"""Offline damped-trend/entering-fatigue audit; no live model or planner changes.

Fits are chronological in corrected run outcomes, but correction calibration is
frozen at today's shared settings. This is conditional retrospective validation,
NOT a prospective replay of weather/HR/elevation calibration or physiological
fitness validation. May's seven-month horizon exceeds the scored holdouts.
"""
from argparse import ArgumentParser
from datetime import date, datetime, timedelta, timezone
import json
import math
from pathlib import Path
import sqlite3
from urllib.request import urlopen
from zoneinfo import ZoneInfo

import numpy as np

from run_analysis.aerobic_candidate import measurement_inputs
from run_analysis.aerobic_forecast_core import damped_days, fit_trend
from run_analysis.aerobic_performance import LocalPaceTrend
from run_analysis.pace_temperature import comparison_covariance






def forecast(fit, horizon, *, half_life=56, feature_value=None):
    value = fit['log_level'] + fit['slope'] * damped_days(horizon, half_life)
    if fit['feature_coefficient'] is not None:
        if feature_value is None or not math.isfinite(feature_value):
            raise ValueError('Known feature required for conditional prediction')
        value += fit['feature_coefficient'] * (feature_value - fit['feature_center']) / fit['feature_scale']
    return math.exp(value)


def score(predictions):
    if not predictions:
        return dict(pairs=0, unique_targets=0)
    # Repeated forecasts for the same outcome share one vote in aggregate.
    counts = {p['target_id']: sum(v['target_id'] == p['target_id'] for v in predictions)
        for p in predictions}
    w = np.array([1/counts[p['target_id']] for p in predictions]); w /= w.sum()
    error = np.array([(p['predicted'] - p['actual']) * 60 for p in predictions])
    return dict(pairs=len(predictions), unique_targets=len(counts),
        mae_seconds_per_mile=float(w @ np.abs(error)),
        bias_seconds_per_mile=float(w @ error),
        median_absolute_error_seconds_per_mile=float(np.median(np.abs(error))),
        p80_absolute_error_seconds_per_mile=float(np.quantile(np.abs(error), .8)),
        note='Future run or multi-run outcomes as specified; overlapping origins/targets are dependent')


def trend_holdouts(rows, times, variance, *, target_mode='run'):
    if target_mode not in {'run', 'block'}:
        raise ValueError('Run or block target required')
    y = np.log([r['pace_min_mile'] for r in rows])
    models = dict(persistence='persistence', linear=None, damp28=28, damp56=56, damp84=84)
    output = []
    last_origin = -math.inf
    for i in range(len(rows)):
        if i < 11 or times[i] - times[0] < 42 or times[i] - last_origin < 7:
            continue
        # Never cross a >28-day recording gap for a trend forecast.
        gaps = np.flatnonzero(np.diff(times[:i+1]) > 28)
        start = int(gaps[-1] + 1) if len(gaps) else 0
        fit = fit_trend(y[start:i+1], times[start:i+1], variance[start:i+1])
        if fit is None:
            continue
        last_origin = times[i]
        curve = LocalPaceTrend(y[start:i+1], variance[start:i+1],
            np.zeros((i+1-start, 0)), np.zeros((0, 0)), times[start:i+1])
        # Responsive current level, broad slope. Same current level for all
        # methods, so damping cannot win by changing the baseline estimator.
        fit['log_level'] = curve.level(times[i])['log_pace']
        for horizon in (14, 28, 56, 84):
            eligible = [j for j in range(i+1, len(rows))
                if abs(times[j] - times[i] - horizon) <= 7]
            if not eligible:
                continue
            selected = [min(eligible, key=lambda j: abs(times[j] - times[i] - horizon))] if target_mode == 'run' else eligible
            if target_mode == 'block' and len(selected) < 2:
                continue
            if np.any(np.diff(times[i:max(selected)+1]) > 28):
                continue
            outcome_weights = 1/variance[selected]; outcome_weights /= outcome_weights.sum()
            actual_horizon = float(outcome_weights @ times[selected] - times[i])
            actual = math.exp(float(outcome_weights @ y[selected]))
            identity = rows[selected[0]]['activity_id'] if target_mode == 'run' else ','.join(str(rows[j]['activity_id']) for j in selected)
            for model, half_life in models.items():
                p = math.exp(fit['log_level']) if model == 'persistence' else forecast(fit, actual_horizon, half_life=half_life)
                output.append(dict(model=model, requested_horizon=horizon,
                    actual_horizon=actual_horizon, origin_id=rows[i]['activity_id'],
                    origin_date=rows[i]['start_time'], target_mode=target_mode,
                    outcome_runs=len(selected), target_id=identity, predicted=p, actual=actual))
    return output


def fatigue_audit(rows, times, variance, values):
    """Time-controlled association plus next-run paired chronological checks.

    Test-time fatigue is the value known before that actual run. It is NOT
    assumed knowable months earlier. No same-run completed load enters it.
    """
    y = np.log([r['pace_min_mile'] for r in rows]); values = np.array(values, dtype=float)
    matched = np.isfinite(values)
    ii = np.flatnonzero(matched)
    empty = dict(available=False, runs=int(matched.sum()), predictions=[])
    if len(ii) < 10:
        return empty
    fit = fit_trend(y[ii], times[ii], variance[ii], history_days=366, feature=values[ii])
    base = fit_trend(y[ii], times[ii], variance[ii], history_days=366)
    if fit is None or base is None:
        return empty
    q25, q75 = np.quantile(values[ii], [.25, .75])
    iqr_effect = forecast(fit, 0, feature_value=float(q75)) - forecast(fit, 0, feature_value=float(q25))
    conditional = []; last_origin = -math.inf
    for index in range(12, len(ii)):
        training = ii[:index]; j = ii[index]; i = training[-1]
        if times[i] - times[training[0]] < 42 or times[i] - last_origin < 4 or times[j] - times[i] > 14:
            continue
        if np.any(np.diff(times[i:j+1]) > 28):
            continue
        fit0 = fit_trend(y[training], times[training], variance[training])
        fit1 = fit_trend(y[training], times[training], variance[training], feature=values[training])
        if fit0 is None or fit1 is None:
            continue
        last_origin = times[i]
        for name, f in [('time_only', fit0), ('time_plus_fatigue', fit1)]:
            p = forecast(f, float(times[j] - times[i]),
                feature_value=float(values[j]) if name == 'time_plus_fatigue' else None)
            conditional.append(dict(model=name, target_id=rows[j]['activity_id'],
                origin_id=rows[i]['activity_id'], predicted=p, actual=math.exp(y[j]),
                fatigue_coefficient=f['feature_coefficient']))
    summaries = {model: score([p for p in conditional if p['model'] == model])
        for model in ('time_only', 'time_plus_fatigue')}
    return dict(available=True, runs=len(ii), first_run=rows[ii[0]]['start_time'],
        last_run=rows[ii[-1]]['start_time'], fatigue_q25=float(q25), fatigue_q75=float(q75),
        pace_effect_high_vs_low_seconds_per_mile=float(iqr_effect*60),
        note='Time-adjusted association, not causal recovery or an estimated fatigue-free pace',
        summaries=summaries, predictions=conditional)


def scheduler_features(connection, root, rows, as_of):
    from run_analysis.progress import prepare_progress_data
    from run_analysis.progress_wip import _canonical_load
    from run_analysis.progress_fatigue import residual
    from run_analysis.adaptive.program_objective_trial import DemandScenario
    prepared = prepare_progress_data(connection)
    ledger = _canonical_load(connection, root, prepared.sessions, as_of)
    model = ledger.get('fatigue_model')
    if not model or not model.get('history_start'):
        return [None]*len(rows), dict(available=False)
    zone = ZoneInfo(model['timezone']); scenario = DemandScenario(**model['scenario'])
    first = datetime.fromisoformat(model['history_start']).astimezone(zone).date()
    observed = [r for r in ledger['rows'] if r['accounting_key'] == model['accounting_key']]
    found = {r['activity_id'] for r in observed}
    missing = [s.start_time.astimezone(zone).date() for s in prepared.sessions if s.activity_id not in found]
    lead = math.ceil(5 * scenario.memory_days); output = []
    for row in rows:
        day = datetime.fromisoformat(row['start_time']).astimezone(zone).date()
        complete = first <= day-timedelta(days=lead) and not any(day-timedelta(days=lead) <= d < day for d in missing)
        output.append(residual(observed, day, model['scales'], scenario, zone, entering=True)['value'] if complete else None)
    return output, dict(available=True, history_start=model['history_start'],
        memory_days=scenario.memory_days, snapshot_as_of=ledger['snapshot_as_of'],
        basis='Current committed scheduler ledger reconstructed before each run date; missing coverage excluded')


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(); root = Path(__file__).resolve().parents[1]
    with urlopen(args.url + '/api/progress-wip?window_days=365&include_exploratory=false&include_legacy_aerobic=false', timeout=150) as response:
        data = json.load(response)
    state = data['aerobic_performance']; report = data['pace_comparison']
    hidden = set(state.get('hidden_activity_ids', []))
    rows = sorted([r for r in report['runs'] if r.get('included_in_trend') and r.get('trend_weight', 0) > 0
        and r['activity_id'] not in hidden and not r.get('manually_excluded') and r.get('workout_type') not in {'hike','bike'}],
        key=lambda r: (r['start_time'], r['activity_id']))
    dates = [datetime.fromisoformat(r['start_time']) for r in rows]
    times = np.array([(d-dates[-1]).total_seconds()/86400 for d in dates])
    cov = comparison_covariance(report['calibration']); m, g = measurement_inputs(rows, cov)
    variance = m + np.einsum('ij,jk,ik->i', g, cov, g)
    y = np.log([r['pace_min_mile'] for r in rows])
    curve = LocalPaceTrend(y, m, g, cov, times)
    current = curve.level(0.)['pace_min_mile']
    assert math.isclose(current, state['current']['pace_min_mile'], abs_tol=1e-10)
    predictions = trend_holdouts(rows, times, variance)
    block_predictions = trend_holdouts(rows, times, variance, target_mode='block')
    validation = {str(h): {model: score([p for p in predictions if p['requested_horizon'] == h and p['model'] == model])
        for model in ('persistence', 'linear', 'damp28', 'damp56', 'damp84')} for h in (14, 28, 56, 84)}
    block_validation = {str(h): {model: score([p for p in block_predictions if p['requested_horizon'] == h and p['model'] == model])
        for model in ('persistence', 'linear', 'damp28', 'damp56', 'damp84')} for h in (14, 28, 56, 84)}
    zone = ZoneInfo(data['timezone'])
    by_day = {datetime.fromisoformat(r['as_of']).astimezone(zone).date(): r for r in data['training_load']['daily']}
    features = {}
    for name, channel in [('recorded_hr_2d', 'short'), ('recorded_hr_7d', 'recent')]:
        features[name] = [(by_day.get(d.astimezone(zone).date()-timedelta(days=1), {}).get(channel) or {}).get('value') for d in dates]
    volume_by_day = {datetime.fromisoformat(r['as_of']).astimezone(zone).date(): r['distance_7d']
        for r in data['exposure']['daily']}
    features['prior_7d_volume'] = [volume_by_day.get(d.astimezone(zone).date()-timedelta(days=1)) for d in dates]
    from run_analysis.config import load_config, resolve_project_path
    config = load_config(root/'config.yaml'); db = resolve_project_path(root, config['paths']['database'])
    with sqlite3.connect(db.resolve().as_uri()+'?mode=ro', uri=True) as c:
        c.row_factory = sqlite3.Row
        features['scheduler_2d'], provenance = scheduler_features(c, root, rows, datetime.fromisoformat(data['as_of']))
        assert c.total_changes == 0
    fatigue = {k: fatigue_audit(rows, times, variance, [np.nan if v is None else v for v in values]) for k, values in features.items()}
    target = data['goal_journey']['projection']['target_date']
    today = date.fromisoformat(data['goal_journey']['today']); horizon = (date.fromisoformat(target)-today).days
    segment = curve.segments[-1]; scenarios = []
    for history in (56, 84, 112, 140, 182, 366):
        f = fit_trend(y[segment], times[segment], variance[segment], history_days=history)
        if f is None:
            continue
        f['log_level'] = math.log(current)
        for half_life in (28, 56, 84):
            scenarios.append(dict(history_days=history, half_life_days=half_life,
                forecast_pace_min_mile=forecast(f, horizon, half_life=half_life), fit=f))
    # A transparent exploratory envelope from the longest tested horizon.
    # This is not a 211-day confidence interval, nor a forecast of 17-mile pace.
    outcomes = [p for p in predictions if p['model'] == 'persistence' and p['requested_horizon'] == 84]
    factors = np.exp(np.quantile([math.log(p['actual']/p['predicted']) for p in outcomes], [.1, .9])) if outcomes else None
    candidate = dict(available=factors is not None, center_pace_min_mile=current,
        label='Plausible aerobic pace, if performance remains near its current level',
        basis='10th–90th percentile of 84-day persistence forecast errors; baseline chosen after this comparison',
        future_date=target, observed_validation_horizon_days=84, examples=len(outcomes),
        future_horizon_validated=False, physiological_confidence_interval=False,
        limitation='Exploratory performance envelope, not a calibrated seven-month expected-pace range')
    if factors is not None:
        low, high = (current*float(f) for f in factors)
        candidate.update(low_pace_min_mile=low, high_pace_min_mile=high,
            display_low_seconds_per_mile=int(math.floor(low*60/5)*5),
            display_high_seconds_per_mile=int(math.ceil(high*60/5)*5))
    result = dict(version='offline-damped-pace-forecast-v1', as_of=data['as_of'],
        reference_hr_bpm=state['target_hr_bpm'], current_pace_min_mile=current,
        target_date=target, horizon_days=horizon, run_count=len(rows), validation=validation,
        predictions=predictions, block_predictions=block_predictions, block_validation=block_validation,
        fatigue=fatigue, fatigue_provenance=provenance,
        scenarios=scenarios, candidate_display=candidate,
        limitations=['Frozen current retrospective pace normalization; not a fully prospective calibration replay',
            'Individual-run outcomes contain day-to-day performance variation, not measured fitness',
            'Chronological folds overlap; pair count is not independent evidence count',
            'Future-fatigue conditional checks use observed pre-run load, not a months-ahead load forecast',
            'Scheduler coverage starts July 10; earlier fatigue is an HR-load proxy only',
            'No validation at the 211-day target horizon; scenarios are not calibrated prediction intervals',
            'Fixed damping half-lives are modeling assumptions, not physiological constants',
            'No assumption that benchmark pace can be sustained for 17 miles'],
        live_model_changed=False, planner_changed=False, database_changed=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    from run_analysis.privacy import private_file
    private_file(args.output)
    print(json.dumps({k: result[k] for k in ('current_pace_min_mile','target_date','validation')}, indent=2))
    print('fatigue',json.dumps({k:{n:v for n,v in value.items() if n != 'predictions'} for k,value in fatigue.items()}, indent=2))
    print('block_validation',json.dumps(block_validation,indent=2))
    print('scenarios',json.dumps([{k:v for k,v in s.items() if k!='fit'} for s in scenarios], indent=2))
    print('Saved',args.output)


if __name__ == '__main__':
    main()
