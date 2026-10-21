"""28/42/56-day causal smoothing comparison with a FIXED forecast target.

Changing the thing being scored can mechanically make forecasts look better.
Primary outcomes here always remain the frozen 28-day Progress benchmark;
independent corrected future-run pace is a second common target. Scoring each
candidate against its own smoother is reported only as a cosmetic control.
No live model, planner, corrections, settings or activity evidence changes.
"""
from argparse import ArgumentParser
from datetime import datetime
import json
import math
from pathlib import Path
import sqlite3
from urllib.request import urlopen

import numpy as np

from run_analysis.aerobic_candidate import measurement_inputs
from run_analysis.aerobic_performance import LocalPaceTrend
from run_analysis.config import load_config, resolve_project_path
from run_analysis.pace_temperature import comparison_covariance
from run_analysis.privacy import private_file
from run_analysis.training_response import POLICIES, fit_response
from scripts.audit_contextual_training_response import sequential_blocks
from scripts.audit_empirical_training_response import disjoint, score
from scripts.audit_pace_outlook_error_margin import interval, matched_growth

WINDOWS = (28, 42, 56)
MODES = ('joint', 'rate_only', 'anchor_only', 'unchanged_anchor')


def prefix_levels(snapshot, bandwidth):
    y = np.log([r['pace_min_mile'] for r in snapshot['rows']])
    m = np.array(snapshot['measurement'])
    g = np.array(snapshot['gradients'])
    c = np.array(snapshot['covariance'])
    if not c.size:
        c = c.reshape((g.shape[1], g.shape[1]))
    dates = [datetime.fromisoformat(r['start_time']) for r in snapshot['rows']]
    times = np.array([(d-dates[0]).total_seconds()/86400 for d in dates])
    result = {}
    for i, row in enumerate(snapshot['rows']):
        curve = LocalPaceTrend(y[:i+1], m[:i+1], g[:i+1], c, times[:i+1], bandwidth_days=bandwidth)
        result[row['activity_id']] = curve.level(times[i])['log_pace']
    return result


def rebuild(examples, levels):
    result = []
    for row in examples:
        start = levels[row['origin_id']]
        end = sum(v['weight']*levels[i] for i, v in zip(row['outcome_ids'], row['training_by_outcome']))
        result.append({**row, 'start_level': start, 'start_pace': math.exp(start),
            'actual': math.exp(end), 'change': end-start})
    return result


def common_targets(examples, snapshot):
    raw = {r['activity_id']: math.log(r['pace_min_mile']) for r in snapshot['rows']}
    return {(r['origin'], r['target_id']): dict(benchmark=r['actual'], start=r['start_pace'],
        raw=math.exp(sum(v['weight']*raw[i] for i, v in zip(r['outcome_ids'], r['training_by_outcome']))))
        for r in examples}


def fit_at(policies, origin):
    fits = {}
    for policy, rows in policies.items():
        prior = [r for r in rows if r['outcome_end'] < origin and r['consistency']['regime'] == 'consistent']
        fits[policy] = {model: fit_response(prior, model) for model in ('calendar', 'log_miles')}
    if any(v is None for values in fits.values() for v in values.values()):
        return None
    return fits


def forecast(fits, row, start):
    values = [start*math.exp(fits[p][m]['coefficients']['elapsed' if m == 'calendar' else 'log_miles'] *
        row['elapsed' if m == 'calendar' else 'log_miles']) for p in fits for m in ('calendar', 'log_miles')]
    return (min(values)+max(values))/2


def evaluate(policies, baseline_policies, targets, bandwidth):
    primary = policies['gap7_fraction0.500']
    results = []
    for origin in sorted({r['origin'] for r in primary}):
        fitted = fit_at(policies, origin)
        baseline = fit_at(baseline_policies, origin)
        if fitted is None or baseline is None:
            continue
        latest = max(f['latest_outcome'] for fs in fitted.values() for f in fs.values())
        for row in (r for r in primary if r['origin'] == origin):
            target = targets[origin, row['target_id']]
            for mode in MODES:
                start = target['start'] if mode == 'rate_only' else row['start_pace']
                prediction = (start if mode == 'unchanged_anchor' else
                    forecast(baseline if mode == 'anchor_only' else fitted, row, start))
                results.append(dict(bandwidth=bandwidth, mode=mode, origin=origin, origin_id=row['origin_id'],
                    target_id=row['target_id'], outcome_ids=row['outcome_ids'], outcome_end=row['outcome_end'],
                    horizon=row['horizon'], elapsed=row['elapsed'], regime=row['consistency']['regime'],
                    predicted=prediction, fixed_benchmark=target['benchmark'], raw_target=target['raw'],
                    own_benchmark=row['actual'], persistence=target['start'], latest_training_outcome=latest,
                    log_error=math.log(prediction/target['benchmark']), actual=target['benchmark']))
    return results


def comparisons(predictions):
    result = {}
    baseline = {(r['origin'],r['target_id']):r for r in predictions if r['bandwidth']==28 and r['mode']=='joint'}
    for label, horizon, regime, latest in (
            ('consistent_all',None,'consistent',False), ('consistent_14',14,'consistent',False),
            ('consistent_28',28,'consistent',False), ('all_contexts',None,None,False),
            ('all_56',56,None,False), ('all_84',84,None,False), ('latest_consistent',None,'consistent',True)):
        result[label] = {}
        for bandwidth in WINDOWS:
            result[label][str(bandwidth)] = {}
            for mode in MODES:
                chosen = [r for r in predictions if r['bandwidth']==bandwidth and r['mode']==mode
                    and (horizon is None or r['horizon']==horizon) and (regime is None or r['regime']==regime)
                    and (not latest or r['origin']>='2026-09-01') and (r['origin'],r['target_id']) in baseline]
                summaries = {}
                for subset_name, subset in [('all',chosen),('outcome_disjoint',disjoint(chosen)),('sequential',sequential_blocks(chosen))]:
                    scores = {}
                    for target, field in [('fixed_benchmark','fixed_benchmark'),('raw_runs','raw_target'),('own_smoother','own_benchmark')]:
                        a = [{**r,'actual':r[field],'log_error':math.log(r['predicted']/r[field])} for r in subset]
                        b = [{**r,'predicted':baseline[r['origin'],r['target_id']]['predicted'],
                            'log_error':math.log(baseline[r['origin'],r['target_id']]['predicted']/r['actual'])} for r in a]
                        # For own-smoother scoring, baseline is shown against
                        # that SAME candidate target, not a different smoother.
                        scores[target] = dict(candidate=score(a), baseline=score(b))
                    summaries[subset_name] = scores
                result[label][str(bandwidth)][mode] = summaries
    return result


def load_snapshot(path, base):
    if path.exists():
        snapshot = json.loads(path.read_text())
        if snapshot['as_of'] != base['as_of']:
            raise ValueError('Snapshot must match frozen response as_of')
        return snapshot
    with urlopen('http://127.0.0.1:8000/api/progress-wip?window_days=365&include_exploratory=false&include_legacy_aerobic=false',timeout=150) as f:
        data = json.load(f)
    report = data['pace_comparison']
    hidden = set(data['aerobic_performance']['hidden_activity_ids'])
    start, end = datetime.fromisoformat(base['era_start']), datetime.fromisoformat(base['as_of'])
    root = Path(__file__).resolve().parents[1]
    config = load_config(root/'config.yaml')
    database = resolve_project_path(root, config['paths']['database'])
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as c:
        metadata = {r[0]:r[1] for r in c.execute('SELECT id,total_elapsed_time_s FROM activities')}
        assert c.total_changes == 0
    from datetime import timedelta
    rows = sorted([dict(r) for r in report['runs'] if r['included_in_trend'] and r['trend_weight']>0
        and r['activity_id'] not in hidden and not r.get('manually_excluded') and r.get('workout_type') not in {'hike','bike'}
        and start<=datetime.fromisoformat(r['start_time'])<=end
        and datetime.fromisoformat(r['start_time'])+timedelta(seconds=metadata[r['activity_id']] or 0)<=end],
        key=lambda r:(r['start_time'],r['activity_id']))
    covariance = comparison_covariance(report['calibration'])
    m,g = measurement_inputs(rows,covariance)
    snapshot = dict(as_of=base['as_of'], rows=rows, measurement=m.tolist(), gradients=g.tolist(), covariance=covariance.tolist())
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(snapshot,indent=2,allow_nan=False)+'\n')
    private_file(path)
    return snapshot


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument('--benchmark',type=Path,default=Path('output/empirical-training-response-benchmark-20261020.json'))
    p.add_argument('--consistent',type=Path,default=Path('output/consistent-training-response-20261020.json'))
    p.add_argument('--reference',type=Path,default=Path('output/contextual-training-response-20261020.json'))
    p.add_argument('--snapshot',type=Path,default=Path('output/pace-smoothing-input-20261020.json'))
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    base, consistent, reference = (json.loads(path.read_text()) for path in (args.benchmark,args.consistent,args.reference))
    if len({base['as_of'],consistent['as_of'],reference['as_of']})!=1:
        raise ValueError('Frozen audit sources must share an as_of')
    snapshot = load_snapshot(args.snapshot,base)
    levels = {days:prefix_levels(snapshot,days) for days in WINDOWS}
    controls = rebuild(base['examples'],levels[28])
    error = max(max(abs(a['start_level']-b['start_level']),abs(a['change']-b['change'])) for a,b in zip(controls,base['examples']))
    if error>1e-9:
        raise ValueError(f'Current corrections no longer reproduce frozen responses: {error}')
    targets = common_targets(base['examples'],snapshot)
    policies = {days:{policy:rebuild(data['examples'],levels[days]) for policy,data in consistent['policies'].items()} for days in WINDOWS}
    predictions = [r for days in WINDOWS for r in evaluate(policies[days],policies[28],targets,days)]
    old = {(r['origin'],r['target_id']):r['predicted'] for r in reference['policies'][reference['primary_policy']]['predictions'] if r['method']=='current/v3_midpoint'}
    actual = {(r['origin'],r['target_id']):r['predicted'] for r in predictions if r['bandwidth']==28 and r['mode']=='joint'}
    if old.keys()!=actual.keys() or max(abs(old[k]-actual[k]) for k in old)>1e-9:
        raise ValueError('28-day current-v3 control must reproduce the preceding audit exactly')
    summary = comparisons(predictions)
    final = {}
    latest_id = snapshot['rows'][-1]['activity_id']
    for days in WINDOWS:
        fitted = fit_at(policies[days],base['as_of'])
        future = dict(elapsed=base['may_horizon_days']/28,
            log_miles=math.log1p(base['scenarios']['calendar']['goal'][-1]['completed_miles']/base['era_miles']))
        anchor = math.exp(levels[days][latest_id])
        baseline_anchor = math.exp(levels[28][latest_id])
        final[str(days)] = dict(current_pace=anchor,may_joint=forecast(fitted,future,anchor),
            may_rate_only=forecast(fitted,future,baseline_anchor),
            calendar_rates={policy:fitted[policy]['calendar']['coefficients']['elapsed'] for policy in fitted})
    baseline_errors = [r for r in predictions if r['bandwidth']==28 and r['mode']=='joint']
    power = matched_growth(baseline_errors)['exponent']
    # Keep the preceding error-growth assumption FIXED across candidates:
    # otherwise a window could appear to narrow May uncertainty merely by
    # changing a noisy extrapolation exponent as well as the smoother.
    margins = {}
    for days in WINDOWS:
        margins[str(days)] = {}
        for mode in ('joint','rate_only','anchor_only'):
            chosen = [r for r in predictions if r['bandwidth']==days and r['mode']==mode
                and r['regime']=='consistent' and r['horizon'] in {14,28}]
            margins[str(days)][mode] = interval(chosen,days=base['may_horizon_days'],
                pace=final['28']['may_joint'],exponent=power)
    output = dict(version='pace-smoothing-forecast-v1',as_of=base['as_of'],windows=WINDOWS,
        observations=len(snapshot['rows']),response_examples=len(base['examples']),
        frozen_response_max_log_error=error,baseline_prediction_max_error=max(abs(old[k]-actual[k]) for k in old),
        predictions=predictions,summary=summary,final_sensitivity=final,
        may_margin_sensitivity=dict(fixed_error_growth_exponent=power,
            fixed_central_pace=final['28']['may_joint'],same_scale_comparison=margins,
            note='Diagnostic only: fixed prior short-horizon scaling assumption, not validated May coverage'),
        protocol=['Same observations, correction covariance, response precision, exposure and health cohorts in every candidate',
            'Every smoother anchor rebuilt from its own chronological prefix only',
            'Every response fit uses outcomes completed strictly before the test origin',
            'Primary target is unchanged frozen 28-day benchmark; corrected future-run pace is a second common target',
            'Own-smoother error is cosmetic control only, not grounds for adoption or smaller uncertainty',
            'Joint, rate-only and anchor-only ablations separate response changes from baseline lag',
            '28-day control must exactly reproduce frozen targets and prior exact active-v3 predictions',
            'May projections are sensitivity only; never select the window by its May result'],
        limitations=['Shared corrected observations and overlapping causal benchmark histories are not independent trials',
            'Very few consistent four-week intervals; longer available cases are health affected',
            'Frozen retrospective corrections, not a historically reconstructed calibration',
            'No May error interval can be validated from this window experiment'],
        live_forecast_changed=False,planner_changed=False,database_changed=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    private_file(args.output)
    print(json.dumps(dict(final_sensitivity=final,
        summary={scope:{days:{mode:{target:round(v['candidate']['mae_seconds'],3) for target,v in data['all'].items()}
            for mode,data in methods.items() if mode in {'joint','rate_only'}}
            for days,methods in summary[scope].items()}
            for scope in ('consistent_all','consistent_28','all_56','all_84')},
        margin_radii={days:{mode:v['absolute_error_80_log'] for mode,v in methods.items()}
            for days,methods in margins.items()}),indent=2))


if __name__ == '__main__':
    main()
