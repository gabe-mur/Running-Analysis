"""Read-only literature-informed thermal HR replacement on current evidence."""
from argparse import ArgumentParser
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
from time import perf_counter

import numpy as np

from run_analysis.aerobic_hr_timing import statistics, fit, aligned_windows, response_corrections
from run_analysis.aerobic_response_evidence import TAUS
from run_analysis.aerobic_performance import _fit as trend_fit
from run_analysis.config import load_config, resolve_project_path
from run_analysis.model_windows import load_overlapping_model_windows
from run_analysis.objective_modeling import _primary_window_weight, _pre_quality_indexes
from run_analysis.pace_comparability import build_pace_comparability, compare_run_windows
from run_analysis.quality_phases import detect_continuous_quality_phase
from run_analysis.read_cache import invariant_snapshot
from scripts.thermal_hr_trial import study_shape, thermal_windows, fit_weather, forward, day


def collect(connection, config, *, window_diagnostic=None):
    base = build_pace_comparability(connection, config)
    originals = {r['activity_id']: r for r in base['runs'] if r['included_in_trend'] and r['trend_weight'] > 0}
    source_ordinals = {r['activity_id']: i for i, r in enumerate(base['runs'])}
    windows, _ = load_overlapping_model_windows(connection, config, window_seconds=120, stride_seconds=60,
        check_elevation=True, response_calibration=base['calibration'])
    groups = defaultdict(list)
    for w in windows:
        if w['activity_id'] in originals:
            groups[w['activity_id']].append(w)
    response = {k: fit(statistics(v[0]['_aerobic_response']['blocks'])) for k, v in groups.items()}
    meta = {r['id']: dict(r) for r in connection.execute('SELECT id,creator,start_time_local FROM activities')}
    # Read only a coarse area label; no coordinates enter the report.
    areas = {r['activity_id']: (math.floor(r['latitude']*10), math.floor(r['longitude']*10)) for r in connection.execute('''
        SELECT activity_id,latitude,longitude FROM trackpoints WHERE id IN
        (SELECT min(id) FROM trackpoints WHERE gps_valid=1 GROUP BY activity_id)''')}
    labels = {v: str(i) for i, v in enumerate(sorted(set(areas.values())))}
    neutral = deepcopy(base['calibration']); neutral['heat']['mean'] = 0.
    observations = []; preview_runs = []
    for ordinal, (identity, original) in enumerate(originals.items()):
        rows = groups[identity]; context = rows[0]['_aerobic_response']
        others = [f['losses'] for k, f in response.items() if k != identity]
        i = int(np.argmin(np.mean(others, axis=0))) if len(others) >= 3 else 0
        rows = [{k: v for k, v in w.items() if k != '_aerobic_response'} for w in rows]
        if rows and rows[0].get('workout_type') == 'tempo_threshold':
            phase = detect_continuous_quality_phase(connection, config, identity)
            rows = [rows[j] for j in _pre_quality_indexes(rows, list(range(len(rows))), phase)]
        weights = [_primary_window_weight(w, config) for w in rows]
        corrections = response_corrections(context['blocks'], float(response[identity]['inverse_reference_speed'][i]), TAUS[i])
        aligned = aligned_windows(rows, corrections, origin_epoch=context['origin_epoch'], config=config, calibration=base['calibration'])
        current = compare_run_windows(aligned, weights, config, base['calibration'], precision_seed=20261015+source_ordinals[identity])
        none = compare_run_windows(aligned, weights, config, neutral)
        variants = {}; diagnostics = {}
        for exposure in ('wbgt', 'air'):
            shifted, details = thermal_windows(aligned, config, exposure=exposure)
            # A missing thermal input cannot be treated as known neutral weather.
            complete = all(d['applied'] for d in details)
            candidate = compare_run_windows(shifted, weights, config, neutral) if complete else current
            variants[exposure] = candidate['pace_min_mile']; diagnostics[exposure] = dict(
                available=complete, applied_windows=sum(d['applied'] for d in details), original_windows=len(aligned),
                reasons=sorted({d['reason'] for d in details if d['reason']}),
                mean_hr_shift_bpm=float(np.average([d['hr_shift_bpm'] or 0 for d in details], weights=weights)) if complete else None)
        def average(key):
            values = [(r.get(key), w) for r, w in zip(aligned, weights) if isinstance(r.get(key), (int, float)) and math.isfinite(r[key])]
            return float(np.average([v for v, w in values], weights=[w for v, w in values])) if values else None
        local = datetime.fromisoformat(meta[identity]['start_time_local']); hour = local.hour+local.minute/60
        r = dict(activity_id=identity, start_time_utc=original['start_time'], workout_type=original['workout_type'],
            available=diagnostics['wbgt']['available'], temperature_f=average('temperature_f'),
            sampled_hr_bpm=current['sampled_hr_bpm'], sampled_minutes=average('moving_minutes_into_run'),
            duration_minutes=context['coverage']['moving_seconds']/60, device=(meta[identity]['creator'] or 'unknown').lower(),
            area=labels.get(areas.get(identity), 'unknown'), hour_sin=math.sin(math.tau*hour/24), hour_cos=math.cos(math.tau*hour/24),
            stop_fraction=context['coverage']['stopped_seconds']/max(1., context['coverage']['stopped_seconds']+context['coverage']['moving_seconds']),
            previous_7d_miles=rows[0].get('previous_7d_miles'), previous_28d_miles=rows[0].get('previous_28d_miles'),
            pace_no_weather=none['pace_min_mile'], current_pace=current['pace_min_mile'],
            population_pace=variants['wbgt'], air_pace=variants['air'],
            current_burden=math.log(none['pace_min_mile']/current['pace_min_mile']),
            thermal_burden=math.log(none['pace_min_mile']/variants['wbgt']),
            air_burden=math.log(none['pace_min_mile']/variants['air']), thermal_context=diagnostics,
            windows_retained=len(aligned), original_windows=len(rows))
        if window_diagnostic is not None:
            r.update(window_diagnostic(aligned, weights, config, neutral))
        observations.append(r)
        preview_runs.append({**original, **current})
    return sorted(observations, key=day), {**base, 'runs': preview_runs}


def paired_difference(current, candidate):
    baseline = {r['activity_id']: r for r in current['cases']}
    pairs = [(baseline[r['activity_id']], r) for r in candidate['cases'] if r['activity_id'] in baseline]
    if not pairs:
        return dict(cases=0)
    delta = np.array([abs(b['error_seconds'])-abs(a['error_seconds']) for a, b in pairs])
    labels = np.array([int(datetime.fromisoformat(b['date']).timestamp()/86400//28) for a, b in pairs])
    blocks = [delta[labels == k] for k in sorted(set(labels))]
    rng = np.random.default_rng(20261015)
    draws = [float(np.mean(np.concatenate([blocks[i] for i in rng.integers(len(blocks), size=len(blocks))]))) for _ in range(2000)]
    return dict(cases=len(pairs), calendar_blocks=len(blocks), candidate_minus_current_mae_seconds=float(delta.mean()),
        descriptive_block_range_95=np.quantile(draws, [.025, .975]).tolist() if len(blocks) > 1 else None)


def audit(rows, report):
    eligible = [r for r in rows if r['available']]
    models = {'current': forward(eligible), 'none': forward(eligible, model='none'), 'population': forward(eligible, model='population'),
              'personal': forward(eligible, model='personal'), 'air_population': forward(eligible, model='air_population'),
              'air_personal': forward(eligible, model='air')}
    fitted = {key: fit_weather(eligible, model=model) for key, model in (('personal', 'personal'), ('air', 'air'))}
    leave_one_out = [dict(activity_id=r['activity_id'], scale=fit_weather(
        [v for v in eligible if v['activity_id'] != r['activity_id']], model='personal')['scale']) for r in eligible]
    sensitivity = {}
    for period, sd in ((28., 1.), (84., 1.), (56., .5), (56., 2.)):
        name = f'calendar_{period:g}_prior_sd_{sd:g}'
        value = forward(eligible, model='personal', period=period, prior_sd=sd)
        control = forward(eligible, period=period)
        sensitivity[name] = dict(candidate=value['summary'], current=control['summary'],
            paired=paired_difference(control, value), fit=fit_weather(eligible, model='personal', period=period, prior_sd=sd))
    joined = {r['activity_id']: r for r in rows}; variants = {}
    latest = max(datetime.fromisoformat(r['start_time_utc']) for r in rows)
    for name, scale, key in (('current', 1., 'current_burden'), ('population', 1., 'thermal_burden'),
                              ('personal', fitted['personal']['scale'], 'thermal_burden')):
        adjusted = [{**r, 'pace_min_mile': joined[r['activity_id']]['pace_no_weather']*math.exp(-scale*joined[r['activity_id']][key])
            if joined[r['activity_id']]['available'] else r['pace_min_mile']} for r in report['runs']]
        curve = trend_fit({**report, 'runs': adjusted})
        variants[name] = dict(current=curve['current'],
            runs=[dict(activity_id=r['activity_id'], date=r['start_time'], pace_min_mile=r['pace_min_mile'],
                delta_from_current_seconds=60*(r['pace_min_mile']-joined[r['activity_id']]['current_pace'])) for r in adjusted])
    planted = []
    for recent_only in (False, True):
        shifted = [dict(r, pace_no_weather=r['pace_no_weather']/(1.02 if not recent_only or datetime.fromisoformat(r['start_time_utc']) >= latest-timedelta(days=28) else 1.)) for r in eligible]
        updated = fit_weather(shifted, model='personal'); original_scale = fitted['personal']['scale']
        affected = [r for r in shifted if not recent_only or datetime.fromisoformat(r['start_time_utc']) >= latest-timedelta(days=28)]
        old = {r['activity_id']: r for r in eligible}
        recovered = [100*(old[r['activity_id']]['pace_no_weather']*math.exp(-original_scale*r['thermal_burden'])/
            (r['pace_no_weather']*math.exp(-updated['scale']*r['thermal_burden']))-1) for r in affected]
        planted.append(dict(recent_only=recent_only, planted_speed_percent=2., fitted_scale=updated['scale'],
            mean_recovered_percent=float(np.mean(recovered)), minimum_recovered_percent=min(recovered), maximum_recovered_percent=max(recovered)))
    return dict(version='thermal-hr-replacement-trial-v3', live_changed=False, source_read_only=True,
        study=study_shape(), observations=rows, original_runs=len(rows), weather_supported_runs=len(eligible),
        missing_weather_originals_retained=[r['activity_id'] for r in rows if not r['available']],
        models=models, paired={key: paired_difference(models['current'], value) for key, value in models.items() if key != 'current'},
        fits=fitted, sensitivities=sensitivity, trend_previews=variants, planted_changes=planted,
        leave_one_run_out=dict(minimum_scale=min(r['scale'] for r in leave_one_out),
            maximum_scale=max(r['scale'] for r in leave_one_out), runs=leave_one_out),
        thermal_coverage=dict(below_40f=sum(r['temperature_f'] < 40 for r in eligible),
            from_40_to_55f=sum(40 <= r['temperature_f'] < 55 for r in eligible),
            from_55_to_70f=sum(55 <= r['temperature_f'] < 70 for r in eligible),
            above_70f=sum(r['temperature_f'] >= 70 for r in eligible)),
        assumptions=[
            'Offline only. Every original run/window retained; missing weather uses its original estimate, not known-neutral imputation.',
            'Legacy heat disabled in the thermal-HR candidate; no temperature cost is charged twice.',
            'Population shape from four repeated published group means, not athlete outlier selection or four independent experiments.',
            'Signed shade WBGT transfers the lab temperature contrast to humidity-sensitive outdoor conditions; dry-air-only sensitivity retained.',
            'Scale prior 1 +/- 1 is broad engineering transfer uncertainty, not a clinical or study-derived interval.',
            'Personal scale and nuisance coefficients use earlier runs only for each forward target; same unchanged no-weather target across models.',
            'Existing HR timing, calibration and source corrections frozen retrospectively. This is incremental weather validation, not a fully prospective fitness estimator.',
            '56-day calendar intercept/slope, device, coarse area, HR, sample time, duration, time of day, stops and recent volume control confounding.',
            'No fit is selected for flattening April or making a more attractive trend. All preset sensitivities and failures retained.',
            'Trend previews use frozen existing precision to isolate point-estimate changes; no new physiological confidence band.',
            'Single-athlete observational prediction does not prove a causal weather coefficient or independent latent-fitness accuracy.'
        ])


def main():
    p = ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reuse-observations', type=Path); args = p.parse_args()
    if args.output.exists():
        p.error('Use a new output path')
    started = perf_counter()
    if args.reuse_observations:
        saved = json.loads(args.reuse_observations.read_text())
        saved = saved.get('reusable_observations', saved)
        rows, report = saved['rows'], saved['report']
    else:
        config = load_config('config.yaml'); database = resolve_project_path(Path.cwd(), config['paths']['database'])
        with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True) as db:
            db.row_factory = sqlite3.Row
            with invariant_snapshot(db):
                rows, report = collect(db, config)
                assert db.total_changes == 0
    extracted = perf_counter(); result = audit(rows, report)
    result['reusable_observations'] = dict(rows=rows, report=report)
    result['timing'] = dict(extraction_seconds=extracted-started, analysis_seconds=perf_counter()-extracted)
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as out:
        json.dump(result, out, indent=2, allow_nan=False)
    print(json.dumps(dict(study=result['study'], original_runs=result['original_runs'], supported=result['weather_supported_runs'],
        models={k: {a: b for a, b in v.items() if a not in ('cases', 'not_evaluated')} for k, v in result['models'].items()},
        fits={k: {a: b for a, b in v.items() if a not in ('source_ids', 'nuisance_beta', 'nuisance_spec')} for k, v in result['fits'].items()},
        paired=result['paired'], planted=result['planted_changes'], coverage=result['thermal_coverage'],
        leave_one_run_out={k: v for k, v in result['leave_one_run_out'].items() if k != 'runs'}, timing=result['timing']), indent=2))


if __name__ == '__main__':
    main()
