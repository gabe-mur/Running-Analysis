"""Explore additional pace/HR signals without changing athlete data or models.

One observation per physical run. Fixed candidate groups, local calendar
controls, block uncertainty, and earlier-run fits avoid selecting a correction
merely because its pooled correlation is attractive. Existing pace corrections
are a frozen, full-history representation: chronological checks below validate
only incremental associations, not the complete physiological estimator.
"""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from datetime import datetime
import json
import math
from pathlib import Path
import sqlite3

import numpy as np

from run_analysis.config import load_config
from run_analysis.pace_comparability import build_pace_comparability
from run_analysis.performance_factors import load_factor_observations
from run_analysis.model_windows import load_overlapping_model_windows
from run_analysis.objective_modeling import _primary_window_weight, _pre_quality_indexes, _wbgt_c
from run_analysis.read_cache import invariant_snapshot
from run_analysis.session_feedback import with_session_feedback


BASE = ('sampled_hr_bpm', 'sampled_minutes', 'heat_c', 'headwind_mph',
        'crosswind_mph', 'duration_minutes', 'device', 'area', 'workout_type', 'weekend')
GROUPS = {
    'time_of_day': ('hour_sin', 'hour_cos'),
    'recent_volume': ('previous_7d_miles', 'previous_28d_miles'),
    'recovery_spacing': ('log_days_since_run', 'log_days_since_hard'),
    'recent_hr_load': ('prior_3d_load', 'prior_21d_daily_load'),
    'key_session_carryover': ('prior_72h_quality_load', 'prior_72h_long_load'),
    'stops': ('stop_fraction',),
    'cold_weather': ('cold_below_50f',),
    'additional_weather': ('precipitation_in', 'surface_pressure_hpa'),
}
CATEGORIES = {'device', 'area', 'workout_type'}


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def day(row):
    return datetime.fromisoformat(row['start_time_utc']).timestamp() / 86400


def design(rows, keys, spec=None):
    if spec is None:
        spec = []
        for key in keys:
            values = [r.get(key) for r in rows]
            if key in CATEGORIES:
                levels = Counter(str(v or 'missing') for v in values)
                reference = levels.most_common(1)[0][0]
                spec.extend(dict(key=key, kind='category', level=level)
                            for level in sorted(levels) if level != reference)
            else:
                known = [v for v in values if finite(v)]
                if len(known) < 2 or np.std(known) < 1e-9:
                    continue
                spec.append(dict(key=key, kind='numeric', center=float(np.mean(known)),
                                 scale=float(np.std(known))))
                if len(known) != len(rows):
                    spec.append(dict(key=key, kind='missing'))
    columns = []
    for f in spec:
        if f['kind'] == 'numeric':
            columns.append([(r[f['key']]-f['center'])/f['scale']
                            if finite(r.get(f['key'])) else 0 for r in rows])
        elif f['kind'] == 'missing':
            columns.append([float(not finite(r.get(f['key']))) for r in rows])
        else:
            columns.append([float(str(r.get(f['key']) or 'missing') == f['level']) for r in rows])
    return np.asarray(columns).T.reshape(len(rows), len(spec)), spec


def fit(rows, keys, *, period=56, ridge=4., outcome='pace', draws=600):
    x, spec = design(rows, keys)
    days = np.array([day(r) for r in rows]); blocks = np.floor(days/period).astype(int)
    columns = []
    for block in np.unique(blocks):
        mask = blocks == block
        columns.extend((mask.astype(float), np.where(mask, (days-days[mask].mean())/period, 0)))
    z = np.column_stack(columns)
    y = np.log([r[outcome] for r in rows])
    rx = x-z@np.linalg.lstsq(z, x, rcond=None)[0]
    ry = y-z@np.linalg.lstsq(z, y, rcond=None)[0]
    inv = np.linalg.inv(rx.T@rx+ridge*np.eye(x.shape[1]))
    beta = inv@rx.T@ry
    error = ry-rx@beta
    effective = float(np.trace(inv@rx.T@rx))
    correction = math.sqrt(len(rows)/max(1., len(rows)-np.linalg.matrix_rank(z)-effective))
    rng = np.random.default_rng(20261014)
    unique = np.unique(blocks)
    signs = rng.choice([-1., 1.], (len(unique), draws))
    sign_index = np.searchsorted(unique, blocks)
    samples = beta[:, None]+inv@rx.T@(error[:, None]*signs[sign_index]*correction)
    return dict(beta=beta, samples=samples, spec=spec, residual=error, x=rx,
                calendar_blocks=len(unique), calendar_rank=int(np.linalg.matrix_rank(z)))


def contrast(rows, model, keys, changes):
    anchor = {}
    for key in keys:
        values = [r.get(key) for r in rows]
        known = [v for v in values if finite(v)]
        anchor[key] = Counter(values).most_common(1)[0][0] if key in CATEGORIES else (float(np.median(known)) if known else None)
    first, second = dict(anchor), dict(anchor)
    for key, (lo, hi) in changes.items():
        first[key], second[key] = lo, hi
    x, _ = design([first, second], keys, model['spec']); dx = x[1]-x[0]
    seconds = float(np.median([r['pace'] for r in rows]))*60
    values = np.expm1(np.clip(dx@model['samples'], -2, 2))*seconds
    return dict(seconds_per_mile=float(math.expm1(float(dx@model['beta']))*seconds),
                conditional_block_range_95=np.quantile(values, [.025, .975]).tolist())


def chronological(rows, candidate):
    blocks = sorted({int(day(r)//56) for r in rows})
    results = []
    for lo, hi in ((.5, .65), (.65, .8), (.8, 1.)):
        start, end = int(len(blocks)*lo), int(len(blocks)*hi)
        train = [r for r in rows if int(day(r)//56) in blocks[:start]]
        if len(train) < 16:
            continue
        models = [fit(train, keys, draws=2) for keys in (BASE, BASE+candidate)]
        for block in blocks[start:end]:
            test = [r for r in rows if int(day(r)//56) == block]
            matrices = [design(test, keys, model['spec'])[0]
                        for keys, model in zip((BASE, BASE+candidate), models)]
            errors = []
            for i, first in enumerate(test):
                for j in range(i+1, len(test)):
                    if day(test[j])-day(first) > 7:
                        continue
                    errors.append([abs(first['pace']*math.exp(float((x[j]-x[i])@m['beta']))-test[j]['pace'])*60
                                   for x, m in zip(matrices, models)])
            if errors:
                errors = np.array(errors)
                results.append(dict(block=block, pairs=len(errors),
                                    baseline=float(errors[:, 0].mean()), candidate=float(errors[:, 1].mean())))
    if not results:
        return dict(blocks=0, pairs=0)
    differences = np.array([r['candidate']-r['baseline'] for r in results])
    rng = np.random.default_rng(20261014)
    boot = rng.choice(differences, (2000, len(differences)), replace=True).mean(axis=1)
    return dict(blocks=len(results), pairs=sum(r['pairs'] for r in results),
                baseline_mae_seconds=float(np.mean([r['baseline'] for r in results])),
                candidate_mae_seconds=float(np.mean([r['candidate'] for r in results])),
                error_change_seconds=float(differences.mean()),
                block_bootstrap_95=np.quantile(boot, [.025, .975]).tolist(), details=results)


def matched_time_checks(rows):
    """Match on context, never on outcome; reuse no physical run within a check."""
    results = []
    for max_gap in (7, 14):
        candidates = []
        for morning in rows:
            if morning['hour'] >= 12:
                continue
            for evening in rows:
                if evening['hour'] < 16 or morning['area'] != evening['area'] or morning['workout_type'] != evening['workout_type']:
                    continue
                gap = abs(day(evening)-day(morning))
                if gap > max_gap:
                    continue
                tolerances = (('sampled_hr_bpm', 5), ('heat_c', 3), ('duration_minutes', 15), ('headwind_mph', 4))
                if any(not finite(morning.get(k)) or not finite(evening.get(k)) or abs(morning[k]-evening[k]) > tolerance
                       for k, tolerance in tolerances):
                    continue
                candidates.append((gap, morning, evening))
        used = set(); pairs = []
        for gap, morning, evening in sorted(candidates, key=lambda x:(x[0],x[1]['activity_id'],x[2]['activity_id'])):
            if morning['activity_id'] in used or evening['activity_id'] in used:
                continue
            used.update((morning['activity_id'],evening['activity_id']))
            pairs.append(dict(morning_id=morning['activity_id'],evening_id=evening['activity_id'],gap_days=gap,
                evening_minus_morning_seconds=(evening['pace']-morning['pace'])*60))
        differences = [p['evening_minus_morning_seconds'] for p in pairs]
        results.append(dict(max_gap_days=max_gap, candidate_pairs=len(candidates), independent_pairs=len(pairs),
            mean_evening_minus_morning_seconds=float(np.mean(differences)) if differences else None, pairs=pairs))
    return dict(assumptions='Morning before noon, evening from 16:00; same role/coarse area; HR within 5 bpm, shade heat index within 3 C, duration within 15 min, headwind within 4 mph; nearest date first, without replacement.',
        results=results, caveat='Sparse context-matched observational pairs; calendar fitness changes and unmeasured influences remain.')


def collect(c, config):
    broad, coverage = load_factor_observations(c, config)
    factors = {r['activity_id']: r for r in broad}
    report = build_pace_comparability(c, config)
    sensor_windows, _ = load_overlapping_model_windows(c, config,
        window_seconds=int(config['model']['primary_window_seconds']),
        stride_seconds=int(config['model']['primary_window_stride_seconds']), check_elevation=True)
    by_run = defaultdict(list)
    for window in sensor_windows:
        by_run[window['activity_id']].append(window)
    meta = with_session_feedback(c, c.execute('''SELECT a.*,m.calculated_moving_time_s,
        m.session_zone_load,m.hr_load_coverage,m.stop_fraction,m.detected_workout_type,
        m.analysis_distance_m,o.perceived_exertion,o.health_tag
        FROM activities a JOIN activity_metrics m ON m.activity_id=a.id
        LEFT JOIN run_overrides o ON o.activity_id=a.activity_id''').fetchall())
    metadata = {r['id']: r for r in meta}
    from run_analysis.progress import prepare_progress_data
    prepared = prepare_progress_data(c)
    roles = {identity: value['workout'].value for identity, value in prepared.details.items()
             if value.get('workout') is not None}
    stats = {r['activity_id']: dict(r) for r in c.execute('''SELECT activity_id,
        avg(average_cadence_spm) AS cadence_spm,avg(temperature_f) AS temperature_f,
        avg(precipitation_in) AS precipitation_in,avg(surface_pressure_hpa) AS surface_pressure_hpa
        FROM segments WHERE is_pathological=0 GROUP BY activity_id''')}
    places = {}
    for r in c.execute('''SELECT activity_id,latitude,longitude FROM trackpoints
        WHERE id IN (SELECT min(id) FROM trackpoints WHERE gps_valid=1 GROUP BY activity_id)'''):
        # Private, coarse geographic controls; no coordinate is emitted.
        places[r['activity_id']] = (math.floor(r['latitude']*10), math.floor(r['longitude']*10))
    place_names = {p: f'area_{i+1}' for i, p in enumerate(sorted(set(places.values())))}
    history = sorted(meta, key=lambda r: datetime.fromisoformat(r['start_time_utc']))
    output = []
    for estimate in report['runs']:
        identity = estimate['activity_id']
        if not estimate['included_in_trend'] or estimate['workout_type'] not in {'easy', 'long', 'recovery'}:
            continue
        row = {**factors.get(identity, {}), **stats.get(identity, {})}
        windows = by_run[identity]
        windows = [windows[i] for i in _pre_quality_indexes(windows, list(range(len(windows))), None)]
        weights = [_primary_window_weight(w, config) for w in windows]
        def sampled(key):
            pairs = [(w.get(key), weight) for w, weight in zip(windows, weights) if finite(w.get(key))]
            return float(np.average([p[0] for p in pairs], weights=[p[1] for p in pairs])) if pairs else None
        heat = []
        for w, weight in zip(windows, weights):
            try:
                value = _wbgt_c(w)
                if finite(value):
                    heat.append((value, weight))
            except (ValueError, TypeError, KeyError):
                pass
        row.update(sampled_minutes=sampled('moving_minutes_into_run'),
            headwind_mph=sampled('headwind_signed_mph'), crosswind_mph=sampled('crosswind_mph'),
            heat_c=float(np.average([p[0] for p in heat], weights=[p[1] for p in heat])) if heat else None)
        m = metadata[identity]
        at = datetime.fromisoformat(estimate['start_time'])
        local = datetime.fromisoformat(m['start_time_local'])
        hour = local.hour+local.minute/60+local.second/3600
        row.update(activity_id=identity, start_time_utc=estimate['start_time'],
                   pace=estimate['pace_min_mile'], observed=estimate['observed_pace_min_mile'],
                   workout_type=estimate['workout_type'], sampled_hr_bpm=estimate['sampled_hr_bpm'],
                   hour=hour, hour_sin=math.sin(math.tau*hour/24), hour_cos=math.cos(math.tau*hour/24),
                   weekend=float(local.weekday() >= 5),
                   area=place_names.get(places.get(identity), 'missing'),
                   timezone_source=m['timezone_source'],
                   stop_fraction=m['stop_fraction'], perceived_exertion=m['perceived_exertion'],
                   workout_feel=m['workout_feel'], feedback_conflict=m['feedback_conflict'],
                   cold_below_50f=max(0.,50-row['temperature_f']) if finite(row.get('temperature_f')) else None)
        priors = [(h, (at-datetime.fromisoformat(h['start_time_utc'])).total_seconds()/86400)
                  for h in history if datetime.fromisoformat(h['start_time_utc']) < at]
        def load_sum(days, role=None):
            if (at-datetime.fromisoformat(history[0]['start_time_utc'])).total_seconds()/86400 < days:
                return None
            relevant = [h for h, gap in priors if 0 < gap <= days
                        and h['id'] in roles and (role is None or roles[h['id']] in role)]
            if any(not finite(h['session_zone_load']) or (h['hr_load_coverage'] or 0) < .7 for h in relevant):
                return None  # Unknown preceding load must not become a rest day.
            return sum(h['session_zone_load'] for h in relevant)
        daily = load_sum(21)
        row.update(prior_3d_load=load_sum(3), prior_21d_daily_load=daily/21 if daily is not None else None,
                   prior_72h_quality_load=load_sum(3, {'intervals','tempo_threshold','hill_repeats','fartlek','progression'}),
                   prior_72h_long_load=load_sum(3, {'long'}))
        for key in ('days_since_run', 'days_since_hard'):
            row['log_'+key] = math.log1p(row[key]) if finite(row.get(key)) else None
        output.append(row)
    return sorted(output, key=day), dict(imported_activities=len(meta), comparable_runs=len(report['runs']),
        normal_easy_long_runs=len(output), broader_stable_runs=len(broad), loader_coverage=coverage,
        activities_without_stored_hr=c.execute('''SELECT count(*) FROM activities a
            WHERE a.average_hr_bpm IS NULL AND NOT EXISTS
            (SELECT 1 FROM trackpoints t WHERE t.activity_id=a.id AND t.heart_rate_bpm IS NOT NULL)''').fetchone()[0],
        fit_sensor_activities=c.execute('SELECT count(distinct activity_id) FROM activity_sensor_streams').fetchone()[0],
        readiness_reports=c.execute('SELECT count(*) FROM user_health_reports').fetchone()[0],
        metadata_health_counts=dict(Counter(r.get('health_tag') or 'unreported' for r in meta)))


def analyze(rows):
    results = {}
    last = max(map(day, rows))
    for name, candidate in GROUPS.items():
        keys = BASE+candidate
        changes = ({'hour_sin': (math.sin(math.tau*10/24), math.sin(math.tau*18/24)),
                    'hour_cos': (math.cos(math.tau*10/24), math.cos(math.tau*18/24))}
                   if name == 'time_of_day' else
                   {k: tuple(np.quantile([r[k] for r in rows if finite(r.get(k))], [.25,.75]))
                    for k in candidate if any(finite(r.get(k)) for r in rows)})
        # A quartile contrast is meaningless for sparse, zero-inflated exposure
        # (both quartiles can be zero despite nonzero exposed runs).
        for key, (lo, hi) in list(changes.items()):
            if lo == hi:
                positive = [r[key] for r in rows if finite(r.get(key)) and r[key] > 0]
                changes[key] = (0., float(np.median(positive))) if positive else (0., 0.)
        variants = []
        for label, cohort, period, ridge, outcome in (
            ('primary', rows, 56, 4., 'pace'),
            ('calendar_28d', rows, 28, 4., 'pace'),
            ('calendar_112d', rows, 112, 4., 'pace'),
            ('stronger_shrinkage', rows, 56, 16., 'pace'),
            ('trailing_365d', [r for r in rows if day(r)>=last-365], 56, 4., 'pace'),
            ('2026_only', [r for r in rows if r['start_time_utc'].startswith('2026')], 56, 4., 'pace'),
            ('observed_sections', rows, 56, 4., 'observed'),
        ):
            model = fit(cohort, keys, period=period, ridge=ridge, outcome=outcome)
            variants.append(dict(variant=label, runs=len(cohort), **contrast(cohort, model, keys, changes)))
        results[name] = dict(contrast=changes, variants=variants, chronological=chronological(rows, candidate),
            chronological_2026=chronological([r for r in rows if r['start_time_utc'].startswith('2026')], candidate))
        if name == 'time_of_day':
            results[name]['sensitivity_extra'] = []
            for label, cohort in (
                ('easy_only', [r for r in rows if r['workout_type'] == 'easy']),
                ('2026_easy_only', [r for r in rows if r['workout_type'] == 'easy' and r['start_time_utc'].startswith('2026')]),
                ('before_new_watch', [r for r in rows if r['device'] != 'forerunner 265s']),
                ('gps_timezone_only', [r for r in rows if r['timezone_source'] == 'gps_nyc']),
            ):
                model = fit(cohort, keys)
                results[name]['sensitivity_extra'].append(dict(cohort=label, runs=len(cohort),
                    **contrast(cohort, model, keys, changes)))
    # A descriptive joint fit checks whether separate candidate effects survive
    # competition. It is not a feature-selection or production fitting step.
    all_keys = tuple(dict.fromkeys(BASE+sum(GROUPS.values(), ())))
    joint = fit(rows, all_keys)
    recent = [r for r in rows if r['start_time_utc'].startswith('2026')]
    joint_recent = fit(recent, all_keys)
    for name in results:
        results[name]['joint'] = contrast(rows, joint, all_keys, results[name]['contrast'])
        results[name]['joint_2026'] = contrast(recent, joint_recent, all_keys, results[name]['contrast'])
        if name != 'time_of_day':
            keys = BASE+GROUPS[name]; primary = fit(rows, keys)
            results[name]['separate_components'] = {key:dict(contrast=values,
                primary=contrast(rows, primary, keys, {key:values}),
                joint=contrast(rows, joint, all_keys, {key:values}),
                joint_2026=contrast(recent, joint_recent, all_keys, {key:values}))
                for key, values in results[name]['contrast'].items()}
    return dict(signals=results, coverage={k:dict(known=sum(finite(r.get(k)) for r in rows),
        values=sorted(set(r[k] for r in rows if finite(r.get(k))))) for k in tuple(dict.fromkeys(sum(GROUPS.values(), ())))},
        hours=dict(Counter(int(r['hour']//3)*3 for r in rows)),
        years=dict(Counter(r['start_time_utc'][:4] for r in rows)),
        devices=dict(Counter(r['device'] for r in rows)),
        matched_time_checks=matched_time_checks(rows),
        feedback={k:sum(finite(r.get(k)) and not r['feedback_conflict'] for r in rows)
                  for k in ('perceived_exertion','workout_feel')})


def main():
    parser = ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); config = load_config('config.yaml')
    with sqlite3.connect('file:data/run_analysis.sqlite?mode=ro', uri=True) as c:
        c.row_factory = sqlite3.Row
        with invariant_snapshot(c):
            rows, coverage = collect(c, config)
            assert c.total_changes == 0
    result = dict(version='additional-pace-signals-v1', coverage_summary=coverage, **analyze(rows),
        observations=rows, database_unchanged=True, live_model_changed=False,
        limitations=['Exploratory single-athlete associations, not causal correction factors.',
            'Comparability representation calibrated on full history; chronological checks test incremental candidate coefficients only.',
            'Nearby-run pairs reuse runs; summary and uncertainty resample calendar blocks, not trackpoints or pairs.',
            'Eight candidate groups examined without selecting or tuning a production model.',
            'Cadence excluded: mechanically linked to speed. Post-run feedback is not an independent fitness adjustment.',
            'Coarse area and device controls can be inseparable from calendar period.',
            'Imported historical activity coverage is assumed; absence of a file cannot establish that no training occurred.',
            'Smashrun creator labels identify an export source, not necessarily the physical watch or HR sensor.',
            'Unlogged sleep, nutrition, hydration, surface and equipment remain unknown.'])
    from run_analysis.privacy import private_file
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n'); private_file(args.output)
    print('Saved:', args.output, 'runs:', len(rows))
    for name, value in result['signals'].items():
        print(name, json.dumps(dict(estimate_seconds=value['variants'][0]['seconds_per_mile'],
            recent_estimate_seconds=value['variants'][4]['seconds_per_mile'],
            future_error_change=value['chronological'].get('error_change_seconds'),
            future_2026_error_change=value['chronological_2026'].get('error_change_seconds'))))


if __name__ == '__main__':
    main()
