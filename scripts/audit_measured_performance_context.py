"""Fixed-family exploratory context checks, never production corrections.

One row per physical run. Normalization is a frozen full-history representation;
chronological checks concern incremental context, NOT prospective fitness truth.
Watch-period labels refer to the run recorder, not verified wellness provenance.
"""
from argparse import ArgumentParser
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import sqlite3
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np

from run_analysis.config import load_config, resolve_project_path
from run_analysis.continuous_training_dose import ContinuousDoseReference
from run_analysis.fit import _open
from run_analysis.model_windows import load_overlapping_model_windows
from run_analysis.movement import classify_movement
from run_analysis.objective_modeling import _primary_window_weight, _pre_quality_indexes
from run_analysis.privacy import private_file
from run_analysis.processing import _load_points
from run_analysis.progress import prepare_progress_data
from run_analysis.progress_fatigue import residual
from run_analysis.progress_wip import _canonical_load
from run_analysis.adaptive.program_objective_trial import DemandScenario
from run_analysis.read_cache import invariant_snapshot
from run_analysis.training_signals import HeartRateReference, observed_cardio_trace
from run_analysis.wellness_signals import read_wellness
from scripts.audit_additional_pace_signals import BASE, contrast, day, design, finite, fit
from scripts.audit_relative_weather import paired_range


PRIMARY = tuple(dict.fromkeys(BASE + ('temperature_f', 'temperature_squared',
    'heat_squared', 'hour_sin', 'hour_cos')))
# Fixed in advance, no winner selected for production. Correlated descriptors
# stay in separate families; joint high-dimensional feature selection is absent.
FAMILIES = {
    'reconstructed_running_exposure': ('entering_running_minutes',),
    'reconstructed_cardio_exposure': ('entering_cardio',),
    'reconstructed_entering_load': ('entering_running_minutes', 'entering_cardio'),
    'committed_scheduler_fatigue': ('scheduler_fatigue',),
    'sleep_duration': ('sleep_hours',),
    'three_night_sleep': ('sleep_three_nights_hours',),
    'previous_date_sleep_alignment': ('previous_date_sleep_hours',),
    'daily_resting_hr': ('daily_resting_hr',),
    'daily_hrv': ('daily_hrv',),
    'sleep_score': ('sleep_score',),
    'early_pacing': ('early_speed_excess_percent',),
    'early_hr_settling': ('early_hr_rise',),
    'rain_and_pressure': ('precipitation_in', 'surface_pressure_hpa'),
    'recording_interruptions': ('stop_fraction', 'longest_recording_gap_seconds'),
}


def sleep_context(series, local_date):
    """No future date, interpolation, or missing-night-as-zero convention."""
    current = series.get(local_date.isoformat(), {})
    before = series.get((local_date-timedelta(days=1)).isoformat(), {})
    nights = [series.get((local_date-timedelta(days=i)).isoformat(), {}).get('sleep_seconds') for i in range(3)]
    return dict(sleep_hours=current['sleep_seconds']/3600 if finite(current.get('sleep_seconds')) else None,
        previous_date_sleep_hours=before['sleep_seconds']/3600 if finite(before.get('sleep_seconds')) else None,
        sleep_three_nights_hours=float(np.mean(nights))/3600 if all(finite(v) for v in nights) else None,
        daily_resting_hr=current.get('resting_hr_bpm'), daily_hrv=current.get('hrv_ms'),
        sleep_score=current.get('sleep_score'))


def entering_load(history, at, *, memory_days=2., timezone_name='America/New_York'):
    """Use completed prior local dates, like the scheduler's daily accounting.

    Two axes only, reconstructed from immutable evidence; no invented work axis.
    Missing reference support within five time constants withholds HR estimate.
    Older gaps are explicitly screened rather than called measured recovery.
    """
    zone = ZoneInfo(timezone_name)
    target = at.astimezone(zone).date()
    earlier = [r for r in history if datetime.fromisoformat(r['end']) < at
               and date.fromisoformat(r['local_date']) < target]
    if not earlier or min(date.fromisoformat(r['local_date']) for r in history) > target-timedelta(days=math.ceil(5*memory_days)):
        return dict(entering_running_minutes=None, entering_cardio=None)
    running = low = high = 0.
    unsupported = False
    for r in earlier:
        age = (target-date.fromisoformat(r['local_date'])).days
        weight = math.exp(-age/memory_days)
        if finite(r.get('moving_minutes')):
            running += weight*r['moving_minutes']
        elif age <= 5*memory_days:
            return dict(entering_running_minutes=None, entering_cardio=None)
        if r.get('cardio_low') is None or r.get('cardio_high') is None:
            unsupported |= age <= 5*memory_days
            continue
        low += weight*r['cardio_low']
        high += weight*r['cardio_high']
    midpoint = (low+high)/2
    # A narrow point estimate is not fabricated from broadly missing HR.
    usable = not unsupported and high-low <= .10*max(midpoint, 1e-9)
    return dict(entering_running_minutes=running, entering_cardio=midpoint if usable else None,
                entering_cardio_low=low, entering_cardio_high=high)


def route_cells(points):
    """Approximate 30 m route footprint; kept in memory, never emitted."""
    return {(math.floor(p.latitude*111320/30),
             math.floor(p.longitude*111320*math.cos(math.radians(40.7))/30))
            for p in points if p.gps_valid and p.latitude is not None and p.longitude is not None}


def route_overlap(a, b):
    if not a or not b:
        return None
    def hits(first, second):
        return sum(any((x+dx, y+dy) in second for dx in (-1, 0, 1) for dy in (-1, 0, 1)) for x, y in first)/len(first)
    return min(hits(a, b), hits(b, a))


def raw_context(connection, config):
    """Reconstruct recent histories without modifying the committed ledger."""
    history, features, routes = [], {}, {}
    reference = ContinuousDoseReference('Offline measured-context reconstruction')
    hr = HeartRateReference(config['resting_hr'], config['max_hr'], 'current fixed configured references')
    zone = ZoneInfo(config['timezone_default'])
    for a in connection.execute('''SELECT a.*,m.calculated_moving_time_s
        FROM activities a JOIN activity_metrics m ON m.activity_id=a.id
        WHERE a.sport='Running' AND a.start_time_utc>='2026-01-01' ORDER BY a.start_time_utc'''):
        at = datetime.fromisoformat(a['start_time_utc'])
        points = _load_points(connection, a['id'])
        movement = classify_movement(points, config['moving_time'])
        activity = SimpleNamespace(start_time_utc=at, trackpoints=points,
            total_elapsed_time_s=a['total_elapsed_time_s'], lap_recorded_time_s=a['lap_recorded_time_s'])
        trace = observed_cardio_trace(activity, movement.intervals, hr, reference.policy, compute_density=False)
        bounds = trace['load_bounds_within_reference']
        history.append(dict(activity_id=a['id'], start=at.isoformat(),
            end=(at+timedelta(seconds=trace['elapsed_seconds'])).isoformat(),
            local_date=at.astimezone(zone).date().isoformat(),
            moving_minutes=sum(i.moving_time_s for i in movement.intervals)/60,
            cardio_low=bounds['low'] if bounds else None, cardio_high=bounds['high'] if bounds else None,
            hr_coverage=trace['coverage']))
        moving = 0.
        early, settled, bins = [], [], [[], []]
        for interval in movement.intervals:
            if moving < 180:
                early.append(interval)
            if 300 <= moving < 600:
                settled.append(interval)
            bins[0 if moving < 300 else 1].append(interval)
            moving += interval.moving_time_s
        def speed(rows):
            seconds = sum(i.moving_time_s for i in rows)
            return sum(i.distance_m for i in rows)/seconds if seconds >= 120 else None
        def mean_hr(rows):
            pairs = [(i, (i.start.heart_rate_bpm+i.end.heart_rate_bpm)/2)
                for i in rows if finite(i.start.heart_rate_bpm) and finite(i.end.heart_rate_bpm)]
            seconds = sum(i.moving_time_s for i, _ in pairs)
            return sum(i.moving_time_s*v for i, v in pairs)/seconds if seconds >= 120 else None
        e, s, eh, sh = speed(early), speed(settled), mean_hr(early), mean_hr(settled)
        features[a['id']] = dict(
            early_speed_excess_percent=100*(e/s-1) if e and s else None,
            early_hr_rise=sh-eh if eh is not None and sh is not None else None,
            longest_recording_gap_seconds=max((i.elapsed_s for i in movement.intervals), default=None))
        routes[a['id']] = route_cells(points)
    return history, features, routes


def source_performance(connection, root):
    """Recover numeric source fields offline, including enhanced respiration.

    Only one newest FIT per physical activity; no importer or database mutation.
    """
    import fitdecode
    selected = {}
    for r in connection.execute('''SELECT s.activity_id,f.path,f.id FROM activity_sources s
        JOIN source_files f ON f.id=s.source_file_id ORDER BY f.id DESC'''):
        if str(r['path']).lower().endswith(('.fit', '.fit.gz')):
            selected.setdefault(r['activity_id'], r['path'])
    result = {}
    names = {'power', 'vertical_oscillation', 'stance_time', 'vertical_ratio', 'step_length',
             'respiration_rate', 'enhanced_respiration_rate'}
    for identity, relative in selected.items():
        records = []
        with fitdecode.FitReader(_open(resolve_project_path(root, relative))) as reader:
            for frame in reader:
                if frame.frame_type != fitdecode.FIT_FRAME_DATA or frame.name != 'record':
                    continue
                values = {f.name: f.value for f in frame.fields}
                timestamp = values.get('timestamp')
                if not isinstance(timestamp, datetime):
                    continue
                at = timestamp.replace(tzinfo=timezone.utc) if timestamp.tzinfo is None else timestamp
                row = {name: values[name] for name in names if finite(values.get(name))}
                # Prefer the enhanced field, never double-count the same sample.
                if 'enhanced_respiration_rate' in row:
                    row['respiration_rate'] = row.pop('enhanced_respiration_rate')
                if row:
                    records.append(dict(timestamp=at.timestamp(), **row))
        result[identity] = records
    return result


def sampled_performance(windows, records, config):
    estimates = defaultdict(list)
    for w in windows:
        start = datetime.fromisoformat(w['window_start_time_utc']).timestamp()
        end = datetime.fromisoformat(w['window_end_time_utc']).timestamp()
        selected = [r for r in records if start <= r['timestamp'] <= end]
        for name in ('power', 'vertical_oscillation', 'stance_time', 'vertical_ratio', 'step_length', 'respiration_rate'):
            values = [r[name] for r in selected if finite(r.get(name))]
            if len(values) >= .70*max(1, len(selected)) and values:
                estimates[name].append((float(np.mean(values)), _primary_window_weight(w, config)))
    return {name: float(np.average([v for v, _ in pairs], weights=[w for _, w in pairs]))
            for name, pairs in estimates.items()}


def later_check(rows, baseline, candidates, *, outcome='pace'):
    """Prefix-only coefficients/scales, previous observation anchored."""
    cases = []
    for i in range(24, len(rows)):
        training, target, anchor = rows[:i], rows[i], rows[i-1]
        if day(training[-1])-day(training[0]) < 56 or day(target)-day(anchor) > 14:
            continue
        errors = {}
        for name, keys in (('baseline', baseline), ('candidate', baseline+candidates)):
            model = fit(training, keys, outcome=outcome, draws=2)
            x, _ = design([anchor, target], keys, model['spec'])
            predicted = anchor[outcome]*math.exp(float((x[1]-x[0])@model['beta']))
            errors[name] = abs(predicted-target[outcome])*60
        cases.append(dict(activity_id=target['activity_id'], date=target['start_time_utc'], **errors))
    paired = paired_range([dict(c, baseline_absolute_error=c['baseline'],
        candidate_absolute_error=c['candidate']) for c in cases]) if cases else None
    if paired and paired['calendar_blocks'] < 3:
        paired['descriptive_95_block_range'] = None
        paired['range_unavailable_reason'] = 'Fewer than three independent calendar blocks'
    return dict(cases=len(cases), baseline_mae_seconds=float(np.mean([c['baseline'] for c in cases])) if cases else None,
        candidate_mae_seconds=float(np.mean([c['candidate'] for c in cases])) if cases else None,
        paired_error_range=paired,
        case_results=cases, at_least_ten_later_targets=len(cases) >= 10,
        validated_predictive_value=False)


def check_family(rows, keys, *, baseline=PRIMARY, outcome='pace'):
    cohort = [r for r in rows if all(finite(r.get(k)) for k in keys)]
    span = day(cohort[-1])-day(cohort[0]) if cohort else 0
    out = dict(runs=len(cohort), span_days=span, keys=list(keys), outcome=outcome,
        start=cohort[0]['start_time_utc'] if cohort else None,
        end=cohort[-1]['start_time_utc'] if cohort else None)
    if len(cohort) < 12 or span < 28:
        return dict(out, status='descriptive_only_insufficient_independent_runs_or_span')
    fitted = fit(cohort, baseline+keys, outcome=outcome)
    changes = {k: tuple(float(v) for v in np.quantile([r[k] for r in cohort], [.25, .75])) for k in keys}
    effect = contrast(cohort, fitted, baseline+keys, changes)
    if fitted['calendar_blocks'] < 3:
        effect['conditional_block_range_95'] = None
        effect['range_unavailable_reason'] = 'Fewer than three independent calendar blocks; do not present a narrow uncertainty interval'
    # contrast() scales to the cohort's comparable pace, also when examining
    # raw-section sensitivity; label the common reference scale explicitly.
    return dict(out, status='exploratory_conditional_association', quartile_contrast=changes,
        calendar_blocks=fitted['calendar_blocks'],
        association=effect, later=later_check(cohort, baseline, keys, outcome=outcome))


def matched_routes(rows, routes):
    candidates = []
    for i, a in enumerate(rows):
        for b in rows[i+1:]:
            gap = day(b)-day(a)
            if gap > 14:
                break
            if a['device'] != b['device'] or a['workout_type'] != b['workout_type']:
                continue
            overlap = route_overlap(routes.get(a['activity_id']), routes.get(b['activity_id']))
            if overlap is None or overlap < .60:
                continue
            if any(not finite(a.get(k)) or not finite(b.get(k)) or abs(a[k]-b[k]) > threshold
                   for k, threshold in (('sampled_hr_bpm', 5), ('heat_c', 3), ('duration_minutes', 15), ('headwind_mph', 4))):
                continue
            candidates.append((gap, -overlap, a, b))
    used, pairs = set(), []
    for gap, negative_overlap, a, b in sorted(candidates, key=lambda p: (p[0], p[2]['activity_id'], p[3]['activity_id'])):
        if a['activity_id'] in used or b['activity_id'] in used:
            continue
        used.update((a['activity_id'], b['activity_id']))
        pairs.append(dict(earlier_id=a['activity_id'], later_id=b['activity_id'], gap_days=gap,
            approximate_overlap=-negative_overlap, pace_change_seconds=60*(b['pace']-a['pace']),
            sleep_hours_change=(b['sleep_hours']-a['sleep_hours']) if all(finite(r.get('sleep_hours')) for r in (a,b)) else None,
            reconstructed_fatigue_change=(b['entering_cardio']-a['entering_cardio']) if all(finite(r.get('entering_cardio')) for r in (a,b)) else None))
    return dict(candidate_pairs=len(candidates), independent_pairs=len(pairs), pairs=pairs,
        interpretation='Descriptive context-matched route pairs, selected without pace or sleep outcomes; approximate route geometry is not matched surface/shade')


def build(connection, root, config, frozen, outliers, rr_pilot):
    history, raw, routes = raw_context(connection, config)
    prepared = prepare_progress_data(connection)
    now = datetime.now(timezone.utc)
    ledger = _canonical_load(connection, root, prepared.sessions, now)
    model = ledger.get('fatigue_model')
    found = {r['activity_id'] for r in ledger['rows']}
    wellness = read_wellness(root) or {}
    series = {r['date']: r for r in wellness.get('series', [])}
    dates = {r['id']: r['start_time_local'][:10] for r in connection.execute('SELECT id,start_time_local FROM activities')}
    windows, _ = load_overlapping_model_windows(connection, config,
        window_seconds=config['model']['primary_window_seconds'],
        stride_seconds=config['model']['primary_window_stride_seconds'], check_elevation=True)
    grouped = defaultdict(list)
    for w in windows:
        grouped[w['activity_id']].append(w)
    performances = source_performance(connection, root)
    previous_audit = {r['activity_id']: r for r in outliers['runs']}
    rr = {r['activity_id']: r for r in rr_pilot['runs']}
    rows = []
    for original in frozen['observations']:
        if original['start_time_utc'] < '2026-03-01':
            continue
        r = dict(original)
        identity = r['activity_id']
        at = datetime.fromisoformat(r['start_time_utc'])
        local_date = date.fromisoformat(dates[identity])
        r.update(raw.get(identity, {}), **sleep_context(series, local_date))
        r.update(entering_load(history, at, timezone_name=config['timezone_default']))
        r['scheduler_fatigue'] = None
        if model:
            day0 = at.astimezone(ZoneInfo(model['timezone'])).date()
            lookback = math.ceil(5*model['scenario']['memory_days'])
            start = datetime.fromisoformat(model['history_start']).astimezone(ZoneInfo(model['timezone'])).date()
            missing = any(h['activity_id'] not in found and day0-timedelta(days=lookback) <= date.fromisoformat(h['local_date']) < day0 for h in history)
            if start <= day0-timedelta(days=lookback) and not missing:
                r['scheduler_fatigue'] = residual(ledger['rows'], day0, model['scales'],
                    DemandScenario(**model['scenario']), ZoneInfo(model['timezone']), entering=True)['value']
        available = grouped[identity]
        selected = [available[i] for i in _pre_quality_indexes(available, list(range(len(available))), None)]
        r['source_performance'] = sampled_performance(selected, performances.get(identity, []), config)
        audit = previous_audit.get(identity)
        near = audit['windows']['scenarios']['near_twenty'] if audit else None
        r['near_twenty_pace'] = near['pace_min_mile'] if near else None
        r['retrospective_local_residual_seconds'] = audit['loo_trend_residual_seconds_per_mile'] if audit else None
        r['exercise_alpha1'] = rr.get(identity, {}).get('median_alpha1')
        r['recorder_period'] = '265' if '265' in r['device'] else '245' if '245' in r['device'] else 'other'
        rows.append(r)
    checks = {name: check_family(rows, keys) for name, keys in FAMILIES.items()}
    for name in ('reconstructed_running_exposure', 'reconstructed_cardio_exposure',
                 'reconstructed_entering_load', 'sleep_duration', 'early_pacing', 'early_hr_settling'):
        checks[name]['raw_section_sensitivity'] = check_family(rows, FAMILIES[name], outcome='observed')
    checks['early_pacing']['separate_later_section_sensitivity'] = check_family(
        [r for r in rows if finite(r['near_twenty_pace'])], FAMILIES['early_pacing'], outcome='near_twenty_pace')
    sleep_periods = {period: {
        'sleep': check_family([r for r in rows if r['recorder_period'] == period], ('sleep_hours',)),
        'observations': [dict(activity_id=r['activity_id'], date=dates[r['activity_id']],
            sleep_hours=r['sleep_hours'], adjusted_pace=r['pace'], residual_seconds=r['retrospective_local_residual_seconds'])
            for r in rows if r['recorder_period'] == period and finite(r['sleep_hours'])]}
        for period in ('245', '265')}
    # Source-quality sensitivity, not a learned cutoff or deletion. All raw
    # reported durations remain in rows and in the primary results above.
    sleep_quality = dict(
        flag='Reported sleep over 14 h; broad inspection flag, not a clinical maximum or production filter',
        flagged_activity_ids=[r['activity_id'] for r in rows if finite(r['sleep_hours']) and r['sleep_hours'] > 14],
        excluding_flagged_sensitivity=check_family(
            [r for r in rows if finite(r['sleep_hours']) and r['sleep_hours'] <= 14], ('sleep_hours',)))
    sensors = [dict(activity_id=r['activity_id'], date=dates[r['activity_id']],
        adjusted_pace=r['pace'], sampled_hr=r['sampled_hr_bpm'], temperature_f=r['temperature_f'],
        duration_minutes=r['duration_minutes'], workout_type=r['workout_type'], **r['source_performance'])
        for r in rows if r['source_performance']]
    pairs = matched_routes(rows, routes)
    for identity in (33, 34):
        if identity in routes:
            prior = [r for r in rows if r['activity_id'] != identity and abs(day(r)-day(next(a for a in rows if a['activity_id'] == identity))) <= 21]
            pairs[f'july_{identity}_best_nearby_route_overlap'] = sorted(
                [dict(activity_id=r['activity_id'], overlap=route_overlap(routes[identity], routes.get(r['activity_id'])))
                    for r in prior if routes.get(r['activity_id'])], key=lambda r: -r['overlap'])[:3]
    return dict(version='measured-performance-context-v1', source_database_read_only=True,
        live_changed=False, planner_called=False, family_checks=checks, sleep_recorder_periods=sleep_periods,
        route_checks=pairs, source_performance=sensors, rows=rows, sleep_source_quality=sleep_quality,
        reconstructed_load_history=history, committed_fatigue_model=model,
        exercise_hrv=dict(available_rr_runs=rr_pilot['available_rr_runs'], comparable_runs=rr_pilot['comparable_runs'],
                          matched_segments=rr_pilot['matched_segments']),
        reported_context=dict(
            post_run_effort_observations=sum(finite(r.get('perceived_exertion')) for r in rows),
            post_run_feeling_observations=sum(finite(r.get('workout_feel')) for r in rows),
            illness_recovery_rows=[dict(activity_id=r['activity_id'], date=dates[r['activity_id']],
                adjusted_pace=r['pace'], residual_seconds=r['retrospective_local_residual_seconds'])
                for r in rows if r.get('health_tag') == 'illness_recovery'],
            interpretation='Post-run ratings are descriptive responses, not independent pre-run predictors. Illness-recovery flags cluster in one late-July/early-August episode.'),
        assumptions=[
            'Primary entering-load memory is the existing two-day exponential e-folding; no decay fitted to outcomes.',
            'Reconstructed duration and continuous HR are two separate axes; no invented quality-work axis before the committed ledger.',
            'All known earlier completed local dates contribute. HR estimate withheld for broad missing bounds or unsupported recent traces.',
            'Recorded sleep date is the primary retrospective join; previous date is a disclosed sensitivity, never selected for better fit.',
            'Three-night sleep requires three actual values; no missing-night imputation.',
            'Sleep is checked pooled with recorder control and separately within 245/265 recording periods; upstream sleep device is not verified by the API.',
            'Fixed 56-day intercept/slope controls, fixed ridge 4, earlier-only feature scaling and coefficients; no feature/horizon selection.',
            'Later checks start after 24 earlier complete-family observations spanning 56 days; fewer than ten targets is explicitly limited.',
            'Canonical pace normalization is frozen full-history; later checks are conditional associations, not full as-known-then forecasts.',
            'Only running dates are independent observations; source duplicates and overlapping sensor windows are not additional runs.',
            'Early pacing/HR settling and breathing/dynamics are same-run diagnostics, not demonstrated pre-run predictors.',
            'A separate near-20-minute outcome checks early-pacing overlap; HR drift is not used to normalize away fitness.',
            'Route matching uses approximate 30m footprints with adjacent-cell tolerance, same role/recorder and fixed context tolerances.',
            'Power, stride length and cadence are not independent causal corrections for pace; sparse respiration/HRV remain descriptive.',
            'All candidate families and sensitivities are reported. Multiple exploratory comparisons are not independent confirmatory discoveries.',
            'No medication model, live corrections, planner evaluation, network requests, wellness refresh or DB writes.'
        ])


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rr-report', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('New output path required')
    root = Path.cwd()
    inputs = [root/'output/relative-weather-with-training-temperature-20261015.json',
              root/'output/comparable-run-outlier-audit-events-20261015.json', args.rr_report]
    frozen, outliers, rr = [json.loads(p.read_text()) for p in inputs]
    config = load_config(root/'config.yaml')
    database = resolve_project_path(root, config['paths']['database'])
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True) as connection:
        connection.row_factory = sqlite3.Row
        with invariant_snapshot(connection):
            result = build(connection, root, config, frozen, outliers, rr)
            assert connection.total_changes == 0
    result['input_sha256'] = {p.name: sha256(p.read_bytes()).hexdigest() for p in inputs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as target:
        json.dump(result, target, indent=2, allow_nan=False)
    private_file(args.output)
    print(json.dumps({name: dict(runs=value['runs'], status=value['status'],
        association=value.get('association'), later={k: v for k, v in value.get('later', {}).items()
            if k != 'case_results'}) for name, value in result['family_checks'].items()}, indent=2))


if __name__ == '__main__':
    main()
