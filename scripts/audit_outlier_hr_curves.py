"""Read-only outlier diagnostic with direct HR/time overlap, not extrapolation.

Both signs of leave-one-run-out local-trend residuals are ranked. Every eligible
recent run is diagnosed. Matching uses context, never pace agreement. Overlapping
windows do not count as independent runs; diagnostic sections are nonoverlapping.
"""
from argparse import ArgumentParser
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
import json
import math
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfo

import numpy as np

from run_analysis.aerobic_candidate import measurement_inputs
from run_analysis.config import load_config, resolve_project_path
from run_analysis.model_windows import load_model_window_sets
from run_analysis.objective_modeling import _wbgt_c
from run_analysis.pace_comparability import build_pace_comparability, _features
from run_analysis.privacy import private_file
from run_analysis.read_cache import invariant_snapshot
from scripts.aerobic_local_trend import LocalPaceTrend
from scripts.audit_comparable_outliers import raw_audit


def mean(values, weights=None):
    if not values:
        return None
    return float(np.average(values, weights=weights))


def after_warmup(row, minimum_minutes=5.):
    moving_seconds = row['elapsed_time_s']-row['stopped_time_s']
    return row['moving_minutes_into_run']-moving_seconds/120 >= minimum_minutes-1e-8


def rank_residuals(rows, calibration):
    covariance = np.zeros((4, 4))
    covariance[:3, :3] = calibration['covariance']
    covariance[3, 3] = calibration['heat']['sd']**2
    m, g = measurement_inputs(rows, covariance)
    origin = datetime.fromisoformat(rows[-1]['start_time'])
    times = np.array([(datetime.fromisoformat(r['start_time'])-origin).total_seconds()/86400 for r in rows])
    y = np.log([r['pace_min_mile'] for r in rows])
    output = {}
    for index, row in enumerate(rows):
        selected = np.arange(len(rows)) != index
        fit = LocalPaceTrend(y[selected], m[selected], g[selected], covariance, times[selected]).level(times[index])
        output[row['activity_id']] = 60*(row['pace_min_mile']-fit['pace_min_mile']) if fit['available'] else None
    return output


def section(row, config, calibration, midpoint):
    f = _features(row, config)
    speed = f['speed']*f['grade']*math.exp(calibration['parameters'][2]*f['wind_log'])
    speed *= math.exp(calibration['heat']['mean']*(f['heat'] or 0))
    minute = row['moving_minutes_into_run']
    try:
        heat = _wbgt_c(row)
    except (TypeError, ValueError, KeyError):
        heat = None
    return dict(minute=minute, hr=row['average_hr_bpm'], raw=row['moving_pace_min_mile'],
        environmental=1609.344/(60*speed),
        seconds=row['elapsed_time_s']-row['stopped_time_s'],
        stability=math.exp(-abs(row.get('heart_rate_change_bpm') or 0)/5
                           -abs(row.get('speed_change_fraction') or 0)/.05),
        half='early' if minute <= midpoint else 'late',
        temperature_f=row.get('temperature_f'), heat_c=heat,
        grade_ratio=f['grade'], wind_log=f['wind_log'],
        gps_coverage=row.get('gps_complete_fraction'),
        device_distance_coverage=row.get('device_distance_fraction'))


def matched_sections(target, peers, *, hr_tolerance=3., minute_tolerance=5., closer_conditions=False):
    result = []
    for current in target:
        matches = []
        for identity, windows in peers.items():
            options = [p for p in windows
                if abs(current['hr']-p['hr']) <= hr_tolerance
                and abs(current['minute']-p['minute']) <= minute_tolerance]
            if closer_conditions:
                options = [p for p in options if current.get('heat_c') is not None and p.get('heat_c') is not None
                    and abs(current['heat_c']-p['heat_c']) <= 3
                    and abs(math.log(current['grade_ratio']/p['grade_ratio'])) <= .03
                    and abs(current['wind_log']-p['wind_log']) <= .02]
            if not options:
                continue
            # Context/stability only: no pace, residual or favorable half.
            chosen = min(options, key=lambda p: (
                ((p['hr']-current['hr'])/hr_tolerance)**2
                +((p['minute']-current['minute'])/minute_tolerance)**2
                +.1*(1-p['stability']), p['minute']))
            matches.append(dict(activity_id=identity, **chosen))
        if len(matches) < 2:
            continue
        # Every peer run supplies one value, whatever its recording frequency.
        reference = {key: float(np.median([r[key] for r in matches])) for key in ('raw', 'environmental', 'hr', 'minute')}
        result.append(dict(current, peer=reference, peer_ids=[r['activity_id'] for r in matches],
            raw_difference_seconds=60*(current['raw']-reference['raw']),
            environmental_difference_seconds=60*(current['environmental']-reference['environmental'])))
    return result


def summarize_matches(matches):
    def summarize(rows):
        # A diagnostic of selection bias must not silently repeat the score's
        # preference for stable/near-reference sections. Use recorded minutes.
        weights = [r['seconds'] for r in rows]
        return dict(windows=len(rows), matched_minutes=sum(r['seconds'] for r in rows)/60,
            peer_runs=len({identity for r in rows for identity in r['peer_ids']}),
            hr_range=[min(r['hr'] for r in rows), max(r['hr'] for r in rows)] if rows else None,
            minute_range=[min(r['minute'] for r in rows), max(r['minute'] for r in rows)] if rows else None,
            raw_difference_seconds=mean([r['raw_difference_seconds'] for r in rows], weights) if rows else None,
            environmental_difference_seconds=mean([r['environmental_difference_seconds'] for r in rows], weights) if rows else None,
            raw_faster_fraction=mean([float(r['raw_difference_seconds'] < 0) for r in rows], weights) if rows else None,
            environmental_faster_fraction=mean([float(r['environmental_difference_seconds'] < 0) for r in rows], weights) if rows else None)
    return dict(all=summarize(matches), early=summarize([r for r in matches if r['half'] == 'early']),
                late=summarize([r for r in matches if r['half'] == 'late']))


def hr_bands(matches):
    result = []
    for low, high in ((128, 134), (134, 140), (140, 146), (146, 152), (152, 158), (158, 166.001)):
        selected = [r for r in matches if low <= r['hr'] < high]
        if selected:
            result.append(dict(low=low, high=high, **summarize_matches(selected)['all']))
    return result


def diagnose(connection, config):
    report = build_pace_comparability(connection, config)
    eligible = sorted([r for r in report['runs'] if r['included_in_trend'] and r['trend_weight'] > 0
                       and r['start_time'] >= '2026-03-01'], key=lambda r: r['start_time'])
    residuals = rank_residuals(eligible, report['calibration'])
    meta = {r['id']: dict(r) for r in connection.execute('''SELECT a.id,a.creator,a.start_time_local,
        m.calculated_moving_time_s FROM activities a JOIN activity_metrics m ON m.activity_id=a.id''')}
    # Only extend the time ceiling for diagnosis of the whole long run.
    # Original HR, movement, pause and coverage screens remain unchanged.
    broad = deepcopy(config)
    broad['model']['maximum_reliable_segment_minutes'] = 1440
    sets, counters = load_model_window_sets(connection, broad, (120,), check_elevation=True)
    grouped = defaultdict(list)
    for w in sets[120]:
        grouped[w['activity_id']].append(w)
    def recorder(identity):
        value = (meta[identity]['creator'] or 'unknown').lower()
        return '265' if '265' in value else '245' if '245' in value else value
    sections = {r['activity_id']: [section(w, config, report['calibration'],
                    (5+meta[r['activity_id']]['calculated_moving_time_s']/60)/2)
                    for w in grouped[r['activity_id']] if after_warmup(w)] for r in eligible}
    results = []
    for row in eligible:
        identity = row['activity_id']
        current_time = datetime.fromisoformat(row['start_time'])
        peers = {r['activity_id']: sections[r['activity_id']] for r in eligible
            if r['activity_id'] != identity and recorder(r['activity_id']) == recorder(identity)
            and abs((datetime.fromisoformat(r['start_time'])-current_time).total_seconds()) <= 28*86400}
        matches = matched_sections(sections[identity], peers)
        strict = matched_sections(sections[identity], peers, hr_tolerance=1., minute_tolerance=3.)
        closer = matched_sections(sections[identity], peers, closer_conditions=True)
        shorter = {key: windows for key, windows in peers.items()
            if abs((datetime.fromisoformat(next(r['start_time'] for r in eligible if r['activity_id'] == key))-current_time).total_seconds()) <= 14*86400}
        sensors = raw_audit(connection, identity, config)
        results.append(dict(activity_id=identity, date=meta[identity]['start_time_local'], recorder=recorder(identity),
            role=row['workout_type'], miles=row['distance_miles'], moving_minutes=meta[identity]['calculated_moving_time_s']/60,
            canonical_adjusted_pace=row['pace_min_mile'], canonical_recorded_sections_pace=row['observed_pace_min_mile'],
            net_adjustment_seconds=60*(row['pace_min_mile']-row['observed_pace_min_mile']),
            adjustment_contributions=row['contributions_seconds_per_mile'],
            loo_trend_residual_seconds=residuals[identity],
            whole_run=sensors['whole_run'], postwarmup_halves=sensors['postwarmup_halves'],
            recording_events=sensors['recording_events'],
            point_hr_coverage=sensors['point_hr_coverage'],
            largest_hr_step_within_ten_seconds=sensors['largest_hr_step_within_ten_seconds'],
            available_diagnostic_minutes=sum(s['seconds'] for s in sections[identity])/60,
            peer_candidate_runs=len(peers), matched=summarize_matches(matches),
            strict_hr_time=summarize_matches(strict), closer_environment=summarize_matches(closer),
            fourteen_day_peers=summarize_matches(matched_sections(sections[identity], shorter)),
            hr_bands=hr_bands(matches), sections=sections[identity], matched_sections=matches))
    ranked = sorted([r for r in results if r['loo_trend_residual_seconds'] is not None],
                    key=lambda r: r['loo_trend_residual_seconds'])
    return dict(version='outlier-direct-hr-curve-diagnostic-v1', live_changed=False,
        source_database_read_only=True, planner_called=False, calibration=report['calibration'],
        runs=results, strongest_ids=[r['activity_id'] for r in ranked[:8]],
        weakest_ids=[r['activity_id'] for r in ranked[-8:][::-1]],
        unranked_boundary_ids=[r['activity_id'] for r in results if r['loo_trend_residual_seconds'] is None],
        diagnostic_window_counters=counters,
        assumptions=[
            'Every eligible run since March is diagnosed; top eight per signed residual are reported, not only July 7.',
            'Outlier rank uses a leave-one-run-out retrospective 28-day local trend under the current harmonized calibration.',
            'The first/last point of an isolated history segment is not ranked by forced trend extrapolation.',
            'Diagnostic windows are nonoverlapping two-minute sections, after warmup, with the existing HR and movement screens.',
            'Only the original 60-minute ceiling is extended to inspect the whole long run; this is not a production acceptance change.',
            'Peers are other eligible physical runs from the same recorder family within 28 days, regardless of their score.',
            'Primary matching uses actual HR within 3 bpm and moving-run position within 5 minutes, at least two distinct peer runs per section.',
            'One closest-context section per peer run. Selection never uses pace or desired agreement.',
            'Strict HR/time (1 bpm/3 min), closer environment (heat 3 C, grade cost 3%, wind log cost .02), and 14-day peer sensitivities are all retained.',
            'Raw matched comparisons apply NO HR or time normalization. Environmental-only comparison adjusts terrain, wind and heat, still NO HR or time normalization.',
            'Early/late halves are split at half of the entire post-five-minute moving duration, not half of selected windows.',
            'Differences and consistency fractions are moving-minute weighted, without canonical target-HR/time or stability weights.',
            'Consistency fractions and HR/time ranges are descriptive support, not confidence intervals or physiological diagnoses.',
            'Sections and reused peer references are correlated; no window-count p-values or inference of independent repeated runs.',
            'Peer calendar changes, route/surface, sensor bias and incomplete environment remain; strong matching does not prove the cause or latent fitness.',
            'Clean recorded signals cannot independently establish absolute HR or GPS accuracy. No data deletion, reclassification or model correction.'
        ])


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('New output path required')
    config = load_config('config.yaml')
    database = resolve_project_path(Path.cwd(), config['paths']['database'])
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True) as connection:
        connection.row_factory = sqlite3.Row
        with invariant_snapshot(connection):
            result = diagnose(connection, config)
            assert connection.total_changes == 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as target:
        json.dump(result, target, indent=2, allow_nan=False)
    private_file(args.output)
    selected = set(result['strongest_ids']+result['weakest_ids']+[34])
    print(json.dumps([dict(id=r['activity_id'], date=r['date'], residual=r['loo_trend_residual_seconds'],
        raw_gap=r['matched']['all']['raw_difference_seconds'],
        environmental_gap=r['matched']['all']['environmental_difference_seconds'],
        matched_minutes=r['matched']['all']['matched_minutes'],
        early_gap=r['matched']['early']['raw_difference_seconds'],
        late_gap=r['matched']['late']['raw_difference_seconds'])
        for r in result['runs'] if r['activity_id'] in selected], indent=2))


if __name__ == '__main__':
    main()
