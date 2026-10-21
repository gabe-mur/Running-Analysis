"""Offline factorial audit of run sampling and post-pause sensitivity.

Frozen shared calibration isolates sampling from refitting. Held-out run halves
test representativeness, NOT accuracy against independently measured fitness.
No variant is selected for making the history smoother or outliers disappear.
"""
from argparse import ArgumentParser
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sqlite3

import numpy as np

from run_analysis.config import load_config, resolve_project_path
from run_analysis.model_windows import load_overlapping_model_windows, _pause_restarts
from run_analysis.movement import classify_movement
from run_analysis.objective_modeling import _primary_window_weight
from run_analysis.pace_comparability import build_pace_comparability, compare_run_windows
from run_analysis.processing import _load_points
from run_analysis.privacy import private_file
from run_analysis.read_cache import invariant_snapshot
from scripts.audit_outlier_hr_curves import after_warmup


VARIANTS = {
    'current': ('current', 'existing'),
    'no_time_preference': ('no_time', 'existing'),
    'broad_stable': ('stable', 'existing'),
    'moving_minutes': ('minutes', 'existing'),
    'broad_after_full_warmup': ('stable', 'full_warmup'),
    'current_pause_5m': ('current', 'pause_5'),
    'current_pause_8m': ('current', 'pause_8'),
    'current_before_first_pause': ('current', 'before_pause'),
    'broad_before_first_pause': ('stable', 'before_pause'),
}


def bounds(row):
    half = (row['elapsed_time_s']-row['stopped_time_s'])/120
    return row['moving_minutes_into_run']-half, row['moving_minutes_into_run']+half


def weight(row, config, mode):
    if mode == 'current':
        return _primary_window_weight(row, config)
    settings = config['model']
    moving = row['elapsed_time_s']-row['stopped_time_s']
    if mode == 'minutes':
        return moving
    stability = math.exp(-abs(row.get('heart_rate_change_bpm') or 0)/settings['primary_hr_change_weight_scale_bpm']
                         -abs(row.get('speed_change_fraction') or 0)/settings['primary_speed_change_weight_scale_fraction'])
    stop_fraction = row['stopped_time_s']/max(1., row['elapsed_time_s'])
    # Keep the original non-time factors exactly: multiplying by duration here
    # would confound removal of time/HR relevance when window lengths vary.
    value = stability*max(.05, (1-stop_fraction)**2)
    if mode == 'no_time':
        value *= math.exp(-.5*((row['average_hr_bpm']-config['target_hr'])/settings['primary_hr_weight_scale_bpm'])**2)
    elif mode != 'stable':
        raise ValueError('Unknown weight mode')
    return max(1e-8, value)


def select(rows, restarts, rule):
    if rule == 'existing':
        return list(rows)
    if rule == 'full_warmup':
        return [r for r in rows if after_warmup(r)]
    if rule == 'before_pause':
        # A sensitivity bound, not a claim that every later section is invalid.
        return [r for r in rows if not restarts or bounds(r)[1] <= min(restarts)+1e-8]
    if rule not in ('pause_5', 'pause_8'):
        raise ValueError('Unknown pause rule')
    duration = 5 if rule == 'pause_5' else 8
    return [r for r in rows if not any(bounds(r)[0] < restart+duration and bounds(r)[1] > restart
                                      for restart in restarts)]


def estimate(rows, config, calibration, mode):
    if not rows:
        return None
    return compare_run_windows(rows, [weight(r, config, mode) for r in rows], config, calibration)


def support_minutes(rows):
    """Union of windows: overlapping observations are not extra running time."""
    total = 0.
    end = -math.inf
    for low, high in sorted(bounds(r) for r in rows):
        total += max(0., high-max(low, end))
        end = max(end, high)
    return total


def half_check(rows, moving_minutes, config, calibration, mode, *, target_rows=None):
    midpoint = (5+moving_minutes)/2
    # One-minute guard either side; no train/test windows share trackpoints.
    halves = [[r for r in rows if bounds(r)[1] <= midpoint-1],
              [r for r in rows if bounds(r)[0] >= midpoint+1]]
    reference = rows if target_rows is None else target_rows
    target_halves = [[r for r in reference if bounds(r)[1] <= midpoint-1],
                     [r for r in reference if bounds(r)[0] >= midpoint+1]]
    if any(support_minutes(h) < 4 for h in halves+target_halves):
        return None
    estimates = [estimate(h, config, calibration, mode)['pace_min_mile'] for h in halves]
    # Target is independently held-out MINUTE-weighted normalized evidence,
    # not another reference/stability-weighted score. Calibration stays frozen.
    targets = [estimate(h, config, calibration, 'minutes')['pace_min_mile'] for h in target_halves]
    return dict(early_to_late_error_seconds=60*(estimates[0]-targets[1]),
                late_to_early_error_seconds=60*(estimates[1]-targets[0]),
                mean_absolute_error_seconds=30*(abs(estimates[0]-targets[1])+abs(estimates[1]-targets[0])),
                early_late_score_gap_seconds=60*(estimates[1]-estimates[0]),
                early_minutes=support_minutes(halves[0]), late_minutes=support_minutes(halves[1]))


def gain_check(rows, config, calibration, mode):
    base = estimate(rows, config, calibration, mode)['pace_min_mile']
    results = []
    for fraction in (-.03, .03):
        shifted = [dict(r, moving_pace_min_mile=r['moving_pace_min_mile']/(1+fraction)) for r in rows]
        changed = estimate(shifted, config, calibration, mode)['pace_min_mile']
        # Physical wind cost depends on pace. Keep that small interaction visible.
        results.append(dict(planted_speed_percent=100*fraction,
                            recovered_speed_percent=100*(base/changed-1)))
    return results


def paired_summary(runs, name):
    pairs = [(r['variants']['current'], r['variants'][name]) for r in runs
             if r['variants']['current'] is not None and r['variants'][name] is not None]
    shifts = [b['delta_from_current_seconds'] for a, b in pairs]
    checks = [(a['half_check'], b['half_check']) for a, b in pairs if a['half_check'] and b['half_check']]
    result = dict(available_runs=len(pairs), missing_runs=len(runs)-len(pairs),
                  median_shift_seconds=float(np.median(shifts)) if shifts else None,
                  median_absolute_shift_seconds=float(np.median(np.abs(shifts))) if shifts else None,
                  maximum_absolute_shift_seconds=max(map(abs, shifts), default=None),
                  retained_union_minutes=sum(b['support_minutes'] for a, b in pairs),
                  paired_half_check_runs=len(checks))
    if checks:
        a = np.array([a['mean_absolute_error_seconds'] for a, b in checks])
        b = np.array([b['mean_absolute_error_seconds'] for a, b in checks])
        result['half_check'] = dict(current_mean_error_seconds=float(a.mean()),
            variant_mean_error_seconds=float(b.mean()), mean_change_seconds=float((b-a).mean()),
            median_change_seconds=float(np.median(b-a)), improved_runs=int(np.sum(b<a-1e-8)),
            worsened_runs=int(np.sum(b>a+1e-8)))
    return result


def audit(connection, config, frozen):
    report = build_pace_comparability(connection, config)
    calibration = report['calibration']
    rows, counters = load_overlapping_model_windows(connection, config,
        window_seconds=int(config['model']['primary_window_seconds']),
        stride_seconds=int(config['model']['primary_window_stride_seconds']), check_elevation=True)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['activity_id']].append(row)
    original = {r['activity_id']: r for r in report['runs'] if r['included_in_trend'] and r['trend_weight'] > 0
                and r['start_time'] >= '2026-03-01'}
    # Fixed outlier cohort from the prior diagnostic, not reselected for results.
    frozen_ids = {r['activity_id'] for r in frozen['runs']}
    if set(original) != frozen_ids:
        raise ValueError('Evidence changed since frozen diagnostic; review cohort before comparing')
    np.testing.assert_allclose(calibration['parameters'], frozen['calibration']['parameters'], rtol=1e-12, atol=1e-12)
    frozen_rows = {r['activity_id']: r for r in frozen['runs']}
    results = []
    for identity, baseline in sorted(original.items(), key=lambda pair: pair[1]['start_time']):
        windows = grouped[identity]
        current = estimate(windows, config, calibration, 'current')
        # Eligible easy/long rows must reproduce live comparison exactly.
        np.testing.assert_allclose(current['pace_min_mile'], baseline['pace_min_mile'], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(current['pace_min_mile'], frozen_rows[identity]['canonical_adjusted_pace'],
                                   rtol=1e-12, atol=1e-12)
        movement = classify_movement(_load_points(connection, identity), config['moving_time'])
        restarts = [s/60 for s in _pause_restarts(movement.intervals, config)]
        moving = sum(i.moving_time_s for i in movement.intervals)/60
        variants = {}
        for name, (mode, rule) in VARIANTS.items():
            selected = select(windows, restarts, rule)
            result = estimate(selected, config, calibration, mode)
            if result is None:
                variants[name] = None
                continue
            variants[name] = dict(pace_min_mile=result['pace_min_mile'],
                observed_pace_min_mile=result['observed_pace_min_mile'],
                delta_from_current_seconds=60*(result['pace_min_mile']-baseline['pace_min_mile']),
                support_minutes=support_minutes(selected), windows=len(selected),
                sampled_hr_bpm=result['sampled_hr_bpm'], contributions=result['contributions_seconds_per_mile'],
                half_check=half_check(selected, moving, config, calibration, mode, target_rows=windows),
                planted_changes=gain_check(selected, config, calibration, mode))
        prior = frozen_rows[identity]
        results.append(dict(activity_id=identity, date=prior['date'], moving_minutes=moving,
            role=baseline['workout_type'], pause_restart_moving_minutes=restarts,
            canonical_residual_seconds=prior['loo_trend_residual_seconds'],
            direct_raw_match=prior['matched'], variants=variants))
        all_paces = [v['pace_min_mile'] for v in variants.values() if v]
        weight_paces = [variants[k]['pace_min_mile'] for k in ('current', 'no_time_preference', 'broad_stable', 'moving_minutes')]
        results[-1]['sampling_sensitivity_span_seconds'] = 60*(max(weight_paces)-min(weight_paces))
        results[-1]['all_sensitivity_span_seconds'] = 60*(max(all_paces)-min(all_paces))
    summary = {name: paired_summary(results, name) for name in VARIANTS}
    paused = [r for r in results if r['pause_restart_moving_minutes']]
    gains = [g for r in results for v in r['variants'].values() if v for g in v['planted_changes']]
    return dict(version='standardization-sampling-experiment-v1', deployed=False,
        source_database_read_only=True, planner_called=False, live_changed=False,
        cohort_runs=len(results), paused_runs=len(paused), calibration=calibration,
        exact_current_reproduction=True, variants=VARIANTS, summary=summary,
        paused_summary={name: paired_summary(paused, name) for name in VARIANTS},
        planted_gain_maximum_error_percentage_points=max(abs(g['planted_speed_percent']-g['recovered_speed_percent']) for g in gains),
        fixed_strongest_ids=frozen['strongest_ids'], fixed_weakest_ids=frozen['weakest_ids'],
        runs=results, window_counters=counters, assumptions=[
            'Primary estimand remains common HR and reference minute; 60-minute ceiling and HR bounds unchanged.',
            'Shared calibration frozen across variants, isolating sampling from refitting; retrospective not prospective validation.',
            'Current reproduced exactly on every eligible run; role/health inclusion unchanged.',
            'No-time removes only time relevance; broad-stable also removes target-HR relevance but retains stability and stops.',
            'Moving-minute-only is a sensitivity, not an assertion that unstable HR sections are equally reliable.',
            'Whole-window-after-five-minutes is separated from weighting, not silently bundled into a proposed fix.',
            'Existing >=60-second stop detection retained. Five/eight-minute exclusions are unvalidated sensitivities, not chosen physiology constants.',
            'Before-first-pause is a continuous-segment sensitivity bound, NOT a claim that all later evidence is invalid.',
            'Held-out halves have a one-minute guard on each side; at least four minutes union support in each half.',
            'Held-out error tests representation of other normalized sections, not measured fitness; real duration deterioration can increase error.',
            'Held-out target rows remain identical original windows for all variants, including pause/warmup exclusions; only training sampling changes.',
            'Runs, not windows, receive equal weight in mean half-check errors; no physiological CI or window-count p-values.',
            'Planted +/-3% speed shifts are mechanical preservation checks with fixed calibration; they do not validate HR/environment models.',
            'Prior outlier signs/IDs kept fixed; no trimming clean outliers or optimizing graph smoothness.',
            'Medication, sleep, fatigue, illness and other speculative corrections not added.'
        ])


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frozen', type=Path, default=Path('output/outlier-hr-curve-diagnostic-final-20261015.json'))
    args = parser.parse_args()
    if args.output.exists():
        parser.error('New output path required')
    root = Path(__file__).resolve().parents[1]
    config = load_config(root/'config.yaml')
    database = resolve_project_path(root, config['paths']['database'])
    frozen = json.loads(args.frozen.read_text())
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True) as connection:
        connection.row_factory = sqlite3.Row
        with invariant_snapshot(connection):
            result = audit(connection, config, frozen)
            assert connection.total_changes == 0
    result['frozen_diagnostic_sha256'] = hashlib.sha256(args.frozen.read_bytes()).hexdigest()
    result['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as target:
        json.dump(result, target, indent=2, allow_nan=False)
    private_file(args.output)
    print(json.dumps({k: result[k] for k in ('cohort_runs', 'paused_runs', 'exact_current_reproduction',
        'summary', 'paused_summary', 'planted_gain_maximum_error_percentage_points')}, indent=2))


if __name__ == '__main__':
    main()
