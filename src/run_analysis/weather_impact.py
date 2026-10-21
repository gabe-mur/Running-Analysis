"""Read-only weather/performance attribution prototype, not pace correction.

Run-level observations, calendar-local fitness nuisance trends, fixed controls,
and block uncertainty. There is deliberately no connection to the active pace
model or scheduler and no external weather/data request in this module.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import math
import sqlite3
from typing import Any

import numpy as np

from .model_windows import load_overlapping_model_windows
from .objective_modeling import (
    _primary_window_weight, _reference_time_support, _robust_weighted_speed,
    _shared_parameter_eligible, _weighted_mean, _wbgt_c,
    pace_to_speed_mps, speed_to_pace_min_mile,
)


CONTROLS = ('sampled_hr_bpm', 'sampled_minutes', 'grade_percent', 'headwind_mph',
            'log_duration', 'log_previous_7d_miles', 'log_days_since_hard')
MODEL_CONTRACT = 'weather-association-v1-calendar-local'


def _day(value: str) -> float:
    moment = datetime.fromisoformat(value)
    if moment.tzinfo is None:
        raise ValueError('Weather observations require timezone-aware run times')
    return moment.timestamp() / 86400.0


def load_weather_observations(connection: sqlite3.Connection, config: dict) -> tuple[list[dict], dict]:
    """Reproduce observed-pace aggregation, retaining measured context only."""
    windows, diagnostics = load_overlapping_model_windows(
        connection, config, window_seconds=int(config['model']['primary_window_seconds']),
        stride_seconds=int(config['model']['primary_window_stride_seconds']))
    grouped = defaultdict(list)
    for row in windows:
        grouped[row['activity_id']].append(row)
    duration = {r[0]: float(r[1]) / 60 if r[1] is not None else None
                for r in connection.execute('SELECT activity_id,calculated_moving_time_s FROM activity_metrics')}
    low, high = map(float, config['zones']['z2'])
    excluded, observations, candidate_weather = Counter(), [], []
    for activity_id, run in grouped.items():
        if not _shared_parameter_eligible(run[0]):
            excluded['quality_health_or_sensor_context'] += 1; continue
        try:
            # Coverage only, before Z2/reference-position selection. Do not
            # silently lose warmer or cooler candidates from the denominator.
            candidate_weather.append((float(np.mean([_wbgt_c(r) for r in run])), activity_id))
        except (ValueError, TypeError, KeyError):
            pass
        selected = [r for r in run if low <= r['average_hr_bpm'] <= high]
        if not selected:
            excluded['no_observed_z2_windows'] += 1; continue
        if not _reference_time_support([r['moving_minutes_into_run'] for r in selected],
                                      config['reference_conditions']['within_run_minutes'], config)[0]:
            excluded['reference_position_not_supported'] += 1; continue
        weights = [_primary_window_weight(r, config) for r in selected]
        def measured_mean(key):
            if any(r.get(key) is None for r in selected): return None
            return _weighted_mean([float(r[key]) for r in selected], weights)
        try:
            wbgt = _weighted_mean([_wbgt_c(r) for r in selected], weights)
        except (ValueError, TypeError, KeyError):
            wbgt = None
        minutes = duration.get(activity_id)
        previous_volume, days_since_hard = run[0].get('previous_7d_miles'), run[0].get('days_since_previous_hard_run')
        grade = measured_mean('average_grade_percent') if all(r.get('grade_energy_ratio') is not None for r in selected) else None
        pace = speed_to_pace_min_mile(_robust_weighted_speed(
            [pace_to_speed_mps(r['moving_pace_min_mile']) for r in selected], weights))
        observations.append(dict(activity_id=activity_id, start_time_utc=run[0]['start_time_utc'],
            observed_pace_min_mile=pace, wbgt_c=wbgt,
            temperature_f=measured_mean('temperature_f'), dewpoint_f=measured_mean('dewpoint_f'),
            sampled_hr_bpm=measured_mean('average_hr_bpm'), sampled_minutes=measured_mean('moving_minutes_into_run'),
            grade_percent=grade, headwind_mph=measured_mean('headwind_signed_mph'),
            log_duration=math.log(minutes) if minutes is not None and minutes > 0 else None,
            log_previous_7d_miles=math.log1p(previous_volume) if previous_volume is not None and previous_volume >= 0 else None,
            log_days_since_hard=math.log1p(days_since_hard) if days_since_hard is not None and days_since_hard >= 0 else None))
    selected_ids = {r['activity_id'] for r in observations}
    complete_ids = {r['activity_id'] for r in observations if all(r.get(k) is not None for k in CONTROLS)}
    weather_support = []
    if candidate_weather:
        for label, chunk in zip(('lower_heat_third', 'middle_heat_third', 'higher_heat_third'),
                                np.array_split(sorted(candidate_weather), 3)):
            if not len(chunk): continue
            ids = {int(row[1]) for row in chunk}
            weather_support.append({'group': label, 'candidate_runs': len(ids),
                'observed_z2_supported': len(ids & selected_ids), 'complete_context_runs': len(ids & complete_ids),
                'candidate_mean_wbgt_range_c': [float(np.min(chunk[:,0])), float(np.max(chunk[:,0]))]})
    total = connection.execute('SELECT count(*) FROM activities').fetchone()[0]
    return sorted(observations, key=lambda r: (r['start_time_utc'], r['activity_id'])), {
        'imported_activities': total, 'runs_with_window_candidates': len(grouped),
        'observed_aerobic_runs': len(observations), 'excluded_before_context_checks': dict(excluded),
        'support_by_candidate_weather_third': weather_support,
        'window_filter_diagnostics': diagnostics,
    }


def _complete(rows):
    selected, missing = [], Counter()
    seen = set()
    for row in rows:
        identity = row.get('activity_id')
        if identity is None:
            missing['missing_activity_identity'] += 1; continue
        if identity in seen:
            raise ValueError('Weather observations must contain each physical activity only once')
        seen.add(identity)
        try:
            _day(row['start_time_utc'])
        except (ValueError, TypeError, KeyError):
            missing['invalid_run_time'] += 1; continue
        required = ('observed_pace_min_mile', 'wbgt_c', *CONTROLS)
        absent = []
        for name in required:
            try:
                valid = row.get(name) is not None and math.isfinite(float(row[name]))
            except (ValueError, TypeError):
                valid = False
            if not valid: absent.append(name)
        if absent:
            missing.update(absent); continue
        if row['observed_pace_min_mile'] <= 0:
            missing['nonpositive_pace'] += 1; continue
        selected.append(row)
    return sorted(selected, key=lambda r: (_day(r['start_time_utc']), r['activity_id'])), dict(missing)


def _fit(rows, *, period=28, offset=0, bootstrap=2000):
    if not rows: return {'status': 'insufficient_observations'}
    day = np.array([_day(r['start_time_utc']) for r in rows])
    group = np.floor((day - offset) / period).astype(int)
    features = np.array([[float(r[key]) for key in CONTROLS] for r in rows])
    scales = features.std(axis=0)
    scales = np.where(scales > 1e-10, scales, 1.0)
    centered = (features - features.mean(axis=0)) / scales
    calendar = []
    for key in np.unique(group):
        mask = group == key
        calendar.append(mask.astype(float))
        calendar.append(np.where(mask, (day - day[mask].mean()) / period, 0))
    z = np.column_stack([*calendar, centered])
    y = np.log([r['observed_pace_min_mile'] for r in rows])
    heat = np.array([r['wbgt_c'] for r in rows])
    baseline = np.linalg.lstsq(z, y, rcond=None)[0]
    residual_heat = heat - z @ np.linalg.lstsq(z, heat, rcond=None)[0]
    residual_y = y - z @ baseline
    rank = int(np.linalg.matrix_rank(z))
    dof = len(rows) - rank - 1
    denominator = float(residual_heat @ residual_heat)
    summary = {'runs': len(rows), 'calendar_blocks': len(np.unique(group)), 'residual_degrees_of_freedom': dof,
               'nuisance_columns': int(z.shape[1]), 'nuisance_rank': rank,
               'remaining_weather_sd_c': float(residual_heat.std()),
               'weather_variance_remaining_fraction': float(np.var(residual_heat) / np.var(heat)) if np.var(heat) > 1e-12 else 0.0,
               'observed_weather_sd_c': float(heat.std()), 'period_days': period, 'offset_days': offset}
    if dof <= 0 or denominator <= max(1e-10, float(heat @ heat) * 1e-12):
        return {'status': 'weather_not_identifiable', **summary}
    beta = float(residual_heat @ residual_y / denominator)
    adjusted = np.linalg.lstsq(z, y - beta * heat, rcond=None)[0]
    residual = y - z @ adjusted - beta * heat
    # Cluster scores avoid treating windows or multiple contrasts as extra runs.
    scores = np.array([float(np.sum(residual_heat[group == key] * residual[group == key]))
                       for key in np.unique(group) if float(np.sum(residual_heat[group == key] ** 2)) > 1e-10])
    if len(scores) < 2:
        return {'status': 'insufficient_weather_support_blocks', **summary}
    correction = math.sqrt(len(scores) / (len(scores) - 1) * (len(rows) - 1) / dof)
    rng = np.random.default_rng(20261013)
    draws = beta + (rng.choice([-1, 1], size=(bootstrap, len(scores))) @ scores) / denominator * correction
    interval = np.quantile(draws, [.025, .975])
    reference_pace = float(np.median([r['observed_pace_min_mile'] for r in rows]))
    def effect(coefficient):
        # Association contrast, not a per-run weather credit or optimal climate.
        log_change = float(coefficient) * 5
        if abs(log_change) > 50: return None
        return {'pace_change_percent': math.expm1(log_change) * 100,
                'seconds_per_mile_at_observed_median': math.expm1(log_change) * reference_pace * 60}
    result = {'status': 'estimated_association', **summary, 'weather_support_calendar_blocks': len(scores),
              'log_pace_change_per_wbgt_c': beta, 'coefficient_interval_95': interval.tolist(),
              'five_c_warmer_association': effect(beta), 'five_c_warmer_interval_95': [effect(v) for v in interval],
              'association_includes_zero': bool(interval[0] <= 0 <= interval[1]),
              'reference_observed_pace_min_mile': reference_pace}
    # Both models use the same controls; their only difference is weather.
    return {**result, '_context_only': baseline[-len(CONTROLS):] / scales,
            '_context_with_weather': adjusted[-len(CONTROLS):] / scales}


def _public(fit):
    return {key: value for key, value in fit.items() if not key.startswith('_')}


def _chronological_contrasts(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[math.floor(_day(row['start_time_utc']) / 28)].append(row)
    keys = sorted(grouped)
    blocks, folds, pair_count, invalid_predictions = [], [], 0, 0
    for left, right in ((.5, .65), (.65, .8), (.8, 1)):
        start, end = int(len(keys) * left), int(len(keys) * right)
        training = [r for key in keys[:start] for r in grouped[key]]
        fit = _fit(training, bootstrap=200)
        folds.append({'training_runs': len(training), 'held_out_calendar_blocks': end-start, 'fit_status': fit['status']})
        if fit['status'] != 'estimated_association': continue
        for key in keys[start:end]:
            run = grouped[key]
            base_errors, weather_errors = [], []
            for i, first in enumerate(run):
                for second in run[i+1:]:
                    if abs(_day(second['start_time_utc']) - _day(first['start_time_utc'])) > 7: continue
                    heat = second['wbgt_c'] - first['wbgt_c']
                    if abs(heat) < 3: continue
                    delta = np.array([second[k] - first[k] for k in CONTROLS])
                    baseline = float(delta @ fit['_context_only'])
                    with_weather = float(delta @ fit['_context_with_weather']) + fit['log_pace_change_per_wbgt_c'] * heat
                    if abs(baseline) > 50 or abs(with_weather) > 50:
                        invalid_predictions += 1; continue
                    # Anchor the second run's prediction at the first run's
                    # observed pace; do not refit to the second run's outcome.
                    base_errors.append(abs(first['observed_pace_min_mile'] * math.exp(baseline) - second['observed_pace_min_mile']) * 60)
                    weather_errors.append(abs(first['observed_pace_min_mile'] * math.exp(with_weather) - second['observed_pace_min_mile']) * 60)
                    pair_count += 1
            if base_errors:
                blocks.append((float(np.mean(base_errors)), float(np.mean(weather_errors)), len(run)))
    if invalid_predictions:
        return {'status': 'unstable_held_out_predictions', 'invalid_predictions': invalid_predictions, 'folds': folds}
    if not blocks:
        return {'status': 'insufficient_held_out_weather_contrasts', 'folds': folds}
    baseline, weather = np.array(blocks)[:, 0], np.array(blocks)[:, 1]
    delta = weather - baseline
    rng = np.random.default_rng(20261013)
    interval = np.quantile(np.mean(rng.choice(delta, size=(2000, len(delta)), replace=True), axis=1), [.025, .975])
    return {'status': 'evaluated_relative_contrasts' if len(blocks)>1 else 'one_held_out_block_only',
            'held_out_blocks': len(blocks), 'pairs_not_independent': pair_count,
            'context_only_mean_absolute_contrast_error_seconds': float(baseline.mean()),
            'with_weather_mean_absolute_contrast_error_seconds': float(weather.mean()),
            'error_change_seconds': float(delta.mean()), 'block_bootstrap_interval_95': interval.tolist() if len(blocks)>1 else None, 'folds': folds,
            'limitation': 'Relative observed run contrasts, not future absolute pace. Fitness/readiness can change within seven days.'}


def analyze_weather_impact(observations: list[dict[str, Any]]) -> dict:
    rows, missing = _complete(observations)
    primary = _fit(rows)
    sensitivity = [_public(_fit(rows, period=period, offset=offset)) for period, offset in ((28,14), (56,0))]
    cv = _chronological_contrasts(rows)
    fits = [primary, *sensitivity]
    all_estimated = all(f['status'] == 'estimated_association' for f in fits)
    intervals = [f['coefficient_interval_95'] for f in fits if f['status'] == 'estimated_association']
    boundary_agreement = all_estimated and (all(low > 0 for low, high in intervals) or all(high < 0 for low, high in intervals))
    verified_gain = cv.get('status') == 'evaluated_relative_contrasts' and cv['block_bootstrap_interval_95'][1] < 0
    # Even consistent prediction is observational evidence, not causal attribution.
    status = 'supported_observational_association' if boundary_agreement and verified_gain else 'uncertain_association'
    if primary['status'] != 'estimated_association': status = 'not_estimable'
    return {'contract': MODEL_CONTRACT, 'status': status, 'observed_runs': len(observations), 'complete_context_runs': len(rows),
            'missing_context_counts_not_exclusive': missing, 'controls': list(CONTROLS), 'primary': _public(primary),
            'calendar_sensitivity': sensitivity, 'chronological_validation': cv,
            'boundaries_agree_on_nonzero_direction': bool(boundary_agreement),
            'adjusted_trend_promoted': False, 'interpretation': (
                'A weather association survives the fixed calendar sensitivities and held-out contrast check; it is not proof of causation.'
                if status == 'supported_observational_association' else
                'The available data do not yet separate a reliable weather contribution from time and run context.'),
            'limitations': ['Estimated shade WBGT omits direct solar exposure and street-level wind differences.',
                           'Calendar-local linear trends do not measure latent fitness or eliminate all seasonal confounding.',
                           'Prior mileage and time since hard running are context proxies, not measured fatigue.',
                           'Conditional association at comparable HR is not the total causal effect of heat.',
                           'Weather response is approximated as linear over the observed range; a nonlinear or acclimation-dependent effect may be missed.',
                           'Block intervals are conditional on this model; unmeasured sleep, health, sunlight and sensor changes are not fully represented.',
                           'Adjacent calendar blocks can still share training patterns; block resampling does not guarantee independence.',
                           'No per-run pace credit, VO2 recalibration or scheduler input is published.']}
