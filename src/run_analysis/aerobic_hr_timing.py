"""One run-level pace inferred jointly from its chronological HR response.

Observed HR initializes a continuous bout, not each scored window. Known stops
and unknown gaps reset at the next actual measured HR rather than manufacturing
a decay toward sleeping resting HR. Every readable moving section contributes.
"""
from collections import defaultdict
import math

import numpy as np

from .aerobic_response_evidence import TAUS

VERSION = 'observed-initial-hr-timing-v1'


def statistics(blocks, *, evaluation_seconds=120., group_ids=None):
    """Exact piecewise-constant response; compare at a fixed evaluation scale.

    Prediction is X * inverse_reference_speed + initial_HR * exp(-t/tau).
    Input microblocks should be shorter than the evaluation scale. Measured
    internal HR never teacher-forces the state, except at genuine bout starts.
    """
    if not blocks or not math.isfinite(evaluation_seconds) or evaluation_seconds <= 0:
        raise ValueError('Observed chronological microblocks and positive scale required')
    if group_ids is not None and len(group_ids) != len(blocks):
        raise ValueError('One explicit evaluation group per microblock required')
    parts = defaultdict(lambda: dict(seconds=0., observed=0., x=np.zeros(len(TAUS)), offset=np.zeros(len(TAUS))))
    x = np.zeros(len(TAUS)); offset = np.zeros(len(TAUS)); end = None
    for ordinal, b in enumerate(blocks):
        if b.first_hr_effort is None:
            raise ValueError('Measured bout-initial HR required')
        if end is not None and b.start < end-1e-6:
            raise ValueError('Ordered nonoverlapping microblocks required')
        if end is None or abs(b.start-end) > 1e-6:
            x[:] = 0.; offset[:] = b.first_hr_effort
        v = b.effort_distance/b.seconds
        means_x = np.zeros(len(TAUS)); means_offset = np.zeros(len(TAUS))
        for i, tau in enumerate(TAUS):
            if tau == 0:
                means_x[i] = x[i] = v; means_offset[i] = offset[i] = 0.
            else:
                a = math.exp(-b.seconds/tau)
                integral_fraction = tau/b.seconds*(-math.expm1(-b.seconds/tau))
                means_x[i] = v+(x[i]-v)*integral_fraction
                means_offset[i] = offset[i]*integral_fraction
                x[i] = v+(x[i]-v)*a; offset[i] *= a
        # Allocate a microblock wholly by its midpoint. No cross-run or
        # overlapping-window votes; one physical run supplies one loss.
        key = int((b.start+b.seconds/2)/evaluation_seconds) if group_ids is None else group_ids[ordinal]
        part = parts[key]
        part['seconds'] += b.seconds; part['observed'] += b.hr_area
        part['x'] += means_x*b.seconds; part['offset'] += means_offset*b.seconds
        end = b.start+b.seconds
    dt = np.array([p['seconds'] for p in parts.values()])
    h = np.array([p['observed'] for p in parts.values()])/dt
    x = np.array([p['x'] for p in parts.values()])/dt[:, None]
    offset = np.array([p['offset'] for p in parts.values()])/dt[:, None]
    return dict(seconds=dt, observed=h, x=x, offset=offset)


def fit(stats, mask=None):
    dt, h, x, offset = [stats[k] for k in ('seconds', 'observed', 'x', 'offset')]
    mask = np.ones(len(dt), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if mask.shape != dt.shape or not np.any(mask):
        return None
    w = dt[mask]/dt[mask].sum()
    denominator = np.sum(w[:, None]*x[mask]**2, axis=0)
    if np.any(denominator <= 0):
        return None
    q = np.sum(w[:, None]*x[mask]*(h[mask, None]-offset[mask]), axis=0)/denominator
    residual = h[:, None]-(x*q+offset)
    return dict(inverse_reference_speed=q,
        losses=np.average(residual[mask]**2, axis=0, weights=w), residual=residual,
        valid=np.isfinite(q) & (q > 0))


def cross_half_loss(stats, index):
    midpoint = stats['seconds'].sum()/2
    cumulative = np.cumsum(stats['seconds'])
    early = cumulative <= midpoint
    late = cumulative-stats['seconds'] >= midpoint
    outputs = []
    for source, target in ((early, late), (late, early)):
        fitted = fit(stats, source)
        if fitted is not None and np.any(target) and fitted['valid'][index]:
            outputs.append(float(np.average(fitted['residual'][target, index]**2, weights=stats['seconds'][target])))
    return float(np.mean(outputs)) if len(outputs) == 2 else None


def response_corrections(blocks, inverse_speed, tau):
    """Integrated instantaneous-minus-delayed HR effort for each microblock.

    Add this difference to observed HR effort, rather than replace observation
    by a fitted curve. Any residual measured HR cost remains in the evidence.
    """
    from run_analysis.hr_response_alignment import response_step
    if not math.isfinite(inverse_speed) or inverse_speed <= 0:
        raise ValueError('Positive fitted inverse reference speed required')
    state = end = None; result = []
    for b in blocks:
        if end is None or abs(b.start-end) > 1e-6:
            state = b.first_hr_effort
        target = b.effort_distance/b.seconds*inverse_speed
        average, state = response_step(state, target, b.seconds, tau)
        result.append(dict(start=b.start, end=b.start+b.seconds,
                           effort_adjustment=target-average))
        end = b.start+b.seconds
    return result


def aligned_windows(rows, corrections, *, origin_epoch, config, calibration):
    """Original window set/pace/context, with only an explicit HR timing term.

    Unknown context retains the original estimate and a reason, not a dropped
    run or an invented measurement. Recompute comparison precision downstream.
    """
    from datetime import datetime
    beta = calibration['parameters'][0]
    reserve = config['target_hr']-config['resting_hr']
    result = []
    for row in rows:
        low = datetime.fromisoformat(row['window_start_time_utc']).timestamp()-origin_epoch
        high = datetime.fromisoformat(row['window_end_time_utc']).timestamp()-origin_epoch
        total = weighted = 0.
        for c in corrections:
            dt = max(0., min(high, c['end'])-max(low, c['start']))
            total += dt; weighted += dt*c['effort_adjustment']
        copy = dict(row)
        timing = dict(observed_hr_bpm=row['average_hr_bpm'], available=False,
            reason='incomplete_recorded_context', covered_seconds=total)
        if total >= high-low-1e-5:
            h = ((row['average_hr_bpm']-config['resting_hr'])/reserve)**beta+weighted/total
            if h > 0:
                copy['average_hr_bpm'] = config['resting_hr']+reserve*h**(1/beta)
                timing.update(available=True, reason='observed_hr_plus_modeled_timing_difference',
                    aligned_hr_bpm=copy['average_hr_bpm'])
            else:
                timing['reason'] = 'nonpositive_modeled_hr_effort'
        copy['hr_timing'] = timing
        result.append(copy)
    return result


def cached_hr_aligned_comparability(connection, report, config):
    """Progress prototype only; canonical scores and planner remain untouched."""
    import json
    from .read_cache import invariant_read
    inputs = {k: config.get(k) for k in ('model', 'moving_time', 'elevation', 'resting_hr', 'max_hr', 'target_hr', 'reference_conditions')}
    return invariant_read(connection, (VERSION, json.dumps(report, sort_keys=True), json.dumps(inputs, sort_keys=True, default=str)),
        lambda: build_hr_aligned_comparability(connection, report, config))


def build_hr_aligned_comparability(connection, report, config, *, window_collector=None, include_precision=True):
    from .model_windows import load_overlapping_model_windows
    from .objective_modeling import _primary_window_weight, _pre_quality_indexes
    from .pace_comparability import compare_run_windows
    from .quality_phases import detect_continuous_quality_phase

    eligible = {r['activity_id'] for r in report['runs'] if r['included_in_trend'] and r['trend_weight'] > 0}
    if not eligible:
        return {**report, 'hr_response': dict(version=VERSION, runs=0, excluded_runs=0)}
    windows, _ = load_overlapping_model_windows(connection, config, window_seconds=120, stride_seconds=60,
        check_elevation=True, response_calibration=report['calibration'])
    groups = defaultdict(list)
    for row in windows:
        if row['activity_id'] in eligible:
            groups[row['activity_id']].append(row)
    evidence = {}; contexts = {}
    for identity, rows in groups.items():
        context = rows[0].get('_aerobic_response')
        if context and context['blocks']:
            contexts[identity] = context
            evidence[identity] = fit(statistics(context['blocks']))
    results = []
    for ordinal, original in enumerate(report['runs']):
        identity = original['activity_id']
        if identity not in eligible or identity not in evidence or evidence[identity] is None:
            results.append({**original, 'hr_response': dict(version=VERSION, available=False, reason='original_evidence_retained')})
            continue
        others = [f['losses'] for k, f in evidence.items() if k != identity and f is not None]
        i = int(np.argmin(np.mean(others, axis=0))) if len(others) >= 3 else 0
        f = evidence[identity]; context = contexts[identity]
        if not f['valid'][i]:
            results.append({**original, 'hr_response': dict(version=VERSION, available=False, reason='nonpositive_fit_original_retained')})
            continue
        rows = [{k: v for k, v in r.items() if k != '_aerobic_response'} for r in groups[identity]]
        if rows and rows[0].get('workout_type') == 'tempo_threshold':
            phase = detect_continuous_quality_phase(connection, config, identity)
            rows = [rows[j] for j in _pre_quality_indexes(rows, list(range(len(rows))), phase)]
        if not rows:
            results.append(original)
            continue
        weights = [_primary_window_weight(r, config) for r in rows]
        corrections = response_corrections(context['blocks'], float(f['inverse_reference_speed'][i]), TAUS[i])
        aligned = aligned_windows(rows, corrections, origin_epoch=context['origin_epoch'], config=config, calibration=report['calibration'])
        if window_collector is not None:
            window_collector(identity, aligned, weights, context)
        value = compare_run_windows(aligned, weights, config, report['calibration'],
            precision_seed=20261015+ordinal if include_precision else None)
        results.append({**original, **value, 'sampled_hr_bpm': original['sampled_hr_bpm'],
            'aligned_sampled_hr_bpm': value['sampled_hr_bpm'],
            'hr_response': dict(version=VERSION, available=True, tau_seconds=TAUS[i],
                original_pace_min_mile=original['pace_min_mile'], aligned_pace_min_mile=value['pace_min_mile'],
                delta_seconds_per_mile=60*(value['pace_min_mile']-original['pace_min_mile']),
                original_windows=len(rows), retained_windows=len(aligned),
                adjusted_windows=sum(r['hr_timing']['available'] for r in aligned), coverage=context['coverage'])})
    return {**report, 'runs': results, 'hr_response': dict(version=VERSION, runs=len(eligible), excluded_runs=0,
        definition='Recorded HR plus an estimated timing difference; observed residuals retained',
        initialization='Actual HR at each continuous recorded bout, never an assumed sleeping-HR reset',
        calibration='Shared response selected without the target run; retrospective existing normalization retained',
        uncertainty='Existing comparison precision recomputed; response-time model uncertainty not a physiological confidence interval')}
