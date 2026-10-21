"""Frozen, chronological performance evidence, separate from cardiovascular load.

Raw pace/HR comparisons are evidence, not proof of a fitness change. Session
dispersion is not a prediction interval. No live planner or fitted-cache access.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import exp, isclose, isfinite, log
from statistics import median

from .workload_accounting import Quantity


def geometric_mean(values, weights=None):
    values = tuple(values)
    weights = tuple(weights) if weights is not None else (1.,)*len(values)
    estimate = exp(sum(w*log(v) for v, w in zip(values, weights))/sum(weights))
    # exp(log(v)) can sit one ULP outside [v, v]; retain Quantity's strict bounds.
    return min(max(values), max(min(values), estimate))


@dataclass(frozen=True)
class AerobicReference:
    issued_at: datetime
    accounting_key: str
    # Per-session (HR reserve, speed) knots. A long run is not many independent runs.
    sessions: tuple[tuple[str, tuple[tuple[float, float], ...]], ...]
    lookback_days: float
    minimum_reserve: float


def freeze_aerobic_reference(state, *, lookback_days=84., minimum_reserve=.25):
    """Save pre-run observations; completion must never refit its own baseline.

    The HRR floor avoids unstable ratios near resting HR. It is a numerical
    domain assumption, not a fitness or workout-acceptance requirement.
    """
    if (not isfinite(lookback_days) or lookback_days <= 0
            or not isfinite(minimum_reserve) or not 0 < minimum_reserve < 1):
        raise ValueError('Valid reference window and reserve floor required')
    cutoff = state.as_of-timedelta(days=lookback_days)
    rows = []
    for s in state.sessions:
        if not s.capacity_eligible or s.end <= cutoff:
            continue
        knots = {}
        for p in s.pace_samples:
            if p.phase == 'easy' and p.mean_hr_reserve is not None and minimum_reserve <= p.mean_hr_reserve < 1:
                knots.setdefault(p.mean_hr_reserve, []).append(p.meters/p.seconds)
        if knots:
            rows.append((s.key, tuple((r, median(v)) for r, v in sorted(knots.items()))))
    return AerobicReference(state.as_of, state.accounting_key, tuple(rows), lookback_days, minimum_reserve)


def _speed_at(knots, reserve):
    for r, speed in knots:
        if isclose(r, reserve, abs_tol=1e-12):
            return speed, False
    for (r0, v0), (r1, v1) in zip(knots, knots[1:]):
        if r0 < reserve < r1:
            return v0+(v1-v0)*(reserve-r0)/(r1-r0), False
    # Explicit provisional transfer, not a measured VO2max or a sprint model.
    # Use the nearest observed anchor instead of extrapolating a noisy slope.
    r, speed = min(knots, key=lambda p: abs(p[0]-reserve))
    return speed*reserve/r, True


def aerobic_speed(reference, reserve):
    if not isfinite(reserve) or not reference.minimum_reserve <= reserve < 1:
        raise ValueError('Submaximal reserve inside the declared domain required')
    values = [_speed_at(knots, reserve) for _, knots in reference.sessions]
    speeds = [v for v, _ in values]
    return dict(speed_mps=(Quantity(min(speeds), max(speeds), geometric_mean(speeds),
        'session_dispersion_not_prediction_interval') if speeds else None),
        source_keys=tuple(key for key, _ in reference.sessions), source_count=len(values),
        transferred_source_count=sum(transferred for _, transferred in values),
        status='provisional_aerobic_reference' if speeds else 'no_aerobic_reference',
        context_adjusted=False, duration_adjusted=False, prediction_error_calibrated=False,
        transfer_assumption='Outside observed HR domain: speed proportional to HR reserve; not validated interval/sprint physiology')


def phase_pace_reference(state, phase, *, equilibrium_reserve, seconds=None, meters=None,
                         supplied_pace=None, lookback_days=84.):
    """Generalize phase pace without an exact-workout or sample-count gate.

    Work-dose similarity changes a continuous weight, never eligibility. If no
    phase evidence exists, a declared submaximal aerobic transfer supplies a
    provisional pace. Truly absent pace/HR history still requires an explicit
    pace for distance prescriptions; timed HR-load forecasts need neither.
    """
    if phase not in {'easy', 'work', 'recovery'}:
        raise ValueError('Known phase required')
    if not isfinite(lookback_days) or lookback_days <= 0:
        raise ValueError('Positive reference window required')
    for target in (seconds, meters):
        if target is not None and (not isfinite(target) or target <= 0):
            raise ValueError('Positive requested dose required')
    if equilibrium_reserve.high > 1:
        raise ValueError('HR reserve assumptions must lie in [0, 1]')
    if supplied_pace is not None:
        if supplied_pace.low <= 0 or supplied_pace.basis == 'unspecified':
            raise ValueError('Positive supplied pace with provenance required')
        return dict(speed_mps=supplied_pace, status='supplied_pace', source_count=0,
                    source_keys=(), context_adjusted=False, prediction_error_calibrated=False)
    cutoff = state.as_of-timedelta(days=lookback_days)
    speeds, keys, relevance, durations, distances, sample_speeds = [], [], [], [], [], []
    for s in state.sessions:
        if not s.capacity_eligible or s.end <= cutoff:
            continue
        samples = [p for p in s.pace_samples if p.phase == phase]
        if phase == 'easy':
            samples = [p for p in samples if p.mean_hr_reserve is not None and
                       equilibrium_reserve.low <= p.mean_hr_reserve <= equilibrium_reserve.high]
        if not samples:
            continue
        weights = [1/(1+sum(log(actual/target)**2 for actual, target in
            ((p.seconds, seconds), (p.meters, meters)) if target is not None)) for p in samples]
        values = [p.meters/p.seconds for p in samples]
        sample_speeds.extend(values)
        speeds.append(geometric_mean(values, weights))
        relevance.append(sum(weights)/len(weights))
        keys.append(s.key)
        durations.extend(p.seconds for p in samples)
        distances.extend(p.meters for p in samples)
    if speeds:
        outside = any(target is not None and not min(values) <= target <= max(values)
                      for target, values in ((seconds, durations), (meters, distances)))
        return dict(speed_mps=Quantity(min(speeds), max(speeds), geometric_mean(speeds, relevance),
                        'session_phase_pace_dispersion_not_prediction_interval'),
            source_keys=tuple(keys), source_count=len(keys),
            source_relevance=tuple(relevance),
            status='phase_transfer' if outside else 'phase_reference',
            outside_observed_work_dose=outside, observed_duration_seconds=(min(durations), max(durations)),
            observed_distance_meters=(min(distances), max(distances)),
            observed_sample_speed_range=(min(sample_speeds), max(sample_speeds)),
            context_adjusted=False, prediction_error_calibrated=False,
            limitation='Numeric similarity weighting does not establish sustainable pace for longer efforts')
    ref = freeze_aerobic_reference(state, lookback_days=lookback_days)
    if equilibrium_reserve.low < ref.minimum_reserve or equilibrium_reserve.high >= 1:
        return dict(speed_mps=None, status='supplied_pace_needed_outside_transfer_domain',
            source_count=0, source_keys=(), context_adjusted=False,
            prediction_error_calibrated=False,
            limitation='No aerobic extrapolation toward resting/maximal HR; timed HR-load forecast remains available')
    center = equilibrium_reserve.estimate
    if center is None:
        center = (equilibrium_reserve.low+equilibrium_reserve.high)/2
    result = aerobic_speed(ref, center)
    if result['speed_mps'] is not None:
        # Preserve intensity uncertainty as well as between-session dispersion.
        low = aerobic_speed(ref, equilibrium_reserve.low)['speed_mps']
        high = aerobic_speed(ref, equilibrium_reserve.high)['speed_mps']
        central = result['speed_mps']
        result['speed_mps'] = Quantity(min(low.low, central.low, high.low),
            max(low.high, central.high, high.high), central.estimate,
            'aerobic_transfer_sensitivity_not_prediction_interval')
    return result


def compare_aerobic_completion(reference, actual):
    """Signed performance residual against the frozen pre-run expectation.

    Faster at the same HR and unchanged pace at lower HR can both contribute
    positive evidence. Lower load alone never does. Each completion contributes
    one session, not independent votes for every window. New snapshots may use
    that session; this reference remains unchanged.
    """
    if actual.accounting_key != reference.accounting_key:
        raise ValueError('Mixed performance accounting')
    if actual.start < reference.issued_at or actual.key in {k for k, _ in reference.sessions}:
        raise ValueError('Performance completion must be held out')
    base = dict(reference_issued_at=reference.issued_at,
        history_keys=tuple(k for k, _ in reference.sessions),
        context_adjusted=False, duration_adjusted=False,
        fitness_gain_established=False, adaptation_authorized=False,
        uncertainty='Unmeasured conditions remain; residual is not a calibrated significance test')
    if not actual.capacity_eligible:
        return dict(base, status='excluded_capacity_evidence', speed_change_fraction=None, windows=())
    windows = []
    for p in actual.pace_samples:
        if p.phase != 'easy' or p.mean_hr_reserve is None or not reference.minimum_reserve <= p.mean_hr_reserve < 1:
            continue
        expected = aerobic_speed(reference, p.mean_hr_reserve)
        q = expected['speed_mps']
        if q is None:
            continue
        speed = p.meters/p.seconds
        windows.append(dict(observed_speed_mps=speed, mean_hr_reserve=p.mean_hr_reserve,
            expected_speed_mps=q.estimate, speed_change_fraction=speed/q.estimate-1,
            expected_session_dispersion=(q.low, q.high),
            transferred_source_count=expected['transferred_source_count']))
    change = median(w['speed_change_fraction'] for w in windows) if windows else None
    return dict(base, status=('no_comparable_evidence' if change is None else
        'as_expected' if isclose(change, 0., abs_tol=1e-12) else
        'positive_aerobic_evidence' if change > 0 else 'negative_aerobic_evidence'),
        speed_change_fraction=change, windows=tuple(windows),
        interpretation='Raw pace at observed HR versus prior reference; not a proven fitness gain or loss')
