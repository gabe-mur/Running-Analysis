"""Chronological, session-gated numeric comparison trial; no planner consumer.

Historical envelopes are descriptive, not calibrated prediction intervals.
An out-of-envelope response is not proof of fatigue or authority to move a run.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite

from .workload_accounting import Quantity, observed_workload
from .workload_profile import prescribed_profile


@dataclass(frozen=True)
class Intent:
    distance_meters: Quantity | None
    duration_minutes: Quantity | None
    work_seconds: Quantity | None
    longest_work_seconds: Quantity | None
    easy_hr_ceiling: float | None = None
    work_distance_meters: Quantity | None = None
    longest_work_distance_meters: Quantity | None = None


@dataclass(frozen=True)
class Reference:
    key: str
    available_at: datetime
    profile: dict


@dataclass(frozen=True)
class ForecastPolicy:
    distance_ratio_limit: float
    work_ratio_limit: float
    minimum_sessions: int
    minimum_hr_coverage: float
    hr_allowance_bpm: float
    speed_allowance_mps: float

    def __post_init__(self):
        numbers = (self.distance_ratio_limit, self.work_ratio_limit,
                   self.minimum_hr_coverage, self.hr_allowance_bpm, self.speed_allowance_mps)
        if (not all(isfinite(x) for x in numbers) or self.distance_ratio_limit < 1
                or self.work_ratio_limit < 1 or type(self.minimum_sessions) is not int
                or self.minimum_sessions < 1
                or not 0 < self.minimum_hr_coverage <= 1
                or min(self.hr_allowance_bpm, self.speed_allowance_mps) < 0):
            raise ValueError('Invalid explicit forecast policy')


def _utc(at):
    if at.tzinfo is None or at.utcoffset() is None:
        raise ValueError('Aware forecast/reference timestamps required')
    return at.astimezone(timezone.utc)


def intent_from_prescription(recommendation, zones):
    profile = prescribed_profile(recommendation)
    exposure, shape = profile['running_exposure'], profile['intensity_structure']
    distance = exposure['distance_range_miles']
    target = exposure['distance_target_miles']
    distance = (Quantity(distance[0]*1609.344, distance[1]*1609.344,
                         target*1609.344 if target is not None else None, 'issued_band') if distance else None)
    duration = recommendation.duration_range_minutes
    duration = Quantity(*duration, basis='issued_band') if duration else None
    low, high, longest_low, longest_high = 0., 0., 0., 0.
    for bout in shape['bouts']:
        a, b = bout.get('duration_range_seconds') or (bout['work_seconds'], bout['work_seconds'])
        low += a
        high += b
        longest_low, longest_high = max(longest_low, a), max(longest_high, b)
    unresolved = 'work_duration_missing' in profile['limitations']
    # Only explicit zone ceilings are read. 'Z4 effort' is NOT measured Z4 HR.
    labels = {x.casefold().strip() for x in recommendation.target_zones}
    easy_ceiling = (float(zones[max(labels)][1])+1
                    if labels and labels <= {'z1', 'z2'} and not shape['bouts'] else None)
    return Intent(distance, duration,
        Quantity(low, high, basis='issued_work_time') if shape['bouts'] and not unresolved else None,
        Quantity(longest_low, longest_high, basis='issued_work_time') if shape['bouts'] and not unresolved else None,
        easy_ceiling)


def observed_features(profile, *, include_inferred_work=False):
    workload = observed_workload(profile)
    features = dict(distance_meters=workload.distance_meters, duration_minutes=workload.running_minutes,
                    cardio_points=workload.cardio_points)
    shape = profile['intensity_structure']
    recorded_signals = shape['recorded_signals']
    basis = 'recorded_lap_contrast'
    inferred = shape.get('pace_inferred_work')
    role = shape.get('session_workout_type')
    if (role == 'intervals' and not shape['bout_count'] and include_inferred_work
            and inferred and inferred['usable_for_comparison']):
        shape = inferred
        basis = 'pace_stream_inference_unquantified'
    # Absence of detected laps is unknown work, not proof of zero quality.
    if shape['bout_count'] and role in {None, 'intervals'}:
        features['work_seconds'] = Quantity.point(shape['total_work_seconds'], basis)
        features['longest_work_seconds'] = Quantity.point(shape['longest_work_seconds'], basis)
        distances = [b['distance_meters'] for b in shape['bouts']]
        features['work_distance_meters'] = (Quantity.point(sum(distances), basis)
                                             if all(x is not None for x in distances) else None)
        features['longest_work_distance_meters'] = (Quantity.point(max(distances), basis)
                                                     if all(x is not None for x in distances) else None)
    else:
        features.update(work_seconds=None, longest_work_seconds=None,
                        work_distance_meters=None, longest_work_distance_meters=None)
    signals = recorded_signals
    for prefix, key, value in (('hr', 'hr_duration_curve', 'peak_mean_bpm'),
                                ('speed', 'speed_duration_curve', 'peak_mean_mps')):
        for point in signals[key]:
            features[f'{prefix}_{point["window_seconds"]:g}s'] = (
                Quantity.point(point[value], 'recorded_signal') if point[value] is not None else None)
    return features


def _near(value, expected, ratio):
    return expected.low/ratio <= value <= expected.high*ratio


def _work_matches(intent, features, ratio):
    keys = ('work_seconds', 'longest_work_seconds',
            'work_distance_meters', 'longest_work_distance_meters')
    constraints = [(getattr(intent, key), features[key]) for key in keys
                   if getattr(intent, key) is not None]
    return bool(constraints) and all(done is not None and
        _near(done.low, expected, ratio) and _near(done.high, expected, ratio)
        for expected, done in constraints)


def forecast(intent, references, issued_at, policy, *, include_inferred_work=False):
    """Select numeric references within app-resolved session eligibility.

    Never inspect the target response, parse workout titles or use later runs.
    Work-time comparisons currently require resolved historical numeric work
    evidence; no unsupported HR/pace curve is manufactured for an unfamiliar
    structure. Signal extraction itself requires neither laps nor repetitions.
    """
    issued_at = _utc(issued_at)
    references = tuple(references)
    if len({r.key for r in references}) != len(references):
        raise ValueError('Duplicate reference sessions')
    selected, work_selected, rejected = [], [], []
    for ref in references:
        reason = None
        if _utc(ref.available_at) > issued_at:
            reason = 'not_yet_available'
        else:
            features = observed_features(ref.profile, include_inferred_work=include_inferred_work)
            work_matches = _work_matches(intent, features, policy.work_ratio_limit)
            if work_matches:
                # Numeric work-pace evidence does not require identical warmup,
                # cooldown or whole-session mileage (nor HR for a pace ratio).
                # This pool MUST NOT forecast whole-session HR/cardio curves.
                work_selected.append((ref, features))
            distance = features['distance_meters']
            if (intent.distance_meters is not None and
                    (distance is None or not _near(distance.low, intent.distance_meters, policy.distance_ratio_limit))):
                reason = 'distance_not_comparable'
            elif ref.profile['cardiovascular']['hr_coverage'] < policy.minimum_hr_coverage:
                reason = 'hr_coverage'
            elif intent.easy_hr_ceiling is not None:
                # Match the intended low-HR envelope, not a stored 'easy' label.
                peak = features.get('hr_300s')
                if peak is None or peak.high > intent.easy_hr_ceiling+policy.hr_allowance_bpm:
                    reason = 'sustained_hr_not_comparable'
            elif intent.work_seconds is not None or intent.work_distance_meters is not None:
                if not work_matches:
                    reason = 'numeric_work_not_comparable'
            else:
                reason = 'intensity_intent_unresolved'
        if reason:
            rejected.append(dict(key=ref.key, reason=reason))
        else:
            selected.append((ref, features))
    expected = {key: getattr(intent, key) for key in
                ('distance_meters', 'duration_minutes', 'work_seconds', 'longest_work_seconds',
                 'work_distance_meters', 'longest_work_distance_meters')}
    support = {}
    inferred_support = {}
    keys = sorted({key for _, f in selected for key in f} |
                  {'work_seconds', 'longest_work_seconds'})
    for key in keys:
        if key in expected and expected[key] is not None:
            continue
        values = []
        contributors = []
        inferred_contributors = []
        pool = work_selected if key in {'work_seconds', 'longest_work_seconds'} else selected
        for ref, features in pool:
            value = features.get(key)
            if value is None:
                continue
            if value.basis == 'pace_stream_inference_unquantified' or (
                    intent.easy_hr_ceiling is None and any(v is not None and
                        v.basis == 'pace_stream_inference_unquantified'
                        for k, v in features.items() if 'work' in k)):
                inferred_contributors.append(ref.key)
            if key in {'work_seconds', 'longest_work_seconds'}:
                distance_key = ('work_distance_meters' if key == 'work_seconds' else 'longest_work_distance_meters')
                distance, target = features[distance_key], getattr(intent, distance_key)
                if target is None or distance is None or distance.low <= 0:
                    continue
                value = Quantity(value.low/distance.high*target.low,
                                 value.high/distance.low*target.high, basis='prior_work_pace_times_intended_distance')
            elif key in {'work_distance_meters', 'longest_work_distance_meters'}:
                continue
            elif key in {'duration_minutes', 'cardio_points'}:
                # Per-distance rates preserve a distance range instead of silently
                # conditioning the forecast on how far the target actually ran.
                distance = features['distance_meters']
                if intent.distance_meters is None or distance is None or distance.low <= 0:
                    continue
                value = Quantity(value.low/distance.high*intent.distance_meters.low,
                                 value.high/distance.low*intent.distance_meters.high,
                                 basis='historical_rate_times_issued_distance_band')
            values.append(value)
            contributors.append(ref.key)
        support[key] = contributors
        inferred_support[key] = inferred_contributors
        if len(values) < policy.minimum_sessions:
            expected.setdefault(key, None)
            continue
        allowance = (policy.hr_allowance_bpm if key.startswith('hr_') else
                     policy.speed_allowance_mps if key.startswith('speed_') else 0.)
        expected[key] = Quantity(max(0., min(x.low for x in values)-allowance),
                                 max(x.high for x in values)+allowance,
                                 basis='historical_envelope_inferred_work_unquantified' if inferred_contributors
                                 else 'historical_envelope_not_calibrated')
    return dict(issued_at=issued_at, expected=expected, support=support,
        inferred_support=inferred_support, include_inferred_work=include_inferred_work,
        selected_reference_keys=[r.key for r, _ in selected], rejected=rejected,
        work_reference_keys=[r.key for r, _ in work_selected],
        meets_session_count_setting=len(selected) >= policy.minimum_sessions,
        meets_work_count_setting=len(work_selected) >= policy.minimum_sessions,
        calibrated=False,
        limitations=['Historical min/max envelopes are not calibrated prediction intervals',
                     'Current settings frozen; completion time used as availability, not original upload time',
                     'Weather, grade, sensor changes and fatigue are not separated',
                     'Work-only matches inform work pace, not whole-session response or equivalent fatigue',
                     'Optional pace inference retains extra candidates and unquantified boundary error',
                     'Novel numeric structure can remain unsupported; missing evidence does not order rest'],
        planner_influence=False)


def compare_distribution(expectation, actual_profile, tolerances):
    """Tolerance-aware differences, not a fatigue diagnosis or adaptation trigger."""
    actual = observed_features(actual_profile,
        include_inferred_work=expectation.get('include_inferred_work', False))
    signals = actual_profile['intensity_structure']['recorded_signals']
    exposure = actual_profile['running_exposure']
    moving = exposure.get('moving_seconds_range', (None, exposure['moving_seconds']))[1]
    rows = {}
    for key in sorted(set(actual) | set(expectation['expected'])):
        expected, done = expectation['expected'].get(key), actual.get(key)
        tolerance = tolerances.get(key, 0.)
        if not isfinite(tolerance) or tolerance < 0:
            raise ValueError('Comparison tolerances must be finite and nonnegative')
        partial = False
        if key.startswith(('hr_', 'speed_')):
            channel = 'hr_seconds' if key.startswith('hr_') else 'speed_seconds'
            partial = moving is None or signals['coverage'][channel]+1e-6 < moving
        if expected is None or done is None:
            status = 'unknown'
        elif done.low > expected.high+tolerance+1e-8:
            status = 'above_expectation'
        elif partial:
            # A valid observed peak can establish a lower bound; missing samples
            # prevent claiming there was no higher unobserved peak.
            status = 'partial_observation'
        elif done.high < expected.low-tolerance-1e-8:
            status = 'below_expectation'
        elif done.low >= expected.low-tolerance-1e-8 and done.high <= expected.high+tolerance+1e-8:
            status = 'within_expectation'
        else:
            status = 'overlapping_uncertainty'
        inferred = any(q is not None and 'unquantified' in q.basis for q in (expected, done))
        nominal_status = status
        if inferred and status != 'unknown':
            status = 'inferred_estimate_only'
        rows[key] = dict(status=status, nominal_status=nominal_status,
                         expected=expected, observed=done, tolerance=tolerance,
                         partial_observation=partial, unquantified_inference=inferred)
    differences = [key for key, value in rows.items() if value['status'] in {'above_expectation', 'below_expectation'}]
    return dict(features=rows, differences=differences,
        interpretation='Differences require context; do not equate elevated HR with harder execution or fatigue',
        adaptation_authorized=False, planner_influence=False)
