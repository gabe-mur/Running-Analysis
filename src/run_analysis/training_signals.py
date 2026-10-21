"""Connect observed TCX/FIT evidence to the experimental response model.

No database writes, workout reclassification, readiness verdict or planner cost.
Measured demand, contextual observations and speculative response are separate.
The same normalized trace integrator accepts supplied projected HR traces; it
does not generate them or pretend that a prescribed zone is instantaneous HR.
"""
from dataclasses import asdict, dataclass
from math import ceil, exp, isfinite

from .movement import classify_movement
from .recovery_response import ResponseExpectation, assess_response
from .response_model import Policy, rate
from .run_feedback import assess_cardiac_drift
from .session_feedback import feedback_from_session_fields
from .signal_structure import SignalSpan
from .web.schemas import WorkoutType
from .workload_profile import observed_profile


@dataclass(frozen=True)
class HeartRateReference:
    resting_bpm: float
    maximum_bpm: float
    source: str

    def __post_init__(self):
        if (not self.source.strip() or any(isinstance(x, bool) or not isfinite(x)
                for x in (self.resting_bpm, self.maximum_bpm))
                or not 0 < self.resting_bpm < self.maximum_bpm):
            raise ValueError('Ordered positive HR references with provenance required')

    def reserve(self, hr):
        if hr is None or isinstance(hr, bool) or not isfinite(hr):
            return None
        # Do not clip high HR or automatically revise the athlete's reference.
        if not self.resting_bpm <= hr <= self.maximum_bpm:
            return None
        return (hr-self.resting_bpm)/(self.maximum_bpm-self.resting_bpm)


def _mean_rate(a, b, policy):
    """Five-point Gauss integration of g(linear HRR), not g(mean HRR).

    Over the supported [0,1] domain at the reference exponents this agrees with
    fine-grid integration to numerical precision. No time resampling is needed.
    """
    nodes = (0., -.5384693101056831, .5384693101056831,
             -.9061798459386640, .9061798459386640)
    weights = (.5688888888888889, .4786286704993665, .4786286704993665,
               .2369268850561891, .2369268850561891)
    return sum(w*rate((a+b)/2+(b-a)*x/2, policy) for x, w in zip(nodes, weights))/2


def integrate_cardio_trace(spans, elapsed_seconds, policy=Policy(), *, density_step_seconds=1., compute_density=True):
    """Normalized HRR accounting over wall time, including observed stopped HR.

    Missing spans stay unknown. Partial load is reported separately, never used
    as a whole-session dose. Density is an explicitly experimental observation,
    computed only for complete traces (no invented decay through unknown gaps).
    """
    if not isfinite(elapsed_seconds) or elapsed_seconds < 0:
        raise ValueError('Finite nonnegative elapsed duration required')
    if not isfinite(density_step_seconds) or density_step_seconds <= 0:
        raise ValueError('Finite positive integration step required')
    spans = tuple(spans)
    previous = 0.
    for s in spans:
        if s.start < previous-1e-8 or s.end > elapsed_seconds+1e-8:
            raise ValueError('Ordered nonoverlapping spans within the session required')
        if not 0 <= s.first <= 1 or not 0 <= s.last <= 1:
            raise ValueError('HR outside the reference requires review, not clipping')
        previous = s.end
    covered = sum(s.end-s.start for s in spans)
    complete = elapsed_seconds > 0 and abs(covered-elapsed_seconds) < 1e-6
    observed = sum((s.end-s.start)/60*_mean_rate(s.first, s.last, policy) for s in spans)
    mean_reserve = (sum((s.end-s.start)*(s.first+s.last)/2 for s in spans)/covered
                    if covered else None)
    peak = None
    if complete and compute_density:
        density = peak = 0.
        for s in spans:
            # One-second midpoint approximation is for the candidate dynamic
            # term only; the accounting integral above is separate and accurate.
            count = max(1, ceil((s.end-s.start)/density_step_seconds))
            minutes = (s.end-s.start)/count/60
            decay = exp(-minutes/policy.density_memory_minutes)
            for i in range(count):
                reserve = s.first+(s.last-s.first)*(i+.5)/count
                drive = max(0., rate(reserve, policy)-1.)
                density = drive+(density-drive)*decay
                peak = max(peak, density)
    return dict(method='normalized_hrr_trace_v1', unit='minutes_equivalent_at_reference_hrr',
        policy=asdict(policy), elapsed_seconds=elapsed_seconds,
        covered_seconds=covered, unknown_seconds=max(0., elapsed_seconds-covered),
        coverage=covered/elapsed_seconds if elapsed_seconds else None,
        complete=complete, observed_load=observed if covered else None,
        total_load=observed if complete else None,
        load_bounds_within_reference=dict(low=observed,
            high=observed+max(0., elapsed_seconds-covered)/60*rate(1., policy),
            basis='Unknown time spans HRR 0 to 1; conditional accounting bounds, not confidence limits'),
        mean_reserve=mean_reserve,
        mean_hr_shortcut_load=covered/60*rate(mean_reserve, policy) if covered else None,
        density_peak_candidate=peak,
        response_load_candidate=observed*(1+policy.density_weight*peak) if peak is not None else None,
        zero_density_weight_load=observed if complete else None,
        response_calibrated=False)


def observed_cardio_trace(activity, intervals, reference, policy=Policy(), *,
                          include_spans=False, compute_density=True):
    points = [p.timestamp_utc for p in activity.trackpoints if p.timestamp_utc is not None]
    origin = activity.start_time_utc or (min(points) if points else None)
    elapsed = max(float(activity.total_elapsed_time_s or 0),
                  float(activity.lap_recorded_time_s or 0),
                  (max(points)-origin).total_seconds() if points and origin else 0.)
    spans = []
    reasons = dict(recording_gap_seconds=0., missing_hr_seconds=0.,
                   outside_reference_seconds=0.)
    for i in intervals:
        if 'large_recording_gap' in i.flags:
            reasons['recording_gap_seconds'] += i.elapsed_s
            continue
        a, b = i.start.heart_rate_bpm, i.end.heart_rate_bpm
        if any(x is None or isinstance(x, bool) or not isfinite(x) or x <= 0 for x in (a, b)):
            reasons['missing_hr_seconds'] += i.elapsed_s
            continue
        first, last = reference.reserve(a), reference.reserve(b)
        if first is None or last is None:
            reasons['outside_reference_seconds'] += i.elapsed_s
            continue
        spans.append(SignalSpan((i.start.timestamp_utc-origin).total_seconds(),
                               (i.end.timestamp_utc-origin).total_seconds(), first, last))
    result = integrate_cardio_trace(spans, elapsed, policy, compute_density=compute_density)
    if include_spans:
        result['normalized_spans'] = tuple(spans)
    result.update(reference=asdict(reference), exclusions=reasons,
        unrepresented_seconds=max(0., result['unknown_seconds']-sum(reasons.values())),
        stopped_hr_policy='Include valid observed HR during stops; never bridge recording gaps')
    if reasons['outside_reference_seconds']:
        result['load_bounds_within_reference'] = None
    return result


def observed_training_signals(activity, config, *, workout_type=None, profile=None,
                              feedback=None, interval_analysis=None,
                              expected_response=None, policy=Policy()):
    """Build a model-facing report using app-resolved roles and existing analyses.

    feedback, when supplied, is the source-resolved result (manual effort wins).
    interval_analysis is the existing workout analysis, not a new rep detector.
    Expectations must have been issued beforehand; never infer them from outcome.
    """
    movement = classify_movement(activity.trackpoints, config['moving_time'])
    intervals = movement.intervals
    reference = HeartRateReference(config['resting_hr'], config['max_hr'],
                                   'configured_athlete_hr_references')
    cardio = observed_cardio_trace(activity, intervals, reference, policy)
    profile = profile or observed_profile(activity, config, workout_type=workout_type)
    if profile['intensity_structure']['session_workout_type'] != workout_type:
        raise ValueError('Profile must use the same app-resolved workout type')
    try:
        role = WorkoutType(workout_type or 'unknown')
    except ValueError:
        role = WorkoutType.UNKNOWN
    valid_movement = all(i.distance_source != 'unknown'
        and not {'large_recording_gap', 'implausible_distance_speed', 'distance_reset'} & set(i.flags)
        for i in intervals)
    if role not in {WorkoutType.EASY, WorkoutType.LONG}:
        drift = dict(valid=False, reason='No resolved steady easy/long context',
                     decoupling_percent=None, confidence='unavailable')
    elif not cardio['complete'] or not valid_movement:
        drift = dict(valid=False, reason='Incomplete HR or unreliable movement; no drift inference',
                     decoupling_percent=None, confidence='unavailable')
    else:
        drift = assess_cardiac_drift(intervals, role).model_dump(mode='json')
    stream = activity.sensor_stream
    resolved = (dict(feedback) if feedback is not None else
                feedback_from_session_fields(stream.session_fields if stream else {}))
    expected = expected_response or ResponseExpectation('No issued response expectation available')
    response = assess_response(expected,
        effort=resolved.get('perceived_exertion'), feeling=resolved.get('workout_feel'),
        feedback_conflict=resolved.get('feedback_conflict', False))
    # JSON-ready without losing the declared expected ranges/provenance.
    for channel in response['channels'].values():
        if channel['expected'] is not None:
            channel['expected'] = asdict(channel['expected'])
    rep_response = None
    if interval_analysis is not None and role in {WorkoutType.INTERVALS, WorkoutType.RUN_WALK}:
        rep_response = interval_analysis.model_dump(mode='json')
    rep_quality = dict(available=rep_response is not None and rep_response['available'],
        complete_trace=cardio['complete'], reliable_movement=valid_movement,
        interpretation='Preserved existing lap/pace analysis; HR drop depends on recovery duration/pace. '
                       'Not a standardized post-exercise HR-recovery test or a readiness score.')
    interpretation = [
        'Cardiovascular load is an HR-derived demand estimate, not fitness or measured fatigue.',
        'HR distribution and density describe the same cardiovascular evidence; do not sum them.',
        'Speed curves and work bouts preserve external work even when HR lags short efforts.',
        'Drift and between-rep HR drop are contextual observations, not recovery clearance.',
        'Exertion and feeling remain distinct; neither is a leg-soreness measurement.',
    ]
    if cardio['unknown_seconds'] > 1e-6:
        interpretation.append('Partial trace: observed load is not a complete session total.')
    if cardio['exclusions']['outside_reference_seconds']:
        interpretation.append('Recorded HR exceeds the reference domain; review references/signal.')
    if expected_response is None:
        interpretation.append('Reports are retained, but easier/harder than expected is unknown without an issued expectation.')
    return dict(version='connected_training_signals_v1', source='observed',
        activity_start=activity.start_time_utc.isoformat() if activity.start_time_utc else None,
        workout_type=workout_type, cardiovascular=cardio,
        running_exposure=profile['running_exposure'],
        external_and_structural_evidence=profile['intensity_structure'],
        response=dict(cardiac_drift=drift, interval_response=rep_response,
            interval_response_context=rep_quality,
            feedback=resolved, expected_comparison=response, leg_fatigue=None),
        hrv=dict(rr_interval_count=sum(x is not None for batch in stream.rr_batches
                                      for x in batch.intervals_seconds) if stream else 0,
            status='preserved_not_interpreted' if stream and stream.rr_batches else 'unavailable',
            model_influence=False),
        fast_exposure_score=None,
        fast_exposure_status='Not scored: requires supported easy-speed reference and noise/terrain handling',
        interpretation=interpretation, planner_influence=False)


def training_signals_from_run_feedback(activity, config, app_feedback, *,
                                       expected_response=None, policy=Policy()):
    """Bridge the existing source-resolved run analysis, without another classifier.

    Current activity's pace-at-HR and weather are context, not pre-run references
    or a second independent load channel. This does not refit the fitness model.
    """
    if activity.start_time_utc is None or app_feedback.run.start_time != activity.start_time_utc:
        raise ValueError('Run feedback and original activity must have the same start')
    analysis = app_feedback.workout_analysis
    result = observed_training_signals(activity, config,
        workout_type=app_feedback.run.workout_type.value,
        feedback=app_feedback.metadata.model_dump(mode='json'),
        interval_analysis=analysis.interval_analysis if analysis else None,
        expected_response=expected_response, policy=policy)
    observation = app_feedback.run.fitness_observation
    result['performance_context'] = dict(
        pace_hr_observation=observation.model_dump(mode='json') if observation else None,
        weather=app_feedback.weather.model_dump(mode='json') if app_feedback.weather else None,
        interpretation='Existing current-run evidence, not a prior capacity forecast. '
                       'Pace-at-HR and derived VO2max are related evidence, not independent votes.',
        additional_load_charge=False)
    return result
