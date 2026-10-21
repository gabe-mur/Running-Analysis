"""Shadow-only workload measurements; no readiness score or DFA dependency.

Duration-weighted bout length is a descriptive measure of sustained work, not
a fatigue multiplier. Planned intensity is not evidence of performed intensity.
"""
from dataclasses import asdict, dataclass
from math import isfinite

from .models import Activity
from .movement import classify_movement
from .training_load import calculate_session_load
from .workload_accounting import continuous_zone_seconds
from .signal_structure import describe_signals
from .prescription_load import structured_duration_bounds
from .workout_detection import inferred_work_groups, recorded_interval_work_positions, usable_recorded_laps


@dataclass(frozen=True)
class WorkBout:
    work_seconds: float
    recovery_after_seconds: float | None
    distance_meters: float | None = None
    target_zones: tuple[str, ...] = ()
    duration_range_seconds: tuple[float, float] | None = None


def bout_summary(bouts):
    for bout in bouts:
        if not isfinite(bout.work_seconds) or bout.work_seconds <= 0:
            raise ValueError('Work duration must be finite and positive')
        if bout.recovery_after_seconds is not None and (
                not isfinite(bout.recovery_after_seconds) or bout.recovery_after_seconds < 0):
            raise ValueError('Recovery duration must be finite and nonnegative')
    total = sum(b.work_seconds for b in bouts)
    between = [b.recovery_after_seconds for b in bouts[:-1]]
    recovery = sum(between) if all(x is not None for x in between) else None
    return dict(bout_count=len(bouts), total_work_seconds=total,
        longest_work_seconds=max((b.work_seconds for b in bouts), default=0.),
        duration_weighted_work_seconds=sum(b.work_seconds**2 for b in bouts)/total if total else 0.,
        between_work_recovery_seconds=recovery,
        work_fraction_of_work_and_between_recovery=(total/(total+recovery)
            if total and recovery is not None else None), bouts=[asdict(b) for b in bouts])


def inferred_bout_summary(intervals):
    """Reuse run-detail pace segmentation, without declaring confirmed quality.

    No new thresholds or rep filtering: retain extra candidates for inspection.
    A candidate crossing unreliable movement cannot support numeric comparison.
    """
    groups = inferred_work_groups(intervals)
    bouts, windows, invalid = [], [], []
    bad_flags = {'large_recording_gap', 'implausible_distance_speed', 'distance_reset'}
    for n, (start, end) in enumerate(groups):
        work = intervals[start:end]
        if any(bad_flags & set(i.flags) or abs(i.moving_time_s-i.elapsed_s) > 1e-8
               or i.distance_source == 'unknown' for i in work):
            invalid.append(n)
        recovery = intervals[end:groups[n+1][0]] if n+1 < len(groups) else None
        after = (sum(i.elapsed_s for i in recovery) if recovery is not None
                 and not any('large_recording_gap' in i.flags for i in recovery) else None)
        bouts.append(WorkBout(sum(i.elapsed_s for i in work), after,
                              sum(i.distance_m for i in work)))
        windows.append(dict(start=work[0].start.timestamp_utc.isoformat(),
                            end=work[-1].end.timestamp_utc.isoformat()))
    return dict(evidence='pace_stream_inference', confidence='moderate' if len(groups) >= 2 else 'low',
        boundary_uncertainty='unquantified', confirmed_quality=False,
        usable_for_comparison=len(groups) >= 2 and not invalid,
        invalid_candidate_indexes=invalid, windows=windows, **bout_summary(bouts))


def observed_profile(activity: Activity, config: dict, *, workout_type: str | None = None) -> dict:
    """Measure all runs; reconstruct pace reps only in the app's eligible roles.

    The caller supplies the app-resolved session type, not the planned workout.
    Without that context, raw pace fluctuation cannot enable interval inference.
    """
    movement = classify_movement(activity.trackpoints, config['moving_time'])
    intervals = movement.intervals
    moving_seconds = sum(i.moving_time_s for i in intervals)
    uncertain_movement = [i for i in intervals
                          if 'large_recording_gap' in i.flags and i.classification != 'stopped']
    covered_elapsed = sum(i.elapsed_s for i in intervals)
    unobserved_seconds = max(0., (activity.lap_recorded_time_s or 0)-covered_elapsed)
    moving_low = max(0., moving_seconds-sum(i.moving_time_s for i in uncertain_movement))
    moving_high = max(moving_seconds,
                     moving_low+sum(i.elapsed_s for i in uncertain_movement)+unobserved_seconds)
    zones = continuous_zone_seconds(intervals, config['zones'])
    cardio = calculate_session_load(zones, moving_seconds)
    rows = [asdict(lap) for lap in activity.laps]
    usable = usable_recorded_laps(rows)
    positions = sorted(recorded_interval_work_positions(rows,intervals))
    # A single faster middle lap is not automatically an interval session.
    if len(positions) < 2:
        positions = []
    bouts = []
    for n, pos in enumerate(positions):
        lap = usable[pos]
        after = (sum(float(x['total_time_s']) for x in usable[pos+1:positions[n+1]])
                 if n+1 < len(positions) else None)
        bouts.append(WorkBout(float(lap['total_time_s']), after, float(lap['distance_m'])))
    return dict(version='workload_profile_v3_shadow', source='observed',
        running_exposure=dict(distance_meters=activity.total_distance_m,
            moving_seconds=moving_seconds, moving_seconds_range=(moving_low, moving_high),
            movement_observed=bool(intervals), elapsed_seconds=activity.total_elapsed_time_s),
        cardiovascular=dict(method='edwards_linear_hr_contiguous_bins', **asdict(cardio), zone_seconds=zones),
        intensity_structure=dict(evidence='recorded_lap_pace_contrast' if bouts else 'unresolved',
            session_workout_type=workout_type,
            **bout_summary(bouts),
            pace_inferred_work=(inferred_bout_summary(intervals) if not bouts
                and workout_type in {'intervals', 'run_walk'} else None),
            recorded_signals=describe_signals(intervals,
                hr_thresholds_bpm=[config['zones'][name][0] for name in ('z3', 'z4', 'z5')])),
        limitations=['No recovery or injury prediction', 'Lap contrasts are descriptive, not absolute intensity',
            'Missing bouts do not prove absence of quality work',
            'Signal distribution is descriptive; physiological intensity, terrain and novelty remain unresolved'],
        planner_influence=False)


def prescribed_profile(recommendation, *, expected_duration_minutes: float | None = None) -> dict:
    """Retain exact repetitions, target zones and explicit uncertainty.

    No estimated HR trace, fixed pace or invented work duration is used here.
    Total-duration/cardio projections are deliberately unresolved until supplied.
    """
    bouts, unresolved = [], []
    minimum_seconds_low, minimum_seconds_high = (x*60 for x in structured_duration_bounds(recommendation))
    for step in recommendation.structure:
        if step.phase != 'work':
            continue
        count = step.repetitions or 1
        seconds_range = None
        if step.work_duration_minutes is not None:
            seconds = step.work_duration_minutes*60
        elif step.work_duration_range_minutes is not None:
            seconds_range = tuple(x*60 for x in step.work_duration_range_minutes)
            seconds = sum(seconds_range)/2
        elif step.duration_minutes is not None:
            seconds = step.duration_minutes*60/count
        else:
            unresolved.append('work_duration_missing')
            continue
        for index in range(count):
            recovery = (step.recovery_duration_minutes*60
                        if index < count-1 and step.recovery_duration_minutes is not None else None)
            bouts.append(WorkBout(seconds, recovery, target_zones=tuple(step.target_zones),
                                  duration_range_seconds=seconds_range))
    if expected_duration_minutes is not None and (
            not isfinite(expected_duration_minutes) or expected_duration_minutes <= 0):
        raise ValueError('Expected duration must be finite and positive')
    return dict(version='workload_profile_v2_shadow', source='prescribed',
        running_exposure=dict(distance_target_miles=recommendation.planned_distance_miles,
            distance_range_miles=recommendation.distance_range_miles,
            expected_duration_minutes=expected_duration_minutes,
            minimum_duration_range_seconds=(minimum_seconds_low, minimum_seconds_high)),
        cardiovascular=dict(method='not_projected', zone_load=None),
        intensity_structure=dict(evidence='prescription_only', **bout_summary(bouts)),
        limitations=unresolved+['Expected cardio range and recovery model remain pending'],
        planner_influence=False)
