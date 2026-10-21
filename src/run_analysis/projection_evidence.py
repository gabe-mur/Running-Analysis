"""Original-run adapter for chronological state; never reads fitted model caches."""
from dataclasses import asdict
from datetime import timedelta

from .athlete_projection import PaceSample, SessionEvidence, accounting_signature
from .movement import classify_movement
from .response_model import Policy
from .training_signals import HeartRateReference, observed_cardio_trace
from .workload_accounting import Quantity
from .workload_profile import observed_profile
from .workout_detection import usable_recorded_laps, recorded_interval_work_positions


def steady_pace_samples(intervals, origin, config, reference, *, with_windows=False):
    """Retain valid fixed wall-time windows instead of rejecting entire runs.

    Reuse configured aerobic-analysis duration/steadiness settings. Any missing
    movement/HR breaks its window, not all windows before and after it. No weather
    adjustment or extrapolation beyond the measured HR/pace domain is invented.
    Offline diagnostics may request original window coordinates; default sample
    objects and all validity decisions are unchanged.
    """
    settings = config.get('model', {})
    width = float(settings.get('primary_window_seconds', 120))
    warmup = float(settings.get('minimum_reliable_segment_minutes', 5))*60
    finish = float(settings.get('maximum_reliable_segment_minutes', 60))*60
    hr_change = float(settings.get('steady_benchmark_maximum_hr_change_bpm', 6))
    speed_change = float(settings.get('steady_benchmark_maximum_speed_change_fraction', .15))
    if width <= 0:
        raise ValueError('Positive capacity window required')
    bins = {}
    bad_flags = {'large_recording_gap', 'distance_reset', 'implausible_distance_speed'}
    stopped = suppression = 0.
    stop_threshold = float(settings.get('post_pause_minimum_stop_seconds', 60))
    suppress_seconds = float(settings.get('post_pause_suppression_moving_seconds', 180))
    for i in intervals:
        if i.moving_time_s <= 0:
            stopped += i.elapsed_s
            continue
        if stopped >= stop_threshold:
            suppression = suppress_seconds
        stopped = 0.
        if suppression > 0:
            suppression = max(0., suppression-i.moving_time_s)
            continue
        low = (i.start.timestamp_utc-origin).total_seconds()
        high = (i.end.timestamp_utc-origin).total_seconds()
        if (low < 0 or high <= low or bad_flags & set(i.flags) or i.distance_source == 'unknown'
                or abs(i.moving_time_s-i.elapsed_s) > 1e-8):
            continue
        a, b = reference.reserve(i.start.heart_rate_bpm), reference.reserve(i.end.heart_rate_bpm)
        if a is None or b is None:
            continue
        for index in range(int(low//width), int(high//width)+1):
            start, end = max(low, index*width), min(high, (index+1)*width)
            if end <= start or index*width < warmup or (index+1)*width > finish:
                continue
            first = a+(b-a)*(start-low)/(high-low)
            last = a+(b-a)*(end-low)/(high-low)
            bins.setdefault(index, []).append((start, end, i.distance_m/(high-low), first, last))
    samples = []
    for index, rows in sorted(bins.items()):
        if abs(sum(b-a for a,b,_,_,_ in rows)-width) > 1e-6:
            continue
        half_stats = []
        for left, right in ((index*width, (index+.5)*width), ((index+.5)*width, (index+1)*width)):
            distance = hr = 0.
            for a,b,speed,r0,r1 in rows:
                start, end = max(a,left), min(b,right)
                if end > start:
                    distance += (end-start)*speed
                    first = r0+(r1-r0)*(start-a)/(b-a)
                    last = r0+(r1-r0)*(end-a)/(b-a)
                    hr += (end-start)*(first+last)/2
            half_stats.append((distance/(width/2), hr/(width/2)))
        (v1,r1),(v2,r2) = half_stats
        if (v1 <= 0 or v2 <= 0 or abs(v2/v1-1) > speed_change
                or abs(r2-r1)*(reference.maximum_bpm-reference.resting_bpm) > hr_change):
            continue
        sample = PaceSample('easy', width, (v1+v2)/2*width,
                            'steady_raw_window_configured_validity_rules', (r1+r2)/2)
        samples.append((index*width,(index+1)*width,sample) if with_windows else sample)
    return samples


def evidence_from_activity(activity, config, *, key, available_at, workout_type,
                           capacity_eligible=True, profile=None, policy=Policy()):
    intervals = classify_movement(activity.trackpoints, config['moving_time']).intervals
    reference = HeartRateReference(config['resting_hr'], config['max_hr'], 'configured_at_issue')
    cardio = observed_cardio_trace(activity, intervals, reference, policy)
    bounds = cardio['load_bounds_within_reference']
    load = (Quantity(bounds['low'], bounds['high'], cardio['total_load'], bounds['basis'])
            if bounds is not None else None)
    profile = profile or observed_profile(activity, config, workout_type=workout_type)
    if profile['intensity_structure']['session_workout_type'] != workout_type:
        raise ValueError('Use the same resolved role for profile and projection evidence')
    samples = []
    bad_flags = {'large_recording_gap', 'distance_reset', 'implausible_distance_speed'}
    def reliable_lap(lap):
        if lap['start_time_utc'] is None:
            return False
        low = lap['start_time_utc']
        high = low+timedelta(seconds=lap['total_time_s'])
        covered = 0.
        for i in intervals:
            seconds = (min(high, i.end.timestamp_utc)-max(low, i.start.timestamp_utc)).total_seconds()
            if seconds <= 0:
                continue
            if (bad_flags & set(i.flags) or i.distance_source == 'unknown'
                    or abs(i.moving_time_s-i.elapsed_s) > 1e-8):
                return False
            covered += seconds
        return abs(covered-lap['total_time_s']) < .001
    moving = sum(i.moving_time_s for i in intervals)
    distance = sum(i.distance_m for i in intervals)
    if workout_type in {'easy', 'long'}:
        samples.extend(steady_pace_samples(intervals, activity.start_time_utc, config, reference))
    if workout_type == 'intervals':
        rows = usable_recorded_laps([asdict(lap) for lap in activity.laps])
        positions = sorted(recorded_interval_work_positions(rows))
        if len(positions) >= 2:
            for index in positions:
                lap = rows[index]
                # A lap total can hide pauses; do not turn it into moving pace
                # when the raw session has unresolved timing/movement.
                if reliable_lap(lap):
                    samples.append(PaceSample('work', lap['total_time_s'], lap['distance_m'],
                                              'recorded_work_lap'))
            for left, right in zip(positions, positions[1:]):
                for lap in rows[left+1:right]:
                    if reliable_lap(lap):
                        samples.append(PaceSample('recovery', lap['total_time_s'], lap['distance_m'],
                                                  'recorded_between_work_lap'))
        else:
            inferred = profile['intensity_structure'].get('pace_inferred_work')
            if inferred and inferred['usable_for_comparison']:
                for bout in inferred['bouts']:
                    if bout['distance_meters'] and bout['work_seconds']:
                        samples.append(PaceSample('work', bout['work_seconds'], bout['distance_meters'],
                                                  'existing_pace_inference_boundary_uncertainty'))
    return SessionEvidence(str(key), activity.start_time_utc,
        activity.start_time_utc+timedelta(seconds=cardio['elapsed_seconds']), available_at,
        workout_type, activity.total_distance_m or distance, moving, load,
        accounting_signature(config['resting_hr'], config['max_hr'], policy),
        tuple(samples), capacity_eligible)
