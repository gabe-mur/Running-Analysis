"""Reuse existing observed work segmentation; no new interval classifier."""
from dataclasses import asdict
from datetime import datetime

from .delivered_training import continuous_phase_from_activity
from .signal_structure import SignalSpan
from .work_stimulus import WorkSignals
from .workload_profile import observed_profile
from .workout_detection import recorded_interval_work_positions, usable_recorded_laps


def observed_work_signals(run, activity, config, intervals, hrr, *, easy_speed_mps=None,
                          reference_at=None, reference_source=''):
    if run.source != 'observed' or run.start != activity.start_time_utc:
        raise ValueError('Original completed activity must match the run')
    if not run.bouts:
        return None
    windows = []
    if run.work_evidence_basis == 'recorded_lap_contrast':
        laps = usable_recorded_laps([asdict(lap) for lap in activity.laps])
        selected = [laps[pos] for pos in sorted(recorded_interval_work_positions(laps,intervals))]
        for lap in selected:
            if lap['start_time_utc'] is None:
                return None
            start = (lap['start_time_utc']-run.start).total_seconds()
            windows.append((start, start+lap['total_time_s']))
    elif run.work_evidence_basis == 'recorded_continuous_lap':
        phase = continuous_phase_from_activity(activity, config)
        if phase is None:
            return None
        lap = next(lap for lap in activity.laps if lap.lap_index == phase.lap_index)
        if lap.start_time_utc is None:
            return None
        start = (lap.start_time_utc-run.start).total_seconds()
        windows.append((start, start+phase.duration_seconds))
    elif run.work_evidence_basis == 'pace_inferred':
        profile = observed_profile(activity, config, workout_type='intervals')
        for window in profile['intensity_structure']['pace_inferred_work']['windows']:
            windows.append(tuple((datetime.fromisoformat(window[k])-run.start).total_seconds()
                                 for k in ('start', 'end')))
    else:
        return None
    if len(windows) != len(run.bouts) or any(abs((b-a)-bout.work_seconds) > 1e-3
                                           for (a, b), bout in zip(windows, run.bouts)):
        return None
    speeds = []
    bad = {'large_recording_gap', 'distance_reset', 'implausible_distance_speed'}
    for i in intervals:
        if (i.elapsed_s <= 0 or abs(i.moving_time_s-i.elapsed_s) > 1e-8
                or bad & set(i.flags) or i.distance_source == 'unknown'):
            continue
        start = (i.start.timestamp_utc-run.start).total_seconds()
        end = (i.end.timestamp_utc-run.start).total_seconds()
        speeds.append(SignalSpan(start, end, i.distance_m/i.elapsed_s, i.distance_m/i.elapsed_s))
    return WorkSignals(run.key, run.source, run.start, run.end, run.available_at,
                       tuple(windows), tuple(hrr), tuple(speeds), run.work_evidence_basis,
                       easy_speed_mps, reference_at, reference_source)
