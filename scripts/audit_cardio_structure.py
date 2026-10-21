"""Read-only, additive decomposition of cardio points; no fatigue ordering."""
from argparse import ArgumentParser
from dataclasses import asdict, replace
from datetime import timedelta
import json
from pathlib import Path

from run_analysis.config import load_config
from run_analysis.fit import parse_fit
from run_analysis.tcx import parse_tcx
from run_analysis.movement import classify_movement
from run_analysis.training_load import ZONE_WEIGHTS
from run_analysis.workload_accounting import continuous_zone_seconds
from run_analysis.workout_detection import recorded_interval_work_positions, usable_recorded_laps
from run_analysis.signal_structure import describe_signals


def summarize(intervals, zones):
    seconds = continuous_zone_seconds(intervals, zones)
    known = [i for i in intervals if i.moving_time_s > 0
             and 'large_recording_gap' not in i.flags
             and all(x is not None and x > 0 for x in (i.start.heart_rate_bpm, i.end.heart_rate_bpm))]
    known_seconds = sum(i.moving_time_s for i in known)
    moving = sum(i.moving_time_s for i in intervals)
    distance = sum(i.distance_m for i in intervals if i.moving_time_s > 0)
    return dict(moving_minutes=moving/60, moving_distance_meters=distance,
        pace_min_mile=moving/60/(distance/1609.344) if distance else None,
        mean_hr=(sum(i.moving_time_s*(i.start.heart_rate_bpm+i.end.heart_rate_bpm)/2
                     for i in known)/known_seconds if known_seconds else None),
        zone_minutes={k: v/60 for k, v in seconds.items()},
        zone_points={k: seconds[k]*v/60 for k, v in ZONE_WEIGHTS.items()},
        points=sum(seconds[k]*v/60 for k, v in ZONE_WEIGHTS.items()))


def slice_interval(interval, low, high):
    """Clip at wall-time boundaries; preserve linear HR integration exactly."""
    a, b = interval.start, interval.end
    duration = (b.timestamp_utc-a.timestamp_utc).total_seconds()
    left = (low-a.timestamp_utc).total_seconds()/duration
    right = (high-a.timestamp_utc).total_seconds()/duration
    def point(at, fraction):
        hr = (a.heart_rate_bpm+(b.heart_rate_bpm-a.heart_rate_bpm)*fraction
              if a.heart_rate_bpm is not None and b.heart_rate_bpm is not None else None)
        return replace(a, timestamp_utc=at, heart_rate_bpm=hr)
    return replace(interval, start=point(low, left), end=point(high, right),
        elapsed_s=duration*(right-left), moving_time_s=interval.moving_time_s*(right-left),
        stopped_time_s=interval.stopped_time_s*(right-left), distance_m=interval.distance_m*(right-left))


def audit_file(path, config):
    parser = {'.tcx': parse_tcx, '.fit': parse_fit}[path.suffix.lower()]
    activity = parser(path, default_timezone=config['timezone_default']).activities[0]
    intervals = classify_movement(activity.trackpoints, config['moving_time']).intervals
    rows = [asdict(lap) for lap in activity.laps]
    usable = usable_recorded_laps(rows)
    positions = sorted(recorded_interval_work_positions(rows))
    work_laps = [usable[i] for i in positions] if len(positions) >= 2 else []
    windows = [(lap['start_time_utc'], lap['start_time_utc']+timedelta(seconds=lap['total_time_s']))
               for lap in work_laps if lap['start_time_utc'] is not None]
    groups = {name: [] for name in ('pre_work', 'work', 'between_work', 'post_work', 'unresolved')}
    for interval in intervals:
        low, high = interval.start.timestamp_utc, interval.end.timestamp_utc
        cuts = sorted({low, high, *(at for span in windows for at in span if low < at < high)})
        for a, b in zip(cuts, cuts[1:]):
            midpoint = a+(b-a)/2
            key = ('unresolved' if not windows else
                   'work' if any(start <= midpoint < end for start, end in windows) else
                   'pre_work' if midpoint < windows[0][0] else
                   'post_work' if midpoint >= windows[-1][1] else 'between_work')
            groups[key].append(slice_interval(interval, a, b))
    summaries = {key: summarize(value, config['zones']) for key, value in groups.items() if value}
    total = summarize(intervals, config['zones'])
    if abs(sum(x['points'] for x in summaries.values())-total['points']) > 1e-8:
        raise ValueError('Phase cardio points do not sum to whole-session points')
    if abs(sum(x['moving_minutes'] for x in summaries.values())-total['moving_minutes']) > 1e-8:
        raise ValueError('Phase durations do not sum to whole-session duration')
    return dict(source=path.name, start=activity.start_time_utc.isoformat(), total=total,
        phases=summaries, recorded_work_seconds=[lap['total_time_s'] for lap in work_laps],
        recorded_signals=describe_signals(intervals,
            hr_thresholds_bpm=[config['zones'][name][0] for name in ('z3', 'z4', 'z5')],
            # Shared descriptive grid, not personal effort thresholds or a classifier.
            speed_thresholds_mps=(2.5, 3., 3.5, 4.)),
        zone_shift_sensitivity={str(shift): summarize(intervals,
            {name: [v+shift for v in bounds] for name, bounds in config['zones'].items()})['points']
            for shift in (-3, 0, 3)},
        interpretation='Recorded lap contrasts used only to explain the sum, not to rank fatigue. '
            'Post-work includes final recovery and cooldown. Moving-time HR only; no EPOC or HRV.',
        planner_influence=False)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('sources', nargs='+', type=Path)
    parser.add_argument('--config', type=Path, default=Path('config.yaml'))
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output path')
    result = [audit_file(path, load_config(args.config)) for path in args.sources]
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    for session in result:
        print(session['source'], {k: round(v['points'], 3) for k, v in session['phases'].items()},
              'total', round(session['total']['points'], 3))


if __name__ == '__main__':
    main()
