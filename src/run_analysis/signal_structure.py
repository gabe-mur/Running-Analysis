"""Label/lap-independent descriptions of recorded effort distribution.

No workout classification or fatigue multiplier. HR thresholds describe response;
raw speed describes external movement, not grade-adjusted metabolic intensity.
Invalid samples, stops and recording gaps break continuity rather than becoming
recovery evidence. Analysis windows/thresholds are inspectable, not safety limits.
"""
from bisect import bisect_right
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class SignalSpan:
    start: float
    end: float
    first: float
    last: float

    def __post_init__(self):
        if not all(isfinite(x) for x in (self.start, self.end, self.first, self.last)) or self.end <= self.start:
            raise ValueError('Finite ordered signal span required')


def _validate(spans):
    if any(b.start < a.end-1e-8 for a, b in zip(spans, spans[1:])):
        raise ValueError('Signal spans must be ordered and nonoverlapping')


def above_threshold(spans, threshold):
    """Exact duration above a threshold for piecewise-linear observations."""
    spans = tuple(spans)
    _validate(spans)
    if not isfinite(threshold):
        raise ValueError('Finite threshold required')
    episodes = []
    for span in spans:
        a, b = span.first, span.last
        if max(a, b) < threshold:
            continue
        low, high = span.start, span.end
        if min(a, b) < threshold:
            crossing = low+(high-low)*(threshold-a)/(b-a)
            if a < threshold:
                low = crossing
            else:
                high = crossing
        if high-low <= 1e-8:
            continue
        if episodes and abs(episodes[-1][1]-low) < 1e-8:
            episodes[-1][1] = high
        else:
            episodes.append([low, high])
    durations = [b-a for a, b in episodes]
    gaps = []
    for previous, following in zip(episodes, episodes[1:]):
        low, high = previous[1], following[0]
        covered = sum(max(0., min(s.end, high)-max(s.start, low)) for s in spans)
        gaps.append(dict(elapsed_seconds=high-low, known_below_threshold_seconds=covered,
                         unknown_seconds=max(0., high-low-covered)))
    total = sum(durations)
    known = sum(s.end-s.start for s in spans)
    return dict(threshold=threshold, known_seconds=known, above_seconds=total,
        fraction_of_known=total/known if known else None,
        longest_continuous_seconds=max(durations, default=0.) if spans else None,
        duration_weighted_episode_seconds=sum(d*d for d in durations)/total if total else None,
        episode_count=len(episodes) if spans else None,
        episodes=[dict(start_seconds=a, end_seconds=b) for a, b in episodes], gaps=gaps)


def duration_curve(spans, windows_seconds):
    """Exact peak time-weighted mean for piecewise-linear signals, O(W N log N).

    Test boundaries and interior stationary points of the rolling integral.
    Never bridge missing/stopped time, and never treat sample count as duration.
    """
    spans = tuple(spans)
    _validate(spans)
    blocks = []
    for span in spans:
        if not blocks or abs(blocks[-1][-1].end-span.start) > 1e-8:
            blocks.append([span])
        else:
            blocks[-1].append(span)
    result = []
    for window in windows_seconds:
        if not isfinite(window) or window <= 0:
            raise ValueError('Finite positive duration window required')
        best, best_start = None, None
        for block in blocks:
            if block[-1].end-block[0].start < window-1e-8:
                continue
            times = [block[0].start, *(s.end for s in block)]
            sums = [0.]
            for span in block:
                sums.append(sums[-1]+(span.end-span.start)*(span.first+span.last)/2)
            def integral(at):
                index = bisect_right(times, at)-1
                if index >= len(block):
                    return sums[-1]
                span = block[index]
                delta = at-times[index]
                slope = (span.last-span.first)/(span.end-span.start)
                return sums[index]+delta*span.first+delta*delta*slope/2
            def value(at):
                index = min(len(block)-1, bisect_right(times, at)-1)
                span = block[index]
                return span.first+(span.last-span.first)*(at-span.start)/(span.end-span.start)
            starts = {max(times[0], min(times[-1]-window, t))
                      for t in (*times, *(t-window for t in times))}
            boundaries = sorted(starts)
            for left, right in zip(boundaries, boundaries[1:]):
                if right-left < 1e-8:
                    continue
                a, b = left+(right-left)/3, left+2*(right-left)/3
                # d/dt integral[t,t+w] = value(t+w)-value(t), affine here.
                da, db = value(a+window)-value(a), value(b+window)-value(b)
                slope = (db-da)/(b-a)
                if abs(slope) > 1e-10:
                    root = a-da/slope
                    if left < root < right:
                        starts.add(root)
            for start in sorted(starts):
                mean = (integral(start+window)-integral(start))/window
                if best is None or mean > best+1e-10:
                    best, best_start = mean, start
        result.append(dict(window_seconds=window, peak_mean_value=best,
                           start_seconds=best_start))
    return result


def speed_duration_curve(spans, windows_seconds):
    spans = tuple(spans)
    if any(s.first != s.last for s in spans):
        raise ValueError('Speed duration curve requires interval-average speeds')
    return [dict(window_seconds=x['window_seconds'], peak_mean_mps=x['peak_mean_value'],
                 start_seconds=x['start_seconds']) for x in duration_curve(spans, windows_seconds)]


def describe_signals(intervals, *, hr_thresholds_bpm, speed_thresholds_mps=(),
                     windows_seconds=(30., 60., 120., 300., 600.)):
    """Consume movement intervals only: never lap labels, run type or a plan."""
    hr, speed = [], []
    origin = intervals[0].start.timestamp_utc if intervals else None
    for interval in intervals:
        if (interval.moving_time_s <= 0 or 'large_recording_gap' in interval.flags
                or abs(interval.moving_time_s-interval.elapsed_s) > 1e-8):
            continue
        start = (interval.start.timestamp_utc-origin).total_seconds()
        end = (interval.end.timestamp_utc-origin).total_seconds()
        if end <= start:
            continue
        a, b = interval.start.heart_rate_bpm, interval.end.heart_rate_bpm
        if all(x is not None and isfinite(x) and x > 0 for x in (a, b)):
            hr.append(SignalSpan(start, end, a, b))
        if (interval.distance_source != 'unknown' and interval.distance_m >= 0
                and not {'implausible_distance_speed', 'distance_reset'} & set(interval.flags)):
            value = interval.distance_m/interval.elapsed_s
            if isfinite(value):
                speed.append(SignalSpan(start, end, value, value))
    return dict(version='recorded_signal_structure_v1',
        hr_exposure=[above_threshold(hr, x) for x in hr_thresholds_bpm],
        hr_duration_curve=[dict(window_seconds=x['window_seconds'], peak_mean_bpm=x['peak_mean_value'],
                                start_seconds=x['start_seconds']) for x in duration_curve(hr, windows_seconds)],
        speed_exposure=[above_threshold(speed, x) for x in speed_thresholds_mps],
        speed_duration_curve=speed_duration_curve(speed, windows_seconds),
        coverage=dict(hr_seconds=sum(s.end-s.start for s in hr),
                      speed_seconds=sum(s.end-s.start for s in speed),
                      total_elapsed_seconds=sum(i.elapsed_s for i in intervals)),
        assumptions=['Linear HR between valid endpoints; constant interval-average speed',
                     'Threshold episodes are not automatically workout repetitions',
                     'Raw speed is affected by terrain, wind and GPS; not absolute intensity',
                     'Missing/stopped time breaks continuity and is not assumed low effort',
                     'Duration windows and thresholds describe the signal, not recovery costs'],
        planner_influence=False)
