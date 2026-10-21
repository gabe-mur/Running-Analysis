"""Recorded time to an HR band, not tissue readiness or a minimum effective dose."""
from math import ceil, floor, isfinite

from .signal_structure import SignalSpan


def moving_hr_spans(activity, intervals):
    """Use existing movement classification; stops and unknown HR break coverage."""
    spans = []
    for i in intervals:
        if ('large_recording_gap' in i.flags or i.moving_time_s <= 0 or
                abs(i.moving_time_s-i.elapsed_s) > 1e-8):
            continue
        a, b = i.start.heart_rate_bpm, i.end.heart_rate_bpm
        if any(x is None or not isfinite(x) or x <= 0 for x in (a, b)):
            continue
        start = (i.start.timestamp_utc-activity.start_time_utc).total_seconds()
        end = (i.end.timestamp_utc-activity.start_time_utc).total_seconds()
        if start >= 0 and end > start:
            spans.append(SignalSpan(start, end, a, b))
    return tuple(spans)


def _band_piece(s, low, high):
    cuts = [0., 1.]
    if s.first != s.last:
        cuts += [(edge-s.first)/(s.last-s.first) for edge in (low, high)
                 if 0 < (edge-s.first)/(s.last-s.first) < 1]
    cuts.sort()
    for a, b in zip(cuts, cuts[1:]):
        value = s.first+(s.last-s.first)*(a+b)/2
        if low <= value < high:
            yield s.start+a*(s.end-s.start), s.start+b*(s.end-s.start)


def describe_entry(spans, *, elapsed_seconds, low_bpm, high_bpm_exclusive,
                   confirmation_seconds, required_fraction):
    """Exact linear band exposure; sustained-entry starts searched on a 1s grid.

    Confirmation requires complete moving-HR coverage and tolerates known brief
    out-of-band fluctuations. Higher HR is not credited as easy zone-2 exposure.
    The look-ahead confirmation is descriptive, never a real-time causal signal.
    """
    if (not all(isfinite(x) for x in (elapsed_seconds, low_bpm, high_bpm_exclusive, required_fraction))
            or elapsed_seconds < 0 or not 0 < low_bpm < high_bpm_exclusive
            or type(confirmation_seconds) is not int or confirmation_seconds <= 0
            or not 0 < required_fraction <= 1):
        raise ValueError('Valid explicit duration, band and confirmation assumptions required')
    spans = tuple(spans)
    if any(s.start < 0 or s.end > elapsed_seconds+1e-8 for s in spans) or any(
            b.start < a.end-1e-8 for a, b in zip(spans, spans[1:])):
        raise ValueError('Ordered nonoverlapping spans within recorded elapsed duration required')
    n = ceil(elapsed_seconds)
    known, band = [0.]*n, [0.]*n
    def add(values, start, end):
        for index in range(floor(start), min(n, ceil(end))):
            values[index] += max(0., min(end, index+1)-max(start, index))
    pieces = []
    for s in spans:
        add(known, s.start, s.end)
        for start, end in _band_piece(s, low_bpm, high_bpm_exclusive):
            pieces.append((start, end))
            add(band, start, end)
    def prefix(values):
        result = [0.]
        for value in values:
            result.append(result[-1]+value)
        return result
    pk, pb = prefix(known), prefix(band)
    window = confirmation_seconds
    settled = None
    for start in range(max(0, floor(elapsed_seconds)-window+1)):
        if (band[start] > 1e-8 and pk[start+window]-pk[start] >= window-1e-6
                and pb[start+window]-pb[start] >= required_fraction*window-1e-6):
            # Search resolution is one second, not claimed exact physiology.
            settled = start
            break
    return dict(low_bpm=low_bpm, high_bpm_exclusive=high_bpm_exclusive,
        confirmation_seconds=window, required_fraction=required_fraction,
        first_entry_seconds=pieces[0][0] if pieces else None,
        sustained_entry_seconds=settled,
        confirmation_available_seconds=settled+window if settled is not None else None,
        known_moving_hr_seconds=pk[-1], zone_seconds=pb[-1],
        zone_seconds_from_sustained_entry=(pb[-1]-pb[settled] if settled is not None else None),
        first_observed_hr_offset_seconds=spans[0].start if spans else None,
        first_observed_hr_bpm=spans[0].first if spans else None,
        already_at_or_above_band_on_first_observation=bool(spans and spans[0].first >= low_bpm),
        status='observed_sustained_entry' if settled is not None else 'no_qualifying_window_observed',
        physiological_warmup_established=False, planner_influence=False)
