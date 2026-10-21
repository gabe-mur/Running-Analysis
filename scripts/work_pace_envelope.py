"""Offline duration-specific demonstrated output; no fitted critical-speed scalar.

Only work segments enter a work envelope. Missing durations remain unsupported,
not zero, and never erase actual dose. This is not a maximal-capacity estimator.
"""
from datetime import timedelta
from math import isfinite

from run_analysis.athlete_projection import aware
from run_analysis.signal_structure import SignalSpan, duration_curve


def work_spans(evidence):
    result = []
    for left, right in evidence.windows:
        for s in evidence.speed_mps:
            a, b = max(left, s.start), min(right, s.end)
            if b > a:
                at = lambda t: s.first+(s.last-s.first)*(t-s.start)/(s.end-s.start)
                result.append(SignalSpan(a, b, at(a), at(b)))
    return tuple(result)


def demonstrated_envelope(evidence, durations, as_of, *, lookback_days=84.):
    aware(as_of)
    evidence, durations = tuple(evidence), tuple(durations)
    if len({e.key for e in evidence}) != len(evidence):
        raise ValueError('Unique session identities required')
    if any(e.source != 'observed' for e in evidence):
        raise ValueError('Demonstrated performance requires observed evidence')
    if not isfinite(lookback_days) or lookback_days <= 0:
        raise ValueError('Positive finite lookback required')
    if not durations or len(set(durations)) != len(durations) or any(
            not isfinite(d) or d <= 0 for d in durations):
        raise ValueError('Unique positive finite durations required')
    prior = sorted((e for e in evidence if as_of-timedelta(days=lookback_days) <= e.start
                    and e.end < as_of and e.available_at <= as_of), key=lambda e: e.key)
    curves = [(e.key, duration_curve(work_spans(e), durations)) for e in prior]
    points = []
    for index, duration in enumerate(durations):
        available = [(key, curve[index]) for key, curve in curves if curve[index]['peak_mean_value'] is not None]
        if not available:
            points.append(dict(seconds=duration, speed_mps=None, source_key=None,
                               supporting_sessions=0, status='no_prior_continuous_work_support'))
            continue
        key, peak = min(available, key=lambda item: (-item[1]['peak_mean_value'], item[0]))
        points.append(dict(seconds=duration, speed_mps=peak['peak_mean_value'], source_key=key,
            start_seconds=peak['start_seconds'], supporting_sessions=len(available),
            status='demonstrated_work_output_not_maximal_capacity'))
    return dict(issued_at=as_of, lookback_days=lookback_days, source_keys=[e.key for e in prior],
        points=points, physiological_threshold_established=False, selection_ready=False,
        limitations=['No recovery concatenation, unsupported-duration extrapolation or fitted asymptote',
            'Reference is best observed work output, not expected pace, maximal capacity or a confidence interval',
            'Different repetitions, recovery spacing, effort and terrain remain distinct contexts',
            'No prior support does not erase completed work or block a novel prescription'])
