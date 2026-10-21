"""Experimental exercise RR analysis, never a readiness or fitness score.

Raw RR remains immutable. FIT record brackets locate windows approximately;
they do not timestamp beats. No stitching across missing brackets or pauses,
no rescaling to a timer, and no universal DFA threshold inference.
"""
from collections import Counter
from datetime import datetime
from functools import lru_cache
import math

import numpy as np

VERSION = 'exercise-hrv-experimental-v1'
SCALES = tuple(range(4, 17))


def screen_rr(values, relative_threshold=.20):
    """Local screen, NOT a clinically validated artifact detector."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not 0 < relative_threshold < 1:
        raise ValueError('One dimensional RR and fractional threshold required')
    medians = np.array([np.median(values[max(0, i-5):i+6]) for i in range(len(values))])
    return ((~np.isfinite(values)) | (values < .25) | (values > 2.)
            | (np.abs(values-medians) > relative_threshold*medians))


def dfa_alpha1(values, scales=SCALES):
    """Linear DFA, beat-domain 4–16 scales, deterministic least-squares fit.

    Fifty-percent overlapping subseries, pooled mean-square fluctuation. The
    final exactly aligned subseries is omitted to match nolds 0.6.2's declared
    overlap convention. Independent vector projection avoids per-box polyfits.
    """
    values = np.asarray(values, dtype=float)
    if (values.ndim != 1 or len(values) < 10*max(scales)
            or not np.isfinite(values).all() or len(set(scales)) < 2
            or any(type(n) is not int or n < 4 for n in scales)):
        raise ValueError('Finite series with ten boxes at the largest scale required')
    if np.ptp(values) <= np.finfo(float).eps*max(1., float(np.max(abs(values)))):
        raise ValueError('Constant RR has no identifiable DFA exponent')
    walk = np.cumsum(values-values.mean())
    fluctuation = []
    for n in scales:
        starts = np.arange(0, len(values)-n, n//2)
        boxes = walk[starts[:, None]+np.arange(n)]
        x = np.arange(n, dtype=float)-(n-1)/2
        centered = boxes-boxes.mean(axis=1, keepdims=True)
        residual = centered-(centered@x/(x@x))[:, None]*x
        fluctuation.append(float(np.sqrt(np.mean(residual**2))))
    if min(fluctuation) <= 0:
        raise ValueError('Degenerate fluctuation curve')
    logn, logf = np.log(scales), np.log(fluctuation)
    centered = logn-logn.mean()
    slope = float(centered@(logf-logf.mean())/(centered@centered))
    fitted = logf.mean()+slope*centered
    total = float(np.sum((logf-logf.mean())**2))
    return dict(alpha1=slope, scales=list(scales), fluctuation=fluctuation,
                log_fit_r_squared=1-float(np.sum((logf-fitted)**2))/total if total else None)


@lru_cache(maxsize=256)
def _smoothness_cholesky(n, strength):
    # I + lambda^2 D2.T D2 is pentadiagonal. No new scipy dependency or
    # full-session dense inversion; this is exact banded Cholesky arithmetic.
    second = np.diff(np.eye(3), n=2, axis=0)[0]
    diagonal, lower1, lower2 = np.ones(n), np.zeros(n), np.zeros(n)
    for i in range(n-2):
        diagonal[i:i+3] += strength**2*second**2
        lower1[i+1:i+3] += strength**2*second[1:]*second[:-1]
        lower2[i+2] += strength**2
    l0, l1, l2 = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(n):
        if i >= 2:
            l2[i] = lower2[i]/l0[i-2]
        if i >= 1:
            l1[i] = (lower1[i]-l2[i]*l1[i-1])/l0[i-1]
        l0[i] = math.sqrt(diagonal[i]-l1[i]**2-l2[i]**2)
    for vector in (l0, l1, l2):
        vector.flags.writeable = False
    return l0, l1, l2


def smoothness_detrend(values, strength=500.):
    """Tarvainen-form smoothness prior in beat domain, explicit sensitivity.

    Lambda is an engineering preprocessing choice here, not a claim to use
    Kubios' time-domain/filter settings. Only derived copies are transformed.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 4 or not np.isfinite(values).all() or not math.isfinite(strength) or strength <= 0:
        raise ValueError('Finite series and positive smoothing strength required')
    l0, l1, l2 = _smoothness_cholesky(len(values), strength)
    forward, trend = np.zeros(len(values)), np.zeros(len(values))
    # Center first so large mean RR cannot magnify numerical cancellation.
    centered = values-values.mean()
    for i in range(len(values)):
        forward[i] = (centered[i]-(l1[i]*forward[i-1] if i else 0)
                      -(l2[i]*forward[i-2] if i >= 2 else 0))/l0[i]
    for i in range(len(values)-1, -1, -1):
        trend[i] = (forward[i]-(l1[i+1]*trend[i+1] if i+1 < len(values) else 0)
                    -(l2[i+2]*trend[i+2] if i+2 < len(values) else 0))/l0[i]
    return centered-trend


def correction_sensitivity(values, *, max_flagged_fraction=.03):
    """Sparse interior interpolation only; compare all fixed screen settings.

    Clusters, edge defects and highly flagged windows stay unavailable. A
    retained result is exploratory, not reference-validated artifact repair.
    """
    rr = np.asarray(values, dtype=float)
    if rr.ndim != 1 or len(rr) < 160 or not np.isfinite(rr).all() or np.any(rr <= 0):
        return dict(status='unsupported_rr', variants=[], usable_for_comparison=False)
    if np.ptp(rr) <= np.finfo(float).eps:
        return dict(status='constant_rr', variants=[], usable_for_comparison=False)
    variants, reasons = [], []
    for threshold in (.10, .20, .30):
        flagged = screen_rr(rr, threshold)
        fraction = float(flagged.mean())
        indexes = np.flatnonzero(flagged)
        clustered = any(b == a+1 for a, b in zip(indexes, indexes[1:]))
        if fraction > max_flagged_fraction or (len(indexes) and (indexes[0] == 0 or indexes[-1] == len(rr)-1)) or clustered:
            reasons.append(dict(screen_threshold=threshold, flagged_fraction=fraction,
                                reason='too_many_flags_or_edge_or_consecutive_flags'))
            continue
        cleaned = rr.copy()
        if indexes.size:
            cleaned[indexes] = np.interp(indexes, np.flatnonzero(~flagged), rr[~flagged])
        for strength in (250., 500., 1000.):
            try:
                result = dfa_alpha1(smoothness_detrend(cleaned, strength))
            except ValueError:
                reasons.append(dict(screen_threshold=threshold, reason='degenerate_after_preprocessing'))
                continue
            variants.append(dict(screen_threshold=threshold, flagged_fraction=fraction,
                                 smoothing_strength=strength, **result))
    if not variants:
        return dict(status='screening_rejected', variants=[], rejection_details=reasons,
                    usable_for_comparison=False)
    central = next((r for r in variants if r['screen_threshold'] == .20 and r['smoothing_strength'] == 500.), None)
    spread = max(r['alpha1'] for r in variants)-min(r['alpha1'] for r in variants)
    stable = len(variants) == 9 and spread <= .10
    return dict(status='exploratory_stable' if stable else 'preprocessing_sensitive',
        alpha1=central['alpha1'] if central else None, raw_alpha1=dfa_alpha1(rr)['alpha1'],
        preprocessing_range=[min(r['alpha1'] for r in variants), max(r['alpha1'] for r in variants)],
        variants=variants, rejection_details=reasons, usable_for_comparison=stable,
        clinical_artifact_rate_established=False)


def _date(value):
    return datetime.fromisoformat(value) if isinstance(value, str) else value


def exercise_rr_windows(payload, start, elapsed_seconds, *, window_seconds=120, stride_seconds=30):
    """Approximate wall-time windows preserving whole ordered source brackets."""
    if (not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0
            or type(window_seconds) is not int or window_seconds <= 0
            or type(stride_seconds) is not int or stride_seconds <= 0):
        raise ValueError('Positive finite duration and integer windows required')
    start = _date(start)
    groups = []
    for batch in payload.get('rr_batches', []):
        a, b = batch.get('preceding_record_utc'), batch.get('following_record_utc')
        values = batch['intervals_seconds']
        # FIT's unused array positions are trailing None padding. An internal
        # invalid slot is not silently deleted from a beat sequence.
        finite_positions = [i for i, v in enumerate(values) if v is not None]
        internal_missing = bool(finite_positions and any(v is None for v in values[:finite_positions[-1]]))
        rr = [float(v) for v in values if v is not None]
        if not rr:
            continue
        bracket = (a, b)
        if not groups or groups[-1]['bracket'] != bracket:
            groups.append(dict(bracket=bracket, rr=[], internal_missing=False))
        groups[-1]['rr'].extend(rr)
        groups[-1]['internal_missing'] |= internal_missing
    valid = []
    for group in groups:
        a, b = group['bracket']
        if a and b:
            group.update(a=(_date(a)-start).total_seconds(), b=(_date(b)-start).total_seconds())
            valid.append(group)
    windows = []
    events = sorted(((_date(e['timestamp'])-start).total_seconds(), e['event_type'])
                    for e in payload.get('timer_events', []))
    for offset in range(0, max(0, int(elapsed_seconds)-window_seconds+1), stride_seconds):
        end = offset+window_seconds
        selected = [g for g in valid if offset <= g['a'] and g['b'] <= end]
        base = dict(start_seconds=offset, end_seconds=end, timing='source_brackets_approximate',
                    physiological_threshold_established=False)
        previous = [kind for at, kind in events if at <= offset]
        stopped = not previous or previous[-1] != 'start' or any(
            offset < at < end and kind != 'start' for at, kind in events)
        if stopped:
            windows.append(dict(base, status='paused_or_timer_unknown', usable_for_comparison=False)); continue
        if (not selected or any(g['internal_missing'] or g['b'] <= g['a'] or g['b']-g['a'] > 15 for g in selected)
                or any(abs(a['b']-b['a']) > 1e-6 for a, b in zip(selected, selected[1:]))):
            windows.append(dict(base, status='missing_or_discontinuous_rr', usable_for_comparison=False)); continue
        rr = np.array([r for group in selected for r in group['rr']])
        covered = selected[-1]['b']-selected[0]['a']
        agreement = float(rr.sum()-covered)
        if covered < .90*window_seconds or not np.isfinite(rr).all() or abs(agreement) > max(2., .02*covered):
            windows.append(dict(base, status='incomplete_or_clock_mismatch', covered_seconds=covered,
                                rr_minus_bracket_seconds=agreement if math.isfinite(agreement) else None,
                                usable_for_comparison=False)); continue
        result = correction_sensitivity(rr)
        windows.append(dict(base, covered_seconds=covered, rr_minus_bracket_seconds=agreement,
            rr_count=len(rr), mean_rr_hr_bpm=60/float(rr.mean()), **result))
    return dict(version=VERSION, windows=windows, status_counts=dict(Counter(w['status'] for w in windows)),
        rr_count=sum(len(g['rr']) for g in groups),
        unanchored_rr_count=sum(len(g['rr']) for g in groups if not all(g['bracket'])),
        assumptions=dict(window_seconds=window_seconds,stride_seconds=stride_seconds,
            scales_beats=list(SCALES), maximum_screen_flag_fraction=.03,
            maximum_preprocessing_spread=.10, maximum_record_bracket_seconds=15,
            minimum_wall_time_coverage=.90, clock_tolerance='max(2 seconds, 2% of bracket span)',
            preprocessing='beat-domain smoothness prior lambda 250/500/1000; sparse interior linear interpolation',
            interpretation='Engineering screens and sensitivity ranges, not clinical artifact rates or confidence intervals'),
        independent_observations='Physical runs, not overlapping windows',
        planner_influence=False, fitness_influence=False)
