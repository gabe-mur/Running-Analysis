"""Offline empirical frontier: candidate CS reference, never proven physiology.

The 3/7/12-minute windows follow Hunter et al. (2023). A linear distance-time
fit is a model of demonstrated training output. Its residuals and deletion
sensitivity do not measure uncertainty about maximal capacity or a threshold.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import isfinite, fsum

from run_analysis.athlete_projection import aware

WINDOWS = (180., 420., 720.)


@dataclass(frozen=True)
class ObservedSpeedCurve:
    key: str
    start: datetime
    end: datetime
    available_at: datetime
    # duration, peak continuous mean speed; no concatenation of repetitions
    points: tuple[tuple[float, float], ...]

    def __post_init__(self):
        for t in (self.start, self.end, self.available_at):
            aware(t)
        object.__setattr__(self, 'points', tuple(tuple(p) for p in self.points))
        if not self.key or not self.start < self.end <= self.available_at:
            raise ValueError('Completed dated observation required')
        if len({p[0] for p in self.points}) != len(self.points):
            raise ValueError('Unique duration points required')
        if any(not isfinite(t) or not isfinite(v) or t <= 0 or v <= 0
               or t > (self.end-self.start).total_seconds() for t, v in self.points):
            raise ValueError('Positive finite, session-supported duration and speed required')


def fit_frontier(rows):
    """Maximum demonstrated distance at each fixed duration; deterministic ties."""
    anchors = []
    for seconds in WINDOWS:
        candidates = [(speed, row.key) for row in rows for t, speed in row.points if t == seconds]
        if not candidates:
            return dict(status='incomplete_duration_support', anchors=anchors,
                        slope_mps=None, intercept_meters=None, r_squared=None)
        speed, key = min(candidates, key=lambda pair: (-pair[0], pair[1]))
        anchors.append(dict(seconds=seconds, meters=seconds*speed, speed_mps=speed, source_key=key))
    tbar = fsum(p['seconds'] for p in anchors)/3
    dbar = fsum(p['meters'] for p in anchors)/3
    slope = fsum((p['seconds']-tbar)*(p['meters']-dbar) for p in anchors)/fsum(
        (p['seconds']-tbar)**2 for p in anchors)
    intercept = dbar-slope*tbar
    residuals = [p['meters']-(intercept+slope*p['seconds']) for p in anchors]
    total = fsum((p['meters']-dbar)**2 for p in anchors)
    r2 = 1-fsum(e*e for e in residuals)/total if total else None
    shape = slope > 0 and intercept > 1e-8 and all(
        a['speed_mps'] >= b['speed_mps'] for a, b in zip(anchors, anchors[1:]))
    return dict(status='empirical_frontier_only' if shape else 'incompatible_or_uninformative_shape',
        slope_mps=slope, intercept_meters=intercept, r_squared=r2, anchors=anchors,
        residual_meters=residuals, distinct_anchor_sessions=len({a['source_key'] for a in anchors}))


def freeze_frontier(rows, as_of, *, lookback_days=42.):
    aware(as_of)
    rows = tuple(rows)
    if len({r.key for r in rows}) != len(rows):
        raise ValueError('Unique activity identities required')
    if not isfinite(lookback_days) or lookback_days <= 0:
        raise ValueError('Positive finite lookback required')
    prior = tuple(sorted((r for r in rows if as_of-timedelta(days=lookback_days) <= r.start
        and r.end < as_of and r.available_at <= as_of), key=lambda r: (r.start, r.key)))
    fit = fit_frontier(prior)
    contributors = sorted({a['source_key'] for a in fit['anchors']})
    deletions = {key: fit_frontier(tuple(r for r in prior if r.key != key)) for key in contributors}
    slopes = [fit['slope_mps'], *(f['slope_mps'] for f in deletions.values())]
    slopes = [s for s in slopes if s is not None]
    return dict(issued_at=as_of, lookback_days=lookback_days, source_keys=[r.key for r in prior],
        fit=fit, contributor_deletions=deletions,
        deletion_sensitivity_mps=(min(slopes), max(slopes)) if slopes else None,
        selection_ready=False, physiological_threshold_established=False,
        uncertainty='Deletion sensitivity is not a confidence or prediction interval',
        limitations=['Habitual efforts are not known maximal; a high R-squared cannot establish maximality',
            'Raw speed is not corrected for grade, wind, surface, GPS or device differences',
            'Slope is a candidate CS estimate, not a validated individual threshold',
            'Intercept is not validated anaerobic capacity or a recoverable work budget',
            'No target benefit, readiness gate, exact-workout prerequisite or live planner influence'])
