"""Target-specific work exposure, not a fitted adaptation or recovery model.

No default target bands: their values are declared coaching scenarios. HR and
pace are separate questions, never summed or maximized into one benefit score.
Only supported work windows count; load outside them stays in the load model.
"""
from dataclasses import dataclass, field
from datetime import datetime
from math import fsum, isfinite, sqrt

from .athlete_projection import aware
from .signal_structure import SignalSpan
from .workload_accounting import Quantity


@dataclass(frozen=True)
class StimulusTarget:
    name: str
    channel: str  # hrr or speed_over_easy; neither is a physiological threshold.
    band: tuple[float, float, float, float]  # zero, full, full, zero
    minutes: float
    source: str
    version: str = field(default='work_signal_suitability_v2_shared_reference', init=False)

    def __post_init__(self):
        object.__setattr__(self, 'band', tuple(self.band))
        if (not self.name or not self.source or self.channel not in {'hrr', 'speed_over_easy'}
                or len(self.band) != 4 or any(not isfinite(x) for x in self.band)
                or not 0 <= self.band[0] < self.band[1] <= self.band[2] < self.band[3]
                or not isfinite(self.minutes) or self.minutes <= 0
                or (self.channel == 'hrr' and self.band[3] > 1)):
            raise ValueError('Explicit finite target band, channel and positive dose required')

    def suitability(self, value):
        a, b, c, d = self.band
        return max(0., min(1., (value-a)/(b-a), (d-value)/(d-c)))


@dataclass(frozen=True)
class CappedStimulusTarget:
    """Declared exposure requirement: ramp to full credit, then cap per minute.

    Higher intensity earns no extra minute credit, but does not erase already
    supported exposure. Excess intensity remains a cost/constraint question.
    The old band target and its archived identities are unchanged.
    """
    name: str
    channel: str
    onset: float
    full: float
    minutes: float
    source: str
    version: str = field(default='work_signal_capped_exposure_v1', init=False)

    def __post_init__(self):
        if (not self.name or not self.source or self.channel not in {'hrr', 'speed_over_easy', 'speed_mps'}
                or any(not isfinite(x) for x in (self.onset, self.full, self.minutes))
                or not 0 <= self.onset < self.full or self.minutes <= 0
                or self.channel == 'hrr' and self.full > 1):
            raise ValueError('Explicit ordered exposure anchors, positive dose and channel required')

    @property
    def band(self):
        return self.onset, self.full

    def suitability(self, value):
        return max(0., min(1., (value-self.onset)/(self.full-self.onset)))


@dataclass(frozen=True)
class WorkSignals:
    key: str
    source: str
    start: datetime
    end: datetime
    available_at: datetime
    windows: tuple[tuple[float, float], ...]
    hrr: tuple[SignalSpan, ...]
    speed_mps: tuple[SignalSpan, ...]
    basis: str
    easy_speed_mps: Quantity | None = None
    reference_at: datetime | None = None
    reference_source: str = ''

    def __post_init__(self):
        for at in (self.start, self.end, self.available_at):
            aware(at)
        for name in ('windows', 'hrr', 'speed_mps'):
            object.__setattr__(self, name, tuple(tuple(w) for w in self.windows) if name == 'windows'
                               else tuple(getattr(self, name)))
        if not self.key or self.source not in {'observed', 'planned'} or not self.basis or self.end <= self.start:
            raise ValueError('Identified evidence with ordered session times required')
        if ((self.source == 'observed' and self.available_at < self.end)
                or (self.source == 'planned' and self.available_at > self.start)):
            raise ValueError('Evidence provenance and availability disagree')
        seconds = (self.end-self.start).total_seconds()
        previous = 0.
        for left, right in self.windows:
            if not all(isfinite(v) for v in (left, right)) or not previous <= left < right <= seconds+1e-6:
                raise ValueError('Ordered nonoverlapping work windows within the session required')
            previous = right
        for spans, upper in ((self.hrr, 1.), (self.speed_mps, float('inf'))):
            previous = 0.
            for s in spans:
                if not previous <= s.start < s.end <= seconds+1e-6 or not 0 <= min(s.first, s.last) <= max(s.first, s.last) <= upper:
                    raise ValueError('Valid ordered nonoverlapping signal spans required')
                previous = s.end
        if self.easy_speed_mps is not None:
            if self.reference_at is None or not self.reference_source or self.easy_speed_mps.low <= 0:
                raise ValueError('Positive pace normalization requires dated reference provenance')
            aware(self.reference_at)
            if self.reference_at > min(self.start, self.available_at):
                raise ValueError('Freeze athlete reference before issuing/completing this run')


def _pieces(spans, left, right):
    for s in spans:
        lo, hi = max(left, s.start), min(right, s.end)
        if hi > lo:
            at = lambda t: s.first+(s.last-s.first)*(t-s.start)/(s.end-s.start)
            yield hi-lo, at(lo), at(hi)


def _integrated_suitability(first, last, target):
    cuts = [0., 1.]
    if first != last:
        cuts += [(x-first)/(last-first) for x in target.band if 0 < (x-first)/(last-first) < 1]
    cuts.sort()
    return sum((b-a)*(target.suitability(first+(last-first)*a)+
                       target.suitability(first+(last-first)*b))/2 for a, b in zip(cuts, cuts[1:]))


def _shared_reference_bounds(pieces, reference, target):
    """Exact extrema over ONE reference for all supplied linear speed spans.

    In reciprocal-reference coordinates x=1/ref, suitability is a sum of two
    (capped) or four (band) hinges. Integrating each gives A*x+B+C/x
    between endpoint crossings. Sweep crossings and any stationary points; no
    reference grid, independent per-record normalizers, or optimizer RNG.
    O(N log N) time/O(N) space. Floating-point arithmetic, not interval numerics.
    """
    pieces = tuple(pieces)
    if reference.low == reference.high:
        total = fsum(dt*_integrated_suitability(a/reference.low, b/reference.low, target)
                     for dt, a, b in pieces)
        return total, total
    a, b = target.band[:2]
    hinges = ((a, 1/(b-a)), (b, -1/(b-a)))
    if len(target.band) == 4:
        c, d = target.band[2:]
        hinges += ((c, -1/(d-c)), (d, 1/(d-c)))
    slope_bound = max(abs(weight) for _, weight in hinges)
    events = {}
    rounding_enclosure = 0.
    def add(x, coeffs):
        events.setdefault(x, []).append(coeffs)
    for seconds, first, last in pieces:
        low, high = sorted((first, last))
        if high == 0:
            continue
        if 0 < high-low <= 1e-6*max(1., high):
            # Avoid cancellation of enormous opposing hinge coefficients for
            # almost-flat ramps. A Lipschitz enclosure bounds the midpoint
            # approximation; it cannot falsely narrow the returned range.
            rounding_enclosure += seconds*(high-low)/(2*reference.low)*slope_bound
            low = high = (low+high)/2
        for threshold, weight in hinges:
            scale = seconds*weight
            if threshold == 0:
                add(0., (scale*(low+high)/2, 0., 0.))
            elif low == high:
                add(threshold/high, (scale*high, -scale*threshold, 0.))
            else:
                delta = high-low
                partial = (scale*high*high/(2*delta), -scale*high*threshold/delta,
                           scale*threshold*threshold/(2*delta))
                add(threshold/high, partial)
                if low > 0:
                    add(threshold/low, (scale*(low+high)/2-partial[0],
                                       -scale*threshold-partial[1], -partial[2]))
    left, right = 1/reference.high, 1/reference.low
    coefficients = [[], [], []]
    interior = []
    for x in sorted(events):
        change = tuple(fsum(row[i] for row in events[x]) for i in range(3))
        if x <= left:
            for i, value in enumerate(change):
                coefficients[i].append(value)
        elif x < right:
            interior.append((x, change))
    A, B, C = (fsum(v) for v in coefficients)
    values = []
    for end, change in (*interior, (right, (0., 0., 0.))):
        points = [left, end]
        if A != 0 and C/A > 0:
            stationary = sqrt(C/A)
            if left < stationary < end:
                points.append(stationary)
        values.extend(fsum((A*x, B, C/x)) for x in points)
        A, B, C = A+change[0], B+change[1], C+change[2]
        left = end
    total_seconds = fsum(dt for dt, _, _ in pieces)
    return (max(0., min(total_seconds, min(values)-rounding_enclosure)),
            max(0., min(total_seconds, max(values)+rounding_enclosure)))


def assess_work_stimulus(evidence, target):
    per_bout = []
    session_speed_pieces = []
    for left, right in evidence.windows:
        known = lower = upper = 0.
        if target.channel in {'hrr', 'speed_mps'}:
            spans = evidence.hrr if target.channel == 'hrr' else evidence.speed_mps
            for seconds, a, b in _pieces(spans, left, right):
                amount = seconds*_integrated_suitability(a, b, target)
                lower += amount; upper += amount; known += seconds
        elif evidence.easy_speed_mps is not None:
            pieces = tuple(_pieces(evidence.speed_mps, left, right))
            session_speed_pieces.extend(pieces)
            known = fsum(seconds for seconds, _, _ in pieces)
            lower, upper = _shared_reference_bounds(pieces, evidence.easy_speed_mps, target)
        unknown = max(0., right-left-known)
        per_bout.append(dict(work_seconds=right-left, known_seconds=known, unknown_seconds=unknown,
            supported_minutes=Quantity(lower/60, (upper+unknown)/60, basis='work_only_target_suitability')))
    minutes = Quantity(sum(b['supported_minutes'].low for b in per_bout),
                       sum(b['supported_minutes'].high for b in per_bout), basis='work_only_target_suitability')
    if target.channel == 'speed_over_easy' and evidence.easy_speed_mps is not None:
        # Marginal bout extrema are NOT additive: each may occur at a different
        # reference. Recompute total across all bouts using a single parameter.
        low, high = _shared_reference_bounds(session_speed_pieces, evidence.easy_speed_mps, target)
        minutes = Quantity(low/60, (high+sum(b['unknown_seconds'] for b in per_bout))/60,
                           basis='work_only_target_suitability')
    return dict(target=target, supported_minutes=minutes,
        coverage=Quantity(min(1., minutes.low/target.minutes), min(1., minutes.high/target.minutes),
                          basis='target_specific_exposure_not_adaptation'),
        per_bout=per_bout, work_seconds=sum(b['work_seconds'] for b in per_bout),
        unknown_seconds=sum(b['unknown_seconds'] for b in per_bout),
        reference_scope='single_shared_session_reference', per_bout_bounds='marginal_not_additive',
        context_adjusted=False, physiological_equivalence_established=False)
