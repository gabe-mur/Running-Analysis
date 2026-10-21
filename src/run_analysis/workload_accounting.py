"""Common observed/projected workload units, with explicit uncertainty.

Shadow only. Bounds are accounting assumptions, not confidence intervals or
physiological safety limits. Recorded values are never rewritten for compliance.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .training_load import ZONE_WEIGHTS


@dataclass(frozen=True)
class Quantity:
    low: float
    high: float
    estimate: float | None = None
    basis: str = 'unspecified'

    def __post_init__(self):
        if not all(isfinite(x) for x in (self.low, self.high)) or not 0 <= self.low <= self.high:
            raise ValueError('Quantity bounds must be finite, nonnegative and ordered')
        if self.estimate is not None and (
            not isfinite(self.estimate) or not self.low <= self.estimate <= self.high
        ):
            raise ValueError('Estimate must lie within quantity bounds')

    @classmethod
    def point(cls, value, basis):
        return cls(value, value, value, basis)

    def scale(self, factor):
        if not isfinite(factor) or factor < 0:
            raise ValueError('Scale must be finite and nonnegative')
        return Quantity(self.low*factor, self.high*factor,
                        self.estimate*factor if self.estimate is not None else None, self.basis)


@dataclass(frozen=True)
class Workload:
    source: str
    distance_meters: Quantity | None
    running_minutes: Quantity | None
    cardio_points: Quantity | None
    structure: dict
    limitations: tuple[str, ...] = ()

    def __post_init__(self):
        if self.source not in {'observed', 'projected'}:
            raise ValueError('Workload source must be observed or projected')


def continuous_zone_seconds(intervals, zones):
    """Integrate linearly between valid HR endpoints without integer-bin holes.

    No interpolation across flagged recording gaps or missing endpoints. This
    retains the moving-time convention; paused cardiovascular strain is not
    estimated. The legacy production zone calculation is deliberately untouched.
    """
    names = [f'z{i}' for i in range(1, 6)]
    bounds = [zones[name] for name in names]
    if any(not all(isfinite(v) for v in b) or b[0] > b[1] for b in bounds):
        raise ValueError('Invalid HR zone bounds')
    if any(a[1]+1 != b[0] for a, b in zip(bounds, bounds[1:])):
        raise ValueError('Expected contiguous integer HR zones')
    edges = [b[0] for b in bounds] + [bounds[-1][1]+1]
    bins = ['below_z1', *names, 'above_z5']
    totals = {name: 0. for name in (*bins, 'unknown')}
    for interval in intervals:
        seconds = interval.moving_time_s
        if seconds <= 0:
            continue
        a, b = interval.start.heart_rate_bpm, interval.end.heart_rate_bpm
        if ('large_recording_gap' in interval.flags
                or any(x is None or not isfinite(x) or x <= 0 for x in (a, b))):
            totals['unknown'] += seconds
            continue
        cuts = [0., 1.]
        if a != b:
            cuts += [(edge-a)/(b-a) for edge in edges if 0 < (edge-a)/(b-a) < 1]
        cuts.sort()
        for left, right in zip(cuts, cuts[1:]):
            hr = a+(b-a)*(left+right)/2
            name = bins[sum(hr >= edge for edge in edges)]
            totals[name] += seconds*(right-left)
    return totals


def observed_workload(profile):
    if profile['source'] != 'observed':
        raise ValueError('Expected an observed profile')
    exposure, cardio = profile['running_exposure'], profile['cardiovascular']
    zones = cardio['zone_seconds']
    known_points = sum(zones.get(name, 0.)*weight/60 for name, weight in ZONE_WEIGHTS.items())
    unknown_minutes = zones.get('unknown', 0.)/60
    distance, seconds = exposure['distance_meters'], exposure['moving_seconds']
    duration = None
    if seconds is not None:
        low, high = exposure.get('moving_seconds_range', (seconds, seconds))
        unknown_minutes += max(0., high-seconds)/60
        estimate = seconds/60 if exposure.get('movement_observed', True) else None
        if high > 0 or exposure.get('movement_observed', True):
            duration = Quantity(low/60, high/60, estimate, 'movement_classification_with_gap_bounds')
    limitations = tuple(profile['limitations'])
    if unknown_minutes:
        limitations += ('Unknown HR has a 0–5 points/minute bound; no average-HR filling',)
    return Workload('observed',
        Quantity.point(distance, 'recorded_distance') if distance is not None else None,
        duration,
        Quantity(known_points, known_points+unknown_minutes*5,
                 known_points if not unknown_minutes else None, 'integrated_hr_with_missing_bounds')
        if duration is not None else None,
        profile['intensity_structure'], limitations)


def projected_workload(profile, *, duration_minutes: Quantity,
                       base_rate: Quantity | None = None,
                       work_rate: Quantity | None = None,
                       recovery_rate: Quantity | None = None):
    """Project using explicit phase HR-rate assumptions, never a fixed pace.

    Rates are Edwards points/minute, supplied independently of workout labels.
    Work duration and total duration may vary together, so compute joint extrema
    rather than adding incompatible independent phase bounds. Between-work
    recovery may use its own rate: slower pace does not mean HR has fallen to
    the warm-up level. Without it, base_rate is an explicit fallback assumption.
    HR lag belongs in the
    caller's range, not in a fabricated measured trace.
    """
    if profile['source'] != 'prescribed':
        raise ValueError('Expected a prescribed profile')
    for rate in (base_rate, work_rate, recovery_rate):
        if rate is not None and rate.high > 5:
            raise ValueError('Edwards rates must be between 0 and 5')
    exposure, structure = profile['running_exposure'], profile['intensity_structure']
    work_low = work_high = 0.
    for bout in structure['bouts']:
        duration_range = bout.get('duration_range_seconds')
        low, high = duration_range or (bout['work_seconds'], bout['work_seconds'])
        work_low += low/60
        work_high += high/60
    recovery = structure['between_work_recovery_seconds']
    if recovery_rate is not None and recovery is None:
        raise ValueError('Separate recovery HR requires known recovery duration')
    minimum_recovery = recovery/60 if recovery is not None else 0.
    minimum_block = exposure.get('minimum_duration_range_seconds', (0, 0))[1]/60
    if duration_minutes.low+1e-9 < max(work_high+minimum_recovery, minimum_block):
        raise ValueError('Total moving duration cannot accommodate all prescribed work/recovery')
    distance_range = exposure['distance_range_miles']
    target = exposure['distance_target_miles']
    distance = (Quantity(distance_range[0]*1609.344, distance_range[1]*1609.344,
                         target*1609.344 if target is not None else None,
                         'issued_distance_and_adherence_band') if distance_range else None)
    limitations = ['Phase HR rates are assumptions, not measured or calibrated forecasts',
                   'No terrain, novelty or short-fast neuromuscular recovery assessment']
    if 'work_duration_missing' in profile['limitations']:
        work_low, work_high = 0., duration_minutes.high
        limitations.append('Unresolved work duration spans the session')
    base = base_rate or Quantity(0, 5, basis='unknown_base_rate')
    work = work_rate or Quantity(0, 5, basis='unknown_work_rate')
    recovery_hr = recovery_rate or base
    if recovery_rate is None:
        limitations.append('Recovery HR uses the base-rate assumption, not measured recovery')
    values = [w*r+minimum_recovery*rr+(t-w-minimum_recovery)*b
              for t in (duration_minutes.low, duration_minutes.high)
              for w in (work_low, min(work_high, t-minimum_recovery))
              for r in (work.low, work.high) for b in (base.low, base.high)
              for rr in (recovery_hr.low, recovery_hr.high)]
    # A central cardio estimate would require joint duration/intensity estimates;
    # do not substitute the middle of a wide unknown interval for evidence.
    cardio = Quantity(min(values), max(values), basis='explicit_phase_rate_bounds')
    return Workload('projected', distance, duration_minutes, cardio, structure, tuple(limitations))


def compare_execution(expected: Workload, actual: Workload):
    """Compare each dimension; do not let distance compliance erase harder HR.

    Numerical containment only. No new materiality threshold or scheduling lock.
    A broad missing-data bound overlapping the plan is not confirmed compliance.
    """
    if expected.source != 'projected' or actual.source != 'observed':
        raise ValueError('Compare projected to observed work')
    result = {}
    for name in ('distance_meters', 'running_minutes', 'cardio_points'):
        plan, done = getattr(expected, name), getattr(actual, name)
        if plan is None or done is None:
            status = 'unknown'
        elif done.high < plan.low-1e-8:
            status = 'below'
        elif done.low > plan.high+1e-8:
            status = 'above'
        elif done.low >= plan.low-1e-8 and done.high <= plan.high+1e-8:
            status = 'within'
        else:
            status = 'overlapping_uncertainty'
        result[name] = status
    return result
