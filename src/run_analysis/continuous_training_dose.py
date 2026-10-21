"""Continuous HR demand and generic quality exposure, not measured adaptation.

No new workout classifier. The HR integral is label-independent. External work
is retained separately because short repetitions can finish before HR responds.
Reference values are explicit sensitivity assumptions, not fitted physiology.
"""
from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from math import isfinite

from .response_model import Policy, rate
from .signal_structure import above_threshold
from .training_signals import _mean_rate, integrate_cardio_trace
from .workload_accounting import Quantity
from .workload_profile import bout_summary


@dataclass(frozen=True)
class ContinuousDoseReference:
    source: str
    intensity_exponent: float = 1.92
    easy_hrr: float = .60
    quality_hrr: float = .85
    quality_minutes: float = 15.
    quality_mapping: str = field(default='supported_work_coverage_v2', init=False)

    def __post_init__(self):
        values = (self.intensity_exponent, self.easy_hrr, self.quality_hrr, self.quality_minutes)
        if (not self.source.strip() or any(isinstance(v, bool) or not isfinite(v) for v in values)
                or self.intensity_exponent < 0 or not 0 < self.easy_hrr < self.quality_hrr <= 1
                or self.quality_minutes <= 0):
            raise ValueError('Explicit ordered HRR references and positive reference duration required')

    @property
    def policy(self):
        # Only the raw integral is used. No density multiplier or decay policy.
        return Policy(intensity_exponent=self.intensity_exponent, reserve_reference=self.easy_hrr)

    @property
    def accounting_key(self):
        mapping = dict(exponent=self.intensity_exponent, reference=self.easy_hrr)
        return 'continuous_hrr_minutes_v1:' + sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest()


def continuous_dose(spans, elapsed_seconds, reference):
    """Shared actual/forecast mapping of supplied linear HRR traces.

    Quality exposure integrates max(g(HRR)-g(easy),0), normalized to the
    excess at quality_hrr. Splitting at the crossing avoids a bucket jump and
    integrates the hinge accurately. This is a descriptor, not a threshold
    estimate, VO2max stimulus measurement, or established equivalence of reps.
    """
    spans = tuple(spans)
    policy = reference.policy
    result = integrate_cardio_trace(spans, elapsed_seconds, policy, compute_density=False)
    excess = 0.
    for s in spans:
        a, b, seconds = s.first, s.last, s.end-s.start
        if max(a, b) <= reference.easy_hrr:
            continue
        if min(a, b) < reference.easy_hrr:
            seconds *= (max(a, b)-reference.easy_hrr)/abs(b-a)
            a, b = reference.easy_hrr, max(a, b)
        excess += seconds/60 * max(0., _mean_rate(a, b, policy)-1.)
    denominator = rate(reference.quality_hrr, policy)-1.
    bounds = result['load_bounds_within_reference']
    return dict(
        cardio=Quantity(bounds['low'], bounds['high'], basis='continuous_hrr_integral'),
        high_effort_minutes=Quantity(excess/denominator,
            (excess+result['unknown_seconds']/60*(rate(1., policy)-1.))/denominator,
            basis='continuous_excess_hr_exposure_not_proven_adaptation'),
        coverage=result['coverage'], unknown_seconds=result['unknown_seconds'],
        sustained_exposure=above_threshold(spans, reference.quality_hrr),
        mean_hr_shortcut_load=result['mean_hr_shortcut_load'],
        accounting_key=reference.accounting_key, reference=asdict(reference))


def supported_work_minutes(bouts, declared_minutes, *, evidence_basis, source, elapsed_seconds):
    """Validate the measured/projected work extent, not its physiological purpose.

    Return None for missing work evidence. It is neither zero detected work nor
    permission to substitute all session minutes. Structure still depends on the
    existing detector; pace contrast alone is not absolute intensity evidence.
    """
    observed = {'recorded_lap_contrast', 'recorded_continuous_lap', 'pace_inferred', 'explicit_segments'}
    if not isfinite(elapsed_seconds) or elapsed_seconds <= 0:
        raise ValueError('Positive finite session duration required')
    if source not in {'observed', 'planned'}:
        raise ValueError('Observed or planned work provenance required')
    allowed = observed if source == 'observed' else {'prescribed'}
    if evidence_basis == 'unknown':
        if bouts:
            raise ValueError('Work segments require explicit evidence provenance')
        return None
    if evidence_basis not in allowed | {'no_supported_work'}:
        raise ValueError('Work evidence must match observed/planned provenance')
    if evidence_basis == 'no_supported_work' and bouts:
        raise ValueError('No-work evidence cannot contain work segments')
    bout_summary(bouts)
    ranges = [b.duration_range_seconds or (b.work_seconds, b.work_seconds) for b in bouts]
    for b, bounds in zip(bouts, ranges):
        if (len(bounds) != 2 or any(not isfinite(v) for v in bounds)
                or not 0 < bounds[0] <= b.work_seconds <= bounds[1]):
            raise ValueError('Ordered positive work-duration bounds required')
    result = Quantity(sum(v[0] for v in ranges)/60, sum(v[1] for v in ranges)/60,
                      basis=evidence_basis)
    if result.high*60+sum(b.recovery_after_seconds or 0 for b in bouts[:-1]) > elapsed_seconds+1e-6:
        raise ValueError('Work and recovery exceed elapsed duration')
    if any(abs(a-b) > 1e-6 for a, b in zip((result.low, result.high),
                                           (declared_minutes.low, declared_minutes.high))):
        raise ValueError('Work demand and supported segments must agree')
    return result


def general_quality_credit(work_minutes, *, role, reference):
    """Coverage of supported work, not a whole-session HR benefit estimate.

    HR load remains separate: no amount of warm-up, recovery, heat or drift
    can supply missing work credit. Repetitions or an observed continuous work
    phase supply extent without requiring an exact template or repeat history.
    This intentionally does not equate their specific adaptations.
    """
    if role not in {'easy', 'long', 'quality', 'unknown'}:
        raise ValueError('Resolved session context required')
    if role in {'easy', 'long'}:
        return Quantity.point(0., 'no_quality_context_hr_demand_retained')
    if work_minutes is None:
        return Quantity(0., 1., basis='unresolved_work_extent_not_hr_substitution')
    lower = work_minutes.low/reference.quality_minutes
    upper = work_minutes.high/reference.quality_minutes
    return Quantity(min(1., lower) if role == 'quality' else 0., min(1., upper),
                    basis='supported_work_coverage_not_specific_stimulus')
