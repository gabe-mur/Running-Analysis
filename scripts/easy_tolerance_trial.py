"""Offline, assumption-labelled easy-dose permission. Not program approval."""
from dataclasses import dataclass
from datetime import timedelta
from math import isfinite

from run_analysis.athlete_projection import aware
from run_analysis.easy_dose import AerobicMinimum, EasyDurationReference, assess_easy_dose
from run_analysis.workload_accounting import Quantity

KEY_PURPOSES = {'long', 'quality', 'threshold', 'high_aerobic', 'speed_skill', 'race_specific'}


@dataclass(frozen=True)
class PriorExposureReference:
    purpose: str
    ceilings: tuple[tuple[str, float], ...]
    source_keys: tuple[str, ...]

    def __post_init__(self):
        if (not self.purpose or not self.source_keys or not self.ceilings
                or len(dict(self.ceilings)) != len(self.ceilings)
                or any(not isfinite(v) or v < 0 for _, v in self.ceilings)):
            raise ValueError('Frozen descriptive exposure references required')


@dataclass(frozen=True)
class EasyTolerancePolicy:
    reference: EasyDurationReference
    minimum: AerobicMinimum
    ordinary_miles: float
    reduced_fraction: float
    prior_references: tuple[PriorExposureReference, ...]
    source: str

    def __post_init__(self):
        if (not self.source or not isfinite(self.ordinary_miles) or self.ordinary_miles <= 0
                or abs(self.ordinary_miles*10-round(self.ordinary_miles*10)) > 1e-8
                or not isfinite(self.reduced_fraction) or not 0 < self.reduced_fraction < 1
                or len({r.purpose for r in self.prior_references}) != len(self.prior_references)):
            raise ValueError('Explicit frozen ordinary command and reduction scenario required')


def quantity(event, channel):
    terms = [t for t in event.terms if t.channel == channel and t.coefficient]
    if any(t.bounds is None for t in terms):
        return None
    return Quantity(sum(t.coefficient*t.bounds.low for t in terms),
                    sum(t.coefficient*t.bounds.high for t in terms), basis='unchanged_event_accounting')


def prior_exposure(event, policy):
    role = 'quality' if event.purpose in KEY_PURPOSES-{'long'} else event.purpose
    ref = next((r for r in policy.prior_references if r.purpose == role), None)
    if ref is None:
        return dict(status='unknown', channels={}, source_keys=())
    channels = {}
    for channel, ceiling in ref.ceilings:
        q = quantity(event, channel)
        channels[channel] = ('unknown' if q is None else
            'above_reference' if q.low > ceiling+1e-8 else
            'within_reference' if q.high <= ceiling+1e-8 else 'overlaps_reference')
    status = ('above_reference' if 'above_reference' in channels.values() else
              'within_reference' if set(channels.values()) == {'within_reference'} else 'uncertain')
    return dict(status=status, channels=channels, source_keys=ref.source_keys,
                interpretation='descriptive_separate_channel_maxima_not_demonstrated_tolerance')


def assess(event, preceding, policy, *, as_of, context=None, unexpected_previous_response=False):
    """Return permission to consider an easy dose, never physiological clearance.

    No ledger is modified, no future event expands frozen reference ceilings, and
    no negative load / recovery bonus is supplied. Caller retains sequence review.
    Dates here are already-local dates; the frozen panel uses UTC report dates.
    """
    aware(as_of)
    if policy.reference.issued_at > as_of:
        raise ValueError('Future-issued reference unavailable')
    preceding = tuple(preceding)
    if (event.source != 'planned' or event.day < as_of.date()
            or len({e.key for e in (*preceding, event)}) != len(preceding)+1
            or any(e.day >= event.day or e.accounting_key != event.accounting_key for e in preceding)
            or len({e.day for e in preceding}) != len(preceding)):
        raise ValueError('Reconciled prior ledger, unique dates, matching units and future command required')
    previous = max(preceding, key=lambda e: e.day) if preceding else None
    known = context if context and context.available_at <= as_of else None
    if known and known.applies_on > as_of.date():
        raise ValueError('A current report cannot assert future readiness')
    state = known.state if known and known.applies_on == event.day else 'unknown'
    if known and known.state in {'health_concern', 'conflicting'}:
        state = known.state
    adjacent_key = bool(previous and event.day-previous.day == timedelta(days=1)
                        and previous.purpose in KEY_PURPOSES)
    evidence = prior_exposure(previous, policy) if previous else dict(status='unknown', channels={})
    result = dict(key=event.key, day=event.day, state=state, adjacent_key=adjacent_key,
        previous_key=previous.key if previous else None, prior_exposure=evidence,
        previous_is_conditional=bool(previous and previous.source == 'planned'),
        policy_source=policy.source, reduced_fraction=policy.reduced_fraction,
        program_approved=False, recovery_cleared=False, planner_influence=False,
        sequence_review_required=True)

    def finish(status):
        return dict(result, status=status)

    if event.purpose != 'easy':
        return finish('purpose_not_covered')
    if state in {'health_concern', 'conflicting'}:
        return finish('review_current_context')
    if known and known.state == 'little_tired' and event.day > as_of.date():
        return finish('request_session_date_context')
    if state == 'unknown' and (unexpected_previous_response or known and known.state == 'little_tired'):
        return finish('request_current_context')
    work = quantity(event, 'work_minutes')
    if work is None or work.high > 1e-8 or event.structure:
        return finish('easy_intensity_not_established')
    miles = quantity(event, 'miles')
    if miles is None or miles.low != miles.high:
        raise ValueError('Committed planned distance required')
    dose = assess_easy_dose(miles.low, policy.reference, policy.minimum)
    result['dose'] = dose
    if not dose['passes_planned_minimum']:
        return finish('below_aerobic_minimum')
    if evidence['status'] == 'above_reference':
        return finish('review_prior_dose')
    reduced = adjacent_key or state == 'little_tired'
    cap_minutes = policy.ordinary_miles*policy.reference.minutes_per_mile*(
        policy.reduced_fraction if reduced else 1.)
    result.update(reduced_required=reduced, maximum_minutes_at_reference_pace=cap_minutes,
        dose_fraction_of_ordinary=miles.low/policy.ordinary_miles,
        current_response_unobserved=state == 'unknown')
    return finish('provisional_dose_option' if dose['total_minutes_at_reference_pace'] <= cap_minutes+1e-8
                  else 'outside_local_dose_policy')
