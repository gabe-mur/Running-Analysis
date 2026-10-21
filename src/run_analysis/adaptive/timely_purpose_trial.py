"""Experimental timing utility, not physiological fatigue or a due-date rule."""

from dataclasses import dataclass
from datetime import timedelta
from math import exp, expm1, isfinite, log

from run_analysis.adaptive.program_objective_trial import PURPOSES, credits


@dataclass(frozen=True)
class PurposeTimingPolicy:
    memory_scale: float
    purposes: tuple[str, ...] = ('long', 'quality')
    grace_fraction: float = 0.  # Legacy comparator; new stage caller explicitly uses 1.

    def __post_init__(self):
        object.__setattr__(self, 'purposes', tuple(self.purposes))
        if not isfinite(self.memory_scale) or self.memory_scale <= 0:
            raise ValueError('Positive finite explicit timing sensitivity scale required')
        if not isfinite(self.grace_fraction) or self.grace_fraction<0:
            raise ValueError('Finite nonnegative purpose-coverage plateau required')
        if (not self.purposes or len(set(self.purposes)) != len(self.purposes)
                or any(p not in {'long', 'quality'} for p in self.purposes)):
            raise ValueError('Select distinct long/quality purposes explicitly')


class UnknownPurposeHistory(ValueError):
    """The trial cannot initialize a temporal utility from unknown completion."""


def interval_loss(coverage, days, memory_days):
    """Exact integral of (1 - coverage * exp(-t/T))**2; no daily sampling bias."""
    if (not all(isfinite(v) for v in (coverage, days, memory_days))
            or not 0 <= coverage <= 1 or days < 0 or memory_days <= 0):
        raise ValueError('Bounded coverage, nonnegative duration, positive memory required')
    return max(0., days + 2 * coverage * memory_days * expm1(-days / memory_days)
               - coverage**2 * memory_days / 2 * expm1(-2 * days / memory_days))


CADENCE_CONTRACTS = {'amplitude_v1', 'proportional_grace_v1'}


def cadence_parameters(credit,grace,contract):
    """Coaching recency, not delivered work or a physiological dose response.

    Partial verified work buys proportionally fewer covered days, rather than
    making a fresh session deficient immediately. Zero/missing remains absent.
    The historical amplitude contract is retained for archived decisions.
    """
    if contract not in CADENCE_CONTRACTS:
        raise ValueError('Explicit supported cadence credit contract required')
    return (float(credit>0),credit*grace) if contract=='proportional_grace_v1' else (credit,grace)


def coverage_at(credit,age,grace,memory,*,credit_contract='amplitude_v1'):
    amplitude,event_grace=cadence_parameters(credit,grace,credit_contract)
    return amplitude*exp(-max(0.,age-event_grace)/memory)


def envelope_day(events,day,*,grace,memory,initial=0.,initial_day=None,
                 credit_contract='amplitude_v1'):
    """Exact one-day loss of a nonbankable plateau-then-decay max envelope.

    Separate constant and decaying contributions between plateau endpoints.
    Their maximum crosses at most once inside each segment. No sampling or
    recommendation-time sensitivity, and no hard exclusion after the cadence.
    Initial uncertainty decays immediately; it is not a fabricated fresh workout.
    """
    known=[((day-date).days,*cadence_parameters(credit,grace,credit_contract))
        for date,credit in events if date<=day and credit>0]
    cuts=sorted({0.,1.,*(g-age for age,_,g in known if 0<g-age<1)})
    initial=initial*exp(-(day-initial_day).days/memory) if initial_day is not None else 0.
    def value(t):
        return max([initial*exp(-t/memory),*(c*exp(-max(0.,age+t-g)/memory) for age,c,g in known)])
    loss=0.
    for a,b in zip(cuts,cuts[1:]):
        mid=(a+b)/2
        constant=max((c for age,c,g in known if age+mid<=g),default=0.)
        decaying=max([initial*exp(-a/memory),*(c*exp(-max(0.,age+a-g)/memory)
            for age,c,g in known if age+mid>g)])
        span=b-a
        if decaying<=constant:
            loss+=span*(1-constant)**2
        elif constant==0:
            loss+=interval_loss(decaying,span,memory)
        else:
            crossing=min(span,memory*log(decaying/constant))
            loss+=interval_loss(decaying,crossing,memory)+(span-crossing)*(1-constant)**2
    return dict(loss=max(0.,loss),opening=value(0.),closing=value(1.))


def timing_components(history, program, *, stage_start, decision, end, policy, timing,
                      opening_coverage=(), quality_cadence_contract='amplitude_v1'):
    """Caller validates unique events and observed/planned interval ownership.

    The max-envelope cannot bank clustered/partial completions. A role label is
    not enough for quality: its explicitly supplied delivered credit is required.
    The long fraction is still the existing trial's distance/reference mapping;
    this function does not establish that mapping as a sufficient endurance dose.
    """
    span = (end - stage_start).days
    if not stage_start <= decision < end:
        raise ValueError('Ordered fixed stage and future scope required')
    result = {}
    for purpose in timing.purposes:
        contract=quality_cadence_contract if purpose=='quality' else 'amplitude_v1'
        index = PURPOSES.index(purpose)
        rate = getattr(policy, f'{purpose}_per_7')
        memory = 7 / rate * timing.memory_scale
        if timing.grace_fraction:
            grace=7/rate*timing.grace_fraction
            events=sorted((r.day,float(credits(r,policy)[index])) for r in (*history,*program)
                          if float(credits(r,policy)[index])>0)
            initial=dict(opening_coverage).get(purpose,0.)
            daily=[]
            frontier=[]
            cursor=0
            for offset in range((end-decision).days):
                day=decision+timedelta(days=offset)
                delivered=0.
                while cursor<len(events) and events[cursor][0]<=day:
                    event=events[cursor]
                    # A later event with >= credit dominates an older event
                    # for every future age under the SAME plateau/decay map.
                    # Never remove an earlier event before the new event arrives.
                    frontier=[old for old in frontier if old[1]>event[1]]
                    frontier.append(event)
                    if event[0]==day:
                        delivered=max(delivered,event[1])
                    cursor+=1
                measured=envelope_day(frontier,day,grace=grace,memory=memory,initial=initial,
                    initial_day=decision,credit_contract=contract)
                daily.append(dict(day=day,opening_coverage=measured['opening'],delivered_credit=delivered,
                    after_session_coverage=measured['opening'],loss=measured['loss']))
            loss=sum(r['loss'] for r in daily)/span
            result[purpose]=dict(loss=loss,weighted_loss=loss*policy.purpose_weights[index],memory_days=memory,
                grace_days=grace,opening_coverage=daily[0]['opening_coverage'],closing_coverage=measured['closing'],
                daily=daily,coverage_model=('nonbankable_proportional_grace_then_decay'
                    if contract=='proportional_grace_v1' else 'nonbankable_cadence_plateau_then_decay'))
            continue
        preceding = [(row, float(credits(row, policy)[index])) for row in history if row.day < decision]
        preceding = [(row, credit) for row, credit in preceding if credit > 0]
        supplied = dict(opening_coverage)
        if purpose in supplied:
            coverage = supplied[purpose]
            if not isfinite(coverage) or not 0 <= coverage <= 1:
                raise ValueError('Explicit opening coverage must be bounded')
        elif not preceding:
            raise UnknownPurposeHistory(f'No known positive {purpose} purpose credit before decision')
        else:
            coverage = max(credit * exp(-(decision-row.day).days / memory) for row, credit in preceding)
        opening = coverage
        future = {row.day: float(credits(row, policy)[index]) for row in program}
        daily = []
        for offset in range((end - decision).days):
            day = decision + timedelta(days=offset)
            credit = future.get(day, 0.)
            before = coverage
            coverage = max(coverage, credit)
            loss = interval_loss(coverage, 1., memory)
            daily.append(dict(day=day, opening_coverage=before, delivered_credit=credit,
                              after_session_coverage=coverage, loss=loss))
            coverage *= exp(-1 / memory)
        loss = sum(row['loss'] for row in daily) / span
        result[purpose] = dict(loss=loss, weighted_loss=loss * policy.purpose_weights[index],
            memory_days=memory, opening_coverage=opening, closing_coverage=coverage,
            daily=daily)
    return result
