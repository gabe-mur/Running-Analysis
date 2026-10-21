"""Offline fixed-stage decision score. No physiological clearance or live imports."""
from dataclasses import dataclass
from datetime import date, timedelta
from functools import cached_property,lru_cache
from math import exp, isfinite

import numpy as np

from run_analysis.workload_accounting import Quantity
from .demand_interaction import session_cost
from .identity_json import FrozenIdentity

AXES = ('running_minutes', 'cardio', 'work_minutes')
PURPOSES = ('volume', 'ordinary_easy', 'long', 'quality')


def demand_value(quantity, fraction):
    """Nominal scenario is an explicit estimate, not an asymmetric-box midpoint.

    Missing estimates keep the existing bounds-only interpolation. Low/high are
    sensitivity endpoints, not probabilities or a worst-case admission rule.
    """
    nominal = quantity.estimate
    if nominal is None:
        return quantity.low+(quantity.high-quantity.low)*fraction
    if fraction <= .5:
        return quantity.low+(nominal-quantity.low)*fraction*2
    return nominal+(quantity.high-nominal)*(fraction-.5)*2


@dataclass(frozen=True)
class TrialSession(FrozenIdentity):
    key: str
    day: date
    miles: float
    role: str
    demand: tuple[Quantity, Quantity, Quantity]
    quality_credit: float = 0.
    source: str = 'planned'

    @cached_property
    def _continuation_terms(self):
        # Nonfield, object-owned arithmetic reuse. Normalized records are
        # frozen; custom mutable diagnostic demands must never be memoized.
        return {} if self._identity_fragments is not None else None

    def __post_init__(self):
        if (not self.key or type(self.day) is not date or self.role not in {'easy','long','quality','unknown','race'}
                or not isfinite(self.miles) or self.miles <= 0 or len(self.demand) != 3
                or any(not isinstance(q, Quantity) for q in self.demand)
                or not isfinite(self.quality_credit) or not 0 <= self.quality_credit <= 1
                or self.source not in {'observed','planned'}):
            raise ValueError('Explicit finite physical dose, demand and purpose required')


@dataclass(frozen=True)
class TrialPolicy:
    weekly_miles: float = 18.5
    ordinary_miles: float = 4.4
    long_miles: float = 8.4
    ordinary_per_7: float = 1.
    long_per_7: float = 1.
    quality_per_7: float = 1.
    ordinary_power: float = 2.
    purpose_weights: tuple[float, ...] = (1.,1.,1.,1.)
    quality_work_minutes: float | None = None  # Stage purpose dose, not demand normalization.
    volume_window_days: int | None = None  # Explicit rolling-accounting experiment, not enabled implicitly.

    def __post_init__(self):
        values = (self.weekly_miles,self.ordinary_miles,self.long_miles,self.ordinary_per_7,
                  self.long_per_7,self.quality_per_7,self.ordinary_power)
        # Zero supporting-volume weight is an explicit single-volume experiment.
        # All other purposes retain positive weights; historical defaults unchanged.
        if (len(self.purpose_weights) != 4 or any(not isfinite(x) or x <= 0 for x in values)
                or any(not isfinite(x) or x < 0 or i != 1 and x == 0
                       for i,x in enumerate(self.purpose_weights))):
            raise ValueError('Positive frozen policy scales required')
        if self.quality_work_minutes is not None and (not isfinite(self.quality_work_minutes) or self.quality_work_minutes <= 0):
            raise ValueError('Positive declared quality-purpose dose required')
        if self.volume_window_days is not None and (type(self.volume_window_days) is not int or self.volume_window_days<1):
            raise ValueError('Positive explicit volume accounting window required')


@dataclass(frozen=True)
class DemandScenario:
    memory_days: float = 2.  # Exponential e-folding, NOT a recovery deadline.
    bound_fraction: float = .5
    timing_compression_days: float = 0.
    axis_weights: tuple[float, ...] = (1.,1.,1.)

    def __post_init__(self):
        if (not isfinite(self.memory_days) or self.memory_days <= 0
                or not 0 <= self.bound_fraction <= 1 or not 0 <= self.timing_compression_days < 1
                or len(self.axis_weights) != 3
                or any(not isfinite(x) or x < 0 for x in self.axis_weights)
                or sum(self.axis_weights) <= 0):
            raise ValueError('Explicit finite demand scenario required')


@dataclass(frozen=True)
class EasyDurationPolicy:
    """Explicit offline alternative to preferred-size/count credit, not physiology."""
    minimum_minutes: float
    penalty_free_minutes: float
    maximum_penalty: float
    source: str
    supporting_easy_miles_per_7: float | None = None

    def __post_init__(self):
        if (any(not isfinite(v) for v in (self.minimum_minutes, self.penalty_free_minutes,
                                          self.maximum_penalty))
                or not 0 < self.minimum_minutes < self.penalty_free_minutes
                or self.maximum_penalty < 0 or not self.source.strip()):
            raise ValueError('Ordered explicit easy-duration bounds and finite nonnegative penalty required')
        if self.supporting_easy_miles_per_7 is not None and (
                not isfinite(self.supporting_easy_miles_per_7) or self.supporting_easy_miles_per_7 <= 0):
            raise ValueError('Explicit positive supporting-easy volume target required')

    def penalty(self, minutes):
        if not isfinite(minutes) or minutes < 0:
            raise ValueError('Finite nonnegative duration required')
        fraction = min(1., max(0., (self.penalty_free_minutes-minutes) /
                              (self.penalty_free_minutes-self.minimum_minutes)))
        return self.maximum_penalty*fraction**2


def easy_duration_components(history, program, *, stage_start, decision, delivery_end,
                             policy, bound_fraction):
    """Per-seven-day finite cost; eligibility separate from score and actual credit.

    Completed stage work retains its cost so expected completion cannot change
    relative comparisons. Actual short runs never become eligibility violations.
    Uncertain forecast duration is reported, not silently passed at its midpoint.
    """
    relevant = tuple(e for e in (*history, *program)
                     if stage_start <= e.day < delivery_end and e.role == 'easy')
    penalty_low = sum(policy.penalty(e.demand[0].high) for e in relevant)
    penalty_high = sum(policy.penalty(e.demand[0].low) for e in relevant)
    factor = 7/(delivery_end-stage_start).days
    planned = tuple(e for e in relevant if e.source == 'planned' and e.day >= decision)
    funding = None
    if policy.supporting_easy_miles_per_7 is not None:
        target = policy.supporting_easy_miles_per_7/factor
        delivered = sum(e.miles for e in relevant)
        funding = dict(target_miles=target, delivered_miles=delivered,
            credited_fraction=min(1., delivered/target),
            shortfall=max(0., 1-delivered/target)**2,
            basis='aggregate supporting easy volume; not per-session size/count')
    nominal_cost = sum(policy.penalty(demand_value(e.demand[0],bound_fraction))
        if e.demand[0].estimate is not None else
        policy.penalty(e.demand[0].low)*(1-bound_fraction)+policy.penalty(e.demand[0].high)*bound_fraction
        for e in relevant)*factor
    return dict(cost=nominal_cost,
        supporting_easy_volume=funding,
        cost_bounds=(penalty_low*factor, penalty_high*factor),
        below_minimum_keys=tuple(e.key for e in planned if e.demand[0].high < policy.minimum_minutes),
        uncertain_minimum_keys=tuple(e.key for e in planned
            if e.demand[0].low < policy.minimum_minutes <= e.demand[0].high),
        source=policy.source, ordinary_size_credit_replaced=True,
        physiological_minimum_established=False)


def credits(session, policy):
    if session.role == 'race':
        return np.zeros(4)
    # Total running, not a false claim that all quality miles are aerobic/easy.
    return np.array((session.miles,
        min(session.miles/policy.ordinary_miles,1.)**policy.ordinary_power if session.role == 'easy' else 0.,
        min(session.miles/policy.long_miles,1.) if session.role == 'long' else 0.,
        session.quality_credit if session.role == 'quality' else 0.))


@lru_cache(maxsize=128)
def _rolling_frame(history, foregone, decision, end, window, smoothing_days,
                   weekly_miles, supporting_miles, volume_targets, support_targets, opening_before):
    """Candidate-independent calendar work; immutable, bounded and value keyed.

    Keep historical summands, rather than a rounded subtotal: Python's sum
    must still see the identical history-then-program sequence for each day.
    No candidate miles, demands, eligibility or losses are memoized here.
    """
    from run_analysis.program_delivery import _target_total
    result=[]
    for n in range(1-smoothing_days,(end-decision).days):
        day=decision+timedelta(days=n)
        start=day-timedelta(days=window-1)
        parameters=[]
        for name,rate,segments in (('volume',weekly_miles,volume_targets),
                                  ('ordinary_easy',supporting_miles,support_targets)):
            if rate is None:
                continue
            target=_target_total(segments,start,day+timedelta(days=1)) if segments else rate*window/7
            historical=tuple(miles for at,miles,role in history if start<=at<=day
                and role!='race' and (name=='volume' or role=='easy'))
            relief=sum(miles for at,miles,role in foregone if start<=at<=day
                and (name=='volume' or role=='easy'))
            opening_gap=0.
            if opening_before is not None:
                stop=min(opening_before,day+timedelta(days=1))
                if stop>start:
                    opening_target=_target_total(segments,start,stop) if segments else rate*(stop-start).days/7
                    opening_actual=sum(miles for at,miles,role in history if start<=at<stop
                        and role!='race' and (name=='volume' or role=='easy'))
                    opening_gap=opening_target-opening_actual
            parameters.append((name,target,historical,relief,opening_gap))
        result.append((n,day,start,tuple(parameters)))
    return tuple(result)


def rolling_volume_components(history,program,*,stage_start,decision,end,policy,supporting_miles,
                              volume_targets=(),support_targets=(),smoothing_days=1,foregone_volume=(),opening_before=None):
    """Additive long-window delivery loss; not a near-term quota or hard floor.

    A session affects only windows containing it. A remote terminal-day addition
    cannot grant credit for missing early volume. Raw distance stays unchanged.
    Explicit target segments prevent new progression goals being applied to the
    past. Without segments this is a fixed-stage experiment, disclosed below.
    """
    if type(smoothing_days) is not int or smoothing_days<1:
        raise ValueError('Positive explicit volume smoothing days required')
    window=policy.volume_window_days
    history,program=tuple(history),tuple(program)
    losses=dict(volume=0.,ordinary_easy=0.)
    daily=[]
    preceding=[]
    frame=_rolling_frame(tuple((r.day,r.miles,r.role) for r in history),
        tuple((r.day,r.miles,r.role) for r in foregone_volume),decision,end,window,smoothing_days,
        policy.weekly_miles,supporting_miles,tuple(volume_targets),tuple(support_targets),opening_before)
    for n,day,window_start,parameters in frame:
        active=[r for r in program if window_start<=r.day<=day]
        row=dict(day=day)
        for name,target,historical,relief,opening_gap in parameters:
            delivered=sum((*historical,*(r.miles for r in active if r.role!='race'
                and (name=='volume' or r.role=='easy'))))
            row[name]=dict(target=target,delivered=delivered,foregone=relief,opening_gap=opening_gap)
        preceding.append(row)
        if len(preceding)>smoothing_days:
            preceding.pop(0)
        if n<0:
            continue
        smoothed=dict(day=day)
        for name in losses:
            if name not in row:
                continue
            # Average signed physical delivery BEFORE clipping. Only trailing
            # windows participate: no distant future or permanent credit bank.
            target=sum(r[name]['target'] for r in preceding)/smoothing_days
            delivered=sum(r[name]['delivered'] for r in preceding)/smoothing_days
            relief=sum(r[name]['foregone'] for r in preceding)/smoothing_days
            relief+=max(0.,sum(r[name]['opening_gap'] for r in preceding)/smoothing_days)
            loss=max(0.,1-(delivered+relief)/target)**2
            losses[name]+=loss/(end-stage_start).days
            smoothed[name]=dict(target=target,delivered=delivered,loss=loss)
            if foregone_volume or opening_before is not None:smoothed[name]['foregone']=relief
            if smoothing_days>1:
                smoothed[name].update(raw_target=row[name]['target'],raw_delivered=row[name]['delivered'])
        daily.append(smoothed)
    return dict(window_days=window,losses=losses,daily=daily,
                smoothing_days=smoothing_days,
                past_stage_target_trajectory='dated_segments' if volume_targets and support_targets else 'not_supplied_fixed_stage_experiment')


def _validate(history, program, stage_start, decision, end, scales):
    if not stage_start <= decision < end or len(scales) != 3 or any(not isfinite(x) or x <= 0 for x in scales):
        raise ValueError('Fixed stage, future interval and positive reference scales required')
    all_rows = (*history,*program)
    if len({e.key for e in all_rows}) != len(all_rows) or len({e.day for e in all_rows}) != len(all_rows):
        raise ValueError('Unique reconciled identities and dates required')
    if any(e.day >= decision or e.source != 'observed' for e in history):
        raise ValueError('Only earlier observed history allowed')
    if any(not decision <= e.day < end or e.source != 'planned' for e in program):
        raise ValueError('Only remaining planned work in declared scope allowed')


def components(history, program, *, stage_start, decision, end, scales, policy, scenario,
               delivery_end=None, demand_interaction='independent_v1'):
    """Same fixed-stage benefit; only future-owned overlap cost, including tail.

    A completed projected prefix changes cost by its sunk contribution only.
    Earlier stage credit remains in the benefit ledger. No rolling stage reset.
    `end` covers demand; optional `delivery_end` freezes benefit/target scope and
    normalization while including supplied work after it in demand only.
    """
    history, program = tuple(history), tuple(program)
    _validate(history,program,stage_start,decision,end,scales)
    delivery_end = end if delivery_end is None else delivery_end
    if type(delivery_end) is not date or not decision < delivery_end <= end:
        raise ValueError('Delivery endpoint must follow the decision and not exceed demand coverage')
    span = (delivery_end-stage_start).days
    delivered = sum((credits(e,policy) for e in (*history,*program)
                     if stage_start <= e.day < delivery_end),np.zeros(4))
    targets = np.array((policy.weekly_miles,policy.ordinary_per_7,policy.long_per_7,policy.quality_per_7))*span/7
    shortfall = np.maximum(0.,1-delivered/targets)**2
    rows = sorted((*history,*program),key=lambda e:e.day)
    q = np.array([[demand_value(v,scenario.bound_fraction) for v in e.demand] for e in rows]).reshape((-1,3))
    q = q/np.array(scales)
    accumulated = np.zeros(3)
    sessions = []
    preceding = np.zeros(3)
    previous_day = None
    compression = exp(scenario.timing_compression_days/scenario.memory_days)
    for i,e in enumerate(rows):
        if previous_day is not None:
            preceding *= exp(-(e.day-previous_day).days/scenario.memory_days)
        if e.day >= decision:
            # Unique daily ownership means every preceding age is >=1 day,
            # while compression is <1. Factoring its shared multiplier is
            # exactly the old pair sum, without an O(n**2) history scan.
            cost = (.5*q[i]**2+q[i]*preceding*compression if demand_interaction=='independent_v1'
                else session_cost(q[i],preceding,compression,demand_interaction))
            accumulated += cost
            sessions.append(dict(key=e.key,day=e.day,axis_cost=dict(zip(AXES,cost.tolist()))))
        preceding += q[i]
        previous_day = e.day
    # Delivery/normalization scope stays fixed when supplied downstream work
    # extends demand coverage. That work carries costs, not extra target credit.
    per_7 = accumulated*7/span
    return dict(delivery=dict(zip(PURPOSES,delivered.tolist())), targets=dict(zip(PURPOSES,targets.tolist())),
        shortfall_by_purpose=dict(zip(PURPOSES,shortfall.tolist())),
        delivery_loss=float(np.dot(shortfall,policy.purpose_weights)),
        demand_by_axis=dict(zip(AXES,per_7.tolist())),
        demand_cost=float(np.average(per_7,weights=scenario.axis_weights)),session_costs=sessions,
        stage_start=stage_start,decision=decision,end=end,delivery_end=delivery_end,
        physiology_established=False,planner_influence=False)


def score(parts, exchange):
    if not isfinite(exchange) or exchange < 0:
        raise ValueError('Nonnegative explicit exchange required')
    return (parts['delivery_loss']+exchange*parts['demand_cost']
            + parts.get('easy_duration', {}).get('cost', 0.))


def exchange_region(contrasts):
    """Strict preferred loss < counterexample loss; no candidate tuning."""
    low, high, impossible = 0., float('inf'), []
    constraints = []
    for name,preferred,other in contrasts:
        benefit = preferred['delivery_loss']-other['delivery_loss']
        demand = preferred['demand_cost']-other['demand_cost']
        constraints.append(dict(name=name,loss_intercept=benefit,loss_slope=demand))
        if abs(demand) < 1e-12:
            if benefit >= -1e-12:
                impossible.append(name)
        elif demand > 0:
            high = min(high,-benefit/demand)
        else:
            low = max(low,-benefit/demand)
    feasible = not impossible and low < high and high > 0
    return dict(feasible=feasible,lower_exclusive=low,upper_exclusive=high if isfinite(high) else None,
        impossible=impossible,constraints=constraints,
        nominal_exchange=(low+high)/2 if feasible and isfinite(high) else None)


def describe(program, history, start, days):
    end = start+timedelta(days=days)
    rows = [e for e in program if start <= e.day < end]
    dates = sorted(e.day for e in rows)
    role_gaps = {role:[(b.day-a.day).days for a,b in zip(
        [e for e in rows if e.role==role],[e for e in rows if e.role==role][1:])]
        for role in ('long','quality')}
    totals = [sum(e.miles for e in (*history,*rows) if day-timedelta(days=6) <= e.day <= day)
        for day in (start+timedelta(days=i) for i in range(days))]
    streak = longest = 0
    all_dates = {e.day for e in (*history,*rows)}
    # Include recorded lead-in in any streak crossing the plan boundary.
    for offset in range(-len(history)*7,days):
        streak = streak+1 if start+timedelta(days=offset) in all_dates else 0
        if offset >= 0:
            longest = max(longest,streak)
    return dict(miles=sum(e.miles for e in rows),runs=len(rows),miles_per_7=sum(e.miles for e in rows)*7/days,
        runs_per_7=len(rows)*7/days,rolling_7_miles_range=[min(totals),max(totals)],
        role_counts={role:sum(e.role==role for e in rows) for role in ('easy','long','quality')},
        long_gaps=role_gaps['long'],quality_gaps=role_gaps['quality'],
        maximum_bracketed_run_gap=max(((b-a).days for a,b in zip(dates,dates[1:])),default=None),
        maximum_consecutive_run_dates_including_lead_in=longest,
        gap_edges_censored=True)
