"""Offline prefix/continuation boundary experiment, not an issued future plan.

At each deterministic handoff, retain the candidate prefix and continue the
accepted stage at mean daily volume. Integrate remaining volume deficits until
all prefix exposure expires, and charge its interaction with future mean demand.
Average LOSSES across handoffs, not credit before the nonlinear loss. Future-only
costs common to the compared candidates are omitted. Discrete future feasibility
and volume fluctuation are not established by this fluid approximation.
"""
from datetime import timedelta
from functools import lru_cache
from types import SimpleNamespace
from .race_event import training_fraction,dated_purpose_fraction
from math import ceil, exp, expm1, isfinite

import numpy as np

from run_analysis.program_delivery import _target_total
from .program_objective_trial import PURPOSES, TrialSession, credits, demand_value
from .timely_purpose_trial import envelope_day
from .demand_interaction import interaction as demand_overlap


def purpose_handoff_losses(context,ledger,handoffs,*,timing_history=None,
                           timing_opening_coverage=None,quality_cadence_contract='amplitude_v1'):
    """Remaining purpose loss until the conditional reference completion.

    Use the existing daily-grid uniform-phase convention: a nominal C-day
    cadence has next arrival floor(U), U uniform on [0,C). Loss on offset d is
    retained with probability max(0,1-(d+1)/C). After a full completion, its
    coverage dominates every bounded prefix credit; future-only loss is common
    and omitted. No credit, date or physical work enters the candidate ledger.
    V20 preserves the latest full completion's known nominal phase while it is
    still on time. Partial credits improve coverage but never postpone that
    completion clock. The explicit coached proportional-grace contract instead
    preserves the latest verified renewal's fractional next-arrival clock in
    BOTH selectable and continuation domains, without changing raw work credit.
    Missing/overdue full phase retains v19's uncertain-arrival
    control; do not fabricate an immediate catch-up. This is a nominal training
    continuation, not a discrete feasibility or physiological guarantee.
    """
    if not handoffs or tuple(handoffs)!=tuple(sorted(set(handoffs))):
        raise ValueError('Ordered unique purpose handoffs required')
    physical_future=tuple(r for r in ledger if r.source=='planned')
    rows=tuple(ledger) if timing_history is None else (*timing_history,*physical_future)
    opening=dict(context.opening_coverage if timing_opening_coverage is None
                 else timing_opening_coverage)
    span=(context.end-context.stage_start).days
    result={}
    for purpose in context.timing.purposes:
        contract=quality_cadence_contract if purpose=='quality' else 'amplitude_v1'
        index=PURPOSES.index(purpose)
        cadence=7/getattr(context.policy,purpose+'_per_7')
        grace=cadence*context.timing.grace_fraction
        memory=cadence*context.timing.memory_scale
        events=sorted((r.day,value) for r in rows
                      if (value:=float(credits(r,context.policy)[index]))>0)
        cursor=0; frontier=[]; cache={}; losses=[]; phases=[]; latest_full=None; latest_renewal=None
        conditioned=getattr(context,'purpose_continuation_phase','uniform_v1')=='known_full_phase_v1'
        for handoff in handoffs:
            while cursor<len(events) and events[cursor][0]<handoff:
                event=events[cursor]
                frontier=[old for old in frontier if old[1]>event[1]]
                frontier.append(event); cursor+=1
                if event[1]>=1.-1e-12:
                    latest_full=event[0]
                if contract=='proportional_grace_v1':
                    renewal=(event[0],event[1])
                    if latest_renewal is None or (event[0]-latest_renewal[0]).days+event[1]*cadence>=latest_renewal[1]*cadence:
                        latest_renewal=renewal
            age=(handoff-latest_full).days if latest_full is not None else None
            known=conditioned and age is not None and age<=cadence
            delay=max(0,ceil(cadence-age)) if known else None
            if contract=='proportional_grace_v1':
                age=(handoff-latest_renewal[0]).days if latest_renewal is not None else None
                known=conditioned and age is not None and age<=latest_renewal[1]*cadence
                delay=max(0,ceil(latest_renewal[1]*cadence-age)) if known else None
            anchored=purpose=='long' and context.long_run_weekday is not None
            if anchored:
                # Continuation must not assume a renewal on an impossible day.
                known=True
                earliest=max(handoff,context.long_run_not_before) if context.long_run_not_before else handoff
                arrival=earliest+timedelta(days=(context.long_run_weekday-earliest.weekday())%7)
                delay=(arrival-handoff).days
            loss=0.
            for offset in range(delay if known else max(0,ceil(cadence)-1)):
                day=handoff+timedelta(days=offset)
                identity=(tuple(frontier),day)
                if identity not in cache:
                    cache[identity]=envelope_day(frontier,day,grace=grace,memory=memory,
                        initial=opening.get(purpose,0.),initial_day=context.decision,
                        credit_contract=contract)['loss']
                fraction = dated_purpose_fraction(context,day,purpose)
                loss+=(1. if known else max(0.,1-(offset+1)/cadence))*cache[identity]*fraction
            losses.append(loss/span)
            if conditioned:
                phases.append(dict(handoff=handoff,last_full_completion=latest_full,
                    age_days=age,arrival_delay_days=delay,
                    basis='preferred_weekday_phase' if anchored else
                        'known_full_nominal_phase' if known else 'uniform_missing_or_overdue_full_phase'))
                if contract=='proportional_grace_v1':
                    phases[-1].update(last_verified_renewal=latest_renewal,
                        basis='preferred_weekday_phase' if anchored else
                            'known_proportional_renewal_phase' if known else 'uniform_missing_or_overdue_renewal_phase')
        result[purpose]=dict(loss=sum(losses)/len(handoffs),by_handoff=losses,
            cadence_days=cadence,phase_basis=('known_full_phase_with_uniform_fallback' if conditioned
                else 'daily_grid_floor_uniform_nominal_cadence'),
            prefix_only=True,future_full_reset_assumed=True)
        if conditioned:
            result[purpose]['phase_by_handoff']=phases
    return result


@lru_cache(maxsize=128)
def _continuation_frame(values, references, support):
    """Immutable candidate-independent continuation arithmetic, exactly once.

    The key contains every read field, including race revisions, targets,
    demand bounds and dates. No prefix ledger, relief, eligibility or candidate
    cost is cached here. All returned arrays are read-only.
    """
    context=SimpleNamespace(**dict(values));policy=context.policy
    references={role:(miles,demand) for role,miles,demand in references}
    easy_miles,easy_demand=references['easy'];long_miles,long_demand=references['long']
    quality_miles,quality_demand=references['quality']
    nominal=lambda demand:np.array([demand_value(q,context.scenario.bound_fraction) for q in demand])/context.scales
    window,smoothing=policy.volume_window_days,context.volume_smoothing_days
    start=context.decision-timedelta(days=window+smoothing-2)
    stop=context.end+timedelta(days=window+smoothing-2)
    race=context.race_goal
    if race and race.goal:
        stop=max(stop,race.race_date+timedelta(days=race.return_days+window+smoothing))
    dates=tuple(start+timedelta(days=i) for i in range((stop-start).days))
    targets=[]
    for segments,rate in ((context.volume_targets,policy.weekly_miles),(context.support_targets,support)):
        if segments:
            final=segments[-1].miles_per_7_days;base=final
            if race and race.goal:
                supported=next((s for s in reversed(segments)
                    if training_fraction(race,context.race_event,s.end-timedelta(days=1))>0),None)
                if supported is not None:
                    base=supported.miles_per_7_days/training_fraction(race,context.race_event,supported.end-timedelta(days=1))
            targets.append(np.array([_target_total(segments,day,day+timedelta(days=1))
                if day<segments[-1].end else max(.1,base*training_fraction(race,context.race_event,day))/7
                if race and race.goal else final/7 for day in dates]))
        else:
            targets.append(np.full(len(dates),rate/7))
    key_miles=(long_miles*policy.long_per_7+quality_miles*policy.quality_per_7)/7
    purpose_factors=np.array([dated_purpose_fraction(context,day,'long') for day in dates]) if race or context.race_goal_history else np.ones(len(dates))
    key_miles=key_miles*purpose_factors if race or context.race_goal_history else key_miles
    if policy.purpose_weights[1]==0:
        if np.any(targets[0]<key_miles-1e-12):
            raise ValueError('Key-session continuation exceeds the single total-volume budget')
        easy_daily=np.maximum(0.,targets[0]-key_miles)
    else:
        easy_daily=np.maximum(targets[1],targets[0]-key_miles)
    total_daily=easy_daily+key_miles
    mean_demand=(easy_daily[:,None]*nominal(easy_demand)/easy_miles+
        purpose_factors[:,None]*(nominal(long_demand)*policy.long_per_7+nominal(quality_demand)*policy.quality_per_7)/7)
    if race and race.goal:
        mean_demand[np.array([day==race.race_date for day in dates])]=0.
    kernel=np.convolve(np.ones(window),np.ones(smoothing)/smoothing)
    target_windows=np.array([np.convolve(v,kernel,mode='valid') for v in targets])
    futures=[]
    for boundary,day in enumerate(dates):
        value=None
        if context.taper_start<day<=context.end:
            count_future=len(dates)-boundary
            decay=np.exp(-np.arange(count_future)/context.scenario.memory_days)
            value=(decay@mean_demand[boundary:]+mean_demand[-1]*
                exp(-count_future/context.scenario.memory_days)/(-expm1(-1/context.scenario.memory_days)))
            value.setflags(write=False)
        futures.append(value)
    for array in (*targets,easy_daily,total_daily,mean_demand,kernel,target_windows):array.setflags(write=False)
    return start,dates,tuple(targets),easy_daily,total_daily,mean_demand,kernel,target_windows,tuple(futures)


def _residual_terms(row,handoffs,scales,scenario):
    """The same per-row multiplies, reused without regrouping the prefix sum."""
    cache=row._continuation_terms if type(row) is TrialSession else None
    numbers=(*scales,scenario.bound_fraction,scenario.timing_compression_days,scenario.memory_days)
    key=(handoffs,tuple((type(value),float(value).hex()) for value in numbers))
    if cache is not None and key in cache:
        return cache[key]
    nominal=np.array([demand_value(q,scenario.bound_fraction) for q in row.demand])/scales
    values=tuple(nominal*exp(-((handoff-row.day).days-scenario.timing_compression_days)
        /scenario.memory_days) if row.day<handoff else None for handoff in handoffs)
    for value in values:
        if value is not None:value.setflags(write=False)
    if cache is not None and len(cache)<4:
        cache[key]=values
    return values


def continuation_boundary_cost(context, evaluation, references, *, distributed=True,
                               include_linear_volume=False,timing_history=None,
                               timing_opening_coverage=None,handoff_probabilities=None,
                               demand_interaction='independent_v1',quality_cadence_contract='amplitude_v1'):
    if 'continuation_boundary' in evaluation.parts:
        raise ValueError('Continuation boundary already included; do not charge it twice')
    if (not context.soft_horizon_taper or context.delivery_contract != 'sustained_v1'
            or context.downstream_end is not None or context.steady_continuation_references):
        raise ValueError('Explicit sustained/tapered context without another terminal policy required')
    if set(references) != {'easy', 'long', 'quality'}:
        raise ValueError('Physical easy/long/quality references required')
    support = context.easy_duration_policy.supporting_easy_miles_per_7
    if support is None:
        raise ValueError('Supporting easy target required')
    policy = context.policy
    easy_miles, easy_demand = references['easy']
    long_miles, long_demand = references['long']
    quality_miles, quality_demand = references['quality']
    if any(not np.isfinite(m) or m <= 0 for m, _ in references.values()):
        raise ValueError('Positive physical reference doses required')
    nominal = lambda demand: np.array([demand_value(q, context.scenario.bound_fraction)
                                      for q in demand]) / context.scales
    # P(a selectable date precedes handoff) exactly matches the existing taper.
    handoffs = tuple(context.taper_start + timedelta(days=i)
                    for i in range(1, (context.end - context.taper_start).days + 1)) if distributed else (context.end,)
    probabilities = None if handoff_probabilities is None else tuple(handoff_probabilities)
    if probabilities is not None and (not distributed or len(probabilities)!=len(handoffs)
            or any(not isfinite(p) or p<=0 for p in probabilities)
            or abs(sum(probabilities)-1.)>1e-12):
        raise ValueError('Positive normalized probabilities for every existing handoff required')
    ledger = evaluation.conditional_ledger
    # Reuse the identical normalization within this evaluation only. Keep row
    # identity (not possibly repeated keys), sum order and decay math unchanged.
    residual_terms={id(row):_residual_terms(row,handoffs,context.scales,context.scenario) for row in ledger}
    window, smoothing = policy.volume_window_days, context.volume_smoothing_days
    span = (context.end - context.stage_start).days
    race = context.race_goal
    channels = ('volume', 'ordinary_easy')
    values=tuple((name,getattr(context,name,()) if name=='race_event_history' else getattr(context,name)) for name in ('policy','scenario','scales','decision','end',
        'volume_smoothing_days','taper_start','volume_targets','support_targets','race_goal','race_event',
        'race_goal_history','race_event_history'))
    reference_values=tuple(sorted((role,miles,tuple(demand)) for role,(miles,demand) in references.items()))
    start,dates,targets,easy_daily,total_daily,mean_demand,kernel,target_windows,futures=_continuation_frame(values,reference_values,support)
    width = len(kernel)
    from .foregone_volume import relief_windows as volume_relief
    relief_windows=volume_relief(context,dates,kernel,targets,ledger)
    totals = np.zeros(2)
    linear_volume = 0.
    cross_axes = np.zeros(3)
    scenarios = []
    for number,handoff in enumerate(handoffs):
        probability = None if probabilities is None else probabilities[number]
        prefix = tuple(r for r in ledger if r.day < handoff)
        amounts = np.zeros((2, len(dates)))
        for row in prefix:
            index = (row.day - start).days
            if 0 <= index < len(dates):
                if row.role != 'race':
                    amounts[0, index] += row.miles
                if row.role == 'easy':
                    amounts[1, index] += row.miles
        boundary = (handoff - start).days
        amounts[:, boundary:] = np.array([total_daily[boundary:],easy_daily[boundary:]])
        balances = np.array([np.convolve(values, kernel, mode='valid') for values in amounts])
        # After the last prefix influence expires, continuing at the reference
        # rate is candidate-independent. Nothing past this endpoint is needed.
        count = (handoff + timedelta(days=width - 1) - context.decision).days
        deficits=np.maximum(0.,1-(balances[:,:count]+relief_windows[:,:count])/target_windows[:,:count])
        losses = deficits ** 2
        if include_linear_volume:
            value_linear = float(deficits[0].sum()/span)
            linear_volume += value_linear/len(handoffs) if probability is None else value_linear*probability
        value = losses.sum(axis=1) / span
        totals += value / len(handoffs) if probability is None else value*probability
        residual = sum((residual_terms[id(r)][number] for r in prefix), np.zeros(3))
        if context.race_demand and context.decision <= race.race_date < handoff:
            residual += nominal(context.race_demand)*exp(-((handoff-race.race_date).days
                -context.scenario.timing_compression_days)/context.scenario.memory_days)
        future_demand=futures[boundary]
        overlap = (future_demand * residual if demand_interaction=='independent_v1'
            else demand_overlap(future_demand,residual,demand_interaction))
        cross = overlap * 7 / span
        if context.race_demand and race.race_date >= handoff:
            # The event survives EVERY handoff, including one before the event
            # enters the selectable horizon. Mean training interacts with it;
            # it supplies neither volume credit nor training-distance headroom.
            ages=np.array([abs((race.race_date-day).days)-context.scenario.timing_compression_days
                           for day in dates[boundary:]])
            interaction=np.exp(-ages/context.scenario.memory_days)@mean_demand[boundary:]
            cross += interaction*nominal(context.race_demand)*7/span
        cross_axes += cross / len(handoffs) if probability is None else cross*probability
        scenarios.append(dict(handoff=handoff, losses=dict(zip(channels, value.tolist())),
            cross_demand_by_axis=cross.tolist()))
    weights = dict(zip(PURPOSES, policy.purpose_weights))
    old = sum(weights[k] * evaluation.parts['shortfall_by_purpose'][k] for k in channels)
    volume = sum(weights[k] * v for k, v in zip(channels, totals))
    raw_cross = float(np.average(cross_axes, weights=context.scenario.axis_weights))
    cross = raw_cross * context.exchange
    replacement_losses=dict(zip(channels,totals.tolist()))
    purpose=None; purpose_cost=0.
    if context.purpose_continuation:
        purpose=purpose_handoff_losses(context,ledger,handoffs,timing_history=timing_history,
                                      timing_opening_coverage=timing_opening_coverage,
                                      quality_cadence_contract=quality_cadence_contract)
        if probabilities is not None:
            purpose={name:dict(value,loss=sum(p*v for p,v in zip(probabilities,value['by_handoff'])))
                     for name,value in purpose.items()}
        for name,value in purpose.items():
            # The existing tapered pre-handoff loss is already charged once.
            # Add ONLY the remaining conditional loss after each handoff.
            replacement_losses[name]=evaluation.parts['shortfall_by_purpose'][name]+value['loss']
            purpose_cost+=weights[name]*value['loss']
        for i,scenario in enumerate(scenarios):
            scenario['purpose_tail_losses']={k:v['by_handoff'][i] for k,v in purpose.items()}
    result = dict(cost=evaluation.total_cost - old + volume + cross,
        volume_cost=volume, previous_volume_cost=old, cross_demand_cost=cross,
        raw_cross_demand_cost=raw_cross, cross_demand_by_axis=cross_axes.tolist(),
        losses=replacement_losses, scenarios=scenarios,
        expected_miles_per_7=float(total_daily[-1]*7), expected_support_per_7=float(easy_daily[-1]*7),
        distribution='uniform_existing_soft_tail_handoffs' if distributed else 'fixed_full_endpoint',
        taper_start=context.taper_start,protected_prefix_days=context.continuation_guard_days,
        physical_delivery_unchanged=True, activated_by_default=False,
        limitations=['Fluid continuation is not a future prescription or established feasible schedule',
                    'Mean volume follows the declared bounded target path; no undeclared later stages assumed',
                    'Purpose timing and short-session costs retain the existing taper',
                    'No calibrated physiological or stability claim'])
    if purpose is not None:
        result['cost']+=purpose_cost
        result['purpose_tail_cost']=purpose_cost
        result['purpose_continuation']=purpose
        result['limitations'][2]=('Purpose coverage preserves known full nominal phase with missing/overdue fallback; short costs retain the taper'
            if getattr(context,'purpose_continuation_phase','uniform_v1')=='known_full_phase_v1'
            else 'Purpose coverage continues until a phase-unknown full reset; short costs retain the taper')
        result['limitations'].append('Nominal cadence/full purpose reset are conditional reference assumptions, not issued work or health clearance')
    if include_linear_volume:
        result['linear_volume_loss'] = linear_volume
    if probabilities is not None:
        result.update(distribution='explicit_handoff_probabilities',handoff_probabilities=probabilities)
    return result


def reference_keys(policy, catalog, available):
    """Same deterministic physical reference selection for benchmark and lifecycle."""
    selected={}
    for role,target in (('easy',policy.ordinary_miles),('long',policy.long_miles),('quality',0)):
        keys=[k for k in available if catalog[k].role==role]
        if not keys:
            raise ValueError('Distributed continuation requires supported quality and easy/long forecasts')
        if role=='quality':
            # Preferences may exclude the archived generic interval command.
            # Use a selectable full-work reference, never a disabled command or
            # the cheaper reduced-work alternative for continuation accounting.
            work=lambda k:sum(seconds for phase,seconds in catalog[k].phases if phase=='work')
            target_work=policy.quality_work_minutes*60
            selected[role]=('quality:ordinary' if 'quality:ordinary' in keys else
                min(keys,key=lambda k:(abs(work(k)-target_work),k)))
        else:
            selected[role]=min(keys,key=lambda k:(abs(catalog[k].miles-target),k))
    return selected
