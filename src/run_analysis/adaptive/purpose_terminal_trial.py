"""Experimental renewal-policy terminal value, not physiological calibration.

The existing purpose loss and one-session self-cost define a simple recurring
reference policy. Subtract its average daily cost to obtain a finite relative
cost-to-go. This closes purpose-delivery boundary incentives without a new
continuity weight, imposed run date or prescribed repeating calendar.

This first diagnostic omits cross-purpose future fatigue and stage transitions;
it must NOT be mistaken for the full continuation value or activated by default.
"""
from datetime import timedelta
from math import ceil,isfinite
import numpy as np

from .program_objective_trial import PURPOSES,credits,demand_value
from .timely_purpose_trial import envelope_day,interval_loss


def renewal_reference(*,session_cost,grace,memory,weight):
    if (not all(isfinite(v) for v in (session_cost,grace,memory,weight))
            or session_cost<0 or grace<0 or memory<=0 or weight<=0):
        raise ValueError('Explicit finite existing objective terms required')
    # At twelve e-folds, the residual coverage is below 0.000007. This is
    # numerical tail coverage, not a maximum training gap or recovery rule.
    limit=max(1,ceil(grace+12*memory))
    rows=[]
    for days in range(1,limit+1):
        loss=weight*interval_loss(1.,max(0.,days-grace),memory)
        rows.append(((session_cost+loss)/days,days))
    average,period=min(rows)
    if average>=weight:
        return dict(status='no_finite_renewal_advantage',average_cost=weight,period_days=None,
                    session_cost=session_cost,integration_days=limit)
    return dict(status='finite_reference_policy',average_cost=average,period_days=period,
                session_cost=session_cost,integration_days=limit)


def terminal_value(context,evaluation,references):
    if context.timing is None or context.timing.grace_fraction<=0:
        raise ValueError('This diagnostic requires the nonbankable plateau purpose policy')
    if set(references)!=set(context.timing.purposes):
        raise ValueError('A frozen full-purpose physical reference for each scored purpose is required')
    components={}
    ledger=evaluation.conditional_ledger
    for purpose in context.timing.purposes:
        q=np.array([demand_value(v,context.scenario.bound_fraction) for v in references[purpose]])/context.scales
        session_cost=context.exchange*7*float(np.average(.5*q*q,weights=context.scenario.axis_weights))
        index=PURPOSES.index(purpose)
        weight=context.policy.purpose_weights[index]
        cadence=7/getattr(context.policy,purpose+'_per_7')
        grace=cadence*context.timing.grace_fraction
        memory=cadence*context.timing.memory_scale
        reference=renewal_reference(session_cost=session_cost,grace=grace,memory=memory,weight=weight)
        if reference['status']!='finite_reference_policy':
            raise ValueError('The existing objective does not prefer a finite recurring reference; do not manufacture one')
        events=[(r.day,float(credits(r,context.policy)[index])) for r in ledger if r.day<context.end]
        elapsed_loss=0.
        choices=[(session_cost,0)]
        for delay in range(1,reference['integration_days']+1):
            day=context.end+timedelta(days=delay-1)
            elapsed_loss+=weight*envelope_day(events,day,grace=grace,memory=memory)['loss']
            choices.append((session_cost+elapsed_loss-reference['average_cost']*delay,delay))
        value,delay=min(choices)
        if value < -1e-8:
            raise ValueError('Terminal state unexpectedly dominates a freshly completed full-purpose reference')
        components[purpose]=dict(reference=reference,relative_cost=max(0.,value),reference_reset_delay_days=delay)
    total=sum(r['relative_cost'] for r in components.values())/(context.end-context.stage_start).days
    return dict(cost=total,components=components,
        scope='renewal self-cost/purpose approximation only; cross-purpose future fatigue, volume and stage changes omitted',
        activated_by_default=False)
