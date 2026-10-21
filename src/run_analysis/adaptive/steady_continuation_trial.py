"""Candidate-state consequences under an explicit coarse continuation.

After the selectable horizon, sustain the accepted stage in expectation. Charge
remaining rolling-volume deficits, purpose coverage until a phase-unknown next
session, and interactions of existing load with future mean demand. Future-only
costs common to all candidates cancel. No new fatigue/continuity coefficient.

This is an approximation, not a searched future calendar: loss of mean volume is
optimistic relative to mean loss of fluctuating volume (Jensen's inequality).
Future stage progression, weather and discrete future feasibility remain outside
this terminal approximation. It must be validated before activation.
"""
from dataclasses import replace
from datetime import timedelta
from math import ceil,exp,expm1
import numpy as np

from run_analysis.program_delivery import _target_total
from .program_objective_trial import PURPOSES,credits,demand_value
from .timely_purpose_trial import envelope_day


def terminal_value(context,evaluation,references):
    if set(references)!={'easy','long','quality'}:
        raise ValueError('Frozen physical easy/long/quality reference doses required')
    if context.policy.volume_window_days is None or context.easy_duration_policy is None or context.timing is None:
        raise ValueError('Explicit rolling, support and purpose policies required')
    span=(context.end-context.stage_start).days
    window=context.policy.volume_window_days
    policy=context.policy
    long_miles,long_demand=references['long']
    quality_miles,quality_demand=references['quality']
    easy_miles,easy_demand=references['easy']
    support=context.easy_duration_policy.supporting_easy_miles_per_7
    if support is None or any(m<=0 for m,_ in references.values()):
        raise ValueError('Positive reference distances and supporting volume required')
    easy_per_7=max(support,policy.weekly_miles-long_miles*policy.long_per_7-quality_miles*policy.quality_per_7)
    miles_per_7=easy_per_7+long_miles*policy.long_per_7+quality_miles*policy.quality_per_7
    def nominal(demand):
        return np.array([demand_value(v,context.scenario.bound_fraction) for v in demand])/context.scales
    daily_demand=(nominal(long_demand)*policy.long_per_7+nominal(quality_demand)*policy.quality_per_7+
                  nominal(easy_demand)*easy_per_7/easy_miles)/7
    rows=tuple(r for r in evaluation.conditional_ledger if r.day<context.end)
    residual=sum((nominal(r.demand)*exp(-((context.end-r.day).days-context.scenario.timing_compression_days)/
        context.scenario.memory_days) for r in rows),np.zeros(3))
    cross=daily_demand*residual/(-expm1(-1/context.scenario.memory_days))
    raw_demand_cost=7/span*float(np.average(cross,weights=context.scenario.axis_weights))
    demand_cost=context.exchange*raw_demand_cost
    def target(segments,day,rate):
        if not segments:
            return rate*window/7
        extended=(*segments[:-1],replace(segments[-1],end=context.end+timedelta(days=window)))
        return _target_total(extended,day-timedelta(days=window-1),day+timedelta(days=1))
    volume=dict(volume=0.,ordinary_easy=0.)
    for offset in range(window-1):
        day=context.end+timedelta(days=offset)
        retained=[r for r in rows if r.day>=day-timedelta(days=window-1)]
        for purpose,arrival,rate,segments in (
            ('volume',miles_per_7,policy.weekly_miles,context.volume_targets),
            ('ordinary_easy',easy_per_7,support,context.support_targets)):
            delivered=sum(r.miles for r in retained if purpose=='volume' or r.role=='easy')+arrival*(offset+1)/7
            volume[purpose]+=max(0.,1-delivered/target(segments,day,rate))**2/span
    timing={}
    for purpose in context.timing.purposes:
        index=PURPOSES.index(purpose)
        period=7/getattr(policy,purpose+'_per_7')
        events=[(r.day,float(credits(r,policy)[index])) for r in rows]
        loss=0.
        for offset in range(max(0,ceil(period)-1)):
            # Unknown phase of the next nominal-cadence completion. Once it
            # occurs, a full reset dominates all pre-boundary purpose evidence.
            survival=max(0.,1-(offset+1)/period)
            loss+=survival*envelope_day(events,context.end+timedelta(days=offset),
                grace=period*context.timing.grace_fraction,memory=period*context.timing.memory_scale)['loss']
        timing[purpose]=loss/span
    delivery=sum(policy.purpose_weights[PURPOSES.index(k)]*v for k,v in {**volume,**timing}.items())
    return dict(cost=demand_cost+delivery,demand=demand_cost,raw_demand_cost=raw_demand_cost,
        demand_by_axis=dict(zip(('running_minutes','cardio','work_minutes'),(7/span*cross).tolist())),
        volume=volume,timing=timing,
        expected_miles_per_7=miles_per_7,expected_support_per_7=easy_per_7,
        scope='fixed accepted stage, fluid-volume and phase-unknown full-purpose continuation; not an issued calendar',
        omitted='future-only common cost, discrete volume fluctuation, later stages and new context',activated_by_default=False)
