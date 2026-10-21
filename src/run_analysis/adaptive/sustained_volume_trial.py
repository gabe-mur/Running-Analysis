"""Offline trailing signed-volume averaging; no publication or gate change.

One cadence of trailing 28-day balances is averaged before rectification. This
is a bounded smoothing experiment, not a physiological adaptation time constant.
Completed exposure expires; hypothetical work cannot offset earlier shortages.
"""
from .program_objective_trial import rolling_volume_components,PURPOSES
from .soft_horizon_trial import day_weight


def sustained_volume_cost(context,evaluation,*,smoothing_days):
    if context.policy.volume_window_days is None or context.steady_continuation_references or context.downstream_end is not None:
        raise ValueError('Explicit rolling-volume context without another terminal experiment required')
    ledger=evaluation.conditional_ledger
    volume=rolling_volume_components(tuple(r for r in ledger if r.source=='observed'),
        tuple(r for r in ledger if r.source=='planned'),stage_start=context.stage_start,
        decision=context.decision,end=context.end,policy=context.policy,
        supporting_miles=context.easy_duration_policy.supporting_easy_miles_per_7
            if context.easy_duration_policy else None,
        volume_targets=context.volume_targets,support_targets=context.support_targets,
        smoothing_days=smoothing_days,foregone_volume=context.foregone_volume,
        opening_before=context.volume_shortfall_since if context.volume_shortfall_policy=='forgone_issued_v2' else None)
    span=(context.end-context.stage_start).days
    losses=dict(volume=0.,ordinary_easy=0.)
    for row in volume['daily']:
        weight=day_weight(context,row['day']) if context.soft_horizon_taper else 1.
        for name in losses:
            if name in row:
                losses[name]+=weight*row[name]['loss']/span
    weights=dict(zip(PURPOSES,context.policy.purpose_weights))
    old=sum(weights[n]*evaluation.parts['shortfall_by_purpose'][n] for n in losses)
    new=sum(weights[n]*v for n,v in losses.items())
    return dict(cost=evaluation.total_cost-old+new,volume_cost=new,previous_volume_cost=old,
        losses=losses,rolling_volume=volume,smoothing_days=smoothing_days,
        additional_memory_days=smoothing_days-1,other_score_terms_unchanged=True,
        planner_influence=False,scope='offline finite trailing smoothing; not progression acceptance')
