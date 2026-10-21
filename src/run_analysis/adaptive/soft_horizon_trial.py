"""Score-only O2 experiment: full accounting-prefix weight, tapered projection.

Reweights the SAME daily delivery, purpose, duration and demand components. It
does not lock a prefix, reward continuity, or alter eligibility/physical load.
This is a finite-horizon approximation to continuing operation, not evidence that
distant fatigue is physiologically smaller. No default planner activation.
"""
from .program_objective_trial import PURPOSES,AXES,demand_value


def day_weight(context,day,*,uniform=False):
    if day>=context.end:
        return 0.
    if uniform or day<context.taper_start:
        return 1.
    if getattr(context,'continuation_distribution','linear_v1')=='c1_v1':
        from .continuation_distribution import c1_survival
        return c1_survival((day-context.taper_start).days/(context.end-context.taper_start).days)
    return (context.end-day).days/(context.end-context.taper_start).days


def tapered_cost(context,evaluation,*,uniform=False,day_weight_function=None):
    if uniform and day_weight_function is not None:
        raise ValueError('Do not combine uniform and alternate taper pricing')
    weight = (lambda day: day_weight(context,day,uniform=uniform)) if day_weight_function is None else day_weight_function
    if context.steady_continuation_references or context.downstream_end is not None:
        raise ValueError('Price this ablation without a second terminal/downstream objective')
    if context.policy.volume_window_days is None or context.timing is None or context.easy_duration_policy is None:
        raise ValueError('Daily rolling-volume, purpose and easy-duration components required')
    # Same normalization as the comparator: common to all candidates at a
    # decision. Do not normalize components separately and change their exchange.
    span=(context.end-context.stage_start).days
    weights=dict(zip(PURPOSES,context.policy.purpose_weights))
    parts=evaluation.parts
    losses={k:0. for k in PURPOSES}
    for row in parts['rolling_volume']['daily']:
        for name in ('volume','ordinary_easy'):
            if name in row:
                losses[name]+=weight(row['day'])*row[name]['loss']/span
    for purpose,data in parts['timing'].items():
        losses[purpose]+=sum(weight(row['day'])*row['loss']
            for row in data['daily'])/span
    delivery=sum(weights[k]*v for k,v in losses.items())
    axis_weights=context.scenario.axis_weights
    axes={name:sum(weight(row['day'])*row['axis_cost'][name]
        for row in parts['session_costs'])*7/span for name in AXES}
    raw_demand=sum(w*axes[name] for name,w in zip(AXES,axis_weights))/sum(axis_weights)
    demand=raw_demand*context.exchange
    short=0.
    for row in evaluation.conditional_ledger:
        if row.role!='easy' or not context.stage_start<=row.day<context.end:
            continue
        q=row.demand[0]
        penalty=context.easy_duration_policy.penalty
        fraction=context.scenario.bound_fraction
        value=penalty(demand_value(q,fraction)) if q.estimate is not None else (
            penalty(q.low)*(1-fraction)+penalty(q.high)*fraction)
        short+=weight(row.day)*value*7/span
    return dict(cost=delivery+demand+short,delivery=delivery,demand=demand,short=short,
        raw_demand_cost=raw_demand,demand_by_axis=axes,losses=losses,
        scope='same training exchanges; unit weight through accounting coverage then linear soft-horizon taper',
        fully_weighted_until=context.taper_start,
        activated_by_default=False)
