"""Offline survival experiments, never a publication contract.

Replace the arbitrary linear end-of-tail collapse with constant handoff hazard
on the existing volume-memory scale. Collapse remaining probability at the last
explicit endpoint, so neither work nor downstream demand disappears. Dates,
physical forecasts, training utility and admission remain unchanged. This still
approximates future operation: a smoother taper does not guarantee stability.
The memory-hazard experiment regressed the general opening witness and is not
retained as a model contract. The C1 experiment has no endpoint pile-up and uses
the minimum-degree cubic with zero slope at both existing taper endpoints.
"""
from dataclasses import replace
from datetime import timedelta
from math import exp, expm1
from types import SimpleNamespace

from .continuation_boundary_trial import continuation_boundary_cost
from .soft_horizon_trial import tapered_cost
from .race_accounting import event_cost
from .continuation_distribution import c1_survival


def distribution(context, *, basis='c1_soft_handoff'):
    memory = context.policy.volume_window_days+context.volume_smoothing_days-2
    count = (context.end-context.taper_start).days
    if memory<=0 or count<1 or not context.continuation_boundary_references:
        raise ValueError('Positive volume memory and existing distributed continuation required')
    if basis not in {'memory_survival','c1_soft_handoff'}:
        raise ValueError('Named explicit continuation survival basis required')
    def survival(offset):
        if basis=='memory_survival':
            return exp(-offset/memory)
        return c1_survival(offset/count)
    probabilities = tuple((exp(-i/memory)*(-expm1(-1/memory)) if basis=='memory_survival'
        else survival(i)-survival(i+1)) for i in range(count-1))
    probabilities += (survival(count-1),)
    handoffs = tuple(context.taper_start+timedelta(days=i+1) for i in range(count))
    def weight(day):
        if day>=context.end:
            return 0.
        return survival(max(0,(day-context.taper_start).days))
    return dict(probabilities=probabilities,handoffs=handoffs,weight=weight,
        memory_days=memory,endpoint_probability=probabilities[-1],basis=basis)


def evaluate_trial(context, baseline, *, probabilities=None, weight=None):
    """Reprice raw components, not a correction fitted to the selected winner.

    Optional probabilities/weight support uniform-distribution parity controls.
    The returned evaluation is diagnostic only; its old fingerprint is NOT a
    valid new-score identity and must never be persisted or published.
    """
    # Snapshot accounting supplies observed rows to the evaluator separately
    # from its persisted context. Retain that scoped physical ledger here too.
    context = replace(context,history=tuple(r for r in baseline.conditional_ledger if r.source=='observed'))
    if (probabilities is None)!=(weight is None):
        raise ValueError('Explicit survival and handoff probabilities must be supplied together')
    if probabilities is None:
        spec = distribution(context)
        probabilities,weight = spec['probabilities'],spec['weight']
    taper = tapered_cost(context,baseline,day_weight_function=weight)
    parts = {k:v for k,v in baseline.parts.items() if k!='continuation_boundary'}
    parts.update(soft_horizon=taper,shortfall_by_purpose=taper['losses'],
        delivery_loss=taper['delivery'],demand_cost=taper['raw_demand_cost'],
        demand_by_axis=taper['demand_by_axis'],
        easy_duration=dict(parts['easy_duration'],cost=taper['short']))
    boundary = continuation_boundary_cost(context,SimpleNamespace(total_cost=taper['cost'],
        parts=parts,conditional_ledger=baseline.conditional_ledger),
        {r:(m,q) for r,m,q in context.continuation_boundary_references},
        include_linear_volume=bool(context.volume_linear_coefficient),
        handoff_probabilities=probabilities)
    added = context.volume_linear_coefficient*context.policy.purpose_weights[0]*boundary.get('linear_volume_loss',0.)
    delivery = taper['delivery']-boundary['previous_volume_cost']+boundary['volume_cost']+boundary.get('purpose_tail_cost',0.)+added
    physical = taper['demand']+boundary['cross_demand_cost']
    event = event_cost(context,tuple(r for r in baseline.conditional_ledger if r.source=='planned'),day_weight_function=weight)
    if event:
        physical += event['cost']
        parts['race_event'] = event
    parts.update(continuation_boundary=boundary,
        volume_benefit=dict(parts.get('volume_benefit',{}),added_volume_cost=added),
        shortfall_by_purpose=dict(taper['losses'],**boundary['losses']),
        trial_scope='Diagnostic survival trial; not a valid persisted score identity')
    frequency = parts.get('frequency_preference',{}).get('cost',0.)
    return replace(baseline,total_cost=delivery+physical+taper['short']+frequency,
        delivery_cost=delivery,weighted_demand_cost=physical,short_run_cost=taper['short'],parts=parts)
