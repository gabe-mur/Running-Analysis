"""Two-purpose delivery through the SAME cadence/taper/continuation evaluator.

Purpose shares partition one opportunity budget. Replace its generic loss by
the larger of shared-opportunity loss and the share-weighted descriptor losses;
do not add three competing quality penalties or force workout-name rotation.
"""
from dataclasses import dataclass, replace
from math import exp, fsum

from run_analysis.workload_accounting import Quantity
from .quality_purpose_contract import QualityPurposeContract


@dataclass(frozen=True)
class PurposeDelivery:
    key: str
    sustained: Quantity
    faster: Quantity
    opportunity: Quantity

    def __post_init__(self):
        if not self.key or any(q.low<0 or q.high>1 for q in (self.sustained,self.faster,self.opportunity)):
            raise ValueError('Identified bounded purpose delivery required')


@dataclass(frozen=True)
class QualityProgramme:
    contract: QualityPurposeContract
    observed: tuple[PurposeDelivery,...]
    planned: tuple[PurposeDelivery,...]  # Immutable catalogue keys, not dated completed activities.

    def __post_init__(self):
        for rows in (self.observed,self.planned):
            if not isinstance(rows,tuple) or len({r.key for r in rows})!=len(rows):
                raise ValueError('Unique immutable purpose evidence required')

    def delivery(self,key,source):
        if source=='planned':
            prefix,day,dose=key.split(':',2)
            if prefix!='candidate':
                raise ValueError('Planned purpose requires the explicit candidate key owner')
            matches=[r for r in self.planned if r.key==dose]
        else:
            matches=[r for r in self.observed if r.key==key]
        if len(matches)!=1:
            raise ValueError('Missing or ambiguous quality purpose evidence: '+key)
        return matches[0]


def scenario(q,fraction):
    from .program_objective_trial import demand_value
    return demand_value(q,fraction)


def programme_loss(context,future,existing):
    """Apply existing chronology, taper and handoff phase to each descriptor."""
    from .timely_purpose_trial import timing_components
    from .soft_horizon_trial import day_weight
    from .continuation_boundary_trial import purpose_handoff_losses
    qp=context.quality_programme
    if abs(qp.contract.cadence_days-7/context.policy.quality_per_7)>1e-8:
        raise ValueError('Purpose contract and programme must share the accepted cadence')
    if context.timing is None or not context.timing.grace_fraction:
        raise ValueError('Two-purpose programme requires the reviewed plateau cadence evaluator')
    span=(context.end-context.stage_start).days
    result={}
    for req in qp.contract.requirements:
        def scoped(row):
            credit=(scenario(getattr(qp.delivery(row.key,row.source),req.purpose),context.scenario.bound_fraction)
                if row.role=='quality' else 0.)
            return replace(row,quality_credit=credit)
        history=tuple(map(scoped,context.history));planned=tuple(map(scoped,future))
        policy=replace(context.policy,quality_per_7=context.policy.quality_per_7*req.opportunity_share)
        origin=min((context.stage_start,*(r.day for r in history)))
        memory=7/policy.quality_per_7*context.timing.memory_scale
        opening=(('quality',context.scenario.bound_fraction*exp(-(context.decision-origin).days/memory)),)
        view=replace(context,history=history,policy=policy,opening_coverage=opening,quality_programme=None)
        timing=replace(context.timing,purposes=('quality',))
        view=replace(view,timing=timing)
        temporal=timing_components(history,planned,stage_start=context.stage_start,decision=context.decision,
            end=context.end,policy=policy,timing=timing,opening_coverage=opening)['quality']
        if context.race_goal is not None or context.race_goal_history:
            from .race_accounting import apply_purpose_window
            temporal=apply_purpose_window(view,{'quality':temporal})['quality']
        loss=fsum((day_weight(context,r['day']) if context.soft_horizon_taper else 1.)*r['loss']
            for r in temporal['daily'])/span
        continuation=None
        if context.purpose_continuation:
            from datetime import timedelta
            handoffs=tuple(context.taper_start+timedelta(days=i)
                for i in range(1,(context.end-context.taper_start).days+1))
            continuation=purpose_handoff_losses(view,(*history,*planned),handoffs)['quality']
            probabilities=context.handoff_probabilities
            loss+=fsum(p*v for p,v in zip(probabilities or [1/len(handoffs)]*len(handoffs),continuation['by_handoff']))
        result[req.purpose]=dict(loss=loss,share=req.opportunity_share,cadence_days=7/policy.quality_per_7,
            timing=temporal,continuation=continuation)
    descriptor=fsum(r['share']*r['loss'] for r in result.values())
    loss=max(existing,descriptor)
    return dict(loss=loss,shared_opportunity_loss=existing,descriptor_loss=descriptor,purposes=result,
        contract_key=qp.contract.key,
        aggregation='max(shared opportunity loss, share-weighted purpose losses); replacement, not added penalties',
        physiological_equivalence_established=False)
