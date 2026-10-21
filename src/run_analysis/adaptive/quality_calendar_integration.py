"""Reviewed numeric recipes through the existing direct phase-load pipeline.

No HR simulator, effort correction, new scoring weights or template-history
requirement. Phase pace is a frozen extent scenario, not a validated prediction
of the athlete's response to each effort cue.
"""
from dataclasses import replace
from math import fsum

from run_analysis.session_decision import CoachingContract, DoseLimits
from run_analysis.workload_accounting import Quantity
from run_analysis.staged_progression import MILE
from .quality_dose_policy import QualityDoseStage, recipe_requests
from .quality_prescription_contract import PaceReference, compile_quality

VERSION = 'quality_direct_calendar_v1'


def compile_calendar_choices(h, anchor, contract):
    """Freeze reviewed native recipes once, retaining all archived alternatives.

    Reuse the accepted command's existing work-bout distance allocation. Do not
    fit a new pace model or pretend that a novel recipe's effort was observed.
    The accepted anchor defines the exposure stage; generic purpose accounting
    is unchanged and does not establish equal adaptations between recipes.
    """
    from .new_model_daily_harness import Dose
    from .web_materialization import phase_distances
    budget = fsum(s for kind,s in anchor.phases if kind=='work')
    if budget <= 0:
        raise ValueError('Quality catalogue needs an accepted positive work exposure')
    distances = phase_distances(h, anchor, dict(
        distance_miles=dict(target=anchor.miles),
        phases=[dict(kind=k,seconds=dict(target=s)) for k,s in anchor.phases]))
    references = {}
    for name,work in (('easy',False),('work',True)):
        positions = [i for i,(k,_) in enumerate(anchor.phases) if (k=='work')==work]
        seconds = fsum(anchor.phases[i][1] for i in positions)
        meters = fsum(distances[i] for i in positions)*MILE
        if not seconds > 0 or not meters > 0:
            raise ValueError('Accepted anchor needs explicit positive work and easy pace scenarios')
        references[name] = PaceReference(Quantity.point(seconds/meters,
            VERSION+';accepted anchor '+name+' extent scenario, not cue-conditioned execution'),
            contract.issued_at,(anchor.key,),contract.source+';frozen accepted '+name+' pace scenario')
    issued_at=max(contract.issued_at,h.stage.issued_at if h.stage else contract.issued_at,
        h.last_as_of or contract.issued_at)
    stage = QualityDoseStage(h.stage.key if h.stage else anchor.key,issued_at,
        h.stage.ordinary_miles if h.stage else anchor.miles,budget/60,
        7/h.training_policy.quality_per_7,contract.source)
    requests = recipe_requests(stage,issued_at=issued_at,
        distance_rep_seconds={m:m*references['work'].seconds_per_meter.estimate for m in (400,500)},
        open_recovery_seconds=90,
        scenario_source=VERSION+';existing 90-second jog scenario; recovery by feel remains open')
    rows=[]; groups={}
    for request in requests:
        prescription=compile_quality(request,easy_pace=references['easy'],work_pace=references['work'])
        definition=prescription.definition()
        definition.update(calendar_integration=VERSION,
            load_model='direct_quality_load;recent_execution_proxy',
            requested_effort_conditioned=False,
            purpose_accounting='existing generic work-extent coverage; not equal adaptation')
        scope=replace(anchor.forecast_scope,duration_source=VERSION+';frozen numeric primary commands',
            movement_source=VERSION+';accepted anchor pace scenarios') if anchor.forecast_scope else None
        margin=h.policy.phase_fraction_margin
        dose=Dose(prescription.key,'quality',prescription.miles,
            tuple((p.kind,p.seconds.estimate) for p in prescription.phases),scope,
            tuple((p.seconds.low*(1-margin),p.seconds.high*(1+margin)) for p in prescription.phases),definition)
        work=[p.seconds.estimate for p in prescription.phases if p.kind=='work']
        ratios=[prescription.phases[i+1].seconds.estimate/p.seconds.estimate
            for i,p in enumerate(prescription.phases[:-1])
            if p.kind=='work' and prescription.phases[i+1].kind=='recovery'] or [0.]
        limits=DoseLimits(Quantity.point(dose.miles,VERSION),Quantity.point(fsum(work),VERSION),
            Quantity(min(work),max(work)),Quantity(min(ratios),max(ratios)))
        group=(request.family,request.variant.removesuffix('_ordinary').removesuffix('_reduced'))
        groups.setdefault(group,[]).append((dose,limits))
        rows.append(dose)
    contracts={}
    for group in groups.values():
        ordered=sorted(group,key=lambda p:(p[1].work_seconds.estimate,p[0].miles,p[0].key))
        ordinary=ordered[-1][1]
        reduced=ordered[0][1] if len(ordered)>1 and ordered[0][1].work_seconds.high<ordinary.work_seconds.low else None
        shared=CoachingContract(contract.issued_at,contract.source+';'+VERSION,
            contract.purpose,contract.intensity_key,ordinary,reduced,
            ordinary if contract.trial else None,ordinary.work_seconds.estimate if contract.trial else None)
        contracts.update({dose.key:shared for dose,_ in group})
    return tuple(rows),contracts
