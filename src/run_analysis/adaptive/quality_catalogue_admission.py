"""Stage-owned native catalogue bounds, independent of calendar search.

Generation does not authorize a dose. The accepted work envelope and complete
session envelope are supplied by the stage owner, not inferred from a winner.
Execution sensitivity is not an additional hard physiological ceiling.
"""
from dataclasses import dataclass
from datetime import datetime
from math import fsum, isfinite

from .quality_dose_policy import QualityDoseStage
from .quality_recipe_catalogue import DoseWindow, compile_catalogue, reviewed_recipes


@dataclass(frozen=True)
class CatalogueAdmission:
    stage: QualityDoseStage
    maximum_session_miles: float
    maximum_session_seconds: float
    maximum_repetitions: int
    source: str

    def __post_init__(self):
        if (not self.source or any(not isfinite(v) or v<=0 for v in
                (self.maximum_session_miles,self.maximum_session_seconds))
                or type(self.maximum_repetitions) is not int or self.maximum_repetitions<1):
            raise ValueError('Explicit accepted stage session envelope required')

    def window(self,purpose):
        if purpose not in {'sustained','faster'}:
            raise ValueError('Explicit supported purpose required')
        # The common purpose targets (e.g. 8 vs 15 exposure minutes) normalize
        # credit; they are NOT different physical stage ceilings. Rounding a
        # faster target down into a hard cap creates artificial partial slots.
        maximum=60*self.stage.quality_work_minutes
        # No extra arbitrary minimum: the native recipe grid defines meaningful
        # bouts. Smaller alternatives receive only their delivered purpose credit.
        return DoseWindow(1.,maximum,self.maximum_session_miles,self.maximum_session_seconds,
            self.maximum_repetitions,self.source+';'+purpose+' accepted work envelope')


def admit_candidate(candidate,admission):
    """Authored command ranges must fit; uncertain pace does not become a veto."""
    p=candidate.prescription
    if admission.stage.issued_at>p.request.issued_at:
        raise ValueError('Accepted stage must precede the command')
    window=admission.window(candidate.purpose)
    work=fsum(q.seconds.high for q in p.phases if q.kind=='work')
    # Time ranges are choices actually offered to the runner (e.g. fartlek).
    # Distance-command pace ranges describe uncertain execution, not additional
    # distance prescribed. Do not reject a fixed distance due to timing alone.
    command_work=fsum(q.seconds.high if q.primary.unit=='time' else q.seconds.estimate
        for q in p.phases if q.kind=='work')
    command_seconds=fsum(q.seconds.high if q.primary.unit=='time' else q.seconds.estimate for q in p.phases)
    reasons=[]
    if command_work>window.maximum_work_seconds+1e-8:reasons.append('offered_work_range_exceeds_stage')
    if p.miles>window.maximum_session_miles+1e-8:reasons.append('complete_session_distance_exceeds_stage')
    if command_seconds>window.maximum_session_seconds+1e-8:reasons.append('complete_session_command_time_exceeds_stage')
    if candidate.repetitions>window.maximum_repetitions:reasons.append('repetition_count_exceeds_stage')
    return dict(accepted=not reasons,reasons=tuple(reasons),stage_key=admission.stage.key,
        source=admission.source,command_work_seconds=command_work,
        maximum_work_seconds=window.maximum_work_seconds,execution_work_high_seconds=work,
        physiological_clearance=False,whole_program_admission_required=True)


def compile_admitted_catalogue(admission,*,issued_at,**recipe_inputs):
    if not isinstance(issued_at,datetime) or admission.stage.issued_at>issued_at:
        raise ValueError('Available stage and command issuance required')
    recipes=reviewed_recipes(issued_at=issued_at,**recipe_inputs)
    raw=compile_catalogue(recipes,{p:admission.window(p) for p in ('sustained','faster')})
    accepted=[];rejected=[]
    for candidate in raw['candidates']:
        result=admit_candidate(candidate,admission)
        if result['accepted']:accepted.append(candidate)
        else:rejected.append(dict(key=candidate.prescription.key,recipe=candidate.recipe_key,**result))
    return dict(candidates=tuple(accepted),rejected=tuple(rejected),
        empty_recipes=tuple(r.key for r in recipes if not any(c.recipe_key==r.key for c in accepted)),
        stage_key=admission.stage.key,source=admission.source,
        whole_program_admission_required=True,planner_influence=False)
