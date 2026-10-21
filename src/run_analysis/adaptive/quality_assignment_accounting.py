"""Issued-command completion, separate from physical fatigue and adaptation.

The caller must reconcile observed effort/structure with the issued purpose.
No activity title, high HR alone, generic role or whole-run miles proves that.
Time/distance commands retain their primary units, including progression's
fraction of committed distance. Physical load is never rewritten by credit.
"""
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from math import fsum

from run_analysis.athlete_projection import aware
from run_analysis.workload_accounting import Quantity


@dataclass(frozen=True)
class IssuedQualityAssignment:
    stage_key: str
    rank: int
    prescription_key: str
    issued_at: datetime
    purpose: str
    primary_unit: str
    required: Quantity

    def __post_init__(self):
        aware(self.issued_at)
        if (not self.stage_key or type(self.rank) is not int or self.rank<0
                or not self.prescription_key or self.purpose not in {'sustained','faster'}
                or self.primary_unit not in {'time','distance'} or self.required.low<=0
                or self.required.estimate is None):
            raise ValueError('Frozen sourced quality command and primary dose required')

    @property
    def key(self):
        return sha256(json.dumps(asdict(self),sort_keys=True,default=_json).encode()).hexdigest()


def _json(value):
    if isinstance(value,datetime):return aware(value).astimezone(timezone.utc).isoformat()
    if hasattr(value,'__dataclass_fields__'):return asdict(value)
    raise TypeError(type(value).__name__)


def issue_assignment(sequence, rank, catalog):
    recipe=sequence.assignment(rank)
    phases=[p for p in catalog[recipe.key].workout_definition['numeric_contract']['phases']
        if p['kind']=='work']
    units={p['primary']['unit'] for p in phases}
    if not phases or len(units)!=1:
        raise ValueError('Mixed/absent primary work units need an explicit fulfillment contract')
    unit=units.pop()
    # Portion commands are distance fractions, NOT predicted time commands.
    measures=[p['meters'] if unit=='portion' else p['primary']['amount'] for p in phases]
    if unit=='portion':unit='distance'
    required=Quantity(fsum(m['low'] for m in measures),fsum(m['high'] for m in measures),
        fsum(m['estimate'] for m in measures),'Issued primary work-dose acceptance range')
    return IssuedQualityAssignment(sequence.stage_key,sequence.consumed+rank,recipe.key,
        datetime.fromisoformat(sequence.issued_at),recipe.purpose,unit,required)


def evidence_identity(run):
    # Reprocessing source evidence must invalidate a receipt, not reuse credit.
    fields=('key','start','end','available_at','miles','role','demand','bouts',
        'accounting_key','work_evidence_basis')
    return sha256(json.dumps({k:asdict(v) if hasattr(v,'__dataclass_fields__') else v
        for k in fields for v in (getattr(run,k),)},sort_keys=True,
        default=_json).encode()).hexdigest()


@dataclass(frozen=True)
class QualityCompletion:
    assignment: IssuedQualityAssignment
    activity_key: str
    available_at: datetime
    evidence_key: str
    credit: Quantity
    purpose_matched: bool | None

    def __post_init__(self):
        aware(self.available_at)
        if (not self.activity_key or not self.evidence_key or self.available_at<self.assignment.issued_at
                or self.purpose_matched is not None and type(self.purpose_matched) is not bool
                or self.credit.low<0 or self.credit.high>1):
            raise ValueError('Identified available reconciled completion required')

    def for_run(self,run):
        if (run.source!='observed' or run.key!=self.activity_key or self.available_at!=run.available_at or
                evidence_identity(run)!=self.evidence_key):
            raise ValueError('Completion receipt no longer matches the original physical evidence')
        if reconcile_assignment(self.assignment,run,purpose_matched=self.purpose_matched).credit!=self.credit:
            raise ValueError('Stored completion credit disagrees with verified primary evidence')
        return self.credit


def reconcile_assignment(assignment, run, *, purpose_matched):
    """Explicit reconciliation + verified work extent, not inferred adherence.

    An authored range is acceptable throughout that range. Below it receives
    fractional credit; excess cannot bank extra slots. Missing verified work
    remains unresolved, and mismatched intent earns zero. A real partial session
    can advance variety; a missed or unresolved session cannot consume a slot.
    """
    if (run.source!='observed' or assignment.issued_at>run.start
            or purpose_matched is not None and type(purpose_matched) is not bool):
        raise ValueError('Observed work and pre-issued, explicitly reconciled intent required')
    if purpose_matched is False or run.role!='quality':
        credit=Quantity.point(0.,'Observed work does not match issued quality purpose')
    elif purpose_matched is None:
        credit=Quantity(0.,1.,basis='Observed purpose unresolved; not a verified mismatch')
    elif (not run.bouts or run.supported_work is None
            or run.work_evidence_basis in {'unknown','prescribed'}):
        credit=Quantity(0.,1.,basis='Observed primary quality dose unresolved')
    else:
        if assignment.primary_unit=='time':
            ranges=[b.duration_range_seconds or (b.work_seconds,b.work_seconds) for b in run.bouts]
        else:
            ranges=[(b.distance_meters,b.distance_meters) if b.distance_meters is not None else None
                for b in run.bouts]
        if any(x is None for x in ranges):
            credit=Quantity(0.,1.,basis='Observed primary quality dose unresolved')
        else:
            credit=Quantity(min(1.,fsum(x[0] for x in ranges)/assignment.required.low),
                min(1.,fsum(x[1] for x in ranges)/assignment.required.low),
                basis='Verified primary work relative to issued accepted dose; not adaptation')
    return QualityCompletion(assignment,run.key,run.available_at,evidence_identity(run),credit,purpose_matched)


def advance_sequence(sequence, receipt):
    if (receipt.assignment.stage_key!=sequence.stage_key or receipt.assignment.rank!=sequence.consumed
            or receipt.assignment.prescription_key!=sequence.assignment(0).key
            or receipt.assignment.issued_at!=datetime.fromisoformat(sequence.issued_at)):
        raise ValueError('Receipt must match the current persisted issued sequence')
    if receipt.purpose_matched is not True or receipt.credit.low<=0:
        return sequence
    return sequence.after_completion(receipt.assignment.prescription_key)


def completion_json(receipt):
    return json.dumps(asdict(receipt),sort_keys=True,default=_json,allow_nan=False)


def read_completion(value):
    data=json.loads(value)
    issued=dict(data.pop('assignment'))
    issued['required']=Quantity(**issued['required'])
    issued['issued_at']=aware(datetime.fromisoformat(issued['issued_at']))
    data['available_at']=aware(datetime.fromisoformat(data['available_at']))
    data['credit']=Quantity(**data['credit'])
    return QualityCompletion(assignment=IssuedQualityAssignment(**issued),**data)


def assignment_json(assignment):
    return json.dumps(asdict(assignment),sort_keys=True,default=_json,allow_nan=False)


def read_assignment(value):
    data=json.loads(value);data['issued_at']=aware(datetime.fromisoformat(data['issued_at']))
    data['required']=Quantity(**data['required'])
    return IssuedQualityAssignment(**data)
