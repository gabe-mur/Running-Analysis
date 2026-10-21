"""Offline issued target/range/actual contract. No scheduler or live DB writes.

Adherence bounds, conditional model exposure, and prediction-error allowances are
different concepts. No default margin is fitted to the two historical examples.
"""
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path

from run_analysis.athlete_projection import aware
from run_analysis.workload_accounting import Quantity
from run_analysis.adaptive.direct_quality_load import KINDS, POLICY, PhaseDose, PhaseHistory, forecast_direct
from run_analysis.adaptive.forecast_handoff import ForecastScope, planning_handoff, read_scope, scoped_forecast


VERSION = 'execution_forecast_contract_v3'


@dataclass(frozen=True)
class TargetBand:
    target: float
    low: float | None
    high: float | None
    source: str

    def __post_init__(self):
        if not self.source or not isfinite(self.target) or self.target <= 0:
            raise ValueError('Positive target and explicit source required')
        if (self.low is None) != (self.high is None):
            raise ValueError('Both adherence bounds or neither required')
        if self.low is not None and (not all(isfinite(x) for x in (self.low,self.high)) or not 0 < self.low <= self.target <= self.high):
            raise ValueError('Ordered positive adherence bounds containing target required')

    def classify(self,value):
        if value is None or self.low is None:
            return 'unknown'
        if not isfinite(value) or value < 0:
            raise ValueError('Finite nonnegative execution required')
        return 'below' if value < self.low else 'above' if value > self.high else 'within'


@dataclass(frozen=True)
class IntendedPhase:
    kind: str
    seconds: TargetBand

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError('Explicit phase kind required')


@dataclass(frozen=True)
class LoadAllowance:
    below: float
    above: float
    source: str

    def __post_init__(self):
        if not self.source or any(not isfinite(x) or x < 0 for x in (self.below,self.above)):
            raise ValueError('Explicit nonnegative load-error allowance required')


@dataclass(frozen=True)
class IntendedExecution:
    session_key: str
    issued_at: datetime
    suggested_start: datetime
    phases: tuple[IntendedPhase,...]
    distance_miles: TargetBand | None
    accounting_key: str
    measurement_context: str
    load_allowance: LoadAllowance | None = None
    forecast_scope: ForecastScope | None = None
    prescription_contract: dict | None = None
    quality_evidence: dict | None = None  # Explicit offline runtime, absent preserves old direct forecasts.

    def __post_init__(self):
        aware(self.issued_at)
        aware(self.suggested_start)
        if not self.session_key or not self.accounting_key or not self.measurement_context or not self.phases:
            raise ValueError('Session, units/HR normalization provenance and phases required')
        if not isinstance(self.phases,tuple):
            raise ValueError('Immutable phase sequence required')
        if self.distance_miles and abs(self.distance_miles.target*10-round(self.distance_miles.target*10)) > 1e-8:
            raise ValueError('Commit new distance target to one decimal before forecasting')
        if self.prescription_contract is not None:
            from .quality_block_contract import read_quality_contract
            prescription=read_quality_contract(self.prescription_contract)
            if (prescription.request.issued_at>self.issued_at or len(prescription.phases)!=len(self.phases)
                    or any(p.kind!=i.kind or abs(p.seconds.estimate-i.seconds.target)>1e-7
                           for p,i in zip(prescription.phases,self.phases))
                    or self.distance_miles is None or prescription.miles!=self.distance_miles.target):
                raise ValueError('Frozen numeric prescription must match issued phases, distance and availability')
        if self.quality_evidence is not None and self.prescription_contract is None:
            raise ValueError('Quality evidence requires a frozen numeric command')
        # suggested_start is advisory, not an availability/recovery gate.


@dataclass(frozen=True)
class CompletedExecution:
    key: str
    session_key: str
    start: datetime
    end: datetime
    available_at: datetime
    phases: tuple[tuple[str,float],...] | None
    distance_miles: float | None
    cardio: Quantity
    accounting_key: str
    measurement_context: str
    phase_evidence: str = 'explicit'

    def __post_init__(self):
        for at in (self.start,self.end,self.available_at):
            aware(at)
        if not self.key or not self.session_key or not self.start < self.end <= self.available_at:
            raise ValueError('Completed original activity identity/timing required')
        if self.phase_evidence not in {'explicit','inferred'}:
            raise ValueError('Explicit or inferred phase provenance required')
        if self.distance_miles is not None and (not isfinite(self.distance_miles) or self.distance_miles < 0):
            raise ValueError('Invalid recorded distance')
        if self.phases is not None:
            if not isinstance(self.phases,tuple) or any(k not in KINDS or not isfinite(s) or s <= 0 for k,s in self.phases):
                raise ValueError('Immutable typed observed phases required')
            if abs(sum(s for _,s in self.phases)-(self.end-self.start).total_seconds()) > 1e-5:
                raise ValueError('Complete elapsed phase accounting or explicit unknown required')


def layout(phases):
    rows,at = [],0.
    for kind,seconds in phases:
        rows.append((kind,at,at+seconds))
        at += seconds
    return tuple(rows)


def _encode(value):
    if is_dataclass(value):
        result=asdict(value)
        if isinstance(value,IntendedExecution) and value.quality_evidence is None:
            result.pop('quality_evidence')
        return result
    if isinstance(value,datetime):
        return value.isoformat()
    raise TypeError(type(value).__name__)


@dataclass(frozen=True)
class FrozenExecution:
    serialized: str
    fingerprint: str

    def __post_init__(self):
        if sha256(self.serialized.encode()).hexdigest() != self.fingerprint:
            raise ValueError('Issued record integrity mismatch')
        if json.loads(self.serialized)['version'] not in {'execution_forecast_contract_v1', 'execution_forecast_contract_v2', VERSION}:
            raise ValueError('Unsupported frozen contract version')

    @property
    def payload(self):
        # Detached copy: callers cannot mutate the issued record.
        return json.loads(self.serialized)


def _history(payload):
    return tuple(PhaseHistory(h['key'],datetime.fromisoformat(h['start']),datetime.fromisoformat(h['end']),
        datetime.fromisoformat(h['available_at']),h['accounting_key'],tuple(PhaseDose(**p) for p in h['phases']))
        for h in payload['history'])


def issue_execution(intent,history,*,parent=None,revision_reason=None):
    if (parent is None) != (revision_reason is None) or revision_reason is not None and not revision_reason.strip():
        raise ValueError('Revision requires both predecessor and explicit reason')
    if parent:
        previous = parent.payload['intent']
        if previous['session_key'] != intent.session_key or datetime.fromisoformat(previous['issued_at']) >= intent.issued_at:
            raise ValueError('Revisions must advance the same session without rewriting history')
    history = tuple(history)
    encoded_intent = _encode(intent)
    if intent.quality_evidence is None:
        target = scoped_forecast(layout((p.kind,p.seconds.target) for p in intent.phases),history,
            as_of=intent.issued_at,accounting_key=intent.accounting_key,scope=intent.forecast_scope)
    else:
        from .quality_forecast_runtime import quality_target, read_profiles
        target_intent=json.loads(json.dumps(encoded_intent,default=_encode))
        profiles=read_profiles(target_intent['quality_evidence']['profiles'])
        target=quality_target(target_intent,history,prepared_profiles=profiles)
    # Save only available contributing evidence; no unrelated/future originals.
    accepted = tuple(h for h in history if h.key in target['source_keys'])
    scenarios = {}
    if all(p.seconds.low is not None for p in intent.phases):
        for edge in ('low','high'):
            if intent.quality_evidence is None:
                result = scoped_forecast(layout((p.kind,getattr(p.seconds,edge)) for p in intent.phases),accepted,
                    as_of=intent.issued_at,accounting_key=intent.accounting_key,scope=intent.forecast_scope)
            else:
                result=quality_target(target_intent,accepted,prepared_profiles=profiles,
                    layout=layout((p.kind,getattr(p.seconds,edge)) for p in intent.phases))
            scenarios[edge] = result['predicted_load']
    # All-low/all-high scenarios need not be extrema for a nonlinear predictor.
    # Bound this frozen model over the duration box via raw observed rate extrema
    # (positive convex mixtures), NOT via the scenario endpoints or their mean.
    phase_bounds = []
    for p in intent.phases:
        obs = [o for h in accepted for o in h.phases if o.kind==p.kind and o.load is not None]
        if not obs and p.kind != 'work':
            obs = [o for h in accepted for o in h.phases if o.kind!='work' and o.load is not None]
        if not obs or p.seconds.low is None:
            phase_bounds = []
            break
        rates = [o.load/o.known_seconds for o in obs]
        phase_bounds.append((p.seconds.low*min(rates),p.seconds.high*max(rates)))
    handoff = planning_handoff(intent.phases,target,accepted,intent.forecast_scope)
    if intent.quality_evidence is not None:
        from .quality_forecast_runtime import quality_handoff
        handoff=quality_handoff(intent.phases,target,accepted,intent.forecast_scope,intent.quality_evidence)
    if handoff['usable']:
        phase_bounds = [(handoff['cardio']['low'],handoff['cardio']['high'])]
    elif intent.forecast_scope is not None:
        phase_bounds = []
    payload = dict(version=VERSION,intent=encoded_intent,policy=asdict(POLICY),
        parent_fingerprint=parent.fingerprint if parent else None,revision_reason=revision_reason,
        history=[asdict(h) for h in sorted(accepted,key=lambda h:h.key)],
        target_forecast=target,execution_scenarios=scenarios,planning_handoff=handoff,
        conditional_model_load_bounds=(sum(p[0] for p in phase_bounds),sum(p[1] for p in phase_bounds)) if phase_bounds else None,
        bounds_meaning='Bounds of frozen model over declared phase-duration box, not physiological limits or predictive confidence; distance/phase feasibility not solved',
        live_scheduler_influence=False)
    serialized = json.dumps(payload,sort_keys=True,separators=(',',':'),default=_encode,allow_nan=False)
    return FrozenExecution(serialized,sha256(serialized.encode()).hexdigest())


def save_execution(record,path):
    """New offline artifact only; exclusive create never overwrites an issue."""
    with Path(path).open('x') as stream:
        json.dump(dict(fingerprint=record.fingerprint,payload=record.serialized),stream)


def read_execution(path):
    with Path(path).open() as stream:
        saved = json.load(stream)
    return FrozenExecution(saved['payload'],saved['fingerprint'])


def compare_execution(record,actual,*,as_of,readiness_changes=()):
    aware(as_of)
    saved = record.payload
    intent = saved['intent']
    issue = datetime.fromisoformat(intent['issued_at'])
    if actual.session_key != intent['session_key'] or actual.start < issue or actual.available_at > as_of:
        raise ValueError('Correct session, pre-run issuance and available completion required')
    if (actual.accounting_key,actual.measurement_context) != (intent['accounting_key'],intent['measurement_context']):
        raise ValueError('Changed units or HR normalization require explicit remeasurement, not silent comparison')
    if saved['policy'] != asdict(POLICY):
        raise ValueError('Frozen forecast policy differs; explicit migration required')
    distance = TargetBand(**intent['distance_miles']).classify(actual.distance_miles) if intent['distance_miles'] else 'not_prescribed'
    kinds = [p['kind'] for p in intent['phases']]
    structure = ('unknown' if actual.phases is None else 'same' if kinds==[k for k,_ in actual.phases]
        else 'unresolved_alignment' if actual.phase_evidence=='inferred' else 'changed')
    phase_status = ([TargetBand(**p['seconds']).classify(s) for p,(_,s) in zip(intent['phases'],actual.phases)]
                    if structure=='same' else [])
    signals = []
    suggested=datetime.fromisoformat(intent['suggested_start'])
    date_status='same_day' if actual.start.astimezone(suggested.tzinfo).date()==suggested.date() else 'different_day'
    if date_status=='different_day':
        signals.append('execution_date_changed')
    if structure=='changed':
        signals.append('workout_structure_changed')
    if distance in {'below','above'} or any(s in {'below','above'} for s in phase_status):
        signals.append('execution_outside_declared_range')
    if readiness_changes:
        signals.append('readiness_changed')
    conditional = None
    if actual.phases is not None:
        if intent.get('quality_evidence') is None:
            conditional = scoped_forecast(layout(actual.phases),_history(saved),as_of=issue,
                accounting_key=intent['accounting_key'],scope=read_scope(intent.get('forecast_scope')))['predicted_load']
        else:
            from .quality_forecast_runtime import quality_target
            conditional=quality_target(intent,_history(saved),layout=layout(actual.phases))['predicted_load']
    residual = (actual.cardio.low-conditional,actual.cardio.high-conditional) if conditional is not None else None
    allowance = intent['load_allowance']
    response = 'unresolved_forecast' if residual is None else 'allowance_not_calibrated_or_declared'
    if structure=='changed':
        response = 'changed_intent_not_adherent_response_test'
    elif structure in {'unknown','unresolved_alignment'}:
        response = 'alignment_unknown'
    elif residual is not None and allowance is not None:
        lo,hi = -allowance['below'],allowance['above']
        response = ('higher_than_declared_allowance' if residual[0] > hi else
            'lower_than_declared_allowance' if residual[1] < lo else
            'within_declared_allowance' if lo <= residual[0] <= residual[1] <= hi else 'uncertain_due_to_load_bounds')
        if response in {'higher_than_declared_allowance','lower_than_declared_allowance'}:
            signals.append('load_response_outside_declared_allowance')
    routine = (structure=='same' and all(s=='within' for s in phase_status)
        and distance in {'within','not_prescribed'} and response=='within_declared_allowance' and not signals)
    return dict(issued_fingerprint=record.fingerprint,actual_key=actual.key,
        distance_adherence=distance,structure=structure,phase_adherence=phase_status,
        execution_date_status=date_status,
        phase_evidence=actual.phase_evidence,
        target_load=saved['target_forecast']['predicted_load'],execution_conditioned_load=conditional,
        actual_cardio=asdict(actual.cardio),actual_distance_miles=actual.distance_miles,
        actual_elapsed_seconds=(actual.end-actual.start).total_seconds(),response_residual_bounds=residual,
        response_status=response,readiness_changes=list(readiness_changes),review_signals=signals,
        routine_variation_without_new_replan_signal=routine,calendar_action='none',
        limits='Review signals are not schedule edits or fitness gains; actual measured load remains unchanged. No error allowance is inferred from adherence bounds.')
