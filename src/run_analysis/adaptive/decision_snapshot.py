"""Frozen offline A1 evidence boundary, not stage construction or a new model.

Reuse original projection evidence when supplied. Aggregate-only callers explicitly
have no measured pace samples; whole-session average pace is not aerobic efficiency.
"""
from dataclasses import dataclass, fields, is_dataclass, replace
from datetime import date, datetime, timedelta, timezone as utc_timezone
from hashlib import sha256
from functools import cached_property, lru_cache
from copy import deepcopy
import json
from zoneinfo import ZoneInfo

from run_analysis.athlete_projection import (
    AthleteSnapshot, PaceSample, SessionEvidence, accounting_signature, aware, snapshot,
)
from run_analysis.demonstrated_training import demonstrated_baseline
from run_analysis.performance_reference import AerobicReference, freeze_aerobic_reference
from run_analysis.program_constraints import DateRestriction, ReportedDistanceReference
from run_analysis.program_delivery import TargetSegment
from run_analysis.recovery_response import ResponseExpectation
from run_analysis.session_feedback import with_session_feedback
from run_analysis.health_reports import read_health_reports
from run_analysis.session_decision import (
    CoachingContract, CurrentContext, DatedResponse, DoseLimits,
    PrescriptionBinding, ProgramDecisionInput,
)
from run_analysis.training_signals import HeartRateReference
from run_analysis.workload_accounting import Quantity
from run_analysis.workload_profile import WorkBout
from run_analysis.adaptive.continuous_program_adapter import evaluate_continuous, validate_continuous_runs
from run_analysis.adaptive.direct_quality_load import PhaseDose, PhaseHistory
from run_analysis.adaptive.program_evidence_adapter import AccountedRun
from run_analysis.adaptive.program_objective_trial import DemandScenario, EasyDurationPolicy, TrialPolicy, TrialSession
from run_analysis.adaptive.shared_program_evaluator import ProgramContext
from run_analysis.adaptive.timely_purpose_trial import PurposeTimingPolicy
from run_analysis.adaptive.prospective_progression import ProspectiveVolumeReview,expected_volume_review
from run_analysis.adaptive.frequency_range_trial import (
    FrequencyRangePolicy, FrequencyStageTarget, FrequencyMidpointPreference,
    FrequencyContract, assess_frequency, compile_frequency_exposure)
from run_analysis.adaptive.planner_preferences import PlannerPreferencesRevision
from run_analysis.adaptive.race_goal import RaceGoalRevision, RaceGoalApplication
from run_analysis.adaptive.race_event import RaceEventResolution
from .quality_program_contract import QualityProgramme, PurposeDelivery
from .quality_purpose_contract import QualityPurposeContract, PurposeRequirement
from run_analysis.work_stimulus import CappedStimulusTarget, StimulusTarget
from .quality_coaching_policy import QualityCoachingSequence, CoachedRecipe
from .quality_assignment_accounting import QualityCompletion, IssuedQualityAssignment
from .foregone_volume import ForegoneVolume


@dataclass(frozen=True)
class CapturedAppInputs:
    captured_at: datetime
    contexts: tuple[CurrentContext, ...]
    restrictions: tuple[DateRestriction, ...]
    responses: tuple[DatedResponse, ...]
    # Physical key, capacity eligibility, explicit role override (if any).
    metadata: tuple[tuple[str, bool, str | None], ...]
    saved_health_status: str | None = None
    health_report_time_known: bool = False


def capture_app_inputs(connection, *, captured_at, timezone, activity_keys, expectations=None, previous=None):
    """Read existing app tables once, without backdating mutable manual metadata.

    activity_keys associates DB identities with the already reconciled physical
    ledger; expectations contains only actual pre-issued expectations. A missing
    expectation preserves unknown response, never manufactures a target effort.
    Capture now and retain the snapshot for replay, not a historical SQL query.
    """
    aware(captured_at)
    zone = ZoneInfo(timezone)
    expectations = expectations or {}
    if previous is not None and previous.captured_at > captured_at:
        raise ValueError('Previous capture must not be from the future')
    previous_responses = {r.key: r for r in previous.responses} if previous else {}
    if len(set(activity_keys.values())) != len(activity_keys):
        raise ValueError('Unique database-to-physical identity mapping required')
    own_transaction = not connection.in_transaction
    if own_transaction:
        connection.execute('BEGIN')
    try:
        state = {r['key']: r for r in connection.execute(
            "SELECT key,value_json,updated_at_utc FROM app_state WHERE key IN ('current_health','weekly_forced_rest_dates')")}
        contexts, restrictions, responses, metadata = [], [], [], []
        reports=read_health_reports(connection,as_of=captured_at)
        saved_health_status = None
        # Absence of a previously captured restriction is not an explicit clear.
        # Normal health / an explicit empty date list supersede it when present.
        if previous is not None:
            if 'current_health' not in state:
                contexts.extend(previous.contexts)
            if 'weekly_forced_rest_dates' not in state:
                restrictions.extend(previous.restrictions)
        for key, r in state.items():
            if key == 'current_health' and reports:
                # This mutable cache is not evidence when a causal user-report
                # journal exists, including when a later refresh rewrote it.
                continue
            available = aware(datetime.fromisoformat(r['updated_at_utc']))
            if available > captured_at:
                raise ValueError('Mutable app state cannot reconstruct this historical cutoff')
            value = json.loads(r['value_json'])
            if key == 'current_health':
                status = {'normal': 'usual', 'little_tired': 'little_tired',
                    'sick_or_recovering': 'health_concern', 'pain_or_injury_concern': 'health_concern'}.get(value.get('health_status'))
                if status is None:
                    raise ValueError('Unrecognized saved current-health evidence')
                saved_health_status = value['health_status']
                # Both manual requests AND automatic schedule regeneration write
                # this row. updated_at is NOT a reliable self-report timestamp.
                if status == 'health_concern':
                    contexts.append(CurrentContext(available.astimezone(zone).date(), available,
                        status, 'app:current_health'))
                elif status == 'little_tired':
                    contexts.append(CurrentContext(available.astimezone(zone).date(), available,
                        'unverified', 'app:current_health'))
                elif previous is not None:
                    # Undated saved "normal" is not documented clearance of an
                    # existing concern. Explicit current reports enter separately.
                    contexts.extend(r for r in previous.contexts if r.state in {'health_concern', 'conflicting'})
            else:
                if not isinstance(value, list):
                    raise ValueError('Malformed protected dates cannot mean no restrictions')
                for day in sorted({date.fromisoformat(v) for v in value}):
                    restrictions.append(DateRestriction('app:rest:'+day.isoformat(), day,
                        day+timedelta(days=1), available, 'reserved_rest', 'app:weekly_forced_rest_dates'))
        if reports:
            saved_health_status=reports[-1]['status']
            for report in reports:
                status={'normal':'usual','little_tired':'little_tired',
                    'sick_or_recovering':'health_concern','pain_or_injury_concern':'health_concern'}[report['status']]
                contexts.append(CurrentContext(report['reported_at'].astimezone(zone).date(),
                    report['reported_at'],status,'app:current_health'))
        for activity_id, physical_key in sorted(activity_keys.items()):
            record = connection.execute('''SELECT a.id,a.created_at_utc,a.updated_at_utc,
                o.perceived_exertion,o.health_tag,o.illness,o.include_in_model,o.workout_type
                FROM activities a LEFT JOIN run_overrides o ON o.activity_id=COALESCE(NULLIF(a.activity_id,''),a.activity_uid)
                WHERE a.id=?''', (activity_id,)).fetchone()
            if record is None:
                raise ValueError('Unresolved activity identity')
            if any(aware(datetime.fromisoformat(record[k])) > captured_at for k in ('created_at_utc', 'updated_at_utc')):
                raise ValueError('Mutable activity metadata is newer than the capture cutoff')
            for imported, in connection.execute('''SELECT s.imported_at_utc FROM source_files s
                JOIN activity_sources a ON a.source_file_id=s.id WHERE a.activity_id=?''', (activity_id,)):
                if aware(datetime.fromisoformat(imported)) > captured_at:
                    raise ValueError('Current source set cannot reconstruct this historical cutoff')
            feedback = with_session_feedback(connection, [record])[0]
            issued, expected = expectations.get(physical_key, (None, ResponseExpectation('no saved pre-run expectation')))
            # Immutable FIT availability survives the first capture/restart.
            # Mutable manual overrides still require capture-time provenance.
            response_at = (max(aware(datetime.fromisoformat(stamp))
                               for stamp in feedback['feedback_source_imported_at'])
                           if feedback['feedback_time_known'] else captured_at)
            report = DatedResponse(physical_key, issued, response_at, expected,
                'captured app feedback; effort='+str(feedback['effort_source'])+'; feeling='+str(feedback['feeling_source']),
                feedback['perceived_exertion'], feedback['workout_feel'], feedback['feedback_conflict'])
            old = previous_responses.get(physical_key)
            responses.append(old if old is not None and replace(report, available_at=old.available_at) == old else report)
            eligible = (not record['illness'] and record['health_tag'] in (None, '', 'normal')
                        and record['include_in_model'] != 0)
            metadata.append((physical_key, eligible, record['workout_type']))
        return CapturedAppInputs(captured_at, tuple(contexts), tuple(restrictions), tuple(responses),
                                 tuple(metadata), saved_health_status,bool(reports))
    finally:
        if own_transaction:
            connection.rollback()  # Release read snapshot; no writes performed.


@dataclass(frozen=True)
class DecisionSnapshot:
    as_of: datetime
    timezone: str
    stage_key: str
    stage_issued_at: datetime
    source: str
    context: ProgramContext
    history: tuple[AccountedRun, ...]
    phase_history: tuple[PhaseHistory, ...]
    contracts: tuple[tuple[str, CoachingContract], ...]
    current: CurrentContext | None
    responses: tuple[DatedResponse, ...]
    restrictions: tuple[DateRestriction, ...]
    athlete: AthleteSnapshot
    aerobic_reference: AerobicReference
    resting_hr: float
    maximum_hr: float
    references_available_at: datetime
    references_source: str
    measurement_context: str
    app_inputs: CapturedAppInputs | None = None
    prospective_volume_review: ProspectiveVolumeReview | None = None
    frequency_contract: FrequencyContract | None = None
    planner_preferences: PlannerPreferencesRevision | None = None
    race_goal: RaceGoalRevision | None = None
    race_goal_history: tuple[RaceGoalApplication, ...] = ()
    quality_coaching: QualityCoachingSequence | None = None
    quality_coaching_reference: tuple = ()
    quality_completions: tuple[QualityCompletion, ...] = ()

    @cached_property
    def fingerprint(self):
        return sha256(serialize(self).encode()).hexdigest()

    def demonstrated(self):
        return demonstrated_baseline(self.athlete.sessions, self.as_of, timezone=self.timezone)

    @cached_property
    def frequency_exposure(self):
        zone = ZoneInfo(self.timezone)
        return compile_frequency_exposure(self.frequency_contract,
            tuple(r.start.astimezone(zone).date() for r in self.history),
            decision=self.context.decision, coverage_end=self.context.end,
            taper_start=self.context.taper_start)


# Closed, typed JSON format: no pickle, dynamic imports or executable objects.
TYPES = {cls.__name__: cls for cls in (
    DecisionSnapshot, CapturedAppInputs, ProgramContext, TrialPolicy, DemandScenario, EasyDurationPolicy,
    PurposeTimingPolicy, TrialSession, AccountedRun, Quantity, WorkBout, PhaseHistory,
    PhaseDose, CoachingContract, DoseLimits, CurrentContext, DatedResponse,
    ResponseExpectation, DateRestriction, AthleteSnapshot, SessionEvidence,
    PaceSample, AerobicReference,TargetSegment,ProspectiveVolumeReview,
    FrequencyRangePolicy, FrequencyStageTarget, FrequencyMidpointPreference, FrequencyContract,
    PlannerPreferencesRevision, RaceGoalRevision, RaceGoalApplication, RaceEventResolution,
    QualityProgramme, PurposeDelivery, QualityPurposeContract, PurposeRequirement, CappedStimulusTarget, StimulusTarget,
    QualityCoachingSequence, CoachedRecipe, QualityCompletion, IssuedQualityAssignment,
    ForegoneVolume, ReportedDistanceReference,
)}


def _encode(value):
    if isinstance(value, datetime):
        aware(value)
        return {'type': 'datetime', 'value': value.astimezone(utc_timezone.utc).isoformat()}
    if type(value) is date:
        return {'type': 'date', 'value': value.isoformat()}
    if is_dataclass(value) and type(value).__name__ in TYPES:
        # An absent delivery contract has the historical aggregate meaning.
        # Preserve canonical old snapshot bytes; new contracts are explicit.
        return {'type': type(value).__name__, 'fields': {f.name: _encode(getattr(value, f.name)) for f in fields(value)
            if not (isinstance(value,ProgramContext) and f.name=='delivery_contract'
                and value.delivery_contract=='aggregate_v1')
            and not (isinstance(value,ProgramContext) and f.name=='quality_programme'
                and value.quality_programme is None)
            and not (isinstance(value,ProgramContext) and f.name=='reported_distance'
                and value.reported_distance is None)
            and not (isinstance(value,ProgramContext) and f.name in {'volume_shortfall_policy','volume_shortfall_since'}
                and getattr(value,f.name) is None)
            and not (isinstance(value,ProgramContext) and f.name=='foregone_volume' and not value.foregone_volume)
            and not (isinstance(value,QualityCoachingSequence) and f.name=='cadence_contract'
                and value.cadence_contract=='amplitude_v1')
            and not (isinstance(value,QualityCoachingSequence) and f.name=='focus'
                and value.focus=='variety')
            and not (isinstance(value,ProgramContext) and f.name=='continuation_boundary_references'
                and not value.continuation_boundary_references)
            and not (isinstance(value,ProgramContext) and f.name=='continuation_guard_days'
                and not value.continuation_guard_days)
            and not (isinstance(value,ProgramContext) and f.name=='volume_linear_coefficient'
                and not value.volume_linear_coefficient)
            and not (isinstance(value,ProgramContext) and f.name=='purpose_continuation'
                and not value.purpose_continuation)
            and not (isinstance(value,ProgramContext) and f.name=='purpose_continuation_phase'
                and value.purpose_continuation_phase=='uniform_v1')
            and not (isinstance(value,ProgramContext) and f.name=='continuation_distribution'
                and value.continuation_distribution=='linear_v1')
            and not (isinstance(value,DecisionSnapshot) and f.name=='prospective_volume_review'
                and value.prospective_volume_review is None)
            and not (isinstance(value,DecisionSnapshot) and f.name=='frequency_contract'
                and value.frequency_contract is None)
            and not (isinstance(value,DecisionSnapshot) and f.name=='planner_preferences'
                and value.planner_preferences is None)
            and not (isinstance(value,PlannerPreferencesRevision) and f.name in {'long_run_weekday','long_run_not_before'}
                and getattr(value,f.name) is None)
            and not (isinstance(value,ProgramContext) and f.name in {'long_run_weekday','long_run_not_before'}
                and getattr(value,f.name) is None)
            and not (isinstance(value,DecisionSnapshot) and f.name=='quality_coaching'
                and value.quality_coaching is None)
            and not (isinstance(value,DecisionSnapshot) and f.name in {'quality_coaching_reference','quality_completions'}
                and not getattr(value,f.name))
            and not (isinstance(value,(DecisionSnapshot,ProgramContext)) and f.name=='race_goal'
                and value.race_goal is None)
            and not (isinstance(value,(DecisionSnapshot,ProgramContext)) and f.name=='race_goal_history'
                and not value.race_goal_history)
            and not (isinstance(value,ProgramContext) and f.name=='race_demand'
                and not value.race_demand)
            and not (isinstance(value,ProgramContext) and f.name=='race_event'
                and value.race_event is None)
            and not (isinstance(value,ProgramContext) and f.name=='race_event_history'
                and not value.race_event_history)
            and not (isinstance(value,RaceGoalRevision) and f.name=='general_long_goal_miles'
                and value.general_long_goal_miles is None)
            and not (isinstance(value,FrequencyMidpointPreference) and f.name=='center_fraction'
                and value.center_fraction == .5)}}
    if isinstance(value, tuple):
        return {'type': 'tuple', 'value': [_encode(v) for v in value]}
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise ValueError(f'Unsupported snapshot value: {type(value).__name__}')


def _decode(value):
    if not isinstance(value, dict):
        if value is not None and type(value) not in (str, int, float, bool):
            raise ValueError('Invalid snapshot scalar')
        return value
    kind = value['type']
    if kind == 'datetime':
        return aware(datetime.fromisoformat(value['value']))
    if kind == 'date':
        return date.fromisoformat(value['value'])
    if kind == 'tuple':
        return tuple(_decode(v) for v in value['value'])
    if kind not in TYPES:
        raise ValueError('Unknown snapshot type')
    if kind in {'StimulusTarget','CappedStimulusTarget'}:
        decoded={k:_decode(v) for k,v in value['fields'].items()}
        version=decoded.pop('version')
        target=TYPES[kind](**decoded)
        if target.version!=version:
            raise ValueError('Unknown stimulus target version')
        return target
    return TYPES[kind](**{k: _decode(v) for k, v in value['fields'].items()})


def serialize(state):
    return json.dumps({'version': 'a1_snapshot_v1', 'state': _encode(state)},
                      sort_keys=True, separators=(',', ':'), allow_nan=False)


def restore(serialized, *, expected_fingerprint):
    if sha256(serialized.encode()).hexdigest() != expected_fingerprint:
        raise ValueError('Snapshot content does not match its recorded identity')
    # Memoize validation of exact immutable bytes, never a live harness/decision.
    # Copies keep caller-local state and cached properties independent.
    return deepcopy(_restore_canonical(serialized))


@lru_cache(maxsize=64)
def _restore_canonical(serialized):
    value = json.loads(serialized)
    if value['version'] != 'a1_snapshot_v1':
        raise ValueError('Unknown snapshot version')
    state = _decode(value['state'])
    if not isinstance(state, DecisionSnapshot) or serialize(state) != serialized:
        raise ValueError('Canonical decision snapshot required')
    if any(r.day>=state.as_of.astimezone(ZoneInfo(state.timezone)).date() for r in state.context.foregone_volume):
        raise ValueError('Same-day or future prescriptions cannot be treated as expired volume')
    if state.context.quality_programme is not None and state.context.quality_programme.contract.issued_at>state.as_of:
        raise ValueError('Quality purpose contract must be available at the decision')
    if state.planner_preferences is not None and state.planner_preferences.available_at > state.as_of:
        raise ValueError('Planner preferences must be available at the decision')
    if state.context.long_run_weekday != (state.planner_preferences.long_run_weekday if state.planner_preferences else None):
        raise ValueError('Long-run weekday must match the frozen preference')
    if state.context.long_run_not_before != (state.planner_preferences.long_run_not_before if state.planner_preferences else None):
        raise ValueError('Long-run transition must match the frozen preference')
    if state.quality_coaching is not None:
        if (state.quality_coaching.stage_key!=state.stage_key or
                datetime.fromisoformat(state.quality_coaching.issued_at)>state.as_of or
                not state.quality_coaching_reference or state.quality_coaching_reference[0]!='quality'):
            raise ValueError('Coaching assignment must belong to this available stage and reference')
        actual={r.key:r for r in state.history}
        for receipt in state.quality_completions:
            if receipt.available_at>state.as_of or receipt.activity_key not in actual:
                raise ValueError('Completion must belong to available original history')
            receipt.for_run(actual[receipt.activity_key])
    elif state.quality_completions or state.quality_coaching_reference:
        raise ValueError('Completion/reference needs a frozen coaching sequence')
    if state.race_goal != state.context.race_goal or (state.race_goal and state.race_goal.available_at > state.as_of):
        raise ValueError('Race objective must match the dated decision context')
    goals={a.revision.key:a.revision for a in state.race_goal_history}
    if state.race_goal:
        goals[state.race_goal.key]=state.race_goal
    outcomes=(*state.context.race_event_history,*((state.context.race_event,) if state.context.race_event else ()))
    for event in outcomes:
        if event.race_key not in goals or not goals[event.race_key].goal:
            raise ValueError('Race outcome must belong to applied objective history')
        local_report=event.available_at.astimezone(ZoneInfo(state.timezone)).date()
        if event.available_at > state.as_of or event.available_at <= goals[event.race_key].available_at:
            raise ValueError('Race outcome must follow its goal and be available')
        if event.effective_on != local_report or local_report < goals[event.race_key].race_date:
            raise ValueError('Race outcome effective date must match its local report after the event')
        if event.physical_key is not None and not any(r.key == event.physical_key and
                r.source == 'observed' and r.end <= event.available_at and r.available_at <= event.available_at
                and r.start.astimezone(ZoneInfo(state.timezone)).date() == goals[event.race_key].race_date
                for r in state.history):
            raise ValueError('Race outcome requires available original evidence')
    history = state.race_goal_history
    if state.context.race_goal_history and state.context.race_goal_history != history:
        raise ValueError('Race target history must match the decision owner')
    if history and (history[-1].revision != state.race_goal or
            any(h.applied_at > state.as_of or h.effective_on < h.applied_at.astimezone(ZoneInfo(state.timezone)).date()
                for h in history) or
            any(a.applied_at >= b.applied_at or a.effective_on > b.effective_on
                for a,b in zip(history,history[1:]))):
        raise ValueError('Race application history must be causal and match the applied goal')
    return state


def migrate_snapshot_defaults(serialized, *, expected_fingerprint):
    """Explicit read-only migration of named historical default fields.

    Never silently reinterpret an old record under a new default. The caller
    must retain both identities and the change list; original bytes are untouched.
    Normal restoration intentionally remains strict.
    """
    if sha256(serialized.encode()).hexdigest()!=expected_fingerprint:
        raise ValueError('Snapshot content does not match its recorded identity')
    value=json.loads(serialized)
    if value.get('version')!='a1_snapshot_v1' or json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)!=serialized:
        raise ValueError('Canonical supported source snapshot required')
    defaults={'PurposeTimingPolicy':{'grace_fraction':0.},'TrialPolicy':{'quality_work_minutes':None,'volume_window_days':None},
              'ProgramContext':{'volume_targets':(),'support_targets':(),'steady_continuation_references':(),'soft_horizon_taper':False},
              'AccountedRun':{'advisory_timezone':None}}
    changes=[]
    def visit(v):
        if not isinstance(v,dict):
            return
        if v.get('type') in defaults:
            for key,default in defaults[v['type']].items():
                if key not in v['fields']:
                    v['fields'][key]=_encode(default)
                    changes.append(dict(type=v['type'],field=key,historical_default=default))
        for child in v.values():
            if isinstance(child,dict):
                visit(child)
            elif isinstance(child,list):
                for item in child:
                    visit(item)
    visit(value)
    updated=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
    fingerprint=sha256(updated.encode()).hexdigest()
    return dict(state=restore(updated,expected_fingerprint=fingerprint),changes=changes,
                source_fingerprint=expected_fingerprint,migrated_fingerprint=fingerprint)


def save_snapshot(path, state):
    """Create an immutable decision artifact; an identical retry is harmless."""
    content = serialize(state)
    _save_snapshot_content(path, content)


def save_committed_snapshot(path, content, *, expected_fingerprint):
    """Validate committed bytes and export them without serializing them twice."""
    if sha256(content.encode()).hexdigest() != expected_fingerprint:
        raise ValueError('Snapshot content does not match its recorded identity')
    _restore_canonical(content)  # Validation only; no caller needs a mutable copy.
    _save_snapshot_content(path, content)


def _save_snapshot_content(path, content):
    try:
        with path.open('x') as stream:
            stream.write(content)
    except FileExistsError:
        if path.read_text() != content:
            raise ValueError('Snapshot path already contains different decision evidence')


def resolve_current(reports, *, as_of, timezone):
    """Same-source later report supersedes; unresolved cross-source health persists.

    Never silently choose between simultaneous or independent contradictory reports.
    No source-priority or health-clearance inference from a favorable run rating.
    """
    today = as_of.astimezone(ZoneInfo(timezone)).date()
    known = [r for r in reports if r.available_at <= as_of and r.applies_on <= today]
    latest = []
    for source in sorted({r.source for r in known}):
        rows = [r for r in known if r.source == source]
        at = max(r.available_at for r in rows)
        latest.extend(r for r in rows if r.available_at == at)
    relevant = [r for r in latest if r.applies_on == today or r.state in
                {'health_concern', 'conflicting', 'little_tired', 'unverified'}]
    if not relevant:
        return None
    states = {r.state for r in relevant if r.state != 'unknown'}
    if len(states) > 1 or 'conflicting' in states:
        return CurrentContext(today, max(r.available_at for r in relevant), 'conflicting',
                              'unresolved sources: '+','.join(sorted({r.source for r in relevant})))
    return sorted(relevant, key=lambda r: (r.available_at, r.source, r.state))[-1]


def build_snapshot(*, as_of, timezone, context, stage_key, stage_issued_at, source,
                   history, phase_history, contracts, reference, resting_hr, maximum_hr,
                   references_available_at, references_source, measurement_context, contexts=(), responses=(),
                   restrictions=(), session_evidence=None, app_inputs=None):
    """Cut availability before summaries/inference; freeze once per decision.

    Future evidence is excluded from both content and identity. This snapshot does
    not promise historical reconstruction of mutable application tables.
    """
    for at in (as_of, stage_issued_at, references_available_at):
        aware(at)
    if stage_issued_at > as_of or references_available_at > as_of or not stage_key or not source or not references_source:
        raise ValueError('Available sourced stage and HR references required')
    HeartRateReference(resting_hr, maximum_hr, references_source)
    if json.loads(measurement_context) != {'resting_hr': resting_hr, 'max_hr': maximum_hr}:
        raise ValueError('HR normalization changed; recompile evidence before snapshot')
    if app_inputs is not None:
        if app_inputs.captured_at > as_of:
            raise ValueError('Captured mutable app inputs cannot be backdated')
        contexts = (*contexts, *app_inputs.contexts)
        responses = (*responses, *app_inputs.responses)
        restrictions = (*restrictions, *app_inputs.restrictions)
    known = tuple(sorted((r for r in history if r.available_at <= as_of and r.end <= as_of),
                         key=lambda r: (r.start, r.key)))
    if any(r.source != 'observed' for r in known) or len({r.key for r in known}) != len(known):
        raise ValueError('Unique actual history required')
    validate_continuous_runs(known, (), reference)
    phases = tuple(sorted((r for r in phase_history if r.available_at <= as_of and r.end <= as_of),
                          key=lambda r: (r.start, r.key)))
    by_key = {r.key: r for r in known}
    if len({p.key for p in phases}) != len(phases):
        raise ValueError('Duplicate phase history')
    for p in phases:
        r = by_key.get(p.key)
        if r is None or (p.start, p.end, p.accounting_key) != (r.start, r.end, r.accounting_key):
            raise ValueError('Phase and physical history must reconcile before snapshot')
    declared = tuple(sorted(contracts.items()))
    if any(c.issued_at > as_of for _, c in declared):
        raise ValueError('Stage cannot use future coaching contracts')
    local_day = as_of.astimezone(ZoneInfo(timezone)).date()
    decision = local_day+timedelta(days=1) if any(r.start.astimezone(ZoneInfo(timezone)).date() == local_day for r in known) else local_day
    if context.decision != decision or context.history or context.demand_accounting_key != reference.accounting_key:
        raise ValueError('Context must use snapshot decision date/units and no separate history')
    if session_evidence is None:
        # Physical exposure only; never manufacture steady pace from total distance.
        evidence = tuple(SessionEvidence(r.key, r.start, r.end, r.available_at,
            'intervals' if r.role == 'quality' else r.role, r.miles*1609.344,
            r.demand[0].low*60, r.demand[1], r.accounting_key) for r in known)
    else:
        evidence = tuple(r for r in session_evidence if r.available_at <= as_of and r.end <= as_of)
        if {r.key for r in evidence} != set(by_key) or len(evidence) != len(known):
            raise ValueError('Projection and physical histories must have identical available identities')
        for e in evidence:
            r = by_key[e.key]
            if ((e.start, e.end) != (r.start, r.end) or abs(e.meters-r.miles*1609.344) > .01
                    or abs(e.moving_seconds-r.demand[0].low*60) > .01):
                raise ValueError('Projection evidence must describe the same physical session')
            if e.pace_samples and e.accounting_key not in {
                    reference.accounting_key, accounting_signature(resting_hr, maximum_hr, reference.policy)}:
                raise ValueError('Pace/HR samples require the same frozen HR normalization')
        # Original projection_evidence uses a different cardio signature. Keep
        # original pace samples; take load only from the reconciled continuous ledger.
        evidence = tuple(replace(e, cardio=by_key[e.key].demand[1], accounting_key=reference.accounting_key)
                         for e in evidence)
    if app_inputs is not None:
        flags = {key: (eligible, role) for key, eligible, role in app_inputs.metadata}
        if set(flags)-set(by_key):
            raise ValueError('App metadata requires corresponding physical history')
        roles = {'easy': 'easy', 'recovery': 'easy', 'medium_long': 'easy', 'long': 'long',
                 'intervals': 'quality', 'tempo_threshold': 'quality', 'race': 'quality'}
        for key, (_, role) in flags.items():
            if role is not None and roles.get(role) != by_key[key].role:
                raise ValueError('Role changed: reconcile physical work evidence before snapshot')
        evidence = tuple(replace(e, capacity_eligible=e.capacity_eligible and flags.get(e.key, (True, None))[0]) for e in evidence)
    from .race_event import physical_keys
    race_keys=physical_keys(context)
    if race_keys:
        evidence=tuple(replace(e,capacity_eligible=False) if e.key in race_keys else e for e in evidence)
    athlete = snapshot(evidence, as_of, reference.accounting_key)
    reports = tuple(sorted((r for r in responses if r.available_at <= as_of), key=lambda r: r.key))
    if len({r.key for r in reports}) != len(reports):
        raise ValueError('Response revisions must be resolved before snapshot')
    for r in reports:
        actual = by_key.get(r.key)
        if actual is None or (r.expected_at is not None and r.expected_at > actual.start) or r.available_at < actual.end:
            raise ValueError('Response must match actual history and a pre-run expectation')
    protected = tuple(sorted((r for r in restrictions if r.available_at <= as_of), key=lambda r: r.key))
    if len({r.key for r in protected}) != len(protected):
        raise ValueError('Restriction revisions must be resolved before snapshot')
    state = DecisionSnapshot(as_of, timezone, stage_key, stage_issued_at, source, context,
        known, phases, declared, resolve_current(contexts, as_of=as_of, timezone=timezone),
        reports, protected, athlete, freeze_aerobic_reference(athlete), resting_hr, maximum_hr,
        references_available_at, references_source, measurement_context, app_inputs,
        race_goal=context.race_goal,race_goal_history=context.race_goal_history)
    # Detach mutable caller-owned collections nested in otherwise frozen records.
    encoded = serialize(state)
    return restore(encoded, expected_fingerprint=sha256(encoded.encode()).hexdigest())


def evaluate_snapshot(state, planned, contract_keys, *, reference, evaluator=evaluate_continuous):
    """One state for every candidate; candidate identity contains the snapshot ID."""
    contracts = dict(state.contracts)
    if len(planned) != len(contract_keys):
        raise ValueError('Every candidate session needs a declared contract key')
    bindings = tuple(PrescriptionBinding(r.key, contracts[k].purpose, contracts[k].intensity_key,
                                       contracts[k]) for r, k in zip(planned, contract_keys))
    inputs = ProgramDecisionInput(state.stage_key, state.stage_issued_at,
        state.source+'; snapshot='+state.fingerprint, bindings, state.current, state.responses)
    if state.quality_coaching is not None and evaluator is evaluate_continuous:
        from .coached_pipeline import connected_evaluator
        evaluator=connected_evaluator(state.quality_coaching,state.quality_coaching_reference,state.quality_completions)
    result=evaluator(state.context, state.history, planned, reference=reference,
        as_of=state.as_of, timezone=state.timezone, purpose_fraction=.5,
        restrictions=state.restrictions, decision_input=inputs)
    from .planner_preferences import long_day_admission
    long_day = long_day_admission(state,planned)
    if long_day is not None:
        result['long_run_day_admission']=long_day
        if not long_day['eligible']:
            result['construction_eligible']=False
            result['decision_eligible']=False
            result['constraints']=dict(result['constraints'],status='preferred_long_run_day_conflict')
    if state.prospective_volume_review is not None:
        review=state.prospective_volume_review
        if review.stage_key!=state.stage_key or review.available_at!=state.as_of or review.timezone!=state.timezone:
            raise ValueError('Prospective review must belong to the frozen decision/stage')
        result['construction_eligible']=result['decision_eligible']
        result['progression_delivery']=expected_volume_review(review,planned)
        result['decision_eligible']=result['decision_eligible'] and result['progression_delivery']['funded']
    if state.frequency_contract is not None:
        contract = state.frequency_contract
        if contract.adopted_at > state.as_of:
            raise ValueError('Cannot use future frequency adoption')
        zone = ZoneInfo(state.timezone)
        dates = tuple(r.start.astimezone(zone).date() for r in planned)
        audit = (state.frequency_exposure.assess(dates) if contract.version in {'frequency_midpoint_v2', 'frequency_midpoint_v3'}
            else assess_frequency(contract.targets,
                tuple(r.start.astimezone(zone).date() for r in state.history), dates,
                decision=state.context.decision, coverage_end=state.context.end, policy=contract.policy))
        preference = contract.preference.assess(audit)
        result['construction_eligible'] = result.get('construction_eligible', result['decision_eligible']) and audit['construction_eligible']
        result['decision_eligible'] = result['decision_eligible'] and audit['eligible']
        base = result['evaluation']
        fingerprint = sha256(json.dumps(dict(version=contract.version,
            model_fingerprint=base.fingerprint, snapshot=state.fingerprint,
            frequency=audit, preference=preference), sort_keys=True, allow_nan=False).encode()).hexdigest()
        result['evaluation'] = replace(base, total_cost=base.total_cost+preference['cost'],
            model_version=base.model_version+'/'+contract.version, fingerprint=fingerprint,
            parts=dict(base.parts, pre_frequency_total_cost=base.total_cost,
                pre_frequency_fingerprint=base.fingerprint, frequency_range=audit,
                frequency_preference=preference))
        result['frequency_range'], result['frequency_preference'] = audit, preference
    if state.race_goal is not None:
        from types import SimpleNamespace
        from .race_accounting import allowed_dose
        zone=ZoneInfo(state.timezone)
        stage=SimpleNamespace(long_miles=state.context.policy.long_miles)
        violations=tuple(r.key for r,key in zip(planned,contract_keys)
            if not allowed_dose(state.race_goal,r.start.astimezone(zone).date(),
                SimpleNamespace(key=key,role=r.role,miles=r.miles),stage,state.context.race_event))
        result['race_phase_admission']=dict(eligible=not violations,violations=violations,
            participation_cleared=False)
        if violations:
            result['construction_eligible']=False
            result['decision_eligible']=False
    return result
