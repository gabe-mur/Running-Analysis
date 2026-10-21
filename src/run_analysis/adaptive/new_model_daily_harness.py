"""Offline daily search/upload harness for the continuous evaluator.

The bundled synthetic scenario is an engineering control, not an athlete model
or training recommendation. No legacy objective, live DB or network calls.
Default search reuses the established calendar/distance refinement engines through
representation adapters. The original simplified search remains a control only.
"""
from argparse import ArgumentParser
from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, timedelta, time, timezone
from math import isfinite
from pathlib import Path
from hashlib import sha256
from threading import Lock
from time import perf_counter
from zoneinfo import ZoneInfo
import json

from run_analysis.config import load_config
from run_analysis.athlete_projection import SessionEvidence
from run_analysis.workload_accounting import Quantity
from run_analysis.web.upload_service import UploadPayload
from run_analysis.adaptive.continuous_program_adapter import evaluate_continuous
from run_analysis.adaptive.direct_quality_load import PhaseDose, PhaseHistory, phase_observations
from run_analysis.adaptive.execution_forecast_contract import (
    IntendedExecution, IntendedPhase, TargetBand, LoadAllowance, issue_execution, layout,
)
from run_analysis.adaptive.execution_upload_sandbox import ExecutionUploadSandbox
from run_analysis.adaptive.forecast_handoff import ForecastScope, read_scope
from run_analysis.adaptive.program_evidence_adapter import accounted_phases, read_accounted
from run_analysis.adaptive.program_objective_trial import TrialPolicy, DemandScenario, describe, TrialSession, demand_value
from run_analysis.adaptive.shared_program_evaluator import MODEL_VERSION, CONTINUATION_MODEL_VERSION, MEMORY_CONTINUATION_MODEL_VERSION, SINGLE_VOLUME_MODEL_VERSION, VOLUME_BENEFIT_MODEL_VERSION, PURPOSE_CONTINUATION_MODEL_VERSION, PHASE_CONTINUATION_MODEL_VERSION, ProgramContext
from run_analysis.adaptive.timely_purpose_trial import PurposeTimingPolicy
from run_analysis.adaptive.continuous_search_adapter import search as migrated_search
from run_analysis.adaptive.stage_calendar_seeds import complete_stage_seeds
from run_analysis.signal_structure import SignalSpan
from run_analysis.session_decision import CoachingContract, CurrentContext, DatedResponse, DoseLimits
from run_analysis.recovery_response import ResponseExpectation
from run_analysis.program_constraints import DateRestriction
from run_analysis.adaptive.program_objective_trial import EasyDurationPolicy
from run_analysis.adaptive.decision_snapshot import build_snapshot, evaluate_snapshot, restore, save_snapshot,save_committed_snapshot,serialize
from run_analysis.stage_target_history import StageTarget,target_segments
from run_analysis.staged_progression import DevelopmentStage,ProgressionPolicy
from run_analysis.adaptive.frequency_range_trial import FrequencyContract, read_frequency_contract


def point(value):
    return Quantity.point(value,'explicit scenario expectation, not predictive confidence')


def dump(value):
    return json.dumps(value,default=lambda x: x.isoformat(),sort_keys=True,indent=2,allow_nan=False)


# Cache only successful content comparisons, never mutable decisions or harnesses.
# Reading and hashing both sources each time catches edits even if file size and
# modification time are unchanged. Digest-only keys keep memory bounded and small.
_verified_decision_exports = OrderedDict()
_decision_export_headers = OrderedDict()
_decision_exports_lock = Lock()
_DECISION_EXPORT_CACHE_SIZE = 256


def _validate_decision_export(serialized, event, exported):
    identity = (sha256(serialized.encode()).digest(), sha256(exported.encode()).digest())
    # Always decode the exported event into a fresh object, just as resume did
    # before optimization. Semantic equality alone does not preserve JSON key
    # order or integer-versus-float representations in an edited export.
    result = json.loads(exported)
    with _decision_exports_lock:
        if identity in _verified_decision_exports:
            _verified_decision_exports.move_to_end(identity)
            return result
    # Preserve semantic JSON equality, including exports with different spacing.
    if event is None:
        event = json.loads(serialized)
    if result != event:
        raise ValueError('Decision export differs from committed journal')
    with _decision_exports_lock:
        _verified_decision_exports[identity] = None
        _verified_decision_exports.move_to_end(identity)
        _decision_export_headers[identity[0]] = (event['index'], event['snapshot_file'], event['snapshot_fingerprint'])
        _decision_export_headers.move_to_end(identity[0])
        while len(_verified_decision_exports) > _DECISION_EXPORT_CACHE_SIZE:
            _verified_decision_exports.popitem(last=False)
        while len(_decision_export_headers) > _DECISION_EXPORT_CACHE_SIZE:
            _decision_export_headers.popitem(last=False)
    return result


@dataclass(frozen=True)
class Dose:
    key: str
    role: str
    miles: float
    phases: tuple[tuple[str,float],...]
    forecast_scope: ForecastScope | None = None
    phase_scenarios: tuple[tuple[float,float],...] | None = None
    workout_definition: dict | None = None

    def __post_init__(self):
        if (not self.key or self.role not in {'easy','long','quality'} or not isfinite(self.miles)
                or self.miles<=0 or abs(self.miles*10-round(self.miles*10))>1e-8):
            raise ValueError('Explicit role and positive one-decimal dose required')
        if not self.phases or any(not isfinite(s) or s<=0 for _,s in self.phases):
            raise ValueError('Complete positive phase timing required')
        if self.workout_definition is not None:
            from run_analysis.web.schemas import QualitySessionType,WorkoutType,WorkoutStep
            QualitySessionType(self.workout_definition['family'])
            WorkoutType(self.workout_definition['workout_type'])
            steps=self.workout_definition['structure']
            if self.role!='quality' or (steps and abs(sum(WorkoutStep.model_validate(s).duration_minutes
                    for s in steps)*60-sum(s for _,s in self.phases))>1e-6):
                raise ValueError('Workout instructions must describe the scored command duration')
            numeric=self.workout_definition.get('numeric_contract')
            if numeric is not None:
                from .quality_block_contract import read_quality_contract
                p=read_quality_contract(numeric)
                if (p.key!=self.key or p.miles!=self.miles or len(p.phases)!=len(self.phases)
                        or any(s.kind!=kind or abs(s.seconds.estimate-seconds)>1e-7
                               for s,(kind,seconds) in zip(p.phases,self.phases))):
                    raise ValueError('Dose must preserve its frozen numeric prescription identity and extents')
        if self.phase_scenarios is not None and (len(self.phase_scenarios)!=len(self.phases) or any(
                not all(isfinite(v) for v in (lo,hi)) or not 0<lo<=s<=hi
                for (_,s),(lo,hi) in zip(self.phases,self.phase_scenarios))):
            raise ValueError('Per-phase positive scenario bounds must contain the command')


@dataclass(frozen=True)
class HarnessPolicy:
    hard_days: int = 21
    tail_days: int = 14
    evaluations: int = 3000
    horizon: str = 'receding'
    # Explicit engineering margins, never production defaults.
    distance_margin: float = .2
    phase_fraction_margin: float = .1
    response_allowance: float | None = None
    stability_tolerance: float | None = None  # New score units; never import the old number.
    steady_continuation: bool = False  # Explicit O2 experiment, not a silent new default.
    soft_horizon_taper: bool = False
    delivery_contract: str = 'aggregate_v1'
    continuation_boundary: bool = False
    continuation_guard_days: int = 0
    volume_linear_coefficient: float = 0.
    prospective_review_hour: int | None = None  # Explicit offline growth trial; absent preserves archived admission.
    purpose_continuation: bool = False  # Scoring-version migration, not a user preference.
    purpose_continuation_phase: str = 'uniform_v1'  # Resume old contracts without reinterpretation.
    complete_first_search: bool = False  # Shared allocation experiment; not a race or coaching preference.
    final_winner_evaluations: int = 0  # Explicit additional offline pair coverage; counted separately.
    coordinated_final_evaluations: int = 0  # Offline exact catalogue refinement; additional full checks.
    continuation_distribution: str = 'linear_v1'  # Archived/live default; C1 is an explicit offline contract.
    general_progression: str | None = None  # Explicit lifecycle version; archived ledgers stay unchanged.
    calendar_merge_evaluations: int = 0  # Explicit release/diagnostic migration; archived defaults unchanged.
    progression_admission: str = 'checkpoint_v1'  # Archived forecast deadline; v2 reviews actual delivery only.
    progression_credit: str = 'issued_nominal_v1'  # Explicit migration; archived reviews remain reproducible.
    volume_shortfall_policy: str | None = None  # Explicit score-only missed-opportunity treatment.

    def __post_init__(self):
        if self.volume_shortfall_policy not in {None,'forgone_issued_v1','forgone_issued_v2'} or (
                self.volume_shortfall_policy and (self.delivery_contract!='sustained_v1' or not self.continuation_boundary)):
            raise ValueError('Foregone volume requires named sustained continuation accounting')
        if (self.progression_credit not in {'issued_nominal_v1', 'actual_distance_v2'}
                or self.progression_credit == 'actual_distance_v2'
                and self.progression_admission != 'observed_rolling_v2'):
            raise ValueError('Actual-distance progression credit requires observed rolling admission')
        if (self.progression_admission not in {'checkpoint_v1', 'observed_rolling_v2'}
                or self.progression_admission == 'observed_rolling_v2'
                and (self.delivery_contract != 'sustained_v1' or not self.continuation_boundary)):
            raise ValueError('Observed progression requires sustained volume and continuation accounting')
        if type(self.calendar_merge_evaluations) is not int or self.calendar_merge_evaluations<0 or (
                self.calendar_merge_evaluations and (self.calendar_merge_evaluations<4 or not self.coordinated_final_evaluations)):
            raise ValueError('Calendar/dose diagnostic requires a bounded allowance and exact dose compiler')
        if self.general_progression not in {None, 'easy_first_coordinated_v1', 'easy_first_coordinated_v2', 'continuous_general_v1', 'continuous_general_v2', 'continuous_general_v3', 'continuous_general_v4', 'continuous_general_v5','continuous_general_v6','continuous_general_v7'} or (
                self.general_progression is not None and self.delivery_contract != 'sustained_v1'):
            raise ValueError('General progression requires a named sustained-delivery policy')
        if self.general_progression in {'continuous_general_v1','continuous_general_v2','continuous_general_v3','continuous_general_v4','continuous_general_v5','continuous_general_v6','continuous_general_v7'} and (
                self.progression_admission != 'observed_rolling_v2' or self.progression_credit != 'actual_distance_v2'):
            raise ValueError('Continuous progression requires observed admission and actual-distance credit')
        if (self.continuation_distribution not in {'linear_v1','c1_v1'} or
                self.continuation_distribution != 'linear_v1' and not self.continuation_boundary):
            raise ValueError('Named continuation distribution requires distributed handoff accounting')
        if type(self.complete_first_search) is not bool:
            raise ValueError('Explicit shared opening-search switch required')
        if type(self.final_winner_evaluations) is not int or self.final_winner_evaluations<0:
            raise ValueError('Explicit nonnegative final-winner evaluation allowance required')
        if (type(self.coordinated_final_evaluations) is not int or self.coordinated_final_evaluations < 0
                or self.coordinated_final_evaluations and not self.continuation_boundary):
            raise ValueError('Coordinated refinement requires a nonnegative allowance and distributed continuation')
        if (self.purpose_continuation_phase not in {'uniform_v1','known_full_phase_v1'}
                or self.purpose_continuation_phase!='uniform_v1' and not self.purpose_continuation):
            raise ValueError('Purpose continuation phase requires a named enabled contract')
        if (type(self.purpose_continuation) is not bool or
                self.purpose_continuation and not self.continuation_boundary):
            raise ValueError('Purpose continuation requires distributed handoff accounting')
        if self.prospective_review_hour is not None and (
                type(self.prospective_review_hour) is not int or not 0<=self.prospective_review_hour<=23
                or self.delivery_contract!='sustained_v1'):
            raise ValueError('Prospective review requires an explicit local checkpoint hour and sustained contract')
        if (not isfinite(self.volume_linear_coefficient) or self.volume_linear_coefficient<0
                or self.volume_linear_coefficient and self.delivery_contract!='sustained_v1'):
            raise ValueError('Linear volume benefit requires a finite nonnegative sustained contract')
        if (type(self.continuation_guard_days) is not int or not 0<=self.continuation_guard_days<=self.hard_days
                or self.continuation_guard_days and (not self.continuation_boundary or self.horizon!='receding')):
            raise ValueError('Memory-covered continuation requires a receding boundary contract')
        if type(self.continuation_boundary) is not bool or self.continuation_boundary and (
                not self.soft_horizon_taper or self.delivery_contract!='sustained_v1' or self.steady_continuation):
            raise ValueError('Distributed continuation requires sustained/tapered policy')
        if self.delivery_contract not in {'aggregate_v1','sustained_v1'}:
            raise ValueError('Unknown delivery contract')
        if (type(self.steady_continuation) is not bool or type(self.soft_horizon_taper) is not bool
                or self.steady_continuation and self.soft_horizon_taper
                or self.hard_days<2 or self.tail_days<1 or self.evaluations<2
                or self.horizon not in {'receding','fixed'} or self.distance_margin<0
                or not 0<=self.phase_fraction_margin<1
                or self.response_allowance is not None and (not isfinite(self.response_allowance) or self.response_allowance<0)
                or self.stability_tolerance is not None and
                (not isfinite(self.stability_tolerance) or self.stability_tolerance<0)):
            raise ValueError('Invalid explicit harness policy')


def accounted(key,start,miles,phases,cardio,role,ref,*,source,available,moving=None,advisory_timezone=None):
    # Synthetic seeds explicitly declare movement; uploads use persisted measured movement.
    return accounted_phases(key,start,miles,phases,cardio,role,ref,source=source,available=available,
        moving=moving or point(sum(s for _,s in phases)/60),advisory_timezone=advisory_timezone)


def planned_accounted(key,start,dose,handoff,ref,*,available):
    if not handoff['usable']:
        raise ValueError('Unresolved forecast cannot become a free candidate')
    run = accounted(key,start,dose.miles,dose.phases,Quantity(**handoff['cardio']),dose.role,ref,
        source='planned',available=available,moving=Quantity(**handoff['running_minutes']),
        advisory_timezone=getattr(start.tzinfo,'key',None) or ('UTC' if start.utcoffset()==timedelta(0) else None))
    return replace(run,command_running_minutes=sum(s for _,s in dose.phases)/60)


def signature(program):
    return tuple(sorted(program))  # (local ISO date, catalogue key), not volatile issue IDs.


def neighbors(program,dates,keys):
    """Insert/delete, change dose/role, move ±1/±2, and swap occupied roles.

    Full objective evaluates every proposal; no legacy proposal score. Yield
    changes across the whole hard+tail scope, never freeze a near-term prefix.
    """
    current = dict(program)
    for day in dates:
        if day in current:
            yield signature((d,k) for d,k in program if d!=day)
        for key in keys:
            if current.get(day)!=key:
                yield signature((*((d,k) for d,k in program if d!=day),(day,key)))
    for day,key in program:
        for step in (-2,-1,1,2):
            other = (datetime.fromisoformat(day).date()+timedelta(days=step)).isoformat()
            if other not in dates:
                continue
            changed = dict(current)
            changed.pop(day)
            if other in current:
                changed[day] = current[other]
            changed[other] = key
            yield signature(changed.items())


def prototype_search(*,dates,keys,previous,evaluate,budget):
    """Reuse prior program as a candidate, not a continuity penalty or lock.

    Empty plus prior seeds; descend full neighborhoods until no improvement or
    budget exhaustion. Exact ties prefer the existing incumbent. Cached only
    within one decision, so new evidence cannot reuse stale costs.
    """
    started = perf_counter()
    cache = {}
    def assess(program):
        program = signature(program)
        if program not in cache:
            if len(cache)>=budget:
                return None
            cache[program] = evaluate(program)
        return cache[program]
    def cost(program):
        result = cache[program]
        return result['cost'] if result['eligible'] else float('inf')
    prior = signature((d,k) for d,k in previous if d in dates)
    assess(prior)
    assess(())
    best = min(cache,key=lambda p:(cost(p),p!=prior,p))
    complete = False
    iterations = 0
    while len(cache)<budget:
        iterations += 1
        incumbent = best
        exhausted = False
        for proposal in neighbors(incumbent,dates,keys):
            result = assess(proposal)
            if result is None:
                exhausted = True
                break
            if cost(proposal)<cost(best)-1e-12:
                best = proposal
        if best==incumbent:
            complete = not exhausted
            break
    if not isfinite(cost(best)):
        raise ValueError('No candidate passes checked constraints; no fallback to an unscored plan')
    return dict(program=best,cost=cost(best),evaluations=len(cache),iterations=iterations,
        budget_exhausted=not complete,local_neighborhood_complete=complete,
        prior_cost=cost(prior) if isfinite(cost(prior)) else None,
        selected=cache[best],seconds=perf_counter()-started,
        search='deterministic_bounded_neighborhood_descent_not_global_optimum')


def churn(previous,current,decision,catalog):
    before,after = dict(previous),dict(current)
    result = {}
    for window in (4,7):
        edits = []
        for n in range(window):
            day = (decision+timedelta(days=n)).isoformat()
            a,b = before.get(day),after.get(day)
            if a==b:
                continue
            old,new = catalog.get(a),catalog.get(b)
            edits.append(dict(day=day,before=a,after=b,date_changed=(a is None)!=(b is None),
                role_changed=bool(old and new and old.role!=new.role),
                distance_change=(new.miles-old.miles) if old and new else None))
        result[str(window)] = edits
    return result


def training_metrics(rows,history,start,days,zone):
    def trial(r):
        return TrialSession(r.key,r.start.astimezone(zone).date(),r.miles,r.role,r.demand,r.quality.low,r.source)
    future,past = tuple(map(trial,rows)),tuple(map(trial,history))
    result = describe(future,past,start,days)
    all_rows = sorted((*past,*future),key=lambda r:r.day)
    end = start+timedelta(days=days)
    result['boundary_gaps'] = {}
    result['gap_coverage'] = {}
    for role in ('all','long','quality'):
        filtered = [r for r in all_rows if r.day<end and (role=='all' or r.role==role)]
        result['boundary_gaps'][role] = [dict(before=a.day,after=b.day,days=(b.day-a.day).days)
            for a,b in zip(filtered,filtered[1:]) if start<=b.day<end]
        previous=[r.day for r in filtered if r.day<start]
        inside=[r.day for r in filtered if r.day>=start]
        result['gap_coverage'][role]=dict(last_before_start=max(previous,default=None),
            first_in_window=min(inside,default=None),last_in_window=max(inside,default=None),
            longest_bracketed_gap_including_lead_in=max(
                (g['days'] for g in result['boundary_gaps'][role]),default=None),
            trailing_days_without_completion=(end-timedelta(days=1)-filtered[-1].day).days if filtered else None,
            trailing_gap_right_censored=True,history_left_censored=not previous)
    for window in (7,14):
        rolling=[]
        for i in range(days):
            day=start+timedelta(days=i)
            within=[r for r in all_rows if day-timedelta(days=window-1)<=r.day<=day]
            rolling.append(dict(day=day,miles=sum(r.miles for r in within),runs=len(within),
                moving_minutes=sum(demand_value(r.demand[0],.5) for r in within),
                long_runs=sum(r.role=='long' for r in within),quality_runs=sum(r.role=='quality' for r in within),
                largest_run_share=max((r.miles for r in within),default=0)/sum(r.miles for r in within) if within else None))
        result[f'rolling_{window}']=rolling
    selected=[r for r in future if start<=r.day<end]
    result['moving_minutes']=sum(demand_value(r.demand[0],.5) for r in selected)
    result['moving_minutes_basis']='observed or explicit nominal scenario; bounds-only quantities use midpoint for reporting'
    result['by_role']={role:dict(runs=sum(r.role==role for r in selected),
        miles=sum(r.miles for r in selected if r.role==role),
        moving_minutes=sum(demand_value(r.demand[0],.5) for r in selected if r.role==role))
        for role in ('easy','long','quality','unknown')}
    result['role_sequence'] = [(r.day,r.role,r.miles) for r in future if start<=r.day<end]
    result['compressed_key_pairs'] = [dict(before=a.day,after=b.day,roles=(a.role,b.role),gap_days=(b.day-a.day).days)
        for a,b in zip(all_rows,all_rows[1:]) if start<=b.day<end and (b.day-a.day).days<=1
        and a.role in {'long','quality'} and b.role in {'long','quality'}]
    result['compression_is_diagnostic_not_automatic_failure'] = True
    result['short_window_rate_warning'] = days<7
    return result


class DailyHarness:
    @classmethod
    def resume(cls, root, config, *, as_of):
        """Restore an archived decision and overlay durable completions.

        Recover complete committed batches; fail explicitly if standalone
        issuance exists that no archived decision owns.
        """
        root = Path(root)
        scenario = json.loads((root/'scenario.json').read_text())
        # Recover missing exports only from a committed, fully owned batch.
        journal=ExecutionUploadSandbox(root,config)
        committed_exports = {}
        try:
            for serialized,content in journal.serialized_decisions():
                digest = sha256(serialized.encode()).digest()
                with _decision_exports_lock:
                    header = _decision_export_headers.get(digest)
                event = None
                if header is None:
                    event = json.loads(serialized)
                    header = (event['index'], event['snapshot_file'], event['snapshot_fingerprint'])
                index, snapshot_file, snapshot_fingerprint = header
                save_committed_snapshot(root/snapshot_file,content,
                    expected_fingerprint=snapshot_fingerprint)
                path=root/f"decision-{index+1:04d}.json"
                if path.exists():
                    event = _validate_decision_export(serialized,event,path.read_text())
                else:
                    if event is None:
                        event = json.loads(serialized)
                    with path.open('x') as stream:
                        stream.write(dump(event))
                committed_exports[path] = event
        finally:
            journal.close()
        # Keep legacy standalone exports and ordering, but do not decode an
        # already-verified committed export a third time.
        events = [committed_exports[p] if p in committed_exports else json.loads(p.read_text())
                  for p in sorted(root.glob('decision-*.json'))]
        if not events:
            raise ValueError('Resume requires a completely archived decision')
        latest = events[-1]
        runtime=latest.get('stage_runtime',scenario)
        if as_of < datetime.fromisoformat(latest['as_of']):
            raise ValueError('Cannot resume before the latest archived decision')
        state = restore((root/latest['snapshot_file']).read_text(),expected_fingerprint=latest['snapshot_fingerprint'])
        obj = cls.__new__(cls)
        obj.sandbox = ExecutionUploadSandbox(root,config)
        try:
            obj.config,obj.origin,obj.policy = config,datetime.fromisoformat(scenario['origin']),HarnessPolicy(**scenario['search_policy'])
            obj.ref,obj.zone = obj.sandbox.reference,ZoneInfo(config['timezone_default'])
            obj.catalog = {d['key']:Dose(**dict(d,phases=tuple(tuple(p) for p in d['phases']),
                phase_scenarios=tuple(tuple(p) for p in d['phase_scenarios']) if d.get('phase_scenarios') else None,
                forecast_scope=read_scope(d.get('forecast_scope')))) for d in runtime['catalog']}
            obj.history = [read_accounted(r) for r in scenario['history']]
            from .review_seed import read_entry,validate_seed
            obj.seed_review_entries=validate_seed(tuple(read_entry(r) for r in scenario.get('seed_review_entries',())),
                obj.history,obj.origin)
            obj.seed_keys = {r.key for r in obj.history}
            obj.phase_history = tuple(PhaseHistory(**dict(h,
                **{k:datetime.fromisoformat(h[k]) for k in ('start','end','available_at')},
                phases=tuple(PhaseDose(**p) for p in h['phases']))) for h in scenario['phase_history'])
            obj.scales,obj.training_policy = state.context.scales,state.context.policy
            obj.policy=replace(obj.policy,delivery_contract=state.context.delivery_contract,
                volume_shortfall_policy=state.context.volume_shortfall_policy,
                volume_linear_coefficient=state.context.volume_linear_coefficient,
                purpose_continuation=state.context.purpose_continuation,
                purpose_continuation_phase=state.context.purpose_continuation_phase,
                continuation_distribution=state.context.continuation_distribution)
            if latest.get('selection_contract') == 'observed_rolling_progression_v2':
                # The committed decision owns admission even if a crash kept
                # the separate scenario-policy export on its previous version.
                if state.prospective_volume_review is not None:
                    raise ValueError('Observed progression decision contains a forecast deadline')
                obj.policy=replace(obj.policy,progression_admission='observed_rolling_v2')
            if latest.get('progression_credit') == 'actual_distance_v2':
                # The committed decision also owns credit after an interrupted
                # separate policy export. Historical decisions are untouched.
                obj.policy=replace(obj.policy,progression_credit='actual_distance_v2')
            obj.general_progression_state=runtime.get('general_progression_state')
            obj.volume_shortfall_since=state.context.volume_shortfall_since
            if obj.general_progression_state:
                obj.policy=replace(obj.policy,general_progression=obj.general_progression_state['version'])
            obj.demand_scenario,obj.exchange,obj.timing = state.context.scenario,state.context.exchange,state.context.timing
            obj.coaching_contracts = dict(state.contracts)
            obj.quality_runtime=runtime.get('quality_runtime')
            obj.easy_duration_policy,obj.decision_source = state.context.easy_duration_policy,scenario['decision_source']
            stage=runtime.get('stage')
            obj.stage=DevelopmentStage(**dict(stage,issued_at=datetime.fromisoformat(stage['issued_at']),
                policy=ProgressionPolicy(**stage['policy']))) if stage else None
            obj.target_history=tuple(StageTarget(**dict(r,available_at=datetime.fromisoformat(r['available_at']),
                review_anchor=datetime.fromisoformat(r['review_anchor']) if r.get('review_anchor') else None,
                effective_on=date.fromisoformat(r['effective_on']))) for r in runtime.get('target_history',()))
            obj.frequency_contract=state.frequency_contract
            obj.reported_distance=state.context.reported_distance
            obj.planner_preferences=state.planner_preferences
            obj.race_goal=state.race_goal
            obj.race_event=state.context.race_event
            obj.race_event_history=state.context.race_event_history
            from .race_goal import RaceGoalApplication
            obj.race_goal_history=state.race_goal_history or ((RaceGoalApplication(state.race_goal,
                state.race_goal.available_at,state.race_goal.available_at.astimezone(obj.zone).date()),)
                if state.race_goal else ())
            if read_frequency_contract(runtime.get('frequency_contract')) != obj.frequency_contract:
                raise ValueError('Frequency runtime differs from committed decision snapshot')
            obj.current_context=state.current
            obj.capacity_eligibility={r.key:r.capacity_eligible for r in state.athlete.sessions}
            obj.responses=state.responses
            obj.restrictions=state.restrictions
            obj.events,obj.publication,obj.last_as_of = events,latest['publication'],datetime.fromisoformat(latest['as_of'])
            from .coached_runtime import restore_runtime
            restore_runtime(obj,runtime,state)
            obj.evidence_as_of = obj.last_as_of
            obj.previous = tuple(tuple(p) for p in latest['program'])
            obj.active,obj.executed,obj.issued_by_day = dict(latest['active']),{},{}
            for event in events:
                for day, issued in event['active'].items():
                    obj.issued_by_day[day] = issued['fingerprint']
                    obj.executed[day] = issued
            archived = {item['fingerprint'] for event in events for item in event['active'].values()}
            issued = {r[0] for r in obj.sandbox.db.execute('SELECT fingerprint FROM sandbox_issues')}
            if issued-archived:
                raise ValueError('Unarchived publication requires explicit recovery')
            obj.sync_completions(as_of)
            return obj
        except Exception:
            obj.sandbox.close()
            raise

    def sync_completions(self, as_of):
        """Refresh the in-memory view from the one durable completion owner."""
        if as_of < self.evidence_as_of:
            raise ValueError('Cannot backdate an already advanced execution view')
        from .seed_reconciliation import apply as apply_seed_revisions
        apply_seed_revisions(self,as_of)
        from run_analysis.uploaded_evidence import imported_feedback_response
        comparison_by_key={json.loads(row['actual'])['key']:json.loads(row['comparison'])
                           for row in self.sandbox.completion_rows(as_of)}
        for session_key, actual in self.sandbox.completed_accounted(as_of):
            old = next((r for r in self.history if r.key == actual.key), None)
            if old is not None and old != actual:
                if not self.sandbox.superseded(old):
                    raise ValueError('In-memory and durable physical evidence disagree')
                self.history[self.history.index(old)]=actual
            if old is None:
                if any(r.start == actual.start and r.end == actual.end for r in self.history):
                    raise ValueError('Physical completion already exists under another identity')
                self.history.append(actual)
            response=imported_feedback_response(comparison_by_key[actual.key],actual)
            if response is not None:
                captured=next((r for r in self.responses if r.key==actual.key
                    and r.source.startswith('captured app feedback;')),None)
                # The app resolver may include a newer manual override or
                # cross-source conflict. Resyncing the same physical completion
                # must not replace that source-resolved report with FIT alone.
                if captured is None:
                    self.responses=tuple(r for r in self.responses if r.key!=actual.key)+(response,)
            if session_key.startswith('unplanned:'):
                # Actual load is retained without inventing an issued target,
                # declaring compliance or consuming an unrelated prescription.
                continue
            # Association is the issued slot, not the actual execution date.
            day = session_key.removeprefix('slot:')
            chosen = self.active.pop(day, None)
            fingerprint = self.sandbox.db.execute('SELECT fingerprint FROM sandbox_completions WHERE session_key=?',
                                                  (session_key,)).fetchone()[0]
            if chosen is None or chosen['fingerprint'] != fingerprint:
                chosen = next((e['active'][day] for e in self.events if day in e['active']
                               and e['active'][day]['fingerprint'] == fingerprint), None)
            if chosen is None:
                raise ValueError('Consumed issuance is not owned by an archived decision')
            self.executed[day] = chosen
            self.previous = tuple(p for p in self.previous if p[0] != day)
        # A later duplicate/query is not new evidence and must not move the
        # physical transition clock. First availability belongs to the stored run.
        self.evidence_as_of = max((self.evidence_as_of, *(r.available_at for r in self.history)))
        if getattr(self,'coaching_base',None) is not None:
            from .coached_runtime import sync
            sync(self,as_of)

    def __init__(self,root,config,*,origin,catalog,history,phase_history,scales,
                 training_policy,search_policy,demand_scenario,exchange,timing,
                 coaching_contracts,easy_duration_policy,decision_source,
                 restrictions=(),current_context=None,responses=(),stage=None,target_history=(),
                 capacity_eligibility=None,frequency_contract=None,seed_review_entries=(),quality_runtime=None):
        catalog,history,phase_history = tuple(catalog),tuple(history),tuple(phase_history)
        if search_policy.volume_linear_coefficient and (training_policy.purpose_weights[1]!=0
                or training_policy.volume_window_days is None or search_policy.steady_continuation):
            raise ValueError('Linear volume benefit requires one rolling-volume objective')
        if origin.tzinfo is None or len({d.key for d in catalog})!=len(catalog) or not catalog:
            raise ValueError('Aware origin and unique explicit catalogue required')
        if any(r.end>=origin or r.available_at>origin for r in history):
            raise ValueError('Seed physical history must be available before origin')
        if any(h.end>=origin or h.available_at>origin for h in phase_history):
            raise ValueError('Seed phase history must be available before origin')
        capacity_eligibility=dict(capacity_eligibility or {})
        if set(capacity_eligibility)-{r.key for r in history} or any(
                type(v) is not bool for v in capacity_eligibility.values()):
            raise ValueError('Capacity eligibility must identify seed physical history with boolean flags')
        self.sandbox = ExecutionUploadSandbox(root,config,create=True)
        self.config,self.origin,self.policy = config,origin,search_policy
        self.ref = self.sandbox.reference
        self.zone = ZoneInfo(config['timezone_default'])
        self.catalog = {d.key:d for d in catalog}
        self.quality_runtime=quality_runtime
        self.history,self.phase_history = list(history),tuple(phase_history)
        from .review_seed import validate_seed
        self.seed_review_entries=validate_seed(seed_review_entries,history,origin)
        self.scales,self.training_policy = scales,training_policy
        self.demand_scenario,self.exchange,self.timing = demand_scenario,exchange,timing
        self.coaching_contracts = dict(coaching_contracts)
        if set(self.coaching_contracts) != set(self.catalog) or not decision_source:
            raise ValueError('Every declared catalogue dose needs a pre-issued coaching contract and source')
        self.easy_duration_policy,self.decision_source = easy_duration_policy,decision_source
        self.stage,self.target_history=stage,tuple(target_history)
        self.general_progression_state=None
        self.volume_shortfall_since=None
        if frequency_contract is not None and (not isinstance(frequency_contract, FrequencyContract)
                or frequency_contract.adopted_at > origin):
            raise ValueError('Available typed frequency contract required')
        self.frequency_contract=frequency_contract
        self.planner_preferences=None
        self.race_goal=None
        maintenance_path=bool(stage and self.target_history and
            stage.source=='unconfirmed_checkpoint_maintenance_v1' and
            self.target_history[-1].source==stage.source and
            self.target_history[-1].volume_per_7==stage.weekly_equivalent_miles and
            not self.target_history[-1].ramp_days)
        if search_policy.prospective_review_hour is not None and (
                stage is None or stage.policy.volume_path!='gradual' or not self.target_history
                or not (self.target_history[-1].ramp_days or maintenance_path) or stage.policy.review_days>search_policy.hard_days):
            raise ValueError('Prospective review trial requires a bounded gradual phase inside the hard horizon')
        if stage is not None and (stage.issued_at>origin or
                (stage.weekly_equivalent_miles,stage.ordinary_miles,stage.long_miles)!=
                (training_policy.weekly_miles,training_policy.ordinary_miles,training_policy.long_miles)):
            raise ValueError('Accepted stage and compiled training targets must agree')
        self.current_context,self.responses = current_context,tuple(responses)
        self.capacity_eligibility=capacity_eligibility
        self.restrictions = tuple(restrictions)
        self.previous = ()
        self.events = []
        self.publication = 0
        self.last_as_of = None
        self.evidence_as_of = origin
        self.active = {}
        self.executed = {}
        self.issued_by_day = {}
        self.seed_keys = {r.key for r in history}
        with (self.sandbox.root/'scenario.json').open('x') as stream:
            stream.write(dump(dict(origin=origin,search_policy=asdict(search_policy),
                training_policy=asdict(training_policy),scales=scales,catalog=[asdict(d) for d in catalog],
                demand_scenario=asdict(demand_scenario),exchange=exchange,timing=asdict(timing),
                history=[asdict(r) for r in history],phase_history=[asdict(h) for h in phase_history],
                seed_review_entries=[asdict(r) for r in self.seed_review_entries],
                restrictions=[asdict(r) for r in restrictions],
                coaching_contracts={key:asdict(value) for key,value in self.coaching_contracts.items()},
                easy_duration_policy=asdict(easy_duration_policy),decision_source=decision_source,
                current_context=asdict(current_context) if current_context else None,
                responses=[asdict(r) for r in self.responses],
                capacity_eligibility=self.capacity_eligibility,
                stage=asdict(stage) if stage else None,target_history=[asdict(r) for r in self.target_history],
                frequency_contract=asdict(self.frequency_contract) if self.frequency_contract else None,
                **({'quality_runtime':quality_runtime} if quality_runtime is not None else {}),
                measurement_context=self.sandbox.context,accounting_key=self.ref.accounting_key)))

    def decision(self,as_of):
        today = as_of.astimezone(self.zone).date()
        return today+timedelta(days=1) if any(r.start.astimezone(self.zone).date()==today for r in self.history) else today

    def session_evidence(self):
        """One representation for snapshot, restart and stage construction."""
        from .race_event import physical_keys
        race_keys=physical_keys(self)
        return tuple(SessionEvidence(r.key,r.start,r.end,r.available_at,
            'intervals' if r.role=='quality' else r.role,r.miles*1609.344,
            r.demand[0].low*60,r.demand[1],r.accounting_key,
            capacity_eligible=self.capacity_eligibility.get(r.key,True) and
                r.key not in race_keys) for r in self.history)

    def context(self,as_of):
        day = self.decision(as_of)
        origin = self.origin.astimezone(self.zone).date()
        start=max(self.stage.issued_at.astimezone(self.zone).date(),self.target_history[-1].effective_on) if self.stage and self.target_history else origin
        if self.stage and self.stage.policy.volume_path=='gradual':
            # Endpoint confirmation must not reset the objective's accounting origin.
            start=self.target_history[-1].effective_on
        if getattr(self,'general_progression_state',None):
            start=date.fromisoformat(self.general_progression_state['accounting_start'])
        boundary = day if self.policy.horizon=='receding' else origin
        from math import ceil
        smoothing=ceil(max(7/self.training_policy.long_per_7,7/self.training_policy.quality_per_7)) if self.policy.delivery_contract=='sustained_v1' else 1
        full_days=max(self.policy.hard_days,self.policy.continuation_guard_days+
            (self.training_policy.volume_window_days or 1)+smoothing-2) if self.policy.continuation_guard_days else self.policy.hard_days
        end=boundary+timedelta(days=full_days+self.policy.tail_days)
        volume,support=target_segments(self.target_history,as_of=as_of,
            start=origin-timedelta(days=(self.training_policy.volume_window_days or 1)+smoothing-2),end=end) if self.target_history else ((),())
        from .race_goal import overlay_applications, application_history
        race = getattr(self, 'race_goal', None)
        applications = application_history(self, as_of)
        resolution = getattr(self, 'race_event', None)
        if applications:
            volume = overlay_applications(volume,applications,timezone=str(self.zone),resolution=resolution,
                resolutions=getattr(self,'race_event_history',()))
            support = overlay_applications(support,applications,timezone=str(self.zone),resolution=resolution,
                resolutions=getattr(self,'race_event_history',()))
        since=None;foregone=()
        if self.policy.volume_shortfall_policy:
            from .foregone_volume import from_reviews
            from .adaptive_stage_lifecycle import review_entries
            since=getattr(self,'volume_shortfall_since',None) or as_of.astimezone(self.zone).date()
            # Expired kernel support cannot affect any candidate. Do not carry
            # years of old misses through every future scoring fingerprint.
            cutoff=max(since,day-timedelta(days=(self.training_policy.volume_window_days or 1)+smoothing-2))
            foregone=from_reviews(review_entries(self,as_of),since=cutoff,today=as_of.astimezone(self.zone).date())
        return ProgramContext((),start,day,boundary+timedelta(days=self.policy.hard_days),
            end,
            self.scales,self.training_policy,self.demand_scenario,self.exchange,self.timing,
            demand_accounting_key=self.ref.accounting_key,easy_duration_policy=self.easy_duration_policy,
            volume_targets=volume,support_targets=support,soft_horizon_taper=self.policy.soft_horizon_taper,
            delivery_contract=self.policy.delivery_contract,continuation_guard_days=self.policy.continuation_guard_days,
            volume_linear_coefficient=self.policy.volume_linear_coefficient,race_goal=race,
            volume_shortfall_policy=self.policy.volume_shortfall_policy,
            volume_shortfall_since=since,foregone_volume=foregone,
            reported_distance=getattr(self,'reported_distance',None),
            race_goal_history=applications,race_event=resolution,
            race_event_history=getattr(self,'race_event_history',()))

    def intent(self,dose,day,as_of,session_key):
        margin = self.policy.phase_fraction_margin
        if dose.forecast_scope and dose.forecast_scope.intensity_key != self.coaching_contracts[dose.key].intensity_key:
            raise ValueError('Forecast scope must match the issued coaching effort')
        return IntendedExecution(session_key,as_of,datetime.combine(day,time(18),self.zone),
            tuple(IntendedPhase(k,TargetBand(s,lo,hi,'declared candidate duration scenario'))
                  for (k,s),(lo,hi) in zip(dose.phases,dose.phase_scenarios or
                      tuple((s*(1-margin),s*(1+margin)) for _,s in dose.phases))),
            TargetBand(dose.miles,max(.1,dose.miles-self.policy.distance_margin),
                       dose.miles+self.policy.distance_margin,'declared harness distance margin'),
            self.ref.accounting_key,self.sandbox.context,
            LoadAllowance(self.policy.response_allowance,self.policy.response_allowance,'engineering scenario allowance')
            if self.policy.response_allowance is not None else None,
            dose.forecast_scope,
            dose.workout_definition.get('numeric_contract') if dose.workout_definition else None,
            getattr(self,'quality_decision_evidence',None) if dose.role=='quality'
                and getattr(self,'quality_runtime',None) is not None else None)

    def replan(self,as_of,*,reason,publish=True,require_stage_funding=False,prepared=None,commit_stage=None,replayed_selection=None,coaching_first_purpose=None):
        persisted_coaching=getattr(self,'coaching_base',None) is not None
        if persisted_coaching:
            from .coached_runtime import offline_guard
            offline_guard(self)
            if replayed_selection is not None:
                raise ValueError('Coached stage transition/reuse requires its own scoring contract')
        if coaching_first_purpose is not None and (publish or prepared is not None or
                replayed_selection is not None or require_stage_funding or
                getattr(self,'quality_runtime',None) is None):
            raise ValueError('Coached opening requires an unpublished native-catalogue sandbox decision')
        if prepared is not None and replayed_selection is not None:
            raise ValueError('Prepared stage preview and archived search reuse are distinct paths')
        if require_stage_funding and self.policy.delivery_contract=='sustained_v1':
            raise ValueError('Sustained delivery uses ordinary admission for every decision')
        if self.last_as_of is not None and as_of<=self.last_as_of:
            raise ValueError('Decision event timestamps must advance')
        self.sync_completions(as_of)
        if persisted_coaching:
            from .coached_runtime import revise_preferences
            revise_preferences(self,as_of)
        quality_signals=None
        if getattr(self,'quality_runtime',None) is not None:
            from .quality_catalogue_runtime import decision_evidence
            self.quality_decision_evidence,quality_signals=decision_evidence(self,as_of)
        context = self.context(as_of)
        dates = tuple((context.decision+timedelta(days=i)).isoformat()
                      for i in range((context.end-context.decision).days))
        history = (*self.phase_history,*self.sandbox.completed_history(as_of))
        state = build_snapshot(as_of=as_of,timezone=self.config['timezone_default'],context=context,
            stage_key=self.stage.key if self.stage else 'harness-stage:'+self.origin.isoformat(),
            stage_issued_at=self.stage.issued_at if self.stage else self.origin,
            source=self.decision_source,history=self.history,phase_history=history,
            contracts=self.coaching_contracts,reference=self.ref,resting_hr=self.config['resting_hr'],
            maximum_hr=self.config['max_hr'],references_available_at=self.origin,
            references_source='sandbox frozen settings',measurement_context=self.sandbox.context,
            contexts=() if self.current_context is None else (self.current_context,),
            session_evidence=self.session_evidence(),
            responses=self.responses,restrictions=self.restrictions)
        from .race_event import preparation_active
        race_build = not preparation_active(context,context.decision) or context.race_goal.phase(context.decision)=='build'
        if (self.policy.progression_admission == 'checkpoint_v1'
                and self.policy.prospective_review_hour is not None and race_build
                and self.target_history[-1].ramp_days):
            from .prospective_progression import freeze_volume_review
            from .adaptive_stage_lifecycle import review_entries
            # The original dated phase owns the checkpoint. Stage confirmation,
            # upload, refresh and restart must not move it to a new horizon.
            path=self.target_history[-1]
            phase=replace(self.stage,issued_at=path.review_anchor or path.available_at)
            checkpoint=datetime.combine(phase.review_at.astimezone(self.zone).date(),
                time(self.policy.prospective_review_hour),self.zone)
            if checkpoint<phase.review_at:checkpoint+=timedelta(days=1)
            if as_of<checkpoint:
                review=freeze_volume_review(phase,self.target_history,review_entries(self,as_of),as_of,
                    review_at=checkpoint,timezone=self.config['timezone_default'])
                if context.race_goal and context.race_goal.goal:
                    from run_analysis.stage_target_history import target_miles_between
                    review=replace(review,required_miles=target_miles_between(
                        context.volume_targets,review.start,review.end,self.zone))
                state=replace(state,prospective_volume_review=review)
            elif self.stage.weekly_equivalent_miles+1e-8<path.volume_goal:
                # No sliding deadline, unsafe automatic catch-up or silent
                # fallback. Interpretation of a missed checkpoint is separate.
                raise ValueError('Prospective progression checkpoint missed; maintenance interpretation required')
        history = state.phase_history
        snapshot_name = f'snapshot-{len(self.events)+1:04d}.json'
        # Freeze each dose forecast once per decision, reuse across candidate dates.
        predictions,handoffs,unresolved,quality_forecasts = {},{},{},{}
        from .quality_choices import allowed, enabled_types, quality_focus, VERSION as QUALITY_CHOICES_VERSION
        from .live_bridge import enabled
        adaptive_enabled=enabled(self.config)
        for key,dose in self.catalog.items():
            if getattr(self,'quality_runtime',None) is not None and dose.role=='quality':
                from .quality_forecast_runtime import VERSION as QUALITY_RUNTIME_VERSION
                if not dose.workout_definition or dose.workout_definition.get('calendar_integration')!=QUALITY_RUNTIME_VERSION:
                    continue
            if adaptive_enabled and not allowed(dose,self.coaching_contracts[key],self.config):
                continue
            intent = self.intent(dose,context.decision,as_of,f'proposal:{key}')
            record = issue_execution(intent,history)
            load = record.payload['target_forecast']['predicted_load']
            handoff = record.payload['planning_handoff']
            if not handoff['usable']:
                unresolved[key] = handoff
                continue
            predictions[key] = load
            handoffs[key] = handoff
            if dose.role=='quality':quality_forecasts[key]=record.payload['target_forecast']
        if getattr(self,'quality_runtime',None) is not None:
            from .quality_catalogue_runtime import programme
            context=replace(context,quality_programme=programme(self,history,quality_forecasts,as_of,quality_signals))
            state=replace(state,context=context)
        if not handoffs:
            raise ValueError(f'No supported or explicitly bounded dose forecasts: {unresolved}')
        runs = {}
        for day in dates:
            for key,dose in self.catalog.items():
                if key not in handoffs:
                    continue
                start = datetime.combine(datetime.fromisoformat(day).date(),time(18),self.zone)
                runs[(day,key)] = planned_accounted(f'candidate:{day}:{key}',start,dose,handoffs[key],
                    self.ref,available=as_of)
        if self.policy.steady_continuation:
            references=[]
            for role in ('easy','long','quality'):
                candidates=[d for d in self.catalog.values() if d.role==role and d.key in handoffs]
                if not candidates:
                    raise ValueError('Coarse continuation requires supported full-purpose reference forecasts')
                dose=(min(candidates,key=lambda d:(abs(d.miles-self.training_policy.ordinary_miles),d.key))
                    if role=='easy' else max(candidates,key=lambda d:(
                        sum(s for kind,s in d.phases if kind=='work') if role=='quality' else d.miles,d.key)))
                references.append((role,dose.miles,runs[(dates[0],dose.key)].demand))
            context=replace(context,steady_continuation_references=tuple(references))
            state=replace(state,context=context)
        if self.policy.continuation_boundary:
            from .continuation_boundary_trial import reference_keys
            keys=reference_keys(self.training_policy,self.catalog,handoffs)
            context=replace(context,continuation_boundary_references=tuple(
                (role,self.catalog[key].miles,runs[(dates[0],key)].demand) for role,key in keys.items()),
                purpose_continuation=self.policy.purpose_continuation,
                purpose_continuation_phase=self.policy.purpose_continuation_phase,
                continuation_distribution=self.policy.continuation_distribution)
            state=replace(state,context=context)
        if self.frequency_contract is not None:
            state=replace(state,frequency_contract=self.frequency_contract)
        preferences=getattr(self,'planner_preferences',None)
        if preferences is not None:
            if preferences.available_at > as_of:
                raise ValueError('Cannot issue a future planner preference')
            context=replace(context,long_run_weekday=preferences.long_run_weekday,
                long_run_not_before=preferences.long_run_not_before)
            state=replace(state,planner_preferences=preferences,context=context)
        race = getattr(self, 'race_goal', None)
        if race is not None:
            from .race_accounting import event_demand, allowed_dose
            context = replace(context, race_demand=() if context.race_event else
                event_demand(race,context.continuation_boundary_references))
            from .race_goal import application_history
            state = replace(state,context=context,race_goal=race,
                            race_goal_history=application_history(self,as_of))
        def evaluate(program):
            if any(key not in handoffs for _,key in program):
                return dict(cost=1e12,eligible=False,construction_eligible=False,
                    constraints='unsupported_or_disabled_workout')
            if race is not None:
                from .race_accounting import allowed_dose
                if any(not allowed_dose(race,date.fromisoformat(day),self.catalog[key],self.stage,context.race_event)
                       for day,key in program):
                    return dict(cost=1e12,eligible=False,construction_eligible=False,
                        constraints='dated_race_phase_restriction')
            result = evaluate_snapshot(state,tuple(runs[p] for p in program),
                tuple(p[1] for p in program),reference=self.ref,evaluator=evaluate_continuous)
            parts=result['evaluation'].parts
            funding=parts['easy_duration']['supporting_easy_volume']
            funded=(all(parts['delivery'][p]+1e-8>=parts['targets'][p] for p in ('volume','long','quality'))
                and funding is not None and funding['delivered_miles']+1e-8>=funding['target_miles'])
            return dict(cost=result['evaluation'].total_cost,
                eligible=result['decision_eligible'] and (funded or not require_stage_funding),
                construction_eligible=result.get('construction_eligible',result['decision_eligible']),
                constraints=result['constraints']['status'],
                recovery_feasibility=result['constraints']['recovery_feasibility'],
                suitability=result['suitability'],
                score_fingerprint=result['evaluation'].fingerprint,
                model_version=result['evaluation'].model_version,
                **({'frequency_range':result['frequency_range'],
                    'frequency_preference':result['frequency_preference']} if 'frequency_range' in result else {}),
                **({'progression_delivery':result['progression_delivery']} if 'progression_delivery' in result else {}))
        available_prior = tuple(p for p in self.previous if p[1] in handoffs)
        long_dates=None
        if preferences is not None and preferences.long_run_weekday is not None:
            from .planner_preferences import long_day_allowed
            long_dates=frozenset(day for day in dates if long_day_allowed(preferences,date.fromisoformat(day)))
        from .calendar_constraint_repair import role_date_domains
        allowed_search_dates=role_date_domains(dates,
            {k:self.catalog[k].role for k in handoffs},restrictions=state.restrictions,
            as_of=as_of,long_dates=long_dates)
        if race is not None:
            available_prior = tuple((day,key) for day,key in available_prior
                if allowed_dose(race,date.fromisoformat(day),self.catalog[key],self.stage,context.race_event))
        coaching=None
        compiler_state=state
        program_transform=None
        if coaching_first_purpose is not None or persisted_coaching:
            from .quality_coaching_policy import freeze_sequence
            from .coached_pipeline import coached_evaluator, fund_bound_prior
            sequence=self.coaching_sequence if persisted_coaching else freeze_sequence({k:self.catalog[k] for k in handoffs},stage_key=state.stage_key,
                issued_at=as_of.isoformat(),ordinary_miles=self.training_policy.ordinary_miles,
                quality_work_minutes=self.training_policy.quality_work_minutes,
                first_purpose=coaching_first_purpose, focus=quality_focus(self.config))
            receipts=getattr(self,'coaching_receipts',()) if persisted_coaching else ()
            case=dict(state=state,catalog=self.catalog,dates=dates,physical=runs,stage=self.stage)
            engine=coached_evaluator(case,sequence,completions=receipts)
            state=replace(state,quality_coaching=sequence,quality_coaching_reference=engine['quality_reference'],
                quality_completions=receipts)
            engine=coached_evaluator(dict(case,state=state),sequence,completions=receipts)
            evaluate=engine['evaluate']
            program_transform=engine['bind']
            compiler_state=engine['effective_state']
            # Bind the ACTUAL live prior before filtering obsolete quality keys.
            # The older generic quality keys intentionally remain in the catalogue.
            available_prior=program_transform(tuple(p for p in self.previous if p[0] in dates))
            permitted={r.key for r in sequence.ordinary}
            handoffs={k:v for k,v in handoffs.items() if self.catalog[k].role!='quality' or k in permitted}
            available_prior,repair=fund_bound_prior(available_prior,dates=dates,
                catalog={k:self.catalog[k] for k in handoffs},physical=runs,state=state,evaluate=evaluate)
            coaching=dict(sequence=sequence,engine=engine,prior_funding=repair)
        seed_policy=self.training_policy
        seed_support=self.easy_duration_policy.supporting_easy_miles_per_7
        if self.stage and self.stage.policy.volume_path=='gradual' and self.target_history[-1].ramp_days:
            # Candidate generation sees the endpoint from the first search too;
            # confirmation must not suddenly reveal a richer seed population.
            seed_policy=replace(seed_policy,weekly_miles=self.target_history[-1].volume_goal)
            seed_support=self.target_history[-1].support_goal
        if self.policy.general_progression in {'continuous_general_v1','continuous_general_v2','continuous_general_v3','continuous_general_v4','continuous_general_v5','continuous_general_v6','continuous_general_v7'}:
            path=self.target_history[-1]
            seed_policy=replace(seed_policy,weekly_miles=path.rate(context.end-timedelta(days=1),'volume_per_7'))
            seed_support=path.rate(context.end-timedelta(days=1),'supporting_easy_per_7')
        easy_keys=[k for k in handoffs if self.catalog[k].role=='easy']
        seed_keys=set()
        for target in (self.training_policy.ordinary_miles,self.training_policy.ordinary_miles/2):
            if easy_keys:
                seed_keys.add(min(easy_keys,key=lambda k:(abs(self.catalog[k].miles-target),k)))
        for role in ('long','quality'):
            options=[k for k in handoffs if self.catalog[k].role==role]
            if options:
                seed_keys.add(max(options,key=lambda k:(sum(s for kind,s in self.catalog[k].phases if kind=='work')
                    if role=='quality' else self.catalog[k].miles,k)))
        if prepared is not None:
            if state.fingerprint!=prepared['state'].fingerprint:
                raise ValueError('Prepared stage decision evidence changed; search again')
            result=dict(prepared['search'],program=prepared['program'])
            if evaluate(prepared['program'])!=result['selected']:
                raise ValueError('Prepared winner differs from revalidated physical candidate')
        elif replayed_selection is not None:
            # Reuse a previously computed whole-plan search only with exact
            # evidence identity, cost, physical admission and funding parity.
            if state.fingerprint!=replayed_selection['snapshot_fingerprint']:
                raise ValueError('Archived coordinated decision differs from the harness evidence')
            program=signature(tuple(p) for p in replayed_selection['program'])
            checked=evaluate(program)
            if not checked['eligible'] or abs(checked['cost']-replayed_selection['cost'])>1e-12:
                raise ValueError('Archived selection failed independent harness revalidation')
            result=dict(program=program,selected=checked,cost=checked['cost'],seconds=0.,evaluations=0,
                local_neighborhood_complete=replayed_selection['local_neighborhood_complete'],
                budget_exhausted=replayed_selection['budget_exhausted'],search='exact_archived_search_reuse',
                archived_search_reuse=dict(original_seconds=replayed_selection['seconds'],
                    original_evaluations=replayed_selection['evaluations'],
                    manifest=replayed_selection['manifest']))
        else:
            from run_analysis.adaptive.decimal_dose_search import calendar_representatives
            seeds = complete_stage_seeds(dates,{k:self.catalog[k] for k in handoffs},seed_policy,
                supporting_miles=seed_support,
                **({'long_dates':long_dates} if long_dates is not None else {}),
                limit=max(1,min(1680,self.policy.evaluations//5)))
            if race is not None:
                seeds = tuple(tuple((day,key) for day,key in seed
                    if allowed_dose(race,date.fromisoformat(day),self.catalog[key],self.stage,context.race_event)) for seed in seeds)
            result = migrated_search(dates=dates,keys=tuple(sorted(handoffs)),previous=available_prior,
            exploration_keys=calendar_representatives({k:self.catalog[k] for k in handoffs},
                ordinary=self.training_policy.ordinary_miles,
                support=self.easy_duration_policy.supporting_easy_miles_per_7),
            dose_miles={k:self.catalog[k].miles for k in handoffs},
            evaluate=evaluate,budget=self.policy.evaluations,hard_end=context.hard_end.isoformat(),
            roles={k:self.catalog[k].role for k in handoffs},stability_tolerance=self.policy.stability_tolerance,
            seed_keys=tuple(sorted(seed_keys)),
            previous_end=self.events[-1]['end'] if self.events else None,
            recombination_dates=(context.taper_start.isoformat(),),
            seed_programs=seeds,complete_first=self.policy.complete_first_search,
            # The native check must see the mileage that will actually publish,
            # after the compiled allocation and merge refiners below. Reuse its
            # existing allowance there instead of checking a stale pre-polish plan.
            final_winner_budget=(0 if getattr(self,'quality_runtime',None) is not None
                else self.policy.final_winner_evaluations),
            fair_coordinates=getattr(self,'quality_runtime',None) is not None,
            program_transform=program_transform,
            **({'allowed_role_dates':allowed_search_dates} if allowed_search_dates else {}))
            from run_analysis.planner_observation import phase
            phase('refine')
            if self.policy.coordinated_final_evaluations:
                from .compiled_calendar_doses import compile_calendar, refine
                stamp = perf_counter()
                original = tuple(result['program'])
                # Compile the actual full-score ledger, not an approximation to HR.
                raw = (coaching['engine']['detail'](original) if coaching else
                    evaluate_snapshot(state,tuple(runs[p] for p in original),
                        tuple(k for _,k in original),reference=self.ref,evaluator=evaluate_continuous))
                compiled = compile_calendar(compiler_state,original,physical=runs,result=raw)
                checked_cache = {original: result['selected']}
                full_calls = 1  # The ledger compilation above is an additional full evaluation.
                def checked(program):
                    nonlocal full_calls
                    if program not in checked_cache:
                        checked_cache[program] = evaluate(program)
                        full_calls += 1
                    return checked_cache[program]
                polished = refine(compiled,checked,
                    max_full_evaluations=self.policy.coordinated_final_evaluations)
                winner = polished.pop('program')
                elapsed = perf_counter()-stamp
                changed = winner != original
                result.update(program=winner,selected=checked_cache[winner],cost=polished['cost'],
                    seconds=result['seconds']+elapsed,evaluations=result['evaluations']+full_calls,
                    budget_exhausted=result.get('budget_exhausted',False) or polished['budget_exhausted'])
                result['refinement'] = dict(result.get('refinement') or {},coordinated_final_doses=dict(
                    polished,seconds=elapsed,additional_full_evaluations=full_calls,
                    changed=changed,configured_allowance=self.policy.coordinated_final_evaluations))
                if changed:
                    # The old calendar neighbourhood was checked at different doses.
                    result['local_neighborhood_complete'] = False
            if self.policy.calendar_merge_evaluations:
                from .calendar_merge_refinement import refine_merges
                def details(program):
                    if coaching:
                        return coaching['engine']['detail'](program)
                    raw = evaluate_snapshot(state,tuple(runs[p] for p in program),
                        tuple(k for _,k in program),reference=self.ref,evaluator=evaluate_continuous)
                    parts=raw['evaluation'].parts
                    funding=parts['easy_duration']['supporting_easy_volume']
                    funded=(all(parts['delivery'][p]+1e-8>=parts['targets'][p] for p in ('volume','long','quality'))
                        and funding is not None and funding['delivered_miles']+1e-8>=funding['target_miles'])
                    raw['decision_eligible'] = raw['decision_eligible'] and (funded or not require_stage_funding)
                    return raw
                joint=refine_merges(compiler_state,result['program'],physical=runs,
                    evaluate_details=details,max_full_evaluations=self.policy.calendar_merge_evaluations)
                winner=joint.pop('program')
                result.update(program=winner,selected=evaluate(winner),cost=joint['cost'],
                    seconds=result['seconds']+joint['seconds'],
                    evaluations=result['evaluations']+joint['full_evaluations']+1)
                result['refinement']=dict(result.get('refinement') or {},calendar_merge_doses=joint)
                if joint['changed']:
                    result['local_neighborhood_complete']=False
            if getattr(self,'quality_runtime',None) is not None and self.policy.final_winner_evaluations:
                from .native_quality_final_refinement import refine as refine_native_final
                native=refine_native_final(result['program'],result['selected'],
                    evaluate=evaluate,dates=dates,roles={k:self.catalog[k].role for k in handoffs},
                    miles={k:self.catalog[k].miles for k in handoffs},
                    budget=self.policy.final_winner_evaluations,program_transform=program_transform)
                winner,selected=native.pop('program'),native.pop('selected')
                result.update(program=winner,selected=selected,cost=native['cost'],
                    seconds=result['seconds']+native['seconds'],
                    evaluations=result['evaluations']+native['evaluations'],
                    budget_exhausted=result.get('budget_exhausted',False) or native['budget_exhausted'])
                result['refinement']=dict(result.get('refinement') or {},native_terminal_doses=native)
                if native['changed']:
                    result['local_neighborhood_complete']=False
        program = result.pop('program')
        if not publish:
            return dict(state=state,program=program,planned=tuple(runs[p] for p in program),
                contract_keys=tuple(k for _,k in program),search=result,
                requires_stage_funding=require_stage_funding,
                **({'coaching':coaching,'physical':runs,'handoffs':handoffs} if coaching else {}))
        save_snapshot(self.sandbox.root/snapshot_name,state)
        edits = churn(self.previous,program,context.decision,self.catalog) if self.events else {'4':[],'7':[]}
        requests=[]
        # Archive only issued hard prescriptions; the soft tail is not a promise.
        for day,key in program:
            if day>=context.hard_end.isoformat():
                continue
            session_key = f'slot:{day}'
            intent = self.intent(self.catalog[key],datetime.fromisoformat(day).date(),as_of,session_key)
            parent = self.issued_by_day.get(day)
            requests.append((day,key,intent,parent))
        event = dict(index=len(self.events),as_of=as_of.isoformat(),reason=reason,decision=context.decision.isoformat(),
            model_version=(PHASE_CONTINUATION_MODEL_VERSION if context.purpose_continuation_phase=='known_full_phase_v1' else
                PURPOSE_CONTINUATION_MODEL_VERSION if context.purpose_continuation else
                VOLUME_BENEFIT_MODEL_VERSION if context.volume_linear_coefficient else
                SINGLE_VOLUME_MODEL_VERSION if context.policy.purpose_weights[1]==0 else
                (MEMORY_CONTINUATION_MODEL_VERSION if context.continuation_guard_days else
                 CONTINUATION_MODEL_VERSION) if context.continuation_boundary_references else MODEL_VERSION),
            snapshot_file=snapshot_name,snapshot_fingerprint=state.fingerprint,
            demonstrated_exposure=state.demonstrated(),
            hard_end=context.hard_end.isoformat(),end=context.end.isoformat(),program=program,churn=edits,
            **result,publication=self.publication+1,active={},
            stage_runtime=dict(stage=asdict(self.stage) if self.stage else None,
                target_history=[asdict(r) for r in self.target_history],catalog=[asdict(d) for d in self.catalog.values()],
                **({'general_progression_state':self.general_progression_state}
                    if getattr(self,'general_progression_state',None) else {}),
                **({'quality_runtime':self.quality_runtime} if getattr(self,'quality_runtime',None) is not None else {}),
                **({'coaching_base':asdict(self.coaching_base)} if persisted_coaching else {}),
                **({'coaching_stage_keys':self.coaching_stage_keys} if persisted_coaching else {}),
                **({'coaching_epochs':[asdict(e) for e in self.coaching_epochs]}
                    if persisted_coaching and len(getattr(self,'coaching_epochs',()))>1 else {}),
                **({'frequency_contract':asdict(self.frequency_contract)} if self.frequency_contract else {})),
            history_keys=[r.key for r in self.history],
            history_cardio_bounds=[[r.key,r.demand[1].low,r.demand[1].high] for r in self.history],
            forecast_load_by_dose=predictions,
            forecast_handoff_by_dose=handoffs,unresolved_forecasts=unresolved,
            **({'quality_preferences':dict(version=QUALITY_CHOICES_VERSION,enabled=enabled_types(self.config),
                focus=quality_focus(self.config))}
                if adaptive_enabled else {}),
            forecast_coverage='partial_catalogue' if unresolved else 'complete_catalogue',
            prior_slots_without_forecast=[p for p in self.previous if p[1] not in handoffs],
            planned_training=training_metrics([runs[p] for p in program],self.history,
                context.decision,(context.hard_end-context.decision).days,self.zone),
            volume=sum(self.catalog[k].miles for d,k in program if d<context.hard_end.isoformat()))
        if coaching:
            from .quality_assignment_accounting import issue_assignment,assignment_json
            rank=0;assignments={}
            for day,key in program:
                if day>=context.hard_end.isoformat():continue
                if self.catalog[key].role=='quality':
                    assignments[day]=assignment_json(issue_assignment(coaching['sequence'],rank,self.catalog))
                    rank+=1
            event['coached_assignments']=assignments
            event['coached_prior_funding']=coaching['prior_funding']
        if race is not None:
            from .race_goal import preparation_status
            event['race_preparation']=preparation_status(self,context.decision,as_of=as_of)
        event['model_version']=result['selected']['model_version']
        if self.policy.progression_admission == 'observed_rolling_v2':
            # Dated rolling-volume/continuation losses still price ALL actual
            # and planned miles. Optional growth is reviewed from completions
            # by advance_stage; it is not a deadline for ordinary admission.
            event['selection_contract'] = 'observed_rolling_progression_v2'
            if self.policy.progression_credit != 'issued_nominal_v1':
                event['progression_credit'] = self.policy.progression_credit
        elif self.policy.prospective_review_hour is not None:
            event['selection_contract']='prospective_volume_review_v1'
        if getattr(self,'general_progression_state',None):
            event['progression_target']=dict(mode='continuous_observed_volume',
                planning_volume_per_7=self.general_progression_state['volume'],
                growth_per_7=self.target_history[-1].growth_per_7,
                conditional_destination=self.target_history[-1].growth_limit,
                phase_handoff=False,measured_fitness=False)
            if self.general_progression_state['version'] in {'continuous_general_v4','continuous_general_v5','continuous_general_v6','continuous_general_v7'}:
                # Keep legacy mode compatibility while making the three
                # different meanings explicit in the issued decision receipt.
                event['progression_target'].update(planning_volume_is_conditional=True,
                    baseline_reference_per_7=self.general_progression_state['progression_reference']['baseline_volume'],
                    observed_volume_per_7=self.general_progression_state['observed_volume_per_7'])
            if self.general_progression_state['version'] in {'continuous_general_v5','continuous_general_v6','continuous_general_v7'}:
                event['progression_target'].update(
                    rate_contract=self.general_progression_state['rate_contract'],
                    rate_status=self.general_progression_state['rate_status'])
            if self.general_progression_state['version']=='continuous_general_v6':
                event['progression_target'].update(
                    execution_fulfillment=self.general_progression_state['execution_fulfillment'],
                    planner_funding=self.general_progression_state['planner_funding'])
            if self.general_progression_state['version']=='continuous_general_v7':
                event['progression_target'].update(
                    reference_signal=self.general_progression_state['reference_signal'],
                    reference_provisional=self.general_progression_state['reference_provisional'],
                    reference_controller=self.general_progression_state['reference_controller'],
                    deferred_reference_volume=self.general_progression_state['deferred_reference_volume'])
        elif self.stage and self.stage.policy.volume_path=='gradual':
            path=self.target_history[-1]
            event['progression_target']=dict(mode='bounded_gradual_volume',
                anchor=path.effective_on.isoformat(),
                endpoint=(path.effective_on+timedelta(days=path.ramp_days)).isoformat(),
                planning_volume_per_7=path.rate(context.decision,'volume_per_7'),
                planning_support_per_7=path.rate(context.decision,'supporting_easy_per_7'),
                phase_goal=path.volume_goal if path.ramp_days else path.volume_per_7,
                reviewed_stage_volume=self.stage.weekly_equivalent_miles,
                measured_fitness=False)
        if self.policy.general_progression=='continuous_general_v7':
            from .realized_reference_runtime import allocate
            config=self.general_progression_state['realized_reference']
            event['realized_reference_allocation']=allocate(compiler_state,program,runs,config,
                reference=self.ref,max_full_evaluations=64,prior_events=self.events)
            event['reference_compute_seconds']=event['realized_reference_allocation']['seconds']
            event['search_plus_reference_seconds']=event['seconds']+event['reference_compute_seconds']
        event=self.sandbox.publish_plan(requests,self.phase_history,event,serialize(state),commit_stage=commit_stage)
        active=event['active']
        self.publication=event['publication']
        self.issued_by_day.update({day:issued['fingerprint'] for day,issued in active.items()})
        self.events.append(event)
        # Runtime removal must not mutate the already-archived event dictionary.
        self.active,self.previous,self.last_as_of = dict(active),program,as_of
        self.race_goal_history=state.race_goal_history
        self.volume_shortfall_since=context.volume_shortfall_since
        self.save()
        return event

    def upload(self,payload,*,day,available_at,observed_role=None,observed_phases=None,correction=None):
        """Actual role/phase evidence is supplied by execution driver, not HR/title.

        Real parsed movement and HR are preserved. Active publication explicitly
        associates the upload; superseded publications remain audit records only.
        """
        automatic=observed_role is None and observed_phases is None
        if not automatic and (observed_phases is None or observed_role not in {'easy','long','quality','unknown'}):
            raise ValueError('Harness driver must supply explicit physical evidence; unresolved execution cannot be fabricated')
        if available_at < self.evidence_as_of:
            raise ValueError('Upload availability cannot backdate the execution view')
        _,activity=self.sandbox.prepare_upload(payload)
        if day is None:
            from run_analysis.run_feedback import _long_run_threshold
            threshold=_long_run_threshold([dict(workout_type=r.role,total_distance_m=r.miles*1609.344)
                for r in self.history if r.available_at<=available_at])
            result=self.sandbox.upload(payload,fingerprint=None,available_at=available_at,
                resolve_evidence=True,long_threshold_miles=threshold,correction=correction)
            self.sync_completions(available_at)
            return result
        consumed=next((r for r in self.sandbox.completion_rows(available_at) if r['session_key']=='slot:'+day),None)
        if consumed is not None:
            # Retry/correction stays associated with the original issue, even
            # after many later calendars have retired that slot.
            chosen=next((e['active'][day] for e in self.events if day in e['active']
                and e['active'][day]['fingerprint']==consumed['fingerprint']),None)
        else:
            known=[e for e in self.events if datetime.fromisoformat(e['as_of'])<=activity.start_time_utc]
            chosen=max(known,key=lambda e:datetime.fromisoformat(e['as_of']))['active'].get(day) if known else None
        if chosen is None:
            raise ValueError('No active pre-run issued session for this upload date; explicit unplanned-work association required')
        if automatic:
            from run_analysis.run_feedback import _long_run_threshold
            threshold=_long_run_threshold([dict(workout_type=r.role,total_distance_m=r.miles*1609.344)
                for r in self.history if r.available_at<=available_at])
            result=self.sandbox.upload(payload,fingerprint=chosen['fingerprint'],available_at=available_at,
                resolve_evidence=True,long_threshold_miles=threshold,correction=correction)
        else:
            result = self.sandbox.upload(payload,fingerprint=chosen['fingerprint'],available_at=available_at,
                observed_phases=observed_phases,phase_evidence='explicit',observed_role=observed_role,correction=correction)
        self.sync_completions(available_at)
        return result

    def save(self):
        # Each decision is immutable and records the complete searched programme.
        path = self.sandbox.root/f'decision-{len(self.events):04d}.json'
        with path.open('x') as stream:
            stream.write(dump(self.events[-1]))

    def summary(self,days,uploads,*,delivery_start=None):
        start = delivery_start or self.origin.astimezone(self.zone).date()
        delivered = [r for r in self.history if r.key not in self.seed_keys and
            r.start.astimezone(self.zone).date()>=start]
        lead = [r for r in self.history if r not in delivered]
        # A post-upload failure leaves the current day unfinished, but the run
        # was completed and must still appear in delivered-volume diagnostics.
        delivery_days=max(1,days,max(((r.start.astimezone(self.zone).date()-start).days+1
            for r in delivered),default=0))
        return dict(version='new_model_daily_harness_v1',days=days,events=self.events,uploads=uploads,
            delivered=training_metrics(delivered,lead,start,delivery_days,self.zone),
            delivery_start=start.isoformat(),
            delivery_window_days=delivery_days,delivery_window_includes_unfinished_day=delivery_days>days,
            search_policy=asdict(self.policy),
            training_policy=asdict(self.training_policy),catalog=[asdict(d) for d in self.catalog.values()],
            demand_scenario=asdict(self.demand_scenario),exchange=self.exchange,timing=asdict(self.timing),
            model=f'evaluate_continuous/{self.events[-1].get("model_version",MODEL_VERSION) if self.events else MODEL_VERSION}',live_scheduler_influence=False,
            limitations=['Synthetic engineering scenario, not athlete validation',
                'Direct phase-load model used on supplied numeric catalogue; easy/long generalization unvalidated',
                'Nominal rate plus declared scenarios, not predictive confidence; no calibrated production margins',
                'Recovery feasibility not established; checked constraints are not training clearance',
                'Bounded discrete search; budget exhaustion and catalogue limits can create churn',
                ('Continuous future growth is conditional planning intent; actual delivery and response own daily review'
                    if self.policy.general_progression in {'continuous_general_v1','continuous_general_v2','continuous_general_v3','continuous_general_v4','continuous_general_v5','continuous_general_v6','continuous_general_v7'} else
                    'Bounded gradual volume paths are planning intent; endpoint confirmation requires observed delivery'),
                'Atomic decision recovery and explicit correction are isolated; full app-route acceptance remains open'])


def synthetic_payload(dose,start,*,distance_factor=1.,duration_factor=1.,hr=140):
    """Known execution fixture emitted as TCX, independent of forecast output.

    HR is supplied by the response scenario, NEVER solved from predicted load.
    Explicit phase evidence comes from this driver, not redetected/assumed intent.
    """
    phases = tuple((k,s*duration_factor) for k,s in dose.phases)
    seconds = sum(s for _,s in phases)
    miles = dose.miles*distance_factor
    times = sorted({0.,seconds,*(float(t) for t in range(0,int(seconds),10))})
    points = ''.join(f'<Trackpoint><Time>{(start+timedelta(seconds=t)).isoformat()}</Time>'
        f'<DistanceMeters>{miles*1609.344*t/seconds}</DistanceMeters>'
        +(f'<HeartRateBpm><Value>{hr}</Value></HeartRateBpm>' if hr is not None else '')
        +'</Trackpoint>' for t in times)
    content = (f'<TrainingCenterDatabase><Activities><Activity Sport="Running"><Id>{start.isoformat()}</Id>'
        f'<Lap StartTime="{start.isoformat()}"><TotalTimeSeconds>{seconds}</TotalTimeSeconds>'
        f'<DistanceMeters>{miles*1609.344}</DistanceMeters><Track>{points}</Track></Lap>'
        '</Activity></Activities></TrainingCenterDatabase>').encode()
    return UploadPayload('synthetic-execution.tcx',content),phases,start+timedelta(seconds=seconds)


def demo(root,*,days=3,budget=3000,hard_days=21,tail_days=14,scenario='expected',stop_on_churn=False):
    if days<1 or scenario not in {'expected','higher_cost','lower_edge','upper_edge','missed'}:
        raise ValueError('Explicit supported scenario and positive simulation days required')
    config = load_config('config.yaml')
    config = dict(config,resting_hr=50,max_hr=200,timezone_default='UTC')
    origin = datetime(2026,9,27,tzinfo=timezone.utc)
    from run_analysis.continuous_training_dose import ContinuousDoseReference
    ref = ContinuousDoseReference('sandbox upload accounting')
    quality = (('warmup',720.),('work',300.),('recovery',120.),('work',300.),
               ('recovery',120.),('work',300.),('cooldown',600.))
    # Non-work uses the contract's existing leading non-work kind; fixture only.
    catalog = (Dose('easy-4.0','easy',4.,(('warmup',2400.),)),
        Dose('easy-4.4','easy',4.4,(('warmup',2640.),)),
        Dose('long-7.6','long',7.6,(('warmup',4560.),)),
        Dose('long-8.4','long',8.4,(('warmup',5040.),)),Dose('quality-3x5','quality',4.,quality))
    catalog = tuple(replace(d,forecast_scope=ForecastScope(
        'controlled_threshold' if d.role=='quality' else 'controlled_easy',
        'explicit synthetic phase times with declared execution margin',
        'synthetic continuous running including jogging recoveries; no planned standing')) for d in catalog)
    history,phases = [],[]
    for day in range(-35,0):
        kind = {0:3,2:4,4:1}.get(day%7)
        if kind is None:
            continue
        dose = catalog[kind]
        start = origin+timedelta(days=day,hours=16)
        seconds = sum(s for _,s in dose.phases)
        end = start+timedelta(seconds=seconds)
        key = f'seed:{day}'
        history.append(accounted(key,start,dose.miles,dose.phases,point(seconds/60),dose.role,
                                 ref,source='observed',available=end))
        phases.append(PhaseHistory(key,start,end,end,ref.accounting_key,
            phase_observations(layout(dose.phases),(SignalSpan(0.,seconds,.6,.6),))))
    # Explicit engineering-stage contracts, frozen before search. This is not
    # athlete-specific contract construction or an endorsement of this catalogue.
    easy_contract = CoachingContract(origin,'declared synthetic stage','easy','controlled_easy',
        DoseLimits(Quantity(4.,4.4,basis='catalogue stage band')),None)
    long_contract = CoachingContract(origin,'declared synthetic stage','long','controlled_easy',
        DoseLimits(point(8.4)),DoseLimits(point(7.6)))
    quality_contract = CoachingContract(origin,'declared synthetic stage','threshold','controlled_threshold',
        DoseLimits(point(4.),point(900.),point(300.),point(.4)),None)
    contracts = {d.key: {'easy':easy_contract,'long':long_contract,'quality':quality_contract}[d.role]
                 for d in catalog}
    harness = DailyHarness(root,config,origin=origin,catalog=catalog,history=history,
        phase_history=phases,scales=(44.,44.,15.),training_policy=TrialPolicy(),
        search_policy=HarnessPolicy(hard_days,tail_days,budget,response_allowance=.5),demand_scenario=DemandScenario(),
        exchange=.01,timing=PurposeTimingPolicy(1.),coaching_contracts=contracts,
        easy_duration_policy=EasyDurationPolicy(10,25,.01,'R1 synthetic stage',4.4),
        decision_source='R1 explicit synthetic stage; not calibrated athlete policy')
    uploads = []
    stopped = None
    try:
        for offset in range(days):
            at = origin+timedelta(days=offset)
            event = harness.replan(at,reason='initial' if offset==0 else 'daily_rollover')
            print(f"{event['as_of']} {event['reason']}: {event['evaluations']} evaluations, "
                  f"{event['seconds']:.2f}s, next-7 edits={len(event['churn']['7'])}",flush=True)
            if stop_on_churn and any(event['churn'].values()):
                stopped = 'near_term_churn'
                break
            day = at.date().isoformat()
            if day not in harness.active:
                continue
            dose = harness.catalog[harness.active[day]['dose']]
            if scenario=='missed' and not uploads:
                uploads.append(dict(day=day,status='missed',issued=harness.active[day]))
                continue
            sign = {'lower_edge':-1,'upper_edge':1}.get(scenario,0)
            distance_factor = (dose.miles+sign*harness.policy.distance_margin)/dose.miles
            duration_factor = 1+sign*harness.policy.phase_fraction_margin
            payload,actual_phases,end = synthetic_payload(dose,at+timedelta(hours=16),
                distance_factor=distance_factor,duration_factor=duration_factor,hr=170 if scenario=='higher_cost' else 140)
            outcome = harness.upload(payload,day=day,available_at=end,observed_role=dose.role,
                                     observed_phases=actual_phases)
            uploads.append(dict(day=day,**outcome))
            after = harness.replan(end+timedelta(seconds=1),reason='post_upload')
            print(f"{after['as_of']} post_upload: {after['evaluations']} evaluations, "
                  f"{after['seconds']:.2f}s, next-7 edits={len(after['churn']['7'])}",flush=True)
            if stop_on_churn and any(after['churn'].values()):
                stopped = 'near_term_churn'
                break
        report = harness.summary(offset+1,uploads)
        report.update(scenario=scenario,stop_reason=stopped,requested_days=days)
        with (Path(root)/'report.json').open('x') as stream:
            stream.write(dump(report))
        return report
    finally:
        harness.sandbox.close()


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--days',type=int,default=3)
    parser.add_argument('--evaluations',type=int,default=3000)
    parser.add_argument('--hard-days',type=int,default=21)
    parser.add_argument('--tail-days',type=int,default=14)
    parser.add_argument('--scenario',choices=('expected','higher_cost','lower_edge','upper_edge','missed'),default='expected')
    parser.add_argument('--stop-on-churn',action='store_true')
    args = parser.parse_args()
    r = demo(args.output_dir,days=args.days,budget=args.evaluations,hard_days=args.hard_days,
             tail_days=args.tail_days,scenario=args.scenario,stop_on_churn=args.stop_on_churn)
    print(dump(dict(events=len(r['events']),uploads=len(r['uploads']),delivered=r['delivered'],stop_reason=r['stop_reason'])))
