"""Offline evaluation boundary for a future search and publication path.

This packages the existing fixed-stage experiment, not a finished coach policy.
No legacy planner imports, search, tail generation, database writes, or implicit
weights. In particular, a finite trial score is NOT physiological clearance.
"""

from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from hashlib import sha256
from functools import cached_property
from math import isfinite,ceil
from run_analysis.program_delivery import TargetSegment
from run_analysis.workload_accounting import Quantity
from types import SimpleNamespace

from run_analysis.adaptive.program_objective_trial import (
    PURPOSES, AXES, DemandScenario, TrialPolicy, TrialSession, EasyDurationPolicy,
    components, score, easy_duration_components,rolling_volume_components,
)
from run_analysis.adaptive.timely_purpose_trial import PurposeTimingPolicy, timing_components
from run_analysis.adaptive.identity_json import (
    identity_fields,identity_dumps,identity_object_dumps,identity_sequence_dumps)
from run_analysis.adaptive.race_goal import RaceGoalRevision, RaceGoalApplication
from run_analysis.adaptive.race_event import RaceEventResolution
from .quality_program_contract import QualityProgramme
from run_analysis.program_constraints import ReportedDistanceReference


MODEL_VERSION = 'shared_fixed_stage_trial_v13_sustained_delivery_contract'
CONTINUATION_MODEL_VERSION = 'shared_fixed_stage_trial_v15_dated_continuation'
MEMORY_CONTINUATION_MODEL_VERSION = 'shared_fixed_stage_trial_v16_memory_covered_continuation'
SINGLE_VOLUME_MODEL_VERSION = 'shared_fixed_stage_trial_v17_single_volume_trial'
VOLUME_BENEFIT_MODEL_VERSION = 'shared_fixed_stage_trial_v18_volume_benefit'
PURPOSE_CONTINUATION_MODEL_VERSION = 'shared_fixed_stage_trial_v19_purpose_continuation'
PHASE_CONTINUATION_MODEL_VERSION = 'shared_fixed_stage_trial_v20_known_purpose_phase'
LIMITATIONS = (
    'Timely long/quality delivery value is not implemented.',
    'Readiness and full program feasibility are not evaluated.',
    'Switching cost is not implemented; no continuity preference is applied.',
    'Forecast demands and continuation are supplied, not generated or calibrated.',
    'Daily accounting only; same-day uploads and multiple daily runs need an adapter.',
)


def _sessions(rows):
    # Frozen dataclasses alone do not protect caller-supplied mutable lists.
    return tuple(sorted((row if type(row) is TrialSession and type(row.demand) is tuple else
                        replace(row, demand=tuple(row.demand)) for row in rows),
                        key=lambda row: (row.day, row.key)))


@dataclass(frozen=True)
class ProgramContext:
    history: tuple[TrialSession, ...]
    stage_start: date
    decision: date
    hard_end: date
    end: date
    scales: tuple[float, float, float]
    policy: TrialPolicy
    scenario: DemandScenario
    exchange: float
    timing: PurposeTimingPolicy | None = None
    opening_coverage: tuple[tuple[str, float], ...] = ()
    evidence_key: str = ''
    demand_accounting_key: str = ''
    # Explicit background work beyond the selectable projection. It is not a
    # generated universal tail, nor may search delete it to avoid its costs.
    downstream: tuple[TrialSession, ...] = ()
    downstream_end: date | None = None
    downstream_basis: str = ''
    easy_duration_policy: EasyDurationPolicy | None = None
    volume_targets: tuple[TargetSegment,...] = ()
    support_targets: tuple[TargetSegment,...] = ()
    steady_continuation_references: tuple[tuple[str,float,tuple[Quantity,...]],...] = ()
    soft_horizon_taper: bool = False
    delivery_contract: str = 'aggregate_v1'  # Historical snapshots retain their meaning.
    continuation_boundary_references: tuple[tuple[str,float,tuple[Quantity,...]],...] = ()
    continuation_guard_days: int = 0  # Zero preserves archived research contracts.
    volume_linear_coefficient: float = 0.  # Explicit policy utility, not physiology.
    purpose_continuation: bool = False  # Absent preserves archived finite-purpose accounting.
    purpose_continuation_phase: str = 'uniform_v1'  # Archived v19 contract; new builds select v20 explicitly.
    race_goal: RaceGoalRevision | None = None
    race_demand: tuple[Quantity, ...] = ()
    race_goal_history: tuple[RaceGoalApplication, ...] = ()
    continuation_distribution: str = 'linear_v1'  # Explicit offline contracts preserve archived meaning.
    race_event: RaceEventResolution | None = None
    race_event_history: tuple[RaceEventResolution,...] = ()
    quality_programme: QualityProgramme | None = None
    long_run_weekday: int | None = None  # User calendar anchor; not physiology.
    long_run_not_before: date | None = None  # Same frozen transition as the preference.
    volume_shortfall_policy: str | None = None
    volume_shortfall_since: date | None = None
    foregone_volume: tuple = ()  # Nonphysical expired opportunity loss, NEVER delivered credit.
    reported_distance: ReportedDistanceReference | None = None

    @cached_property
    def handoff_probabilities(self):
        if self.continuation_distribution=='linear_v1':
            return None
        from .continuation_distribution import c1_handoff_probabilities
        return c1_handoff_probabilities((self.end-self.taper_start).days)

    @cached_property
    def taper_start(self):
        """Do not replace future work before the visible prefix's volume expires.

        A W-day trailing window smoothed over S days has W+S-1 daily weights.
        Its last contribution is W+S-2 days after a run. No calendar is locked.
        """
        if not self.continuation_guard_days:
            return self.hard_end
        return max(self.hard_end,self.decision+timedelta(days=
            self.continuation_guard_days+self.policy.volume_window_days+self.volume_smoothing_days-2))

    @cached_property
    def volume_smoothing_days(self):
        return ceil(max(7/self.policy.long_per_7,7/self.policy.quality_per_7)) if self.delivery_contract=='sustained_v1' else 1

    def __post_init__(self):
        from .foregone_volume import VERSION,LEGACY_VERSION,ForegoneVolume
        rows=tuple(self.foregone_volume)
        if self.volume_shortfall_policy not in {None,VERSION,LEGACY_VERSION}:
            raise ValueError('Named volume shortfall policy required')
        if self.volume_shortfall_policy is None:
            if rows or self.volume_shortfall_since is not None:
                raise ValueError('Foregone volume needs an explicit accounting policy')
        elif (type(self.volume_shortfall_since) is not date
                or self.volume_shortfall_since>self.decision
                or self.delivery_contract!='sustained_v1' or self.policy.purpose_weights[1]!=0):
            raise ValueError('Foregone volume requires a dated single-volume sustained contract')
        if (any(not isinstance(r,ForegoneVolume) or not self.volume_shortfall_since<=r.day<self.decision for r in rows)
                or len({r.day for r in rows})!=len(rows)):
            raise ValueError('Unique expired issued opportunities required; future candidates cannot earn relief')
        object.__setattr__(self,'foregone_volume',tuple(sorted(rows,key=lambda r:r.day)))
        if self.long_run_weekday is not None and (type(self.long_run_weekday) is not int or not 0 <= self.long_run_weekday <= 6):
            raise ValueError('Valid long-run weekday required')
        if self.long_run_not_before is not None and (type(self.long_run_not_before) is not date
                or self.long_run_not_before.weekday()!=self.long_run_weekday):
            raise ValueError('Long-run transition must match the weekday')
        if self.quality_programme is not None and not isinstance(self.quality_programme,QualityProgramme):
            raise ValueError('Explicit frozen quality programme required')
        object.__setattr__(self,'race_event_history',tuple(self.race_event_history))
        if any(not isinstance(r,RaceEventResolution) for r in self.race_event_history) or len({r.race_key for r in self.race_event_history}) != len(self.race_event_history):
            raise ValueError('One current outcome per applied race required')
        if self.race_event is not None and (not isinstance(self.race_event, RaceEventResolution)
                or self.race_goal is None or self.race_event.race_key != self.race_goal.key):
            raise ValueError('Race outcome must belong to the current objective')
        if self.race_goal and any(r.race_key==self.race_goal.key and r!=self.race_event for r in self.race_event_history):
            raise ValueError('Current and historical race outcome views must agree')
        if (self.continuation_distribution not in {'linear_v1','c1_v1'} or
                self.continuation_distribution!='linear_v1' and not self.continuation_boundary_references):
            raise ValueError('Named alternate distribution requires distributed continuation')
        object.__setattr__(self, 'race_goal_history', tuple(self.race_goal_history))
        if self.race_goal_history and self.race_goal_history[-1].revision != self.race_goal:
            raise ValueError('Dated target history must match current race objective')
        if self.race_goal is not None and not isinstance(self.race_goal, RaceGoalRevision):
            raise ValueError('Explicit durable race objective required')
        object.__setattr__(self, 'race_demand', tuple(self.race_demand))
        if self.race_demand and (len(self.race_demand) != 3 or self.race_goal is None
                or self.race_goal.goal is None or any(not isinstance(q, Quantity) for q in self.race_demand)):
            raise ValueError('Three conditional race demand axes and dated objective required')
        if (self.purpose_continuation_phase not in {'uniform_v1','known_full_phase_v1'}
                or self.purpose_continuation_phase!='uniform_v1' and not self.purpose_continuation):
            raise ValueError('Purpose continuation phase requires a named enabled contract')
        if (type(self.purpose_continuation) is not bool or self.purpose_continuation and (
                not self.continuation_boundary_references or self.timing is None
                or self.timing.grace_fraction<=0)):
            raise ValueError('Purpose continuation requires distributed handoffs and plateau timing')
        if (not isfinite(self.volume_linear_coefficient) or self.volume_linear_coefficient < 0
                or self.volume_linear_coefficient and (self.policy.purpose_weights[1] != 0
                    or self.policy.volume_window_days is None
                    or self.steady_continuation_references or self.downstream_end is not None)):
            raise ValueError('Linear volume benefit requires a finite nonnegative single rolling-volume contract')
        if (type(self.continuation_guard_days) is not int or self.continuation_guard_days<0
                or self.continuation_guard_days and (not self.soft_horizon_taper
                    or self.delivery_contract!='sustained_v1' or self.policy.volume_window_days is None)):
            raise ValueError('Memory-covered continuation requires sustained/tapered volume accounting')
        if self.delivery_contract not in {'aggregate_v1','sustained_v1'}:
            raise ValueError('Unknown delivery contract')
        if self.delivery_contract=='sustained_v1' and (self.policy.volume_window_days is None
                or self.steady_continuation_references or self.downstream_end is not None):
            raise ValueError('Sustained delivery requires rolling volume without a second terminal policy')
        if type(self.soft_horizon_taper) is not bool or self.soft_horizon_taper and (
                self.steady_continuation_references or self.downstream_end is not None):
            raise ValueError('Explicit separate soft-horizon taper or terminal policy required')
        refs=tuple((role,miles,tuple(demand)) for role,miles,demand in self.steady_continuation_references)
        if refs and (len(refs)!=3 or {r for r,_,_ in refs}!={'easy','long','quality'}
                or any(not isfinite(m) or m<=0 or len(q)!=3 or any(not isinstance(v,Quantity) for v in q)
                       for _,m,q in refs)):
            raise ValueError('Explicit positive physical continuation references required')
        if refs and self.downstream_end is not None:
            raise ValueError('Do not combine coarse terminal expectation with a supplied downstream calendar')
        object.__setattr__(self,'steady_continuation_references',refs)
        boundary_refs=tuple((role,miles,tuple(demand)) for role,miles,demand in self.continuation_boundary_references)
        if boundary_refs:
            if (not self.soft_horizon_taper or self.delivery_contract!='sustained_v1'
                    or self.steady_continuation_references or self.downstream_end is not None
                    or self.easy_duration_policy is None
                    or self.easy_duration_policy.supporting_easy_miles_per_7 is None):
                raise ValueError('Distributed continuation requires sustained/tapered supporting-volume context')
            if (len(boundary_refs)!=3 or {r for r,_,_ in boundary_refs}!={'easy','long','quality'}
                    or any(not isfinite(m) or m<=0 or len(q)!=3
                           or any(not isinstance(v,Quantity) for v in q) for _,m,q in boundary_refs)):
                raise ValueError('Three positive physical continuation references required')
        object.__setattr__(self,'continuation_boundary_references',tuple(sorted(boundary_refs)))
        for name in ('volume_targets','support_targets'):
            segments=tuple(sorted(getattr(self,name),key=lambda s:s.start))
            if any(not isinstance(s,TargetSegment) for s in segments) or any(b.start<a.end for a,b in zip(segments,segments[1:])):
                raise ValueError('Nonoverlapping explicit dated volume targets required')
            object.__setattr__(self,name,segments)
        if self.easy_duration_policy is not None and not isinstance(self.easy_duration_policy, EasyDurationPolicy):
            raise ValueError('Explicit easy-duration experiment policy required')
        object.__setattr__(self, 'history', _sessions(self.history))
        object.__setattr__(self, 'downstream', _sessions(self.downstream))
        object.__setattr__(self, 'opening_coverage', tuple(tuple(p) for p in self.opening_coverage))
        if (len(dict(self.opening_coverage)) != len(self.opening_coverage)
                or any(p not in {'long', 'quality'} or not isfinite(v) or not 0 <= v <= 1
                       for p, v in self.opening_coverage)):
            raise ValueError('Unique bounded explicit opening purpose coverage required')
        object.__setattr__(self, 'scales', tuple(self.scales))
        object.__setattr__(self, 'policy', replace(
            self.policy, purpose_weights=tuple(self.policy.purpose_weights)))
        object.__setattr__(self, 'scenario', replace(
            self.scenario, axis_weights=tuple(self.scenario.axis_weights)))
        if (any(type(day) is not date for day in
                (self.stage_start, self.decision, self.hard_end, self.end))
                or not self.stage_start <= self.decision < self.hard_end < self.end):
            raise ValueError('Explicit ordered stage, decision, hard end and continuation end required')
        if self.continuation_guard_days and (self.continuation_guard_days>(self.hard_end-self.decision).days
                or self.end<=self.taper_start):
            raise ValueError('Continuation must cover prefix memory and retain a nonempty taper')
        if len(self.scales) != 3 or any(not isfinite(v) or v <= 0 for v in self.scales):
            raise ValueError('Three positive finite fixed reference scales required')
        if not isfinite(self.exchange) or self.exchange < 0:
            raise ValueError('Explicit nonnegative finite exchange required')
        if any(row.source != 'observed' or row.day >= self.decision for row in self.history):
            raise ValueError('History must contain only earlier observed work')
        if (len({row.key for row in self.history}) != len(self.history)
                or len({row.day for row in self.history}) != len(self.history)):
            raise ValueError('History must have unique reconciled identities and dates')
        if self.downstream_end is None:
            if self.downstream or self.downstream_basis:
                raise ValueError('Downstream work/rest requires explicit coverage and provenance')
        elif (type(self.downstream_end) is not date or self.downstream_end <= self.end
              or not isinstance(self.downstream_basis, str) or not self.downstream_basis.strip()):
            raise ValueError('Explicit downstream endpoint and nonempty provenance required')
        if any(row.source != 'planned' or not self.end <= row.day < self.downstream_end
               or abs(row.miles*10-round(row.miles*10)) > 1e-8 for row in self.downstream):
            raise ValueError('Downstream rows must be committed planned work within their scope')
        if (len({row.key for row in (*self.history, *self.downstream)}) != len(self.history)+len(self.downstream)
                or len({row.day for row in self.downstream}) != len(self.downstream)):
            raise ValueError('Unique downstream identities and dates required')


@dataclass(frozen=True)
class ProgramCandidate:
    hard: tuple[TrialSession, ...]
    continuation: tuple[TrialSession, ...]
    continuation_basis: str

    def __post_init__(self):
        object.__setattr__(self, 'hard', _sessions(self.hard))
        object.__setattr__(self, 'continuation', _sessions(self.continuation))
        if not isinstance(self.continuation_basis, str) or not self.continuation_basis.strip():
            raise ValueError('Declare how the continuation was supplied, including explicit rest')
        for row in (*self.hard, *self.continuation):
            if row.source != 'planned':
                raise ValueError('Candidates contain planned work only')
            if abs(row.miles * 10 - round(row.miles * 10)) > 1e-8:
                raise ValueError('Commit a one-decimal prescription before evaluation')


@dataclass(frozen=True)
class ProgramEvaluation:
    fingerprint: str
    model_version: str
    total_cost: float
    delivery_cost: float
    weighted_demand_cost: float
    # None means not evaluated, not an established zero cost or readiness pass.
    switching_cost: float | None
    parts: dict
    conditional_ledger: tuple[TrialSession, ...]
    selection_ready: bool = False
    limitations: tuple[str, ...] = LIMITATIONS
    downstream_accounting: dict | None = None
    short_run_cost: float = 0.


def evaluate_program(context: ProgramContext, candidate: ProgramCandidate, *,
                     timing_history=None, timing_opening_coverage=None,
                     demand_interaction='independent_v1',
                     quality_cadence_contract='amplitude_v1') -> ProgramEvaluation:
    """One entry point for comparing and rechecking committed prescriptions.

    Half-open intervals give each future run exactly one owner. The boundary
    between hard plan and continuation does not change the score: both are
    evaluated together against the same historical prefix and fixed stage.
    No downstream distance allocation, rounding, or role substitution occurs.
    Explicit diagnostic timing overrides affect only purpose cadence credit,
    never physical history, volume, demand or admission. Defaults preserve the
    historical scoring/fingerprint contract; overrides enter the identity.
    Optional context-owned downstream work is an explicit external scenario:
    charge its self/interactions once, without extending current delivery goals.
    It cannot supply benefit credit or be deleted by a candidate. Its generation
    and suitability remain the caller's responsibility, not inferred here.
    """
    from .timely_purpose_trial import CADENCE_CONTRACTS
    if quality_cadence_contract not in CADENCE_CONTRACTS or (
            quality_cadence_contract!='amplitude_v1' and
            (context.timing is None or context.timing.grace_fraction<=0 or context.quality_programme is not None)):
        raise ValueError('Proportional quality cadence requires explicit plateau coaching without a second programme')
    if timing_history is not None or timing_opening_coverage is not None:
        if context.quality_programme is not None:
            raise ValueError('Diagnostic timing override cannot silently bypass two-purpose evidence')
        if context.timing is None:
            raise ValueError('Diagnostic timing override requires a timing policy')
        if timing_history is not None:
            timing_history = _sessions(timing_history)
            if any(r.source != 'observed' or r.day >= context.decision for r in timing_history):
                raise ValueError('Diagnostic timing history must precede the decision')
    if any(not context.decision <= row.day < context.hard_end for row in candidate.hard):
        raise ValueError('Hard prescriptions must lie in [decision, hard_end)')
    from .demand_interaction import MODES
    if demand_interaction not in MODES:
        raise ValueError('Explicit supported demand interaction required')
    if demand_interaction!='independent_v1' and (context.race_goal is not None or
            context.race_demand or context.steady_continuation_references):
        raise ValueError('Shared running-work experiment supports general-fitness boundary continuation only')
    if any(not context.hard_end <= row.day < context.end for row in candidate.continuation):
        raise ValueError('Continuation prescriptions must lie in [hard_end, end)')
    future = (*candidate.hard, *candidate.continuation)
    demand_future = (*future, *context.downstream)
    parts = components(context.history, demand_future, stage_start=context.stage_start,
        decision=context.decision, end=context.downstream_end or context.end, scales=context.scales,
        policy=context.policy, scenario=context.scenario, delivery_end=context.end,
        demand_interaction=demand_interaction)
    limitations = LIMITATIONS
    if context.timing is not None:
        temporal = timing_components(context.history if timing_history is None else timing_history, future,
            stage_start=context.stage_start, decision=context.decision, end=context.end,
            policy=context.policy, timing=context.timing,
            opening_coverage=context.opening_coverage if timing_opening_coverage is None else timing_opening_coverage,
            quality_cadence_contract=quality_cadence_contract)
        if context.race_goal is not None:
            from .race_accounting import apply_purpose_window
            temporal = apply_purpose_window(context, temporal)
        # Replace only the selected aggregate purpose terms; never charge both.
        effective_losses = dict(parts['shortfall_by_purpose'],
                                **{purpose: row['loss'] for purpose, row in temporal.items()})
        parts = dict(parts, aggregate_delivery_loss=parts['delivery_loss'], timing=temporal,
            aggregate_shortfall_by_purpose=parts['shortfall_by_purpose'],
            shortfall_by_purpose=effective_losses,
            delivery_loss=sum(effective_losses[p] * context.policy.purpose_weights[i]
                              for i, p in enumerate(PURPOSES)))
        limitations = ('Timely purpose value is experimental and uncalibrated.', *LIMITATIONS[1:])
    if context.easy_duration_policy is not None:
        easy = easy_duration_components(context.history, future, stage_start=context.stage_start,
            decision=context.decision, delivery_end=context.end, policy=context.easy_duration_policy,
            bound_fraction=context.scenario.bound_fraction)
        # Replace the preferred-size/count objective; do not grant one full
        # ordinary credit per short run or charge both competing formulations.
        funding = easy['supporting_easy_volume']
        effective = dict(parts['shortfall_by_purpose'],
                         ordinary_easy=funding['shortfall'] if funding is not None else 0.)
        parts = dict(parts, prior_shortfall_by_purpose=parts['shortfall_by_purpose'],
            shortfall_by_purpose=effective, easy_duration=easy,
            delivery_loss=sum(effective[p]*context.policy.purpose_weights[i]
                              for i, p in enumerate(PURPOSES)))
        limitations = (*limitations,
            'Easy-duration experiment replaces ordinary-size credit; floor eligibility is reported, not app-wired.')
    if context.policy.volume_window_days is not None:
        volume=rolling_volume_components(context.history,future,stage_start=context.stage_start,
            decision=context.decision,end=context.end,policy=context.policy,
            supporting_miles=context.easy_duration_policy.supporting_easy_miles_per_7 if context.easy_duration_policy else None,
            volume_targets=context.volume_targets,support_targets=context.support_targets,
            smoothing_days=context.volume_smoothing_days,foregone_volume=context.foregone_volume,
            opening_before=context.volume_shortfall_since if context.volume_shortfall_policy=='forgone_issued_v2' else None)
        effective=dict(parts['shortfall_by_purpose'],**volume['losses'])
        parts=dict(parts,finite_interval_losses=parts['shortfall_by_purpose'],rolling_volume=volume,
            shortfall_by_purpose=effective,delivery_loss=sum(effective[p]*context.policy.purpose_weights[i]
                                                          for i,p in enumerate(PURPOSES)))
    if context.steady_continuation_references:
        from .steady_continuation_trial import terminal_value
        terminal=terminal_value(context,SimpleNamespace(conditional_ledger=(*context.history,*future)),
            {role:(miles,demand) for role,miles,demand in context.steady_continuation_references})
        effective={k:v+terminal['volume'].get(k,0.)+terminal['timing'].get(k,0.)
                   for k,v in parts['shortfall_by_purpose'].items()}
        parts=dict(parts,steady_continuation=terminal,selectable_delivery_loss=parts['delivery_loss'],
            selectable_demand_cost=parts['demand_cost'],shortfall_by_purpose=effective,
            delivery_loss=sum(effective[p]*context.policy.purpose_weights[i] for i,p in enumerate(PURPOSES)),
            demand_cost=parts['demand_cost']+terminal['raw_demand_cost'],
            demand_by_axis={k:v+terminal['demand_by_axis'][k] for k,v in parts['demand_by_axis'].items()})
        limitations=(*limitations,terminal['scope'],terminal['omitted'])
    if context.soft_horizon_taper:
        from .soft_horizon_trial import tapered_cost
        tapered=tapered_cost(context,SimpleNamespace(parts=parts,conditional_ledger=(*context.history,*future)))
        parts=dict(parts,soft_horizon=tapered,untapered_delivery_loss=parts['delivery_loss'],
            untapered_demand_cost=parts['demand_cost'],untapered_easy_duration=parts['easy_duration'],
            shortfall_by_purpose=tapered['losses'],delivery_loss=tapered['delivery'],
            demand_cost=tapered['raw_demand_cost'],demand_by_axis=tapered['demand_by_axis'],
            easy_duration=dict(parts['easy_duration'],cost=tapered['short']))
        limitations=(*limitations,tapered['scope'],'Soft horizon taper is experimental; no stability guarantee.')
    total = score(parts, context.exchange)
    version = MODEL_VERSION
    if context.continuation_boundary_references:
        from .continuation_boundary_trial import continuation_boundary_cost
        boundary=continuation_boundary_cost(context,SimpleNamespace(total_cost=total,parts=parts,
            conditional_ledger=(*context.history,*future)),
            {role:(miles,demand) for role,miles,demand in context.continuation_boundary_references},
            include_linear_volume=bool(context.volume_linear_coefficient),
            timing_history=timing_history,timing_opening_coverage=timing_opening_coverage,
            quality_cadence_contract=quality_cadence_contract,
            handoff_probabilities=context.handoff_probabilities,demand_interaction=demand_interaction)
        effective=dict(parts['shortfall_by_purpose'],**boundary['losses'])
        parts=dict(parts,continuation_boundary=boundary,pre_boundary_total_cost=total,
            pre_boundary_delivery_loss=parts['delivery_loss'],pre_boundary_demand_cost=parts['demand_cost'],
            shortfall_by_purpose=effective,
            delivery_loss=sum(effective[p]*context.policy.purpose_weights[i] for i,p in enumerate(PURPOSES)),
            demand_cost=parts['demand_cost']+boundary['raw_cross_demand_cost'],
            demand_by_axis={k:parts['demand_by_axis'][k]+boundary['cross_demand_by_axis'][i]
                           for i,k in enumerate(AXES)})
        total=boundary['cost']
        version=MEMORY_CONTINUATION_MODEL_VERSION if context.continuation_guard_days else CONTINUATION_MODEL_VERSION
        limitations=(*limitations,*boundary['limitations'])
    if context.policy.purpose_weights[1] == 0:
        version=SINGLE_VOLUME_MODEL_VERSION
        limitations=(*limitations,'Supporting volume is diagnostic only; no independent delivery penalty.')
    if context.volume_linear_coefficient:
        from .volume_benefit_trial import linear_volume_loss
        linear = linear_volume_loss(context,SimpleNamespace(parts=parts,conditional_ledger=(*context.history,*future)))
        added = context.volume_linear_coefficient * linear * context.policy.purpose_weights[0]
        parts = dict(parts, volume_benefit=dict(linear_coefficient=context.volume_linear_coefficient,
            linear_volume_loss=linear, added_volume_cost=added,
            squared_volume_loss=parts['shortfall_by_purpose']['volume'],
            baseline_total_cost=total, baseline_delivery_loss=parts['delivery_loss'],
            assumption='Capped marginal training utility; not calibrated physiology'),
            delivery_loss=parts['delivery_loss']+added)
        total += added
        version=VOLUME_BENEFIT_MODEL_VERSION
        limitations=(*limitations,'Linear volume value is a declared policy assumption, not guaranteed delivery.')
    if context.purpose_continuation:
        version=(PHASE_CONTINUATION_MODEL_VERSION if context.purpose_continuation_phase=='known_full_phase_v1'
                 else PURPOSE_CONTINUATION_MODEL_VERSION)
    if context.race_goal is not None:
        from .race_accounting import event_cost
        event = event_cost(context, future)
        if event:
            total += event['cost']
            parts = dict(parts, race_event=event, demand_cost=parts['demand_cost']+event['raw_demand_cost'],
                demand_by_axis={name:parts['demand_by_axis'][name]+event['demand_by_axis'][name] for name in AXES})
        version += ':dated_race_v1'
        limitations = (*limitations, 'Race work is a conditional bounded forecast, not race clearance or training-distance headroom.')
    weighted_demand = context.exchange * parts['demand_cost']
    if context.continuation_distribution!='linear_v1':
        version += ':c1_handoff_v1'
        limitations=(*limitations,'Cubic continuation survival is an accounting assumption, not calibrated physiology or proven stability.')
    if context.quality_programme is not None:
        from .quality_program_contract import programme_loss
        old=parts['shortfall_by_purpose']['quality']
        quality=programme_loss(context,future,old)
        difference=(quality['loss']-old)*context.policy.purpose_weights[3]
        total+=difference
        parts=dict(parts,quality_programme=quality,delivery_loss=parts['delivery_loss']+difference,
            shortfall_by_purpose=dict(parts['shortfall_by_purpose'],quality=quality['loss']))
        version+=':two_quality_purposes_v1'
    if not all(isfinite(value) for value in (total, parts['delivery_loss'], weighted_demand)):
        raise ValueError('Nonfinite evaluation is not a selectable program')
    # Include the entire immutable evaluation context, not merely candidate dates.
    # This is a reproducibility identity; there is deliberately no result cache yet.
    context_identity=identity_fields(context)
    if context.reported_distance is None:
        context_identity.pop('reported_distance')  # Keep all existing athlete identities unchanged.
    if context.volume_shortfall_policy is None:
        for name in ('volume_shortfall_policy','volume_shortfall_since','foregone_volume'):
            context_identity.pop(name)
    else:
        version+=':'+context.volume_shortfall_policy
        parts=dict(parts,foregone_volume=dict(policy=context.volume_shortfall_policy,
            since=context.volume_shortfall_since,opportunities=context.foregone_volume,
            physical_credit=False,progression_credit=False))
    if context.long_run_weekday is None:
        context_identity.pop('long_run_weekday')  # Preserve Automatic/archived identities.
    if context.long_run_not_before is None:
        context_identity.pop('long_run_not_before')
    if quality_cadence_contract!='amplitude_v1':
        context_identity['quality_cadence_contract']=quality_cadence_contract
        parts=dict(parts,quality_cadence_contract=quality_cadence_contract)
        version+=':'+quality_cadence_contract
        limitations=(*limitations,'Partial verified quality work shortens cadence grace; not equivalent adaptation or physiological recovery.')
    if demand_interaction!='independent_v1':
        context_identity['demand_interaction']=demand_interaction
        parts=dict(parts,demand_interaction=demand_interaction)
        version+=':'+demand_interaction
        limitations=(*limitations,'Running and harder-work fatigue share exposure; explicit uncalibrated cost hypothesis, not changed physical load.')
    if context.quality_programme is None:
        context_identity.pop('quality_programme')
    if context.race_event is None:
        context_identity.pop('race_event')
    else:
        context_identity['race_event']=dict(key=context.race_event.key,**context.race_event.payload)
    if not context.race_event_history:
        context_identity.pop('race_event_history')
    else:
        context_identity['race_event_history']=[dict(key=r.key,**r.payload) for r in context.race_event_history]
    if context.race_goal is None:
        context_identity.pop('race_goal')
    else:
        context_identity['race_goal'] = dict(key=context.race_goal.key,**context.race_goal.payload)
    if not context.race_demand:
        context_identity.pop('race_demand')
    if not context.race_goal_history:
        context_identity.pop('race_goal_history')
    else:
        context_identity['race_goal_history']=[dict(revision=dict(key=h.revision.key,**h.revision.payload),
            applied_at=h.applied_at.isoformat(),effective_on=h.effective_on.isoformat()) for h in context.race_goal_history]
    if timing_history is not None or timing_opening_coverage is not None:
        context_identity['diagnostic_timing_override']=dict(
            history=[identity_fields(r) for r in timing_history] if timing_history is not None else None,
            opening_coverage=timing_opening_coverage)
    if not context.continuation_boundary_references:
        context_identity.pop('continuation_boundary_references')
    if not context.continuation_guard_days:
        context_identity.pop('continuation_guard_days')
    if not context.volume_linear_coefficient:
        context_identity.pop('volume_linear_coefficient')
    if not context.purpose_continuation:
        context_identity.pop('purpose_continuation')
    if context.purpose_continuation_phase=='uniform_v1':
        context_identity.pop('purpose_continuation_phase')
    if context.continuation_distribution=='linear_v1':
        context_identity.pop('continuation_distribution')
    context_json=identity_object_dumps(context_identity,
        fragments={'history':identity_sequence_dumps(context.history,default=_json_date,compact=True)},
        default=_json_date,compact=True)
    candidate_json=identity_object_dumps(identity_fields(candidate),
        fragments={name:identity_sequence_dumps(getattr(candidate,name),default=_json_date,compact=True)
            for name in ('hard','continuation')},default=_json_date,compact=True)
    fingerprint = sha256(identity_object_dumps(dict(version=version,
        context=context_identity,candidate=candidate),fragments=dict(context=context_json,candidate=candidate_json),
        default=_json_date,compact=True).encode()).hexdigest()
    if context.steady_continuation_references:
        limitations=(*limitations,'Coarse fixed-stage continuation is approximate, not an issued/feasibility-checked calendar.')
    elif context.downstream_end is None:
        limitations = (*limitations, 'Work beyond the projection is unspecified, not established rest.')
    else:
        limitations = (*limitations,
            'Supplied downstream work is a conditional scenario, not a validated continuation policy.')
    return ProgramEvaluation(fingerprint, version, total, parts['delivery_loss'],
        weighted_demand, None, parts, (*context.history, *demand_future), limitations=limitations,
        short_run_cost=parts.get('easy_duration', {}).get('cost', 0.),
        downstream_accounting=dict(
            status='distributed_mean_continuation' if context.continuation_boundary_references else
                   'coarse_fixed_stage_terminal' if context.steady_continuation_references else
                   'supplied_scenario' if context.downstream_end is not None else 'unspecified',
            delivery_end=context.end, demand_end=context.downstream_end or context.end,
            basis=context.downstream_basis or None, keys=tuple(row.key for row in context.downstream)))


def _json_date(value):
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError(f'Unsupported evaluation identity value: {type(value).__name__}')
