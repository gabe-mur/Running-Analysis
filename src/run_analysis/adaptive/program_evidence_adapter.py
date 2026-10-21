"""Offline workload/purpose/constraint bridge. No new detector or recovery gate."""
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone as utc_timezone
from hashlib import sha256
from math import exp, isfinite
from functools import cached_property,lru_cache
from zoneinfo import ZoneInfo

from run_analysis.athlete_projection import aware
from run_analysis.delivered_training import PurposeEvidence, dose_metrics
from run_analysis.program_constraints import assess_program_constraints
from run_analysis.session_decision import assess_program_decision
from run_analysis.recency_demand import CHANNELS, DemandEvent, DemandTerm
from run_analysis.workload_accounting import Quantity, observed_workload, projected_workload
from run_analysis.workload_profile import WorkBout, bout_summary, prescribed_profile
from run_analysis.adaptive.program_objective_trial import TrialSession
from run_analysis.adaptive.shared_program_evaluator import ProgramCandidate, evaluate_program
from run_analysis.adaptive.identity_json import (
    FrozenIdentity,identity_fields,identity_dumps,identity_object_dumps,identity_sequence_dumps)

ACCOUNTING = 'edwards_contiguous_hr_minutes_v1'


@dataclass(frozen=True)
class QualityReference:
    purpose: str
    metric: str
    total: Quantity
    bout: Quantity
    source: str

    def __post_init__(self):
        if (not self.purpose or not self.source or self.metric not in {'work_seconds', 'work_meters'}
                or self.total.low <= 0 or self.bout.low <= 0):
            raise ValueError('Explicit purpose, total/per-bout dose and reference source required')


def quality_credit(bouts, reference, *, purpose=None, structure_basis='unknown'):
    """Bounded dose coverage, not retrospective adherence or proven stimulus.

    No repeat-history minimum. Verified compatible purpose and supported bout
    extent permit a positive lower bound. Missing purpose/structure leaves a
    bound, not zero or fabricated full completion. Different verified purpose
    does not fulfill this reference, though all physical work remains load.
    """
    if purpose is not None and purpose != reference.purpose:
        return Quantity.point(0., 'different_verified_purpose')
    if not bouts:
        return Quantity(0., 1., basis='unresolved_work_structure')
    ranges = [(b.duration_range_seconds or (b.work_seconds, b.work_seconds))
              if reference.metric == 'work_seconds' else
              ((b.distance_meters, b.distance_meters) if b.distance_meters is not None else None)
              for b in bouts]
    if any(q is None for q in ranges):
        return Quantity(0., 1., basis='unresolved_work_distance')
    upper = min(1., sum(q[1] for q in ranges)/reference.total.low)
    lower = min(1., sum(q[0] for q in ranges)/reference.total.high)
    supported = (purpose == reference.purpose and structure_basis in {
        'recorded_lap_contrast', 'recorded_continuous_lap', 'explicit_segments', 'prescribed'})
    if not supported or any(q[0] < reference.bout.low for q in ranges):
        lower = 0.
    return Quantity(lower, upper, basis='explicit_purpose_and_bout_dose' if supported else
                    'conditional_purpose_or_inferred_bout_dose')


@dataclass(frozen=True)
class AccountedRun(FrozenIdentity):
    key: str
    start: datetime
    end: datetime
    available_at: datetime
    miles: float
    role: str
    demand: tuple[Quantity, Quantity, Quantity]
    quality: Quantity
    source: str
    bouts: tuple[WorkBout, ...] = ()
    accounting_key: str = ACCOUNTING
    quality_reference_key: str = ''
    work_evidence_basis: str = 'unknown'
    command_running_minutes: float | None = None
    advisory_timezone: str | None = None

    @cached_property
    def _trial_rows(self):
        # Nonfield object-local memo; corrected sources construct a new record.
        return {}

    def trial_session(self, day, role, credit):
        # Retain exact credit type and signed zero, not merely float equality.
        key=(day,role,float(credit).hex(),type(credit))
        if key not in self._trial_rows:
            row=TrialSession(self.key,day,self.miles,role,self.demand,credit,self.source)
            if len(self._trial_rows)>=8:return row
            self._trial_rows[key]=row
        return self._trial_rows[key]

    @cached_property
    def supported_work(self):
        # All dependencies are immutable normalized dataclass fields. A changed
        # source/forecast creates a new object; no date-based staleness cache.
        from run_analysis.continuous_training_dose import supported_work_minutes
        return supported_work_minutes(self.bouts,self.demand[2],evidence_basis=self.work_evidence_basis,
            source=self.source,elapsed_seconds=(self.end-self.start).total_seconds())

    def __post_init__(self):
        for at in (self.start, self.end, self.available_at):
            aware(at)
        object.__setattr__(self, 'demand', tuple(self.demand))
        object.__setattr__(self, 'bouts', tuple(replace(b, target_zones=tuple(b.target_zones),
            duration_range_seconds=tuple(b.duration_range_seconds) if b.duration_range_seconds is not None else None)
            for b in self.bouts))
        if (not self.key or not isfinite(self.miles) or self.miles <= 0 or self.end <= self.start
                or len(self.demand) != 3 or any(not isinstance(q, Quantity) for q in self.demand)
                or self.role not in {'easy', 'long', 'quality', 'unknown'}
                or self.source not in {'observed', 'planned'} or not self.accounting_key
                or not isinstance(self.quality, Quantity) or self.quality.high > 1):
            raise ValueError('Identified, timed, unit-tagged physical run with bounded purpose required')
        if self.source == 'observed' and self.available_at < self.end:
            raise ValueError('Observed work cannot be available before completion')
        if self.advisory_timezone is not None:
            zone=ZoneInfo(self.advisory_timezone)
            if self.source!='planned' or self.available_at.astimezone(zone).date()>self.start.astimezone(zone).date():
                raise ValueError('Advisory clock applies only to a current/future planned date')
        if self.source == 'planned' and self.available_at > self.start and self.advisory_timezone is None:
            raise ValueError('Forecast must be issued before work starts')
        if self.command_running_minutes is not None and (self.source != 'planned'
                or not isfinite(self.command_running_minutes) or self.command_running_minutes <= 0
                or self.demand[0].estimate is None
                or abs(self.command_running_minutes-self.demand[0].estimate) > 1e-8
                or abs(self.command_running_minutes-(self.end-self.start).total_seconds()/60) > 1e-8
                or not self.demand[0].low <= self.command_running_minutes <= self.demand[0].high):
            raise ValueError('Planned duration command must match nominal continuous-running duration and workload scenarios')


def observed_run(profile, dose, *, role, reference, as_of, purpose_evidence=None):
    aware(as_of)
    if dose.available_at > as_of or dose.end > as_of:
        raise ValueError('Observed work not yet available')
    if purpose_evidence is not None and purpose_evidence.key != dose.key:
        raise ValueError('Purpose evidence belongs to another run')
    resolved = profile['intensity_structure'].get('session_workout_type')
    expected_role = ('quality' if resolved in {'intervals', 'tempo_threshold'} else
                     'long' if resolved == 'long' else
                     'easy' if resolved in {'easy', 'recovery', 'medium_long'} else 'unknown')
    if role != expected_role:
        raise ValueError('Use observed resolved role, not the planned workout label')
    load = observed_workload(profile)
    if load.distance_meters is None or load.distance_meters.low != load.distance_meters.high or load.running_minutes is None or load.cardio_points is None:
        raise ValueError('Observed distance/duration/cardio accounting unavailable')
    if dose.distance_meters != load.distance_meters or dose.moving_seconds != load.running_minutes.scale(60):
        raise ValueError('Dose and workload must describe the same physical observation')
    metrics = dose_metrics(dose)
    work = (metrics['work_seconds'].scale(1/60) if metrics['work_seconds'] else
            Quantity(0., load.running_minutes.high, basis='unknown_quality_work') if role in {'quality', 'unknown'} else
            Quantity.point(0., 'no_supported_work_in_resolved_nonquality_run'))
    known = purpose_evidence if purpose_evidence and purpose_evidence.available_at <= as_of else None
    credit = (quality_credit(dose.bouts, reference, purpose=known.purpose if known else None,
                            structure_basis=dose.structure_basis) if role in {'quality', 'unknown'} else
              Quantity.point(0., 'resolved_nonquality_run'))
    return AccountedRun(dose.key, dose.start, dose.end,
        max(dose.available_at, known.available_at) if known else dose.available_at,
        load.distance_meters.low/1609.344, role,
        (load.running_minutes, load.cardio_points, work), credit, 'observed', dose.bouts,
        quality_reference_key=reference_key(reference),
        work_evidence_basis=dose.structure_basis if dose.bouts else
            ('unknown' if role in {'quality', 'unknown'} else 'no_supported_work'))


def forecast_run(recommendation, *, key, issued_at, reference, purpose,
                 duration, base_rate, work_rate, recovery_rate=None, explicit_bouts=None):
    profile = prescribed_profile(recommendation)
    if explicit_bouts is not None:
        explicit_bouts = tuple(explicit_bouts)
        if profile['intensity_structure']['bouts']:
            raise ValueError('Do not override already resolved prescription work')
        profile['intensity_structure'] = dict(evidence='explicit_forecast_segments', **bout_summary(explicit_bouts))
        profile['limitations'] = [v for v in profile['limitations'] if v != 'work_duration_missing']
    load = projected_workload(profile, duration_minutes=duration,
        base_rate=base_rate, work_rate=work_rate, recovery_rate=recovery_rate)
    role = ('long' if recommendation.workout_type == 'long' else
            'quality' if recommendation.workout_type in {'intervals', 'tempo_threshold'} else 'easy')
    if recommendation.workout_type not in {'long', 'intervals', 'tempo_threshold', 'easy', 'recovery'}:
        raise ValueError('Unsupported forecast role; do not silently classify it as easy')
    if recommendation.planned_for is None or recommendation.distance_target_miles is None:
        raise ValueError('Commit a timed one-decimal target before forecasting')
    bouts = tuple(WorkBout(**b) for b in profile['intensity_structure']['bouts'])
    if any(b.distance_meters is not None and (not isfinite(b.distance_meters) or b.distance_meters <= 0) for b in bouts):
        raise ValueError('Positive finite explicit bout distance required')
    if sum(b.distance_meters or 0 for b in bouts) > recommendation.distance_target_miles*1609.344:
        raise ValueError('Forecast work distance exceeds session target')
    unresolved = 'work_duration_missing' in profile['limitations']
    work = (Quantity(0., duration.high, basis='unresolved_prescribed_work') if unresolved else
        Quantity(sum((b.duration_range_seconds or (b.work_seconds, b.work_seconds))[0] for b in bouts)/60,
                 sum((b.duration_range_seconds or (b.work_seconds, b.work_seconds))[1] for b in bouts)/60,
                 basis='prescribed_work'))
    credit = (quality_credit(bouts, reference, purpose=purpose,
                            structure_basis='unknown' if unresolved else 'prescribed') if role == 'quality' else
              Quantity.point(0., 'nonquality_prescription'))
    # End is the explicit upper forecast duration, not a measured completion.
    return AccountedRun(key, recommendation.planned_for,
        recommendation.planned_for+timedelta(minutes=duration.high), issued_at,
        recommendation.distance_target_miles, role, (duration, load.cardio_points, work),
        credit, 'planned', bouts, quality_reference_key=reference_key(reference),
        work_evidence_basis='unknown' if unresolved else 'prescribed')


def reconcile_completion(history, planned, actual, *, replaces_key, as_of):
    """An explicit match removes exactly one forecast, preserving raw actual dose."""
    aware(as_of)
    history, planned = tuple(history), tuple(planned)
    if actual.source != 'observed' or actual.available_at > as_of or actual.end > as_of:
        raise ValueError('Available actual completion required')
    if sum(r.key == replaces_key for r in planned) != 1:
        raise ValueError('Completion must replace exactly one identified forecast')
    expected = next(r for r in planned if r.key == replaces_key)
    if expected.source != 'planned' or expected.available_at > actual.start:
        raise ValueError('Completion requires an actual pre-run forecast')
    if (expected.accounting_key,expected.quality_reference_key) != (actual.accounting_key,actual.quality_reference_key):
        raise ValueError('Changed accounting/reference requires explicit reconciliation')
    if actual.key in {r.key for r in history} or any(r.key == actual.key and r.key != replaces_key for r in planned):
        raise ValueError('Actual completion identity already accounted')
    if any(r.start < actual.end and actual.start < r.end for r in history):
        raise ValueError('Overlapping actual history requires source reconciliation')
    return (*history, actual), tuple(r for r in planned if r.key != replaces_key)


def read_accounted(value):
    """Restore persisted physical evidence, never reinterpret the current primary file."""
    return AccountedRun(**dict(value,
        **{k: datetime.fromisoformat(value[k]) for k in ('start', 'end', 'available_at')},
        demand=tuple(Quantity(**q) for q in value['demand']), quality=Quantity(**value['quality']),
        bouts=tuple(WorkBout(**b) for b in value['bouts'])))


def accounted_phases(key, start, miles, phases, cardio, role, ref, *, source,
                     available, moving=None, elapsed_seconds=None, explicit=True,advisory_timezone=None):
    """Shared planned/measured physical mapping; never infer actual intent from plan."""
    from run_analysis.continuous_training_dose import general_quality_credit
    point = lambda v: Quantity.point(v, 'explicit scenario expectation, not predictive confidence')
    seconds = sum(s for _, s in phases) if phases is not None else elapsed_seconds
    if seconds is None or seconds <= 0 or (elapsed_seconds is not None and abs(seconds-elapsed_seconds) > 1e-5):
        raise ValueError('Complete phase timing or explicit elapsed duration required')
    if moving is None:
        if source == 'observed':
            raise ValueError('Observed movement cannot be inferred from elapsed phase time')
        moving = point(seconds/60)
    if phases is not None and explicit:
        indexes = [i for i, (kind, _) in enumerate(phases) if kind == 'work']
        bouts = tuple(WorkBout(phases[i][1], sum(s for _, s in phases[i+1:indexes[n+1]])
            if n+1 < len(indexes) else None) for n, i in enumerate(indexes))
        work = point(sum(b.work_seconds for b in bouts)/60)
        evidence = 'prescribed' if source == 'planned' else 'explicit_segments'
        supported = work
    else:
        bouts = ()
        work = Quantity(0, moving.high, basis='unresolved observed work') if role in {'quality','unknown'} else point(0)
        evidence, supported = 'unknown', None
    return AccountedRun(key, start, start+timedelta(seconds=seconds), available, miles, role,
        (moving, cardio, work), general_quality_credit(supported, role=role, reference=ref), source,
        bouts, ref.accounting_key, reference_key(ref), evidence,advisory_timezone=advisory_timezone)


def _choose(q, fraction):
    return q.low+(q.high-q.low)*fraction


def stage_quality_credit(run, policy):
    """Purpose completion changes with the stage; raw physical work never does.

    This remains generic supported aerobic work, not threshold/speed equivalence.
    Missing extent remains unresolved; no label or high-HR substitution.
    """
    if policy.quality_work_minutes is None:
        return run.quality
    work = run.supported_work
    if run.role in {'easy','long'}:
        return Quantity.point(0.,'no_quality_purpose')
    if work is None:
        return Quantity(0.,1.,basis='unresolved_work_extent_not_hr_substitution')
    return Quantity(min(1.,work.low/policy.quality_work_minutes) if run.role=='quality' else 0.,
        min(1.,work.high/policy.quality_work_minutes),basis='stage_supported_work_completion_not_adaptation_equivalence')


def evaluate_accounted(context, history, planned, *, as_of, timezone, reference,
                       purpose_fraction, restrictions=(), history_complete_since=None,
                       accounting_key=ACCOUNTING, decision_input=None,
                       program_evaluator=evaluate_program, quality_credit_provider=None,
                       quality_cadence_contract='amplitude_v1'):
    """Evaluate raw evidence with explicit independent demand/purpose scenarios.

    This daily model supports one run per local date: after today's completed
    upload, the remaining plan starts tomorrow. Evidence availability stays at
    the true as-of time. No intraday recovery timing or second-run planning.
    """
    aware(as_of)
    from .timely_purpose_trial import CADENCE_CONTRACTS
    if quality_cadence_contract not in CADENCE_CONTRACTS or (
            quality_cadence_contract!='amplitude_v1' and quality_credit_provider is None):
        raise ValueError('Explicit quality cadence contract requires its shared credit provider')
    if not isfinite(purpose_fraction) or not 0 <= purpose_fraction <= 1:
        raise ValueError('Explicit purpose uncertainty scenario required')
    history, planned = tuple(history), tuple(planned)
    rows = (*history, *planned)
    if len({r.key for r in rows}) != len(rows):
        raise ValueError('Actual/projected identities overlap')
    if any(r.accounting_key != accounting_key for r in rows):
        raise ValueError('Do not mix cardiovascular accounting units')
    if ((accounting_key != ACCOUNTING or context.demand_accounting_key)
            and context.demand_accounting_key != accounting_key):
        raise ValueError('Normalization scales must use the same cardiovascular accounting units')
    expected_reference = reference_key(reference)
    if any(r.quality_reference_key != expected_reference for r in rows):
        raise ValueError('Recompute purpose credit when changing its stage reference')
    if any(r.source != 'observed' or r.available_at > as_of or r.end > as_of for r in history):
        raise ValueError('Only available completed observations belong in history')
    if any(r.source != 'planned' or r.available_at > as_of for r in planned):
        raise ValueError('Only already-issued forecasts belong in the candidate')
    zone = ZoneInfo(timezone)
    if any(r.advisory_timezone is not None and r.advisory_timezone!=timezone for r in planned):
        raise ValueError('Advisory planning date must use the decision timezone')
    today = as_of.astimezone(zone).date()
    if decision_input is not None:
        if decision_input.issued_at > as_of:
            raise ValueError('Decision policy is not yet available')
        current = decision_input.current
        decision_input = replace(decision_input,
            current=current if current and current.available_at <= as_of and current.applies_on <= today else None,
            responses=tuple(r for r in decision_input.responses if r.available_at <= as_of))
    local = lambda r: r.start.astimezone(zone).date()
    if any(local(r) != r.end.astimezone(zone).date() for r in rows):
        raise ValueError('Overnight runs require explicit cross-date accounting')
    decision = today+timedelta(days=1) if any(local(r) == today for r in history) else today
    if any(local(r) >= decision for r in history) or any(local(r) < decision for r in planned):
        raise ValueError('Remove completed-day forecasts explicitly; no silent ownership changes')
    def purpose_credit(r):
        credit=(quality_credit_provider(r) if quality_credit_provider is not None else
            context.quality_programme.delivery(r.key,r.source).opportunity
            if context.quality_programme is not None and r.role=='quality' else stage_quality_credit(r,context.policy))
        if (not isinstance(credit,Quantity) or credit.low<0 or credit.high>1
                or r.role!='quality' and quality_credit_provider is not None and credit.high!=0):
            raise ValueError('Explicit bounded quality credit; nonquality roles cannot gain coaching credit')
        return credit
    # One value/source for trial rows, opening conditions and exposed diagnostics.
    # In particular, uploads must not silently revert to generic work coverage.
    purpose_credits={r.key:purpose_credit(r) for r in rows}
    def trial(r):
        if r.key in race_keys:
            return r.trial_session(local(r), 'race', 0.)
        # Unknown role can carry demand, but cannot receive a made-up full purpose.
        credit=purpose_credits[r.key]
        return r.trial_session(local(r), r.role, _choose(credit, purpose_fraction))
    from .race_event import physical_keys
    race_keys=physical_keys(context)
    observed, future = tuple(map(trial, history)), tuple(map(trial, planned))
    openings, opening_bounds = [], {}
    if context.timing:
        for purpose in context.timing.purposes:
            memory = 7/getattr(context.policy, purpose+'_per_7')*context.timing.memory_scale
            grace=7/getattr(context.policy,purpose+'_per_7')*context.timing.grace_fraction
            # Unknown pre-inventory exposure is a bounded initial condition,
            # not a demand to catch up or a fabricated last workout date.
            origin = min((context.stage_start, *(local(r) for r in history)))
            lows, highs = [0.], [exp(-(decision-origin).days/memory)]
            sampled = [purpose_fraction*highs[0]]
            for r in history:
                if r.key in race_keys:
                    continue
                # Match credits(TrialSession): an unresolved role cannot gain
                # purpose merely because the same event moved into history.
                if purpose == 'quality':
                    q = (purpose_credits[r.key]
                        if r.role == 'quality' else Quantity.point(0., 'unresolved_or_nonquality_role'))
                else:
                    q = Quantity.point(min(r.miles/context.policy.long_miles, 1.)
                                       if r.role == 'long' else 0., 'long_dose_reference')
                if purpose=='quality' and quality_cadence_contract!='amplitude_v1':
                    from .timely_purpose_trial import coverage_at
                    measure=lambda credit:coverage_at(credit,(decision-local(r)).days,grace,memory,
                        credit_contract=quality_cadence_contract)
                    lows.append(measure(q.low));highs.append(measure(q.high))
                    sampled.append(measure(_choose(q,purpose_fraction)))
                else:
                    decay = exp(-max(0.,(decision-local(r)).days-grace)/memory)
                    lows.append(q.low*decay); highs.append(q.high*decay)
                    sampled.append(_choose(q, purpose_fraction)*decay)
            bounds = Quantity(max(lows), max(highs), basis='purpose_history_and_unknown_initial_condition')
            opening_bounds[purpose] = bounds
            # Select each event's scenario BEFORE taking the max envelope. In
            # general, interpolation(max(bounds)) != max(interpolation(bounds)).
            # The latter matches the forecast transition and identical upload.
            # Plateau-aware timing evaluates dated known events itself. Pass
            # only the residual unknown initial condition, not a fake new event.
            openings.append((purpose, purpose_fraction*highs[0] if grace else max(sampled)))
    identity_encoder=_canonical_json if context.continuation_boundary_references else _json
    identity = sha256(identity_object_dumps(dict(runs=rows, reference=identity_fields(reference),
        purpose_fraction=purpose_fraction, timezone=timezone,
        **({'coaching_quality_credit':{k:identity_fields(v) for k,v in purpose_credits.items()}}
            if quality_credit_provider is not None else {}),
        decision_input=identity_fields(decision_input) if decision_input is not None else None),
        fragments={'runs':identity_sequence_dumps(rows,default=identity_encoder)},
        default=identity_encoder).encode()).hexdigest()
    scoped = replace(context, history=observed, decision=decision,
                     opening_coverage=tuple(openings), evidence_key=identity)
    candidate = ProgramCandidate(tuple(r for r in future if r.day < context.hard_end),
        tuple(r for r in future if r.day >= context.hard_end), 'explicit evidence-backed continuation')
    evaluated = program_evaluator(scoped, candidate)
    if (quality_cadence_contract!='amplitude_v1' and
            evaluated.parts.get('quality_cadence_contract')!=quality_cadence_contract):
        raise ValueError('Evidence opening and program evaluator must use the same quality cadence contract')
    def demand(r):
        q = (Quantity.point(r.miles, 'actual_or_committed_distance'), *r.demand)
        return DemandEvent(r.key, local(r), tuple(DemandTerm(c, r.key+':'+c, 1., v)
            for c, v in zip(CHANNELS, q)), r.accounting_key, r.role,
            tuple(b.work_seconds for b in r.bouts) or None, r.source)
    # The spike comparison records physical distance, not a routine training
    # allowance. Event association changes purpose/capacity credit, not the
    # fact that this distance was run. Stage limits and readiness stay separate.
    guards = assess_program_constraints(tuple(map(demand, planned)), tuple(map(demand, history)),
        as_of=as_of, timezone=timezone, restrictions=restrictions,
        history_complete_since=history_complete_since, plan_complete=True, decision_date=decision,
        reported_distance=context.reported_distance)
    suitability = assess_program_decision(decision_input, planned, history, as_of=as_of,
        timezone=timezone, constraints=guards, easy_duration=evaluated.parts.get('easy_duration'))
    return dict(evaluation=evaluated, constraints=guards, decision=decision,
        suitability=suitability, decision_eligible=suitability['candidate_eligible'],
        constraint_eligible=guards['status'] == 'passes_checked_constraints_conditionally',
        selection_ready=False, opening_coverage_bounds=opening_bounds,
        quality_credit=purpose_credits,
        physical_reference_quality={r.key:r.quality for r in rows}, accounting_key=accounting_key,
        limitations=['Conditional score is not readiness or authorization to train',
            'Quality reference is an explicit stage comparison, not retrospective adherence',
            'Different verified quality purposes do not fulfill this same reference',
            'Purpose and demand uncertainty must be varied independently',
            'One run per local date; same-day future second runs and overnight runs need separate support'])


def _json(value):
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def _canonical_json(value):
    # Same instant must retain the same identity across snapshot UTC restoration.
    # Legacy contracts retain their historical (offset-sensitive) identity bytes.
    if isinstance(value,datetime):
        return value.astimezone(utc_timezone.utc).isoformat()
    return _json(value)


@lru_cache(maxsize=128)
def reference_key(reference):
    return sha256(identity_dumps(identity_fields(reference), sort_keys=True, allow_nan=False).encode()).hexdigest()
