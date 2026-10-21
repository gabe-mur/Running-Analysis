"""Offline coaching-contract decisions for individual and whole-program candidates."""
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime
from math import isfinite
from zoneinfo import ZoneInfo

from .athlete_projection import aware
from .development_report import ProgramOption
from .program_constraints import assess_program_constraints
from .recency_demand import compare_recency
from .workload_accounting import Quantity
from .recovery_response import ResponseExpectation, assess_response

QUALITY = {'threshold', 'high_aerobic', 'speed_skill', 'race_specific'}


@dataclass(frozen=True)
class DoseLimits:
    miles: Quantity
    work_seconds: Quantity | None = None
    bout_seconds: Quantity | None = None
    recovery_work_ratio: Quantity | None = None
    running_minutes: Quantity | None = None

    def __post_init__(self):
        if self.miles.low <= 0 or any(q is not None and q.low <= 0 for q in
                (self.work_seconds, self.bout_seconds, self.running_minutes)):
            raise ValueError('Meaningful positive dose limits required')


@dataclass(frozen=True)
class CoachingContract:
    issued_at: datetime
    source: str
    purpose: str
    intensity_key: str
    ordinary: DoseLimits
    reduced: DoseLimits | None
    trial: DoseLimits | None = None
    trial_target: float | None = None  # miles for easy/long; work seconds for quality

    def __post_init__(self):
        aware(self.issued_at)
        if not self.source or not self.intensity_key or self.purpose not in {'easy', 'long', *QUALITY}:
            raise ValueError('Explicit purpose, intensity and coaching provenance required')
        for limits in (self.ordinary, self.reduced, self.trial):
            if limits is None:
                continue
            if self.purpose in QUALITY and any(q is None for q in
                    (limits.work_seconds, limits.bout_seconds, limits.recovery_work_ratio)):
                raise ValueError('Quality contract needs total work, bout and recovery limits')
        main = 'work_seconds' if self.purpose in QUALITY else 'miles'
        if self.reduced is not None and getattr(self.reduced, main).high >= getattr(self.ordinary, main).low:
            raise ValueError('Reduced dose must be meaningfully separate from ordinary band')
        if (self.trial is None) != (self.trial_target is None):
            raise ValueError('A trial needs its own committed target, not just an acceptance band')
        if self.trial_target is not None:
            q = getattr(self.trial, main)
            if not isfinite(self.trial_target) or not q.low <= self.trial_target <= q.high:
                raise ValueError('Trial target must lie within its explicit limits')
            if main == 'miles' and abs(self.trial_target*10-round(self.trial_target*10)) > 1e-8:
                raise ValueError('Trial distance must commit to one decimal')


@dataclass(frozen=True)
class SessionAlternative:
    key: str
    program: ProgramOption
    target_key: str
    intensity_key: str
    between_recovery_seconds: tuple[float, ...] = ()

    def __post_init__(self):
        if not self.key or not self.intensity_key or self.target_key not in {e.key for e in self.program.demands}:
            raise ValueError('Identified planned opportunity and intensity required')
        bouts = self.event.structure or ()
        if len(self.between_recovery_seconds) != max(0, len(bouts)-1) or any(
                not isfinite(x) or x < 0 for x in self.between_recovery_seconds):
            raise ValueError('Every between-work recovery must be supplied')

    @property
    def event(self):
        return next(e for e in self.program.demands if e.key == self.target_key)

    @property
    def exposure(self):
        return next(e for e in self.program.exposures if e.key == self.target_key)


@dataclass(frozen=True)
class CurrentContext:
    applies_on: date
    available_at: datetime
    state: str
    source: str

    def __post_init__(self):
        aware(self.available_at)
        if type(self.applies_on) is not date or not self.source or self.state not in {
                'unknown', 'usual', 'little_tired', 'health_concern', 'conflicting', 'unverified'}:
            raise ValueError('Explicit dated current context required')


@dataclass(frozen=True)
class PrescriptionBinding:
    key: str
    purpose: str
    intensity_key: str
    contract: CoachingContract

    def __post_init__(self):
        if not self.key or not self.intensity_key or self.purpose not in {'easy', 'long', *QUALITY}:
            raise ValueError('Explicit prescription identity, purpose and intensity required')


@dataclass(frozen=True)
class DatedResponse:
    key: str
    expected_at: datetime | None
    available_at: datetime
    expectation: ResponseExpectation
    source: str
    effort: float | None = None
    feeling: float | None = None
    conflict: bool = False

    def __post_init__(self):
        if self.expected_at is not None:
            aware(self.expected_at)
        elif self.expectation.effort is not None or self.expectation.feeling is not None:
            raise ValueError('Response bounds require a pre-issued expectation timestamp')
        aware(self.available_at)
        if not self.key or not self.source or (self.expected_at is not None and self.expected_at > self.available_at):
            raise ValueError('Identified response and pre-issued expectation required')
        assess_response(self.expectation, effort=self.effort, feeling=self.feeling,
                        feedback_conflict=self.conflict)


@dataclass(frozen=True)
class ProgramDecisionInput:
    stage_key: str
    issued_at: datetime
    source: str
    prescriptions: tuple[PrescriptionBinding, ...]
    current: CurrentContext | None = None
    responses: tuple[DatedResponse, ...] = ()

    def __post_init__(self):
        aware(self.issued_at)
        object.__setattr__(self, 'prescriptions', tuple(self.prescriptions))
        object.__setattr__(self, 'responses', tuple(self.responses))
        if (not self.stage_key or not self.source or
                len({p.key for p in self.prescriptions}) != len(self.prescriptions) or
                len({r.key for r in self.responses}) != len(self.responses)):
            raise ValueError('Sourced frozen stage and unique prescription/response identities required')


def assess_program_decision(inputs, planned, history, *, as_of, timezone, constraints,
                            easy_duration=None):
    """R2 candidate suitability; sequence demand remains in the shared objective.

    Accepts the existing accounted-run interface, not a second load calculation.
    Scores do not bypass contracts/health. No lead-in-increase veto, fatigue
    ceiling, sensitivity vote or incumbent preference. This assesses supplied
    stage contracts; it does not construct/accept an athlete's training stage.
    """
    aware(as_of)
    today = as_of.astimezone(ZoneInfo(timezone)).date()
    base = dict(decision_policy_version='r2_boundary_v1', r1_policy='r1-provisional-2026-09-27-v1',
        candidate_eligible=False, physiological_recovery_established=False,
        selection_ready=False, session_checks={}, responses={}, reasons=[],
        sequence_assessment='contracts_and_constraints_plus_separately_scored_whole_sequence_demand')
    if inputs is None:
        return dict(base, status='not_assessed', action='review_policy_inputs',
                    reasons=['decision_inputs_missing'])
    if inputs.issued_at > as_of:
        raise ValueError('Decision policy is not yet available')
    bindings = {p.key: p for p in inputs.prescriptions}
    observed = {r.key: r for r in history}
    current = inputs.current
    if current and (current.available_at > as_of or current.applies_on > today):
        current = None
    fresh = current if current and (current.applies_on == today or
        current.state in {'health_concern', 'conflicting', 'unverified'}) else None
    state = fresh.state if fresh else 'unknown'
    base.update(stage_key=inputs.stage_key, context_state=state,
                policy_source=inputs.source, context_source=fresh.source if fresh else None)
    concerns = []
    for report in inputs.responses:
        if report.available_at > as_of:
            continue
        run = observed.get(report.key)
        if run is None or (report.expected_at is not None and report.expected_at > run.start) or report.available_at < run.end:
            raise ValueError('Response must match completed work and a pre-run expectation')
        assessed = assess_response(report.expectation, effort=report.effort,
            feeling=report.feeling, feedback_conflict=report.conflict)
        base['responses'][report.key] = dict(assessed, channels={name: dict(channel,
            expected=asdict(channel['expected']) if channel['expected'] is not None else None)
            for name, channel in assessed['channels'].items()})
        if assessed['review_needed'] and not (fresh and fresh.available_at >= report.available_at
                                              and state in {'usual', 'little_tired'} and not report.conflict):
            concerns.append(report.key)
    if state in {'health_concern', 'conflicting'}:
        return dict(base, status='context_review', action='review_current_context',
                    reasons=['current_health_or_conflicting_context'])
    if state == 'unverified':
        return dict(base, status='context_review', action='request_current_context',
                    reasons=['current_report_time_unknown'])
    if concerns or (current and current.state == 'little_tired' and fresh is None):
        return dict(base, status='context_review', action='request_current_context',
                    reasons=['unresolved_response_or_stale_fatigue'], response_keys=concerns)

    reasons = []
    if planned and constraints['status'] != 'passes_checked_constraints_conditionally':
        reasons.append('program_constraints_not_satisfied')
    if easy_duration is None:
        reasons.append('easy_duration_policy_missing')
    else:
        if easy_duration.get('supporting_easy_volume') is None:
            reasons.append('supporting_easy_volume_policy_missing')
        if easy_duration['below_minimum_keys']:
            reasons.append('planned_easy_below_minimum')
        if easy_duration['uncertain_minimum_keys']:
            reasons.append('planned_easy_minimum_uncertain')

    def fits(run, limits):
        if limits is None or not _inside(run.miles, limits.miles):
            return False
        # A declared command is distinct from uncertainty about its execution.
        # Older callers without that contract retain their bounds-only behavior.
        command = getattr(run,'command_running_minutes',None)
        durations = (command,) if command is not None and run.source == 'planned' else (run.demand[0].low,run.demand[0].high)
        if limits.running_minutes and not all(_inside(v, limits.running_minutes) for v in durations):
            return False
        if limits.work_seconds and not all(_inside(v*60, limits.work_seconds)
                                           for v in (run.demand[2].low, run.demand[2].high)):
            return False
        if limits.bout_seconds:
            if not run.bouts or any(not all(_inside(v, limits.bout_seconds) for v in
                    (b.duration_range_seconds or (b.work_seconds, b.work_seconds))) for b in run.bouts):
                return False
        if limits.recovery_work_ratio:
            for b in run.bouts[:-1]:
                if b.recovery_after_seconds is None:
                    return False
                bounds = b.duration_range_seconds or (b.work_seconds, b.work_seconds)
                if not all(_inside(b.recovery_after_seconds/v, limits.recovery_work_ratio) for v in bounds):
                    return False
        return True

    actions = []
    for run in sorted(planned, key=lambda r: (r.start, r.key)):
        binding = bindings.get(run.key)
        problems, modes = [], {}
        mode = None
        if binding is None:
            problems.append('prescription_contract_missing')
        else:
            contract = binding.contract
            advisory=getattr(run,'advisory_timezone',None)
            clock_unavailable=(contract.issued_at.astimezone(ZoneInfo(advisory)).date()>
                run.start.astimezone(ZoneInfo(advisory)).date()) if advisory else contract.issued_at>run.start
            if contract.issued_at > as_of or clock_unavailable:
                problems.append('prescription_contract_not_available')
            expected_role = 'quality' if binding.purpose in QUALITY else binding.purpose
            if (run.role != expected_role or contract.purpose != binding.purpose
                    or contract.intensity_key != binding.intensity_key):
                problems.append('prescription_purpose_or_intensity_mismatch')
            if run.role == 'quality' and run.work_evidence_basis != 'prescribed':
                problems.append('quality_work_extent_unsupported')
            modes = {name: fits(run, limits) for name, limits in (
                ('ordinary', contract.ordinary), ('reduced', contract.reduced), ('trial', contract.trial))}
            # A trial band's upper edge is not a newly committed growth target.
            if modes['trial']:
                actual = sum(b.work_seconds for b in run.bouts) if contract.purpose in QUALITY else run.miles
                modes['trial'] = abs(actual-contract.trial_target) < 1e-8
            mode = next((name for name in ('trial', 'reduced', 'ordinary') if modes[name]), None)
            if mode is None:
                problems.append('outside_supported_dose_contract')
            run_today = run.start.astimezone(ZoneInfo(timezone)).date() == today
            if run_today and state == 'little_tired':
                if mode == 'trial':
                    problems.append('current_fatigue_does_not_authorize_growth')
                elif contract.purpose != 'easy' and mode != 'reduced':
                    problems.append('current_fatigue_requires_demanding_session_alternative')
        base['session_checks'][run.key] = dict(problems=problems, fits=modes, mode=mode)
        if problems:
            reasons.append('session_contract_or_context_not_satisfied')
        elif run.start.astimezone(ZoneInfo(timezone)).date() == today:
            actions.append({'ordinary': 'continue_provisionally', 'reduced': 'propose_reduced_trial',
                            'trial': 'propose_bounded_trial'}[mode])
    if reasons:
        return dict(base, status='requires_alternative_or_review', action='review_or_adapt_candidate',
                    reasons=sorted(set(reasons)))
    return dict(base, status='provisionally_supported', candidate_eligible=True,
        action=actions[0] if actions else ('rest_today_future_conditional' if planned else 'rest_candidate'),
        future_context_recheck_required=any(r.start.astimezone(ZoneInfo(timezone)).date() > today for r in planned))


def _inside(value, bounds):
    return bounds is None or value is not None and bounds.low-1e-8 <= value <= bounds.high+1e-8


def _metrics(option):
    bouts = option.event.structure or ()
    ratios = tuple(r/w for r, w in zip(option.between_recovery_seconds, bouts))
    return dict(miles=option.exposure.miles, work_seconds=sum(bouts),
                bouts=bouts, recovery_ratios=ratios)


def _minute_bounds(option):
    terms = [t for t in option.event.terms if t.channel == 'running_minutes' and t.coefficient]
    if not terms or any(t.bounds is None for t in terms):
        return None
    return tuple(sum(t.coefficient*getattr(t.bounds, bound) for t in terms) for bound in ('low', 'high'))


def _fits(option, limits):
    m = _metrics(option)
    if not _inside(m['miles'], limits.miles) or not _inside(m['work_seconds'], limits.work_seconds):
        return False
    if limits.bout_seconds and (not m['bouts'] or not all(_inside(v, limits.bout_seconds) for v in m['bouts'])):
        return False
    if limits.recovery_work_ratio and not all(_inside(v, limits.recovery_work_ratio) for v in m['recovery_ratios']):
        return False
    if limits.running_minutes:
        minutes = _minute_bounds(option)
        if minutes is None:
            return False
        lo, hi = minutes
        if not (_inside(lo, limits.running_minutes) and _inside(hi, limits.running_minutes)):
            return False
    return True


def _same_surrounding_program(issued, candidate):
    return candidate.target_key == issued.target_key and all(
        {e.key: e for e in getattr(issued.program, attr) if e.key != issued.target_key} ==
        {e.key: e for e in getattr(candidate.program, attr) if e.key != candidate.target_key}
        for attr in ('exposures', 'demands'))


def _preceding(option, history):
    return (*history, *(e for e in option.program.demands if e.day < option.event.day))


def _context_comparison(issued, candidate, expected_history, history):
    # Exclude the receiving dose to distinguish lead-in changes from intended growth.
    zero = lambda e: replace(e, terms=tuple(replace(t, coefficient=0.) for t in e.terms))
    return compare_recency(_preceding(issued, expected_history), zero(issued.event),
                           _preceding(candidate, history), zero(candidate.event))


def decide_session(contract, issued, alternatives, *, as_of, history, expected_history,
                   context=None, restrictions=(), history_complete_since=None,
                   unexpected_previous_response=False, trial_requested=False, timezone='UTC'):
    """Action proposal conditional on a justified frozen contract, never recovery clearance.

    No raw workout feeling/RPE is converted to current readiness here. Bounds
    must be justified upstream; fixture bounds do not become production defaults.
    """
    aware(as_of)
    today = as_of.astimezone(ZoneInfo(timezone)).date()
    if contract.issued_at > as_of or issued.event.day < today:
        raise ValueError('Available issued contract and future opportunity required')
    if issued.event.purpose != contract.purpose or issued.intensity_key != contract.intensity_key:
        raise ValueError('Issued purpose/intensity does not match coaching contract')
    if not _fits(issued, contract.ordinary):
        raise ValueError('Issued dose lies outside its own ordinary contract')
    alternatives, history, expected_history, restrictions = map(tuple, (alternatives, history, expected_history, restrictions))
    options = (issued, *alternatives)
    if len({o.key for o in options}) != len(options):
        raise ValueError('Unique candidate identities required')
    known_context = context if context and context.available_at <= as_of and context.applies_on <= today else None
    current = known_context if known_context and (known_context.applies_on == today or
                                                  known_context.state in {'health_concern', 'conflicting'}) else None
    state = current.state if current else 'unknown'
    result = dict(action=None, selected_option_key=None, context_state=state,
        context_source=current.source if current else None, contract_source=contract.source,
        option_checks={}, planner_influence=False, running_authorized=False,
        progression_accepted=False, physiological_recovery_established=False,
        stage_delivery_recheck_required=False, next_date_readiness_recheck_required=False,
        limitations=['Conditional on upstream justification of coaching limits and original program',
            'Current context is explicit input, not inferred from workout RPE or low HR',
            'Single-opportunity proposal; downstream demand and stage delivery are not cleared'])
    def finish(action, reason, option=None):
        result.update(action=action, reason=reason, selected_option_key=option.key if option else None)
        if option:
            result['proposed_day'] = option.event.day
            result['stage_delivery_recheck_required'] = option.key != issued.key
            result['next_date_readiness_recheck_required'] = option.event.day > today
            result['program_miles_change'] = option.exposure.miles-issued.exposure.miles
            result['quality_work_seconds_change'] = _metrics(option)['work_seconds']-_metrics(issued)['work_seconds']
            result['purpose'] = option.event.purpose
            downstream = []
            for target in option.program.demands:
                if target.key == option.target_key or target.day <= min(issued.event.day, option.event.day):
                    continue
                original = next(e for e in issued.program.demands if e.key == target.key)
                comparison = compare_recency(
                    (*expected_history, *(e for e in issued.program.demands if e.day < original.day)), original,
                    (*history, *(e for e in option.program.demands if e.day < target.day)), target)
                downstream.append(dict(key=target.key, day=target.day,
                    recency_demand_order=comparison['recency_demand_order'],
                    channel_orders=comparison['channel_orders']))
            result['downstream_demand_comparisons'] = downstream
        return result
    if state in {'health_concern', 'conflicting'}:
        return finish('review_current_context', 'A health concern or conflicting current reports cannot be cleared by dose reduction or waiting')
    if state == 'little_tired' and issued.event.day > today:
        return finish('request_session_date_context', 'Current fatigue cannot determine the dose on a future date')
    if (unexpected_previous_response or known_context and known_context.state == 'little_tired') and state == 'unknown':
        return finish('request_current_context', 'Unexpected past response does not specify current fatigue or a reduction dose')
    eligible = []
    for option in options:
        problems = []
        if not _same_surrounding_program(issued, option):
            problems.append('surrounding_program_changed')
        if option.event.purpose != contract.purpose or option.intensity_key != contract.intensity_key:
            problems.append('different_purpose_or_intensity')
        if option.event.day < issued.event.day:
            problems.append('earlier_date_requires_spacing_trial')
        if problems:
            result['option_checks'][option.key] = dict(problems=problems)
            continue
        constraints = assess_program_constraints(option.program.demands, history, as_of=as_of, timezone=timezone,
            restrictions=restrictions, history_complete_since=history_complete_since, plan_complete=False)
        lead_in = _context_comparison(issued, option, expected_history, history)
        # Proven positive cumulative difference in candidate demand, not interval overlap.
        greater = [channel for channel, curve in lead_in['cutoff_differences'].items()
                   if any(row['left_minus_right_high'] is not None and row['left_minus_right_high'] < -1e-8 for row in curve)]
        if greater:
            problems.append('greater_preceding_demand_requires_review')
        if constraints['status'] != 'passes_checked_constraints_conditionally':
            problems.append('program_constraints_not_satisfied')
        fits = {name: _fits(option, limits) if limits else False for name, limits in
                (('ordinary', contract.ordinary), ('reduced', contract.reduced), ('trial', contract.trial))}
        result['option_checks'][option.key] = dict(problems=problems, fits=fits,
            constraints=constraints, lead_in_channel_orders=lead_in['channel_orders'],
            greater_preceding_channels=greater)
        if not problems:
            eligible.append(option)
    main = 'work_seconds' if contract.purpose in QUALITY else 'miles'
    def preserve_dose(option):
        m = _metrics(option)
        return (-m[main], -m['miles'], option.key)
    same_day = [o for o in eligible if o.event.day == issued.event.day]
    if state == 'little_tired':
        # A mild current fatigue report is not a blanket dose-reduction order.
        # An ordinary easy prescription remains available only after its existing
        # contract, lead-in and whole-program constraint checks have passed.
        # This does not extend to a long run just because its intensity is easy.
        if contract.purpose == 'easy' and issued in same_day:
            return finish('continue_provisionally',
                'Mild current fatigue alone does not require reducing a familiar easy dose', issued)
        base = _metrics(issued)
        def reduction(o):
            m = _metrics(o)
            before_minutes, after_minutes = _minute_bounds(issued), _minute_bounds(o)
            duration_increased = before_minutes is not None and after_minutes is not None and after_minutes[1] > before_minutes[1]+1e-8
            return (result['option_checks'][o.key]['fits']['reduced'] and m['miles'] <= base['miles'] and
                    m['work_seconds'] <= base['work_seconds'] and
                    not duration_increased and
                    max(m['bouts'], default=0) <= max(base['bouts'], default=0) and
                    min(m['recovery_ratios'], default=float('inf')) >= min(base['recovery_ratios'], default=float('inf')))
        choices = sorted((o for o in same_day if reduction(o)), key=preserve_dose)
        return finish('propose_reduced_trial', 'Current little-tired context invokes the predeclared meaningful reduction', choices[0]) if choices else finish(
            'review_adaptation', 'No supported same-day meaningful reduction; do not invent tomorrow recovery')
    if trial_requested:
        choices = sorted((o for o in same_day if contract.trial and result['option_checks'][o.key]['fits']['trial']
                          and _metrics(o)[main] > _metrics(issued)[main]),
                         key=lambda o: (round(abs(_metrics(o)[main]-contract.trial_target), 8), _metrics(o)[main], o.key))
        if choices:
            return finish('propose_bounded_dose_trial', 'Explicit trial limits, same date and unchanged surrounding program', choices[0])
        return finish('review_trial', 'No supported dose-only trial within the frozen limits')
    if issued in eligible:
        return finish('continue_provisionally', 'Issued contract remains supported; ordinary uncertainty does not force rest', issued)
    check = result['option_checks'][issued.key]
    blocked = [r for row in check.get('constraints', {}).get('sessions', [])
               if row['key'] == issued.target_key for r in row['restrictions']]
    if blocked and all(r['kind'] == 'reserved_rest' for r in blocked):
        choices = sorted((o for o in eligible if o.event.day > issued.event.day
                          and result['option_checks'][o.key]['fits']['ordinary']
                          and _metrics(o) == _metrics(issued)), key=lambda o: (o.event.day, o.key))
        if choices:
            return finish('propose_tentative_deferral', 'Reserved-date conflict; readiness must be rechecked on the proposed date', choices[0])
    return finish('review_adaptation', 'Constraint or lead-in change needs a justified adjustment, not automatic extra rest')
