"""Offline joined evidence report. No candidate selection or feasibility clearance."""
from dataclasses import dataclass
from math import isclose
from zoneinfo import ZoneInfo

from .program_delivery import ProgramExposure, audit_program, compare_allocation, screen_development
from .recency_demand import DemandEvent, compare_recency
from .staged_progression import DevelopmentStage, review_stage
from .program_constraints import assess_program_constraints


@dataclass(frozen=True)
class ProgramOption:
    key: str
    exposures: tuple[ProgramExposure, ...]
    demands: tuple[DemandEvent, ...]

    def __post_init__(self):
        if not self.key:
            raise ValueError('Named program option required')
        e = {x.key: x for x in self.exposures}
        d = {x.key: x for x in self.demands}
        if len(e) != len(self.exposures) or len(d) != len(self.demands) or e.keys() != d.keys():
            raise ValueError('Unique one-to-one exposure/demand identities required')
        for key, exposure in e.items():
            demand = d[key]
            purpose = exposure.quality_purpose if exposure.role == 'quality' else exposure.role
            if (exposure.source != 'planned' or demand.source != 'planned' or exposure.day != demand.day
                    or (purpose or 'unknown') != demand.purpose or exposure.work_bout_seconds != demand.structure):
                raise ValueError('Same planned date, source, purpose and structure required')
            terms = [t for t in demand.terms if t.channel == 'miles']
            if any(t.bounds is None for t in terms):
                raise ValueError('Prescribed demand distance must be known')
            if any(not isclose(sum(t.coefficient*getattr(t.bounds, bound) for t in terms), exposure.miles,
                               abs_tol=1e-8, rel_tol=0) for bound in ('low', 'high')):
                raise ValueError('Funding and recency must describe the same distance')
            if exposure.work_bout_seconds is not None:
                work = [t for t in demand.terms if t.channel == 'work_minutes']
                expected = sum(exposure.work_bout_seconds)/60
                if any(t.bounds is None for t in work) or any(not isclose(
                        sum(t.coefficient*getattr(t.bounds, bound) for t in work), expected,
                        abs_tol=1e-8, rel_tol=0) for bound in ('low', 'high')):
                    raise ValueError('Work-demand accounting must match prescribed bouts')


@dataclass(frozen=True)
class OpportunityPair:
    reference_key: str
    candidate_key: str


def development_report(stage: DevelopmentStage, reference: ProgramOption, candidate: ProgramOption, *,
        as_of, reference_targets, candidate_targets, coverage_start, coverage_end,
        evaluation_starts, spans=(21, 28, 42), coverage_complete=False,
        history=(), opportunity_pairs=(), entries=(), ledger_complete=False,
        longest_prior_30d_miles=None, responses=(), timezone='UTC',
        restrictions=(), history_complete_since=None):
    """Recompute linked evidence from inputs, not caller-supplied 'passed' flags.

    Candidate funding targets remain explicit assumptions, not estimated capacity.
    This API intentionally has no feasibility override or acceptance path.
    """
    if reference.key == candidate.key:
        raise ValueError('Distinct option identities required')
    review = review_stage(stage, entries, as_of, ledger_complete=ledger_complete,
        longest_prior_30d_miles=longest_prior_30d_miles, responses=responses)
    today = as_of.astimezone(ZoneInfo(timezone)).date()
    if coverage_start < today:
        raise ValueError('Future-program coverage must not imply observed history completeness')
    history, pairs = tuple(history), tuple(opportunity_pairs)
    if (len({e.key for e in history}) != len(history) or
            any(e.source != 'observed' or e.day >= today for e in history)):
        raise ValueError('Unique prior observed history required; same-day timing unsupported')
    if any(h.key in {e.key for e in option.exposures} for option in (reference, candidate) for h in history):
        raise ValueError('Actual exposure cannot also appear as projected work')
    if len({e.accounting_key for e in (*history, *reference.demands, *candidate.demands)}) > 1:
        raise ValueError('All demand evidence must share cardio accounting units')
    audit_kwargs = dict(coverage_start=coverage_start, coverage_end=coverage_end,
        evaluation_starts=tuple(evaluation_starts), spans=tuple(spans), coverage_complete=coverage_complete)
    before = audit_program(reference.exposures, tuple(reference_targets), **audit_kwargs)
    after = audit_program(candidate.exposures, tuple(candidate_targets), **audit_kwargs)
    changes = compare_allocation(before, after)
    allocation = screen_development(before, after)
    a, b = ({e.key: e for e in option.demands} for option in (reference, candidate))
    if len({p.reference_key for p in pairs}) != len(pairs) or len({p.candidate_key for p in pairs}) != len(pairs):
        raise ValueError('Each matched opportunity must be unique')
    comparisons = []
    def preceding(option, event):
        if any(e.key != event.key and e.day == event.day for e in option.demands):
            raise ValueError('Same-day multiple sessions need explicit timing support')
        return (*history, *(e for e in option.demands if e.day < event.day))
    for pair in pairs:
        if pair.reference_key not in a or pair.candidate_key not in b:
            raise ValueError('Comparison refers to an absent program opportunity')
        x, y = a[pair.reference_key], b[pair.candidate_key]
        if x.day < today or y.day < today:
            raise ValueError('Comparison precedes report date')
        comparisons.append(compare_recency(preceding(reference, x), x, preceding(candidate, y), y))
    def goal_position(option):
        longs = [e for e in option.exposures if e.role == 'long']
        return dict(long_prescriptions=[dict(key=e.key, day=e.day, miles=e.miles,
            gap_to_destination_miles=max(0., stage.goal_long_miles-e.miles),
            exceeds_destination=e.miles > stage.goal_long_miles) for e in longs],
            training_benefit_established=False,
            note='Prescription distance toward a destination, not demonstrated capacity or benefit')
    inspected_start = min(r['start'] for r in after['intervals'])
    inspected_end = max(r['end'] for r in after['intervals'])
    matched = {p.candidate_key for p in pairs}
    unmatched = [e.key for e in candidate.exposures
                 if inspected_start <= e.day < inspected_end and e.key not in matched]
    constraints = assess_program_constraints(candidate.demands, history, as_of=as_of,
        timezone=timezone, restrictions=restrictions, history_complete_since=history_complete_since,
        plan_complete=coverage_complete and coverage_start == today)
    unresolved = ['absolute_dose_specific_feasibility', 'training_benefit_and_demand_tradeoff_policy',
                  'purpose_specific_stimulus_and_substitution_policy']
    if constraints['status'] != 'passes_checked_constraints_conditionally':
        unresolved.append('candidate_constraints_require_resolution')
    reasons = list(allocation['review_reasons'])
    if constraints['status'] == 'constraint_conflict':
        reasons.append('explicit_program_constraint_conflict')
    if unmatched:
        unresolved.append('uncompared_candidate_opportunities')
    if review['response_state'] == 'unknown':
        unresolved.append('response_unknown_not_clearance')
    return dict(as_of=as_of, stage_key=stage.key, reference_key=reference.key, candidate_key=candidate.key,
        execution_and_stage_review=review,
        program_delivery=dict(reference=before, candidate=after, allocation_changes=changes, screen=allocation),
        goal=dict(destination_long_miles=stage.goal_long_miles, reference=goal_position(reference),
                  candidate=goal_position(candidate)),
        demand=dict(left_option=reference.key, right_option=candidate.key, comparisons=comparisons,
            unmatched_candidate_keys=unmatched, aggregate_winner=None,
            interpretation='Conditional demand orders, not physiological readiness or overall superiority'),
        response=dict(state=review['response_state'], source=review['response_source'],
            concern_present=review['response_state'] in {'adverse', 'health_concern'},
            unknown_is_clearance=False),
        feasibility=dict(status='unresolved', unmet_requirements=unresolved,
            constraint_checks=constraints, known_review_reasons=reasons,
            note='No evidence of a conflict is not evidence of feasibility'),
        next_step='resolve_feasibility_and_tradeoff_policy_before_acceptance_or_scheduler_integration',
        selected_candidate=None, progression_accepted=False, running_authorized=False, planner_influence=False)
