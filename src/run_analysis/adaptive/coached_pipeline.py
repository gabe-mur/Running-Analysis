"""Explicit sandbox coaching hooks for the actual live daily planning path.

No new fatigue model. Commands are bound before full physical evaluation;
the compiler uses the identical continuation reference as that evaluator.
"""
from dataclasses import replace
from collections import OrderedDict
from datetime import date
from .quality_coaching_policy import continuation_reference, evaluate_coached_program
from .quality_assignment_accounting import issue_assignment
from .program_evidence_adapter import stage_quality_credit
from .continuous_program_adapter import evaluate_continuous
from .decision_snapshot import evaluate_snapshot
from run_analysis.workload_accounting import Quantity
from run_analysis.continuous_training_dose import ContinuousDoseReference

# Search already retains every compact score. Keeping every complete diagnostic
# trace as well makes a full search grow by hundreds of MB. Evicted details are
# reproducibly repriced from this evaluator's frozen inputs, never approximated.
DETAIL_CACHE_LIMIT = 128


def connected_evaluator(sequence, quality_reference, completions=(), *, demand_interaction='independent_v1'):
    """Same evaluator for fresh search, immutable snapshot replay and restart."""
    receipts={r.activity_key:r for r in completions}
    if len(receipts)!=len(completions):raise ValueError('Unique observed quality completion receipts required')
    def connected(context, history, planned, **kwargs):
        issued={};rank=0
        for run in sorted(planned,key=lambda r:r.start):
            if run.role=='quality':
                assignment=sequence.assignment(rank);rank+=1
                if run.key.split(':',2)[2]!=assignment.key:
                    raise ValueError('Unbound quality command cannot receive fulfillment credit')
                issued[run.key]=assignment
        def credit(run):
            if run.key in receipts:return receipts[run.key].for_run(run)
            if run.role!='quality':return Quantity.point(0.,'Not a quality opportunity')
            if run.source=='planned':
                if run.key not in issued:raise ValueError('Missing frozen quality assignment')
                return Quantity.point(1.,'Expected fulfillment of exact issued ordinary command')
            return (context.quality_programme.delivery(run.key,run.source).opportunity
                if context.quality_programme is not None else stage_quality_credit(run,context.policy))
        def program_evaluator(context,candidate):
            return evaluate_coached_program(context,candidate,sequence=sequence,
                quality_reference=quality_reference,demand_interaction=demand_interaction)
        return evaluate_continuous(context,history,planned,program_evaluator=program_evaluator,
            quality_credit_provider=credit,quality_cadence_contract=sequence.cadence_contract,**kwargs)
    return connected


def coached_evaluator(case, sequence, *, completions=(), demand_interaction='independent_v1'):
    state, catalog, dates = (case[k] for k in ('state', 'catalog', 'dates'))
    bind = lambda program: sequence.bind(program, catalog)
    quality_reference = continuation_reference(sequence, case['physical'], dates[0])
    ref = ContinuousDoseReference('sandbox upload accounting')
    cache = OrderedDict()
    effective_state = replace(state, context=replace(state.context, quality_programme=None,
        continuation_boundary_references=tuple(quality_reference if r[0]=='quality' else r
            for r in state.context.continuation_boundary_references)))

    connected=connected_evaluator(sequence,quality_reference,completions,demand_interaction=demand_interaction)

    def detail(program):
        program=tuple(program)
        if program!=bind(program):raise ValueError('Unbound proposal cannot be priced as coached physical work')
        if program not in cache:
            cache[program]=evaluate_snapshot(state,tuple(case['physical'][p] for p in program),
                tuple(k for _,k in program),reference=ref,evaluator=connected)
        raw=cache[program]
        cache.move_to_end(program)
        while len(cache)>DETAIL_CACHE_LIMIT:
            cache.popitem(last=False)
        if state.race_goal is not None:
            from datetime import date
            from .race_accounting import allowed_dose
            if any(not allowed_dose(state.race_goal,date.fromisoformat(day),catalog[key],
                    case.get('stage',state.context.policy),state.context.race_event) for day,key in program):
                return dict(raw,decision_eligible=False,construction_eligible=False,
                    constraints=dict(raw['constraints'],status='dated_race_phase_restriction'))
        return raw

    def evaluate(program):
        raw=detail(program);e=raw['evaluation']
        return dict(cost=e.total_cost,eligible=raw['decision_eligible'],
            construction_eligible=raw.get('construction_eligible',raw['decision_eligible']),
            model_version=e.model_version,score_fingerprint=e.fingerprint,
            constraints=raw['constraints']['status'])
    return dict(detail=detail,evaluate=evaluate,cache=cache,quality_reference=quality_reference,
        bind=bind,connected=connected,effective_state=effective_state)


def fund_bound_prior(program, *, dates, catalog, physical, state, evaluate):
    """Retain a complete prior as a search candidate after command substitution.

    A smaller new quality command may unfund the prior. Enumerate single easy
    additions in the frozen review interval; never move dates, relax admission,
    or force adoption. The ordinary search still compares other calendars.
    """
    original=evaluate(program)
    if original['eligible']:return program,dict(evaluations=1,changed=False,already_admitted=True)
    restrictions=getattr(state,'restrictions',())
    if any(r.available_at<=state.as_of and any(r.start<=date.fromisoformat(day)<r.end
            for day,_ in program) for r in restrictions):
        # Enlarging an easy dose cannot repair a blocked date. Let the shared
        # calendar repair offer deletion/relocation, not hundreds of impossible
        # dose-only evaluations outside the main search budget.
        return program,dict(evaluations=1,changed=False,already_admitted=False,
            reason='date_constraint_needs_calendar_repair')
    if original.get('constraints')=='preferred_long_run_day_conflict':
        return program,dict(evaluations=1,changed=False,already_admitted=False,
            reason='weekday_change_needs_calendar_search')
    review=state.prospective_volume_review
    candidates=[];calls=1
    for day,key in program:
        if catalog[key].role!='easy' or (review is not None and not
                review.start<=physical[day,key].start<review.end):continue
        for k,dose in sorted(catalog.items()):
            if dose.role!='easy' or dose.miles<=catalog[key].miles or (day,k) not in physical:continue
            p=tuple((d,k if d==day else old) for d,old in program)
            r=evaluate(p);calls+=1
            if r['eligible']:candidates.append((r['cost'],p))
    winner=min(candidates)[1] if candidates else program
    return winner,dict(evaluations=calls,changed=winner!=program,
        already_admitted=False,admitted_alternatives=len(candidates))
