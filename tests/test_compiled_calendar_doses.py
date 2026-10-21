from dataclasses import replace
from datetime import datetime, time, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from test_continuation_boundary_trial import setup
from test_soft_horizon_trial import candidate
from run_analysis.adaptive.compiled_calendar_doses import compile_calendar, neighbors, refine
from run_analysis.adaptive.program_objective_trial import DemandScenario
from run_analysis.adaptive.shared_program_evaluator import evaluate_program
from run_analysis.workload_accounting import Quantity


def fixture(*, fraction=.5, weights=(1.,1.,1.), compression=0., single_volume=True, context_modifier=None):
    context, refs, rows, _ = setup()
    context = replace(context, policy=replace(context.policy,
        purpose_weights=(1.,0. if single_volume else 1.,1.,1.)),
        scenario=DemandScenario(bound_fraction=fraction, axis_weights=weights,
            timing_compression_days=compression),
        continuation_boundary_references=tuple((r,m,q) for r,(m,q) in refs.items()),
        volume_linear_coefficient=.1 if single_volume else 0.)
    if context_modifier is not None:
        context=context_modifier(context)
    easy = [r for r in rows if r.role == 'easy']
    table = {}
    for row in easy:
        for units in range(11,61):
            miles = units/10
            # Deliberately nonlinear, bounded catalogue quantities: the compiler
            # must use the values rather than assuming affine HR versus distance.
            q = tuple(Quantity(v*.8,v*1.2,v,'nonlinear fixture') for v in
                (miles/4, miles/5+.013*(units%3), miles/8+.01*(units%2)))
            begin = datetime.combine(row.day,time(18),timezone.utc)
            table[(row.day.isoformat(),f'easy:{miles:.1f}')] = SimpleNamespace(
                miles=miles,role='easy',demand=q,start=begin,end=begin+timedelta(minutes=miles*10))
    program=[]
    for row in rows:
        day = row.day.isoformat()
        key = f'easy:{row.miles:.1f}' if row.role == 'easy' else row.key
        if row.role != 'easy':
            begin=datetime.combine(row.day,time(18),timezone.utc)
            table[(day,key)] = SimpleNamespace(miles=row.miles,role=row.role,demand=row.demand,
                start=begin,end=begin+timedelta(minutes=row.miles*10))
        program.append((day,key))
    program=tuple(program)
    def details(p):
        revised=[]
        for day,key in p:
            original=next(r for r in rows if r.day.isoformat()==day)
            dose=table[(day,key)]
            revised.append(replace(original,miles=dose.miles,demand=dose.demand))
        evaluation=evaluate_program(context,candidate(context,revised))
        return dict(evaluation=evaluation,decision_eligible=True)
    def evaluate(p):
        r=details(p)
        return dict(cost=r['evaluation'].total_cost,eligible=r['decision_eligible'])
    state=SimpleNamespace(context=context,prospective_volume_review=None)
    compiled=compile_calendar(state,program,physical=table,result=details(program))
    return compiled,evaluate,state,table,program,details


@pytest.mark.parametrize('distance',['5k','10k','half_marathon','marathon'])
@pytest.mark.parametrize('status',['recorded','did_not_start'])
def test_resolved_event_and_dated_return_preserve_compiler_parity(distance,status):
    from run_analysis.adaptive.race_goal import RaceGoalRevision
    from run_analysis.adaptive.race_event import RaceEventResolution
    from run_analysis.adaptive.program_objective_trial import TrialSession
    def modify(ctx):
        event_day=ctx.decision-timedelta(days=1)
        at=datetime.combine(ctx.decision,time(4),timezone.utc)
        race=RaceGoalRevision.create(available_at=at-timedelta(days=30),distance=distance,
            race_date=event_day,preparation_mode='balanced',parent_key='fixture')
        key='event' if status=='recorded' else None
        outcome=RaceEventResolution.create(race_key=race.key,status=status,physical_key=key,
            available_at=at,parent_key=race.key,effective_on=ctx.decision)
        history=(TrialSession('event',event_day,10.,'race',tuple(Quantity.point(5.,'declared event fixture') for _ in range(3)),
            source='observed'),) if key else ()
        return replace(ctx,race_goal=race,race_event=outcome,history=history,
            continuation_distribution='c1_v1')
    compiled,evaluate,*_=fixture(context_modifier=modify)
    choices=[compiled.original_levels,np.zeros(len(compiled.days),dtype=int),
        np.full(len(compiled.days),len(compiled.keys)-1,dtype=int)]
    for x,value in zip(choices,compiled.cost(choices)):
        assert value==pytest.approx(evaluate(compiled.physical(x))['cost'],abs=1e-12)


@pytest.mark.parametrize('status',['recorded','did_not_start'])
def test_cleared_goal_keeps_historical_target_and_purpose_parity(status):
    from run_analysis.adaptive.race_goal import RaceGoalRevision,RaceGoalApplication,overlay_applications
    from run_analysis.adaptive.race_event import RaceEventResolution
    from run_analysis.program_delivery import TargetSegment
    def modify(ctx):
        at=datetime.combine(ctx.decision,time(4),timezone.utc)
        race=RaceGoalRevision.create(available_at=at-timedelta(days=30),distance='10k',
            race_date=ctx.decision-timedelta(days=7),preparation_mode='balanced',parent_key='fixture')
        outcome=RaceEventResolution.create(race_key=race.key,status=status,
            physical_key='event' if status=='recorded' else None,available_at=at-timedelta(days=2),
            parent_key=race.key,effective_on=ctx.decision-timedelta(days=2))
        clear=RaceGoalRevision.create(available_at=at-timedelta(days=1),distance='general_fitness',
            race_date=None,preparation_mode='balanced',parent_key=race.key)
        apps=tuple(RaceGoalApplication(r,r.available_at,r.available_at.date()) for r in (race,clear))
        def targets(rate):
            return overlay_applications((TargetSegment(ctx.decision-timedelta(days=50),ctx.end+timedelta(days=1),rate),),
                apps,timezone='UTC',resolutions=(outcome,))
        return replace(ctx,race_goal=clear,race_goal_history=apps,race_event_history=(outcome,),
            volume_targets=targets(ctx.policy.weekly_miles),
            support_targets=targets(ctx.easy_duration_policy.supporting_easy_miles_per_7),
            continuation_distribution='c1_v1',purpose_continuation=True)
    compiled,evaluate,*_=fixture(context_modifier=modify)
    levels=[compiled.original_levels,np.zeros(len(compiled.days),dtype=int),
        np.full(len(compiled.days),len(compiled.keys)-1,dtype=int)]
    for x,value in zip(levels,compiled.cost(levels)):
        assert value==pytest.approx(evaluate(compiled.physical(x))['cost'],abs=1e-12)


@pytest.mark.parametrize('fraction',[0.,.5,1.])
@pytest.mark.parametrize('weights',[(1.,1.,1.),(2.,3.,0.)])
@pytest.mark.parametrize('single_volume',[True,False])
def test_compilation_preserves_full_score_for_nonlinear_catalogue(fraction,weights,single_volume):
    compiled,evaluate,*_=fixture(fraction=fraction,weights=weights,single_volume=single_volume,
        compression=.4)
    levels=[compiled.original_levels,np.zeros(len(compiled.days),dtype=int),
        np.full(len(compiled.days),len(compiled.keys)-1,dtype=int)]
    levels.extend(np.array([(n*11+i*7)%len(compiled.keys) for i in range(len(compiled.days))])
        for n in range(10))
    for x,cost in zip(levels,compiled.cost(levels)):
        assert cost == pytest.approx(evaluate(compiled.physical(x))['cost'],abs=1e-12)


def test_all_span_pairs_and_unit_triples_are_present_and_keep_exact_volume():
    compiled,*_=fixture()
    x=compiled.original_levels
    streams=list(neighbors(compiled,x))
    assert any(np.count_nonzero(p!=x)==2 and p[0]!=x[0] and p[-1]!=x[-1] for p in streams)
    assert any(np.count_nonzero(p!=x)==3 for p in streams)
    for p in streams:
        if np.count_nonzero(p!=x)>1:
            assert compiled.miles[p].sum() == pytest.approx(compiled.miles[x].sum())


def test_refinement_rechecks_admission_and_never_changes_dates_or_roles():
    compiled,evaluate,*_=fixture()
    accepted=refine(compiled,evaluate)
    assert accepted['cost']<=accepted['original_cost']
    assert accepted['neighborhood_complete']
    assert [d for d,k in accepted['program']]==[d for d,k in compiled.program]
    assert evaluate(accepted['program'])['cost']==pytest.approx(accepted['cost'],abs=1e-12)
    def reject(p):
        return dict(evaluate(p),eligible=p==compiled.program)
    rejected=refine(compiled,reject)
    assert rejected['program']==compiled.program
    assert rejected['cost']==pytest.approx(rejected['original_cost'])
    assert any(row['rejected']>0 for row in rejected['trace'])


def test_prospective_credit_uses_checkpoint_completion_not_total_horizon_volume():
    compiled,evaluate,state,table,program,details=fixture()
    checkpoint=datetime.combine(datetime.fromisoformat(compiled.days[1]).date(),time(4),timezone.utc)
    review=SimpleNamespace(start=checkpoint-timedelta(days=30),end=checkpoint,
        available_at=checkpoint-timedelta(days=10),required_miles=20.,completed_credit=(('past',8.),))
    state.prospective_volume_review=review
    c=compile_calendar(state,program,physical=table,result=details(program))
    for i,day in enumerate(c.days):
        if datetime.fromisoformat(day).date() >= checkpoint.date():
            assert np.all(c.credit[i]==0.)
    assert c.funded(c.original_levels) == (sum(c.credit[i,j] for i,j in enumerate(c.original_levels))
        + 1e-8 >= c.required_credit)


def test_mismatch_fails_closed_instead_of_adopting_a_proxy_score():
    compiled,evaluate,*_=fixture()
    compiled.linear*=1.1
    with pytest.raises(ValueError,match='reproduce'):
        refine(compiled,evaluate)


def test_unrecognized_scoring_contract_is_not_silently_compiled():
    c,evaluate,state,table,p,details=fixture()
    state.context=replace(state.context,continuation_boundary_references=())
    with pytest.raises(ValueError,match='contract'):
        compile_calendar(state,p,physical=table,result=details(p))


def test_full_check_allowance_stops_without_claiming_neighbourhood_completion():
    compiled,evaluate,*_=fixture()
    calls=[]
    def checked(p):
        calls.append(p)
        return dict(evaluate(p),eligible=p==compiled.program)
    result=refine(compiled,checked,max_full_evaluations=2)
    assert len(calls)==result['full_evaluations']==2
    assert result['budget_exhausted'] and not result['neighborhood_complete']
    assert result['program']==compiled.program


def test_capped_refinement_retains_only_a_full_checked_improvement():
    compiled,evaluate,*_=fixture()
    result=refine(compiled,evaluate,max_full_evaluations=2)
    assert result['full_evaluations']<=2
    assert result['cost']<=result['original_cost']
    assert evaluate(result['program'])['cost']==pytest.approx(result['cost'],abs=1e-12)
    assert not result['neighborhood_complete']


@pytest.mark.parametrize('kwargs',[dict(max_full_evaluations=0),dict(max_full_evaluations=True),
    dict(max_sweeps=0),dict(batch_size=0)])
def test_invalid_refinement_allowances_are_rejected(kwargs):
    compiled,evaluate,*_=fixture()
    with pytest.raises(ValueError,match='allowances'):
        refine(compiled,evaluate,**kwargs)
