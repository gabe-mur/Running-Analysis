from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from test_compiled_calendar_doses import fixture
from test_soft_horizon_trial import candidate
from run_analysis.adaptive.compiled_calendar_doses import compile_calendar
from run_analysis.adaptive.continuation_survival_trial import distribution, evaluate_trial
from run_analysis.adaptive.continuation_boundary_trial import continuation_boundary_cost
from run_analysis.adaptive.shared_program_evaluator import evaluate_program
from run_analysis.adaptive.soft_horizon_trial import day_weight


def test_survival_matches_all_existing_handoffs_and_does_not_drop_endpoint_mass():
    _,_,state,_,_,_ = fixture()
    context = state.context
    spec = distribution(context,basis='memory_survival')
    assert spec['memory_days']==33
    assert sum(spec['probabilities'])==pytest.approx(1.,abs=1e-14)
    assert spec['endpoint_probability']>.6
    for day in (context.decision+timedelta(days=i) for i in range(36)):
        expected=sum(p for h,p in zip(spec['handoffs'],spec['probabilities']) if day<h)
        assert spec['weight'](day)==pytest.approx(expected,abs=1e-14)


@pytest.mark.parametrize('fraction',[0.,.5,1.])
@pytest.mark.parametrize('single_volume',[True,False])
def test_uniform_probabilities_reproduce_legacy_full_objective(fraction,single_volume):
    _,_,state,_,program,details=fixture(fraction=fraction,single_volume=single_volume)
    baseline=details(program)['evaluation'];context=state.context
    count=(context.end-context.taper_start).days
    trial=evaluate_trial(context,baseline,probabilities=(1/count,)*count,
        weight=lambda d:day_weight(context,d))
    assert trial.total_cost==pytest.approx(baseline.total_cost,abs=1e-12)
    assert trial.conditional_ledger==baseline.conditional_ledger


def test_survival_preserves_single_volume_budget_and_cannot_credit_deleted_runs():
    _,_,state,_,program,details=fixture()
    context=state.context
    base=details(program)['evaluation']
    full=evaluate_trial(context,base)
    empty=evaluate_trial(context,evaluate_program(context,candidate(context,())))
    assert full.parts['continuation_boundary']['expected_miles_per_7']==context.policy.weekly_miles
    assert empty.parts['continuation_boundary']['volume_cost']>full.parts['continuation_boundary']['volume_cost']
    assert full.conditional_ledger==base.conditional_ledger
    assert full.total_cost==pytest.approx(full.delivery_cost+full.weighted_demand_cost+full.short_run_cost)


@pytest.mark.parametrize('fraction',[0.,.5,1.])
@pytest.mark.parametrize('basis',['memory_survival','c1_soft_handoff'])
def test_compiler_matches_trial_for_nonlinear_catalogue_not_just_initial_winner(fraction,basis):
    _,_,state,physical,program,details=fixture(fraction=fraction,weights=(2.,3.,0.),compression=.4)
    spec=distribution(state.context,basis=basis)
    def priced(p):
        raw=details(p)
        return dict(raw,evaluation=evaluate_trial(state.context,raw['evaluation'],
            probabilities=spec['probabilities'],weight=spec['weight']))
    compiled=compile_calendar(state,program,physical=physical,result=priced(program),
        day_weight_function=spec['weight'],handoff_probabilities=spec['probabilities'])
    choices=[compiled.original_levels,np.zeros(len(compiled.days),dtype=int),
        np.full(len(compiled.days),len(compiled.keys)-1,dtype=int)]
    choices.extend(np.array([(n*11+i*7)%len(compiled.keys) for i in range(len(compiled.days))]) for n in range(8))
    for levels,value in zip(choices,compiled.cost(choices)):
        assert value==pytest.approx(priced(compiled.physical(levels))['evaluation'].total_cost,abs=1e-12)


def test_c1_curve_has_zero_endpoint_slopes_and_a_positive_mass_for_every_handoff():
    from run_analysis.adaptive.continuation_distribution import c1_survival,c1_handoff_probabilities
    step=1e-6
    assert abs((c1_survival(step)-1)/step)<1e-5
    assert abs(c1_survival(1-step)/step)<1e-5
    probabilities=c1_handoff_probabilities(14)
    assert all(p>0 for p in probabilities)
    assert sum(probabilities)==pytest.approx(1.,abs=1e-14)
    _,_,state,_,_,_=fixture()
    context=replace(state.context,continuation_distribution='c1_v1')
    for i in range(36):
        day=context.decision+timedelta(days=i)
        handoffs=distribution(context,basis='c1_soft_handoff')['handoffs']
        assert day_weight(context,day)==pytest.approx(sum(p for h,p in zip(handoffs,probabilities) if day<h))


def test_integrated_c1_contract_matches_diagnostic_and_changes_identity_not_physical_work():
    _,_,state,_,program,details=fixture()
    base=details(program)['evaluation']
    spec=distribution(state.context,basis='c1_soft_handoff')
    trial=evaluate_trial(state.context,base,probabilities=spec['probabilities'],weight=spec['weight'])
    context=replace(state.context,continuation_distribution='c1_v1')
    actual=evaluate_program(context,candidate(context,tuple(r for r in base.conditional_ledger if r.source=='planned')))
    assert actual.total_cost==pytest.approx(trial.total_cost,abs=1e-12)
    assert actual.fingerprint!=base.fingerprint
    assert actual.model_version.endswith(':c1_handoff_v1')
    assert actual.conditional_ledger==base.conditional_ledger
    assert actual.parts['delivery']==base.parts['delivery']


def test_context_persistence_preserves_old_bytes_and_round_trips_new_contract():
    from run_analysis.adaptive.decision_snapshot import _encode,_decode
    _,_,state,_,_,_=fixture()
    before=_encode(state.context)
    assert 'continuation_distribution' not in before['fields']
    assert _decode(before)==state.context
    updated=replace(state.context,continuation_distribution='c1_v1')
    after=_encode(updated)
    assert after['fields']['continuation_distribution']=='c1_v1'
    assert _decode(after)==updated


def test_unknown_or_unbacked_survival_contract_is_rejected():
    _,_,state,_,_,_=fixture()
    with pytest.raises(ValueError,match='distribution'):
        replace(state.context,continuation_distribution='unrecognized')
    with pytest.raises(ValueError,match='distribution'):
        replace(state.context,continuation_boundary_references=(),continuation_distribution='c1_v1')


def test_expected_completion_preserves_pairwise_ranking_with_fixed_c1_handoffs():
    _,_,state,_,program,details=fixture()
    context=replace(state.context,continuation_distribution='c1_v1',purpose_continuation=True,
        purpose_continuation_phase='known_full_phase_v1')
    rows=details(program)['evaluation'].conditional_ledger
    completed=replace(rows[0],source='observed')
    after=replace(context,decision=context.decision+timedelta(days=1),history=(completed,))
    comparisons=[]
    for shift in (0,1):
        varied=tuple(replace(r,day=r.day+timedelta(days=shift)) if r.key=='5' else r for r in rows)
        before_score=evaluate_program(context,candidate(context,varied)).total_cost
        after_score=evaluate_program(after,candidate(after,varied[1:])).total_cost
        comparisons.append((before_score,after_score))
    assert comparisons[0][0]-comparisons[1][0]==pytest.approx(comparisons[0][1]-comparisons[1][1],abs=1e-12)


def test_compiler_and_event_interactions_share_the_c1_survival_contract():
    from datetime import datetime,time,timezone
    from run_analysis.adaptive.race_goal import RaceGoalRevision
    from run_analysis.workload_accounting import Quantity
    _,_,state,physical,program,details=fixture()
    at=datetime.combine(state.context.decision,time.min,timezone.utc)
    goal=RaceGoalRevision.create(available_at=at,distance='10k',
        race_date=state.context.decision+timedelta(days=30),preparation_mode='balanced',parent_key='c1_fixture')
    context=replace(state.context,continuation_distribution='c1_v1',race_goal=goal,
        race_demand=(Quantity.point(3.,'fixture'),)*3)
    state.context=context
    def scored(p):
        rows=details(p)['evaluation'].conditional_ledger
        ev=evaluate_program(context,candidate(context,rows))
        assert ev.parts['race_event']['training_credit']==0.
        return dict(evaluation=ev,decision_eligible=True)
    compiled=compile_calendar(state,program,physical=physical,result=scored(program))
    choices=[compiled.original_levels,np.zeros(len(compiled.days),dtype=int),
        np.full(len(compiled.days),len(compiled.keys)-1,dtype=int)]
    for levels,value in zip(choices,compiled.cost(choices)):
        assert value==pytest.approx(scored(compiled.physical(levels))['evaluation'].total_cost,abs=1e-12)


@pytest.mark.parametrize('probabilities',[(0.,)*14,(.1,)*14,(float('nan'),)*14,(1.,)])
def test_inconsistent_handoff_distribution_fails_closed(probabilities):
    _,_,state,_,program,details=fixture()
    ev=details(program)['evaluation']
    parts={k:v for k,v in ev.parts.items() if k!='continuation_boundary'}
    with pytest.raises(ValueError,match='probabilities'):
        continuation_boundary_cost(state.context,SimpleNamespace(total_cost=ev.total_cost,
            parts=parts,conditional_ledger=ev.conditional_ledger),
            {r:(m,q) for r,m,q in state.context.continuation_boundary_references},
            handoff_probabilities=probabilities)
