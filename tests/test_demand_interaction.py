from dataclasses import replace
from math import exp
import numpy as np
import pytest
from run_analysis.adaptive.demand_interaction import interaction
from run_analysis.adaptive.shared_program_evaluator import ProgramCandidate,evaluate_program
from test_shared_program_evaluator import context,session,day


MODE='running_work_shared_v1'


def test_shared_running_work_is_symmetric_and_reaches_runs_with_zero_work():
    long=np.array([2.,2.,0.]);quality=np.array([.7,.8,1.])
    assert interaction(long,quality)[2]==0.
    assert interaction(long,quality,MODE)[2]==2.
    assert np.array_equal(interaction(long,quality,MODE),interaction(quality,long,MODE))
    assert np.array_equal(interaction(long,long),interaction(long,long,MODE))
    with pytest.raises(ValueError):interaction(long,quality,'guess')


def test_role_is_not_a_fatigue_multiplier_and_raw_ledger_is_unchanged():
    q=session('q',0,role='quality');long=session('l',1,8.4,'long')
    p=ProgramCandidate((q,long),(),'rest tail');ctx=context(history=())
    base=evaluate_program(ctx,p);coupled=evaluate_program(ctx,p,demand_interaction=MODE)
    assert coupled.conditional_ledger==base.conditional_ledger
    assert coupled.delivery_cost==base.delivery_cost
    assert coupled.weighted_demand_cost>base.weighted_demand_cost
    assert coupled.fingerprint!=base.fingerprint
    # Same physical event, different role: overlap is not selected by name.
    changed=evaluate_program(ctx,replace(p,hard=(q,replace(long,role='easy'))),demand_interaction=MODE)
    assert changed.weighted_demand_cost==coupled.weighted_demand_cost
    raw=coupled.parts['session_costs'][1]['axis_cost']['work_minutes']
    expected=(8.4*10/44)*(15/15)*exp(-1/ctx.scenario.memory_days)
    assert raw==pytest.approx(expected)


def test_coupled_completion_preserves_relative_scores_and_boundary_ownership():
    ctx=context(history=());first=session('q',0,role='quality')
    plans=[ProgramCandidate((first,session('long',i,8.4,'long')),(),'rest tail') for i in (2,3)]
    before=[evaluate_program(ctx,p,demand_interaction=MODE) for p in plans]
    rolled=replace(ctx,history=(replace(first,source='observed'),),decision=day(1))
    after=[evaluate_program(rolled,replace(p,hard=p.hard[1:]),demand_interaction=MODE) for p in plans]
    assert before[0].total_cost-before[1].total_cost==pytest.approx(after[0].total_cost-after[1].total_cost)
    tail=replace(plans[0],hard=(first,),continuation=(plans[0].hard[1],))
    assert evaluate_program(replace(ctx,hard_end=day(1)),tail,demand_interaction=MODE).total_cost==before[0].total_cost


def test_shared_interaction_prices_every_continuation_handoff_once():
    from test_continuation_boundary_trial import setup
    from test_soft_horizon_trial import candidate
    from run_analysis.adaptive.continuation_boundary_trial import continuation_boundary_cost
    ctx,refs,rows,base=setup()
    coupled=evaluate_program(ctx,candidate(ctx,rows),demand_interaction=MODE)
    old_boundary=continuation_boundary_cost(ctx,base,refs)
    new_boundary=continuation_boundary_cost(ctx,coupled,refs,demand_interaction=MODE)
    # This fixture has equal normalized running/cardio/work demand. The shared
    # term is exactly two extra running/work products, not another full charge.
    for old,new in zip(old_boundary['scenarios'],new_boundary['scenarios']):
        assert new['cross_demand_by_axis']==pytest.approx([
            old['cross_demand_by_axis'][0],old['cross_demand_by_axis'][1],
            3*old['cross_demand_by_axis'][2]],abs=1e-12)
    integrated=replace(ctx,continuation_boundary_references=tuple(
        (role,miles,demand) for role,(miles,demand) in refs.items()))
    actual=evaluate_program(integrated,candidate(integrated,rows),demand_interaction=MODE)
    original=evaluate_program(integrated,candidate(integrated,rows))
    assert actual.total_cost==pytest.approx(new_boundary['cost'],abs=1e-12)
    assert actual.weighted_demand_cost==pytest.approx(original.weighted_demand_cost*5/3,abs=1e-12)
    assert actual.delivery_cost==original.delivery_cost
    assert actual.conditional_ledger==original.conditional_ledger
    assert actual.total_cost==pytest.approx(actual.weighted_demand_cost+actual.delivery_cost+actual.short_run_cost)
    with pytest.raises(ValueError,match='twice'):
        continuation_boundary_cost(integrated,actual,refs,demand_interaction=MODE)
    from test_steady_continuation_trial import fixture
    steady_ctx,steady_refs,steady_rows=fixture()
    with pytest.raises(ValueError,match='general-fitness'):
        evaluate_program(replace(steady_ctx,steady_continuation_references=tuple(
            (role,miles,demand) for role,(miles,demand) in steady_refs.items())),
            candidate(steady_ctx,steady_rows),demand_interaction=MODE)
