from dataclasses import replace
from datetime import timedelta
import pytest

from test_steady_continuation_trial import fixture
from run_analysis.adaptive.shared_program_evaluator import ProgramCandidate,evaluate_program
from run_analysis.adaptive.soft_horizon_trial import day_weight,tapered_cost
from run_analysis.adaptive.program_objective_trial import DemandScenario


def candidate(context,rows):
    return ProgramCandidate(tuple(r for r in rows if r.day<context.hard_end),
        tuple(r for r in rows if r.day>=context.hard_end),'explicit test')


@pytest.mark.parametrize('fraction',[0.,.5,1.])
@pytest.mark.parametrize('axis_weights',[(1.,1.,1.),(2.,3.,1.)])
def test_uniform_daily_decomposition_recovers_existing_score(fraction,axis_weights):
    context,_,rows=fixture()
    context=replace(context,scenario=DemandScenario(bound_fraction=fraction,axis_weights=axis_weights))
    result=evaluate_program(context,candidate(context,rows))
    uniform=tapered_cost(context,result,uniform=True)
    assert uniform['cost']==pytest.approx(result.total_cost,abs=1e-12)
    assert uniform['delivery']==pytest.approx(result.delivery_cost,abs=1e-12)
    assert uniform['demand']==pytest.approx(result.weighted_demand_cost,abs=1e-12)
    assert uniform['short']==pytest.approx(result.short_run_cost,abs=1e-12)


def test_weight_is_full_through_hard_plan_and_continuous_at_boundary():
    context,_,_=fixture()
    assert day_weight(context,context.decision)==1
    assert day_weight(context,context.hard_end-timedelta(days=1))==1
    assert day_weight(context,context.hard_end)==1
    assert day_weight(context,context.end-timedelta(days=1))==pytest.approx(1/14)
    assert day_weight(context,context.end)==0


def test_integrated_taper_keeps_physical_credit_and_cost_components_consistent():
    context,_,rows=fixture()
    plan=candidate(context,rows)
    base=evaluate_program(context,plan)
    expected=tapered_cost(context,base)
    result=evaluate_program(replace(context,soft_horizon_taper=True),plan)
    assert result.total_cost==pytest.approx(expected['cost'])
    assert result.total_cost==pytest.approx(result.delivery_cost+result.weighted_demand_cost+result.short_run_cost)
    assert result.conditional_ledger==base.conditional_ledger
    assert result.parts['delivery']==base.parts['delivery']
    assert result.parts['targets']==base.parts['targets']
    assert result.parts['demand_by_axis']==expected['demand_by_axis']
    assert result.fingerprint!=base.fingerprint


def test_future_overlap_still_charged_in_projection_instead_of_ending_at_day21():
    context,_,rows=fixture()
    result=evaluate_program(replace(context,soft_horizon_taper=True),candidate(context,rows))
    late=[r for r in result.parts['session_costs'] if r['day']>=context.hard_end]
    assert late and all(sum(r['axis_cost'].values())>0 for r in late)
    assert all(day_weight(context,r['day'])>0 for r in late)


def test_cannot_silently_combine_distinct_horizon_experiments():
    context,references,_=fixture()
    with pytest.raises(ValueError,match='separate'):
        replace(context,soft_horizon_taper=True,steady_continuation_references=tuple(
            (role,miles,demand) for role,(miles,demand) in references.items()))
