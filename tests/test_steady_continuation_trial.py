from datetime import date,timedelta
from types import SimpleNamespace
from dataclasses import replace
from math import exp
import pytest
import numpy as np

from run_analysis.workload_accounting import Quantity
from run_analysis.adaptive.program_objective_trial import TrialSession,TrialPolicy,DemandScenario,EasyDurationPolicy
from run_analysis.adaptive.shared_program_evaluator import ProgramContext
from run_analysis.adaptive.shared_program_evaluator import ProgramCandidate,evaluate_program
from run_analysis.adaptive.timely_purpose_trial import PurposeTimingPolicy,envelope_day
from run_analysis.adaptive.steady_continuation_trial import terminal_value


def fixture():
    start=date(2026,9,27)
    q=lambda n:(Quantity.point(n,'fixture'),)*3
    context=ProgramContext((),start,start,start+timedelta(days=21),start+timedelta(days=35),
        (1.,1.,1.),TrialPolicy(weekly_miles=18.,long_miles=8.,volume_window_days=28),
        DemandScenario(),.01,PurposeTimingPolicy(1.,grace_fraction=1.),
        easy_duration_policy=EasyDurationPolicy(10,25,.01,'fixture',6.))
    references=dict(easy=(3.,q(.75)),long=(8.,q(2)),quality=(4.,q(1)))
    rows=tuple(TrialSession(str(d),start+timedelta(days=d),m,role,q(m/4),quality_credit=1 if role=='quality' else 0)
        for d,m,role in ((0,8,'long'),(3,4,'quality'),(5,3,'easy'),(8,3,'easy'),
                        (14,8,'long'),(18,4,'quality'),(21,3,'easy'),(28,8,'long'),(31,4,'quality'),(33,3,'easy')))
    return context,references,rows


def test_terminal_fatigue_matches_explicit_future_pair_interactions():
    context,references,rows=fixture()
    result=terminal_value(context,SimpleNamespace(conditional_ledger=rows),references)
    daily=(2+1+.75*6/3)/7
    explicit=0.
    for offset in range(200):
        day=context.end+timedelta(days=offset)
        explicit+=sum((r.miles/4)*daily*exp(-(day-r.day).days/2) for r in rows)
    assert result['demand']==pytest.approx(.01*7/35*explicit,abs=1e-12)


def test_terminal_volume_is_remaining_fluid_loss_not_new_credit_in_hard_plan():
    context,references,rows=fixture()
    r=terminal_value(context,SimpleNamespace(conditional_ledger=rows),references)
    for purpose,rate in (('volume',18),('ordinary_easy',6)):
        expected=0.
        for offset in range(27):
            day=context.end+timedelta(days=offset)
            existing=sum(x.miles for x in rows if day-timedelta(days=27)<=x.day< context.end
                         and (purpose=='volume' or x.role=='easy'))
            total=existing+rate*(offset+1)/7
            expected+=max(0,1-total/(rate*4))**2/35
        assert r['volume'][purpose]==pytest.approx(expected,abs=1e-12)
    assert r['expected_miles_per_7']==18
    assert not r['activated_by_default']


def test_purpose_loss_matches_all_uniform_future_phases():
    context,references,rows=fixture()
    r=terminal_value(context,SimpleNamespace(conditional_ledger=rows),references)
    for purpose in ('long','quality'):
        events=[(x.day,1.) for x in rows if x.role==purpose]
        expected=sum(sum(envelope_day(events,context.end+timedelta(days=d),grace=7,memory=7)['loss']
                         for d in range(delay)) for delay in range(7))/7/35
        assert r['timing'][purpose]==pytest.approx(expected,abs=1e-12)


def test_terminal_is_not_a_run_date_lock_and_empty_prefix_has_large_deficit():
    context,references,rows=fixture()
    priced=terminal_value(context,SimpleNamespace(conditional_ledger=rows),references)
    empty=terminal_value(context,SimpleNamespace(conditional_ledger=()),references)
    assert empty['cost']>priced['cost']
    assert empty['demand']==0
    assert 'not an issued calendar' in priced['scope']


def test_integrated_objective_charges_each_term_once_without_fabricating_delivery():
    context,references,rows=fixture()
    candidate=ProgramCandidate(tuple(r for r in rows if r.day<context.hard_end),
        tuple(r for r in rows if r.day>=context.hard_end),'fixture')
    base=evaluate_program(context,candidate)
    expected=terminal_value(context,base,references)
    full=evaluate_program(replace(context,steady_continuation_references=tuple(
        (role,miles,demand) for role,(miles,demand) in references.items())),candidate)
    assert full.total_cost==pytest.approx(base.total_cost+expected['cost'])
    assert full.parts['delivery']==base.parts['delivery']
    assert full.parts['targets']==base.parts['targets']
    assert full.conditional_ledger==base.conditional_ledger
    assert full.downstream_accounting['status']=='coarse_fixed_stage_terminal'
    assert full.total_cost==pytest.approx(full.delivery_cost+full.weighted_demand_cost+full.short_run_cost)
    with pytest.raises(ValueError,match='combine'):
        replace(context,steady_continuation_references=tuple((r,m,q) for r,(m,q) in references.items()),
            downstream_end=context.end+timedelta(days=7),downstream_basis='would double count')
