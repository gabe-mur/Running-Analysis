from dataclasses import replace
from datetime import date, timedelta
from math import exp

import pytest

from scripts.audit_program_objective import calendar, policies, scenarios
from scripts.program_objective_trial import (TrialSession, TrialPolicy, DemandScenario,
    components, credits, describe, exchange_region, score)
from run_analysis.workload_accounting import Quantity

START = date(2026,9,26)
RATES = {role:{channel:dict(bounds=Quantity.point(rate,'synthetic')) for channel,rate in
    [('running_minutes',10.),('cardio',20.)]} for role in ('easy','long','quality')}


def calc(program,history=(),**kwargs):
    options = dict(stage_start=START,decision=START,end=START+timedelta(days=56),
                   scales=(44.,88.,15.),policy=TrialPolicy(),scenario=DemandScenario())
    options.update(kwargs)
    return components(history,program,**options)


@pytest.mark.parametrize('name,total,count',[('A',138.4,24),('B',138.4,32),('C',155.2,32),('D',151.9,28)])
def test_exact_calendar_and_complete_cycles(name,total,count):
    p = calendar(name,START,56,RATES)
    assert sum(e.miles for e in p)==pytest.approx(total)
    assert len(p)==count
    assert all(abs(e.miles*10-round(e.miles*10))<1e-10 for e in p)
    assert all(e.demand[2].low==15 for e in p if e.role=='quality')


def test_funded_plan_beats_rest_at_zero_exchange():
    assert score(calc(calendar('C',START,56,RATES)),0)<score(calc(()),0)
    assert calc(())['delivery_loss']==4
    assert calc(())['demand_cost']==0


def test_ordinary_credit_is_partial_not_a_threshold_or_run_count_reward():
    e = calendar('B',START,7,RATES)[1]
    p = TrialPolicy()
    assert credits(e,p)[1]==.25
    assert credits(replace(e,miles=4.4),p)[1]==1
    assert credits(replace(e,miles=8.8),p)[1]==1
    assert credits(e,p)[1]*2 < credits(replace(e,miles=4.4),p)[1]
    assert credits(e,replace(p,ordinary_power=1))[1]*2==1
    assert p.ordinary_miles==4.4


def test_quality_label_without_dose_has_no_credit():
    q = next(e for e in calendar('C',START,7,RATES) if e.role=='quality')
    assert credits(replace(q,quality_credit=0),TrialPolicy())[3]==0
    assert credits(q,TrialPolicy())[1]==0


def test_more_dose_above_capped_credit_still_adds_demand():
    e = next(e for e in calendar('A',START,7,RATES) if e.role=='easy')
    larger = replace(e,miles=8.8,demand=tuple(q.scale(2) for q in e.demand))
    assert credits(larger,TrialPolicy())[1]==credits(e,TrialPolicy())[1]
    assert calc((larger,))['demand_cost']>calc((e,))['demand_cost']


@pytest.mark.parametrize('memory',[1.,2.,4.])
def test_same_delivery_compression_costs_more(memory):
    a = calc(calendar('C',START,56,RATES),scenario=DemandScenario(memory))
    b = calc(calendar('C_compressed',START,56,RATES),scenario=DemandScenario(memory))
    assert a['delivery']==b['delivery']
    assert a['demand_cost']<b['demand_cost']


def test_overlap_matches_direct_formula_and_preserves_lookback():
    a = TrialSession('h',START-timedelta(days=1),4.4,'easy',tuple(Quantity.point(x,'test') for x in (44,88,0)),source='observed')
    b = replace(a,key='p',day=START,source='planned')
    got = calc((b,),(a,))
    expected = (.5+exp(-.5))*7/56
    assert got['demand_by_axis']['cardio']==pytest.approx(expected)
    assert got['demand_cost']>calc((b,))['demand_cost']


def test_upload_replacement_has_identical_credit_and_only_sunk_cost_difference():
    plans = [calendar(k,START,56,RATES) for k in ('C','B')]
    # Use a genuinely shared executed prefix rather than pretend different doses complied.
    plans[1] = (plans[0][0],*plans[1][1:])
    before = [calc(p) for p in plans]
    history = (replace(plans[0][0],source='observed'),)
    after = [calc(p[1:],history,decision=START+timedelta(days=1)) for p in plans]
    for a,b in zip(before,after):
        assert a['delivery_loss']==b['delivery_loss']
    assert before[0]['demand_cost']-before[1]['demand_cost']==pytest.approx(after[0]['demand_cost']-after[1]['demand_cost'])


def test_same_stage_rollover_on_rest_date_preserves_entire_score():
    p = tuple(replace(e,day=e.day+timedelta(days=1)) for e in calendar('C',START,55,RATES))
    assert calc(p)['demand_cost']==calc(p,decision=START+timedelta(days=1))['demand_cost']
    assert calc(p)['delivery_loss']==calc(p,decision=START+timedelta(days=1))['delivery_loss']


def test_tail_is_charged_and_earlier_session_cost_is_not_rewritten():
    full = calendar('C',START,56,RATES)
    prefix = tuple(e for e in full if e.day < START+timedelta(days=21))
    a,b = calc(prefix),calc(full)
    assert a['session_costs']==b['session_costs'][:len(prefix)]
    assert b['demand_cost']>a['demand_cost']
    assert b['delivery_loss']<a['delivery_loss']


def test_common_tail_remembers_prefix_not_just_last_planned_run():
    b = calendar('B',START,56,RATES)
    c = calendar('C',START,56,RATES)
    seam = START+timedelta(days=21)
    with_c = tuple(e for e in c if e.day < seam)+tuple(e for e in b if e.day >= seam)
    costs_b,costs_c = calc(b),calc(with_c)
    tail_b = sum(sum(e['axis_cost'].values()) for e in costs_b['session_costs'] if e['day']>=seam)
    tail_c = sum(sum(e['axis_cost'].values()) for e in costs_c['session_costs'] if e['day']>=seam)
    assert tail_c>tail_b


def test_unit_conversion_does_not_change_score():
    p=calendar('C',START,56,RATES)
    scaled=tuple(replace(e,demand=(e.demand[0],e.demand[1].scale(100),e.demand[2])) for e in p)
    assert calc(p)['demand_cost']==pytest.approx(calc(scaled,scales=(44,8800,15))['demand_cost'])


def test_older_same_demand_is_not_more_expensive():
    p=calendar('C',START,7,RATES)
    h=replace(p[0],key='actual',day=START-timedelta(days=1),source='observed')
    assert calc(p,(replace(h,day=h.day-timedelta(days=2)),))['demand_cost']<calc(p,(h,))['demand_cost']


def test_unknown_is_not_zero_and_scenarios_preserve_it():
    p=calendar('C',START,7,RATES)
    h=replace(p[0],key='h',day=START-timedelta(days=1),source='observed',
              demand=(*p[0].demand[:2],Quantity(0,90,basis='unresolved')))
    assert calc(p,(h,),scenario=DemandScenario(bound_fraction=1))['demand_cost']>calc(p,(h,),scenario=DemandScenario(bound_fraction=0))['demand_cost']


def test_exchange_region_finds_bounds_and_detects_contradictions():
    a=dict(delivery_loss=0,demand_cost=2)
    b=dict(delivery_loss=1,demand_cost=0)
    region=exchange_region([('training',a,b)])
    assert region['upper_exclusive']==.5
    assert region['nominal_exchange']==.25
    assert not exchange_region([('training',a,b),('opposite',b,a)])['feasible']
    assert not exchange_region([('indistinguishable',a,a)])['feasible']


@pytest.mark.parametrize('bad',[-1,float('nan'),float('inf')])
def test_invalid_exchange_rejected(bad):
    with pytest.raises(ValueError):
        score(dict(delivery_loss=1,demand_cost=1),bad)


def test_duplicate_and_future_history_rejected():
    p=calendar('C',START,7,RATES)
    with pytest.raises(ValueError,match='Unique'):
        calc((*p,p[0]))
    with pytest.raises(ValueError,match='earlier observed'):
        calc(p[1:],(replace(p[0],source='observed'),))


def test_sensitivity_is_predeclared_and_separate_from_candidates():
    assert len(policies())==30
    assert len(scenarios())==72
    assert all(p.ordinary_miles==4.4 for p in policies())


def test_description_reports_low_volume_window_in_eight_day_cycle():
    p=calendar('D',START,64,RATES)
    report=describe(p[4:],tuple(replace(e,source='observed') for e in p[:4]),START+timedelta(days=8),56)
    assert report['miles_per_7']==pytest.approx(18.9875)
    assert report['rolling_7_miles_range']==pytest.approx([13.3,21.7])
    assert set(report['long_gaps'])=={8}
