from datetime import date, timedelta
from itertools import product

import numpy as np
import pytest

from scripts.program_objective_trial import TrialSession, TrialPolicy, DemandScenario, components
from scripts.volume_demand_frontier import PatternCost, LONG, QUALITY, neighbors, seeds, search, exhaustive_small
from run_analysis.workload_accounting import Quantity

VECTORS=np.array([[.25,.3,0],[2.1,2.3,0],[1.1,1.8,1.]])
START=date(2026,9,26)


@pytest.mark.parametrize('memory,timing',tuple(product((1.,2.,4.),(0.,.5))))
def test_matrix_exactly_matches_original_scorer_with_history_and_continuation(memory,timing):
    x=np.array([LONG,22,0,QUALITY,0,35,0])
    hq=np.array([.9,1.2,.7])
    cost=PatternCost(7,3,VECTORS,[(1,hq)],memory_days=memory,timing_compression_days=timing)
    historical=TrialSession('h',START-timedelta(days=1),4.,'quality',
        tuple(Quantity.point(float(v),'test') for v in hq),source='observed')
    future=[]
    for day in range(21):
        value=x[day%7]
        if value==0:
            continue
        role,miles,q=('long',8.4,VECTORS[1]) if value==LONG else (
            ('quality',4.5,VECTORS[2]) if value==QUALITY else ('easy',value/10,VECTORS[0]*value/10))
        future.append(TrialSession(str(day),START+timedelta(days=day),miles,role,
            tuple(Quantity.point(float(v),'test') for v in q),1. if value==QUALITY else 0.))
    original=components((historical,),future,stage_start=START,decision=START,end=START+timedelta(days=21),
        scales=(1.,1.,1.),policy=TrialPolicy(),scenario=DemandScenario(memory,.5,timing))
    assert cost.batch([x])[0]==pytest.approx(original['demand_cost'],abs=1e-12)


def test_no_history_or_tail_disappears():
    x=[LONG,22,0,QUALITY,0,35,0]
    history=[(1,[1.,1.,1.])]
    assert PatternCost(7,3,VECTORS,history).batch([x])[0]>PatternCost(7,3,VECTORS).batch([x])[0]
    # Per-seven average varies with boundary; the full cumulative cost grows.
    assert PatternCost(7,3,VECTORS,history).batch([x])[0]*3>PatternCost(7,1,VECTORS,history).batch([x])[0]


def test_neighbors_preserve_exact_mileage_key_work_and_committed_doses():
    x=np.array([LONG,30,0,QUALITY,0,35,0])
    n=neighbors(x)
    assert len(n)>0
    assert all(np.sum(row==LONG)==1 and np.sum(row==QUALITY)==1 for row in n)
    assert np.all(np.maximum(n,0).sum(axis=1)==65)
    assert np.all((n<=0)|((n>=14)&(n<=60)))
    assert len({tuple(row) for row in n})==len(n)
    assert {sum(row>0) for row in n}>={2,3}
    assert any(row[0]!=LONG for row in n)


def test_seeds_vary_count_and_initial_long_date():
    variants=seeds(21,195)
    assert len({sum(v>0) for v in variants})==3
    assert len({tuple(np.flatnonzero(v==LONG)) for v in variants})==7
    for x in variants:
        assert np.maximum(x,0).sum()==195
        assert sum(x==LONG)==sum(x==QUALITY)==3


@pytest.mark.parametrize('memory',[1.,2.,4.])
def test_search_against_all_small_calendars(memory):
    cost=PatternCost(7,3,VECTORS,[(1,[.7,1.4,.6]),(4,[1.8,2.,0.])],memory_days=memory)
    oracle=exhaustive_small(cost)
    assert oracle['total_candidates']==42*3**5
    for budget in (40,60,80,100):
        result=search(cost,budget,floor=20,ceiling=30,step=10)
        assert result['budget_exhausted_starts']==0
        assert result['cost']==pytest.approx(oracle['by_easy_tenths'][budget]['cost'],abs=1e-10)


def test_identical_search_is_deterministic():
    cost=PatternCost(7,3,VECTORS)
    a=search(cost,60,floor=20,ceiling=30,step=10)
    b=search(cost,60,floor=20,ceiling=30,step=10)
    for key in ('state','cost','evaluated_neighbors','distinct_endpoints'):
        assert a[key]==b[key]


def test_unrepresentable_dose_and_missing_demand_fail_explicitly():
    with pytest.raises(ValueError):
        seeds(7,1)
    with pytest.raises(ValueError):
        PatternCost(7,3,[[None]*3]*3)


def test_cost_ignores_the_old_ordinary_credit_and_volume_exchange():
    cost=PatternCost(7,3,VECTORS)
    a=[LONG,0,0,QUALITY,0,44,0]
    b=[LONG,22,0,QUALITY,0,22,0]
    assert cost.batch([a,b]).shape==(2,)
    # Only physical normalized demand appears in this module's objective.
    assert not hasattr(cost,'ordinary_power') and not hasattr(cost,'weekly_miles')


def test_fixed_key_arm_cannot_buy_efficiency_by_deferring_key_work():
    keys=np.array([LONG,0,0,QUALITY,0,0,0])
    for seed in seeds(7,65,fixed_keys=keys):
        assert np.array_equal(np.where(seed<0,seed,0),keys)
        for row in neighbors(seed,move_keys=False):
            assert np.array_equal(np.where(row<0,row,0),keys)
    result=search(PatternCost(7,3,VECTORS),65,fixed_keys=keys)
    assert np.array_equal(np.where(np.array(result['state'])<0,result['state'],0),keys)


def test_restricting_to_longer_easy_doses_is_reported_as_a_different_domain():
    result=search(PatternCost(7,3,VECTORS),65,floor=30)
    assert all(v>=30 for v in result['state'] if v>0)
