from types import SimpleNamespace

import pytest

from scripts import finalized_distance_search as search
from scripts.continuation_replan import fingerprint
from test_typed_continuation import fixture
from test_recommendation import CONFIG


def test_neighbors_keep_other_days_and_exact_times(monkeypatch):
    state_at,make=fixture()
    days,states=make(10,{0:('easy',3.3),2:('long',8.),4:('easy',3.3),8:('easy',3.3)})
    monkeypatch.setattr(search.planner,'_easy_allocation_bounds',lambda *a,**k:(3.25,4.,3.75,1.))
    monkeypatch.setattr(search.planner,'_easy_allocation_options',lambda *a,**k:([3.25,3.5,3.75,4.],{3.5:(3.2,3.8)}))
    original=fingerprint(days)
    variants=list(search.neighbors(days,dict(config=CONFIG,daily_states=states),state_at,visible_days=7))
    assert any(v[0].recommendation.planned_distance_miles==3.5 for v in variants)
    for candidate in variants:
        changed=[i for i,(a,b) in enumerate(zip(days,candidate)) if a!=b]
        assert len(changed)==1 and changed[0] in (0,4)
        i=changed[0]
        assert candidate[i].recommendation.planned_for==days[i].recommendation.planned_for
        assert candidate[i].recommendation.workout_type==days[i].recommendation.workout_type
        target=candidate[i].recommendation.planned_distance_miles
        assert abs(target*10-round(target*10))<1e-8
    assert fingerprint(days)==original


def test_easy_neighbors_do_not_erase_primary_long_distinction(monkeypatch):
    state_at,make=fixture()
    days,states=make(3,{0:('easy',3.3),2:('long',4.)})
    monkeypatch.setattr(search.planner,'_easy_allocation_bounds',lambda *a,**k:(3.25,4.,3.75,1.))
    monkeypatch.setattr(search.planner,'_easy_allocation_options',lambda *a,**k:([3.25,3.5,3.75,4.],{}))
    variants=list(search.neighbors(days,dict(config=CONFIG,daily_states=states),state_at,visible_days=7))
    assert all(v[0].recommendation.planned_distance_miles<=4./1.15 for v in variants)


def setup_descent(monkeypatch, costs):
    _,make=fixture()
    def candidate(x,y):
        days,_=make(3,{0:('easy',x),2:('easy',y)})
        return dict(days=days,program=(days,[],[]),score=SimpleNamespace(total_cost=costs[(x,y)],offsets=[0,2]),
            row=dict(hard_fingerprint=fingerprint(days)),tail_fingerprint='tail',stable=True)
    def doses(days):
        return tuple(days[i].recommendation.planned_distance_miles for i in (0,2))
    def neighbors(days,*args,**kwargs):
        x,y=doses(days)
        for pair in ((7.-x,y),(x,7.-y)):
            yield candidate(*pair)['days']
    calls=[]
    def evaluate(seed):
        calls.append(seed)
        return candidate(*doses(seed['days']))
    monkeypatch.setattr(search,'neighbors',neighbors)
    return candidate,evaluate,calls


def test_revisits_earlier_coordinate_after_later_improvement(monkeypatch):
    candidate,evaluate,calls=setup_descent(monkeypatch,{(3.,3.):10.,(4.,3.):11.,(3.,4.):8.,(4.,4.):6.})
    result=search.refine_distances([candidate(3.,3.)],evaluate,{},None)
    assert result['lanes'][0]['path']==[10.,8.,6.]
    assert result['lanes'][0]['status']=='locally_converged'
    assert all(c['source']=='distance_neighbor' and c['tail_parent']['tail_fingerprint']=='tail' for c in calls)


def test_budget_exhaustion_is_not_reported_as_convergence(monkeypatch):
    candidate,evaluate,_=setup_descent(monkeypatch,{(3.,3.):10.,(4.,3.):11.,(3.,4.):8.,(4.,4.):6.})
    result=search.refine_distances([candidate(3.,3.)],evaluate,{},None,budget=1)
    assert result['evaluations']==1
    assert result['lanes'][0]['status']=='budget_exhausted'
    assert result['lanes'][0]['path']==[10.]


def test_rejected_or_tied_neighbors_do_not_displace_incumbent(monkeypatch):
    candidate,_,_=setup_descent(monkeypatch,{(3.,3.):10.,(4.,3.):10.,(3.,4.):10.,(4.,4.):10.})
    current=candidate(3.,3.)
    result=search.refine_distances([current],lambda seed:None,{},None)
    assert result['lanes'][0]['path']==[10.]
    result=search.refine_distances([current],lambda seed:current,{},None)
    assert result['lanes'][0]['path']==[10.]


def test_policy_is_scoped_and_disabled_by_default():
    assert search.DISTANCE_REFINEMENT.get() is None
    with search.distance_refinement_policy(enabled=True,budget=5,post_calendar_budget=64):
        assert search.DISTANCE_REFINEMENT.get()['budget']==5
        assert search.DISTANCE_REFINEMENT.get()['candidate_budget']==15
        assert search.DISTANCE_REFINEMENT.get()['visible_days']==21
        assert search.DISTANCE_REFINEMENT.get()['post_calendar_budget']==64
    assert search.DISTANCE_REFINEMENT.get() is None
    with pytest.raises(ValueError):
        with search.distance_refinement_policy(budget=0):
            pass
    with pytest.raises(ValueError):
        with search.distance_refinement_policy(post_calendar_budget=-1):
            pass


def test_independent_and_stable_lanes_both_receive_checks(monkeypatch):
    candidate,evaluate,calls=setup_descent(monkeypatch,{(3.,3.):10.,(4.,3.):11.,(3.,4.):12.,(4.,4.):13.})
    independent=candidate(3.,3.); independent['stable']=False
    stable=candidate(4.,3.)
    stable['tail_fingerprint']='different-tail'
    def reject(seed):
        calls.append(seed)
        return None
    result=search.refine_distances([independent,stable],reject,{},None)
    assert len(result['lanes'])==2
    assert {c['tail_parent']['tail_fingerprint'] for c in calls}=={'tail','different-tail'}
    assert all(l['status']=='locally_converged' for l in result['lanes'])


def test_prepruning_mode_checks_every_retained_physical_candidate(monkeypatch):
    candidate,evaluate,calls=setup_descent(monkeypatch,{(3.,3.):10.,(4.,3.):11.,(3.,4.):12.,(4.,4.):13.})
    first=candidate(3.,3.)
    second=candidate(4.,4.)
    second['tail_fingerprint']='other-tail'

    result=search.refine_distances(
        [first,second],evaluate,{},None,
        seed_candidates=[first,second],
    )

    assert result['seed_mode']=='all_retained'
    assert result['seed_count']==2
    assert len(result['lanes'])==2
    assert {c['tail_parent']['tail_fingerprint'] for c in calls}=={
        'tail','other-tail',
    }


def test_prepruning_sweep_limit_is_not_reported_as_convergence(monkeypatch):
    candidate,evaluate,_=setup_descent(monkeypatch,{(3.,3.):10.,(4.,3.):11.,(3.,4.):8.,(4.,4.):6.})

    result=search.refine_distances(
        [candidate(3.,3.)],evaluate,{},None,
        seed_candidates=[candidate(3.,3.)],max_sweeps=1,
    )

    assert result['lanes'][0]['path']==[10.,8.]
    assert result['lanes'][0]['status']=='sweep_limit_reached'


def test_no_easy_days_has_empty_complete_neighborhood():
    _,make=fixture()
    days,_=make(3,{})
    current=dict(days=days,program=(days,[],[]),score=SimpleNamespace(total_cost=1.,offsets=[]),
        row=dict(hard_fingerprint=fingerprint(days)),tail_fingerprint='tail',stable=False)
    # No projected state is needed for an all-rest visible prefix.
    result=search.refine_distances([current],lambda seed:pytest.fail('No neighbor expected'),{},None)
    assert result['evaluations']==0
    assert result['lanes'][0]['status']=='locally_converged'


def test_large_distance_lane_does_not_starve_alternatives():
    def item(name,cost):
        return dict(days=(name,),program=((name,),),score=SimpleNamespace(total_cost=cost,offsets=()),
            row=dict(hard_fingerprint=name),tail_fingerprint=name,stable=False)
    items=[item(str(i),float(i)) for i in range(3)]
    calls=[]
    def neighbors(current):
        for i in range(10):
            yield (current['days'][0],i)
    result=search.refine_distances(items,lambda s:calls.append(s['tail_parent']['days'][0]),{},None,
        budget=8,seed_candidates=items,neighbor_adapter=neighbors,identity_adapter=tuple)
    assert calls==['0','1','2','0','1','2','0','1']
    assert [l['sweeps'][0]['evaluations'] for l in result['lanes']]==[3,3,2]
    assert all(l['status']=='budget_exhausted' for l in result['lanes'])
