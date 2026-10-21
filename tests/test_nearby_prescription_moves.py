from collections import Counter
from datetime import date,timedelta

import pytest

from run_analysis.adaptive.calendar_operators import nearby_prescription_neighbors
from run_analysis.adaptive.continuous_search_adapter import search
from run_analysis.adaptive import full_score_refinement,finalized_distance_search


def test_exchanges_complete_doses_and_crosses_occupied_day():
    p={0:'quality:full',1:'easy:2.2',4:'long:8.4'}
    rows=nearby_prescription_neighbors(p,range(6))
    assert ((0,'easy:2.2'),(1,'quality:full'),(4,'long:8.4')) in rows
    assert ((1,'easy:2.2'),(2,'quality:full'),(4,'long:8.4')) in rows
    assert all(Counter(dict(r).values())==Counter(p.values()) for r in rows)
    assert all(len(r)==len(dict(r))==len(p) for r in rows)
    assert rows==sorted(set(rows)) and len(rows)<=4*len(p)


def test_allowed_dates_determinism_and_no_identical_dose_noop():
    p={0:'easy',1:'easy',4:'long'}
    allowed={0,1,3,4,5}
    a=nearby_prescription_neighbors(p,allowed)
    assert a==nearby_prescription_neighbors(dict(reversed(list(p.items()))),reversed(sorted(allowed)))
    assert tuple(p.items()) not in a
    assert all(set(dict(r))<=allowed for r in a)
    assert nearby_prescription_neighbors({},allowed)==[]
    assert nearby_prescription_neighbors({2:'easy'},allowed)==[]


@pytest.mark.parametrize('kind',['swap','cross'])
@pytest.mark.parametrize('eligible',[True,False])
def test_actual_search_discovers_joint_improvement_and_revalidates_it(monkeypatch,kind,eligible):
    dates=tuple((date(2026,9,27)+timedelta(days=i)).isoformat() for i in range(5))
    prior=((dates[0],'easy'),(dates[1],'quality'),(dates[4],'long'))
    target=tuple(sorted(((dates[1],'easy'),(dates[0],'quality'),(dates[4],'long')))) if kind=='swap' else (
        (dates[1],'quality'),(dates[2],'easy'),(dates[4],'long'))
    # Coarse seeds cannot contain quality/long, and all intermediates are worse.
    # Isolate the real selector's final search, not a hindsight candidate seed.
    monkeypatch.setattr(full_score_refinement,'refine',lambda *a,**k:dict(lanes=[]))
    monkeypatch.setattr(finalized_distance_search,'refine_distances',lambda *a,**k:dict(lanes=[]))
    calls=[]
    def evaluate(p):
        calls.append(p)
        return dict(cost=0. if p==target else 5. if p==prior else 10.,
            eligible=eligible or p!=target)
    result=search(dates=dates,keys=('easy','quality','long'),previous=prior,
        roles={'easy':'easy','quality':'quality','long':'long'},seed_keys=('easy',),
        hard_end=dates[3],budget=1000,evaluate=evaluate)
    assert target in calls
    assert result['program']==(target if eligible else prior)
    assert result['local_neighborhood_complete']
    assert len(calls)==len(set(calls))==result['evaluations']<=1000
    assert result['continuity']['tolerance'] is None
