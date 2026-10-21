from datetime import date,timedelta
from itertools import product

from run_analysis.adaptive.new_model_daily_harness import Dose,neighbors
from run_analysis.adaptive.program_objective_trial import TrialPolicy
from scripts.audit_adaptive_landscape import families,descend,block_candidates,canonical


DATES=tuple((date(2026,9,27)+timedelta(days=i)).isoformat() for i in range(14))
CATALOG={d.key:d for d in (Dose('short','easy',2.,(('easy',1200.),)),
    Dose('easy','easy',4.,(('easy',2400.),)),Dose('long','long',8.,(('easy',4800.),)),
    Dose('long_reduced','long',6.,(('easy',3600.),)),
    Dose('quality','quality',4.,(('work',1500.),)),Dose('quality_reduced','quality',3.,(('work',1000.),)))}


def test_stratified_samples_are_reproducible_complete_physical_and_diverse():
    anchor=((DATES[0],'long'),(DATES[3],'easy'),(DATES[6],'quality'),(DATES[10],'short'))
    args=(DATES,CATALOG,TrialPolicy(),6.,anchor)
    a=families(*args,count=40,seed=2)
    assert a==families(*args,count=40,seed=2)
    assert a!=families(*args,count=40,seed=3)
    assert set(a)=={'stage_lattice','nonperiodic_spaced','uniform_counts','coupled_incumbent'}
    for rows in a.values():
        assert rows and len(rows)<=40
        for p in rows:
            assert canonical(p)==p
            assert all(d in DATES and k in CATALOG for d,k in p)
    assert len({len(p) for p in a['uniform_counts']})>=5
    assert {k for p in a['uniform_counts'] for _,k in p}>=set(CATALOG)-{'short','easy'}


def test_exhaustive_block_changes_joint_assignments_but_never_its_exterior():
    p=((DATES[0],'easy'),(DATES[1],'long'),(DATES[5],'easy'))
    rows=list(block_candidates(p,DATES[:2],('easy','long')))
    assert len(rows)==len(set(rows))==9
    assert all((DATES[5],'easy') in r for r in rows)
    assert ((DATES[0],'long'),(DATES[1],'easy'),(DATES[5],'easy')) in rows
    assert ((DATES[5],'easy'),) in rows


def test_equal_lane_quota_counts_shared_cached_proposals_and_rejected_candidates():
    p=((DATES[0],'easy'),)
    calls=[]
    cache={}
    def price(p):
        calls.append(p)
        return cache.setdefault(p,dict(cost=0. if p==((DATES[1],'long'),) else 10.,
            eligible=all(k!='long' for _,k in p)))
    a=descend(p,DATES[:3],('easy','long'),price,5)
    b=descend(p,DATES[:3],('easy','long'),price,5)
    assert a==b and a['proposals']==5
    assert a['status']=='budget_exhausted'
    assert a['program']==p


def test_tiny_lane_optimum_matches_exhaustive_without_locking_start():
    dates=DATES[:3]
    target=((dates[0],'easy'),(dates[2],'long'))
    def price(p):
        return dict(cost=sum(dict(p).get(d)!=dict(target).get(d) for d in dates),eligible=True)
    result=descend(((dates[1],'easy'),),dates,('easy','long'),price,100)
    exhaustive=[canonical((d,k) for d,k in zip(dates,ks) if k)
        for ks in product((None,'easy','long'),repeat=3)]
    assert result['cost']==min(price(p)['cost'] for p in exhaustive)==0
    assert result['program']==target
    assert result['status']=='coordinate_complete'


def test_joint_occupied_exchange_crosses_worse_single_role_changes():
    dates=DATES[:3]
    prior=((dates[0],'quality'),(dates[1],'easy'),(dates[2],'long'))
    target=((dates[0],'easy'),(dates[1],'quality'),(dates[2],'long'))
    def price(p):
        return dict(cost=0. if p==target else 5. if p==prior else 10.,eligible=True)
    for d,k in target[:2]:
        one=canonical({**dict(prior),d:k}.items())
        assert price(one)['cost']>price(prior)['cost']
    result=descend(prior,dates,('easy','quality','long'),price,500)
    assert result['program']==target
    assert [r['cost'] for r in result['path']]==[5.,0.]


def test_two_day_crossing_keeps_intervening_session_unchanged():
    prior=((DATES[0],'easy'),(DATES[1],'quality'),(DATES[3],'long'))
    crossed=((DATES[1],'quality'),(DATES[2],'easy'),(DATES[3],'long'))
    assert crossed in set(neighbors(prior,DATES[:4],('easy','quality','long')))
