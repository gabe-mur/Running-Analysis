from types import SimpleNamespace as NS
import pytest
from run_analysis.adaptive.coached_pipeline import fund_bound_prior
from run_analysis.adaptive.native_quality_final_refinement import refine
from run_analysis.adaptive.new_model_daily_harness import DailyHarness


def test_bound_prior_repair_is_candidate_only_and_does_not_move_roles_or_relax_admission():
    dates=('2026-10-09','2026-10-10')
    catalog=dict(q=NS(role='quality',miles=3.),e2=NS(role='easy',miles=2.),
        e3=NS(role='easy',miles=3.),e4=NS(role='easy',miles=4.))
    physical={(dates[1],k):NS(start=10) for k in ('e2','e3','e4')}
    p=((dates[0],'q'),(dates[1],'e2'));calls=[]
    def evaluate(p):
        calls.append(p)
        key=p[1][1]
        return dict(eligible=key=='e3',cost={'e2':0.,'e3':2.,'e4':1.}[key])
    result,report=fund_bound_prior(p,dates=dates,catalog=catalog,physical=physical,
        state=NS(prospective_volume_review=NS(start=0,end=11)),evaluate=evaluate)
    assert result==((dates[0],'q'),(dates[1],'e3'))
    assert report['evaluations']==len(calls)==3
    assert report['admitted_alternatives']==1
    assert all(tuple(d for d,_ in r)==dates and r[0][1]=='q' for r in calls)


def test_healthy_already_funded_prior_needs_no_additions():
    p=(('2026-10-09','e'),)
    result,report=fund_bound_prior(p,dates=(),catalog={},physical={},
        state=NS(),evaluate=lambda p:dict(eligible=True,cost=1.))
    assert result==p and report['evaluations']==1 and not report['changed']


def test_coaching_cannot_be_accidentally_published_or_reuse_a_stale_winner():
    h=DailyHarness.__new__(DailyHarness);h.quality_runtime={}
    for kwargs in ({'publish':True},{'publish':False,'prepared':{}},
            {'publish':False,'replayed_selection':{}},{'publish':False,'require_stage_funding':True}):
        with pytest.raises(ValueError,match='unpublished'):
            h.replan(None,reason='test',coaching_first_purpose='sustained',**kwargs)


def test_terminal_binding_prices_actual_assignments_and_rechecks_dates(monkeypatch):
    import run_analysis.adaptive.native_quality_final_refinement as m
    monkeypatch.setattr(m,'quality_easy_neighbors',lambda *a,**kw:iter(()))
    monkeypatch.setattr(m,'decimal_pair_neighbors',lambda *a,**kw:iter(()))
    dates=('2026-10-09','2026-10-10','2026-10-11')
    p=((dates[0],'qA'),(dates[2],'qB'));calls=[]
    def bind(p):return tuple((d,'qA' if i==0 else 'qB') for i,(d,k) in enumerate(sorted(p)))
    target=((dates[0],'qA'),(dates[1],'qB'))
    def evaluate(p):
        assert p==bind(p);calls.append(p)
        return dict(eligible=True,cost=0. if p==target else 1.)
    result=refine(p,dict(eligible=True,cost=1.),evaluate=evaluate,dates=dates,
        roles=dict(qA='quality',qB='quality'),miles=dict(qA=3.,qB=4.),budget=20,program_transform=bind)
    assert result['program']==target
    assert result['evaluations']==len(set(calls))==len(calls)


def test_terminal_rejects_a_transform_that_changes_dates():
    p=(('2026-10-09','e'),)
    with pytest.raises(ValueError,match='preserve'):
        refine(p,dict(eligible=True,cost=1.),evaluate=lambda p:None,dates=('2026-10-09','2026-10-10'),
            roles=dict(e='easy'),miles=dict(e=3.),budget=1,
            program_transform=lambda p:(('2026-10-10','e'),))
