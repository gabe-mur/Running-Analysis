from dataclasses import replace
from datetime import timedelta
import pytest

from scripts.timely_purpose_trial import envelope_day,PurposeTimingPolicy
from scripts.program_evidence_adapter import reconcile_completion
from test_continuous_program_adapter import AT,run,evaluate,context


def test_full_coverage_has_no_reward_for_repeated_early_refresh():
    once=((AT.date(),1.),)
    daily=tuple((AT.date()+timedelta(days=d),1.) for d in range(7))
    for d in range(7):
        day=AT.date()+timedelta(days=d)
        assert envelope_day(once,day,grace=7,memory=7)['loss']==0.
        assert envelope_day(daily,day,grace=7,memory=7)['loss']==0.


def test_partial_work_cannot_bank_full_credit_and_expiry_is_soft():
    events=tuple((AT.date()+timedelta(days=d),.5) for d in range(7))
    r=envelope_day(events,AT.date()+timedelta(days=6),grace=7,memory=7)
    assert r['opening']==.5 and r['loss']==.25
    one=((AT.date(),1.),)
    loss=[envelope_day(one,AT.date()+timedelta(days=d),grace=7,memory=7)['loss'] for d in range(6,10)]
    assert 0==loss[0]<loss[1]<loss[2]<loss[3]<1


def test_exact_envelope_handles_crossing_and_fractional_cadence():
    from math import exp
    events=((AT.date()-timedelta(days=9),1.),(AT.date(),.72))
    day=AT.date()
    exact=envelope_day(events,day,grace=6.7,memory=7)['loss']
    n=10000
    numerical=sum((1-max(exp(-(9+(i+.5)/n-6.7)/7),.72))**2/n for i in range(n))
    assert exact==pytest.approx(numerical,abs=1e-9)


@pytest.mark.parametrize('stage_work',[3.,12.,30.])
def test_new_coverage_preserves_expected_upload_relative_comparisons(stage_work):
    history=(run(-6,'observed','long',.6,0),run(-3,'observed'))
    expected=run()
    actual=replace(expected,key='actual',source='observed',available_at=expected.end,work_evidence_basis='explicit_segments')
    ctx=replace(context(),policy=replace(context().policy,quality_work_minutes=stage_work),
        timing=PurposeTimingPolicy(1.,grace_fraction=1.))
    values=[]
    for offset in (3,4):
        future=(run(offset),run(7,role='long',hrr=.6,work=0),run(23))
        before=evaluate(history,(expected,*future),ctx=ctx)
        h,p=reconcile_completion(history,(expected,*future),actual,replaces_key=expected.key,as_of=actual.end)
        after=evaluate(h,p,actual.end,ctx=ctx)
        values.append((before['evaluation'].total_cost,after['evaluation'].total_cost))
    assert values[0][0]-values[1][0]==pytest.approx(values[0][1]-values[1][1],abs=1e-12)
