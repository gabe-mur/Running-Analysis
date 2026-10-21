from dataclasses import replace
from datetime import timedelta
import pytest

from run_analysis.program_delivery import TargetSegment
from scripts.program_objective_trial import rolling_volume_components
from scripts.timely_purpose_trial import PurposeTimingPolicy
from scripts.program_evidence_adapter import reconcile_completion
from test_continuous_program_adapter import AT,run,evaluate,context
from test_r3_decision_controls import programme,context as r3context,evaluate as r3evaluate


def test_remote_boundary_volume_cannot_pay_for_an_early_deficit():
    # A far terminal session lies outside every window that includes day zero.
    ctx=replace(context(),policy=replace(context().policy,volume_window_days=28))
    first=(run(0,role='easy',work=0),)
    later=(run(34,role='easy',work=0),)
    a=evaluate((),first,ctx=ctx)['evaluation'].parts['rolling_volume']
    b=evaluate((),later,ctx=ctx)['evaluation'].parts['rolling_volume']
    assert a['daily'][0]['volume']['delivered']>b['daily'][0]['volume']['delivered']
    assert b['daily'][0]['volume']['delivered']==0


def test_new_goal_does_not_retroactively_raise_old_target():
    ctx=context()
    policy=replace(ctx.policy,volume_window_days=28,weekly_miles=21.)
    start=AT.date()-timedelta(days=27)
    end=AT.date()+timedelta(days=35)
    segments=(TargetSegment(start,AT.date(),20.),TargetSegment(AT.date(),end,21.))
    result=rolling_volume_components((),(),stage_start=AT.date(),decision=AT.date(),end=end,
        policy=policy,supporting_miles=7.,volume_targets=segments,
        support_targets=(TargetSegment(start,end,7.),))
    assert result['daily'][0]['volume']['target']==pytest.approx((27*20+21)/7)
    assert result['daily'][27]['volume']['target']==84.
    with pytest.raises(ValueError,match='coverage'):
        rolling_volume_components((),(),stage_start=AT.date(),decision=AT.date(),end=end,policy=policy,
            supporting_miles=7.,volume_targets=segments[1:])


def test_expected_completion_preserves_alternative_ranking_with_rolling_delivery():
    history=(run(-6,'observed','long',.6,0),run(-3,'observed'))
    expected=run()
    actual=replace(expected,key='actual',source='observed',available_at=expected.end,work_evidence_basis='explicit_segments')
    ctx=replace(context(),policy=replace(context().policy,volume_window_days=28,quality_work_minutes=12.),
        timing=PurposeTimingPolicy(1.,grace_fraction=1.))
    values=[]
    for offset in (3,4):
        future=(run(offset),run(7,role='long',hrr=.6,work=0),run(23))
        before=evaluate(history,(expected,*future),ctx=ctx)
        h,p=reconcile_completion(history,(expected,*future),actual,replaces_key=expected.key,as_of=actual.end)
        after=evaluate(h,p,actual.end,ctx=ctx)
        values.append((before['evaluation'].total_cost,after['evaluation'].total_cost))
    assert values[0][0]-values[1][0]==pytest.approx(values[0][1]-values[1][1],abs=1e-12)


def test_undertraining_and_run_fragmentation_are_not_free():
    ctx=replace(r3context(),policy=replace(r3context().policy,volume_window_days=28),
        timing=PurposeTimingPolicy(1.,grace_fraction=1.))
    balanced=programme()
    deficient=tuple(r for r in balanced if r.role!='easy')
    assert r3evaluate(balanced,ctx=ctx)['evaluation'].total_cost<r3evaluate(deficient,ctx=ctx)['evaluation'].total_cost
