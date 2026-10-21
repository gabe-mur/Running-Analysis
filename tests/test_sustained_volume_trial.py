from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace
import pytest

from run_analysis.adaptive.program_objective_trial import rolling_volume_components,TrialPolicy
from run_analysis.adaptive.sustained_volume_trial import sustained_volume_cost
from run_analysis.adaptive.program_evidence_adapter import reconcile_completion
from run_analysis.adaptive.timely_purpose_trial import PurposeTimingPolicy
from run_analysis.program_delivery import TargetSegment
from test_continuous_program_adapter import AT,run,evaluate,context


def balance(values,*,smoothing=2):
    # One-day base window isolates the averaging operation, not athlete policy.
    rows=[SimpleNamespace(day=AT.date()+timedelta(days=i-len(values)+1),miles=v,role='easy')
          for i,v in enumerate(values) if v>0]
    return rolling_volume_components(rows,(),stage_start=AT.date(),decision=AT.date(),
        end=AT.date()+timedelta(days=1),policy=TrialPolicy(weekly_miles=7.,volume_window_days=1),
        supporting_miles=7.,smoothing_days=smoothing)['daily'][0]['volume']


def test_high_low_balance_before_clipping_but_persistent_deficit_is_not_free():
    assert balance((.8,1.2))['loss']==pytest.approx(0.)
    assert balance((1.2,.8))['loss']==pytest.approx(0.)
    assert balance((.8,.8))['loss']==pytest.approx(.04)
    assert balance((0.,0.))['loss']==1.
    assert balance((.8,),smoothing=1)['loss']==pytest.approx(.04)


def test_old_excess_expires_instead_of_banking_forever():
    assert balance((100.,0.,0.))['loss']==1.


@pytest.mark.parametrize('n',[0,-1,1.5,True])
def test_invalid_smoothing_is_rejected(n):
    with pytest.raises(ValueError,match='smoothing'):
        balance((1.,),smoothing=n)


def test_later_run_cannot_repay_a_past_shortage():
    ctx=replace(context(),policy=replace(context().policy,volume_window_days=28))
    a=evaluate((),(run(0,role='easy',work=0),),ctx=ctx)
    b=evaluate((),(run(34,role='easy',work=0),),ctx=ctx)
    x=sustained_volume_cost(ctx,a['evaluation'],smoothing_days=7)['rolling_volume']['daily'][0]
    y=sustained_volume_cost(ctx,b['evaluation'],smoothing_days=7)['rolling_volume']['daily'][0]
    assert x['volume']['delivered']>y['volume']['delivered']==0


def test_expected_completion_preserves_relative_score_with_fixed_endpoint():
    history=(run(-6,'observed','long',.6,0),run(-3,'observed'))
    expected=run()
    actual=replace(expected,key='actual',source='observed',available_at=expected.end,work_evidence_basis='explicit_segments')
    ctx=replace(context(),policy=replace(context().policy,volume_window_days=28,quality_work_minutes=12.),
        timing=PurposeTimingPolicy(1.,grace_fraction=1.))
    scores=[]
    for offset in (3,4):
        future=(run(offset),run(7,role='long',hrr=.6,work=0),run(23))
        before=evaluate(history,(expected,*future),ctx=ctx)
        h,p=reconcile_completion(history,(expected,*future),actual,replaces_key=expected.key,as_of=actual.end)
        after=evaluate(h,p,actual.end,ctx=ctx)
        # The adapter's decision is the day after a completed run; use the
        # actual evaluator scope for both replacement terms.
        later=replace(ctx,decision=AT.date()+timedelta(days=1))
        scores.append((sustained_volume_cost(ctx,before['evaluation'],smoothing_days=7)['cost'],
            sustained_volume_cost(later,after['evaluation'],smoothing_days=7)['cost']))
    assert scores[0][0]-scores[1][0]==pytest.approx(scores[0][1]-scores[1][1],abs=1e-12)


def test_one_day_reduces_exactly_to_existing_objective():
    ctx=replace(context(),policy=replace(context().policy,volume_window_days=28))
    result=evaluate((),(run(0),run(4)),ctx=ctx)['evaluation']
    assert sustained_volume_cost(ctx,result,smoothing_days=1)['cost']==pytest.approx(result.total_cost,abs=1e-12)


def test_dated_target_increases_are_not_backdated_by_smoothing():
    today=AT.date()
    segments=(TargetSegment(today-timedelta(days=33),today,14.),
        TargetSegment(today,today+timedelta(days=35),21.))
    values=rolling_volume_components((),(),stage_start=today,decision=today,
        end=today+timedelta(days=35),policy=TrialPolicy(weekly_miles=21.,volume_window_days=28),
        supporting_miles=None,volume_targets=segments,smoothing_days=7)['daily']
    # Six prior rolling readings at 56 miles, current reading at 57.
    assert values[0]['volume']['target']==pytest.approx((6*56+57)/7)
    assert values[-1]['volume']['target']==pytest.approx(84.)
    with pytest.raises(ValueError,match='coverage'):
        rolling_volume_components((),(),stage_start=today,decision=today,
            end=today+timedelta(days=35),policy=TrialPolicy(weekly_miles=21.,volume_window_days=28),
            supporting_miles=None,volume_targets=segments[1:],smoothing_days=7)


def test_tapered_one_day_matches_current_score_and_long_break_is_not_hidden():
    from test_soft_horizon_trial import fixture,candidate
    from run_analysis.adaptive.shared_program_evaluator import evaluate_program
    ctx,_,rows=fixture()
    ctx=replace(ctx,soft_horizon_taper=True)
    ev=evaluate_program(ctx,candidate(ctx,rows))
    assert sustained_volume_cost(ctx,ev,smoothing_days=1)['cost']==pytest.approx(ev.total_cost,abs=1e-12)
    # A very large run expires after base-window + smoothing - 1 dates.
    today=AT.date()
    history=(SimpleNamespace(day=today-timedelta(days=34),miles=1000.,role='easy'),)
    values=rolling_volume_components(history,(),stage_start=today,decision=today,
        end=today+timedelta(days=2),policy=TrialPolicy(volume_window_days=28),
        supporting_miles=7.,smoothing_days=7)['daily']
    assert all(r['volume']['loss']==1. and r['ordinary_easy']['loss']==1. for r in values)


def test_integrated_contract_matches_prototype_without_changing_physical_credit():
    from test_soft_horizon_trial import fixture,candidate
    from run_analysis.adaptive.shared_program_evaluator import evaluate_program
    ctx,_,rows=fixture()
    ctx=replace(ctx,soft_horizon_taper=True)
    original=evaluate_program(ctx,candidate(ctx,rows))
    integrated=replace(ctx,delivery_contract='sustained_v1')
    result=evaluate_program(integrated,candidate(integrated,rows))
    trial=sustained_volume_cost(ctx,original,smoothing_days=integrated.volume_smoothing_days)
    assert result.total_cost==pytest.approx(trial['cost'],abs=1e-12)
    assert result.parts['delivery']==original.parts['delivery']
    assert result.parts['targets']==original.parts['targets']
    assert result.weighted_demand_cost==original.weighted_demand_cost
    assert result.short_run_cost==original.short_run_cost
