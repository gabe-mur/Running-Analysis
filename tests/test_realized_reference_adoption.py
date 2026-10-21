"""Recognition is not a capped growth increment or a reset of normalization."""
from copy import deepcopy
from datetime import datetime, timedelta
import json

import pytest

from run_analysis.adaptive.realized_growth import shadow_proposal, review_reference
from run_analysis.adaptive.smoothed_progression import advance
from test_realized_growth import course, fit, AT, ZONE


def observed(level=22.,rate=.04,*,day=28):
    # Build actual available distance evidence with the requested end level;
    # the 20-mile no-growth normalization remains fixed and independent.
    rows,refs=course(rate,days=day)
    last=max((e.scheduled_at-AT).days for e in rows)
    rows,refs=course(rate,days=day,level=level/(20*(1+rate)**(last/7)))
    return fit(rows,refs)


def physical(level=22.):
    return dict(actual_per_7=level,delivery_ratio=1.,mean_age_days=14)


def review(previous=None,*,volume=20.,signal=None,permitted=True,funded=20.,destination=100.):
    signal=signal or observed()
    return review_reference(previous,volume=volume,physical=physical(),signal=signal,
        funding=dict(funded_level=funded),baseline_volume=20.,
        as_of=datetime.fromisoformat(signal['as_of']),zone=ZONE,destination=destination,permitted=permitted)


@pytest.mark.parametrize('old_funding',[10.,20.,None])
def test_twenty_two_is_recognized_and_future_four_percent_uses_twenty_two(old_funding):
    signal=observed()
    result=review(signal=signal,funded=old_funding)
    state=result['state']; p=result['proposal']
    assert state['volume']==pytest.approx(22.)
    assert state['observed_reference_volume']==pytest.approx(22.)
    assert state['baseline_volume']==20.
    assert state['growth_per_7']==pytest.approx(.04)
    assert result['newly_recognized_increase']==pytest.approx(2.)
    # This is a CONDITIONAL seven-day projection, not credit for future runs.
    assert advance(state['volume'],p,elapsed_days=7,destination=100.)==pytest.approx(22.88)
    assert result['physical_credit_added']==0.


def test_recognition_of_todays_level_does_not_backdate_growth_on_that_level():
    first=review(signal=observed(level=20.,day=21))
    previous=first['state']
    # Seven elapsed calendar days; 22 is recognized TODAY, not seven days ago.
    signal=observed(level=22.,day=28)
    result=review(previous,volume=previous['volume'],signal=signal)
    assert result['elapsed_days']==7
    assert result['prior_base_accrual']==pytest.approx(.8)
    assert result['state']['volume']==pytest.approx(22.)
    assert result['newly_recognized_increase']==pytest.approx(1.2)
    assert result['state']['volume']!=pytest.approx(22.88)


def test_repeated_and_resumed_review_cannot_recognize_the_same_extra_miles_twice():
    signal=observed(); state=review(signal=signal)['state']
    state=json.loads(json.dumps(state))
    for _ in range(12):
        result=review(state,volume=state['volume'],signal=signal)
        assert result['newly_recognized_increase']==0
        assert result['prior_base_accrual']==0
        assert result['state']==state
        state=result['state']


@pytest.mark.parametrize('rate',[0.,.01,.02,.04,.06])
def test_actual_level_is_not_capped_even_when_additional_growth_is_partial_or_zero(rate):
    result=review(signal=observed(level=22.,rate=rate))
    assert result['state']['volume']==pytest.approx(22.)
    assert result['state']['growth_per_7']==pytest.approx(min(rate,.04))
    future=advance(result['state']['volume'],result['proposal'],elapsed_days=7,destination=100.)
    assert future==pytest.approx(22*(1+min(rate,.04)))


def test_health_hold_blocks_adoption_and_does_not_bank_growth_during_hold():
    signal=observed(day=21)
    held=review(signal=signal,permitted=False)
    assert held['state']['volume']==20.
    assert held['state']['growth_per_7']==0
    assert held['newly_recognized_increase']==0
    fresh=observed(level=22.,day=28)
    resumed=review(held['state'],volume=20.,signal=fresh)
    assert resumed['prior_base_accrual']==0
    assert resumed['state']['volume']==pytest.approx(22.)


@pytest.mark.parametrize('case',['approximate','unknown','no_actual'])
def test_unverified_or_absent_actuals_cannot_raise_the_reference(case):
    signal=observed(); actual=physical()
    if case=='approximate': signal=dict(signal,reference_quality='diagnostic_approximation')
    elif case=='unknown': signal=dict(signal,level_ratio=None,growth_per_7=0.,status='evidence_unresolved')
    else: actual=dict(actual,actual_per_7=0.)
    result=review_reference(None,volume=20.,physical=actual,signal=signal,
        funding=dict(funded_level=60.),baseline_volume=20.,
        as_of=datetime.fromisoformat(signal['as_of']),zone=ZONE,destination=100.)
    assert result['state']['volume']==20.
    assert result['state']['growth_per_7']==0.
    assert result['newly_recognized_increase']==0.


def test_below_current_target_does_not_erase_the_accepted_plan_or_manufacture_support():
    result=review(signal=observed(level=18.,rate=.02))
    assert result['state']['observed_reference_volume']==pytest.approx(18.)
    assert result['state']['volume']==20.
    assert result['state']['growth_per_7']==0
    assert result['state']['rate_status']=='observed_headroom_exhausted'


def test_published_growth_without_new_actuals_cannot_self_ratchet_the_reference():
    signal=observed(level=20.,rate=0.)
    result=review(signal=signal,funded=40.)
    assert result['state']['volume']==pytest.approx(20.)
    assert result['state']['growth_per_7']==0.
    assert advance(result['state']['volume'],result['proposal'],elapsed_days=180,destination=100.)==pytest.approx(20.)


def test_fixed_normalization_and_causal_review_checks_survive_recognition():
    signal=observed(); state=review(signal=signal)['state']
    args=dict(volume=state['volume'],physical=physical(),signal=signal,funding=dict(funded_level=20.),
        baseline_volume=20.,as_of=datetime.fromisoformat(signal['as_of']),zone=ZONE,destination=100.)
    with pytest.raises(ValueError,match='normalization baseline'):
        review_reference(state,**dict(args,baseline_volume=22.))
    with pytest.raises(ValueError,match='normalization baseline'):
        review_reference(state,**dict(args,signal=dict(signal,baseline_key='reset')))
    with pytest.raises(ValueError,match='match committed state'):
        review_reference(state,**dict(args,volume=20.))
    with pytest.raises(ValueError,match='causal review timestamp'):
        review_reference(state,**dict(args,as_of=args['as_of']-timedelta(days=1)))
    changed=deepcopy(signal); changed['level_ratio']+=.01
    with pytest.raises(ValueError,match='Same-time evidence'):
        review_reference(state,**dict(args,signal=changed))


def test_goal_is_separate_from_recognizing_the_observed_level():
    result=review(signal=observed(),destination=21.)
    assert result['state']['observed_reference_volume']==pytest.approx(22.)
    assert result['state']['volume']==21.
    assert advance(21.,result['proposal'],elapsed_days=7,destination=21.)==21.


@pytest.mark.parametrize('scale',[5.,20.,60.])
@pytest.mark.parametrize('multiple',[1.1,1.25])
def test_accelerated_actuals_are_not_limited_by_the_old_growth_envelope(scale,multiple):
    rows,refs=course(.06,days=28,scale=scale)
    last=max((e.scheduled_at-AT).days for e in rows)
    rows,refs=course(.06,days=28,scale=scale,level=multiple/1.06**(last/7))
    signal=fit(rows,refs)
    result=review_reference(None,volume=scale,physical=physical(scale*multiple),signal=signal,
        funding=dict(funded_level=scale),baseline_volume=scale,
        as_of=datetime.fromisoformat(signal['as_of']),zone=ZONE,destination=200.)
    assert result['state']['volume']==pytest.approx(scale*multiple)
    assert result['state']['growth_per_7']==pytest.approx(.04)
    assert advance(result['state']['volume'],result['proposal'],elapsed_days=7,destination=200.)==pytest.approx(scale*multiple*1.04)


def test_new_actual_recognition_can_happen_intraday_without_earning_extra_elapsed_growth():
    signal=observed(level=20.)
    first=review(signal=signal)['state']
    later=dict(observed(level=22.),as_of=(datetime.fromisoformat(signal['as_of'])+timedelta(hours=1)).isoformat())
    result=review(first,volume=first['volume'],signal=later)
    assert result['elapsed_days']==0 and result['prior_base_accrual']==0
    assert result['state']['volume']==pytest.approx(22.)
    assert result['newly_recognized_increase']==pytest.approx(2.)


def test_old_positive_signal_is_bounded_and_cannot_keep_recognizing_an_overshoot():
    signal=observed(level=22.)
    result=review(signal=signal); state=result['state']
    ceiling=result['proposal']['supported_volume_ceiling']
    start=datetime.fromisoformat(signal['as_of'])
    for days in range(1,181):
        stale=dict(signal,as_of=(start+timedelta(days=days)).isoformat())
        result=review(state,volume=state['volume'],signal=stale)
        assert result['newly_recognized_increase']==0
        assert result['state']['volume']<=ceiling
        state=result['state']
    assert state['growth_per_7']==0.


def test_unknown_startup_and_later_missing_evidence_do_not_reset_or_brick_reference():
    full=observed()
    unknown=dict(full,as_of=(datetime.fromisoformat(full['as_of'])-timedelta(days=1)).isoformat(),
        baseline_key=None,level_ratio=None,growth_per_7=0.,status='insufficient_opportunities')
    first=review(signal=unknown)['state']
    assert first['baseline_key'] is None and first['volume']==20.
    second=review(first,volume=20.,signal=full)['state']
    assert second['baseline_key']=='fixed' and second['volume']==pytest.approx(22.)
    later=dict(unknown,as_of=(datetime.fromisoformat(full['as_of'])+timedelta(days=1)).isoformat())
    held=review(second,volume=second['volume'],signal=later)['state']
    assert held['baseline_key']=='fixed' and held['volume']==second['volume']
    assert held['growth_per_7']==0.
