"""Accounting invariants and explicit counterexamples, not controller acceptance."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from scripts.prototype_completion_progression import Allocation, Resolution, reconcile, course

AT=datetime(2026,10,19,tzinfo=timezone.utc)


@pytest.mark.parametrize('volume',[5.,20.,60.])
def test_completed_increase_is_conserved_across_scales(volume):
    result=course([8,6,6],[8,6.4,6.4],[8,6.4,6.4],scale=volume/20)
    assert result['distance_increment_delivered']
    assert result['eligible_increment_low']==pytest.approx(volume*.04)
    assert result['authorized_target_change']==0  # An attribution is not a policy.


@pytest.mark.parametrize('volume',[5.,20.,60.])
def test_flat_actuals_do_not_fund_requested_growth(volume):
    result=course([8,6,6],[8,6.4,6.4],[8,6,6],scale=volume/20)
    assert result['eligible_increment_low']==0
    assert not result['distance_increment_delivered']


def test_redistributing_volume_without_an_increase_cannot_create_credit():
    result=course([8,6,6],[10,4,6],[10,4,6])
    assert result['positive_only_credit']==2  # The rejected clipped approach.
    assert result['signed_increment_low']==0
    assert result['eligible_increment_low']==0


def test_partial_credit_is_visible_immediately_but_is_not_an_irreversible_claim():
    early=course([8,6,6],[8,6.4,6.4],[10,4,6],known=1)
    complete=course([8,6,6],[8,6.4,6.4],[10,4,6])
    assert early['eligible_increment_low']==pytest.approx(.8)
    assert early['status']=='provisional_accounting'
    assert not early['distance_increment_delivered']
    assert early['authorized_target_change']==0
    assert complete['eligible_increment_low']==0
    # Promoting early and refusing to reverse it would invent an increase.


def test_overshoots_can_offset_an_omission_in_volume_not_quality_or_fatigue():
    result=course([8,6,6],[8,6.4,6.4],[8,12.8,None])
    assert result['actual_miles']==pytest.approx(20.8)
    assert result['distance_increment_delivered']
    assert result['eligible_increment_low']==pytest.approx(.8)
    assert result['future_physical_credit']==0


def test_more_runs_do_not_manufacture_total_volume_growth():
    few=course([8,6,6],[8,6.4,6.4],[8,6,6])
    many=course([4,4,3,3,3,3],[4,4,3.2,3.2,3.2,3.2],[4,4,3,3,3,3],known=6)
    for key in ('baseline_miles','proposed_miles','actual_miles','eligible_increment_low'):
        assert few[key]==pytest.approx(many[key])


def test_calendar_shift_without_expiry_does_not_remove_known_credit():
    a=(Allocation('a',AT,20,20.8,'frozen'),)
    done=(Resolution('a',AT+timedelta(days=1),'physical',20.8,20.8),)
    before=reconcile(a,done,as_of=AT+timedelta(days=2))
    after=reconcile(a,done,as_of=AT+timedelta(days=500))
    assert before==after


def test_refresh_and_receipt_order_do_not_multiply_credit():
    a=(Allocation('a',AT,10,10.4,'frozen'),Allocation('b',AT,10,10.4,'frozen'))
    r=(Resolution('a',AT,'one',10.4,10.4),Resolution('b',AT,'two',10.4,10.4))
    result=reconcile(a,r,as_of=AT)
    for _ in range(100):
        assert reconcile(a,tuple(reversed(r)),as_of=AT)==result


def test_measurement_bounds_do_not_manufacture_an_increase():
    a=(Allocation('a',AT,20,20.8,'frozen'),)
    result=reconcile(a,(Resolution('a',AT,'physical',19,21),),as_of=AT)
    assert result['signed_increment_low']==-1
    assert result['signed_increment_high']==1
    assert not result['distance_increment_delivered']


def test_future_receipt_is_not_current_delivery_and_fresh_time_alone_earns_nothing():
    a=(Allocation('a',AT,20,20.8,'frozen'),)
    r=(Resolution('a',AT+timedelta(days=1),'physical',20.8,20.8),)
    result=reconcile(a,r,as_of=AT)
    assert result['eligible_increment_low']==0
    assert not result['all_allocations_settled']
    assert reconcile(a,(),as_of=AT+timedelta(days=1000))['eligible_increment_low']==0


@pytest.mark.parametrize('kind',['duplicate_allocation','duplicate_resolution','duplicate_physical',
                                 'unknown_allocation','retrospective_baseline'])
def test_identities_and_original_baselines_cannot_be_rewritten_to_buy_credit(kind):
    a=[Allocation('a',AT,10,10.4,'frozen'),Allocation('b',AT,10,10.4,'frozen')]
    r=[Resolution('a',AT,'one',10.4,10.4),Resolution('b',AT,'two',10.4,10.4)]
    if kind=='duplicate_allocation':a.append(a[0])
    elif kind=='duplicate_resolution':r.append(r[0])
    elif kind=='duplicate_physical':r[1]=replace(r[1],physical_key='one')
    elif kind=='unknown_allocation':r[1]=replace(r[1],allocation_key='unknown')
    else:a[1]=replace(a[1],frozen_at=AT+timedelta(seconds=1))
    with pytest.raises(ValueError):reconcile(a,r,as_of=AT+timedelta(days=1))


def test_changing_the_allocation_group_changes_when_it_is_settled():
    # Falsifies the idea that this can just become a windowless authorization
    # rule. Group closure is itself a gate even if its name has no day count.
    a=[Allocation('a',AT,10,10.4,'frozen'),Allocation('b',AT,10,10.4,'frozen')]
    r=[Resolution('a',AT,'one',10.4,10.4)]
    short=reconcile(a[:1],r,as_of=AT)
    long=reconcile(a,r,as_of=AT)
    assert short['distance_increment_delivered']
    assert not long['distance_increment_delivered']
    assert short['authorized_target_change']==long['authorized_target_change']==0
