"""Actual miles fund volume; they do not buy extra growth steps or quality credit."""
from dataclasses import replace
from datetime import timedelta
import json

import pytest

from run_analysis.staged_progression import (
    MILE, ProgressionPolicy, ResponseReport, progression_distance_credit,
    review_recent_delivery,
)
from run_analysis.workload_accounting import Quantity
from test_staged_progression import stage, entry
from test_stage_acceptance import ledger_entries


def review(s, rows, **kwargs):
    return review_recent_delivery(s, rows, s.review_at, ledger_complete=True,
        longest_prior_30d_miles=8.4, distance_credit='actual_distance_v2', **kwargs)


def distance(e, miles):
    return replace(e, completion=replace(e.completion,
        distance_meters=Quantity.point(miles*MILE, 'actual')))


@pytest.mark.parametrize('miles', [2., 3.1, 3.2, 3.3, 6.])
def test_measured_distance_not_prescription_owns_new_credit(miles):
    e=ledger_entries(stage(weekly_equivalent_miles=20.))[1]  # 3.2 issued
    c=progression_distance_credit(e.contract, Quantity.point(miles*MILE, 'actual'),
        basis='actual_distance_v2')
    assert c['credited_miles']==pytest.approx(miles)
    # Archived behavior is unchanged, including in-band tolerance.
    old=progression_distance_credit(e.contract, Quantity.point(miles*MILE, 'actual'))
    assert old['credited_miles']<=3.2+1e-8


def test_overshoot_can_fund_reduced_later_prescription_without_increasing_growth_rate():
    s=stage(weekly_equivalent_miles=20.,policy=ProgressionPolicy(14,.05,'easy_volume',21.,'gradual'))
    rows=ledger_entries(s)  # 40 nominal miles
    rows[1]=distance(rows[1],6.2)  # +3 actual
    # Later prescription is shortened by 2 miles and followed exactly.
    later=rows[2]
    target=replace(later.contract.targets[0],quantity=Quantity.point(2*MILE,'issued'))
    rows[2]=distance(replace(later,contract=replace(later.contract,targets=(target,))),2.)
    r=review(s,rows,required_miles=40.9)
    assert r['credited_miles']==pytest.approx(41.)
    assert r['volume_accounting']['prescribed_miles']==pytest.approx(38.)
    assert r['status']=='conditional_development_candidate'
    assert r['candidate'].weekly_equivalent_miles==21.
    assert r['candidate'].long_miles==s.long_miles
    # Very large extra mileage cannot turn one review into multiple growth steps.
    huge=review(s,[rows[0],distance(rows[1],30.),*rows[2:]],required_miles=40.9)
    assert huge['candidate']==r['candidate']
    assert review_recent_delivery(s,rows,s.review_at,ledger_complete=True,
        longest_prior_30d_miles=8.4,required_miles=40.9)['status']=='recent_delivery_hold_targets'


def test_credit_does_not_bypass_missing_endurance_health_or_observation_interval():
    s=stage(weekly_equivalent_miles=20.)
    rows=ledger_entries(s)
    rows[1]=distance(rows[1],20.)
    no_longs=[replace(e,contract=replace(e.contract,purpose='easy_volume')) for e in rows]
    assert review(s,no_longs)['status']=='endurance_reference_not_represented'
    for state,expected in [('health_concern','health_review_required'),('adverse','dose_specific_review_required')]:
        assert review(s,rows,responses=(ResponseReport(s.review_at,state,'athlete'),))['status']==expected
    early=review_recent_delivery(s,rows,s.review_at-timedelta(seconds=1),ledger_complete=True,
        longest_prior_30d_miles=8.4,distance_credit='actual_distance_v2')
    assert early['status']=='continue_existing_stage'


def test_unavailable_incomplete_and_duplicate_evidence_cannot_buy_credit():
    s=stage(weekly_equivalent_miles=20.)
    rows=ledger_entries(s)
    late=distance(rows[1],30.)
    late=replace(late,completion=replace(late.completion,available_at=s.review_at+timedelta(seconds=1)))
    assert review(s,[rows[0],late,*rows[2:]])['credited_miles']==pytest.approx(36.8)
    assert review(s,[rows[0],replace(late,completion=None),*rows[2:]])['credited_miles']==pytest.approx(36.8)
    with pytest.raises(ValueError,match='Completion counted twice'):
        review(s,[*rows,replace(rows[1],key='different-slot')])
    e=entry(s)
    unresolved=progression_distance_credit(e.contract,None,basis='actual_distance_v2')
    assert unresolved['credited_miles']==0
    uncertain=Quantity(9*MILE,11*MILE,10*MILE,'bounded measurement')
    assert progression_distance_credit(e.contract,uncertain,basis='actual_distance_v2')['credited_miles']==pytest.approx(9.)


def test_release_only_changes_credit_and_requires_observed_admission():
    from types import SimpleNamespace
    from run_analysis.adaptive.new_model_daily_harness import HarnessPolicy
    from run_analysis.adaptive.live_bridge import (
        actual_progression_credit_release_policy, observed_progression_release_policy,
        continuation_release_policy, runtime_version, LivePlannerReviewRequired,
    )
    s=stage(policy=ProgressionPolicy(14,.05,'easy_volume',21.,'gradual'))
    policy=continuation_release_policy(HarnessPolicy(delivery_contract='sustained_v1',
        continuation_boundary=True,soft_horizon_taper=True,purpose_continuation=True,
        purpose_continuation_phase='known_full_phase_v1'))
    with pytest.raises(LivePlannerReviewRequired):
        actual_progression_credit_release_policy(policy,s)
    policy=observed_progression_release_policy(policy,s)
    updated=actual_progression_credit_release_policy(policy,s)
    assert updated==replace(policy,progression_credit='actual_distance_v2')
    assert actual_progression_credit_release_policy(updated,s)==updated
    assert runtime_version(SimpleNamespace(policy=updated))==208
    with pytest.raises(ValueError,match='Actual-distance progression credit'):
        HarnessPolicy(progression_credit='actual_distance_v2')


def test_lifecycle_passes_actual_credit_and_restart_retains_committed_rule(tmp_path,monkeypatch):
    from test_observed_progression_admission import completed, enable_observed
    from run_analysis.adaptive.adaptive_stage_lifecycle import advance_stage
    from run_analysis.adaptive.new_model_daily_harness import DailyHarness
    import run_analysis.adaptive.adaptive_stage_lifecycle as lifecycle
    h,inputs,config,at=completed(tmp_path,monkeypatch)
    enable_observed(h)
    h.policy=replace(h.policy,progression_credit='actual_distance_v2')
    seen=[]
    original=lifecycle.review_recent_delivery
    def capture(*args,**kwargs):
        seen.append(kwargs['distance_credit'])
        return original(*args,**kwargs)
    monkeypatch.setattr(lifecycle,'review_recent_delivery',capture)
    try:
        updated,r,event=advance_stage(h,inputs,at)
        assert seen==['actual_distance_v2']
        assert event['progression_credit']=='actual_distance_v2'
        assert r['distance_credit_basis']=='actual_distance_v2'
        again,hold,no_event=advance_stage(updated,inputs,at+timedelta(seconds=1))
        assert no_event is None and hold['status']=='continue_existing_stage'
        assert again.stage==updated.stage
        # Old/missing export must not undo the rule in the committed decision.
        path=h.sandbox.root/'scenario.json'
        value=json.loads(path.read_text());value['search_policy'].pop('progression_credit',None)
        path.write_text(json.dumps(value))
        resumed=DailyHarness.resume(h.sandbox.root,config,as_of=at+timedelta(seconds=1))
        try:
            assert resumed.policy.progression_credit=='actual_distance_v2'
            assert resumed.stage==updated.stage
        finally:
            resumed.sandbox.close()
    finally:
        h.sandbox.close()
