from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace
import json

import pytest

from test_staged_progression import AT, history, stage
from run_analysis.volume_retention import (
    VolumeReviewBlock, VolumeRetentionPolicy, recent_volume_proposal, review_volume_retention,
)
from run_analysis.adaptive.factory import initial_target_history
from run_analysis.stage_target_history import target_segments
from run_analysis.program_delivery import _target_total


def block(index=0, *, rate=10., issued_rate=20., intent='ordinary', complete=True):
    start = AT+timedelta(days=14*index)
    return VolumeReviewBlock(start,start+timedelta(days=14),issued_rate*2,rate*2,rate*2,
                             complete,start+timedelta(days=14),'explicit fixture',intent)


def review(blocks,day,**kwargs):
    return review_volume_retention(stage(weekly_equivalent_miles=20.),blocks,AT+timedelta(days=day),**kwargs)


def test_normal_rest_and_same_evidence_refresh_never_rebase_targets():
    for day in (0,1,3,7,13):
        result=review([],day)
        assert result['proposed_volume_per_7'] is None
        assert result['retained_exposure_volume_per_7']==20.
    rows=[block(rate=20.)]
    for day in (14,15,20,27):
        assert review(rows,day)['retained_exposure_volume_per_7']==20.


def test_reduced_volume_needs_sustained_evidence_and_does_not_recompound_on_refresh():
    rows=[block(),block(1)]
    assert review(rows,14)['status']=='await_sustained_reduction'
    expected=review(rows,28)
    assert expected['proposed_volume_per_7']==15.
    assert expected['requires_return_dose_and_quality_review']
    assert not expected['accepted'] and not expected['physiology_established']
    for day in (28,29,35,41):
        assert review(rows,day)['proposed_volume_per_7']==15.
    assert review(rows+[block(2)],42)['proposed_volume_per_7']==13.5


def test_confirmed_break_decays_each_block_once_not_twice():
    rows=[block(i,rate=0.,issued_rate=0.,intent='confirmed_break') for i in range(3)]
    for day,expected in ((14,14.1),(28,10.),(42,7.1)):
        assert review(rows,day)['proposed_volume_per_7']==expected


def test_sparse_refresh_and_daily_refresh_produce_same_result():
    rows=[block(i) for i in range(3)]
    direct=review(rows,42)
    for day in range(43):
        polled=review(rows,day)
    assert direct==polled


def test_missing_uploads_and_planner_shortfall_cannot_ratchet_target_down():
    cases=[([block(i,complete=False) for i in range(2)],'incomplete_recording_hold'),
           ([block(i,issued_rate=10.) for i in range(2)],'planner_under_delivery_review')]
    for rows,status in cases:
        result=review(rows,28)
        assert result['status']==status
        assert result['proposed_volume_per_7'] is None
        assert result['retained_exposure_volume_per_7']==20.
    assert review([],28)['status']=='missing_review_block_hold'


def test_one_deload_is_protected_but_repeated_intentional_reduction_is_not_immortal():
    rows=[block(i,issued_rate=10.,intent='planned_deload') for i in range(2)]
    assert review(rows,14)['status']=='planned_deload_hold'
    assert review(rows,14)['proposed_volume_per_7'] is None
    assert review(rows,28)['proposed_volume_per_7']==15.
    recovered=[rows[0],block(1,rate=20.)]
    assert review(recovered,28)['retained_exposure_volume_per_7']==20.


def test_late_block_is_not_known_earlier_and_corrections_replay_without_stale_decay():
    rows=[block(),replace(block(1),available_at=AT+timedelta(days=30))]
    assert review(rows,28)['status']=='missing_review_block_hold'
    assert review(rows,30)['proposed_volume_per_7']==15.
    assert review([block(rate=20.),block(1,rate=20.)],30)['proposed_volume_per_7'] is None
    earlier_late=[replace(block(),available_at=AT+timedelta(days=30)),block(1)]
    assert review(earlier_late,28)['status']=='missing_review_block_hold'
    assert review(earlier_late,30)['proposed_volume_per_7']==15.


def test_compliant_band_credit_is_not_raw_load_or_automatic_growth():
    rows=[replace(block(i,rate=20.),actual_miles=37.) for i in range(2)]
    assert review(rows,28)['retained_exposure_volume_per_7']==20.
    assert review([block(i,rate=30.) for i in range(2)],28)['retained_exposure_volume_per_7']==20.


def test_invalid_block_identity_or_policy_fails():
    with pytest.raises(ValueError,match='Contiguous'):
        review([block(),block()],28)
    with pytest.raises(ValueError,match='Contiguous'):
        review([block(1)],28)
    with pytest.raises(ValueError,match='cannot contain'):
        block(intent='confirmed_break')
    with pytest.raises(ValueError):
        VolumeRetentionPolicy(half_life_days=0)


def test_recent_candidate_is_one_time_and_future_data_cannot_change_it():
    rows=history()
    base=stage(weekly_equivalent_miles=10.)
    a=recent_volume_proposal(rows,AT,base,timezone='UTC')
    assert a['candidate'].weekly_equivalent_miles>10.
    assert a['candidate'].long_miles==base.long_miles
    assert a['historical_stage']==base
    assert not a['accepted']
    future=replace(rows[-1],key='future',start=AT+timedelta(days=2),
                   end=AT+timedelta(days=2,hours=1),available_at=AT+timedelta(days=2,hours=1))
    assert recent_volume_proposal(rows+[future],AT,base,timezone='UTC')==a
    later=recent_volume_proposal(rows,AT+timedelta(days=3),base,timezone='UTC')
    assert later['observed_volume_per_7']==a['observed_volume_per_7']
    assert recent_volume_proposal(rows,AT+timedelta(days=20),base,timezone='UTC')['candidate'] is None


def candidate(s,support):
    return dict(stage=s,training_policy=SimpleNamespace(weekly_miles=s.weekly_equivalent_miles),
                easy_duration_policy=SimpleNamespace(supporting_easy_miles_per_7=support))


def test_prospective_initial_target_preserves_historical_volume_and_support():
    old=stage(weekly_equivalent_miles=16.5)
    new=replace(old,key='forward',weekly_equivalent_miles=18.6)
    records=initial_target_history(candidate(new,6.3),AT,'UTC',baseline_candidate=candidate(old,4.4))
    day=AT.date()
    volume,support=target_segments(records,as_of=AT,start=day-timedelta(days=27),end=day+timedelta(days=21))
    assert _target_total(volume,day-timedelta(days=27),day)==pytest.approx(27*16.5/7)
    assert _target_total(volume,day,day+timedelta(days=7))==18.6
    assert _target_total(support,day-timedelta(days=27),day)==pytest.approx(27*4.4/7)
    assert _target_total(support,day,day+timedelta(days=7))==6.3
    assert len(initial_target_history(candidate(old,4.4),AT,'UTC'))==1


def test_prototype_report_serializes_actual_planned_dates():
    from scripts.prototype_volume_target import encode
    value=dict(stage=stage(),planned=[(AT.date(),'easy',4.4)])
    restored=json.loads(json.dumps(value,default=encode))
    assert restored['planned'][0][0]==AT.date().isoformat()
