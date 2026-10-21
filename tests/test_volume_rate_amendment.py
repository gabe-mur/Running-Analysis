from dataclasses import replace
from datetime import datetime,timedelta,timezone
import json

import pytest

from run_analysis.adaptive.volume_rate_amendment import propose, apply
from run_analysis.general_progression import RATE_VERSION
from run_analysis.staged_progression import DevelopmentStage,ProgressionPolicy
from run_analysis.stage_target_history import StageTarget,target_segments,target_miles_between

AT=datetime(2026,10,3,21,tzinfo=timezone.utc)


def fixture():
    stage=DevelopmentStage('old',AT,18.1,4.2,8.9,17.,7.,
        ProgressionPolicy(14,.05,'easy_volume',19.,'gradual'),'test')
    targets=(StageTarget('old-path',AT,AT.date(),18.1,5.4,'test',14,19.,6.3),)
    return stage,targets


def test_amendment_preserves_today_past_baseline_long_and_original_review():
    stage,targets=fixture(); now=AT+timedelta(days=4)
    review=propose(stage,targets,now,'UTC'); new=review['candidate']; path=review['amended_target']
    assert review['status']=='conditional_rate_amendment' and not review['confirmed_delivery']
    assert new.issued_at==stage.issued_at and new.review_at==stage.review_at
    assert (new.weekly_equivalent_miles,new.long_miles,new.ordinary_miles)==(18.1,8.9,4.2)
    assert new.policy.volume_goal_miles==19.4
    assert path.ramp_days==10 and path.review_anchor==AT
    assert path.rate(now.date(),'volume_per_7')==targets[-1].rate(now.date(),'volume_per_7')
    old,_=target_segments(targets,as_of=now,start=AT.date(),end=now.date()+timedelta(days=1))
    revised,_=target_segments((*targets,path),as_of=now,start=AT.date(),end=now.date()+timedelta(days=1))
    assert target_miles_between(old,AT,now,timezone.utc)==target_miles_between(revised,AT,now,timezone.utc)
    assert propose(new,(*targets,path),now+timedelta(days=1),'UTC')['candidate'] is None


def test_no_auto_review_bypass_or_backdated_growth():
    stage,targets=fixture()
    for now in (AT,stage.review_at,stage.review_at+timedelta(days=1)):
        assert propose(stage,targets,now,'UTC')['candidate'] is None


def setup(tmp_path,monkeypatch):
    from test_adaptive_stage_lifecycle import completed_stage
    h,inputs,cfg,_=completed_stage(tmp_path,monkeypatch,growth_focus='easy_volume',gradual=True,
        continuation_boundary=True,single_volume=True,prospective_review_hour=4,
        purpose_continuation=True,continuation_distribution='c1_v1')
    h.sandbox.close()
    # Restore a causal prefix before the review, not the fixture's completed future.
    # The pure/lifecycle fixture below uses the existing origin decision only.
    root=tmp_path/'state'
    import sqlite3
    at=h.origin+timedelta(days=4)
    with sqlite3.connect(root/'sandbox.sqlite') as db:
        for key,available in db.execute('SELECT activity_id,available_at FROM sandbox_completions').fetchall():
            if datetime.fromisoformat(available)>at:
                db.execute('DELETE FROM sandbox_completions WHERE activity_id=?',(key,))
    from run_analysis.adaptive.new_model_daily_harness import DailyHarness
    resumed=DailyHarness.resume(root,cfg,as_of=at)
    resumed.policy=replace(resumed.policy,general_progression=RATE_VERSION)
    return resumed,inputs,cfg,at


def test_real_evaluator_atomic_amendment_restart_and_checkpoint(tmp_path,monkeypatch):
    from run_analysis.adaptive.new_model_daily_harness import DailyHarness
    from run_analysis.adaptive.decision_snapshot import restore
    h,inputs,cfg,at=setup(tmp_path,monkeypatch)
    original=h.stage; oldtargets=h.target_history
    scenario=h.sandbox.root/'scenario.json'; data=json.loads(scenario.read_text())
    from dataclasses import asdict
    data['search_policy']=asdict(h.policy); scenario.write_text(json.dumps(data))
    try:
        updated,review,event=apply(h,inputs,at)
        assert event and event['stage_acceptance'], review
        assert review['confirmed_delivery'] is False
        evidence=json.loads(updated.sandbox.db.execute('SELECT evidence FROM stages WHERE key=?',
            (updated.stage.key,)).fetchone()[0])
        assert evidence['review']['confirmed_delivery'] is False
        assert updated.stage.weekly_equivalent_miles==original.weekly_equivalent_miles
        assert updated.stage.review_at==original.review_at
        assert updated.target_history[:-1]==oldtargets
        assert updated.scales==h.scales
        assert updated.catalog['quality:ordinary']==h.catalog['quality:ordinary']
        state=restore((h.sandbox.root/event['snapshot_file']).read_text(),expected_fingerprint=event['snapshot_fingerprint'])
        assert state.prospective_volume_review.start==original.issued_at+timedelta(hours=4)
        assert state.prospective_volume_review.end==original.review_at+timedelta(hours=4)
        reopened=DailyHarness.resume(h.sandbox.root,cfg,as_of=at)
        try:
            assert reopened.stage==updated.stage and reopened.target_history==updated.target_history
        finally:
            reopened.sandbox.close()
    finally:
        h.sandbox.close()


def test_rejected_preview_does_not_commit(tmp_path,monkeypatch):
    from run_analysis.adaptive.stage_acceptance import StageLedger
    h,inputs,_,at=setup(tmp_path,monkeypatch); before=h.stage,h.target_history,len(h.events)
    def reject(*args): raise ValueError('diagnostic churn gate')
    try:
        with pytest.raises(ValueError,match='churn gate'):
            apply(h,inputs,at,preview_check=reject)
        assert (h.stage,h.target_history,len(h.events))==before
        assert StageLedger.for_sandbox(h.sandbox).current==h.stage
    finally:
        h.sandbox.close()


def test_readiness_prevents_amendment(tmp_path,monkeypatch):
    from run_analysis.session_decision import CurrentContext
    h,inputs,_,at=setup(tmp_path,monkeypatch)
    h.current_context=CurrentContext(at.date(),at,'little_tired','current report')
    try:
        updated,review,event=apply(h,inputs,at)
        assert updated is h and event is None
        assert review['status']=='rate_amendment_readiness_hold'
    finally:
        h.sandbox.close()
