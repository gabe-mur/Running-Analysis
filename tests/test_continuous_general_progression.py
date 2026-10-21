"""Continuous accounting controls; small seed searches are not stability proof."""
from copy import copy
from dataclasses import replace
from datetime import timedelta
import json

import pytest

from run_analysis.adaptive.continuous_general_progression import propose,VERSION
from run_analysis.adaptive.adaptive_stage_lifecycle import advance_stage,review_entries
from run_analysis.adaptive.new_model_daily_harness import DailyHarness,HarnessPolicy
from run_analysis.adaptive.stage_acceptance import StageLedger
from run_analysis.stage_target_history import StageTarget,target_segments
from run_analysis.staged_progression import ResponseReport


def fixture(tmp_path,monkeypatch,**kwargs):
    from test_adaptive_stage_lifecycle import completed_stage
    h,inputs,config,at=completed_stage(tmp_path,monkeypatch,growth_focus='easy_volume',
        gradual=True,continuation_boundary=True,single_volume=True,prospective_review_hour=4,**kwargs)
    h.policy=replace(h.policy,progression_admission='observed_rolling_v2',
        progression_credit='actual_distance_v2',general_progression=VERSION)
    return h,inputs,config,at


def proposal(h,at):
    return propose(h,at,review_entries(h,at),longest=h.stage.long_miles)


def adopt(h,p):
    new=copy(h)
    new.general_progression_state=p['continuous_state']
    new.target_history=(*h.target_history,p['target'])
    return new


def test_policy_is_explicit_and_does_not_reinterpret_archived_defaults():
    assert HarnessPolicy().general_progression is None
    with pytest.raises(ValueError,match='Continuous progression'):
        HarnessPolicy(delivery_contract='sustained_v1',general_progression=VERSION)


def test_explicit_continuous_bridge_version_signature_and_policy_survive_refresh():
    from datetime import timezone
    from types import SimpleNamespace
    from test_prospective_progression import fixture as stage_fixture
    from run_analysis.adaptive.live_bridge import (
        continuation_release_policy,observed_progression_release_policy,
        actual_progression_credit_release_policy,general_progression_release_policy,
        runtime_version,input_signature)
    stage,_=stage_fixture()
    policy=continuation_release_policy(HarnessPolicy(delivery_contract='sustained_v1',
        continuation_boundary=True,soft_horizon_taper=True,purpose_continuation=True,
        purpose_continuation_phase='known_full_phase_v1'))
    policy=actual_progression_credit_release_policy(observed_progression_release_policy(policy,stage),stage)
    updated=replace(policy,general_progression=VERSION)
    assert runtime_version(SimpleNamespace(policy=policy))==208
    assert runtime_version(SimpleNamespace(policy=updated))==209
    assert general_progression_release_policy(updated,stage)==updated
    assert replace(updated,general_progression=policy.general_progression)==policy
    h=SimpleNamespace(config={},zone=timezone.utc,history=(),policy=policy,
        decision=lambda at:at.date(),frequency_contract=None,planner_preferences=None)
    captured=SimpleNamespace(contexts=(),restrictions=(),responses=(),metadata=())
    before=input_signature(h,captured,stage.issued_at)
    h.policy=updated
    assert input_signature(h,captured,stage.issued_at)!=before


def test_experimental_general_path_cannot_silently_fall_back_to_old_race_baseline(tmp_path,monkeypatch):
    h,inputs,_,at=fixture(tmp_path,monkeypatch)
    before=h.stage,h.target_history,len(h.events)
    monkeypatch.setattr('run_analysis.adaptive.race_event.preparation_active',lambda *_:True)
    try:
        with pytest.raises(ValueError,match='explicit race-target migration'):
            advance_stage(h,inputs,at)
        assert (h.stage,h.target_history,len(h.events))==before
    finally:h.sandbox.close()


def test_continuous_path_compounds_daily_caps_destination_and_adds_only_easy_support():
    from datetime import datetime,timezone
    at=datetime(2026,10,1,tzinfo=timezone.utc)
    row=StageTarget('continuous',at,at.date(),20.,8.,'test',growth_per_7=.04,growth_limit=42.5)
    for day in (0,1,7,14,28,100000):
        expected=min(42.5,20.*1.04**(min(day,1000)/7))
        assert row.rate(at.date()+timedelta(days=day),'volume_per_7')==pytest.approx(expected)
        assert row.rate(at.date()+timedelta(days=day),'supporting_easy_per_7')==pytest.approx(expected-12.)
    assert row.rate(at.date()+timedelta(days=15),'volume_per_7')>row.rate(at.date()+timedelta(days=14),'volume_per_7')


@pytest.mark.parametrize('volume,long,goal',[(5.,2.,4.),(20.,8.,17.),(60.,15.,17.)])
def test_scale_controls_use_same_elapsed_rate_without_forcing_volume_above_destination(monkeypatch,volume,long,goal):
    from datetime import datetime,timezone
    from types import SimpleNamespace
    from run_analysis.staged_progression import DevelopmentStage,ProgressionPolicy
    from run_analysis.general_progression import destination
    at=datetime(2026,10,1,tzinfo=timezone.utc)
    stage=DevelopmentStage('scale',at-timedelta(days=30),volume,1.,long,goal,7.,
        ProgressionPolicy(14,.05,'easy_volume',volume,'gradual'),'pure arithmetic control')
    h=SimpleNamespace(stage=stage,planner_preferences=None,general_progression_state=None,
        target_history=(StageTarget('seed',stage.issued_at,at.date()-timedelta(days=40),volume,volume/2,'test'),),
        decision=lambda when:when.date(),zone=timezone.utc,policy=SimpleNamespace(progression_credit='actual_distance_v2'))
    monkeypatch.setattr('run_analysis.adaptive.continuous_general_progression.review_recent_delivery',
        lambda *a,**k:dict(status='observed_delivery_supported',credited_miles=volume*3))
    first=propose(h,at,(),longest=long)
    new=adopt(h,first)
    week=propose(new,at+timedelta(days=7),(),longest=long)
    assert week['continuous_state']['volume']==pytest.approx(min(destination(stage).volume_per_7,volume*1.04))
    assert week['continuous_state']['long_miles']<=long*1.05


def test_intraday_slope_review_never_changes_today_or_rewrites_archived_view():
    from datetime import datetime,timezone
    at=datetime(2026,10,1,tzinfo=timezone.utc)
    old=StageTarget('old',at,at.date(),20.,8.,'test',growth_per_7=0.,growth_limit=30.)
    new=replace(old,stage_key='new',available_at=at+timedelta(hours=1),growth_per_7=.04)
    before,_=target_segments((old,new),as_of=at,start=at.date(),end=at.date()+timedelta(days=8))
    after,_=target_segments((old,new),as_of=new.available_at,start=at.date(),end=at.date()+timedelta(days=8))
    assert before[0]==after[0]
    assert after[-1].miles_per_7_days==pytest.approx(20.*1.04)
    with pytest.raises(ValueError,match='reprice today'):
        target_segments((old,replace(new,volume_per_7=21.)),as_of=new.available_at,
            start=at.date(),end=at.date()+timedelta(days=2))


def test_daily_review_does_not_confirm_endpoint_or_create_new_phase(tmp_path,monkeypatch):
    h,_,_,at=fixture(tmp_path,monkeypatch)
    try:
        p=proposal(h,at)
        assert p['gate_status']=='observed_delivery_supported'
        assert p['status']=='continuous_growth' and p['candidate'] is None
        assert p['continuous_state']['volume']==21.
        assert p['target'].rate(at.date()+timedelta(days=14),'volume_per_7')==pytest.approx(21.*1.04**2)
        new=adopt(h,p)
        again=proposal(new,at+timedelta(hours=1))
        assert again['continuous_state']['volume']==21.
        assert again['continuous_state']['observation_anchor']==h.stage.issued_at.isoformat()
        tomorrow=proposal(new,at+timedelta(days=1))
        assert tomorrow['continuous_state']['volume']==pytest.approx(21.*1.04**(1/7))
        after_gap=proposal(new,at+timedelta(days=30))
        assert after_gap['continuous_state']['volume']==tomorrow['continuous_state']['volume']
        assert not after_gap['continuous_state']['growth_allowed']
    finally:h.sandbox.close()


@pytest.mark.parametrize('cause',['no_delivery','adverse','health_concern'])
def test_hold_prevents_growth_and_does_not_bank_missed_days(tmp_path,monkeypatch,cause):
    h,_,_,at=fixture(tmp_path,monkeypatch)
    try:
        entries=() if cause=='no_delivery' else review_entries(h,at)
        responses=() if cause=='no_delivery' else (ResponseReport(at,cause,'test actual response'),)
        p=propose(h,at,entries,responses=responses,longest=h.stage.long_miles)
        assert p['status']=='continuous_growth_hold' and p['target'].growth_per_7==0
        new=adopt(h,p)
        later=propose(new,at+timedelta(days=3),(),longest=h.stage.long_miles)
        assert later['continuous_state']['volume']==p['continuous_state']['volume']
        assert later['continuous_state']['long_budget']==h.stage.long_miles
    finally:h.sandbox.close()


def test_duplicate_actual_credit_is_rejected_and_recent_minimum_still_required(tmp_path,monkeypatch):
    h,_,_,at=fixture(tmp_path,monkeypatch)
    try:
        entries=review_entries(h,at)
        with pytest.raises(ValueError,match='Duplicate opportunity'):
            propose(h,at,(*entries,entries[0]),longest=h.stage.long_miles)
        early=proposal(h,at-timedelta(days=1))
        assert early['gate_status']=='insufficient_observation_history'
        assert not early['continuous_state']['growth_allowed']
    finally:h.sandbox.close()


def test_full_publication_restart_and_failed_search_are_atomic(tmp_path,monkeypatch):
    h,inputs,config,at=fixture(tmp_path,monkeypatch)
    original=h.stage,h.scales,h.catalog,h.training_policy,h.target_history,h.context(at).stage_start
    try:
        new,review,event=advance_stage(h,inputs,at)
        assert event['reason']=='continuous_general_review' and 'stage_acceptance' not in event
        assert new.stage==original[0] and new.scales==original[1] and new.catalog==original[2]
        assert new.training_policy==original[3] and new.target_history[:-1]==original[4]
        assert new.context(at).stage_start==original[5]
        assert StageLedger.for_sandbox(h.sandbox).current==h.stage
        assert event['progression_target']['mode']=='continuous_observed_volume'
        # Committed runtime owns policy even if scenario export stayed old.
        scenario_path=h.sandbox.root/'scenario.json'
        scenario=json.loads(scenario_path.read_text())
        scenario['search_policy']['general_progression']=None
        scenario_path.write_text(json.dumps(scenario))
        resumed=DailyHarness.resume(h.sandbox.root,config,as_of=at)
        try:
            assert resumed.policy.general_progression==VERSION
            assert resumed.general_progression_state==new.general_progression_state
            assert resumed.target_history==new.target_history
            assert resumed.context(at)==new.context(at)
        finally:resumed.sandbox.close()
        with pytest.raises(ValueError,match='timestamps must advance'):
            advance_stage(new,inputs,at)
        retried,_,_=advance_stage(new,inputs,at+timedelta(microseconds=1))
        assert retried.target_history[-1].volume_per_7==new.target_history[-1].volume_per_7
        assert {k:v for k,v in retried.general_progression_state.items() if k!='available_at'}=={
            k:v for k,v in new.general_progression_state.items() if k!='available_at'}
        new=retried
        from run_analysis.adaptive.continuous_search_adapter import NoEligibleProgram
        def fail(**kwargs):raise NoEligibleProgram('explicit atomic control')
        monkeypatch.setattr('run_analysis.adaptive.new_model_daily_harness.migrated_search',fail)
        before=(new.stage,new.target_history,new.general_progression_state,len(new.events))
        with pytest.raises(NoEligibleProgram):advance_stage(new,inputs,at+timedelta(days=1))
        assert (new.stage,new.target_history,new.general_progression_state,len(new.events))==before
    finally:h.sandbox.close()


def test_long_budget_grows_only_with_observed_support_and_not_from_forecast(tmp_path,monkeypatch):
    h,_,_,at=fixture(tmp_path,monkeypatch)
    try:
        p=proposal(h,at)
        new=adopt(h,p)
        new.general_progression_state=dict(new.general_progression_state,
            volume=30.,support=17.,long_budget=8.49,long_growth_allowed=True)
        def reviewed(*args,**kwargs):
            return dict(status='observed_delivery_supported',credited_miles=60.,stage_key=h.stage.key)
        monkeypatch.setattr('run_analysis.adaptive.continuous_general_progression.review_recent_delivery',reviewed)
        tomorrow=proposal(new,at+timedelta(days=1))
        assert tomorrow['long_progression'] and tomorrow['candidate'].long_miles==8.5
        assert tomorrow['continuous_state']['long_budget']<8.49*1.05
        assert proposal(new,at+timedelta(seconds=1))['candidate'] is None
        new.general_progression_state=dict(new.general_progression_state,volume=20.,support=7.)
        unsupported=proposal(new,at+timedelta(days=1))
        assert not unsupported['long_progression']
        assert unsupported['continuous_state']['long_budget']==h.stage.long_miles
    finally:h.sandbox.close()


def test_supported_growth_depends_on_elapsed_days_not_how_often_app_is_opened(tmp_path,monkeypatch):
    h,_,_,at=fixture(tmp_path,monkeypatch)
    try:
        def reviewed(*args,**kwargs):
            return dict(status='observed_delivery_supported',credited_miles=60.,stage_key=h.stage.key)
        monkeypatch.setattr('run_analysis.adaptive.continuous_general_progression.review_recent_delivery',reviewed)
        initial=adopt(h,proposal(h,at))
        weekly=proposal(initial,at+timedelta(days=7))
        daily=initial
        for n in range(1,8):daily=adopt(daily,proposal(daily,at+timedelta(days=n)))
        assert weekly['continuous_state']['volume']==pytest.approx(daily.general_progression_state['volume'])
        assert weekly['continuous_state']['volume']==pytest.approx(21.*1.04)
    finally:h.sandbox.close()


def test_supported_long_command_update_keeps_continuous_clock_and_commits_atomically(tmp_path,monkeypatch):
    h,inputs,config,at=fixture(tmp_path,monkeypatch,weekly_volume=30.)
    try:
        initial=proposal(h,at)
        h.general_progression_state=dict(initial['continuous_state'],
            day=(at.date()-timedelta(days=1)).isoformat(),volume=31.3,
            support=18.,long_budget=8.49,long_growth_allowed=True,growth_allowed=True)
        old_state=h.general_progression_state
        new,review,event=advance_stage(h,inputs,at)
        assert review['gate_status']=='observed_delivery_supported'
        assert review['credited_miles']==pytest.approx(63.)
        assert review['credited_miles']>=review['required_miles']
        assert event['stage_acceptance']
        assert new.stage.long_miles==8.5
        assert new.scales==h.scales and new.catalog['quality:ordinary']==h.catalog['quality:ordinary']
        assert new.general_progression_state['observation_anchor']==old_state['observation_anchor']
        assert new.context(at).stage_start==h.context(at).stage_start
        resumed=DailyHarness.resume(h.sandbox.root,config,as_of=at)
        try:
            assert resumed.stage==new.stage
            assert resumed.general_progression_state==new.general_progression_state
            assert resumed.context(at)==new.context(at)
        finally:resumed.sandbox.close()
    finally:h.sandbox.close()
