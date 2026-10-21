"""Opt-in lifecycle controls; small fixture seeds do not prove calendar stability."""
from dataclasses import replace
from datetime import timedelta
import json

import pytest

from run_analysis.adaptive.smoothed_general_progression import VERSION
from run_analysis.adaptive.continuous_general_progression import propose
from run_analysis.adaptive.adaptive_stage_lifecycle import review_entries,advance_stage
from run_analysis.adaptive.new_model_daily_harness import DailyHarness,HarnessPolicy
from run_analysis.staged_progression import ResponseReport
from test_continuous_general_progression import fixture as old_fixture,adopt


def fixture(tmp_path,monkeypatch,**kwargs):
    h,inputs,config,at=old_fixture(tmp_path,monkeypatch,**kwargs)
    h.policy=replace(h.policy,general_progression=VERSION)
    return h,inputs,config,at


def proposal(h,at,**kwargs):
    return propose(h,at,review_entries(h,at),longest=h.stage.long_miles,**kwargs)


def test_new_policy_is_explicit_and_requires_existing_observed_actual_contract():
    assert HarnessPolicy().general_progression is None
    with pytest.raises(ValueError,match='Continuous progression'):
        HarnessPolicy(delivery_contract='sustained_v1',general_progression=VERSION)


def test_opt_in_has_distinct_runtime_and_same_physical_state(tmp_path,monkeypatch):
    from run_analysis.adaptive.live_bridge import runtime_version
    h,_,_,at=fixture(tmp_path,monkeypatch)
    try:
        assert runtime_version(h)==212
        before=h.stage,h.scales,h.catalog,h.training_policy,h.target_history
        r=proposal(h,at)
        assert r['smoothed_support']['future_credit'] is False
        assert 0<=r['target'].growth_per_7<=.04+1e-12
        assert (h.stage,h.scales,h.catalog,h.training_policy,h.target_history)==before
        assert r['continuous_state']['version']==VERSION
    finally:h.sandbox.close()


@pytest.mark.parametrize('cause',['health_concern','adverse','missing_volume','early'])
def test_guards_still_hold_without_invented_credit(tmp_path,monkeypatch,cause):
    h,_,_,at=fixture(tmp_path,monkeypatch)
    try:
        rows=() if cause=='missing_volume' else review_entries(h,at)
        responses=(ResponseReport(at,cause,'actual'),) if cause in {'health_concern','adverse'} else ()
        review_at=at-timedelta(days=1) if cause=='early' else at
        r=propose(h,review_at,rows,
            responses=responses,longest=h.stage.long_miles)
        assert r['target'].growth_per_7==0
        assert not r['long_progression']
        assert r['continuous_state']['volume']==h.target_history[-1].rate(h.decision(review_at),'volume_per_7')
    finally:h.sandbox.close()


def test_intraday_refresh_does_not_compound_target_and_no_training_gap_does_not_bank_growth(tmp_path,monkeypatch):
    h,_,_,at=fixture(tmp_path,monkeypatch)
    try:
        first=proposal(h,at);new=adopt(h,first)
        for hours in (1,2,3):
            r=proposal(new,at+timedelta(hours=hours))
            assert r['continuous_state']['volume']==first['continuous_state']['volume']
            new=adopt(new,r)
        r=propose(new,at+timedelta(days=90),(),longest=h.stage.long_miles)
        assert r['continuous_state']['volume']==first['continuous_state']['volume']
        assert r['target'].growth_per_7==0
    finally:h.sandbox.close()


def test_full_publication_and_restart_retain_opt_in_and_fractional_rate(tmp_path,monkeypatch):
    h,inputs,config,at=fixture(tmp_path,monkeypatch)
    try:
        old=h.stage,h.catalog,h.scales,h.training_policy
        updated,r,event=advance_stage(h,inputs,at)
        assert event and event['reason']=='continuous_general_review'
        assert (updated.stage,updated.catalog,updated.scales,updated.training_policy)==old
        assert r['continuous_state']['version']==VERSION
        scenario_path=h.sandbox.root/'scenario.json'
        value=json.loads(scenario_path.read_text())
        value['search_policy']['general_progression']=None
        scenario_path.write_text(json.dumps(value))
        resumed=DailyHarness.resume(h.sandbox.root,config,as_of=at)
        try:
            assert resumed.policy.general_progression==VERSION
            assert resumed.general_progression_state==updated.general_progression_state
            assert resumed.target_history==updated.target_history
            assert resumed.context(at)==updated.context(at)
        finally:resumed.sandbox.close()
    finally:h.sandbox.close()


def test_race_cannot_silently_fall_back_to_old_progression(tmp_path,monkeypatch):
    h,inputs,_,at=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr('run_analysis.adaptive.race_event.preparation_active',lambda *_:True)
    try:
        with pytest.raises(ValueError,match='explicit race-target migration'):
            advance_stage(h,inputs,at)
    finally:h.sandbox.close()


def test_endurance_window_controls_long_growth_not_general_volume():
    from types import SimpleNamespace
    from test_smoothed_progression import running,targets,AT,ANCHOR
    from test_staged_progression import stage
    rows=running()
    # All recent volume is easy; an old long is not sudden permission for a
    # longer long. Its leaving the 14-day set must not veto general volume.
    rows=[replace(e,contract=replace(e.contract,purpose='easy_volume'))
        if e.scheduled_at>=AT-timedelta(days=14) else e for e in rows]
    s=stage(issued_at=ANCHOR,weekly_equivalent_miles=20.)
    h=SimpleNamespace(stage=s,planner_preferences=None,general_progression_state=None,
        target_history=targets(),decision=lambda at:at.date(),zone=AT.tzinfo,
        policy=SimpleNamespace(general_progression=VERSION,progression_credit='actual_distance_v2'))
    r=propose(h,AT,rows,longest=s.long_miles)
    assert r['legacy_window_status']=='endurance_reference_not_represented'
    assert r['target'].growth_per_7>0
    assert not r['long_progression'] and not r['continuous_state']['long_growth_allowed']


def test_observed_long_growth_keeps_atomic_publication_and_clock(tmp_path,monkeypatch):
    h,inputs,config,at=fixture(tmp_path,monkeypatch,weekly_volume=30.)
    try:
        initial=proposal(h,at)
        h.general_progression_state=dict(initial['continuous_state'],
            day=(at.date()-timedelta(days=1)).isoformat(),volume=31.3,
            support=18.,long_budget=8.49,long_growth_allowed=True,growth_allowed=True,
            growth_per_7=.04,supported_volume_ceiling=34.)
        anchor=h.general_progression_state['observation_anchor']
        updated,r,event=advance_stage(h,inputs,at)
        assert r['smoothed_support']['actual_per_7']>=h.stage.long_miles/.4
        assert r['long_progression'] and event['stage_acceptance']
        assert event['reason']=='accepted_continuous_long_dose'
        assert updated.stage.long_miles==8.5
        assert updated.scales==h.scales
        assert updated.general_progression_state['observation_anchor']==anchor
        resumed=DailyHarness.resume(h.sandbox.root,config,as_of=at)
        try:
            assert resumed.stage==updated.stage
            assert resumed.general_progression_state==updated.general_progression_state
        finally:resumed.sandbox.close()
    finally:h.sandbox.close()
