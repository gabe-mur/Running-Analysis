"""Race coaching inputs through the SAME volume measurement and controller."""
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace

import pytest

from run_analysis.adaptive.progression_policy import resolve
from run_analysis.adaptive.race_goal import RaceGoalRevision
from run_analysis.adaptive.cadence_mass_growth import estimate
from run_analysis.adaptive.realized_growth import shadow_proposal
from run_analysis.adaptive.smoothed_general_progression import propose
from run_analysis.adaptive.adaptive_stage_lifecycle import review_entries, advance_stage
from run_analysis.adaptive.new_model_daily_harness import DailyHarness
from run_analysis.race_preparation import destination, MARATHON_POLICIES
from test_cadence_mass_growth import course
from test_prescription_progression import ZONE
from test_smoothed_general_progression import fixture
from test_continuous_general_progression import adopt


def race(h, at, distance='marathon', mode='balanced', days=90):
    h.race_goal = RaceGoalRevision.create(available_at=at, distance=distance,
        race_date=h.decision(at)+timedelta(days=days), preparation_mode=mode,
        parent_key='explicit-test', general_long_goal_miles=h.stage.goal_long_miles)
    return h.race_goal


@pytest.mark.parametrize('distance,mode', [('5k','balanced'),('10k','balanced'),
    ('half_marathon','balanced'),('marathon','shorter'),('marathon','balanced'),('marathon','more')])
def test_existing_race_rate_and_target_are_inputs_not_a_new_algorithm(tmp_path,monkeypatch,distance,mode):
    h,_,_,at=fixture(tmp_path,monkeypatch)
    try:
        h.policy=replace(h.policy,general_progression='continuous_general_v7')
        revision=race(h,at,distance,mode)
        r=resolve(h,at)
        policy=MARATHON_POLICIES[mode].progression if distance=='marathon' else h.stage.policy
        assert r.volume_growth_per_7==pytest.approx((1+policy.growth_fraction)**(7/policy.review_days)-1)
        assert r.target==destination(h.stage,revision.goal)
        assert r.long_share==r.target.long_miles/r.target.volume_per_7
        p=propose(h,at,review_entries(h,at),longest=h.stage.long_miles)
        assert p['continuous_state']['coaching_progression']['target_long_miles']==r.target.long_miles
        assert p['target'].growth_per_7<=r.volume_growth_per_7+1e-12
        assert p['target'].growth_limit<=r.target.volume_per_7
        assert p['target'].rate(revision.taper_start,'volume_per_7')==pytest.approx(
            p['target'].rate(revision.race_date,'volume_per_7'))
    finally:h.sandbox.close()


@pytest.mark.parametrize('cap',[.04,.06,.08])
def test_same_actual_measurement_and_uncapped_credit_only_change_prospective_cap(cap):
    rows,refs,at,current=course(rate=.09)
    s=estimate(rows,refs,as_of=at,zone=ZONE,baseline_volume=20.,maximum_growth_per_7=cap)
    assert s['growth_per_7']==pytest.approx(cap)
    assert s['measured_volume_per_7']==pytest.approx(current)
    p=shadow_proposal(20.,dict(actual_per_7=current,delivery_ratio=1.,mean_age_days=14),s,
        dict(funded_level=20.),baseline_volume=20.,maximum_growth_per_7=cap)
    assert p['growth_per_7']==pytest.approx(cap)
    assert p['recognized_planning_floor']==pytest.approx(current)
    assert p['physical_credit_added']==0


@pytest.mark.parametrize('phase,days', [('taper',10),('race',0),('post_race',-1)])
def test_dated_race_holds_stop_both_ramps_without_resetting_history(tmp_path,monkeypatch,phase,days):
    h,_,_,at=fixture(tmp_path,monkeypatch)
    try:
        h.policy=replace(h.policy,general_progression='continuous_general_v7')
        before=propose(h,at,review_entries(h,at),longest=h.stage.long_miles)
        h=adopt(h,before)
        clock=at+timedelta(days=1)
        revision=race(h,clock,days=days)
        assert revision.phase(h.decision(clock))==phase
        p=propose(h,clock,review_entries(h,clock),longest=h.stage.long_miles)
        assert p['target'].growth_per_7==0 and not p['long_progression']
        assert p['continuous_state']['volume']==before['continuous_state']['volume']
        assert p['continuous_state']['realized_reference']==before['continuous_state']['realized_reference']
        assert p['continuous_state']['coaching_progression']['phase']==phase
    finally:h.sandbox.close()


def test_race_publication_and_restart_keep_shared_controller(tmp_path,monkeypatch):
    h,inputs,config,at=fixture(tmp_path,monkeypatch)
    try:
        h.policy=replace(h.policy,general_progression='continuous_general_v7')
        race(h,at)
        original=h.stage,h.scales,tuple(h.history),h.training_policy
        updated,review,event=advance_stage(h,inputs,at)
        assert event and review['continuous_state']['coaching_progression']['phase']=='build'
        assert (updated.stage,updated.scales,tuple(updated.history),updated.training_policy)==original
        resumed=DailyHarness.resume(h.sandbox.root,config,as_of=at)
        try:
            assert resumed.general_progression_state==updated.general_progression_state
            assert resumed.race_goal==updated.race_goal
            assert resumed.policy.general_progression=='continuous_general_v7'
        finally:resumed.sandbox.close()
    finally:h.sandbox.close()


@pytest.mark.parametrize('ending',['clear','complete','did_not_start'])
def test_general_inputs_return_without_reusing_the_race_rate_or_target(ending):
    from test_staged_progression import stage, AT
    from run_analysis.adaptive.race_event import RaceEventResolution
    h=SimpleNamespace(stage=stage(),planner_preferences=None,decision=lambda clock:clock.date())
    revision=race(h,AT,days=1)
    clock=AT+timedelta(days=2)
    if ending=='clear':
        h.race_goal=RaceGoalRevision.create(available_at=clock,distance='general_fitness',race_date=None,
            preparation_mode='balanced',parent_key=revision.key,general_long_goal_miles=17.)
    elif ending=='complete':clock=AT+timedelta(days=30)
    else:
        h.race_event=RaceEventResolution.create(race_key=revision.key,status='did_not_start',
            available_at=clock,parent_key=revision.key,effective_on=clock.date())
    resolved=resolve(h,clock)
    assert resolved.phase=='general_fitness' and resolved.volume_growth_per_7==.04
    assert resolved.target.long_miles==17. and resolved.long_share==.4
