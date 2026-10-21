"""V4 accounting/lifecycle controls, not searched-plan or physiology acceptance."""
from dataclasses import replace
from datetime import timedelta
import json

import pytest

from run_analysis.adaptive.smoothed_progression import matched_growth,growth,advance
from run_analysis.adaptive.smoothed_general_progression import MATCHED_VERSION
from run_analysis.adaptive.continuous_general_progression import propose
from run_analysis.stage_target_history import StageTarget
from run_analysis.staged_progression import ResponseReport
from test_smoothed_progression import running,evidence,targets,AT,ANCHOR
from test_historical_progression import harness,physical,adopt


def matched_harness(rows=None):
    h=harness(physical(running(19.6) if rows is None else rows))
    h.policy.general_progression=MATCHED_VERSION
    h.target_history=(*h.target_history,StageTarget('ahead-endpoint',AT-timedelta(seconds=1),AT.date(),21.4,9.4,
        'existing conditional endpoint, not completed growth',growth_per_7=0.,growth_limit=42.5))
    return h


def test_one_lag_matched_comparison_does_not_apply_a_second_current_level_ceiling():
    e=dict(actual_per_7=17.78710496564,delivery_ratio=1.01756,mean_age_days=14)
    assert growth(19.4,e)['growth_per_7']==0
    p=matched_growth(19.4,e)
    assert p['growth_per_7']==pytest.approx(.04)
    assert p['observed_volume']==e['actual_per_7']
    assert p['physical_credit_added']==0
    assert p['future_growth_is_conditional']


@pytest.mark.parametrize('ratio',[0.,.8,.94,1.,1.08])
def test_matching_ratio_scales_response_without_new_injury_or_adaptation_numbers(ratio):
    p=matched_growth(20.,dict(actual_per_7=20*ratio,delivery_ratio=ratio,mean_age_days=14))
    assert 0<=p['growth_per_7']<=.04+1e-12
    if ratio<=.8:assert p['growth_per_7']==0
    if ratio>=1:assert p['growth_per_7']==pytest.approx(.04)


def test_health_override_and_real_absence_of_training_are_still_holds():
    e=dict(actual_per_7=20.,delivery_ratio=1.,mean_age_days=14)
    assert matched_growth(20.,e,permitted=False)['growth_per_7']==0
    assert matched_growth(20.,dict(e,actual_per_7=0.,delivery_ratio=0.))['growth_per_7']==0


def test_reference_and_proposed_target_stay_distinct_without_lowering_the_plan():
    h=matched_harness()
    old=h.stage,h.target_history,h.catalog if hasattr(h,'catalog') else None
    r=propose(h,AT,(),longest=8.4)
    state=r['continuous_state']
    assert state['version']==MATCHED_VERSION
    assert state['progression_reference']['baseline_volume']==20.
    assert state['planning_volume']==21.4
    assert state['observed_volume_per_7']<20.
    assert state['historical_recognized_increase']==0
    assert state['planning_volume_is_conditional']
    assert r['target'].volume_per_7==21.4
    assert r['target'].growth_per_7>0
    assert not state['long_growth_allowed']  # Actual easy/total support still governs the long.
    assert (h.stage,h.target_history,None)==old


def test_matched_policy_is_explicit_and_keeps_the_original_actual_evidence_contract():
    from run_analysis.adaptive.new_model_daily_harness import HarnessPolicy
    assert HarnessPolicy().general_progression is None
    with pytest.raises(ValueError,match='Continuous progression'):
        HarnessPolicy(delivery_contract='sustained_v1',general_progression=MATCHED_VERSION)


def test_unavailable_target_cannot_promote_the_planning_level_or_support():
    h=matched_harness()
    before=propose(h,AT,(),longest=8.4)
    h.target_history=(*h.target_history,StageTarget('unavailable',AT+timedelta(days=1),AT.date(),
        1000.,500.,'future knowledge'))
    after=propose(h,AT,(),longest=8.4)
    assert before['target']==after['target']
    assert before['continuous_state']==after['continuous_state']
    assert before['smoothed_support']==after['smoothed_support']


def test_repeated_reviews_preserve_baseline_and_do_not_compound_intraday():
    h=matched_harness();first=propose(h,AT,(),longest=8.4);new=adopt(h,first)
    for hour in (1,2,3):
        r=propose(new,AT+timedelta(hours=hour),(),longest=8.4)
        assert r['continuous_state']['planning_volume']==21.4
        assert r['continuous_state']['progression_reference']==first['continuous_state']['progression_reference']
        new=adopt(new,r)


@pytest.mark.parametrize('state',['health_concern','adverse'])
def test_response_veto_does_not_disappear_with_matched_accounting(state):
    h=matched_harness()
    r=propose(h,AT,(),responses=(ResponseReport(AT,state,'athlete'),),longest=8.4)
    assert r['target'].growth_per_7==0
    assert r['target'].volume_per_7==21.4
    assert not r['long_progression']


def test_retained_historical_init_is_not_a_new_timer_or_another_growth_payout():
    h=matched_harness();first=propose(h,AT,(),longest=8.4)
    new=adopt(h,first)
    state=first['continuous_state']
    new.general_progression_state=dict(state,version='continuous_general_v3')
    second=propose(new,AT+timedelta(days=1),(),longest=8.4)
    assert second['continuous_state']['historical_initialization']==state['historical_initialization']
    assert second['continuous_state']['volume']==state['volume']  # No banked migration growth.
    assert second['continuous_state']['progression_reference']==state['progression_reference']


def test_new_runner_and_unattended_gap_cannot_invent_completed_progress():
    h=matched_harness(running()[-2:])
    first=propose(h,AT,(),longest=8.4)
    assert first['gate_status']=='insufficient_observation_history'
    assert first['target'].growth_per_7==0
    h=matched_harness();first=propose(h,AT,(),longest=8.4);new=adopt(h,first)
    late=propose(new,AT+timedelta(days=180),(),longest=8.4)
    assert late['continuous_state']['volume']==first['continuous_state']['volume']
    assert late['target'].growth_per_7==0


@pytest.mark.parametrize('scale',[5.,20.,60.])
def test_flat_training_stops_growth_despite_original_forecast_rate(scale):
    rows=running(scale,weeks=60);ts=targets(scale);v=scale;rates=[]
    for n in range(181):
        at=AT+timedelta(days=n);e=evidence(rows,at=at,ts=ts);p=matched_growth(v,e)
        before=v
        if n:v=advance(v,p,elapsed_days=1,destination=scale*3)
        assert before<=v<=before*1.04**(1/7)+1e-8
        ts=(*ts,StageTarget('flat'+str(n),at,at.date(),v,scale*.4+v-scale,'control',
            growth_per_7=p['growth_per_7'],growth_limit=scale*3))
        rates.append(p['growth_per_7'])
    assert max(rates[-30:])<1e-8
    assert v/scale==pytest.approx(1.113888799562335,rel=1e-6)
    assert v<scale*1.04**3  # Same bound as the pre-V4 flat-training control.


@pytest.mark.parametrize('scale',[5.,20.,60.])
def test_following_rising_doses_is_scale_invariant_and_does_not_stall(scale):
    rows=running(scale);ts=targets(scale);v=scale;values=[]
    for n in range(43):
        at=AT+timedelta(days=n,hours=4);e=evidence(rows,at=at,ts=ts);p=matched_growth(v,e)
        before=v
        if n:v=advance(v,p,elapsed_days=1,destination=scale*3)
        assert before<=v<=before*1.04**(1/7)+1e-8
        ts=(*ts,StageTarget('rising'+str(n),at,at.date(),v,scale*.4+v-scale,'control',
            growth_per_7=p['growth_per_7'],growth_limit=scale*3))
        if n%7 in (0,2,3,5):
            template=rows[-1];share={0:.45,2:.15,3:.15,5:.25}[n%7]
            q=replace(template.completion.distance_meters,low=v*share*1609.344,
                high=v*share*1609.344,estimate=v*share*1609.344)
            start=at.replace(hour=18);end=start+timedelta(hours=1)
            rows.append(replace(template,key='new'+str(n),scheduled_at=start,
                contract=replace(template.contract,issued_at=at,targets=(replace(template.contract.targets[0],quantity=q),)),
                completion=replace(template.completion,key='new'+str(n),start=start,end=end,available_at=end,distance_meters=q)))
        values.append(v)
    assert .09<values[21]/scale-1<=1.04**3-1+1e-8
    assert values[42]>values[21]


def test_full_atomic_publication_and_restart_preserve_both_reference_and_path(tmp_path,monkeypatch):
    from test_smoothed_general_progression import fixture
    from run_analysis.adaptive.adaptive_stage_lifecycle import advance_stage
    from run_analysis.adaptive.new_model_daily_harness import DailyHarness
    from run_analysis.adaptive.live_bridge import runtime_version
    h,inputs,config,at=fixture(tmp_path,monkeypatch)
    h.policy=replace(h.policy,general_progression=MATCHED_VERSION)
    try:
        old=h.stage,h.scales,h.catalog,h.training_policy
        updated,review,event=advance_stage(h,inputs,at)
        assert event and (updated.stage,updated.scales,updated.catalog,updated.training_policy)==old
        assert runtime_version(updated)==214
        assert event['progression_target']['planning_volume_is_conditional']
        assert event['progression_target']['baseline_reference_per_7']==h.stage.weekly_equivalent_miles
        assert event['progression_target']['observed_volume_per_7']==updated.general_progression_state['observed_volume_per_7']
        path=h.sandbox.root/'scenario.json';raw=json.loads(path.read_text())
        raw['search_policy']['general_progression']=None;path.write_text(json.dumps(raw))
        resumed=DailyHarness.resume(h.sandbox.root,config,as_of=at)
        try:
            assert resumed.policy.general_progression==MATCHED_VERSION
            assert resumed.general_progression_state==updated.general_progression_state
            assert resumed.target_history==updated.target_history
            assert resumed.context(at)==updated.context(at)
        finally:resumed.sandbox.close()
    finally:h.sandbox.close()


def test_race_requires_separate_migration_not_fallback(tmp_path,monkeypatch):
    from test_smoothed_general_progression import fixture
    from run_analysis.adaptive.adaptive_stage_lifecycle import advance_stage
    h,inputs,_,at=fixture(tmp_path,monkeypatch)
    h.policy=replace(h.policy,general_progression=MATCHED_VERSION)
    monkeypatch.setattr('run_analysis.adaptive.race_event.preparation_active',lambda *_:True)
    try:
        with pytest.raises(ValueError,match='explicit race-target migration'):advance_stage(h,inputs,at)
    finally:h.sandbox.close()
