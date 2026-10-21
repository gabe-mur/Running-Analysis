"""V5 nominal-rate contract; old V4 behavior remains separately reproducible."""
from dataclasses import replace
from datetime import timedelta

import pytest

from run_analysis.adaptive.smoothed_progression import nominal_growth,matched_growth,advance
from run_analysis.adaptive.smoothed_general_progression import NOMINAL_VERSION
from run_analysis.adaptive.continuous_general_progression import propose
from run_analysis.stage_target_history import StageTarget
from run_analysis.staged_progression import ResponseReport
from test_smoothed_progression import running,targets,evidence,AT
from test_historical_progression import adopt
from test_matched_progression import matched_harness


def nominal_harness():
    h=matched_harness();h.policy.general_progression=NOMINAL_VERSION
    return h


@pytest.mark.parametrize('ratio',[.9492741635942683,.970465488167793,.9825456424373739,.99,1.,1.08])
def test_historical_gap_does_not_multiply_nominal_growth(ratio):
    e=dict(actual_per_7=20*ratio,delivery_ratio=ratio,mean_age_days=14)
    old=matched_growth(20.,e);new=nominal_growth(20.,e)
    assert new['growth_per_7']==.04
    assert new['supported_volume_ceiling']==old['supported_volume_ceiling']
    assert new['long_support_fraction']==old['support_fraction']
    assert new['physical_credit_added']==0
    assert new['rate_status']=='nominal'
    if ratio<1:assert old['growth_per_7']<new['growth_per_7']


@pytest.mark.parametrize('ratio',[0.,.5,.9,1/1.0816])
def test_real_absence_or_exhausted_headroom_holds_instead_of_unsupported_growth(ratio):
    p=nominal_growth(20.,dict(actual_per_7=20*ratio,delivery_ratio=ratio,mean_age_days=14))
    assert p['growth_per_7']==0
    assert advance(20.,p,elapsed_days=10000,destination=60.)==20.


def test_headroom_remains_a_real_clip_not_just_a_display_label():
    p=nominal_growth(20.,dict(actual_per_7=18.6,delivery_ratio=.93,mean_age_days=14))
    assert p['growth_per_7']==.04
    assert advance(20.,p,elapsed_days=7,destination=60.)==pytest.approx(p['supported_volume_ceiling'])
    assert advance(20.,p,elapsed_days=7,destination=60.)<20*1.04
    assert advance(20.,p,elapsed_days=0,destination=60.)==20.


@pytest.mark.parametrize('state',['health_concern','adverse'])
def test_health_response_is_a_hard_veto(state):
    h=nominal_harness()
    r=propose(h,AT,(),responses=(ResponseReport(AT,state,'athlete'),),longest=8.4)
    assert r['target'].growth_per_7==0
    assert r['continuous_state']['volume']==21.4
    assert not r['long_progression']


def test_absent_actual_cannot_use_inconsistent_ratio_or_optimistic_forecast():
    p=nominal_growth(20.,dict(actual_per_7=0.,delivery_ratio=1.,mean_age_days=14))
    assert p['growth_per_7']==0
    assert nominal_growth(20.,dict(actual_per_7=100.,delivery_ratio=5.,mean_age_days=14),permitted=False)['growth_per_7']==0


def test_explicit_migration_preserves_level_reference_and_history_without_banked_growth():
    h=matched_harness();old=propose(h,AT,(),longest=8.4);h=adopt(h,old)
    h.policy.general_progression=NOMINAL_VERSION
    new=propose(h,AT+timedelta(days=1),(),longest=8.4)
    assert new['continuous_state']['version']==NOMINAL_VERSION
    assert new['continuous_state']['volume']==old['continuous_state']['volume']
    assert new['continuous_state']['progression_reference']==old['continuous_state']['progression_reference']
    assert new['continuous_state']['historical_initialization']==old['continuous_state']['historical_initialization']
    assert new['target'].growth_per_7==.04
    assert not new['long_progression']


def test_repeat_refresh_and_unattended_gap_do_not_bank_growth():
    h=nominal_harness();first=propose(h,AT,(),longest=8.4);h=adopt(h,first)
    for hour in (1,2,3):
        r=propose(h,AT+timedelta(hours=hour),(),longest=8.4)
        assert r['continuous_state']['volume']==21.4
        h=adopt(h,r)
    late=propose(h,AT+timedelta(days=180),(),longest=8.4)
    assert late['continuous_state']['volume']==21.4
    assert late['target'].growth_per_7==0


def test_nominal_volume_does_not_accelerate_the_separate_long_budget():
    from test_historical_progression import harness,physical
    from test_smoothed_progression import ANCHOR
    h=harness(physical(running(24.3)));h.policy.general_progression=NOMINAL_VERSION
    h.target_history=(StageTarget('known-reference',AT,ANCHOR.date(),25.,12.,'test reference'),)
    first=propose(h,AT,(),longest=11.)
    first['continuous_state'].update(long_budget=h.stage.long_miles,long_miles=h.stage.long_miles)
    h=adopt(h,first)
    r=propose(h,AT+timedelta(days=1),(),longest=11.)
    p=r['growth_proposal'];fraction=p['long_support_fraction']
    assert r['target'].growth_per_7==.04 and 0<fraction<1
    assert r['continuous_state']['long_budget']==pytest.approx(
        h.stage.long_miles*1.05**(fraction/14))


def test_nominal_policy_is_explicit_and_causal():
    from run_analysis.adaptive.new_model_daily_harness import HarnessPolicy
    assert HarnessPolicy().general_progression is None
    with pytest.raises(ValueError,match='Continuous progression'):
        HarnessPolicy(delivery_contract='sustained_v1',general_progression=NOMINAL_VERSION)
    h=nominal_harness();before=propose(h,AT,(),longest=8.4)
    h.target_history=(*h.target_history,StageTarget('future',AT+timedelta(days=1),AT.date(),
        1000.,500.,'not yet available'))
    after=propose(h,AT,(),longest=8.4)
    assert before['continuous_state']==after['continuous_state']
    assert before['smoothed_support']==after['smoothed_support']


@pytest.mark.parametrize('scale',[5.,20.,60.])
def test_flat_training_is_bounded_and_stops_but_the_new_headroom_tradeoff_is_explicit(scale):
    rows=running(scale,weeks=60);ts=targets(scale);v=scale;rates=[]
    for n in range(181):
        at=AT+timedelta(days=n);e=evidence(rows,at=at,ts=ts);p=nominal_growth(v,e)
        if n:v=advance(v,p,elapsed_days=1,destination=scale*3)
        ts=(*ts,StageTarget('flat'+str(n),at,at.date(),v,scale*.4+v-scale,'control',
            growth_per_7=p['growth_per_7'],growth_limit=scale*3))
        rates.append(p['growth_per_7'])
    assert max(rates[-30:])==0
    assert v/scale==pytest.approx(1.1633222189521129,rel=1e-6)
    assert v/scale<1.04**4
    # Old V4's tighter 11.39% terminal headroom is NOT claimed for this policy.


@pytest.mark.parametrize('scale',[5.,20.,60.])
def test_followed_rising_training_delivers_nominal_growth_not_just_a_nonzero_rate(scale):
    rows=running(scale);ts=targets(scale);v=scale;values=[]
    for n in range(43):
        at=AT+timedelta(days=n,hours=4);p=nominal_growth(v,evidence(rows,at=at,ts=ts))
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
    assert values[21]==pytest.approx(scale*1.04**3)
    assert values[42]==pytest.approx(scale*1.04**6)


def test_nominal_full_publication_restart_and_receipt(tmp_path,monkeypatch):
    from test_smoothed_general_progression import fixture
    from run_analysis.adaptive.adaptive_stage_lifecycle import advance_stage
    from run_analysis.adaptive.new_model_daily_harness import DailyHarness
    from run_analysis.adaptive.live_bridge import runtime_version
    h,inputs,config,at=fixture(tmp_path,monkeypatch)
    h.policy=replace(h.policy,general_progression=NOMINAL_VERSION)
    try:
        old=h.stage,h.scales,h.catalog,h.training_policy
        updated,review,event=advance_stage(h,inputs,at)
        assert event and (updated.stage,updated.scales,updated.catalog,updated.training_policy)==old
        assert runtime_version(updated)==215
        assert event['progression_target']['rate_contract']=='nominal_4pct_matched_headroom_v1'
        assert event['progression_target']['growth_per_7']==.04
        resumed=DailyHarness.resume(h.sandbox.root,config,as_of=at)
        try:
            assert resumed.policy.general_progression==NOMINAL_VERSION
            assert resumed.general_progression_state==updated.general_progression_state
            assert resumed.target_history==updated.target_history
            assert resumed.context(at)==updated.context(at)
        finally:resumed.sandbox.close()
    finally:h.sandbox.close()


def test_nominal_race_still_requires_explicit_migration(tmp_path,monkeypatch):
    from test_smoothed_general_progression import fixture
    from run_analysis.adaptive.adaptive_stage_lifecycle import advance_stage
    h,inputs,_,at=fixture(tmp_path,monkeypatch)
    h.policy=replace(h.policy,general_progression=NOMINAL_VERSION)
    monkeypatch.setattr('run_analysis.adaptive.race_event.preparation_active',lambda *_:True)
    try:
        with pytest.raises(ValueError,match='explicit race-target migration'):advance_stage(h,inputs,at)
    finally:h.sandbox.close()
