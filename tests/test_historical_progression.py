"""Historical credit must remove a fresh timer without inventing training."""
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace

import pytest

from run_analysis.adaptive.historical_progression import observations,support,VERSION
from run_analysis.adaptive.smoothed_general_progression import HISTORICAL_VERSION
from run_analysis.adaptive.continuous_general_progression import propose
from run_analysis.staged_progression import MILE,ResponseReport,ProgressionPolicy
from run_analysis.stage_target_history import StageTarget
from run_analysis.workload_accounting import Quantity
from test_smoothed_progression import AT,ANCHOR,running,targets
from test_staged_progression import stage


def physical(entries):
    return [SimpleNamespace(key=e.completion.key,start=e.completion.start,end=e.completion.end,
        available_at=e.completion.available_at,miles=e.completion.distance_meters.low/MILE,
        role='long' if e.contract.purpose=='long_endurance' else 'easy',source='observed') for e in entries]


def harness(history,*,volume=20.,issued=AT):
    s=stage(issued_at=issued,weekly_equivalent_miles=volume,
        policy=ProgressionPolicy(14,.05,'easy_volume',volume,'gradual'))
    ts=(StageTarget('historical-reference',AT,ANCHOR.date(),volume,8.,
        'presently adopted reference, not a historical prescription'),)
    return SimpleNamespace(stage=s,planner_preferences=None,general_progression_state=None,
        target_history=ts,decision=lambda at:at.date(),zone=AT.tzinfo,
        history=history,capacity_eligibility={},
        policy=SimpleNamespace(general_progression=HISTORICAL_VERSION,progression_credit='actual_distance_v2'))


def test_unprescribed_original_runs_count_and_receipts_never_double_count():
    rows=running();raw=physical(rows)
    a=observations(raw,(),AT);b=observations(raw,rows,AT)
    assert len(a)==len(b)==len(rows)
    assert sum(r.miles for r in a)==pytest.approx(sum(r.miles for r in b))
    x=support(a,(),targets(),as_of=AT,zone=AT.tzinfo)
    y=support(b,rows,targets(),as_of=AT,zone=AT.tzinfo)
    assert x['actual_per_7']==pytest.approx(y['actual_per_7'])


def test_overshoot_is_actual_credit_even_without_old_prescription():
    rows=running(factor=1.08)
    x=support(observations(physical(rows),rows,AT),rows,targets(),as_of=AT,zone=AT.tzinfo)
    normal=support(observations(physical(running()),(),AT),(),targets(),as_of=AT,zone=AT.tzinfo)
    assert x['actual_per_7']==pytest.approx(normal['actual_per_7']*1.08)


def test_experienced_runner_adopted_today_has_no_new_observation_delay():
    h=harness(physical(running()))
    r=propose(h,AT,(),longest=h.stage.long_miles)
    assert r['gate_status']!='insufficient_observation_history'
    assert r['target'].growth_per_7>0
    init=r['continuous_state']['historical_initialization']
    assert init['ready'] and init['recorded_runs']==80
    assert init['hypothetical_growth_credit']==0
    assert r['continuous_state']['observation_anchor']<h.stage.issued_at.isoformat()
    assert h.general_progression_state is None


def test_existing_higher_actual_level_is_recognized_without_future_growth_credit():
    h=harness(physical(running(30.)),volume=20.)
    r=propose(h,AT,(),longest=13.5)
    a=r['smoothed_support']['actual_per_7']
    assert r['continuous_state']['volume']==pytest.approx(a)
    assert 29.<a<31.
    init=r['continuous_state']['historical_initialization']
    assert init['recognized_increase']==pytest.approx(a-20.)
    assert r['target'].volume_per_7==pytest.approx(a)
    assert not r['smoothed_support']['future_credit']


def test_current_target_is_not_lowered_and_no_growth_is_invented_for_elapsed_weeks():
    h=harness(physical(running(15.)),volume=20.)
    r=propose(h,AT,(),longest=8.4)
    assert r['continuous_state']['volume']==20.
    assert r['target'].growth_per_7==0
    assert r['continuous_state']['historical_initialization']['ready']
    assert r['continuous_state']['historical_initialization']['recognized_increase']==0


def test_historical_long_credit_requires_demonstrated_dose_and_supporting_volume():
    h=harness(physical(running(30.)),volume=20.)
    r=propose(h,AT,(),longest=13.5)
    assert r['long_progression']
    long=r['continuous_state']['long_miles']
    actual=r['smoothed_support']['actual_per_7']
    demonstrated=max(row.miles for row in h.history if row.role=='long'
        and row.start>=AT-timedelta(days=30))
    assert h.stage.long_miles<long<=min(actual*.4,demonstrated,13.5*1.1)
    assert r['candidate'].long_miles==long

    # A lone longer session does not establish adequate supporting volume.
    low=harness(physical(running(15.)),volume=20.)
    held=propose(low,AT,(),longest=13.5)
    assert not held['long_progression']
    assert held['continuous_state']['long_miles']==low.stage.long_miles


def test_initialization_keeps_original_accounting_epoch_and_issued_reference():
    h=harness(physical(running(30.)),volume=20.)
    before=h.target_history
    first=propose(h,AT,(),longest=13.5)
    assert first['continuous_state']['accounting_start']==before[-1].effective_on.isoformat()
    assert h.target_history==before
    assert first['target'].available_at==AT
    assert first['target'].effective_on==AT.date()


def test_new_runner_does_not_get_an_old_observation_clock_from_stage_date():
    recent=physical(running())[-2:]
    h=harness(recent,issued=ANCHOR)
    r=propose(h,AT,(),longest=8.4)
    assert r['gate_status']=='insufficient_observation_history'
    assert r['target'].growth_per_7==0
    assert not r['continuous_state']['historical_initialization']['ready']


def test_health_veto_still_prevents_initialization_or_growth():
    h=harness(physical(running(30.)))
    r=propose(h,AT,(),responses=(ResponseReport(AT,'health_concern','athlete'),),longest=13.5)
    assert r['gate_status']=='health_review_required'
    assert r['continuous_state']['volume']==20.
    assert not r['long_progression']


def test_history_earlier_than_known_targets_does_not_create_backdated_targets():
    rows=physical(running());reference=AT-timedelta(days=30)
    ts=(StageTarget('reference',AT,reference.date(),20.,8.,'known only now'),)
    before=ts
    x=support(observations(rows,(),AT),(),ts,as_of=AT,zone=AT.tzinfo)
    assert x['anchor']<x['reference_anchor']
    assert x['reference_anchor']==reference.isoformat()
    assert ts==before
    assert x['target_per_7']==pytest.approx(20.)


def test_known_old_distance_needs_no_guessed_prescribed_distance():
    rows=running();e=rows[-1]
    target=replace(e.contract.targets[0],quantity=replace(e.contract.targets[0].quantity,estimate=None))
    e=replace(e,contract=replace(e.contract,targets=(target,)))
    records=observations((),(e,),AT)
    assert records[0].miles==pytest.approx(e.completion.distance_meters.low/MILE)


def test_measurement_bounds_own_credit_instead_of_raw_midpoint():
    e=running()[-1]
    bounded=replace(e,completion=replace(e.completion,distance_meters=Quantity(4*MILE,6*MILE,5*MILE,'bounded')))
    raw=physical((e,))
    assert observations(raw,(bounded,),AT)[0].miles==4.


@pytest.mark.parametrize('kind',['duplicate_history','duplicate_receipt','same_physical_different_id','mismatched_distance'])
def test_conflicting_or_duplicate_sources_cannot_buy_more_growth(kind):
    rows=running();raw=physical(rows)
    if kind=='duplicate_history':raw.append(raw[0])
    elif kind=='duplicate_receipt':rows.append(replace(rows[0],key='another-slot'))
    elif kind=='same_physical_different_id':raw.append(SimpleNamespace(**dict(vars(raw[0]),key='different-id')))
    else:raw[0]=SimpleNamespace(**dict(vars(raw[0]),miles=100.))
    with pytest.raises(ValueError):observations(raw,rows,AT)


def test_future_planned_unavailable_and_excluded_history_cannot_buy_credit():
    raw=physical(running());one=raw[-1]
    variants=[SimpleNamespace(**dict(vars(one),key='future',start=AT+timedelta(days=1),
        end=AT+timedelta(days=1,hours=1),available_at=AT+timedelta(days=1,hours=1))),
        SimpleNamespace(**dict(vars(one),key='planned',source='planned')),
        SimpleNamespace(**dict(vars(one),key='late',available_at=AT+timedelta(days=1))),
        SimpleNamespace(**dict(vars(one),key='excluded'))]
    baseline=observations(raw,(),AT)
    assert observations([*raw,*variants],(),AT,excluded_keys={'excluded'})==baseline


def adopt(h,result):
    from copy import copy
    new=copy(h)
    new.general_progression_state=result['continuous_state']
    new.target_history=(*h.target_history,result['target'])
    return new


def test_repeated_initialization_does_not_grant_the_same_growth_again():
    h=harness(physical(running(30.)))
    first=propose(h,AT,(),longest=13.5);new=adopt(h,first)
    for seconds in (1,2,3):
        result=propose(new,AT+timedelta(seconds=seconds),(),longest=13.5)
        assert result['continuous_state']['volume']==first['continuous_state']['volume']
        assert result['continuous_state']['historical_initialization']==first['continuous_state']['historical_initialization']
        new=adopt(new,result)


def test_imported_older_history_removes_wait_immediately_but_does_not_reprice_today():
    rows=physical(running(30.))
    h=harness(rows[-2:])
    early=propose(h,AT,(),longest=13.5);new=adopt(h,early)
    assert not early['continuous_state']['historical_initialization']['ready']
    new.history=rows
    later=propose(new,AT+timedelta(hours=1),(),longest=13.5)
    init=later['continuous_state']['historical_initialization']
    assert init['ready'] and later['gate_status']!='insufficient_observation_history'
    assert init['recognized_volume']>29.
    assert not init['baseline_applied']  # Today's earlier published target stays intact.
    assert later['continuous_state']['volume']==early['continuous_state']['volume']
    new=adopt(new,later)
    following=propose(new,AT+timedelta(days=1),(),longest=13.5)
    assert following['continuous_state']['historical_initialization']['baseline_applied']
    assert following['continuous_state']['volume']==init['recognized_volume']


def test_new_historical_support_has_separate_runtime_and_strict_old_defaults():
    from run_analysis.adaptive.live_bridge import runtime_version
    from run_analysis.adaptive.new_model_daily_harness import HarnessPolicy
    policy=HarnessPolicy(delivery_contract='sustained_v1',soft_horizon_taper=True,
        continuation_boundary=True,progression_admission='observed_rolling_v2',
        progression_credit='actual_distance_v2',general_progression=HISTORICAL_VERSION)
    assert runtime_version(SimpleNamespace(policy=policy))==213
    assert HarnessPolicy().general_progression is None
    with pytest.raises(ValueError,match='Continuous progression'):
        HarnessPolicy(delivery_contract='sustained_v1',general_progression=HISTORICAL_VERSION)


def test_historical_initialization_commits_and_survives_stale_export(tmp_path,monkeypatch):
    import json
    from test_smoothed_general_progression import fixture
    from run_analysis.adaptive.adaptive_stage_lifecycle import advance_stage
    from run_analysis.adaptive.new_model_daily_harness import DailyHarness
    h,inputs,config,at=fixture(tmp_path,monkeypatch)
    h.policy=replace(h.policy,general_progression=HISTORICAL_VERSION)
    try:
        old=h.stage,h.scales,h.catalog,h.training_policy
        updated,result,event=advance_stage(h,inputs,at)
        assert event and (updated.stage,updated.scales,updated.catalog,updated.training_policy)==old
        init=updated.general_progression_state['historical_initialization']
        assert init['ready']
        path=h.sandbox.root/'scenario.json'
        value=json.loads(path.read_text());value['search_policy']['general_progression']=None
        path.write_text(json.dumps(value))
        resumed=DailyHarness.resume(h.sandbox.root,config,as_of=at)
        try:
            assert resumed.policy.general_progression==HISTORICAL_VERSION
            assert resumed.general_progression_state==updated.general_progression_state
            assert resumed.target_history==updated.target_history
            assert resumed.context(at)==updated.context(at)
        finally:resumed.sandbox.close()
    finally:h.sandbox.close()
