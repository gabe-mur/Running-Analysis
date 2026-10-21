"""Execution/planner attribution controls, not calendar or physiology acceptance."""
from dataclasses import asdict,replace
from datetime import timedelta,timezone
from types import SimpleNamespace

import pytest

from run_analysis.adaptive.prescription_progression import fulfillment,published_funding,growth
from run_analysis.adaptive.smoothed_progression import advance
from run_analysis.stage_target_history import StageTarget
from run_analysis.staged_progression import MILE
from run_analysis.workload_accounting import Quantity
from test_smoothed_progression import AT
from test_staged_progression import entry,stage

ZONE=timezone.utc


def opportunity(actual=4.,*,nominal=4.,low=None,when=AT,key='one'):
    e=entry(stage(issued_at=when-timedelta(days=1)),actual,key)
    q=Quantity.point(nominal*MILE,'issued') if low is None else Quantity(low*MILE,nominal*MILE,nominal*MILE,'issued band')
    e=replace(e,scheduled_at=when,contract=replace(e.contract,targets=(replace(e.contract.targets[0],quantity=q),)),
        completion=replace(e.completion,start=when,end=when+timedelta(minutes=40),
            available_at=when+timedelta(minutes=40),moving_seconds=Quantity.point(2400,'actual')))
    return e


def paired(rows,*,at=AT+timedelta(hours=1),records=()):
    return fulfillment(rows,as_of=at,epoch=AT,zone=ZONE,records=records)


def proposal(execution,*,ratio=.95,volume=20.,funded=20.):
    return growth(volume,dict(actual_per_7=20.,delivery_ratio=ratio,mean_age_days=14),execution,
        dict(funded_level=funded,status='known'))


def test_full_adherence_is_independent_of_the_old_continuous_target_gap():
    e=paired((opportunity(),))
    for ratio in (.8,.95,.97,1.,1.1):
        p=proposal(e,ratio=ratio)
        assert p['growth_per_7']==.04 and p['rate_status']=='fulfilled'
        assert p['physical_credit_added']==0


@pytest.mark.parametrize('fraction,expected',[(1.,.04),(.99,.0296),(.98,.0192),(.97,.0088),(.95,0.),(0.,0.)])
def test_actual_underexecution_reduces_growth_immediately_without_a_deadband(fraction,expected):
    e=paired((opportunity(4*fraction),))
    assert proposal(e)['growth_per_7']==pytest.approx(expected)


def test_internal_comparison_margin_is_not_a_prescribed_range():
    e=paired((opportunity(3.8,low=3.8),))
    assert e['weighted_actual']<e['weighted_nominal']
    assert e['distance_basis']=='issued_nominal_v1'
    assert e['fulfillment_ratio']==pytest.approx(.95)
    assert proposal(e)['growth_per_7']==0
    for low in (.1,3.,3.8,4.):
        changed=paired((opportunity(3.8,low=low),))
        assert changed['fulfillment_ratio']==e['fulfillment_ratio']
        assert changed['weighted_actual']==e['weighted_actual']
        assert changed['weighted_required']==e['weighted_required']
    assert proposal(paired((opportunity(4.,low=3.8),)))['growth_per_7']==.04


def test_experimental_minimum_basis_cannot_silently_resume_as_approved_policy():
    with pytest.raises(ValueError,match='Issued target distance required'):
        fulfillment((opportunity(),),as_of=AT+timedelta(hours=1),epoch=AT,
            zone=ZONE,distance_basis='issued_minimum_v1')


def test_same_day_uncompleted_run_is_not_a_miss_but_expired_day_is():
    missed=replace(opportunity(),completion=None)
    assert paired((missed,))['fulfillment_ratio'] is None
    e=paired((missed,),at=AT+timedelta(days=1,hours=4))
    assert e['fulfillment_ratio']==0 and proposal(e)['growth_per_7']==0


def test_overshoot_offsets_undershoot_as_net_actual_distance_not_positive_only_tokens():
    e=paired((opportunity(5.,key='a'),opportunity(3.,key='b')))
    assert e['fulfillment_ratio']==1 and proposal(e)['growth_per_7']==.04
    e=paired((opportunity(5.,key='a'),opportunity(2.,key='b')))
    assert e['fulfillment_ratio']==pytest.approx(7/8)
    assert proposal(e)['growth_per_7']==0


def test_independent_extra_physical_run_is_counted_once_and_can_offset_a_miss():
    e=opportunity(4.)
    raw=SimpleNamespace(key=e.completion.key,start=e.completion.start,end=e.completion.end,
        available_at=e.completion.available_at,miles=4.)
    missed=replace(opportunity(key='b'),completion=None)
    extra=SimpleNamespace(**dict(vars(raw),key='extra',miles=4.))
    result=paired((e,missed),at=AT+timedelta(days=1,hours=4),records=(raw,extra))
    assert result['fulfillment_ratio']==pytest.approx(1.)
    assert result['extra_physical_keys']==['extra']


def test_unknown_distance_is_not_zero_or_full_fulfillment():
    e=opportunity();e=replace(e,completion=replace(e.completion,distance_meters=None))
    result=paired((e,));p=proposal(result)
    assert result['unresolved']==['one'] and p['growth_per_7']==0
    assert p['rate_status']=='execution_evidence_unresolved'


def test_bounded_distance_does_not_falsely_label_uncertainty_as_an_undershoot():
    e=opportunity();e=replace(e,completion=replace(e.completion,
        distance_meters=Quantity(3*MILE,5*MILE,4*MILE,'uncertain')))
    p=proposal(paired((e,)))
    assert p['growth_per_7']==0
    assert p['rate_status']=='execution_evidence_unresolved'


def test_late_upload_replaces_the_omission_without_duplicate_credit():
    e=opportunity();late=replace(e,completion=replace(e.completion,available_at=AT+timedelta(days=2)))
    missed=paired((late,),at=AT+timedelta(days=1,hours=4))
    known=paired((late,),at=AT+timedelta(days=2,hours=4))
    assert proposal(missed)['growth_per_7']==0
    assert proposal(known)['growth_per_7']==.04 and len(known['included'])==1


def test_fixed_policy_adoption_is_not_a_sliding_window_or_a_new_wait():
    old=opportunity(0.,when=AT-timedelta(days=1),key='old')
    new=opportunity()
    assert paired((old,new))==paired((new,))
    assert proposal(paired((new,)))['growth_per_7']==.04


def test_future_evidence_and_duplicate_identity_do_not_buy_growth():
    e=opportunity();future=replace(e,scheduled_at=AT+timedelta(hours=3),
        contract=replace(e.contract,issued_at=AT+timedelta(hours=2)))
    assert paired((future,))['fulfillment_ratio'] is None
    with pytest.raises(ValueError,match='Duplicate opportunity'):paired((e,e))
    with pytest.raises(ValueError,match='Completion counted twice'):paired((e,replace(e,key='other')))


def event(level=20.,miles=None,rate=.04):
    path=StageTarget('funding',AT,AT.date(),level,level*.5,'issued',growth_per_7=rate,growth_limit=level*3)
    expected=sum(path.rate(AT.date()+timedelta(days=i),'volume_per_7')/7 for i in range(21))
    raw=asdict(path);raw.update(available_at=AT.isoformat(),effective_on=AT.date().isoformat())
    return dict(index=0,as_of=AT.isoformat(),decision=AT.date().isoformat(),
        hard_end=(AT+timedelta(days=21)).date().isoformat(),volume=expected if miles is None else miles,
        progression_target={'planning_volume_per_7':level},stage_runtime={'target_history':[raw]})


def test_planner_funding_is_separate_from_athlete_fulfillment():
    good=published_funding(event(),as_of=AT,zone=ZONE)
    short=published_funding(event(miles=30.),as_of=AT,zone=ZONE)
    assert good['funding_ratio']==pytest.approx(1.) and good['funded_level']==pytest.approx(20.)
    assert short['funded_level']<20
    p=growth(20.,dict(actual_per_7=20.,delivery_ratio=1.,mean_age_days=14),paired((opportunity(),)),short)
    assert p['growth_per_7']==0 and p['rate_status']=='planner_funding_exhausted'


@pytest.mark.parametrize('scale',[5.,20.,60.])
def test_flat_fully_followed_plan_cannot_ratchet_the_target_indefinitely(scale):
    e=paired((opportunity(scale),));v=scale
    for n in range(181):
        at=AT+timedelta(days=n)
        funding=published_funding(event(level=v,miles=scale*3),as_of=at,zone=ZONE)
        p=proposal(e,volume=v,funded=funding['funded_level'])
        v=advance(v,p,elapsed_days=bool(n)*1,destination=scale*3)
    assert v<=scale*1.0816+1e-8
    assert p['growth_per_7']==0


@pytest.mark.parametrize('scale',[5.,20.,60.])
def test_new_rising_plan_fully_followed_delivers_four_percent(scale):
    e=paired((opportunity(scale),));v=scale
    for n in range(43):
        funding=published_funding(event(level=v),as_of=AT+timedelta(days=n),zone=ZONE)
        p=proposal(e,volume=v,funded=funding['funded_level'])
        v=advance(v,p,elapsed_days=bool(n)*1,destination=scale*3)
    assert v==pytest.approx(scale*1.04**6)


def test_health_veto_still_overrides_full_execution_and_funding():
    e=paired((opportunity(),));p=growth(20.,dict(actual_per_7=20.,delivery_ratio=1.,mean_age_days=14),e,
        dict(status='known',funded_level=20.),permitted=False)
    assert p['growth_per_7']==0 and p['rate_status']=='response_veto'


@pytest.mark.parametrize('scale',[5.,20.,60.])
@pytest.mark.parametrize('mode',['follow','undershoot','flat_actual','flat_plan'])
def test_causal_daily_courses_separate_adherence_from_planner_funding(scale,mode):
    rows=[];v=scale;values=[];rates=[]
    for n in range(181):
        at=AT+timedelta(days=n,hours=8)
        e=fulfillment(rows,as_of=at,epoch=AT,zone=ZONE)
        funding=published_funding(event(level=v,miles=scale*3 if mode=='flat_plan' else None),as_of=at,zone=ZONE)
        p=growth(v,dict(actual_per_7=scale,delivery_ratio=.95,mean_age_days=14),e,funding)
        if n:v=advance(v,p,elapsed_days=1,destination=scale*3)
        if n%7 in (0,2,3,5):
            share={0:.45,2:.15,3:.15,5:.25}[n%7]
            planned=(scale if mode=='flat_plan' else v)*share
            actual=scale*share if mode=='flat_actual' else planned*(.99 if mode=='undershoot' else 1.)
            rows.append(opportunity(actual,nominal=planned,when=at.replace(hour=18),key=str(n)))
        values.append(v);rates.append(p['growth_per_7'])
    if mode=='follow':assert values[42]==pytest.approx(scale*1.04**6)
    elif mode=='undershoot':
        assert values[42]==pytest.approx(scale*1.0296**6)
        assert rates[-1]==pytest.approx(.0296)
    elif mode=='flat_plan':
        assert v<=scale*1.0816+1e-8
        assert max(rates[-30:])==0
    else:
        assert v<scale*1.1
        assert max(rates[-30:])==0


def test_refined_policy_commits_and_resumes_with_distinct_execution_and_funding_receipts(tmp_path,monkeypatch):
    from test_smoothed_general_progression import fixture
    from run_analysis.adaptive.smoothed_general_progression import PRESCRIPTION_VERSION
    from run_analysis.adaptive.adaptive_stage_lifecycle import advance_stage
    from run_analysis.adaptive.new_model_daily_harness import DailyHarness
    from run_analysis.adaptive.live_bridge import runtime_version
    h,inputs,config,at=fixture(tmp_path,monkeypatch)
    h.policy=replace(h.policy,general_progression=PRESCRIPTION_VERSION)
    try:
        before=h.stage,h.scales,h.catalog,h.training_policy
        updated,review,e=advance_stage(h,inputs,at)
        assert e and (updated.stage,updated.scales,updated.catalog,updated.training_policy)==before
        assert runtime_version(updated)==216
        assert e['progression_target']['rate_contract']=='prescription_fulfillment_v1'
        assert e['progression_target']['execution_fulfillment']['epoch']==at.isoformat()
        assert e['progression_target']['execution_fulfillment']['distance_basis']=='issued_nominal_v1'
        assert updated.general_progression_state['execution_distance_basis']=='issued_nominal_v1'
        assert e['progression_target']['execution_fulfillment']['fulfillment_ratio'] is None
        assert e['progression_target']['planner_funding']['status']=='known'
        resumed=DailyHarness.resume(h.sandbox.root,config,as_of=at)
        try:
            assert resumed.policy.general_progression==PRESCRIPTION_VERSION
            assert resumed.general_progression_state==updated.general_progression_state
            assert resumed.context(at)==updated.context(at)
            from run_analysis.adaptive.continuous_general_progression import propose
            from run_analysis.adaptive.adaptive_stage_lifecycle import review_entries
            again=propose(resumed,at+timedelta(hours=1),
                review_entries(resumed,at+timedelta(hours=1),original_schedule=True),longest=resumed.stage.long_miles)
            assert again['continuous_state']['execution_epoch']==at.isoformat()
            assert again['continuous_state']['execution_distance_basis']=='issued_nominal_v1'
            assert again['continuous_state']['volume']==updated.general_progression_state['volume']
        finally:resumed.sandbox.close()
    finally:h.sandbox.close()


def test_new_policy_cannot_fall_back_to_general_rules_during_a_race(tmp_path,monkeypatch):
    from test_smoothed_general_progression import fixture
    from run_analysis.adaptive.smoothed_general_progression import PRESCRIPTION_VERSION
    from run_analysis.adaptive.adaptive_stage_lifecycle import advance_stage
    h,inputs,_,at=fixture(tmp_path,monkeypatch);h.policy=replace(h.policy,general_progression=PRESCRIPTION_VERSION)
    monkeypatch.setattr('run_analysis.adaptive.race_event.preparation_active',lambda *_:True)
    try:
        with pytest.raises(ValueError,match='explicit race-target migration'):advance_stage(h,inputs,at)
    finally:h.sandbox.close()


def test_future_published_plan_cannot_authorize_current_growth():
    with pytest.raises(ValueError,match='Future plan'):
        published_funding(event(),as_of=AT-timedelta(seconds=1),zone=ZONE)


def test_paired_schedule_uses_original_intent_not_retimed_completion(tmp_path,monkeypatch):
    from test_smoothed_general_progression import fixture
    from run_analysis.adaptive.adaptive_stage_lifecycle import review_entries
    h,_,_,at=fixture(tmp_path,monkeypatch)
    try:
        old=review_entries(h,at);new=review_entries(h,at,original_schedule=True)
        assert [(e.key,e.completion) for e in old]==[(e.key,e.completion) for e in new]
        for e in new:
            if not e.key.startswith('slot:'):continue
            raw=h.sandbox.record(e.contract.source).payload['intent']
            from datetime import datetime
            assert e.scheduled_at==max(e.contract.issued_at,datetime.fromisoformat(raw['suggested_start']))
    finally:h.sandbox.close()
