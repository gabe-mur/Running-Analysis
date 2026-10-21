"""Dated race wiring controls; not evidence of 21-day optimizer stability."""
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from math import isfinite
import sqlite3

import pytest

from run_analysis.adaptive.race_goal import RaceGoalRevision, queue_request, latest_request, overlay_targets
from run_analysis.adaptive.race_goal import RaceGoalApplication,overlay_applications
from run_analysis.adaptive.race_accounting import event_cost, event_demand, allowed_dose
from run_analysis.adaptive.decision_snapshot import restore, serialize
from run_analysis.adaptive.shared_program_evaluator import ProgramCandidate, evaluate_program
from run_analysis.adaptive.program_objective_trial import TrialSession
from run_analysis.program_delivery import TargetSegment, _target_total
from run_analysis.web.schemas import RacePreparationSettings
from run_analysis.workload_accounting import Quantity
from test_decision_snapshot import freeze


def revision(at, distance='10k', days=90, parent='fixture'):
    return RaceGoalRevision.create(available_at=at,distance=distance,
        race_date=at.date()+timedelta(days=days) if distance!='general_fitness' else None,
        preparation_mode='balanced',parent_key=parent)


def test_goal_journal_is_causal_idempotent_and_revision_checked():
    state=freeze()
    db=sqlite3.connect(':memory:')
    goal=RacePreparationSettings(distance='10k',race_date=state.as_of.date()+timedelta(days=90))
    at=state.as_of+timedelta(seconds=1)
    request=queue_request(db,state=state,draft=goal,expected_revision='adopted:'+state.fingerprint,as_of=at)
    assert latest_request(db,as_of=state.as_of) is None
    assert latest_request(db,as_of=at)==request
    assert queue_request(db,state=state,draft=goal,expected_revision='old retry',as_of=at)==None
    with pytest.raises(ValueError,match='Reload'):
        queue_request(db,state=state,draft=RacePreparationSettings(),expected_revision='old',as_of=at+timedelta(seconds=1))
    clear=queue_request(db,state=state,draft=RacePreparationSettings(),expected_revision=request.key,as_of=at+timedelta(seconds=1))
    assert clear.goal is None and clear.parent_key==request.key
    db.close()


@pytest.mark.parametrize('distance',['5k','10k','half_marathon','marathon'])
def test_dated_taper_does_not_move_with_horizon_or_reprice_pre_request_history(distance):
    at=datetime(2026,10,1,tzinfo=timezone.utc)
    r=revision(at,distance,days=35)
    start=at.date()-timedelta(days=28)
    finish=r.race_date+timedelta(days=r.return_days+5)
    full=(TargetSegment(start,finish,30.),)
    over=overlay_targets(full,r,activation_day=at.date())
    assert _target_total(over,start,at.date())==_target_total(full,start,at.date())
    assert _target_total(over,r.race_date-timedelta(days=1),r.race_date)==pytest.approx(15/7)
    later=overlay_targets((TargetSegment(start,finish+timedelta(days=1),30.),),r,activation_day=at.date())
    for day in (r.taper_start, r.race_date-timedelta(days=1),r.race_date+timedelta(days=1)):
        assert _target_total(over,day,day+timedelta(days=1))==_target_total(later,day,day+timedelta(days=1))
    assert r.purpose_fraction(r.race_date,'long')==0
    assert r.phase(finish)=='complete'


def test_race_snapshot_roundtrip_and_old_snapshot_parity():
    state=freeze()
    original=serialize(state)
    assert 'race_goal' not in original and 'race_demand' not in original
    assert serialize(restore(original,expected_fingerprint=state.fingerprint))==original
    r=revision(state.as_of)
    new=replace(state,context=replace(state.context,race_goal=r),race_goal=r)
    assert restore(serialize(new),expected_fingerprint=new.fingerprint)==new
    bad=replace(new,race_goal=None)
    with pytest.raises(ValueError,match='must match'):
        restore(serialize(bad),expected_fingerprint=bad.fingerprint)


def test_goal_date_edit_and_cancellation_preserve_applied_taper_history():
    at=datetime(2026,10,1,tzinfo=timezone.utc)
    first=revision(at,days=10)
    change_at=at+timedelta(days=7)
    second=revision(change_at,days=90,parent=first.key)
    base=(TargetSegment(at.date(),at.date()+timedelta(days=120),30.),)
    applications=(RaceGoalApplication(first,at,at.date()),)
    before=overlay_applications(base,applications,timezone='UTC')
    edited=overlay_applications(base,applications+(RaceGoalApplication(second,change_at,change_at.date()),),timezone='UTC')
    assert _target_total(before,at.date(),change_at.date())==_target_total(edited,at.date(),change_at.date())
    assert _target_total(edited,change_at.date(),change_at.date()+timedelta(days=1))==pytest.approx(30/7)
    cancel=revision(change_at,'general_fitness',parent=first.key)
    cleared=overlay_applications(base,applications+(RaceGoalApplication(cancel,change_at,change_at.date()),),timezone='UTC')
    assert _target_total(cleared,at.date(),change_at.date())==_target_total(edited,at.date(),change_at.date())
    assert _target_total(cleared,change_at.date(),base[0].end)==pytest.approx(30*(base[0].end-change_at.date()).days/7)


def test_future_application_does_not_reprice_completed_day():
    at=datetime(2026,10,1,tzinfo=timezone.utc)
    goal=revision(at,days=3)
    base=(TargetSegment(at.date(),at.date()+timedelta(days=7),30.),)
    tomorrow=at.date()+timedelta(days=1)
    applied=RaceGoalApplication(goal,at,tomorrow)
    result=overlay_applications(base,(applied,),timezone='UTC')
    assert _target_total(result,at.date(),tomorrow)==pytest.approx(30/7)
    assert _target_total(result,tomorrow,tomorrow+timedelta(days=1))<30/7


def test_event_is_priced_without_training_credit_or_guard_predecessor():
    state=freeze()
    r=revision(state.as_of,days=4)
    state=replace(state,context=replace(state.context,opening_coverage=(('long',0.),('quality',0.))))
    q=(Quantity.point(50.,'fixture'),Quantity.point(60.,'fixture'),Quantity.point(15.,'fixture'))
    demand=event_demand(r,(('easy',4.,q),('long',8.,q),('quality',4.,q)))
    context=replace(state.context,race_goal=r,race_demand=demand)
    empty=ProgramCandidate((),(),'explicit rest fixture')
    result=evaluate_program(context,empty)
    old=evaluate_program(state.context,empty)
    assert result.parts['delivery']==old.parts['delivery']
    assert result.parts['race_event']['training_credit']==0
    unpriced=evaluate_program(replace(context,race_demand=()),empty)
    assert result.total_cost-unpriced.total_cost==pytest.approx(result.parts['race_event']['cost'])
    assert not result.parts['race_event']['participation_cleared']
    assert not any(row.day==r.race_date for row in result.conditional_ledger)
    near=TrialSession('near',r.race_date-timedelta(days=1),4.,'easy',q,source='planned')
    far=replace(near,day=r.race_date-timedelta(days=3))
    assert event_cost(context,(near,))['cost']>event_cost(context,(far,))['cost']


def test_real_evaluator_prices_race_taper_and_continuation_without_fake_completion(tmp_path,monkeypatch):
    from test_adaptive_stage_lifecycle import completed_stage
    from run_analysis.adaptive.service import AdaptivePlannerService
    from run_analysis.adaptive.new_model_daily_harness import DailyHarness
    h,inputs,config,at=completed_stage(tmp_path,monkeypatch,growth_focus='easy_volume',gradual=True,
        single_volume=True,continuation_boundary=True,purpose_continuation=True)
    try:
        h.race_goal=revision(at,days=10)
        service=AdaptivePlannerService(h,inputs)
        result=service.refresh(at)
        h=service.harness
        event=result['event']
        assert event['race_preparation']['phase']=='build'
        assert all(day!=h.race_goal.race_date.isoformat() for day,key in event['program'])
        assert event['selected']['eligible'] and isfinite(event['cost'])
        saved=h.sandbox.root/event['snapshot_file']
        state=restore(saved.read_text(),expected_fingerprint=event['snapshot_fingerprint'])
        assert state.race_goal==h.race_goal and len(state.context.race_demand)==3
        h.sandbox.close()
        reopened=DailyHarness.resume(tmp_path/'state',config,as_of=at)
        try:
            assert reopened.race_goal==state.race_goal
            assert not any(r.key.startswith('race:') for r in reopened.history)
        finally:
            reopened.sandbox.close()
    finally:
        h.sandbox.close()


def test_subsequent_phase_is_reviewed_not_elapsed_growth(tmp_path,monkeypatch):
    from test_adaptive_stage_lifecycle import completed_stage
    from run_analysis.adaptive.adaptive_stage_lifecycle import advance_stage
    from run_analysis.adaptive.stage_acceptance import StageLedger
    h,inputs,_,at=completed_stage(tmp_path,monkeypatch,growth_focus='easy_volume',gradual=True,
        single_volume=True,continuation_boundary=True,purpose_continuation=True)
    try:
        h.race_goal=revision(at,'marathon',days=250)
        original=h.stage
        updated,review,event=advance_stage(h,inputs,at)
        assert event is not None,review
        assert updated.stage.weekly_equivalent_miles==21.
        assert updated.stage.policy.volume_goal_miles==22.3
        assert updated.stage.long_miles<=1.1*max(r.miles for r in h.history)
        assert updated.target_history[-1].ramp_days==7
        assert StageLedger.for_sandbox(h.sandbox).current==updated.stage
        assert event['stage_acceptance']
        assert all(a==b for a,b in zip(h.target_history,updated.target_history))
        assert updated.scales==h.scales
        updated2,review2,event2=advance_stage(updated,inputs,at+timedelta(seconds=1))
        assert updated2.stage==updated.stage and event2 is None
    finally:
        h.sandbox.close()


@pytest.mark.parametrize('signal',['missed','adverse'])
def test_race_growth_holds_for_insufficient_delivery_or_adverse_response(tmp_path,monkeypatch,signal):
    from test_adaptive_stage_lifecycle import completed_stage
    from run_analysis.adaptive.adaptive_stage_lifecycle import advance_stage,review_entries
    from run_analysis.session_decision import DatedResponse
    from run_analysis.recovery_response import ResponseExpectation
    h,inputs,_,at=completed_stage(tmp_path,monkeypatch,growth_focus='easy_volume',gradual=True,
        single_volume=True,continuation_boundary=True,purpose_continuation=True)
    try:
        h.race_goal=revision(at,'marathon',days=250)
        before=h.stage
        if signal=='missed':
            entries=tuple(replace(r,completion=None) for r in review_entries(h,at))
            monkeypatch.setattr('run_analysis.adaptive.adaptive_stage_lifecycle.review_entries',lambda *_:entries)
        else:
            h.responses=(DatedResponse('reported',at-timedelta(days=1),at,
                ResponseExpectation('issued fixture',effort=Quantity(2.,5.)),
                'athlete self-report',effort=9.),)
        updated,review,event=advance_stage(h,inputs,at)
        assert event is None and updated.stage==before,review
        assert review['status']!='conditional_development_candidate'
    finally:h.sandbox.close()


def test_race_application_history_survives_publication_and_cancellation(tmp_path,monkeypatch):
    from test_adaptive_stage_lifecycle import completed_stage
    from run_analysis.adaptive.service import AdaptivePlannerService
    from run_analysis.adaptive.new_model_daily_harness import DailyHarness
    h,inputs,config,at=completed_stage(tmp_path,monkeypatch,growth_focus='easy_volume',gradual=True,
        single_volume=True,continuation_boundary=True,purpose_continuation=True)
    try:
        h.race_goal=revision(at,days=10)
        service=AdaptivePlannerService(h,inputs)
        first=service.refresh(at)['event'];h=service.harness
        state=restore((h.sandbox.root/first['snapshot_file']).read_text(),expected_fingerprint=first['snapshot_fingerprint'])
        assert len(state.race_goal_history)==1 and state.race_goal_history==state.context.race_goal_history
        queued=queue_request(h.sandbox.db,state=state,draft=RacePreparationSettings(),
            expected_revision=h.race_goal.key,as_of=at+timedelta(seconds=1),general_long_goal_miles=17.)
        cleared=service.refresh(at+timedelta(seconds=2))['event'];h=service.harness
        assert h.race_goal==queued and len(h.race_goal_history)==2
        h.sandbox.close()
        restored=DailyHarness.resume(tmp_path/'state',config,as_of=at+timedelta(seconds=2))
        try:
            assert restored.race_goal==queued and restored.race_goal_history==h.race_goal_history
            assert restored.context(at+timedelta(seconds=3)).race_goal_history==h.race_goal_history
        finally:restored.sandbox.close()
    finally:h.sandbox.close()


def test_retained_general_destination_survives_goal_queue_and_cancel():
    state=freeze();db=sqlite3.connect(':memory:');at=state.as_of+timedelta(seconds=1)
    request=queue_request(db,state=state,draft=RacePreparationSettings(distance='marathon',
        race_date=at.date()+timedelta(days=250)),expected_revision='adopted:'+state.fingerprint,
        as_of=at,general_long_goal_miles=17.)
    clear=queue_request(db,state=state,draft=RacePreparationSettings(),expected_revision=request.key,
        as_of=at+timedelta(seconds=1))
    assert request.general_long_goal_miles==clear.general_long_goal_miles==17.
    db.close()


def test_api_rejects_unfeasible_goal_and_applies_with_no_calendar_or_overlay_write(tmp_path):
    from dataclasses import asdict
    import json
    from fastapi.testclient import TestClient
    from run_analysis.web.app import create_app
    from test_adaptive_settings import adopted
    from test_race_preparation import stage
    config,runtime,state=adopted(tmp_path,race_ready=True)
    local=tmp_path/'config.local.yaml'
    local.write_text(local.read_text()+'timezone_default: UTC\n')
    s=stage(key=state.stage_key,issued_at=state.stage_issued_at,
            weekly_equivalent_miles=20.,long_miles=5.)
    with sqlite3.connect(runtime/'sandbox.sqlite') as db:
        event=json.loads(db.execute('SELECT event FROM sandbox_decisions').fetchone()[0])
        value=asdict(s);value['issued_at']=s.issued_at.isoformat()
        event['stage_runtime']['stage']=value
        db.execute('UPDATE sandbox_decisions SET event=?',(json.dumps(event),))
        before=db.execute('SELECT * FROM sandbox_decisions').fetchall()
    # Match the test's frozen chronology, not the wall clock / stale actual guard.
    import run_analysis.adaptive.race_settings as module
    original=module.save_race_request
    def frozen(config,root,draft,**kwargs):
        return original(config,root,draft,**dict(kwargs,as_of=state.as_of+timedelta(seconds=1)))
    from unittest.mock import patch
    with patch.object(module,'save_race_request',frozen),patch.object(module,'RACE_ACTIVATION_READY',True):
        client=TestClient(create_app(tmp_path))
        status=client.get('/api/settings/race-preparation/status').json()
        assert status['status']=='available'
        parent=status['expected_revision']
        invalid=client.post('/api/settings/race-preparation/apply',json=dict(expected_revision=parent,
            goal=dict(distance='marathon',race_date=(state.as_of.date()+timedelta(days=3)).isoformat())))
        assert invalid.status_code==422,invalid.text
        overlay=(tmp_path/'config.local.yaml').read_bytes()
        good=client.post('/api/settings/race-preparation/apply',json=dict(expected_revision=parent,
            goal=dict(distance='10k',race_date=(state.as_of.date()+timedelta(days=180)).isoformat())))
        assert good.status_code==200,good.text
        assert good.json()['pending'] and not good.json()['calendar_changed']
        assert not good.json()['participation_cleared']
        with sqlite3.connect(runtime/'sandbox.sqlite') as db:
            assert before==db.execute('SELECT * FROM sandbox_decisions').fetchall()
        assert (tmp_path/'config.local.yaml').read_bytes()==overlay


def test_live_race_apply_is_gated_until_daily_acceptance(tmp_path,monkeypatch):
    from run_analysis.adaptive.race_settings import save_race_request
    monkeypatch.setattr('run_analysis.adaptive.race_settings.RACE_ACTIVATION_READY',False)
    with pytest.raises(ValueError,match='activation is paused'):
        save_race_request({},tmp_path,RacePreparationSettings(distance='10k',race_date=date(2027,5,1)),
            expected_revision='unused',as_of=datetime(2026,10,6,tzinfo=timezone.utc))


def test_failed_goal_publication_keeps_applied_revision_and_stage(tmp_path,monkeypatch):
    from test_adaptive_stage_lifecycle import completed_stage
    from run_analysis.adaptive.service import AdaptivePlannerService
    from run_analysis.adaptive.new_model_daily_harness import DailyHarness
    from run_analysis.adaptive.continuous_search_adapter import NoEligibleProgram
    h,inputs,config,at=completed_stage(tmp_path,monkeypatch,growth_focus='easy_volume',gradual=True,
        single_volume=True,continuation_boundary=True,purpose_continuation=True)
    try:
        event=h.events[-1]
        state=restore((h.sandbox.root/event['snapshot_file']).read_text(),expected_fingerprint=event['snapshot_fingerprint'])
        queued=queue_request(h.sandbox.db,state=state,draft=RacePreparationSettings(distance='10k',
            race_date=at.date()+timedelta(days=180)),expected_revision='adopted:'+state.fingerprint,as_of=at)
        before=h.stage
        def fail(**kwargs): raise NoEligibleProgram('intentional admission failure')
        monkeypatch.setattr('run_analysis.adaptive.new_model_daily_harness.migrated_search',fail)
        service=AdaptivePlannerService(h,inputs)
        with pytest.raises(NoEligibleProgram):service.refresh(at+timedelta(seconds=1))
        assert service.harness.race_goal is None and service.harness.stage==before
        assert latest_request(h.sandbox.db)==queued
        assert h.events[-1]==event
        h.sandbox.close()
        reopened=DailyHarness.resume(tmp_path/'state',config,as_of=at)
        try:
            assert reopened.race_goal is None and reopened.stage==before
            assert latest_request(reopened.sandbox.db)==queued
        finally:reopened.sandbox.close()
    finally:h.sandbox.close()
