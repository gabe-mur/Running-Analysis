"""Event accounting/causality controls; not an optimizer stability claim."""
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace
import sqlite3

import numpy as np
import pytest

from test_decision_snapshot import freeze
from test_r3_decision_controls import AT, REF, row, context
from test_race_goal_integration import revision
from run_analysis.adaptive.race_event import (
    record_resolution, latest_resolution, training_fraction, purpose_fraction)
from run_analysis.adaptive.race_goal import RaceGoalApplication, overlay_applications
from run_analysis.adaptive.new_model_daily_harness import HarnessPolicy
from run_analysis.adaptive.race_accounting import allowed_dose
from run_analysis.adaptive.program_objective_trial import credits, TrialSession
from run_analysis.adaptive.decision_snapshot import serialize, restore
from run_analysis.program_delivery import TargetSegment, _target_total


def established_profile(distance):
    # Declared synthetic established exposure, not fitted athlete readiness.
    return dict(zip(('weekly_volume','long_miles','ordinary_miles','easy_days'),{
        '5k':(10.,3.1,3.,(4,)),
        '10k':(20.,8.4,4.4,(4,)),
        'half_marathon':(25.,10.,5.6,(4,6)),
        'marathon':(40.,20.,5.4,(1,4,6)),
    }[distance]))


def report(db, race, *, status='did_not_start', physical=None, at=None, parent=None):
    return record_resolution(db, revision=race, history=() if physical is None else (physical,),
        timezone_name='UTC', status=status, physical_key=physical.key if physical else None,
        expected_revision=parent or race.key, as_of=at or AT+timedelta(days=1))


def test_journal_is_causal_revision_checked_and_requires_real_event_day_activity():
    db=sqlite3.connect(':memory:')
    race=revision(AT-timedelta(days=5),days=5)
    physical=row(0,6.2,'quality',source='observed')
    resolved=report(db,race,status='recorded',physical=physical)
    assert latest_resolution(db,race.key,as_of=AT) is None
    assert latest_resolution(db,race.key)==resolved
    assert report(db,race,status='recorded',physical=physical,parent=resolved.key,
        at=AT+timedelta(days=2))==resolved
    with pytest.raises(ValueError,match='changed'):
        report(db,race)
    with pytest.raises(ValueError,match='original activity'):
        report(db,race,status='recorded',physical=row(-1,6.,source='observed'),parent=resolved.key,
            at=AT+timedelta(days=2))
    corrected=report(db,race,parent=resolved.key,at=AT+timedelta(days=2))
    assert corrected.parent_key==resolved.key
    assert latest_resolution(db,race.key,as_of=AT+timedelta(days=1))==resolved


def test_journal_rejects_a_validly_hashed_but_disconnected_correction():
    import json
    from run_analysis.adaptive.race_event import RaceEventResolution
    db=sqlite3.connect(':memory:');race=revision(AT-timedelta(days=5),days=5)
    first=report(db,race)
    broken=RaceEventResolution.create(race_key=race.key,status='did_not_start',
        available_at=first.available_at+timedelta(days=1),parent_key=race.key,
        effective_on=first.effective_on+timedelta(days=1))
    db.execute('INSERT INTO adaptive_race_events VALUES (?,?)',(broken.key,json.dumps(broken.payload)))
    with pytest.raises(ValueError,match='chain'):
        latest_resolution(db,race.key)


@pytest.mark.parametrize('distance',['5k','10k','half_marathon','marathon'])
def test_nonparticipation_is_prospective_and_releases_only_assumed_recovery(distance):
    db=sqlite3.connect(':memory:')
    race=revision(AT-timedelta(days=10),distance,days=10)
    outcome=report(db,race,at=AT+timedelta(days=2))
    first=AT.date()+timedelta(days=1)
    after=AT.date()+timedelta(days=2)
    assert training_fraction(race,outcome,first)==race.training_fraction(first)
    assert training_fraction(race,outcome,after)==1.
    assert purpose_fraction(race,outcome,after,'long')==1.
    dose=SimpleNamespace(role='quality',key='quality:full',miles=4.)
    stage=SimpleNamespace(long_miles=8.)
    assert not allowed_dose(race,first,dose,stage,outcome)
    assert allowed_dose(race,after,dose,stage,outcome)
    targets=(TargetSegment(AT.date()-timedelta(days=2),after+timedelta(days=2),30.),)
    applications=(RaceGoalApplication(race,race.available_at,race.available_at.date()),)
    old=overlay_applications(targets,applications,timezone='UTC')
    new=overlay_applications(targets,applications,timezone='UTC',resolution=outcome)
    assert _target_total(old,targets[0].start,after)==_target_total(new,targets[0].start,after)
    assert _target_total(new,after,after+timedelta(days=1))==pytest.approx(30/7)


def test_recorded_event_retains_demand_but_no_training_credit_and_roundtrips():
    db=sqlite3.connect(':memory:')
    race=revision(AT-timedelta(days=5),days=5)
    physical=row(0,6.2,'quality',source='observed')
    outcome=report(db,race,status='recorded',physical=physical)
    at=AT+timedelta(days=1)
    ctx=replace(context(),race_goal=race,race_event=outcome,decision=at.date())
    # The real HR-derived quantities stay untouched; the event is not a second
    # forecast or ordinary quality credit. It is measured distance exposure.
    from run_analysis.adaptive.continuous_program_adapter import evaluate_continuous
    evaluated=evaluate_continuous(ctx,(physical,),(),reference=REF,as_of=at,timezone='UTC',purpose_fraction=.5)
    actual=evaluated['evaluation'].conditional_ledger[0]
    assert actual.role=='race' and actual.demand==physical.demand
    assert np.array_equal(credits(actual,ctx.policy),np.zeros(4))
    assert evaluated['evaluation'].parts['delivery']['volume']==0.
    state=freeze(as_of=at,context=ctx,history=(physical,))
    assert not next(r for r in state.athlete.sessions if r.key==physical.key).capacity_eligible
    assert restore(serialize(state),expected_fingerprint=state.fingerprint)==state


def test_general_snapshot_has_no_new_absent_field():
    assert 'race_event' not in serialize(freeze())


def test_preparation_status_discloses_nonstart_only_after_causal_report():
    from zoneinfo import ZoneInfo
    from test_race_preparation import stage
    from run_analysis.adaptive.race_goal import preparation_status
    db=sqlite3.connect(':memory:');race=revision(AT-timedelta(days=5),days=5)
    outcome=report(db,race,at=AT+timedelta(days=2))
    h=SimpleNamespace(race_goal=race,race_event=outcome,race_event_history=(),
        stage=stage(),history=(),zone=ZoneInfo('UTC'))
    before=preparation_status(h,AT.date()+timedelta(days=1),as_of=AT+timedelta(days=1))
    after=preparation_status(h,outcome.effective_on,as_of=outcome.available_at)
    assert before['phase']=='post_race' and before['outcome']=='unknown'
    assert after['phase']=='did_not_start' and after['outcome']=='did_not_start'
    assert not after['race_participation_cleared']
    db.close()


def test_general_signature_still_notices_historical_event_reconciliation():
    from zoneinfo import ZoneInfo
    from run_analysis.adaptive.live_bridge import input_signature
    from run_analysis.adaptive.race_goal import RaceGoalRevision
    db=sqlite3.connect(':memory:');race=revision(AT-timedelta(days=5),days=5)
    general=RaceGoalRevision.create(available_at=AT+timedelta(hours=1),distance='general_fitness',
        race_date=None,preparation_mode='balanced',parent_key=race.key)
    h=SimpleNamespace(config={},zone=ZoneInfo('UTC'),decision=lambda at:AT.date()+timedelta(days=2),
        history=(),sandbox=SimpleNamespace(db=db),policy=HarnessPolicy(),
        race_goal=general,race_goal_history=(RaceGoalApplication(race,race.available_at,race.available_at.date()),))
    captured=SimpleNamespace(contexts=(),restrictions=(),responses=(),metadata=())
    at=AT+timedelta(days=2)
    before=input_signature(h,captured,at)
    report(db,race,at=AT+timedelta(days=1))
    assert input_signature(h,captured,at)!=before
    db.close()


def test_completed_race_keeps_its_accounting_when_general_goal_is_restored():
    from run_analysis.adaptive.race_goal import RaceGoalRevision
    from run_analysis.adaptive.continuous_program_adapter import evaluate_continuous
    db=sqlite3.connect(':memory:');race=revision(AT-timedelta(days=5),days=5)
    physical=row(0,6.2,'quality',source='observed')
    outcome=report(db,race,status='recorded',physical=physical)
    general=RaceGoalRevision.create(available_at=AT+timedelta(days=1,seconds=1),
        distance='general_fitness',race_date=None,preparation_mode='balanced',parent_key=race.key,
        general_long_goal_miles=17.)
    at=AT+timedelta(days=2)
    ctx=replace(context(),decision=at.date(),race_goal=general,race_event_history=(outcome,),
        race_goal_history=(RaceGoalApplication(race,race.available_at,race.available_at.date()),
            RaceGoalApplication(general,general.available_at,general.available_at.date())))
    result=evaluate_continuous(ctx,(physical,),(),reference=REF,as_of=at,timezone='UTC',purpose_fraction=.5)
    assert result['evaluation'].conditional_ledger[0].role=='race'
    assert result['evaluation'].parts['delivery']['volume']==0.
    state=freeze(as_of=at,context=ctx,history=(physical,))
    assert not state.athlete.sessions[0].capacity_eligible
    assert restore(serialize(state),expected_fingerprint=state.fingerprint)==state


def test_nonparticipation_target_history_survives_a_later_goal_change():
    from run_analysis.adaptive.race_goal import RaceGoalRevision
    from run_analysis.adaptive.race_accounting import apply_purpose_window
    db=sqlite3.connect(':memory:');race=revision(AT-timedelta(days=5),days=5)
    outcome=report(db,race)
    at=AT+timedelta(days=2)
    general=RaceGoalRevision.create(available_at=at,distance='general_fitness',race_date=None,
        preparation_mode='balanced',parent_key=race.key)
    applications=(RaceGoalApplication(race,race.available_at,race.available_at.date()),
        RaceGoalApplication(general,at,at.date()))
    base=(TargetSegment(AT.date(),at.date(),30.),)
    before=overlay_applications(base,applications[:1],timezone='UTC',resolution=outcome)
    after=overlay_applications(base,applications,timezone='UTC',resolutions=(outcome,))
    assert before==after
    reported=outcome.effective_on
    assert _target_total(after,reported,reported+timedelta(days=1))==pytest.approx(30/7)
    ctx=replace(context(),race_goal=general,race_event_history=(outcome,),race_goal_history=applications)
    raw={'long':dict(daily=[dict(day=reported,loss=1.)],loss=1.,weighted_loss=1.)}
    assert apply_purpose_window(ctx,raw)['long']['daily'][0]['loss']==1.


def test_service_holds_unknown_event_even_on_upload_day_without_mutating_goal():
    from run_analysis.adaptive.service import AdaptivePlannerService
    from zoneinfo import ZoneInfo
    race=revision(AT-timedelta(days=5),days=5)
    h=SimpleNamespace(config={},race_goal=race,sandbox=SimpleNamespace(db=sqlite3.connect(':memory:')),
        zone=ZoneInfo('UTC'),decision=lambda at:AT.date()+timedelta(days=1))
    service=AdaptivePlannerService(h,{})
    with pytest.raises(ValueError,match='Confirm the race outcome'):
        service.refresh(AT+timedelta(hours=12))
    assert service.harness is h


def test_event_is_physical_distance_exposure_not_progression_capacity():
    from test_r3_decision_controls import evaluate
    db=sqlite3.connect(':memory:')
    race=revision(AT-timedelta(days=5),days=5)
    ordinary=row(-1,3.,source='observed')
    physical=row(0,10.,'quality',source='observed')
    outcome=report(db,race,status='recorded',physical=physical)
    at=AT+timedelta(days=1)
    ctx=replace(context(),race_goal=race,race_event=outcome,decision=at.date())
    result=evaluate((row(1,6.),),history=(ordinary,physical),ctx=ctx,as_of=at)
    guard=result['constraints']['sessions'][0]
    assert physical.key in guard['reference_keys']
    assert guard['committed_ceiling_miles']==11.
    state=freeze(as_of=at,context=ctx,history=(ordinary,physical))
    assert not next(r for r in state.athlete.sessions if r.key==physical.key).capacity_eligible


def test_post_marathon_distance_guard_does_not_forget_the_measured_event():
    from test_r3_decision_controls import evaluate
    db=sqlite3.connect(':memory:')
    race=revision(AT-timedelta(days=45),'marathon',days=24)
    physical=row(-21,26.2,'quality',source='observed')
    previous_long=row(-35,17.,'long',source='observed')
    outcome=report(db,race,status='recorded',physical=physical,at=AT-timedelta(days=20))
    ctx=replace(context(),race_goal=race,race_event=outcome)
    result=evaluate((row(1,17.,'long'),),history=(previous_long,physical),ctx=ctx,as_of=AT)
    guard=result['constraints']['sessions'][0]
    assert physical.key in guard['reference_keys']
    assert previous_long.key not in guard['reference_keys']
    assert guard['committed_ceiling_miles']==28.8
    # This numerical check is not a clearance or a proposal to run 28.8 miles.


def test_snapshot_rejects_outcome_backdated_effective_day():
    from run_analysis.adaptive.race_event import RaceEventResolution
    race=revision(AT-timedelta(days=5),days=5)
    at=AT+timedelta(days=1)
    bad=RaceEventResolution.create(race_key=race.key,status='did_not_start',
        available_at=at,parent_key=race.key,effective_on=AT.date())
    with pytest.raises(ValueError,match='effective date'):
        freeze(as_of=at,context=replace(context(),race_goal=race,race_event=bad,decision=at.date()))


@pytest.mark.parametrize('distance',['5k','10k','half_marathon','marathon'])
def test_resolved_nonstart_reenters_normal_admission_without_catchup(tmp_path,monkeypatch,distance):
    from test_adaptive_stage_lifecycle import completed_stage
    from run_analysis.adaptive.service import AdaptivePlannerService
    from run_analysis.adaptive.race_event import preparation_active
    h,inputs,config,at=completed_stage(tmp_path,monkeypatch,growth_focus='easy_volume',
        gradual=True,single_volume=True,continuation_boundary=True,purpose_continuation=True,
        purpose_continuation_phase='known_full_phase_v1',continuation_distribution='c1_v1',
        prospective_review_hour=4,**established_profile(distance))
    original=h.stage;physical=tuple(h.history)
    race=revision(h.origin,distance,days=7)
    h.race_goal=race
    h.race_goal_history=(RaceGoalApplication(race,h.origin,h.origin.date()),)
    at+=timedelta(hours=4)
    record_resolution(h.sandbox.db,revision=race,history=h.history,timezone_name='UTC',
        status='did_not_start',physical_key=None,expected_revision=race.key,as_of=at)
    service=AdaptivePlannerService(h,inputs)
    try:
        result=service.refresh(at,reason='explicit reconciled non-start integration control')
        updated=service.harness
        assert result['stage_review']['status']=='conditional_maintenance_candidate'
        assert result['stage_review']['race_interrupted']
        assert not result['stage_review']['confirmed_delivery']
        assert updated.stage.weekly_equivalent_miles==original.weekly_equivalent_miles
        assert updated.stage.long_miles==original.long_miles
        assert tuple(updated.history)==physical
        assert not preparation_active(updated,updated.decision(at))
        assert result['event']['selected']['eligible']
        saved=updated.sandbox.root/result['event']['snapshot_file']
        state=restore(saved.read_text(),expected_fingerprint=result['event']['snapshot_fingerprint'])
        assert state.prospective_volume_review is None
        assert state.context.race_demand==()
        updated.sandbox.close()
        from run_analysis.adaptive.new_model_daily_harness import DailyHarness
        reopened=DailyHarness.resume(tmp_path/'state',config,as_of=at)
        try:
            assert reopened.stage==updated.stage
            assert reopened.race_event.status=='did_not_start'
        finally:reopened.sandbox.close()
    finally:service.harness.sandbox.close()


@pytest.mark.parametrize('distance',['5k','10k','half_marathon','marathon'])
def test_real_unplanned_upload_needs_explicit_event_association_and_retains_load(tmp_path,monkeypatch,distance):
    """Real importer/service on declared established exposure; not optimal search."""
    from test_adaptive_stage_lifecycle import completed_stage
    from run_analysis.adaptive.new_model_daily_harness import synthetic_payload,DailyHarness
    from run_analysis.adaptive.service import AdaptivePlannerService
    h,inputs,config,at=completed_stage(tmp_path,monkeypatch,growth_focus='easy_volume',
        gradual=True,single_volume=True,continuation_boundary=True,purpose_continuation=True,
        purpose_continuation_phase='known_full_phase_v1',continuation_distribution='c1_v1',
        prospective_review_hour=4,**established_profile(distance))
    h.race_goal=revision(h.origin,distance,days=14)
    service=AdaptivePlannerService(h,inputs)
    original_stage=h.stage
    try:
        issued=service.refresh(at,reason='explicit event-day service fixture')['event']
        h=service.harness
        assert not any(day==h.race_goal.race_date.isoformat() for day,key in issued['program'])
        dose=h.catalog['quality:ordinary']
        payload,_,end=synthetic_payload(dose,at+timedelta(hours=16),hr=170,
            distance_factor=h.race_goal.goal.profile.distance_miles/dose.miles)
        uploaded=service.upload(payload,day=None,available_at=end,replan=False)
        actual=h.history[-1]
        assert actual.key==uploaded['upload']['comparison']['actual_key']
        with pytest.raises(ValueError,match='Confirm the race outcome'):
            service.refresh(end+timedelta(microseconds=1))
        assert service.harness.stage==original_stage
        record_resolution(h.sandbox.db,revision=h.race_goal,history=h.history,timezone_name='UTC',
            status='recorded',physical_key=actual.key,expected_revision=h.race_goal.key,
            as_of=end+timedelta(microseconds=2))
        after=service.refresh(end+timedelta(microseconds=3),reason='explicit recorded event fixture')
        h=service.harness
        assert after['stage_review']['status']=='dated_race_growth_hold'
        assert h.stage==original_stage
        assert after['event']['selected']['eligible']
        state=restore((h.sandbox.root/after['event']['snapshot_file']).read_text(),
            expected_fingerprint=after['event']['snapshot_fingerprint'])
        assert state.context.race_demand==()
        assert sum(r.key==actual.key for r in state.history)==1
        assert next(r for r in state.history if r.key==actual.key).demand==actual.demand
        assert not next(r for r in state.athlete.sessions if r.key==actual.key).capacity_eligible
        h.sandbox.close()
        reopened=DailyHarness.resume(tmp_path/'state',config,as_of=end+timedelta(microseconds=3))
        try:
            assert reopened.race_event.physical_key==actual.key
            assert next(r for r in reopened.history if r.key==actual.key).demand==actual.demand
        finally:reopened.sandbox.close()
    finally:service.harness.sandbox.close()


def test_cleared_reconciled_goal_does_not_resurrect_unperformed_growth(tmp_path,monkeypatch):
    from test_adaptive_stage_lifecycle import completed_stage
    from run_analysis.adaptive.service import AdaptivePlannerService
    h,inputs,_,at=completed_stage(tmp_path,monkeypatch,growth_focus='easy_volume',
        gradual=True,single_volume=True,continuation_boundary=True,purpose_continuation=True,
        purpose_continuation_phase='known_full_phase_v1',continuation_distribution='c1_v1',
        prospective_review_hour=4)
    original=h.stage;race=revision(h.origin,'10k',days=7)
    h.race_goal=race
    h.race_goal_history=(RaceGoalApplication(race,h.origin,h.origin.date()),)
    stamp=at+timedelta(hours=3)
    outcome=record_resolution(h.sandbox.db,revision=race,history=h.history,timezone_name='UTC',
        status='did_not_start',physical_key=None,expected_revision=race.key,as_of=stamp)
    clear=revision(stamp+timedelta(seconds=1),distance='general_fitness')
    h.race_goal=clear
    h.race_goal_history=(*h.race_goal_history,RaceGoalApplication(clear,clear.available_at,clear.available_at.date()))
    service=AdaptivePlannerService(h,inputs)
    try:
        result=service.refresh(at+timedelta(hours=4))
        h=service.harness
        assert result['stage_review']['race_interrupted']
        assert not result['stage_review']['confirmed_delivery']
        assert result['event']['selected']['eligible']
        assert h.stage.weekly_equivalent_miles==original.weekly_equivalent_miles
        assert getattr(h,'race_event',None) is None and h.race_event_history==(outcome,)
        state=restore((h.sandbox.root/result['event']['snapshot_file']).read_text(),
            expected_fingerprint=result['event']['snapshot_fingerprint'])
        assert state.prospective_volume_review is None
        assert state.context.race_goal==clear and state.context.race_event_history==(outcome,)
    finally:service.harness.sandbox.close()
