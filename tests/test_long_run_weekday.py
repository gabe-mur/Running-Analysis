"""Optional calendar anchor: control wiring/admission, not recovery calibration."""
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from run_analysis.adaptive.planner_preferences import (
    PlannerPreferencesRevision, queue_request, latest_request, long_day_allowed,
)
from run_analysis.adaptive.decision_snapshot import serialize,restore,evaluate_snapshot
from run_analysis.adaptive.live_bridge import input_signature
from run_analysis.adaptive.stage_calendar_seeds import complete_stage_seeds
from run_analysis.adaptive.continuous_search_adapter import search
from run_analysis.adaptive.new_model_daily_harness import Dose,DailyHarness
from run_analysis.adaptive.program_objective_trial import TrialPolicy
from run_analysis.adaptive.continuation_boundary_trial import purpose_handoff_losses
from run_analysis.program_constraints import DateRestriction
from run_analysis.web.schemas import PlannerPreferencesPatch
from run_analysis.web.app import create_app
from test_adaptive_settings import adopted
from test_decision_snapshot import freeze
from test_r3_decision_controls import REF,row


def revision(state,day=0,not_before=date(2026,9,28)):
    return PlannerPreferencesRevision.create(available_at=state.as_of,
        goal_long_miles=17,frequency_mode='balanced',parent_key='fixture',
        long_run_weekday=day,long_run_not_before=not_before)


def anchored(state,r):
    return replace(state,planner_preferences=r,context=replace(state.context,
        long_run_weekday=r.long_run_weekday,long_run_not_before=r.long_run_not_before))


@pytest.mark.parametrize('value',[-1,7,True,2.5,'5'])
def test_invalid_weekdays_cannot_be_saved(value):
    with pytest.raises(ValueError):
        PlannerPreferencesPatch(long_run_goal_miles=17,run_frequency='balanced',
            expected_revision='fixture',long_run_weekday=value)


def test_automatic_preserves_old_bytes_and_constrained_snapshots_roundtrip():
    state=freeze();original=serialize(state)
    assert 'long_run_weekday' not in original
    loaded=restore(original,expected_fingerprint=state.fingerprint)
    assert serialize(loaded)==original
    r=revision(state)
    changed=anchored(state,r)
    assert restore(serialize(changed),expected_fingerprint=changed.fingerprint)==changed
    with pytest.raises(ValueError,match='identity changed'):
        replace(r,long_run_weekday=1,long_run_not_before=date(2026,9,29))


def test_admission_uses_athlete_local_day_for_hard_and_soft_longs_without_changing_physical_load():
    state=freeze(timezone='America/New_York')
    state=anchored(state,revision(state))
    wrong=(row(0,8.4,'long'),row(23,8.4,'long'))
    r=evaluate_snapshot(state,wrong,('long','long'),reference=REF)
    assert not r['decision_eligible'] and not r['construction_eligible']
    assert r['long_run_day_admission']['violations']==(wrong[0].key,wrong[1].key)
    monday=row(1,8.4,'long')
    at_utc_tuesday=replace(monday,start=monday.start+timedelta(hours=17),end=monday.end+timedelta(hours=17))
    checked=evaluate_snapshot(state,(at_utc_tuesday,),('long',),reference=REF)
    assert checked['long_run_day_admission']['eligible']  # Monday evening in New York.
    assert next(r.day for r in checked['evaluation'].conditional_ledger if r.key==at_utc_tuesday.key).weekday()==0
    auto=replace(state,planner_preferences=None,context=replace(state.context,
        long_run_weekday=None,long_run_not_before=None))
    assert 'long_run_day_admission' not in evaluate_snapshot(auto,wrong,('long','long'),reference=REF)


def test_anchored_long_still_cannot_override_protected_rest():
    state=freeze()
    state=anchored(state,revision(state))
    day=date(2026,9,28)
    state=replace(state,restrictions=(DateRestriction('rest',day,day+timedelta(days=1),state.as_of,'reserved_rest','user'),))
    r=evaluate_snapshot(state,(row(1,8.4,'long'),),('long',),reference=REF)
    assert r['long_run_day_admission']['eligible']
    assert not r['decision_eligible']


def test_transition_delays_from_issued_next_long_then_survives_retries_and_unrelated_edits(tmp_path):
    _,root,state=adopted(tmp_path)
    with __import__('sqlite3').connect(root/'sandbox.sqlite') as db:
        at=state.as_of+timedelta(seconds=1)
        patch=PlannerPreferencesPatch(long_run_goal_miles=17,run_frequency='balanced',
            expected_revision='adopted:'+state.fingerprint,long_run_weekday=5)
        r=queue_request(db,state=state,stage={'long_miles':8.9,'goal_long_miles':17},patch=patch,as_of=at,
            prior_program=(('2026-09-28','long'),))
        assert r.long_run_not_before==date(2026,10,3)
        assert not long_day_allowed(r,date(2026,9,26)) and long_day_allowed(r,date(2026,10,10))
        assert latest_request(db)==r
        assert queue_request(db,state=state,stage={'long_miles':8.9,'goal_long_miles':17},patch=patch,as_of=at+timedelta(seconds=1)) is None
        edit=PlannerPreferencesPatch(long_run_goal_miles=18,run_frequency='more',expected_revision=r.key)
        second=queue_request(db,state=state,stage={'long_miles':8.9,'goal_long_miles':17},patch=edit,as_of=at+timedelta(seconds=2))
        assert second.long_run_weekday==5 and second.long_run_not_before==r.long_run_not_before
        cleared=queue_request(db,state=state,stage={'long_miles':8.9,'goal_long_miles':17},as_of=at+timedelta(seconds=3),
            patch=PlannerPreferencesPatch(long_run_goal_miles=18,run_frequency='more',expected_revision=second.key,long_run_weekday=None))
        assert cleared.long_run_weekday is None and cleared.long_run_not_before is None


def test_missing_transition_anchor_fails_without_saving(tmp_path):
    _,root,state=adopted(tmp_path)
    with __import__('sqlite3').connect(root/'sandbox.sqlite') as db:
        with pytest.raises(ValueError,match='No issued future long run'):
            queue_request(db,state=state,stage={'long_miles':8.9,'goal_long_miles':17},as_of=state.as_of+timedelta(seconds=1),
                patch=PlannerPreferencesPatch(long_run_goal_miles=17,run_frequency='balanced',expected_revision='adopted:'+state.fingerprint,long_run_weekday=5))
        assert latest_request(db) is None


def test_api_weekday_save_is_pending_noop_retry_stale_safe_and_clearable(tmp_path):
    _,root,state=adopted(tmp_path)
    import sqlite3
    anchor=datetime.now(timezone.utc).date()+timedelta(days=1)
    weekday=(anchor.weekday()+2)%7
    with sqlite3.connect(root/'sandbox.sqlite') as db:
        event=json.loads(db.execute('SELECT event FROM sandbox_decisions').fetchone()[0])
        event['program']=[(anchor.isoformat(),'long')]
        db.execute('UPDATE sandbox_decisions SET event=?',(json.dumps(event),))
        original=db.execute('SELECT * FROM sandbox_decisions').fetchall()
    client=TestClient(create_app(tmp_path))
    old=client.get('/api/settings').json()['planner']
    assert old['long_run_weekday'] is None
    patch=dict(long_run_goal_miles=17,run_frequency='balanced',
        expected_revision=old['preferences_revision'],long_run_weekday=weekday)
    response=client.patch('/api/settings',json={'planner_preferences':patch})
    assert response.status_code==200,response.text
    requested=response.json()['planner']
    assert requested['long_run_weekday'] is None and requested['preferences_pending']
    assert requested['requested_long_run_weekday']==weekday
    assert requested['requested_long_run_not_before']==(anchor+timedelta(days=2)).isoformat()
    assert requested['snapshot_fingerprint']==state.fingerprint
    assert client.patch('/api/settings',json={'planner_preferences':patch}).status_code==200
    stale=client.patch('/api/settings',json={'planner_preferences':dict(patch,long_run_weekday=(weekday+1)%7)})
    assert stale.status_code==422
    # Old clients omitting the new field preserve it rather than clearing it.
    edit=dict(long_run_goal_miles=18,run_frequency='balanced',expected_revision=requested['preferences_revision'])
    edited=client.patch('/api/settings',json={'planner_preferences':edit}).json()['planner']
    assert edited['requested_long_run_weekday']==weekday
    cleared=client.patch('/api/settings',json={'planner_preferences':dict(edit,
        long_run_weekday=None,expected_revision=edited['preferences_revision'])})
    assert cleared.status_code==200
    assert cleared.json()['planner']['requested_long_run_weekday'] is None
    assert cleared.json()['planner']['requested_long_run_not_before'] is None
    with sqlite3.connect(root/'sandbox.sqlite') as db:
        assert db.execute('SELECT * FROM sandbox_decisions').fetchall()==original
        assert db.execute('SELECT count(*) FROM adaptive_preference_requests').fetchone()[0]==3


def test_day_and_clear_trigger_refresh_but_automatic_does_not(tmp_path):
    _,_,state=adopted(tmp_path)
    h=SimpleNamespace(config={},zone=__import__('zoneinfo').ZoneInfo('UTC'),history=state.history,
        decision=lambda _:state.context.decision,frequency_contract=state.frequency_contract,
        policy=SimpleNamespace(continuation_distribution='linear_v1',
            progression_admission='checkpoint_v1',progression_credit='issued_nominal_v1',
            general_progression=None),planner_preferences=None)
    captured=SimpleNamespace(contexts=(),restrictions=(),responses=(),metadata=())
    automatic=input_signature(h,captured,state.as_of)
    h.planner_preferences=revision(state)
    assert input_signature(h,captured,state.as_of)!=automatic
    h.planner_preferences=PlannerPreferencesRevision.create(available_at=state.as_of,
        goal_long_miles=17,frequency_mode='balanced',parent_key='fixture')
    assert input_signature(h,captured,state.as_of)==automatic


def library(start=date(2026,9,27),days=35):
    dates=tuple((start+timedelta(days=i)).isoformat() for i in range(days))
    catalog={d.key:d for d in (Dose('e','easy',3.5,(('warmup',2100),)),
        Dose('l','long',8.4,(('warmup',5040),)),Dose('q','quality',3.9,(('work',600),)))}
    allowed=frozenset(d for d in dates if date.fromisoformat(d).weekday()==5 and d>='2026-10-03')
    seeds=complete_stage_seeds(dates,catalog,TrialPolicy(weekly_miles=20),supporting_miles=7.7,long_dates=allowed)
    return dates,catalog,allowed,seeds


def test_seed_generation_covers_quality_placement_and_keeps_absolute_long_weekday_on_rollover():
    dates,catalog,allowed,seeds=library()
    assert seeds and all(d in allowed for p in seeds for d,k in p if k=='l')
    assert len({next(date.fromisoformat(d).weekday() for d,k in p if k=='q') for p in seeds})==6
    _,_,_,later=library(start=date(2026,9,28),days=34)
    assert {tuple((d,k) for d,k in p if k=='l') for p in seeds}=={tuple((d,k) for d,k in p if k=='l') for p in later}


def test_real_search_does_not_spend_model_calls_on_wrong_weekdays():
    dates,catalog,allowed,seeds=library(days=21)
    calls=[]
    def evaluate(p):
        assert all(d in allowed for d,k in p if k=='l')
        calls.append(p)
        return {'cost':abs(60-sum(catalog[k].miles for _,k in p)),
            'eligible':sum(k=='l' for _,k in p)>=2,'construction_eligible':True}
    result=search(dates=dates,keys=tuple(catalog),previous=((dates[0],'l'),),
        evaluate=evaluate,budget=250,hard_end=dates[-1],roles={k:d.role for k,d in catalog.items()},
        seed_programs=seeds,complete_first=True,allowed_role_dates={'long':allowed})
    assert result['selected']['eligible'] and calls
    assert all(d in allowed for d,k in result['program'] if k=='l')


@pytest.mark.parametrize('delayed',[False,True])
def test_continuation_long_renewal_uses_selected_weekday_not_old_phase(delayed):
    from test_continuation_boundary_trial import setup
    ctx,refs,_,ev=setup()
    ctx=replace(ctx,purpose_continuation_phase='known_full_phase_v1',purpose_continuation=True,
        continuation_boundary_references=tuple((role,miles,demand) for role,(miles,demand) in refs.items()),long_run_weekday=5)
    if delayed:
        earliest=ctx.end+timedelta(days=20)
        ctx=replace(ctx,long_run_not_before=earliest+timedelta(days=(5-earliest.weekday())%7))
    handoffs=tuple(ctx.end+timedelta(days=i) for i in range(7))
    r=purpose_handoff_losses(ctx,ev.conditional_ledger,handoffs)
    assert all(p['basis']=='preferred_weekday_phase' and
        (p['handoff']+timedelta(days=p['arrival_delay_days'])).weekday()==5
        for p in r['long']['phase_by_handoff'])
    if delayed:
        assert all(p['handoff']+timedelta(days=p['arrival_delay_days'])>=ctx.long_run_not_before
            for p in r['long']['phase_by_handoff'])


def test_full_harness_applies_anchor_replays_and_restarts_without_stage_reset(tmp_path,monkeypatch):
    from test_adaptive_stage_lifecycle import completed_stage
    from test_frequency_lifecycle import adopted as frequency,replay
    from run_analysis.adaptive.planner_preferences import apply_request
    h,_,config,at=completed_stage(tmp_path,monkeypatch,continuation_boundary=True,
        single_volume=True,purpose_continuation=True,purpose_continuation_phase='known_full_phase_v1')
    h.frequency_contract=frequency(h,at-timedelta(days=1))
    try:
        first=h.replan(at,reason='weekday baseline fixture');state,_,_=replay(h,first)
        old_stage=h.stage;old_history=h.history
        r=queue_request(h.sandbox.db,state=state,stage=first['stage_runtime']['stage'],prior_program=first['program'],
            as_of=at+timedelta(seconds=1),patch=PlannerPreferencesPatch(long_run_goal_miles=17,run_frequency='balanced',
                long_run_weekday=5,expected_revision='adopted:'+state.fingerprint))
        assert apply_request(h,as_of=r.available_at)
        # Real search, not the fixture's seeded-selector shortcut.
        import run_analysis.adaptive.new_model_daily_harness as module
        monkeypatch.setattr(module,'migrated_search',search)
        h.policy=replace(h.policy,evaluations=300,complete_first_search=True)
        second=h.replan(at+timedelta(seconds=2),reason='weekday constrained full pipeline')
        applied,planned,result=replay(h,second)
        assert all(long_day_allowed(r,run.start.date()) for run in planned if run.role=='long')
        assert result['long_run_day_admission']['eligible']
        assert h.stage==old_stage and h.history==old_history
        assert applied.planner_preferences==r
        root=h.sandbox.root
    finally:h.sandbox.close()
    reopened=DailyHarness.resume(root,config,as_of=at+timedelta(seconds=3))
    try:
        assert reopened.planner_preferences==r
        replay(reopened,reopened.events[-1])
    finally:reopened.sandbox.close()
