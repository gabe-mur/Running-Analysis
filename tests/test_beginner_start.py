from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

import pytest
import yaml

from run_analysis.beginner import (BeginnerFeedback, StarterRecipe, advance,
    sessions, save_feedback, progress)
from run_analysis.adaptive.reported_start import start_if_needed, candidate_from_measured_start
from run_analysis.adaptive.beginner_start import KEY
from run_analysis.adaptive.adaptive_candidate_pipeline import load_athlete
from run_analysis.config import load_config
from run_analysis.db import connect, initialize
from run_analysis.importer import import_files
from run_analysis.processing import process_activities
from test_reported_training_start import config_for

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_beginner_calibration import recorded_outing


def setup(tmp_path):
    cfg, database, now = config_for(tmp_path)
    cfg.pop('training_background')
    cfg.pop('training_background_reported_at')
    cfg['beginner_start'] = True
    cfg['weather'].update(historical_enabled=False, forecast_enabled=False)
    (tmp_path/'config.local.yaml').write_text(yaml.safe_dump(cfg))
    return cfg, database, now+timedelta(minutes=1)


def record(db, root, cfg, at, *, minutes=20, **kwargs):
    path=root/f"outing-{int(at.timestamp())}.fit"
    recorded_outing(path, at, dict(minutes=minutes, pace=11., **kwargs))
    import_files(db,root,cfg['timezone_default'],paths=[path])
    process_activities(db,cfg)
    return db.execute('SELECT MAX(id) FROM activities').fetchone()[0]


def test_explicit_beginner_selection_only_and_no_history_is_not_fake(tmp_path):
    cfg, path, now=setup(tmp_path)
    with connect(path) as db:
        assert start_if_needed(db,dict(cfg,beginner_start=False),tmp_path,as_of=now) is None
        plan=start_if_needed(db,cfg,tmp_path,as_of=now)
        assert plan.run_count==1
        assert plan.target_evidence.planning_mode=='baseline_required'
        assert plan.beginner_training['status']=='calibrating'
        assert db.execute('SELECT COUNT(*) FROM activities').fetchone()[0]==0
        assert db.execute("SELECT COUNT(*) FROM app_state WHERE key='adaptive_runtime'").fetchone()[0]==0
        assert [d.recommendation.duration_range_minutes for d in plan.days if d.recommendation]==[(10,30)]


@pytest.mark.parametrize('minutes,other,running,bout',[(10,{},10,10),(20,{},20,20),(30,{},30,30),
    (20,dict(run_seconds=60,walk_seconds=90),8,1),
    (30,dict(run_seconds=240,walk_seconds=60),24,4)])
def test_actual_fit_running_bouts_and_time_not_whole_outing_pace(tmp_path,minutes,other,running,bout):
    cfg,path,now=setup(tmp_path)
    with connect(path) as db:
        key=record(db,tmp_path,cfg,now-timedelta(days=1),minutes=minutes,**other)
        raw=sessions(db,cfg,now)[0]
        assert raw['eligible']
        assert raw['running_s']/60==pytest.approx(running,abs=.2)
        assert raw['bout_s']/60==pytest.approx(bout,abs=.2)
        assert not raw['confirmed']
        save_feedback(db,cfg,now,key,BeginnerFeedback(comfortable=True))
        plan=start_if_needed(db,cfg,tmp_path,as_of=now)
        assert plan.beginner_training['running_minutes']==pytest.approx(running,abs=.5)
        assert plan.target_evidence.baseline_session_minutes==pytest.approx(running,abs=.5)
        assert 1 <= plan.run_count <= 3
        assert all(d.recommendation.planning_role=='easy' for d in plan.days if d.recommendation)
        assert all(d.recommendation.quality_session_type is None for d in plan.days if d.recommendation)
        assert db.execute('SELECT COUNT(*) FROM activities').fetchone()[0]==1
        if other:
            assert all(d.recommendation.workout_type=='run_walk' for d in plan.days if d.recommendation)
        else:
            assert all(d.recommendation.workout_type=='easy' for d in plan.days if d.recommendation)


@pytest.mark.parametrize('options', [dict(target_hr=178),dict(missing_hr=True)])
def test_hard_and_missing_hr_do_not_establish_comfortable_running(tmp_path, options):
    cfg,path,now=setup(tmp_path)
    with connect(path) as db:
        key=record(db,tmp_path,cfg,now-timedelta(days=1),**options)
        save_feedback(db,cfg,now,key,BeginnerFeedback(comfortable=True))
        state=progress(sessions(db,cfg,now),{},now)
        assert state['recipe'] is None
        assert state['handoff'] is None


def test_missing_cadence_requests_times_instead_of_treating_slow_runner_as_walk(tmp_path):
    cfg,path,now=setup(tmp_path)
    with connect(path) as db:
        key=record(db,tmp_path,cfg,now-timedelta(days=1))
        db.execute('UPDATE trackpoints SET cadence=NULL,run_cadence=NULL,cadence_source=NULL')
        db.commit()
        assert sessions(db,cfg,now)[0]['needs_times']
        with pytest.raises(ValueError,match='recorded moving'):
            save_feedback(db,cfg,now,key,BeginnerFeedback(comfortable=True,running_minutes=21,longest_bout_minutes=21))
        save_feedback(db,cfg,now,key,BeginnerFeedback(comfortable=True,running_minutes=15,longest_bout_minutes=5))
        state=progress(sessions(db,cfg,now),{},now)
        assert state['recipe']['running_s']==900
        assert state['recipe']['bout_s']==300
        assert state['handoff'] is None


def test_recorded_stop_does_not_become_continuous_running_even_with_manual_times(tmp_path):
    cfg,path,now=setup(tmp_path)
    with connect(path) as db:
        key=record(db,tmp_path,cfg,now-timedelta(days=1),minutes=22,
            blocks=[(True,600),(False,120),(True,600)],walk_speed=0)
        measured=sessions(db,cfg,now)[0]
        assert measured['continuous_motion_s']<=610
        assert measured['bout_s']<=610
        with pytest.raises(ValueError,match='recorded stop or gap'):
            save_feedback(db,cfg,now,key,BeginnerFeedback(comfortable=True,running_minutes=20,longest_bout_minutes=20))


def sample(key, at, recipe, *, comfortable=True, hard=False):
    return dict(id=key,start=at.isoformat(),date=at.date().isoformat(),confirmed=comfortable and not hard,
        running_s=recipe.running_s,bout_s=recipe.bout_s,walk_s=recipe.walk_s,moving_s=recipe.outing_s,
        miles=recipe.outing_s/60/11.,running_miles=recipe.running_s/60/11.,
        feedback=dict(comfortable=comfortable),hard=hard,needs_times=False)


@pytest.mark.parametrize('seed_recipe', [StarterRecipe(480,60),StarterRecipe(600,600),StarterRecipe(1200,1200),StarterRecipe(1800,1800)])
def test_continuous_replay_from_various_abilities_reaches_real_handoff(seed_recipe):
    now=datetime.now(timezone.utc)
    observed=[sample(1,now,seed_recipe)]
    issues={}
    state=progress(observed,issues,now+timedelta(hours=1))
    assert state['recipe']['running_s']==seed_recipe.running_s
    initial=None
    for i in range(1,100):
        at=now+timedelta(days=(i//3)*7 + (0,2,4)[i%3])
        recipe=StarterRecipe(**state['recipe'])
        issues[at.date().isoformat()]=dict(issued_at=(at-timedelta(days=1)).isoformat(),recipe=asdict(recipe))
        observed.append(sample(i+1,at,recipe))
        next_state=progress(observed,issues,at+timedelta(hours=1))
        assert next_state==progress(observed,issues,at+timedelta(hours=1))
        assert next_state['recipe']['running_s'] <= recipe.running_s*1.1+1e-8
        state=next_state
        if state['handoff']:
            initial=(at-now).days
            break
    assert initial is not None
    assert 7 <= initial <= 100
    assert state['handoff']['weekly_miles']==pytest.approx(90/11)
    assert len(state['handoff']['source_ids'])==3
    print('STARTER REPLAY',asdict(seed_recipe),'handoff day',initial)


def test_misses_do_not_advance_and_hard_run_reduces_not_credits():
    now=datetime.now(timezone.utc)
    recipe=StarterRecipe(1200,1200)
    observed=[sample(1,now,recipe)]
    assert progress(observed,{},now+timedelta(days=6))['recipe']==asdict(recipe)
    assert progress(observed,{},now+timedelta(days=15))['recipe']['running_s']==960
    at=now+timedelta(days=2)
    issues={at.date().isoformat():dict(issued_at=now.isoformat(),recipe=asdict(recipe))}
    observed.append(sample(2,at,recipe,comfortable=False,hard=True))
    state=progress(observed,issues,at+timedelta(hours=1))
    assert state['recipe']['running_s']==960
    assert state['successes']==0
    assert state['completed_ids']==[1]
    assert progress(observed,issues,now+timedelta(days=31))['recipe'] is None
    returning=sample(3,now+timedelta(days=35),StarterRecipe(600,600))
    assert progress(observed+[returning],issues,now+timedelta(days=35,hours=1))['recipe']['running_s']==600


def test_actual_comfortable_sessions_on_other_dates_still_earn_credit():
    now=datetime.now(timezone.utc)
    recipe=StarterRecipe(1200,1200)
    observed=[sample(1,now,recipe),sample(2,now+timedelta(days=3),recipe),sample(3,now+timedelta(days=5),recipe)]
    issues={(now+timedelta(days=2)).date().isoformat():dict(issued_at=(now+timedelta(hours=1)).isoformat(),recipe=asdict(recipe))}
    state=progress(observed,issues,now+timedelta(days=5,hours=1))
    assert state['completed_ids']==[1,2,3]
    assert state['recipe']['running_s']>1200


def test_illness_and_manual_exclusion_cannot_seed_starter(tmp_path):
    cfg,path,now=setup(tmp_path)
    with connect(path) as db:
        key=record(db,tmp_path,cfg,now-timedelta(days=1))
        activity=db.execute('SELECT COALESCE(NULLIF(activity_id,\'\'),activity_uid) FROM activities WHERE id=?',(key,)).fetchone()[0]
        db.execute('INSERT INTO run_overrides(activity_id,health_tag,illness) VALUES (?,?,?)',(activity,'illness',1))
        db.commit()
        save_feedback(db,cfg,now,key,BeginnerFeedback(comfortable=True))
        assert progress(sessions(db,cfg,now),{},now)['recipe'] is None
        db.execute('UPDATE run_overrides SET health_tag=\'normal\',illness=0,include_in_model=0')
        db.commit()
        save_feedback(db,cfg,now,key,BeginnerFeedback(comfortable=True))
        assert progress(sessions(db,cfg,now),{},now)['recipe'] is None


def test_repeated_opening_is_stable_and_rest_constraints_do_not_create_streaks(tmp_path):
    cfg,path,now=setup(tmp_path)
    with connect(path) as db:
        key=record(db,tmp_path,cfg,now-timedelta(days=1))
        save_feedback(db,cfg,now,key,BeginnerFeedback(comfortable=True))
        one=start_if_needed(db,cfg,tmp_path,as_of=now)
        two=start_if_needed(db,cfg,tmp_path,as_of=now+timedelta(seconds=1))
        signature=lambda p:[(d.date,d.recommendation.structure if d.recommendation else None) for d in p.days]
        assert signature(one)==signature(two)
        assert one.beginner_training['running_minutes']==two.beginner_training['running_minutes']==20
        runs=[d.date for d in one.days if d.recommendation]
        db.execute('INSERT INTO app_state VALUES (?,?,?)',('weekly_forced_rest_dates',json.dumps([runs[0].isoformat()]),now.isoformat()))
        db.commit()
        changed=start_if_needed(db,cfg,tmp_path,as_of=now)
        runs=[d.date for d in changed.days if d.recommendation]
        assert all((b-a).days>=2 for a,b in zip(runs,runs[1:]))
        assert changed.beginner_training['running_minutes']==20


def test_normal_candidate_uses_measured_running_pace_and_full_physical_history(tmp_path):
    cfg,path,now=setup(tmp_path)
    with connect(path) as db:
        for i in (0,2,4,7):
            at=now+timedelta(days=i)
            key=record(db,tmp_path,cfg,at,minutes=30)
            save_feedback(db,cfg,at+timedelta(hours=1),key,BeginnerFeedback(comfortable=True))
        later=now+timedelta(days=7,hours=1)
        observed=sessions(db,cfg,later)
        seed=dict(weekly_miles=sum(s['running_miles'] for s in observed[-3:]),runs_per_week=3,
            ordinary_miles=observed[-1]['running_miles'],regular_long_miles=observed[-1]['running_miles'],
            source_ids=[s['id'] for s in observed[-3:]])
    inputs=load_athlete(path,cfg,later)
    candidate=candidate_from_measured_start(inputs,observed,seed,cfg,later)
    assert len(inputs['history'])==4
    assert candidate['stage'].weekly_equivalent_miles==8.2
    assert candidate['stage'].long_miles==2.7
    assert candidate['stage'].ordinary_miles==2.7
    assert all(r.source=='observed' for r in inputs['history'])
    assert any(d.role=='quality' for d in candidate['catalog'])


def test_full_fit_upload_calendar_and_handoff_to_normal_pipeline(tmp_path,monkeypatch):
    """Real import, real issued receipts, real normal optimizer, bounded search.

    No live database, fake history, prebuilt schedule or replaced scoring function.
    """
    import run_analysis.adaptive.live_bridge as bridge
    import os
    from run_analysis.adaptive.factory import resume_service
    original=bridge.continuation_release_policy
    if not os.environ.get('BEGINNER_FULL_SEARCH'):
        monkeypatch.setattr(bridge,'continuation_release_policy',lambda p:replace(original(p),
            evaluations=256,final_winner_evaluations=32,coordinated_final_evaluations=8))
    cfg,path,now=setup(tmp_path)
    now=now.replace(hour=16,minute=0,second=0)+timedelta(days=1)
    with connect(path) as db:
        opening=start_if_needed(db,cfg,tmp_path,as_of=now-timedelta(hours=1))
        for day in (0,2,4,7):
            at=now+timedelta(days=day)
            key=record(db,tmp_path,cfg,at,minutes=30)
            later=at+timedelta(hours=1)
            save_feedback(db,cfg,later,key,BeginnerFeedback(comfortable=True))
            result=start_if_needed(db,cfg,tmp_path,as_of=later)
            assert db.execute('SELECT COUNT(*) FROM activities').fetchone()[0]==(0,2,4,7).index(day)+1
            if day<7:
                assert result.beginner_training
                assert not db.execute("SELECT 1 FROM app_state WHERE key='adaptive_runtime'").fetchone()
        assert result.beginner_training is None
        assert result.planner_status=='ready',result.planner_review_reason
        pointer=json.loads(db.execute("SELECT value_json FROM app_state WHERE key='adaptive_runtime'").fetchone()[0])
        assert start_if_needed(db,cfg,tmp_path,as_of=later) is None
        with pytest.raises(ValueError,match='only during introductory'):
            save_feedback(db,cfg,later,key,BeginnerFeedback(comfortable=True))
    service=resume_service(pointer['root'],cfg,later)
    try:
        h=service.harness
        assert len(h.history)==4
        assert h.stage.weekly_equivalent_miles==8.2
        assert h.stage.ordinary_miles==h.stage.long_miles==2.7
        assert h.policy==bridge.general_progression_release_policy(bridge.continuation_release_policy(h.policy),h.stage)
        assert all(r.source=='observed' for r in h.history)
        assert h.sandbox.db.execute('SELECT COUNT(*) FROM sandbox_completions').fetchone()[0]==0
        scenario=json.loads((Path(pointer['root'])/'scenario.json').read_text())
        assert scenario['beginner_handoff']['optimizer_weights_changed'] is False
        assert scenario['beginner_handoff']['fake_completions_added']==0
        assert load_config(tmp_path/'config.yaml').get('training_background') is None
    finally:
        h.sandbox.close()


def test_walking_warmup_does_not_dilute_running_pace_or_rate_on_handoff(tmp_path):
    from run_analysis.adaptive.reported_start import measured_phase_inputs
    cfg,path,now=setup(tmp_path)
    with connect(path) as db:
        keys=[]
        for i in (0,2,4,7):
            at=now+timedelta(days=i)
            keys.append(record(db,tmp_path,cfg,at,minutes=40,blocks=[(False,300),(True,1800),(False,300)]))
            save_feedback(db,cfg,at+timedelta(hours=1),keys[-1],BeginnerFeedback(comfortable=True))
        later=now+timedelta(days=7,hours=1)
        observed=sessions(db,cfg,later)
        inputs=load_athlete(path,cfg,later)
        measured=measured_phase_inputs(db,inputs,observed,cfg)
        assert measured['history']==inputs['history']
        assert len(measured['history'])==len(measured['phase_history'])==4
        assert all(sum(p.seconds for p in h.phases)==1800 for h in measured['phase_history'])
        assert all(s['moving_s']==2400 and s['running_s']==1800 for s in observed)
        seed=dict(weekly_miles=sum(s['running_miles'] for s in observed[-3:]),ordinary_miles=observed[-1]['running_miles'],
            regular_long_miles=observed[-1]['running_miles'],source_ids=keys[-3:])
        candidate=candidate_from_measured_start(measured,observed,seed,cfg,later)
        assert candidate['construction']['nominal_minutes_per_mile']==pytest.approx(11,abs=.01)


def test_api_setup_feedback_and_watch_hr_update_before_handoff(tmp_path):
    from fastapi.testclient import TestClient
    from run_analysis.web.app import create_app
    from test_web_phase1 import _write_config
    _write_config(tmp_path)
    fixture=load_config(tmp_path/'config.yaml')
    model=yaml.safe_load((Path(__file__).resolve().parents[1]/'config.example.yaml').read_text())['model']
    fixture['model']={**model,**fixture['model']}
    (tmp_path/'config.local.yaml').write_text(yaml.safe_dump(fixture))
    with TestClient(create_app(tmp_path)) as client:
        saved=client.patch('/api/settings',json=dict(beginner_start=True,training_background=None,
            profile=dict(birth_date='1985-01-01',sex='male',weight_lb=150,height_in=68),
            setup=dict(confirmed_steps=['profile','heart_rate','zones'])))
        assert saved.status_code==200,saved.text
        assert saved.json()['beginner_start']
        assert saved.json()['planner']['status']=='introductory'
        cfg=load_config(tmp_path/'config.yaml')
        plan=client.get('/api/weekly-schedule/latest')
        assert plan.status_code==200,plan.text
        assert plan.json()['beginner_training']['status']=='calibrating'
        with connect(tmp_path/cfg['paths']['database']) as db:
            key=record(db,tmp_path,cfg,datetime.now(timezone.utc)-timedelta(days=1))
        feedback=client.post(f'/api/beginner-feedback/{key}',json={'comfortable':True})
        assert feedback.status_code==200,feedback.text
        cfgchange=client.patch('/api/settings',json={'resting_hr':49})
        assert cfgchange.status_code==200,cfgchange.text
        assert all(s['status']!='failed' for s in cfgchange.json()['recalculation']), cfgchange.json()['recalculation']
        plan=client.get('/api/weekly-schedule/latest')
        assert plan.status_code==200,plan.text
        assert plan.json()['beginner_training']['running_minutes']==20
        assert client.patch('/api/settings',json={'beginner_start':True,'training_background':dict(
            weekly_miles=10,runs_per_week=3,regular_long_miles=4,maintaining_training=True)}).status_code==422


@pytest.mark.parametrize('initial', [dict(minutes=10),dict(minutes=20,run_seconds=60,walk_seconds=90)])
def test_recorded_fit_starter_calendar_replay_reaches_handoff_without_invented_adherence(tmp_path,monkeypatch,initial):
    """Recorded multiweek starter replay. Normal search is tested separately.

    Synthetic easy HR and perfect comfortable adherence are explicit inputs,
    not a prediction of a human runner's response.
    """
    import run_analysis.adaptive.reported_start as reported
    class HandoffReached(Exception):
        pass
    captured={}
    def handoff(connection, config, root, as_of, *, measured_start):
        captured['seed']=measured_start[1]
        captured['as_of']=as_of
        raise HandoffReached()
    monkeypatch.setattr(reported,'adopt',handoff)
    cfg,path,now=setup(tmp_path)
    now=now.replace(hour=16,minute=0,second=0,microsecond=0)+timedelta(days=1)
    with connect(path) as db:
        start_if_needed(db,cfg,tmp_path,as_of=now-timedelta(hours=1))
        key=record(db,tmp_path,cfg,now,**initial)
        later=now+timedelta(hours=1)
        save_feedback(db,cfg,later,key,BeginnerFeedback(comfortable=True))
        plan=start_if_needed(db,cfg,tmp_path,as_of=later)
        for n in range(60):
            command=StarterRecipe(**json.loads(db.execute('SELECT value_json FROM app_state WHERE key=?',(KEY,)).fetchone()[0])['recipe'])
            next_day=next(day for day in plan.days if day.recommendation)
            at=datetime.combine(next_day.date,datetime.min.time(),tzinfo=timezone.utc)+timedelta(hours=16)
            assert at>later
            blocks=[(False,300)]+[(kind=='run',seconds) for kind,seconds in command.blocks]+[(False,300)]
            key=record(db,tmp_path,cfg,at,minutes=command.outing_s/60,blocks=blocks)
            later=at+timedelta(hours=1)
            save_feedback(db,cfg,later,key,BeginnerFeedback(comfortable=True))
            try:
                plan=start_if_needed(db,cfg,tmp_path,as_of=later)
            except HandoffReached:
                break
            snapshot=plan.model_dump()
            assert start_if_needed(db,cfg,tmp_path,as_of=later).model_dump()==snapshot
        assert captured
        assert (captured['as_of']-now).days <= 105
        assert captured['seed']['weekly_miles']==pytest.approx(90/11,abs=.1)
        assert db.execute('SELECT COUNT(*) FROM activities').fetchone()[0]==n+2
        print('FIT CALENDAR REPLAY',initial,'handoff day',(captured['as_of']-now).days,'recordings',n+2)
