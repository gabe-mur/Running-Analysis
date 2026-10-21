from datetime import datetime, timedelta, timezone, time
from dataclasses import replace
import json
import yaml
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from run_analysis.config import load_config
from run_analysis.db import connect, initialize
from run_analysis.training_background import TrainingBackground, calibration_activity
from run_analysis.adaptive.reported_start import candidate_from_report, baseline_schedule, frequency_policy, start_if_needed
from run_analysis.adaptive.adaptive_candidate_pipeline import load_athlete
from run_analysis.importer import import_files
from run_analysis.processing import process_activities
from run_analysis.settings_service import save_settings_overlay
from run_analysis.web.schemas import SettingsPatch
from test_web_phase1 import _write_config
from fit_builder import build_run
from fastapi.testclient import TestClient
from run_analysis.web.app import create_app


def config_for(tmp_path, *, volume=24, count=4, long=8, maintaining=True):
    _write_config(tmp_path)
    cfg=load_config(tmp_path/'config.yaml')
    cfg['training_background']=dict(weekly_miles=volume,runs_per_week=count,regular_long_miles=long,maintaining_training=maintaining)
    now=datetime.now(timezone.utc)
    cfg['training_background_reported_at']=(now-timedelta(days=3)).isoformat()
    cfg.setdefault('coaching',{})['strength_training']=False
    cfg['setup']={'confirmed_steps':['heart_rate','zones']}
    cfg['profile']={'birth_date':'1985-01-01','sex':'male','weight_lb':150,'height_in':68}
    (tmp_path/'config.local.yaml').write_text(yaml.safe_dump(cfg))
    database=tmp_path/cfg['paths']['database'];database.parent.mkdir(parents=True,exist_ok=True)
    with connect(database) as db: initialize(db)
    return cfg,database,now


@pytest.mark.parametrize('values', [dict(weekly_miles=8,runs_per_week=4,regular_long_miles=9),
    dict(weekly_miles=24,runs_per_week=4,regular_long_miles=2),
    dict(weekly_miles=float('nan'),runs_per_week=4,regular_long_miles=8),
    dict(weekly_miles=24,runs_per_week=0,regular_long_miles=8)])
def test_report_rejects_inconsistent_levels(values):
    with pytest.raises(ValidationError): TrainingBackground(**values,maintaining_training=True)


def test_report_saves_with_timestamp_is_idempotent_and_can_be_cleared_before_adoption(tmp_path):
    _write_config(tmp_path)
    dbpath=tmp_path/'data/test.sqlite';dbpath.parent.mkdir()
    report=dict(weekly_miles=24,runs_per_week=4,regular_long_miles=8,maintaining_training=True)
    with connect(dbpath) as db:
        initialize(db)
        first=save_settings_overlay(tmp_path/'config.yaml',SettingsPatch(training_background=report),connection=db)
        again=save_settings_overlay(tmp_path/'config.yaml',SettingsPatch(training_background=report),connection=db)
        assert first['training_background_reported_at']==again['training_background_reported_at']
        assert db.execute('SELECT COUNT(*) FROM activities').fetchone()[0]==0
        cleared=save_settings_overlay(tmp_path/'config.yaml',SettingsPatch(training_background=None),connection=db)
        assert cleared['training_background'] is None


def test_adopted_report_cannot_be_used_as_a_live_rebase(tmp_path):
    _write_config(tmp_path)
    dbpath=tmp_path/'data/test.sqlite';dbpath.parent.mkdir()
    with connect(dbpath) as db:
        initialize(db)
        db.execute('INSERT INTO app_state VALUES (?,?,?)',('adaptive_runtime','{}',datetime.now(timezone.utc).isoformat()))
        with pytest.raises(ValueError,match='already adopted'):
            save_settings_overlay(tmp_path/'config.yaml',SettingsPatch(training_background=dict(
                weekly_miles=24,runs_per_week=4,regular_long_miles=8,maintaining_training=True)),connection=db)
        # Echoing an unchanged optional field must not break old Settings clients.
        assert save_settings_overlay(tmp_path/'config.yaml',SettingsPatch(training_background=None),
            connection=db)['training_background'] is None


def test_first_run_uses_normal_reported_distance_without_fake_weekly_history(tmp_path):
    cfg,path,now=config_for(tmp_path)
    with connect(path) as db:
        result=start_if_needed(db,cfg,tmp_path,as_of=now)
        assert result.target_evidence.capacity_reference_miles==24
        assert result.target_run_count==4
        assert result.run_count==1
        assert [d.recommendation.distance_target_miles for d in result.days if d.recommendation]==[5.3]
        assert all(d.day_role=='awaiting_calibration' for d in result.days if not d.recommendation)
        assert result.target_evidence.recent_7d_miles==0
        assert db.execute('SELECT COUNT(*) FROM activities').fetchone()[0]==0
        assert db.execute("SELECT COUNT(*) FROM app_state WHERE key='adaptive_runtime'").fetchone()[0]==0


def test_returning_after_break_does_not_automatically_prescribe_former_volume(tmp_path):
    cfg,path,now=config_for(tmp_path,maintaining=False)
    with connect(path) as db:
        result=start_if_needed(db,cfg,tmp_path,as_of=now)
        assert result.run_count==0
        assert 'after a break' in result.summary


def test_settings_api_and_first_plan_use_report_not_shipped_defaults(tmp_path):
    _write_config(tmp_path)
    report=dict(weekly_miles=24,runs_per_week=4,regular_long_miles=8,maintaining_training=True)
    with TestClient(create_app(tmp_path)) as client:
        response=client.patch('/api/settings',json=dict(training_background=report,
            profile=dict(birth_date='1985-01-01',sex='male',weight_lb=150,height_in=68),
            max_hr=190,resting_hr=48,
            zones={'z1':{'minimum_bpm':114,'maximum_bpm':132},'z2':{'minimum_bpm':133,'maximum_bpm':151},
                'z3':{'minimum_bpm':152,'maximum_bpm':170},'z4':{'minimum_bpm':171,'maximum_bpm':180},
                'z5':{'minimum_bpm':181,'maximum_bpm':190}},
            setup=dict(confirmed_steps=['profile','heart_rate','zones'],zone_method='percent_max')))
        assert response.status_code==200,response.text
        assert response.json()['training_background']==dict(report,last_run_date=None)
        assert response.json()['training_background_reported_at']
        plan=client.get('/api/weekly-schedule/latest')
        assert plan.status_code==200,plan.text
        runs=[d['recommendation'] for d in plan.json()['days'] if d['recommendation']]
        assert len(runs)==1
        assert runs[0]['distance_target_miles']==5.3
        assert plan.json()['target_evidence']['recent_7d_miles']==0
        assert client.get('/api/setup').json()['complete'] is False
        assert client.get('/api/settings').json()['training_background']==dict(report,last_run_date=None)


def test_unconfirmed_calibration_or_current_illness_cannot_start_running(tmp_path):
    from run_analysis.web.schemas import WeeklyScheduleRequest
    cfg,path,now=config_for(tmp_path)
    with connect(path) as db:
        cfg['setup']['confirmed_steps']=[]
        blocked=start_if_needed(db,cfg,tmp_path,as_of=now)
        assert blocked.run_count==0
        assert 'heart-rate setup' in blocked.summary
        cfg['setup']['confirmed_steps']=['heart_rate','zones']
        blocked=start_if_needed(db,cfg,tmp_path,as_of=now,
            request=WeeklyScheduleRequest(health_status='sick_or_recovering'))
        assert blocked.run_count==0


@pytest.mark.parametrize('has_hr',[True,False])
def test_gpx_calibration_and_adaptive_source_loader_agree(tmp_path,has_hr):
    from test_gpx import make_gpx
    cfg,path,now=config_for(tmp_path)
    source=make_gpx(tmp_path/'run.gpx',hr=has_hr,points=121,start=now-timedelta(days=1))
    with connect(path) as db:
        import_files(db,tmp_path,cfg['timezone_default'],paths=[source]);process_activities(db,cfg)
        now=datetime.now(timezone.utc)
        row=calibration_activity(db,cfg,reported_at=datetime.fromisoformat(cfg['training_background_reported_at']),as_of=now)
        assert bool(row)==has_hr
        # File availability matters: a recording not yet imported is not known.
        assert calibration_activity(db,cfg,reported_at=datetime.fromisoformat(cfg['training_background_reported_at']),
            as_of=now-timedelta(hours=1)) is None
    if has_hr:
        assert len(load_athlete(path,cfg,now)['history'])==1


@pytest.mark.parametrize('volume,count,long', [(5,3,2),(24,4,8),(45,5,12)])
def test_one_recording_calibrates_forecast_not_exposure(tmp_path,volume,count,long):
    cfg,path,now=config_for(tmp_path,volume=volume,count=count,long=long)
    fit=tmp_path/'calibration.fit'
    build_run(fit,start=now-timedelta(days=1),points=121,seconds_per_point=10,metres_per_point=25,heart_rate=145)
    with connect(path) as db:
        import_files(db,tmp_path,cfg['timezone_default'],paths=[fit]);process_activities(db,cfg)
        assert calibration_activity(db,cfg,reported_at=datetime.fromisoformat(cfg['training_background_reported_at']))
    now=datetime.now(timezone.utc)
    inputs=load_athlete(path,cfg,now)
    candidate=candidate_from_report(inputs,cfg,now)
    assert candidate['stage'].weekly_equivalent_miles==volume
    assert candidate['stage'].long_miles==long
    assert len(inputs['history'])==1
    assert candidate['construction']['nominal_minutes_per_mile']==pytest.approx(1609.344/150)
    lo,hi=frequency_policy(TrainingBackground.model_validate(cfg['training_background'])).bounds(volume)
    assert lo <= count <= hi


@pytest.mark.parametrize('volume,count,long',[(5,3,2),(45,5,12)])
def test_bounded_optimizer_start_at_other_levels(tmp_path,monkeypatch,volume,count,long):
    # Integration smoke at both ends, not a production-search or stability claim.
    from run_analysis.adaptive import live_bridge
    original=live_bridge.continuation_release_policy
    monkeypatch.setattr(live_bridge,'continuation_release_policy',lambda p: replace(original(p),
        evaluations=256,final_winner_evaluations=32,coordinated_final_evaluations=8))
    cfg,path,now=config_for(tmp_path,volume=volume,count=count,long=long)
    source=tmp_path/'calibration.fit'
    build_run(source,start=now-timedelta(days=1),points=121,seconds_per_point=10,metres_per_point=25,heart_rate=145)
    with connect(path) as db:
        import_files(db,tmp_path,cfg['timezone_default'],paths=[source]);process_activities(db,cfg)
        result=start_if_needed(db,cfg,tmp_path,as_of=datetime.now(timezone.utc))
        assert result.planner_status=='ready',result.planner_review_reason
        assert result.target_evidence.capacity_reference_miles==volume
        runs=[d.recommendation for d in result.planning_days if d.recommendation]
        assert any(r.distance_target_miles==long for r in runs)
        assert volume*3*.8 <= sum(r.distance_target_miles for r in runs) <= volume*3*1.2
        print('BOUNDED START',volume,count,long,len(runs),sum(r.distance_target_miles for r in runs),flush=True)


def test_actual_optimizer_starts_at_reported_level_and_reopens_without_rebootstrap(tmp_path):
    cfg,path,now=config_for(tmp_path)
    fit=tmp_path/'calibration.fit'
    build_run(fit,start=now-timedelta(days=1),points=121,seconds_per_point=10,metres_per_point=25,heart_rate=145)
    with connect(path) as db:
        import_files(db,tmp_path,cfg['timezone_default'],paths=[fit]);process_activities(db,cfg)
        now=datetime.now(timezone.utc)
        result=start_if_needed(db,cfg,tmp_path,as_of=now)
        print('OPENING',result.run_count,result.projected_distance_range_miles,flush=True)
        assert result.planner_status=='ready'
        assert result.planner_version==217
        assert result.target_evidence.capacity_reference_miles==24
        assert 3 <= result.run_count <= 5
        assert any(d.recommendation and d.recommendation.distance_target_miles==8 for d in result.planning_days)
        assert result.projected_distance_range_miles[0] >= 20
        pointer=json.loads(db.execute("SELECT value_json FROM app_state WHERE key='adaptive_runtime'").fetchone()[0])
        assert db.execute('SELECT COUNT(*) FROM activities').fetchone()[0]==1
        full=[d for d in result.planning_days if d.recommendation]
        assert 10 <= len(full) <= 15
        assert 65 <= sum(d.recommendation.distance_target_miles for d in full) <= 78
    from run_analysis.adaptive.factory import resume_service
    service=resume_service(pointer['root'],cfg,now)
    try:
        assert service.harness.stage.weekly_equivalent_miles==24
        assert service.harness.stage.long_miles==8
        assert len(service.harness.history)==1
        assert service.harness.frequency_contract.policy.bounds(24)==pytest.approx((3,5))
        assert service.harness.context(now).reported_distance.miles==8
        assert service.harness.sandbox.db.execute('SELECT COUNT(*) FROM sandbox_completions').fetchone()[0]==0
    finally: service.harness.sandbox.close()
    # The actual upload/replan bridge, with a faster pace and different HR,
    # must add physical work without re-bootstrap from two short recordings.
    first=next(d for d in result.days if d.recommendation)
    at=max(datetime.combine(first.date,time(18),ZoneInfo(cfg['timezone_default'])),now+timedelta(minutes=5))
    points=round(first.recommendation.distance_target_miles*1609.344/30)+1
    uploaded=tmp_path/'next-run.fit'
    build_run(uploaded,start=at,points=points,seconds_per_point=10,metres_per_point=30,heart_rate=150)
    later=at+timedelta(seconds=(points-1)*10,hours=1)
    from run_analysis.adaptive.live_bridge import schedule
    with connect(path) as db:
        import_files(db,tmp_path,cfg['timezone_default'],paths=[uploaded]);process_activities(db,cfg)
        assert start_if_needed(db,cfg,tmp_path,as_of=later) is None
        updated=schedule(db,cfg,tmp_path,as_of=later)
        print('AFTER UPLOAD',updated.planner_status,updated.target_evidence.capacity_reference_miles,flush=True)
        assert updated.planner_status=='ready',updated.planner_review_reason
        assert updated.target_evidence.capacity_reference_miles==24
        assert db.execute('SELECT COUNT(*) FROM activities').fetchone()[0]==2
        assert json.loads(db.execute("SELECT value_json FROM app_state WHERE key='adaptive_runtime'").fetchone()[0])==pointer
        cached=schedule(db,cfg,tmp_path,as_of=later)
        assert cached.model_dump()==updated.model_dump()
    resumed=resume_service(pointer['root'],cfg,later)
    try:
        h=resumed.harness
        assert len(h.history)==2
        assert h.stage.long_miles==8
        assert h.context(later).reported_distance.miles==8
        assert h.sandbox.db.execute('SELECT COUNT(*) FROM sandbox_completions').fetchone()[0]==1
        assert sum(r.miles for r in h.history)==pytest.approx((3000+(points-1)*30)/1609.344)
        assert all(r.source=='observed' for r in h.history)
        assert h.events[-1]['forecast_load_by_dose']['easy:5.3'] != h.events[0]['forecast_load_by_dose']['easy:5.3']
        print('REOPENED',len(h.history),h.stage.weekly_equivalent_miles,h.stage.long_miles,flush=True)
    finally: resumed.harness.sandbox.close()
