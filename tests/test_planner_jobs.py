from threading import Event
from time import monotonic,sleep
from uuid import uuid4

from fastapi.testclient import TestClient
import pytest

from run_analysis.planner_observation import PlannerProgress,phase,budget_progress
from run_analysis.web.planner_jobs import PlannerJobs
from run_analysis.web.app import create_app
from test_web_phase1 import _write_config


def wait_for(jobs,key,state='complete'):
    end=monotonic()+5
    while monotonic()<end:
        result=jobs.snapshot(key)
        if result['state']==state:return result
        sleep(.005)
    raise AssertionError(result)


def test_refresh_rejoins_real_worker_with_progress_and_idempotent_retry():
    tracker=PlannerProgress();jobs=PlannerJobs(tracker);entered=Event();release=Event();calls=[]
    def build():
        calls.append(1);phase('search');budget_progress(50,100);entered.set()
        assert release.wait(5)
        return {'planner_status':'ready','unchanged':'exact result'}
    try:
        a=jobs.start('a','refresh',{},build);assert entered.wait(2)
        assert a['state'] in ('queued','running')
        b=jobs.start('b','refresh',{},lambda:pytest.fail('duplicate refresh'))
        assert b['job_id']=='a' and jobs.resolve('b')=='a'
        assert tracker.snapshot(jobs.resolve('b'))['budget_used']==50
        assert jobs.start('a','refresh',{},build)['job_id']=='a'
        with pytest.raises(ValueError):jobs.start('a','update',{},build)
        release.set();result=wait_for(jobs,'a')
        assert result['result']=={'planner_status':'ready','unchanged':'exact result'}
        assert jobs.start('a','refresh',{},build)==result
        assert len(calls)==1 and tracker.snapshot('a')['state']=='complete'
    finally:release.set();jobs.close()


def test_refresh_cannot_jump_over_or_merge_edits_and_queue_is_bounded():
    jobs=PlannerJobs(PlannerProgress(),max_pending=4);entered=Event();release=Event();order=[]
    def first():entered.set();assert release.wait(5);order.append('first')
    try:
        jobs.start('a','refresh',{},first);assert entered.wait(2)
        jobs.start('b','update',{'health':'tired'},lambda:order.append('update'))
        jobs.start('c','rest',{'date':'today'},lambda:order.append('rest'))
        result=jobs.start('d','refresh',{},lambda:order.append('last'))
        assert result['job_id']=='d'  # Do not join old refresh a across edits.
        with pytest.raises(RuntimeError,match='queue is full'):
            jobs.start('e','update',{},lambda:None)
        assert jobs.start('f','refresh',{},lambda:None)['job_id']=='d'
        release.set();wait_for(jobs,'d')
        assert order==['first','update','rest','last']
    finally:release.set();jobs.close()


def test_failure_expiry_and_shutdown_do_not_cancel_running_publication():
    from run_analysis.adaptive.live_bridge import LivePlannerReviewRequired
    clock=[0.];tracker=PlannerProgress();jobs=PlannerJobs(tracker,clock=lambda:clock[0],retention=10)
    def failed():raise LivePlannerReviewRequired('Reconcile evidence first')
    try:
        jobs.start('failure','refresh',{},failed)
        result=wait_for(jobs,'failure','failed')
        assert result['error']=={'status':409,'detail':'Reconcile evidence first'}
        assert tracker.snapshot('failure')['state']=='failed'
        clock[0]=11;assert jobs.snapshot('failure') is None
        entered=Event();release=Event()
        def running():entered.set();assert release.wait(5);return 'published'
        jobs.start('running','refresh',{},running);assert entered.wait(2)
        jobs.start('queued','update',{},lambda:pytest.fail('shutdown canceled queue'))
        jobs.close();release.set()
        assert wait_for(jobs,'running')['result']=='published'
        assert jobs.snapshot('queued')['state']=='failed'
    finally:jobs.close()


def test_api_rejoins_job_and_health_and_progress_are_readable_while_worker_waits(tmp_path,monkeypatch):
    _write_config(tmp_path)
    entered=Event();release=Event();calls=[]
    def build(*args):
        calls.append(1);phase('search');budget_progress(10,100);entered.set()
        assert release.wait(5)
        return {'planner_status':'ready','unchanged':True}
    monkeypatch.setattr('run_analysis.web.app.ensure_current_weekly_schedule',build)
    app=create_app(tmp_path)
    try:
        with TestClient(app) as client:
            first,second=str(uuid4()),str(uuid4())
            a=client.post('/api/weekly-schedule/jobs',json=dict(request_id=first,operation='refresh'))
            assert a.status_code==202 and entered.wait(2)
            b=client.post('/api/weekly-schedule/jobs',json=dict(request_id=second,operation='refresh'))
            assert b.status_code==202 and b.json()['job_id']==first
            assert client.get('/api/health').status_code==200
            assert client.get('/api/weekly-schedule/progress/'+second).json()['budget_used']==10
            assert client.get('/api/weekly-schedule/jobs/'+first).json()['state']=='running'
            assert client.post('/api/weekly-schedule/jobs',json=dict(request_id=first,operation='update',
                update=dict(health_status='normal'))).status_code==409
            assert client.post('/api/weekly-schedule/jobs',json=dict(request_id=str(uuid4()),operation='rest')).status_code==422
            assert client.get('/api/weekly-schedule/jobs/'+str(uuid4())).status_code==404
            release.set();wait_for(app.state.planner_jobs,first)
            result=client.get('/api/weekly-schedule/jobs/'+first).json()
            assert result['result']=={'planner_status':'ready','unchanged':True} and len(calls)==1
    finally:release.set()


def test_edit_jobs_use_existing_health_and_rest_pipeline_with_worker_owned_connections(tmp_path,monkeypatch):
    from datetime import datetime,timezone
    from zoneinfo import ZoneInfo
    from types import SimpleNamespace
    from threading import get_ident
    import sqlite3
    from run_analysis.recommendation_service import load_forced_rest_dates
    _write_config(tmp_path);calls=[];thread=get_ident()
    def health(connection,status,*,source):
        assert get_ident()!=thread
        calls.append(('health',str(status),source))
    def generate(connection,config,request,root,**kwargs):
        calls.append(('generate',str(request.health_status),kwargs.get('discard_prior_schedule',False)))
        return {'planner_status':'ready','exact_pipeline':True}
    monkeypatch.setattr('run_analysis.health_reports.record_health_report',health)
    monkeypatch.setattr('run_analysis.web.app.generate_weekly_schedule',generate)
    monkeypatch.setattr('run_analysis.web.app.ensure_current_weekly_schedule',lambda *a:SimpleNamespace(days=[]))
    today=datetime.now(timezone.utc).astimezone(ZoneInfo('America/New_York')).date().isoformat()
    app=create_app(tmp_path)
    with TestClient(app) as client:
        key=str(uuid4())
        assert client.post('/api/weekly-schedule/jobs',json=dict(request_id=key,operation='update',
            update=dict(health_status='normal'))).status_code==202
        assert wait_for(app.state.planner_jobs,key)['result']['exact_pipeline']
        rest=str(uuid4())
        assert client.post('/api/weekly-schedule/jobs',json=dict(request_id=rest,operation='rest',
            rest=dict(date=today,is_rest_day=True))).status_code==202
        wait_for(app.state.planner_jobs,rest)
        with sqlite3.connect(tmp_path/'data/test.sqlite') as db:
            assert today in {day.isoformat() for day in load_forced_rest_dates(db)}
        assert calls[0]==('health','normal','POST /weekly-schedule')
        assert calls[1]==('generate','normal',False)
        assert calls[2]==('generate','normal',True)
