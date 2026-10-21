from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from types import SimpleNamespace
from uuid import uuid4
from threading import Event

from fastapi.testclient import TestClient
import pytest

from run_analysis.planner_observation import PlannerProgress, phase, model_check, finished, budget_progress
from run_analysis.planner_observation import planner_job, follow_job
from run_analysis.web.app import create_app
from test_web_phase1 import _write_config


def test_progress_is_request_scoped_budget_based_and_not_a_time_prediction():
    clock = [0.]
    tracker = PlannerProgress(clock=lambda: clock[0])
    assert tracker.snapshot('absent') == {'state': 'unknown'}
    phase('search');model_check();budget_progress(900,1000)  # No active observer.
    with tracker.observe('a'):
        phase('history');model_check()
        phase('search');budget_progress(250,1000)
        with ThreadPoolExecutor() as executor:
            executor.submit(copy_context().run, model_check).result()
        clock[0] = 10
        snap = tracker.snapshot('a')
        assert snap['model_checks'] == 1
        assert snap['budget_used'] == 250 and snap['budget_total'] == 1000
        assert snap['estimated_percent'] == pytest.approx(28.5)
        assert snap['elapsed_seconds'] == 10
        with tracker.observe('b'):
            phase('search');budget_progress(1000,1000);model_check()
            assert tracker.snapshot('b')['estimated_percent'] == 90
            assert tracker.snapshot('a') == snap
            finished()
        phase('refine');model_check()
        assert tracker.snapshot('a')['estimated_percent'] == 92
        assert tracker.snapshot('a')['model_checks'] == 2
        phase('publish');finished()
    clock[0] = 50
    result = tracker.snapshot('a')
    assert result['elapsed_seconds'] == 10
    assert result['state'] == 'complete' and result['estimated_percent'] == 100
    assert result['phases'] == ['waiting','history','search','refine','publish']


def test_failed_stopped_jobs_never_reach_100_and_records_expire():
    clock=[0.]
    tracker=PlannerProgress(clock=lambda:clock[0],retention_seconds=20,capacity=2)
    with pytest.raises(ValueError), tracker.observe('a'):
        phase('search');budget_progress(100,100)
        raise ValueError('expected')
    assert tracker.snapshot('a')['state']=='failed'
    assert tracker.snapshot('a')['estimated_percent']==90
    with tracker.observe('b'):finished(False)
    with tracker.observe('c'):finished()
    assert tracker.snapshot('a')=={'state':'unknown'}
    clock[0]=21
    assert tracker.snapshot('c')=={'state':'unknown'}


def test_actual_search_budget_is_observed_without_changing_search_result():
    from run_analysis.adaptive.continuous_search_adapter import search
    options=dict(dates=('2026-10-10','2026-10-11'),keys=('easy',),previous=(),
        evaluate=lambda p:dict(cost=-len(p),eligible=True),budget=10,
        hard_end='2026-10-12',roles={'easy':'easy'})
    plain=search(**options)
    tracker=PlannerProgress()
    with tracker.observe('search'):
        phase('search')
        watched=search(**options)
        progress=tracker.snapshot('search')
    assert watched['program']==plain['program']
    assert watched['cost']==plain['cost']
    assert watched['evaluations']==plain['evaluations']
    assert watched['selected']==plain['selected']
    assert watched['budget_exhausted']==plain['budget_exhausted']
    assert watched['local_neighborhood_complete']==plain['local_neighborhood_complete']
    assert progress['budget_used']==watched['evaluations']
    assert progress['budget_total']==10


def test_continuous_model_result_is_identical_with_observer():
    from test_continuous_program_adapter import evaluate, run
    history=(run(-2,source='observed',role='long',work=0),)
    planned=(run(role='easy',work=0),)
    plain=evaluate(history,planned)
    tracker=PlannerProgress()
    with tracker.observe('a'):
        phase('search')
        watched=evaluate(history,planned)
        assert tracker.snapshot('a')['model_checks']==1
    assert watched==plain


def test_read_only_progress_endpoint_and_route_header_context(tmp_path):
    _write_config(tmp_path)
    app=create_app(tmp_path)
    request_id=str(uuid4())
    # A tiny route stands in for the expensive planner, exercises the same
    # middleware/threadpool propagation without running or altering a plan.
    @app.get('/api/weekly-schedule')
    def probe():
        phase('search');budget_progress(40,100);model_check()
        return {'ok':True}
    with TestClient(app) as client:
        assert client.get('/api/weekly-schedule/progress/'+request_id).json()=={'state':'unknown'}
        assert client.get('/api/weekly-schedule',headers={'X-Planner-Request':request_id}).json()=={'ok':True}
        result=client.get('/api/weekly-schedule/progress/'+request_id).json()
        assert result['model_checks']==1 and result['state']=='complete'
        assert result['estimated_percent']==100
        assert result['budget_used']==40 and result['budget_total']==100
        assert 'no-store' in client.get('/api/weekly-schedule/progress/'+request_id).headers['cache-control']
        assert client.get('/api/weekly-schedule/progress/not-a-uuid').status_code==422
        assert client.get('/api/weekly-schedule',headers={'X-Planner-Request':'invalid'}).status_code==200
    assert not (tmp_path/'data/test.sqlite').exists()


def test_progress_can_be_polled_while_planner_request_is_running(tmp_path):
    _write_config(tmp_path)
    app = create_app(tmp_path)
    request_id = str(uuid4())
    entered, release = Event(), Event()

    @app.get('/api/weekly-schedule')
    def pending_probe():
        phase('search');budget_progress(50,100);model_check()
        entered.set()
        assert release.wait(5)
        return {'ok': True}

    with TestClient(app) as client, ThreadPoolExecutor() as executor:
        future = executor.submit(client.get, '/api/weekly-schedule', headers={'X-Planner-Request': request_id})
        try:
            assert entered.wait(5)
            response = client.get('/api/weekly-schedule/progress/' + request_id)
            assert response.status_code == 200
            result = response.json()
            assert result['state'] == 'running'
            assert result['phase'] == 'search'
            assert result['estimated_percent'] == 49
            assert result['model_checks'] == 1
            assert result['budget_used'] == 50
        finally:
            release.set()
        assert future.result().json() == {'ok': True}
        assert client.get('/api/weekly-schedule/progress/' + request_id).json()['state'] == 'complete'


def test_refresh_follows_headerless_upload_job_and_keeps_own_completion(tmp_path, monkeypatch):
    from threading import Lock
    _write_config(tmp_path)
    app = create_app(tmp_path)
    owner_entered, waiter_entered, release, loading_result, finish_result = (Event() for _ in range(5))
    lock = Lock()
    request_id = str(uuid4())

    # An untracked HTTP caller (like upload/sync) owns the real planner lock.
    def owner(root, config, payloads):
        with lock, planner_job('runtime-a'):
            phase('search');budget_progress(5000,10000);model_check()
            owner_entered.set()
            assert release.wait(5)
            phase('publish')
        return {'files': [], 'stages': [], 'primary_activity_id': None}

    # Exercise the real multipart/async upload route and its worker-thread
    # context propagation; only expensive import/search work is stood in.
    monkeypatch.setattr('run_analysis.web.app.run_locked_upload_pipeline', owner)

    @app.get('/api/weekly-schedule')
    def queued_refresh():
        phase('waiting');follow_job('runtime-a');waiter_entered.set()
        with lock, planner_job('runtime-a'):
            phase('history')  # Cached plan after the upload's search.
            loading_result.set()
            assert finish_result.wait(5)
            phase('publish')
        return {'ok': True}

    with TestClient(app) as client, ThreadPoolExecutor() as executor:
        owner_future = executor.submit(client.post, '/api/uploads', files={'files':('run.fit',b'fixture','application/octet-stream')})
        assert owner_entered.wait(5)
        waiter_future = executor.submit(client.get, '/api/weekly-schedule', headers={'X-Planner-Request':request_id})
        try:
            assert waiter_entered.wait(5)
            status = client.get('/api/weekly-schedule/progress/' + request_id).json()
            assert status['state'] == 'running' and status['phase'] == 'search'
            assert status['following_active_job']
            assert status['estimated_percent'] == 49
            assert status['model_checks'] == 1 and status['budget_used'] == 5000
            release.set()
            assert owner_future.result().status_code == 200
            assert loading_result.wait(5)
            loading = client.get('/api/weekly-schedule/progress/' + request_id).json()
            assert loading['state'] == 'running' and loading['estimated_percent'] == 97
            assert loading['budget_used'] == 5000
            assert not loading['following_active_job']
        finally:
            release.set();finish_result.set()
        assert waiter_future.result().json() == {'ok': True}
        assert client.get('/api/weekly-schedule/progress/' + request_id).json()['estimated_percent'] == 100


def test_shared_jobs_are_isolated_and_new_search_does_not_reuse_old_counter():
    tracker = PlannerProgress()
    with tracker.provide(), planner_job('a'):
        phase('search');budget_progress(90,100);model_check()
        with tracker.observe('other'):
            follow_job('b')
            assert tracker.snapshot('other')['estimated_percent'] == 0
        with tracker.observe('waiter'):
            follow_job('a')
            assert tracker.snapshot('waiter')['estimated_percent'] == pytest.approx(81.8)
            with planner_job('a'):
                phase('history')
                assert tracker.snapshot('waiter')['budget_used'] == 90
                phase('search');budget_progress(10,100)
                snapshot = tracker.snapshot('waiter')
                assert snapshot['budget_used'] == 10 and snapshot['model_checks'] == 0
                assert snapshot['estimated_percent'] == pytest.approx(16.2)
            finished()


def test_completed_or_failed_shared_work_never_finishes_queued_request():
    tracker = PlannerProgress()
    with tracker.provide(), planner_job('a'):
        phase('search');budget_progress(100,100)
        with tracker.observe('waiter'):
            follow_job('a')
            owner = tracker._jobs['a']
            tracker.finish(owner, 'complete')
            assert tracker.snapshot('waiter')['state'] == 'running'
            assert tracker.snapshot('waiter')['estimated_percent'] == 97
            tracker.finish(owner, 'failed')
            assert tracker.snapshot('waiter')['state'] == 'running'
            finished(False)
            assert tracker.snapshot('waiter')['state'] == 'failed'


def test_disconnecting_owner_request_cannot_hide_search_from_next_refresh():
    tracker = PlannerProgress()
    with tracker.observe('original'):
        with planner_job('a'):
            phase('search');budget_progress(50,100)
            # Simulate cancellation of the HTTP observer while its synchronous
            # worker continues computing under the existing live planner lock.
            tracker.finish(tracker._work['original'], 'failed')
            budget_progress(75,100)
            assert tracker._jobs['a'].state == 'running'
            with tracker.observe('refresh'):
                follow_job('a')
                result = tracker.snapshot('refresh')
                assert result['state'] == 'running' and result['following_active_job']
                assert result['budget_used'] == 75
                assert result['estimated_percent'] == pytest.approx(69.5)
        assert tracker._jobs['a'].state == 'complete'
