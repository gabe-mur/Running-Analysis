from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from fastapi.testclient import TestClient
import pytest

from run_analysis.config import load_config
from run_analysis.db import connect, initialize
from run_analysis.plan_refresh import (
    mark_plan_refresh, pending_refresh, complete_plan_refresh, saved_plan_state,
)
from run_analysis.web.app import create_app
from run_analysis import recommendation_service as plans
from test_prescription_matching import _schedule, _threshold
from test_upload_noop import stub_downstream
from test_web_phase1 import _write_config
from test_web_upload import _tcx_bytes


def saved_schedule(db, when):
    schedule = _schedule(when, _threshold(when))
    db.execute('INSERT OR REPLACE INTO app_state VALUES (?,?,?)',
               ('weekly_schedule', schedule.model_dump_json(), when.isoformat()))
    db.commit()
    return schedule


def test_browser_upload_finishes_analysis_without_planning_and_persists_pending(tmp_path, monkeypatch):
    _write_config(tmp_path)
    calls = stub_downstream(monkeypatch)
    client = TestClient(create_app(tmp_path))
    result = client.post('/api/uploads', files={'files': ('run.tcx', _tcx_bytes())})
    assert result.status_code == 200
    assert any(name == 'fit_models' for name, _ in calls)
    assert any(name == 'match_activities_to_prescriptions' for name, _ in calls)
    assert not any(name == 'generate_weekly_schedule' for name, _ in calls)
    assert next(s['status'] for s in result.json()['stages'] if s['name'] == 'schedule') == 'deferred'
    with connect(tmp_path / 'data/test.sqlite') as db:
        revision = pending_refresh(db)
        assert revision
    # Duplicate uploads cannot clear the pending update or start a search.
    calls.clear()
    assert client.post('/api/uploads', files={'files': ('renamed.tcx', _tcx_bytes())}).status_code == 200
    assert calls == []
    with connect(tmp_path / 'data/test.sqlite') as db:
        assert pending_refresh(db) == revision
    # A new app instance sees the same persisted state; reading cannot plan.
    def forbidden(*args, **kwargs):
        pytest.fail('Saved-plan reader invoked the planner')
    monkeypatch.setattr(plans, 'generate_weekly_schedule', forbidden)
    monkeypatch.setattr('run_analysis.adaptive.live_bridge.schedule', forbidden)
    saved = TestClient(create_app(tmp_path)).get('/api/weekly-schedule/saved')
    assert saved.status_code == 200 and saved.json()['needs_refresh']
    assert client.get('/api/dashboard').json()['weekly_plan_needs_refresh']


def test_explicit_plan_open_consumes_pending_update_and_saved_reads_remain_read_only(tmp_path, monkeypatch):
    _write_config(tmp_path)
    when = datetime.now(ZoneInfo(load_config(tmp_path / 'config.yaml')['timezone_default']))
    with connect(tmp_path / 'data/test.sqlite') as db:
        initialize(db)
        schedule = saved_schedule(db, when)
        mark_plan_refresh(db)
    calls = []
    def build(db, config, request, root, **kwargs):
        calls.append('build')
        return schedule
    monkeypatch.setattr(plans, '_generate_weekly_schedule', build)
    client = TestClient(create_app(tmp_path))
    assert client.get('/api/weekly-schedule/saved').json()['needs_refresh']
    assert calls == []
    assert client.get('/api/weekly-schedule/latest').status_code == 200
    assert calls == ['build']
    assert not client.get('/api/weekly-schedule/saved').json()['needs_refresh']
    assert not client.get('/api/dashboard').json()['weekly_plan_needs_refresh']
    assert calls == ['build']
    with connect(tmp_path / 'data/test.sqlite') as db:
        assert pending_refresh(db) is None


@pytest.mark.parametrize('outcome', ['failure', 'review', 'newer_upload', 'success'])
def test_pending_revision_clears_only_after_success_on_same_inputs(tmp_path, outcome):
    _write_config(tmp_path)
    with connect(tmp_path / 'data/test.sqlite') as db:
        initialize(db)
        mark_plan_refresh(db)
        before = pending_refresh(db)
        def build():
            if outcome == 'failure':
                raise ValueError('Planner failed')
            if outcome == 'newer_upload':
                mark_plan_refresh(db)
            return SimpleNamespace(planner_status='review_required' if outcome == 'review' else 'ready')
        if outcome == 'failure':
            with pytest.raises(ValueError):
                complete_plan_refresh(db, build)
        else:
            complete_plan_refresh(db, build)
        after = pending_refresh(db)
        assert (after is None) == (outcome == 'success')
        if outcome == 'newer_upload':
            assert after != before


def test_saved_plan_is_stale_after_rollover_even_without_upload(tmp_path):
    _write_config(tmp_path)
    cfg = load_config(tmp_path / 'config.yaml')
    when = datetime(2026, 10, 12, 18, tzinfo=timezone.utc)
    with connect(tmp_path / 'data/test.sqlite') as db:
        initialize(db)
        saved_schedule(db, when)
        before = db.total_changes
        assert not saved_plan_state(db, cfg, as_of=when)['needs_refresh']
        assert saved_plan_state(db, cfg, as_of=when + timedelta(days=1))['needs_refresh']
        assert db.total_changes == before


def test_saved_plan_and_pending_marker_share_a_snapshot_during_publication(tmp_path, monkeypatch):
    _write_config(tmp_path)
    cfg = load_config(tmp_path / 'config.yaml')
    when = datetime(2026, 10, 12, 18, tzinfo=timezone.utc)
    path = tmp_path / 'data/test.sqlite'
    with connect(path) as reader, connect(path) as writer:
        initialize(reader)
        saved_schedule(reader, when)
        mark_plan_refresh(reader)
        load = plans.load_latest_weekly_schedule

        def concurrent_publication(db):
            schedule = load(db)
            assert db.in_transaction
            writer.execute("DELETE FROM app_state WHERE key='weekly_schedule_pending_refresh'")
            writer.commit()
            return schedule

        monkeypatch.setattr(plans, 'load_latest_weekly_schedule', concurrent_publication)
        # A marker cleared after the old plan was read cannot label that old
        # response current. The next request sees the completed publication.
        assert saved_plan_state(reader, cfg, as_of=when)['needs_refresh']
        assert not reader.in_transaction
        assert pending_refresh(reader) is None
        monkeypatch.setattr(plans, 'load_latest_weekly_schedule', load)
        assert not saved_plan_state(reader, cfg, as_of=when)['needs_refresh']
