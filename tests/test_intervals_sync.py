from datetime import timedelta
import json

from fastapi.testclient import TestClient
import pytest

from fit_builder import build_run
from run_analysis import intervals_connection as remote
from run_analysis import intervals_sync as sync
from run_analysis.db import connect
from run_analysis.web.app import create_app
from run_analysis.web.upload_service import upload_operation
from test_sensor_streams import START, make_rr_fit
from test_upload_noop import seed, snapshot, stub_downstream
from test_web_phase1 import _write_config
from test_web_upload import _tcx_bytes
from test_zip_feedback import rated_fit

URL = '/api/connections/intervals/sync'
HEADERS = {'X-Run-Analysis-Local': '1'}


@pytest.fixture(autouse=True)
def no_unmocked_wellness_network(monkeypatch):
    monkeypatch.setattr(remote,'list_wellness',lambda root,**kwargs:[])


def remote_files(monkeypatch, recordings):
    calls = []
    monkeypatch.setattr(remote, 'list_recent_activities', lambda root: [
        {'id': identifier, 'type': 'Run', 'file_type': 'fit', 'start_date_local': identifier}
        for identifier in recordings])
    def download(root, identifier):
        calls.append(identifier)
        value = recordings[identifier]
        if isinstance(value, Exception):
            raise value
        return value
    monkeypatch.setattr(remote, 'download_original_fit', download)
    return calls


def test_sync_imports_new_original_once_and_reuses_manual_upload(tmp_path, monkeypatch):
    existing = rated_fit()
    seed(tmp_path, 'manual.fit', existing)
    new = make_rr_fit(tmp_path / 'new.fit').read_bytes()
    downloads = remote_files(monkeypatch, {'i10': existing, 'i20': new})
    downstream = stub_downstream(monkeypatch)
    result = sync.sync_runs(tmp_path, tmp_path / 'config.yaml')
    assert result['new_runs'] == 1 and result['downloaded_files'] == 2
    assert [f['status'] for f in result['upload']['files']] == ['unchanged', 'imported']
    assert not any(name == 'generate_weekly_schedule' for name, _ in downstream)
    assert next(stage['status'] for stage in result['upload']['stages'] if stage['name'] == 'schedule') == 'deferred'
    with connect(tmp_path / 'data/test.sqlite') as db:
        assert db.execute('SELECT count(*) FROM activities').fetchone()[0] == 2
        rr = db.execute('SELECT payload_json FROM activity_sensor_streams ORDER BY source_file_id DESC LIMIT 1').fetchone()[0]
        assert json.loads(rr)['rr_batches'][0]['intervals_seconds'] == [.4, .8, None]
    before = snapshot(tmp_path)
    downstream.clear()
    result = sync.sync_runs(tmp_path, tmp_path / 'config.yaml')
    assert result['already_synced'] == 2 and result['new_runs'] == 0
    assert result['upload'] is None and result['downloaded_files'] == 0
    assert downloads == ['i10', 'i20'] and downstream == []
    assert snapshot(tmp_path) == before
    cache = tmp_path / 'data/intervals-sync.json'
    assert cache.stat().st_mode & 0o777 == 0o600


def test_first_synced_fit_enriches_existing_tcx(tmp_path, monkeypatch):
    seed(tmp_path, 'manual.tcx', _tcx_bytes())
    remote_files(monkeypatch, {'i10': rated_fit()})
    downstream = stub_downstream(monkeypatch)
    result = sync.sync_runs(tmp_path, tmp_path / 'config.yaml')
    assert result['new_runs'] == 0
    assert result['upload']['files'][0]['status'] == 'duplicate'
    assert any(name == 'process_activities' for name, _ in downstream)
    assert not any(name == 'generate_weekly_schedule' for name, _ in downstream)
    with connect(tmp_path / 'data/test.sqlite') as db:
        sensor = json.loads(db.execute('SELECT payload_json FROM activity_sensor_streams').fetchone()[0])
        assert sensor['session_fields']['workout_rpe'] == 40


def test_empty_history_no_key_and_cross_site_requests_do_not_import(tmp_path, monkeypatch):
    client = TestClient(create_app(tmp_path))
    assert client.post(URL, headers=HEADERS).status_code == 400
    assert not list(tmp_path.rglob('*.sqlite'))
    def forbidden(*args):
        pytest.fail('Cross-site sync contacted remote service')
    monkeypatch.setattr(remote, 'list_recent_activities', forbidden)
    assert client.post(URL).status_code == 403
    assert client.post(URL, headers={**HEADERS, 'Origin': 'https://other.example'}).status_code == 403
    assert client.post(URL, headers={**HEADERS, 'Sec-Fetch-Site': 'cross-site'}).status_code == 403
    monkeypatch.setattr(remote, 'list_recent_activities', lambda root: [])
    result = client.post(URL, headers=HEADERS)
    assert result.status_code == 200 and result.json()['new_runs'] == 0
    assert not list(tmp_path.rglob('*.sqlite'))


def test_upload_and_sync_share_busy_guard_and_release_after_failure(tmp_path, monkeypatch):
    _write_config(tmp_path)
    client = TestClient(create_app(tmp_path))
    with upload_operation(tmp_path):
        assert client.post(URL, headers=HEADERS).status_code == 409
        assert client.post('/api/uploads', files={'files': ('run.tcx', _tcx_bytes())}).status_code == 409
    monkeypatch.setattr(remote, 'list_recent_activities', lambda root: [])
    assert client.post(URL, headers=HEADERS).status_code == 200


def test_partial_remote_failure_retries_only_unsuccessful_file(tmp_path, monkeypatch):
    seed(tmp_path, 'manual.fit', rated_fit())
    downloads = remote_files(monkeypatch, {'i10': rated_fit(), 'i20': remote.ConnectionProblem('Original file not ready.')})
    downstream = stub_downstream(monkeypatch)
    result = sync.sync_runs(tmp_path, tmp_path / 'config.yaml')
    assert result['errors'] == ['Original file not ready.']
    assert result['new_runs'] == 0 and downstream == []
    sync.sync_runs(tmp_path, tmp_path / 'config.yaml')
    assert downloads == ['i10', 'i20', 'i20']


def test_batch_limits_continue_on_next_click_not_repeat_first_batch(tmp_path, monkeypatch):
    seed(tmp_path, 'manual.fit', rated_fit())
    other = build_run(tmp_path / 'other.fit', start=START + timedelta(days=2))
    downloads = remote_files(monkeypatch, {'i10': rated_fit(), 'i20': other})
    stub_downstream(monkeypatch)
    monkeypatch.setattr(sync, 'MAX_SYNC_FILES', 1)
    first = sync.sync_runs(tmp_path, tmp_path / 'config.yaml')
    assert first['remaining_runs'] == 1 and downloads == ['i10']
    second = sync.sync_runs(tmp_path, tmp_path / 'config.yaml')
    assert second['remaining_runs'] == 0 and second['new_runs'] == 1
    assert downloads == ['i10', 'i20']


def test_sync_cache_does_not_hide_missing_sensor_backfill(tmp_path, monkeypatch):
    seed(tmp_path, 'manual.fit', rated_fit())
    downloads = remote_files(monkeypatch, {'i10': rated_fit()})
    calls = stub_downstream(monkeypatch)
    sync.sync_runs(tmp_path, tmp_path / 'config.yaml')
    with connect(tmp_path / 'data/test.sqlite') as db:
        db.execute('DELETE FROM activity_sensor_streams')
        db.commit()
    sync.sync_runs(tmp_path, tmp_path / 'config.yaml')
    assert downloads == ['i10', 'i10']
    assert any(name == 'process_activities' for name, _ in calls)
    assert not any(name == 'generate_weekly_schedule' for name, _ in calls)


def test_failed_parse_not_recorded_as_successful_sync(tmp_path, monkeypatch):
    _write_config(tmp_path)
    downloads = remote_files(monkeypatch, {'i10': b'not fit'})
    calls = stub_downstream(monkeypatch)
    for _ in range(2):
        result = sync.sync_runs(tmp_path, tmp_path / 'config.yaml')
        assert result['upload']['files'][0]['status'] == 'failed'
        assert result['new_runs'] == 0
    assert downloads == ['i10', 'i10'] and calls == []
    assert not (tmp_path / 'data/intervals-sync.json').exists()


def test_nonrunning_nonfit_and_duplicate_remote_rows(tmp_path, monkeypatch):
    seed(tmp_path, 'manual.fit', rated_fit())
    downloads = remote_files(monkeypatch, {'i10': rated_fit()})
    monkeypatch.setattr(remote, 'list_recent_activities', lambda root: [
        {'id': 'i10', 'type': 'Run', 'file_type': 'fit'},
        {'id': 'i10', 'type': 'Run', 'file_type': 'fit'},
        {'id': 'i20', 'type': 'Ride', 'file_type': 'fit'},
        {'id': 'i30', 'type': 'Run', 'file_type': 'tcx'},
    ])
    stub_downstream(monkeypatch)
    result = sync.sync_runs(tmp_path, tmp_path / 'config.yaml')
    assert result['checked_runs'] == 2 and downloads == ['i10']
    assert len(result['errors']) == 1


def test_corrupt_cache_only_costs_redownload_and_not_replanning(tmp_path, monkeypatch):
    seed(tmp_path, 'manual.fit', rated_fit())
    downloads = remote_files(monkeypatch, {'i10': rated_fit()})
    calls = stub_downstream(monkeypatch)
    (tmp_path / 'data/intervals-sync.json').write_text('invalid')
    result = sync.sync_runs(tmp_path, tmp_path / 'config.yaml')
    assert result['new_runs'] == 0 and downloads == ['i10'] and calls == []


def test_remote_invalid_id_rejected_before_any_download(tmp_path, monkeypatch):
    downloads = remote_files(monkeypatch, {})
    monkeypatch.setattr(remote, 'list_recent_activities', lambda root: [{'id': '../bad', 'type': 'Run'}])
    result = TestClient(create_app(tmp_path)).post(URL, headers=HEADERS)
    assert result.status_code == 400 and downloads == []
    assert '../bad' not in result.text


def test_wellness_failure_does_not_fail_activity_sync_or_call_planner(tmp_path,monkeypatch):
    monkeypatch.setattr(remote,'list_recent_activities',lambda root:[])
    def fail(*args,**kwargs):raise remote.ConnectionProblem('Remote unavailable')
    monkeypatch.setattr(remote,'list_wellness',fail)
    result=sync.sync_runs(tmp_path,tmp_path/'config.yaml')
    assert result['new_runs']==0 and result['errors']==[]
    assert result['wellness']['status']=='unavailable'
