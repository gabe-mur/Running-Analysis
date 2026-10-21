"""Upload orchestration must distinguish accepted content from new evidence."""
from dataclasses import dataclass
from datetime import date
from io import BytesIO
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

from run_analysis.db import connect
from run_analysis.importer import import_files
from run_analysis.web import upload_service as service
from run_analysis.web.upload_service import UploadPayload
from test_sensor_streams import make_rr_fit
from test_web_phase1 import _write_config
from test_web_upload import _tcx_bytes


@dataclass
class Summary:
    failures: int = 0


def stub_downstream(monkeypatch):
    calls = []
    for name in ('sync_overrides', 'process_activities', 'update_weather', 'fit_models',
                 'load_current_status', 'generate_weekly_schedule', 'match_activities_to_prescriptions'):
        def stub(*args, _name=name, **kwargs):
            calls.append((_name, args[1] if _name == 'match_activities_to_prescriptions' else None))
            if _name == 'match_activities_to_prescriptions':
                return []
            if _name == 'load_current_status':
                return SimpleNamespace(health_status='normal')
            if _name == 'generate_weekly_schedule':
                return SimpleNamespace(end_date=date(2026, 10, 27))
            return Summary()
        monkeypatch.setattr(service, name, stub)
    return calls


def seed(root, name, content):
    _write_config(root)
    path = service._persist_upload(root, UploadPayload(name, content))
    with connect(root / 'data/test.sqlite') as db:
        summary = import_files(db, root, 'America/New_York', paths=[path])
        assert summary.activities_added == 1
        return path, summary.evidence_changed_activity_ids[0]


def snapshot(root):
    with connect(root / 'data/test.sqlite') as db:
        return tuple(db.iterdump())


def assert_noop(result, calls):
    assert calls == []
    assert all(item.status == 'unchanged' for item in result.files)
    assert next(s.status for s in result.stages if s.name == 'import') == 'complete'
    assert all(s.status == 'skipped' for s in result.stages if s.name in ('process', 'weather', 'model', 'schedule'))


@pytest.mark.parametrize('fit', [False, True])
@pytest.mark.parametrize('rename', [False, True])
def test_exact_upload_is_noop_even_with_new_filename(tmp_path, monkeypatch, fit, rename):
    content = make_rr_fit(tmp_path / 'fixture.fit').read_bytes() if fit else _tcx_bytes()
    name = 'run.fit' if fit else 'run.tcx'
    _, activity_id = seed(tmp_path, name, content)
    before = snapshot(tmp_path)
    calls = stub_downstream(monkeypatch)
    result = service.run_upload_pipeline(tmp_path, 'config.yaml', [
        UploadPayload(('intervals-' if rename else '') + name, content)])
    assert_noop(result, calls)
    assert result.primary_activity_id == activity_id
    assert snapshot(tmp_path) == before
    assert len(list((tmp_path / 'uploads').iterdir())) == 1


def test_identical_zip_member_is_noop(tmp_path, monkeypatch):
    content = make_rr_fit(tmp_path / 'fixture.fit').read_bytes()
    seed(tmp_path, 'manual.fit', content)
    buffer = BytesIO()
    with ZipFile(buffer, 'w') as archive:
        archive.writestr('garmin/new-name.fit', content)
    calls = stub_downstream(monkeypatch)
    before = snapshot(tmp_path)
    result = service.run_upload_pipeline(tmp_path, 'config.yaml', [UploadPayload('export.zip', buffer.getvalue())])
    assert_noop(result, calls)
    assert snapshot(tmp_path) == before


def test_mixed_batch_processes_once_and_matches_only_new_evidence(tmp_path, monkeypatch):
    old = _tcx_bytes()
    _, old_id = seed(tmp_path, 'old.tcx', old)
    calls = stub_downstream(monkeypatch)
    new = _tcx_bytes(start='2024-07-03T12:00:00Z', end='2024-07-03T12:00:20Z')
    result = service.run_upload_pipeline(tmp_path, 'config.yaml', [
        UploadPayload('renamed.tcx', old), UploadPayload('new.tcx', new),
        UploadPayload('second-copy.tcx', new)])
    assert [item.status for item in result.files] == ['unchanged', 'imported', 'unchanged']
    matches = [ids for name, ids in calls if name == 'match_activities_to_prescriptions']
    assert matches == [result.files[1].activity_ids]
    assert old_id not in matches[0]
    assert sum(name == 'generate_weekly_schedule' for name, _ in calls) == 1


def test_unchanged_fit_missing_sensors_still_processes_backfill(tmp_path, monkeypatch):
    content = make_rr_fit(tmp_path / 'fixture.fit').read_bytes()
    _, activity_id = seed(tmp_path, 'run.fit', content)
    with connect(tmp_path / 'data/test.sqlite') as db:
        db.execute('DELETE FROM activity_sensor_streams')
        db.commit()
    calls = stub_downstream(monkeypatch)
    result = service.run_upload_pipeline(tmp_path, 'config.yaml', [UploadPayload('run.fit', content)])
    assert result.files[0].status == 'unchanged'
    assert ('match_activities_to_prescriptions', [activity_id]) in calls
    assert next(s.status for s in result.stages if s.name == 'schedule') == 'complete'
    calls.clear()
    assert_noop(service.run_upload_pipeline(tmp_path, 'config.yaml', [UploadPayload('run.fit', content)]), calls)


def test_passive_v3_backfill_through_upload_does_not_refit_or_replan(tmp_path,monkeypatch):
    import json
    content=make_rr_fit(tmp_path/'fixture.fit').read_bytes()
    seed(tmp_path,'run.fit',content)
    with connect(tmp_path/'data/test.sqlite') as db:
        original=[tuple(r) for r in db.execute('SELECT * FROM activities')]
        old=json.loads(db.execute('SELECT payload_json FROM activity_sensor_streams').fetchone()[0])
        old.pop('performance_records');old.pop('performance_session_fields');old['extraction_version']=2
        db.execute('UPDATE activity_sensor_streams SET extraction_version=2,payload_json=?',(json.dumps(old),))
        db.commit()
    calls=stub_downstream(monkeypatch)
    result=service.run_upload_pipeline(tmp_path,'config.yaml',[UploadPayload('run.fit',content)])
    assert_noop(result,calls)
    with connect(tmp_path/'data/test.sqlite') as db:
        assert [tuple(r) for r in db.execute('SELECT * FROM activities')]==original
        assert db.execute('SELECT extraction_version FROM activity_sensor_streams').fetchone()[0]==3


def test_first_fit_after_tcx_is_new_evidence_without_another_run(tmp_path, monkeypatch):
    from test_tcx import make_tcx
    content = make_rr_fit(tmp_path / 'fixture.fit').read_bytes()
    tcx = make_tcx(tmp_path / 'fixture.tcx', start='2026-09-24T00:00:00Z',
                   end='2026-09-24T00:00:20Z', activity_id='2026-09-24T00:00:00Z')
    tcx_content = tcx.read_bytes().replace(b'>40</DistanceMeters>', b'>60</DistanceMeters>')
    _, activity_id = seed(tmp_path, 'run.tcx', tcx_content)
    calls = stub_downstream(monkeypatch)
    result = service.run_upload_pipeline(tmp_path, 'config.yaml', [UploadPayload('run.fit', content)])
    assert result.files[0].status == 'duplicate'
    assert ('match_activities_to_prescriptions', [activity_id]) in calls
    with connect(tmp_path / 'data/test.sqlite') as db:
        assert db.execute('SELECT count(*) FROM activities').fetchone()[0] == 1
        assert db.execute('SELECT count(*) FROM activity_sensor_streams').fetchone()[0] == 1


def test_richer_recording_for_existing_run_still_processes(tmp_path, monkeypatch):
    _, activity_id = seed(tmp_path, 'plain.tcx', _tcx_bytes())
    calls = stub_downstream(monkeypatch)
    result = service.run_upload_pipeline(tmp_path, 'config.yaml', [UploadPayload('gps.tcx', _tcx_bytes(gps=True))])
    assert result.files[0].status == 'duplicate'
    assert ('match_activities_to_prescriptions', [activity_id]) in calls


def test_changed_fit_feedback_still_processes_existing_run(tmp_path, monkeypatch):
    from fit_builder import FitWriter, UINT8, UINT32, ENUM, fit_timestamp
    from test_sensor_streams import START
    from datetime import timedelta
    def recording(rpe):
        writer = FitWriter()
        for i in range(3):
            writer.message(20, {253: (UINT32, fit_timestamp(START + timedelta(seconds=10*i))),
                                3: (UINT8, 150), 5: (UINT32, 3000*i)})
        writer.message(18, {2: (UINT32, fit_timestamp(START)), 5: (ENUM, 1),
                           7: (UINT32, 20000), 8: (UINT32, 20000), 9: (UINT32, 6000),
                           193: (UINT8, rpe)})
        return writer.build()
    _, activity_id = seed(tmp_path, 'run.fit', recording(40))
    calls = stub_downstream(monkeypatch)
    result = service.run_upload_pipeline(tmp_path, 'config.yaml', [UploadPayload('report.fit', recording(70))])
    assert result.files[0].status == 'duplicate'
    assert ('match_activities_to_prescriptions', [activity_id]) in calls


def test_gzip_repack_with_identical_sensor_evidence_skips_downstream(tmp_path, monkeypatch):
    import gzip
    content = make_rr_fit(tmp_path / 'fixture.fit').read_bytes()
    seed(tmp_path, 'run.fit', content)
    calls = stub_downstream(monkeypatch)
    result = service.run_upload_pipeline(tmp_path, 'config.yaml', [UploadPayload('download.fit.gz', gzip.compress(content))])
    assert result.files[0].status == 'duplicate'
    assert calls == []
    assert next(s.status for s in result.stages if s.name == 'schedule') == 'skipped'


def test_failed_content_is_not_accepted_as_unchanged_on_retry(tmp_path, monkeypatch):
    _write_config(tmp_path)
    calls = stub_downstream(monkeypatch)
    for _ in range(2):
        result = service.run_upload_pipeline(tmp_path, 'config.yaml', [UploadPayload('bad.tcx', b'not xml')])
        assert result.files[0].status == 'failed'
        assert next(s.status for s in result.stages if s.name == 'import') == 'failed'
    assert calls == []
