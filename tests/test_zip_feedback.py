from datetime import datetime, timedelta, timezone
from io import BytesIO
import stat
import zipfile

import pytest
from fastapi.testclient import TestClient

from fit_builder import FitWriter, UINT8, UINT32, ENUM, fit_timestamp
from test_web_phase1 import _write_config
from test_web_upload import _tcx_bytes
from run_analysis.config import load_config
from run_analysis.db import connect
from run_analysis.metadata_service import update_run_metadata
from run_analysis.overrides import sync_overrides
from run_analysis.run_feedback import get_run_feedback, list_runs
from run_analysis.web.app import create_app
from run_analysis.web.schemas import RunMetadataPatch
from run_analysis.web.upload_archive import fit_members
from run_analysis.web.upload_service import UploadPayload, run_upload_pipeline

START = datetime(2024, 7, 1, 12, tzinfo=timezone.utc)


def rated_fit(rpe=40, feel=75):
    writer = FitWriter()
    for i in range(3):
        writer.message(20, {253: (UINT32, fit_timestamp(START + timedelta(seconds=10*i))),
                            3: (UINT8, 145), 5: (UINT32, 2000*i)})
    writer.message(18, {2: (UINT32, fit_timestamp(START)), 5: (ENUM, 1),
                       7: (UINT32, 20000), 8: (UINT32, 20000), 9: (UINT32, 4000),
                       192: (UINT8, feel), 193: (UINT8, rpe)})
    return writer.build()


def archive(entries):
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for name, data in entries:
            z.writestr(name, data)
    return buffer.getvalue()


def upload(root, name, data):
    return run_upload_pipeline(root, 'config.yaml', [UploadPayload(name, data)])


def test_zip_endpoint_decodes_binary_ratings_and_repeated_upload_is_idempotent(tmp_path):
    _write_config(tmp_path)
    client = TestClient(create_app(tmp_path))
    payload = archive([('export/ACTIVITY.FIT', rated_fit()), ('info.txt', b'ignored')])
    response = client.post('/api/uploads', files=[('files', ('garmin.zip', payload, 'application/zip'))])
    assert response.status_code == 200, response.text
    run_id = response.json()['primary_activity_id']
    with connect(tmp_path/'data/test.sqlite') as db:
        feedback = get_run_feedback(db, load_config(tmp_path/'config.yaml'), run_id)
        assert feedback.metadata.perceived_exertion == 4
        assert feedback.metadata.workout_feel == 75
        assert feedback.metadata.effort_source.startswith('fit:')
        assert list_runs(db)[0].session_difficulty.perceived_exertion == 4
        assert db.execute('SELECT count(*) FROM run_overrides').fetchone()[0] == 0
    repeat = upload(tmp_path, 'garmin.zip', payload)
    assert repeat.files[0].status == 'unchanged'
    direct = upload(tmp_path, 'ACTIVITY.FIT', rated_fit())
    assert direct.files[0].status == 'unchanged'
    with connect(tmp_path/'data/test.sqlite') as db:
        assert db.execute('SELECT count(*) FROM activities').fetchone()[0] == 1
        assert db.execute('SELECT count(*) FROM activity_sensor_streams').fetchone()[0] == 1


@pytest.mark.parametrize('fit_first', [False, True])
def test_tcx_fit_order_preserves_one_activity_and_manual_edits(tmp_path, fit_first):
    _write_config(tmp_path)
    sources = [('run.tcx', _tcx_bytes()), ('garmin.zip', archive([('run.fit', rated_fit())]))]
    if fit_first:
        sources.reverse()
    first = upload(tmp_path, *sources[0])
    run_id = first.primary_activity_id
    with connect(tmp_path/'data/test.sqlite') as db:
        update_run_metadata(db, tmp_path/'run_overrides.csv', run_id,
                            RunMetadataPatch(perceived_exertion=7.5, notes='keep me'))
    second = upload(tmp_path, *sources[1])
    assert second.files[0].activity_ids == [run_id]
    with connect(tmp_path/'data/test.sqlite') as db:
        assert db.execute('SELECT count(*) FROM activities').fetchone()[0] == 1
        config = load_config(tmp_path/'config.yaml')
        feedback = get_run_feedback(db, config, run_id)
        assert feedback.metadata.perceived_exertion == 7.5
        assert feedback.metadata.effort_source == 'manual'
        assert feedback.metadata.notes == 'keep me'
        assert feedback.metadata.workout_feel == 75
        # Clearing an override reverts to device evidence, not a fabricated zero.
        update_run_metadata(db, tmp_path/'run_overrides.csv', run_id,
                            RunMetadataPatch(perceived_exertion=None))
        assert get_run_feedback(db, config, run_id).metadata.perceived_exertion == 4
        assert not db.execute('PRAGMA foreign_key_check').fetchall()


@pytest.mark.parametrize('rpe,feel,expected', [(45, 0, 4.5), (0, 100, 0), (255, 255, None), (101, 101, None)])
def test_rating_scale_missing_invalid_and_zero(tmp_path, rpe, feel, expected):
    _write_config(tmp_path)
    run_id = upload(tmp_path, 'run.fit', rated_fit(rpe, feel)).primary_activity_id
    with connect(tmp_path/'data/test.sqlite') as db:
        result = get_run_feedback(db, load_config(tmp_path/'config.yaml'), run_id)
        assert result.metadata.perceived_exertion == expected
        assert result.metadata.workout_feel == (feel if feel <= 100 else None)
        if expected == 0:
            assert result.run.session_difficulty.session_rpe_load == 0
        update_run_metadata(db, tmp_path/'ratings.csv', run_id, RunMetadataPatch(perceived_exertion=0))
        sync_overrides(db, tmp_path/'ratings.csv')
        assert get_run_feedback(db, load_config(tmp_path/'config.yaml'), run_id).metadata.perceived_exertion == 0


@pytest.mark.parametrize('name', ['../evil.fit', '/evil.fit', 'C:\\evil.fit', 'safe/../../evil.fit'])
def test_unsafe_zip_rejected_before_any_upload_write(tmp_path, name):
    _write_config(tmp_path)
    with pytest.raises(ValueError, match='Unsafe'):
        upload(tmp_path, 'run.zip', archive([(name, rated_fit())]))
    assert not (tmp_path/'uploads').exists()


def test_bad_empty_nested_and_symlink_archives():
    for content in (b'not zip', archive([('run.tcx', b'xml')]), archive([('run.zip', b'zip')]), archive([('run.fit', b'')])):
        with pytest.raises(ValueError):
            fit_members(content)
    info = zipfile.ZipInfo('link.fit')
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with pytest.raises(ValueError, match='symbolic'):
        fit_members(archive([(info, b'/tmp/target')]))


def test_member_and_expansion_limits(monkeypatch):
    import run_analysis.web.upload_archive as module
    with pytest.raises(ValueError, match='too many'):
        fit_members(archive([(f'{i}.txt', b'') for i in range(101)]))
    monkeypatch.setattr(module, 'MAX_EXPANDED_BYTES', 30)
    with pytest.raises(ValueError, match='expanded'):
        fit_members(archive([('run.fit', b'x'*31)]))


def test_multi_fit_zip_and_ignored_tcx_are_not_double_counted(tmp_path):
    _write_config(tmp_path)
    result = upload(tmp_path, 'runs.zip', archive([
        ('one.fit', rated_fit()), ('two.fit', rated_fit()),
        ('copy.tcx', _tcx_bytes()), ('__MACOSX/._one.fit', b'metadata'),
    ]))
    assert len(result.files) == 2
    # The second member has exactly the same bytes: reuse its completed import
    # rather than add provenance or request downstream work a second time.
    assert {f.status for f in result.files} == {'imported', 'unchanged'}
    with connect(tmp_path/'data/test.sqlite') as db:
        assert db.execute('SELECT count(*) FROM activities').fetchone()[0] == 1


def test_corrupt_fit_in_valid_zip_is_reported_and_other_run_survives(tmp_path):
    _write_config(tmp_path)
    result = upload(tmp_path, 'runs.zip', archive([('bad.fit', b'bad'), ('good.fit', rated_fit())]))
    assert result.files[0].status == 'failed'
    assert result.files[1].status == 'imported'
    assert result.primary_activity_id == result.files[1].activity_ids[0]


def test_conflicting_device_sources_are_visible(tmp_path):
    _write_config(tmp_path)
    run_id = upload(tmp_path, 'one.fit', rated_fit(40)).primary_activity_id
    upload(tmp_path, 'two.fit', rated_fit(60))
    with connect(tmp_path/'data/test.sqlite') as db:
        result = get_run_feedback(db, load_config(tmp_path/'config.yaml'), run_id)
        assert result.metadata.feedback_conflict
        assert result.metadata.perceived_exertion == 4  # canonical source wins
