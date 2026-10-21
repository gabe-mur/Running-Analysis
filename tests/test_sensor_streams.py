from datetime import datetime, timedelta, timezone
import json
import struct
import pytest

from fit_builder import FitWriter, UINT8, UINT32, ENUM, fit_timestamp
from run_analysis.db import connect, initialize, SCHEMA_VERSION
from run_analysis.fit import parse_fit
from run_analysis.importer import import_files
from test_tcx import make_tcx

START = datetime(2026, 9, 24, tzinfo=timezone.utc)


def make_rr_fit(path):
    writer = FitWriter()
    for i in range(3):
        writer.message(20, {253: (UINT32, fit_timestamp(START+timedelta(seconds=10*i))),
                            3: (UINT8, 150), 5: (UINT32, 3000*i)})
        # Real binary HRV message (global 78), uint16 array, milliseconds.
        writer._data += bytes([0x4F])+struct.pack('<BBHB', 0, 0, 78, 1)+bytes([0, 6, 0x84])
        writer._data += bytes([0x0F])+struct.pack('<HHH', 400, 800, 0xFFFF)
    writer.message(18, {2: (UINT32, fit_timestamp(START)), 5: (ENUM, 1),
                       7: (UINT32, 20000), 8: (UINT32, 20000), 9: (UINT32, 6000)})
    path.write_bytes(writer.build())
    return path


def test_rr_units_padding_and_anchors_are_preserved_without_correction(tmp_path):
    activity = parse_fit(make_rr_fit(tmp_path/'run.fit')).activities[0]
    stream = activity.sensor_stream
    assert stream.dfa_status == 'deferred'
    assert len(stream.rr_batches) == 3
    first, last = stream.rr_batches[0], stream.rr_batches[-1]
    assert first.intervals_seconds == (.4, .8, None)
    assert first.preceding_record_utc == START
    assert first.following_record_utc == START+timedelta(seconds=10)
    assert last.following_record_utc is None
    assert first.message_index < last.message_index


def test_duplicate_sources_keep_separate_rr_streams_without_duplicate_work(tmp_path):
    a = make_rr_fit(tmp_path/'a.fit')
    b = make_rr_fit(tmp_path/'b.fit')
    with connect(tmp_path/'test.sqlite') as db:
        summary = import_files(db, tmp_path, 'UTC', paths=[a, b])
        assert summary.activities_added == 1
        assert db.execute('SELECT count(*) FROM activity_sensor_streams').fetchone()[0] == 2
        before = db.total_changes
        initialize(db)
        assert db.total_changes == before
        payload = json.loads(db.execute('SELECT payload_json FROM activity_sensor_streams LIMIT 1').fetchone()[0])
        assert payload['rr_batches'][0]['intervals_seconds'] == [.4, .8, None]
        assert not db.execute('PRAGMA foreign_key_check').fetchall()
        import_files(db, tmp_path, 'UTC', paths=[a, b])
        assert db.execute('SELECT count(*) FROM activity_sensor_streams').fetchone()[0] == 2


def test_v16_upgrade_and_sensor_backfill_preserve_canonical_row(tmp_path):
    path = make_rr_fit(tmp_path/'run.fit')
    with connect(tmp_path/'test.sqlite') as db:
        import_files(db, tmp_path, 'UTC', paths=[path])
        original = dict(db.execute('SELECT * FROM activities').fetchone())
        db.execute('DROP TABLE activity_sensor_streams')
        db.execute("UPDATE schema_metadata SET value='16' WHERE key='schema_version'")
        db.commit()
        initialize(db)
        assert db.execute("SELECT value FROM schema_metadata WHERE key='schema_version'").fetchone()[0] == str(SCHEMA_VERSION)
        summary = import_files(db, tmp_path, 'UTC', paths=[path])
        assert summary.unchanged_files == 1
        assert summary.failed_files == 0
        assert dict(db.execute('SELECT * FROM activities').fetchone()) == original
        assert db.execute('SELECT count(*) FROM activity_sensor_streams').fetchone()[0] == 1


def test_rr_does_not_change_canonical_quality_rank(tmp_path):
    from run_analysis.importer import _activity_quality_key
    activity = parse_fit(make_rr_fit(tmp_path/'run.fit')).activities[0]
    before = _activity_quality_key(activity)
    activity.sensor_stream = None
    assert _activity_quality_key(activity) == before


def test_performance_fields_preserve_fit_units_and_zero_without_changing_trackpoints(tmp_path):
    from fit_builder import UINT16
    writer=FitWriter()
    for i in range(3):
        writer.message(20,{253:(UINT32,fit_timestamp(START+timedelta(seconds=i*10))),
            3:(UINT8,150),5:(UINT32,3000*i),7:(UINT16,0 if i==0 else 210),
            39:(UINT16,825),41:(UINT16,2450),85:(UINT16,10200)})
    writer.message(18,{2:(UINT32,fit_timestamp(START)),5:(ENUM,1),7:(UINT32,20000),
        8:(UINT32,20000),9:(UINT32,6000),20:(UINT16,205),34:(UINT16,212)})
    source=tmp_path/'extra.fit';source.write_bytes(writer.build())
    activity=parse_fit(source).activities[0];stream=activity.sensor_stream
    assert stream.extraction_version==3
    assert stream.performance_records[0]['fields']['power']==dict(value=0,units='watts')
    assert stream.performance_records[1]['fields']['vertical_oscillation']==dict(value=82.5,units='mm')
    assert stream.performance_records[1]['fields']['stance_time']==dict(value=245.,units='ms')
    assert stream.performance_records[1]['fields']['step_length']==dict(value=1020.,units='mm')
    assert stream.performance_session_fields['normalized_power']==dict(value=212,units='watts')
    assert activity.trackpoints[1].distance_m==30


def test_passive_sensor_backfill_does_not_rewrite_activity_or_dirty_planner_evidence(tmp_path):
    source=make_rr_fit(tmp_path/'run.fit')
    with connect(tmp_path/'test.sqlite') as db:
        import_files(db,tmp_path,'UTC',paths=[source])
        original=dict(db.execute('SELECT * FROM activities').fetchone())
        row=db.execute('SELECT * FROM activity_sensor_streams').fetchone()
        old=json.loads(row['payload_json'])
        for key in ('performance_records','performance_session_fields'):old.pop(key)
        old['extraction_version']=2
        db.execute('UPDATE activity_sensor_streams SET extraction_version=2,payload_json=?',(json.dumps(old),))
        db.commit()
        update=import_files(db,tmp_path,'UTC',paths=[source])
        assert update.unchanged_files==1 and update.failed_files==0
        assert update.evidence_changed_activity_ids==[]
        assert dict(db.execute('SELECT * FROM activities').fetchone())==original
        assert db.execute('SELECT extraction_version FROM activity_sensor_streams').fetchone()[0]==3
        assert not db.execute('PRAGMA foreign_key_check').fetchall()


def test_feedback_change_still_invalidates_evidence_with_passive_fields_present(tmp_path):
    from dataclasses import replace
    from run_analysis.importer import _store_sensor_stream
    source=make_rr_fit(tmp_path/'run.fit')
    with connect(tmp_path/'test.sqlite') as db:
        import_files(db,tmp_path,'UTC',paths=[source])
        row=db.execute('SELECT * FROM activity_sensor_streams').fetchone()
        activity=parse_fit(source).activities[0]
        activity.sensor_stream.session_fields['workout_rpe']=40
        assert _store_sensor_stream(db,row['source_file_id'],0,row['activity_id'],activity)


@pytest.mark.parametrize('fit_first', [True, False])
def test_rr_survives_tcx_deduplication_and_force_reimport(tmp_path, fit_first):
    fit = make_rr_fit(tmp_path/'run.fit')
    tcx = make_tcx(tmp_path/'run.tcx', start='2026-09-24T00:00:00Z',
                   end='2026-09-24T00:00:20Z', activity_id='2026-09-24T00:00:00Z')
    tcx.write_text(tcx.read_text().replace('>40</DistanceMeters>', '>60</DistanceMeters>'))
    paths = [fit, tcx] if fit_first else [tcx, fit]
    with connect(tmp_path/'test.sqlite') as db:
        summary = import_files(db, tmp_path, 'UTC', paths=paths)
        assert summary.activities_added == 1
        assert summary.duplicate_activities == 1
        original = db.execute('SELECT payload_json FROM activity_sensor_streams').fetchone()[0]
        import_files(db, tmp_path, 'UTC', paths=[fit], force=True)
        assert db.execute('SELECT count(*) FROM activities').fetchone()[0] == 1
        assert db.execute('SELECT count(*) FROM activity_sensor_streams').fetchone()[0] == 1
        assert db.execute('SELECT payload_json FROM activity_sensor_streams').fetchone()[0] == original
        assert not db.execute('PRAGMA foreign_key_check').fetchall()
