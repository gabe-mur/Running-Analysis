from datetime import datetime, timedelta, timezone
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest
from fastapi.testclient import TestClient

from fit_builder import build_run
from run_analysis.cadence import GPX_UNSPECIFIED_CADENCE_SOURCE
from run_analysis.config import load_config
from run_analysis.db import connect
from run_analysis.geo import haversine_m
from run_analysis.gpx import parse_gpx
from run_analysis.importer import import_files, parser_for
from run_analysis.movement import classify_movement
from run_analysis.processing import process_activities
from run_analysis.web.app import create_app
from test_web_phase1 import _write_config

START = datetime(2026, 7, 1, 12, tzinfo=timezone.utc)


def make_gpx(path, *, hr=True, creator='StravaGPX', kind='9', points=91, start=START):
    records = []
    for index in range(points):
        at = (start + timedelta(seconds=index * 10)).isoformat()
        extensions = (f'<extensions><t:TrackPointExtension><t:hr>145</t:hr>'
                      '<t:cad>82</t:cad></t:TrackPointExtension></extensions>') if hr else ''
        records.append(f'<trkpt lat="{40.71 + index * .00027}" lon="-73.99">'
                       f'<ele>30</ele><time>{at}</time>{extensions}</trkpt>')
    path.write_text(f'<gpx xmlns="http://www.topografix.com/GPX/1/1" '
                    f'xmlns:t="http://www.garmin.com/xmlschemas/TrackPointExtension/v1" '
                    f'creator="{creator}" version="1.1"><trk><name>Morning run</name>'
                    f'<type>{kind}</type><trkseg>{"".join(records)}</trkseg></trk></gpx>')
    return path


def test_gpx_preserves_samples_and_marks_distance_as_gps_not_recorded_device(tmp_path):
    path = make_gpx(tmp_path / 'run.GPX')
    activity = parse_gpx(path).activities[0]
    assert parser_for(path) is parse_gpx
    assert activity.sport == 'Running'
    assert activity.start_time_utc == START
    assert activity.total_elapsed_time_s == 900
    assert activity.total_distance_m == pytest.approx(90 * haversine_m(40.71, -73.99, 40.71027, -73.99))
    assert activity.distance_source == 'gps'
    assert activity.gps_quality == 'gps_complete'
    assert activity.hr_quality == 'hr_complete'
    point = activity.trackpoints[0]
    assert point.heart_rate_bpm == 145
    assert point.altitude_m == 30
    assert point.distance_m is None  # No made-up device distance stream.
    assert point.pause_after_s is None
    assert point.cadence == 82
    assert point.cadence_source == GPX_UNSPECIFIED_CADENCE_SOURCE
    assert point.cadence_spm is None  # GPX cannot settle steps vs strides.
    assert activity.sensor_stream is None


@pytest.mark.parametrize('namespace,hr_tag,cad_tag', [
    ('http://www.garmin.com/xmlschemas/TrackPointExtension/v2', 'hr', 'cad'),
    ('http://www.cluetrust.com/XML/GPXDATA/1/0', 'hr', 'cadence'),
    ('', 'heartrate', 'cadence'),
])
def test_common_extension_namespaces_and_generic_tags(tmp_path, namespace, hr_tag, cad_tag):
    path = make_gpx(tmp_path / 'run.gpx', points=2)
    content = path.read_text().replace('http://www.garmin.com/xmlschemas/TrackPointExtension/v1', namespace)
    content = content.replace('t:hr', f't:{hr_tag}').replace('t:cad', f't:{cad_tag}')
    if not namespace:
        content = content.replace('xmlns:t=""', '').replace('t:', '').replace('<extensions>', '<extensions xmlns="">')
    path.write_text(content)
    assert parse_gpx(path).activities[0].trackpoints[0].heart_rate_bpm == 145


def test_missing_hr_is_kept_but_not_fitness_eligible(tmp_path):
    _write_config(tmp_path)
    path = make_gpx(tmp_path / 'missing.gpx', hr=False)
    with connect(tmp_path / 'test.sqlite') as connection:
        result = import_files(connection, tmp_path, 'America/New_York', paths=[path])
        process_activities(connection, load_config(tmp_path / 'config.yaml'))
        row = connection.execute('SELECT model_eligible,exclusion_reason FROM activity_metrics').fetchone()
        assert result.activities_added == 1
        assert row['model_eligible'] == 0
        assert 'inadequate_hr' in row['exclusion_reason']
        assert connection.execute('SELECT COUNT(*) FROM trackpoints WHERE heart_rate_bpm IS NOT NULL').fetchone()[0] == 0


@pytest.mark.parametrize('kind,eligible', [('9', True), ('', False), ('4', False), ('1', False)])
def test_only_identified_runs_feed_fitness_without_explicit_user_confirmation(tmp_path, kind, eligible):
    _write_config(tmp_path)
    path = make_gpx(tmp_path / 'activity.gpx', kind=kind)
    with connect(tmp_path / 'test.sqlite') as connection:
        import_files(connection, tmp_path, 'America/New_York', paths=[path])
        process_activities(connection, load_config(tmp_path / 'config.yaml'))
        row = connection.execute('SELECT model_eligible,exclusion_reason FROM activity_metrics').fetchone()
        assert bool(row['model_eligible']) is eligible
        if not eligible:
            assert 'gpx_activity_type_not_running' in row['exclusion_reason']


@pytest.mark.parametrize('kind,creator,expected', [
    ('running', 'Other', 'Running'), ('trail_run', 'Other', 'Running'),
    ('9', 'StravaGPX Android', 'Running'), ('1', 'StravaGPX', 'Biking'),
    ('4', 'StravaGPX', 'Hiking'), ('10', 'StravaGPX', 'Walking'),
    ('9', 'Other', '9'), ('', 'Other', 'Unknown'),
])
def test_sport_is_not_guessed_from_filename_or_numeric_codes_from_other_vendors(tmp_path, kind, creator, expected):
    activity = parse_gpx(make_gpx(tmp_path / 'run.gpx', kind=kind, creator=creator)).activities[0]
    assert activity.sport == expected


@pytest.mark.parametrize('content', [
    '<gpx><rte><rtept lat="40" lon="-74"/></rte></gpx>',
    '<gpx><trk><trkseg><trkpt lat="40" lon="-74"/></trkseg></trk></gpx>',
    '<gpx/>', '<TrainingCenterDatabase/>', '<gpx><broken>',
])
def test_routes_untimed_tracks_wrong_formats_and_malformed_files_fail(tmp_path, content):
    path = tmp_path / 'route.gpx'
    path.write_text(content)
    with pytest.raises((ValueError, ET.ParseError)):
        parse_gpx(path)


@pytest.mark.parametrize('encoding', ['utf-8', 'utf-16'])
def test_entity_and_dtd_payloads_are_rejected(tmp_path, encoding):
    path = tmp_path / 'entity.gpx'
    path.write_bytes(('<?xml version="1.0" encoding="' + encoding + '"?>'
                     '<!DOCTYPE gpx [<!ENTITY injected "secret">]><gpx>&injected;</gpx>').encode(encoding))
    with pytest.raises(ValueError, match='document types'):
        parse_gpx(path)


def test_invalid_and_foreign_extension_values_do_not_become_measurements(tmp_path):
    path = make_gpx(tmp_path / 'bad.gpx', points=2)
    content = path.read_text().replace('lat="40.71"', 'lat="NaN"').replace('<t:hr>145', '<t:hr>NaN')
    content = content.replace('<t:cad>82', '<t:cad>inf').replace('<ele>30', '<ele>nan')
    path.write_text(content)
    activity = parse_gpx(path).activities[0]
    assert activity.trackpoints[0].gps_valid is False
    assert activity.trackpoints[0].latitude is None
    assert all(p.heart_rate_bpm is None and p.cadence is None and p.altitude_m is None for p in activity.trackpoints)
    path.write_text(make_gpx(path, points=2).read_text().replace('http://www.garmin.com/xmlschemas/TrackPointExtension/v1', 'https://unrelated.example/sensor'))
    assert parse_gpx(path).activities[0].hr_quality == 'hr_missing'


def test_segment_breaks_do_not_invent_motion_or_hr_across_the_gap(tmp_path):
    _write_config(tmp_path)
    path = make_gpx(tmp_path / 'segments.gpx', points=4)
    text = path.read_text()
    text = text.replace('<trkpt lat="40.71054"', '</trkseg><trkseg><trkpt lat="41.71054"')
    text = text.replace('40.710809999999995', '41.71081').replace('40.71081', '41.71081')
    path.write_text(text)
    activity = parse_gpx(path).activities[0]
    movement = classify_movement(activity.trackpoints, load_config(tmp_path / 'config.yaml')['moving_time'])
    assert len(movement.intervals) == 2
    assert movement.diagnostics['analysis_distance_m'] < 100
    assert movement.diagnostics['analysis_distance_m'] == pytest.approx(activity.total_distance_m)
    assert activity.total_elapsed_time_s == 30
    assert activity.lap_recorded_time_s == 20


@pytest.mark.parametrize('first_fit', [False, True])
def test_cross_format_duplicate_keeps_richer_fit_in_either_order(tmp_path, first_fit):
    gpx = make_gpx(tmp_path / 'run.gpx', points=12)
    distance = parse_gpx(gpx).activities[0].total_distance_m
    fit = tmp_path / 'run.fit'
    build_run(fit, start=START, points=12, metres_per_point=distance / 11, pauses=[(40, 50)])
    paths = [fit, gpx] if first_fit else [gpx, fit]
    with connect(tmp_path / 'test.sqlite') as connection:
        result = import_files(connection, tmp_path, 'America/New_York', paths=paths)
        assert result.activities_added == 1
        assert result.duplicate_activities == 1
        assert connection.execute('SELECT COUNT(*) FROM activities').fetchone()[0] == 1
        assert connection.execute('SELECT MAX(pause_after_s) FROM trackpoints').fetchone()[0] == 10
        assert connection.execute('SELECT COUNT(*) FROM activity_sensor_streams').fetchone()[0] == 1
        repeated = import_files(connection, tmp_path, 'America/New_York', paths=paths)
        assert repeated.unchanged_files == 2


def test_gpx_upload_api_and_repeated_content_do_not_search_the_planner(tmp_path, monkeypatch):
    _write_config(tmp_path)
    path = make_gpx(tmp_path / 'run.gpx')
    def forbidden(*args, **kwargs):
        raise AssertionError('Uploads must not start a planner search')
    monkeypatch.setattr('run_analysis.web.upload_service.generate_weekly_schedule', forbidden)
    client = TestClient(create_app(tmp_path))
    upload = [('files', ('run.gpx', path.read_bytes(), 'application/gpx+xml'))]
    result = client.post('/api/uploads', files=upload)
    assert result.status_code == 200
    assert result.json()['files'][0]['status'] == 'imported'
    assert result.json()['primary_activity_id'] is not None
    assert client.post('/api/uploads', files=upload).json()['files'][0]['status'] == 'unchanged'
    # The same bytes with the wrong extension must parse/fail, not hit a
    # generic non-FIT content cache and silently masquerade as accepted TCX.
    mislabeled = client.post('/api/uploads', files=[('files', ('wrong.tcx', path.read_bytes(), 'text/xml'))])
    assert mislabeled.json()['files'][0]['status'] == 'failed'


def test_gpx_upload_exposes_missing_hr_and_unknown_sport_to_user(tmp_path):
    _write_config(tmp_path)
    path = make_gpx(tmp_path / 'missing.gpx', hr=False, kind='')
    client = TestClient(create_app(tmp_path))
    result = client.post('/api/uploads', files=[('files', ('missing.gpx', path.read_bytes(), 'text/xml'))])
    assert result.status_code == 200
    warnings = result.json()['files'][0]['warnings']
    assert any('no heart-rate samples' in value for value in warnings)
    assert any('does not identify' in value for value in warnings)
