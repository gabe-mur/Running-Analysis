from datetime import datetime, timedelta, timezone
import sqlite3

import pytest

from run_analysis.db import initialize
from run_analysis.config import load_config
from run_analysis.onboarding import automatic_comparison_hr
from run_analysis.settings_service import save_settings_overlay, refresh_automatic_comparison, recalculate_for_settings
from run_analysis.web.schemas import SettingsPatch
from run_analysis.modeling import InsufficientModelDataError
from test_web_phase1 import _write_config

NOW = datetime(2026, 10, 11, tzinfo=timezone.utc)
ZONES = {'z2': [141, 153]}


@pytest.fixture
def db():
    connection = sqlite3.connect(':memory:')
    connection.row_factory = sqlite3.Row
    initialize(connection)
    yield connection
    connection.close()


def run(db, bpm, *, days=1, role='easy', health='normal', eligible=1, count=1,
        miles=3, moving_s=1800, manual=None):
    identifier = db.execute('SELECT COUNT(*) FROM activities').fetchone()[0] + 1
    start = (NOW - timedelta(days=days)).isoformat()
    db.execute('''INSERT INTO activities(id,activity_uid,activity_id,sport,start_time_utc,
        lap_count,trackpoint_count,gps_quality,hr_quality,elevation_quality,cadence_quality,
        distance_source,namespaces_json,data_quality_json,created_at_utc,updated_at_utc)
        VALUES (?,?,?,'Running',?,1,1,'good','good','good','good','device','{}','{}',?,?)''',
        (identifier, str(identifier), str(identifier), start, start, start))
    db.execute('''INSERT INTO activity_metrics(activity_id,metrics_json,calculated_at_utc,
        model_eligible,detected_workout_type) VALUES (?,'{}',?,?,?)''',
        (identifier, start, eligible, role))
    db.execute('UPDATE activities SET total_distance_m=? WHERE id=?', (miles*1609.344, identifier))
    db.execute('UPDATE activity_metrics SET calculated_moving_time_s=? WHERE activity_id=?',
        (moving_s, identifier))
    for index in range(count):
        db.execute('''INSERT INTO segments(activity_id,segment_index,metrics_json,average_hr_bpm,is_pathological)
            VALUES (?,?,'{}',?,0)''', (identifier,index,bpm))
    if health != 'normal':
        db.execute('INSERT INTO run_overrides(activity_id,health_tag) VALUES (?,?)', (str(identifier),health))
    if manual is not None:
        db.execute('INSERT INTO run_overrides(activity_id,workout_type) VALUES (?,?)',
            (str(identifier), manual))
    return identifier


def test_sparse_history_uses_zone_midpoint_not_a_stale_coordinate(db):
    for _ in range(7):
        run(db,143)
    assert automatic_comparison_hr(db,ZONES,current=143,as_of=NOW).recommended_bpm == 147
    assert automatic_comparison_hr(db,{'z2':[150,160]},current=143,as_of=NOW).recommended_bpm == 155


def test_recent_comparable_runs_not_quality_illness_old_future_or_one_long_run(db):
    for _ in range(8):
        run(db,146)
    for kwargs in [dict(days=91),dict(days=-1),dict(role='intervals'),dict(health='illness'),dict(eligible=0)]:
        for _ in range(10):
            run(db,150,**kwargs)
    run(db,150,count=100)
    recommendation = automatic_comparison_hr(db,ZONES,as_of=NOW)
    assert recommendation.recommended_bpm == 146
    assert recommendation.run_count == 8


def test_equal_run_support_retains_current_coordinate(db):
    for _ in range(8):
        run(db,146)
        run(db,148,count=4)
    assert automatic_comparison_hr(db,ZONES,current=146,as_of=NOW).recommended_bpm == 146
    run(db,148)
    assert automatic_comparison_hr(db,ZONES,current=146,as_of=NOW).recommended_bpm == 148


def test_unlabelled_easy_and_long_runs_use_existing_app_roles(db):
    from run_analysis.run_feedback import _infer_workout_type, _long_run_threshold_from_connection

    for _ in range(6):
        run(db,148,role=None)
    for _ in range(2):
        run(db,148,role=None,miles=9,moving_s=5400)
    threshold = _long_run_threshold_from_connection(db)
    assert _infer_workout_type(None,None,3,30,{},threshold).value == 'easy'
    assert _infer_workout_type(None,None,9,90,{},threshold).value == 'long'
    before = tuple(db.iterdump())
    recommendation = automatic_comparison_hr(db,ZONES,current=148,as_of=NOW)
    assert recommendation.recommended_bpm == 148
    assert recommendation.run_count == 8
    assert recommendation.segment_count == 8
    assert tuple(db.iterdump()) == before  # Resolution must not relabel stored activities.


def test_unlabelled_runs_keep_health_time_and_quality_filters(db):
    for _ in range(8):
        run(db,146,role=None)
    for kwargs in (dict(days=91),dict(days=-1),dict(health='illness'),dict(eligible=0),
                   dict(moving_s=0),dict(role='intervals'),dict(role='tempo_threshold')):
        for _ in range(10):
            run(db,150,**{'role':None,**kwargs})
    run(db,150,role=None,count=100)
    recommendation = automatic_comparison_hr(db,ZONES,as_of=NOW)
    assert recommendation.recommended_bpm == 146
    assert recommendation.run_count == 8
    assert next(c.run_count for c in recommendation.candidates if c.heart_rate_bpm == 150) == 1


@pytest.mark.parametrize('manual', ['intervals','tempo','threshold','race','run_walk','hike','bike','other'])
def test_manual_non_aerobic_roles_do_not_become_easy_when_detection_is_absent(db,manual):
    for _ in range(8):
        run(db,148,role=None)
        run(db,150,role=None,manual=manual)
    recommendation = automatic_comparison_hr(db,ZONES,current=148,as_of=NOW)
    assert recommendation.recommended_bpm == 148
    assert [c.heart_rate_bpm for c in recommendation.candidates] == [148]


def test_inferred_support_counts_runs_not_number_of_segments(db):
    for _ in range(8):
        run(db,148,role=None,count=2)
    run(db,146,role=None,count=100)
    recommendation = automatic_comparison_hr(db,ZONES,current=148,as_of=NOW)
    assert recommendation.recommended_bpm == 148
    assert recommendation.run_count == 8
    assert recommendation.segment_count == 16


def test_automatic_save_is_authoritative_and_failed_refit_cannot_relabel_old_pace(db,tmp_path,monkeypatch):
    _write_config(tmp_path)
    identifier = run(db,146)
    db.execute('UPDATE activity_metrics SET standardized_pace_at_target_hr_min_mile=9, selected_model_version="old"')
    db.execute('INSERT INTO model_runs(activity_id,model_name,model_version,result_json) VALUES (?,"standardized_pace_at_target_hr","old","{}")',(identifier,))
    previous = load_config(tmp_path/'config.yaml')
    patch = SettingsPatch(target_hr=150,setup={'comparison_hr_mode':'automatic'})
    updated = save_settings_overlay(tmp_path/'config.yaml',patch,connection=db)
    assert updated['target_hr'] == 147  # Sparse history; submitted value is not authoritative.
    monkeypatch.setattr('run_analysis.settings_service.process_activities',lambda *a,**k: (_ for _ in ()).throw(ValueError('no raw test files')))
    monkeypatch.setattr('run_analysis.settings_service.fit_models',lambda *a,**k: (_ for _ in ()).throw(InsufficientModelDataError('sparse')))
    stages = recalculate_for_settings(db,updated,tmp_path,patch,previous_config=previous)
    assert next(s.status for s in stages if s.name=='model') == 'deferred'
    assert db.execute('SELECT standardized_pace_at_target_hr_min_mile FROM activity_metrics').fetchone()[0] is None
    assert db.execute('SELECT COUNT(*) FROM model_runs').fetchone()[0] == 0
    assert db.execute('SELECT COUNT(*) FROM activities').fetchone()[0] == 1
    assert db.execute('SELECT COUNT(*) FROM segments').fetchone()[0] == 1
    before = tuple(db.iterdump())
    config, changed = refresh_automatic_comparison(tmp_path/'config.yaml',db)
    assert not changed and config['target_hr']==147
    assert tuple(db.iterdump()) == before


def test_background_coordinate_needs_no_legacy_setup_opt_in(db,tmp_path):
    _write_config(tmp_path)
    config, changed = refresh_automatic_comparison(tmp_path/'config.yaml',db)
    assert changed and config['target_hr']==147
    assert config['setup']['comparison_hr_mode']=='automatic'


def test_normal_ui_save_derives_coordinate_without_a_target_or_mode(db,tmp_path):
    _write_config(tmp_path)
    for _ in range(8):
        run(db,148,role=None)
    config = save_settings_overlay(tmp_path/'config.yaml',
        SettingsPatch(setup={'zone_method':'custom'}),connection=db)
    assert config['target_hr']==148
    assert config['setup']['comparison_hr_mode']=='automatic'


def test_unchanged_background_coordinate_preserves_existing_results_without_opt_in(db,tmp_path):
    _write_config(tmp_path)
    for _ in range(8):
        run(db,145,role=None)
    before = tuple(db.iterdump())
    overlay_before = (tmp_path/'config.local.yaml').read_bytes() if (tmp_path/'config.local.yaml').exists() else None
    config, changed = refresh_automatic_comparison(tmp_path/'config.yaml',db)
    assert not changed and config['target_hr']==145
    assert tuple(db.iterdump())==before
    overlay_after = (tmp_path/'config.local.yaml').read_bytes() if (tmp_path/'config.local.yaml').exists() else None
    assert overlay_after==overlay_before


def test_upload_rechecks_automatic_anchor_and_preserves_unchanged_results(db,tmp_path,monkeypatch):
    _write_config(tmp_path)
    save_settings_overlay(tmp_path/'config.yaml',SettingsPatch(setup={'comparison_hr_mode':'automatic'}),connection=db)
    for _ in range(8):
        run(db,143)
    original = automatic_comparison_hr
    monkeypatch.setattr('run_analysis.onboarding.automatic_comparison_hr',
        lambda connection,zones,**kwargs: original(connection,zones,**kwargs,as_of=NOW))
    config, changed = refresh_automatic_comparison(tmp_path/'config.yaml',db)
    assert changed and config['target_hr']==143
    db.execute('UPDATE activity_metrics SET standardized_pace_at_target_hr_min_mile=9')
    db.commit()
    before = tuple(db.iterdump())
    config, changed = refresh_automatic_comparison(tmp_path/'config.yaml',db)
    assert not changed and config['target_hr']==143
    assert tuple(db.iterdump())==before
