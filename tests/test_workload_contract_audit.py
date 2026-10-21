from datetime import datetime, timezone

from run_analysis.db import connect
from run_analysis.importer import import_files
from scripts.audit_workload_contract import trial
from test_tcx import make_tcx
from test_workload_profile import CONFIG


def test_original_file_trial_has_frozen_preceding_reference_and_no_db_writes(tmp_path):
    paths = [make_tcx(tmp_path/f'run-{day}.tcx', start=f'2026-09-{day}T12:00:00Z',
                     end=f'2026-09-{day}T12:00:20Z', activity_id=f'2026-09-{day}T12:00:00Z')
             for day in ('01', '18', '22')]
    database = tmp_path/'runs.sqlite'
    with connect(database) as db:
        import_files(db, tmp_path, 'UTC', paths=paths)
        before = list(db.iterdump())
        since, cutoff, until = [datetime(2026, 9, day, tzinfo=timezone.utc) for day in (1, 17, 25)]
        report = trial(database, CONFIG, since, cutoff, until)
        assert list(db.iterdump()) == before
    assert len(report['reference_session_keys']) == 1
    assert len(report['sessions']) == 3
    assert len(report['policies']) == 27
    assert len(report['cases']) == 36
    assert all(sum(case['counts'].values()) == 81 for case in report['cases'])
    assert not report['planner_influence']


def test_expectation_loader_uses_app_resolved_override_without_writes(tmp_path):
    from scripts.audit_workload_expectation import app_session_types
    from scripts.audit_workload_contract import load_sessions
    path = make_tcx(tmp_path/'run.tcx')
    database = tmp_path/'runs.sqlite'
    with connect(database) as db:
        import_files(db, tmp_path, 'UTC', paths=[path])
        key, external_id = db.execute('SELECT id,activity_id FROM activities').fetchone()
        db.execute('INSERT INTO run_overrides(activity_id,workout_type) VALUES (?,?)',
                   (external_id, 'intervals'))
        db.commit()
        before = list(db.iterdump())
        types = app_session_types(database)
        assert types[str(key)] == 'intervals'
        sessions = load_sessions(database, CONFIG,
            datetime(2000, 1, 1, tzinfo=timezone.utc), datetime(2100, 1, 1, tzinfo=timezone.utc),
            session_types=types)
        assert sessions[0][1]['intensity_structure']['session_workout_type'] == 'intervals'
        # A type alone still cannot manufacture performed reps from no structure.
        assert sessions[0][1]['intensity_structure']['pace_inferred_work']['bout_count'] == 0
        assert list(db.iterdump()) == before


def test_old_archive_optional_schema_is_migrated_only_in_memory(tmp_path):
    from scripts.audit_workload_expectation import app_session_types
    path=make_tcx(tmp_path/'run.tcx')
    database=tmp_path/'old.sqlite'
    with connect(database) as db:
        import_files(db,tmp_path,'UTC',paths=[path])
        db.execute('DROP TABLE activity_sensor_streams')
        db.commit()
        before=list(db.iterdump())
        assert app_session_types(database)
        assert list(db.iterdump())==before


def test_summary_only_archive_record_requires_explicit_inventory_callback(tmp_path,monkeypatch):
    import pytest
    from dataclasses import replace
    import scripts.audit_workload_contract as module
    path=make_tcx(tmp_path/'old.tcx',start='2020-06-01T12:00:00Z',
        end='2020-06-01T12:00:20Z',activity_id='2020-06-01T12:00:00Z')
    database=tmp_path/'old.sqlite'
    parsed=module.parse_tcx(path)
    parsed.activities[0]=replace(parsed.activities[0],total_elapsed_time_s=None,
        lap_recorded_time_s=1465,trackpoints=[])
    with connect(database) as db:
        import_files(db,tmp_path,'UTC',paths=[path])
        before=list(db.iterdump())
        monkeypatch.setattr(module,'parse_tcx',lambda *args,**kwargs:parsed)
        dates=(datetime(2000,1,1,tzinfo=timezone.utc),datetime(2100,1,1,tzinfo=timezone.utc))
        with pytest.raises(ValueError,match='Missing end-time'):
            module.load_sessions(database,CONFIG,*dates)
        captured=[]
        assert module.load_sessions(database,CONFIG,*dates,
            missing_timing_observer=lambda a,k,p:captured.append((a,k,p)))==[]
        assert len(captured)==1
        assert captured[0][0].lap_recorded_time_s==1465
        assert list(db.iterdump())==before


def test_projection_audit_defaults_to_entire_archive_not_current_year(tmp_path,monkeypatch):
    import pytest
    import scripts.audit_athlete_projection as module
    path=make_tcx(tmp_path/'run.tcx')
    database=tmp_path/'runs.sqlite'
    with connect(database) as db:
        import_files(db,tmp_path,'UTC',paths=[path])
    captured=[]
    def stop_after_scope(database,config,since,until,**kwargs):
        captured.append(since)
        raise RuntimeError('scope inspected')
    monkeypatch.setattr(module,'load_sessions',stop_after_scope)
    with pytest.raises(RuntimeError,match='scope inspected'):
        module.audit(database,CONFIG)
    assert captured[-1].year==1
    explicit=datetime(2024,1,1,tzinfo=timezone.utc)
    with pytest.raises(RuntimeError,match='scope inspected'):
        module.audit(database,CONFIG,since=explicit)
    assert captured[-1]==explicit
