from datetime import date, datetime, timedelta, timezone
import json
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

from run_analysis.config import load_config
from run_analysis.db import connect, initialize
from run_analysis.strength_completion import save_completion, load_completions, with_strength_records
from run_analysis.strength_schedule import strength_rotation_payload, _suggestion
from run_analysis.web.schemas import (
    StrengthSessionType as Kind, WeeklyScheduleDay, WeeklyScheduleResponse,
    WeeklyTargetEvidence, TrailingCalendarDay,
)
from run_analysis.recommendation_service import load_latest_weekly_schedule
from run_analysis.web.app import create_app
from test_web_phase1 import _write_config

AT = datetime(2026, 10, 17, 16, tzinfo=timezone.utc)


def seed(root, day=AT.date()):
    _write_config(root)
    config = load_config(root / 'config.yaml')
    days = [WeeklyScheduleDay(date=day + timedelta(days=i), day_role='rest_day', rationale='',
        strength_suggestion=_suggestion(kind, {}, 8)) for i, kind in enumerate((Kind.PULL, Kind.PUSH))]
    evidence = WeeklyTargetEvidence(recent_7d_miles=0, chronic_42d_weekly_miles=0,
        best_sustained_28d_weekly_miles=0, peak_7d_miles=0, demonstrated_run_days_per_week=0,
        capacity_reference_miles=0, rationale='fixture')
    schedule = WeeklyScheduleResponse(generated_at=AT, start_date=day, end_date=day + timedelta(days=6),
        target_run_count=0, run_count=0, target_distance_range_miles=(0, 0),
        projected_distance_range_miles=(0, 0), target_evidence=evidence, summary='fixture', days=days,
        trailing_days=[TrailingCalendarDay(date=day-timedelta(days=1), day_role='rest_day', total_distance_miles=0)])
    with connect(root / 'data/test.sqlite') as db:
        initialize(db)
        for key, value in (
            ('weekly_schedule', schedule.model_dump_json()),
            ('weekly_schedule_internal', json.dumps({'days':[d.model_dump(mode='json') for d in days]})),
            ('strength_rotation', strength_rotation_payload(days, {day-timedelta(days=1):Kind.LEGS}, config))):
            db.execute('INSERT INTO app_state VALUES (?,?,?)', (key, value, AT.isoformat()))
        db.commit()
    return config, schedule


def test_check_undo_retry_and_reload_do_not_change_prescriptions(tmp_path):
    config, schedule = seed(tmp_path)
    with connect(tmp_path / 'data/test.sqlite') as db:
        before = list(db.execute('SELECT * FROM app_state ORDER BY key'))
        first = save_completion(db, config, AT.date(), Kind.PULL, True, as_of=AT)
        assert save_completion(db, config, AT.date(), Kind.PULL, True, as_of=AT+timedelta(hours=1)) == first
        assert load_latest_weekly_schedule(db).days[0].strength_completion == first
        for row in before:
            assert tuple(db.execute('SELECT * FROM app_state WHERE key=?', (row['key'],)).fetchone()) == tuple(row)
        assert save_completion(db, config, AT.date(), Kind.PULL, False, as_of=AT) is None
        assert save_completion(db, config, AT.date(), Kind.PULL, False, as_of=AT) is None
        assert load_latest_weekly_schedule(db).days[0].strength_completion is None


def test_past_suggestion_can_be_completed_and_survives_rotation_replacement(tmp_path):
    config, schedule = seed(tmp_path)
    past = AT.date()-timedelta(days=1)
    with connect(tmp_path / 'data/test.sqlite') as db:
        receipt = save_completion(db, config, past, Kind.LEGS, True, as_of=AT)
        db.execute("DELETE FROM app_state WHERE key='strength_rotation'"); db.commit()
        shown = with_strength_records(db, schedule, config).trailing_days[0]
        assert shown.strength_completion == receipt
        assert shown.strength_suggestion == receipt.session
        assert not schedule.trailing_days[0].strength_completion
        with pytest.raises(ValueError, match='different completed'):
            save_completion(db, config, past, Kind.PUSH, True, as_of=AT)


def test_receipt_preserves_completed_type_not_new_suggestion(tmp_path):
    config, schedule = seed(tmp_path)
    with connect(tmp_path / 'data/test.sqlite') as db:
        receipt = save_completion(db, config, AT.date(), Kind.PULL, True, as_of=AT)
        changed = schedule.model_copy(update={'days':[schedule.days[0].model_copy(
            update={'strength_suggestion':_suggestion(Kind.PUSH, {}, 8)})]})
        shown = with_strength_records(db, changed).days[0]
        assert shown.strength_completion.session.session_type == Kind.PULL
        assert shown.strength_suggestion.session_type == Kind.PUSH
        assert receipt.recorded_at == AT


@pytest.mark.parametrize('day,kind', [(date(2026,10,18),Kind.PUSH), (date(2026,10,17),Kind.LEGS), (date(2026,10,12),Kind.PUSH)])
def test_future_unknown_and_stale_actions_leave_no_receipt(tmp_path,day,kind):
    config, _ = seed(tmp_path)
    with connect(tmp_path / 'data/test.sqlite') as db:
        with pytest.raises(ValueError):
            save_completion(db, config, day, kind, True, as_of=AT)
        assert load_completions(db) == {}


def test_future_guard_uses_athlete_local_day(tmp_path):
    config, _ = seed(tmp_path)
    with connect(tmp_path / 'data/test.sqlite') as db:
        with pytest.raises(ValueError, match='Future'):
            save_completion(db,config,date(2026,10,18),Kind.PUSH,True,
                as_of=datetime(2026,10,18,0,30,tzinfo=timezone.utc))


def test_endpoint_saves_without_calling_running_planner_and_cached_reads_show_it(tmp_path,monkeypatch):
    today = datetime.now(timezone.utc).astimezone(ZoneInfo('America/New_York')).date()
    seed(tmp_path, today)
    def no_planner(*a, **kw):
        raise AssertionError('A lifting checkbox must not run the planner')
    monkeypatch.setattr('run_analysis.web.app.ensure_current_weekly_schedule', no_planner)
    monkeypatch.setattr('run_analysis.web.app.generate_weekly_schedule', no_planner)
    with TestClient(create_app(tmp_path)) as client:
        body = {'date':today.isoformat(),'session_type':'pull','completed':True}
        response = client.post('/api/strength-completions',json=body)
        assert response.status_code == 200
        assert response.json()['strength_completion']['session']['session_type'] == 'pull'
        shown = client.get('/api/weekly-schedule/saved')
        assert shown.status_code == 200
        assert shown.json()['schedule']['days'][0]['strength_completion']
        assert shown.json()['schedule']['trailing_days'][0]['strength_suggestion']['session_type'] == 'legs'
        undone = client.post('/api/strength-completions',json={**body,'completed':False})
        assert undone.status_code == 200 and undone.json()['strength_completion'] is None
        assert client.post('/api/strength-completions',json={**body,'session_type':'fake'}).status_code == 422
