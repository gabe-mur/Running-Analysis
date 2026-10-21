"""Saved-plan freshness, separate from running the expensive planner."""
from datetime import datetime, timedelta, timezone
import json
from uuid import uuid4
from zoneinfo import ZoneInfo

KEY = 'weekly_schedule_pending_refresh'


def pending_refresh(connection):
    row = connection.execute('SELECT value_json FROM app_state WHERE key=?', (KEY,)).fetchone()
    return row[0] if row else None


def mark_plan_refresh(connection):
    # A new token prevents an older concurrent planner from clearing newer input.
    connection.execute('''INSERT INTO app_state VALUES (?,?,?)
        ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json,
        updated_at_utc=excluded.updated_at_utc''',
        (KEY, json.dumps({'revision': uuid4().hex}), datetime.now(timezone.utc).isoformat()))
    connection.commit()


def complete_plan_refresh(connection, build):
    revision = pending_refresh(connection)
    result = build()
    if revision is not None and result.planner_status == 'ready':
        connection.execute('DELETE FROM app_state WHERE key=? AND value_json=?', (KEY, revision))
        connection.commit()
    return result


def saved_plan_state(connection, config, *, as_of=None):
    """No search, runtime lock, weather fetch, or materialization on this path."""
    if not connection.in_transaction:
        from .read_cache import invariant_snapshot
        # Read the saved plan and its freshness marker from the same version;
        # a concurrent publication must not make an older plan look current.
        with invariant_snapshot(connection):
            return saved_plan_state(connection, config, as_of=as_of)
    from .recommendation_service import load_latest_weekly_schedule
    schedule = load_latest_weekly_schedule(connection)
    today = (as_of or datetime.now(timezone.utc)).astimezone(
        ZoneInfo(config.get('timezone_default', 'UTC'))).date()
    boundary_current = schedule is not None and (
        schedule.start_date == today or (
            schedule.start_date == today + timedelta(days=1)
            and schedule.trailing_days and schedule.trailing_days[-1].date == today
            and schedule.trailing_days[-1].activities))
    return {'schedule': schedule, 'needs_refresh': bool(
        not boundary_current or pending_refresh(connection) is not None)}
