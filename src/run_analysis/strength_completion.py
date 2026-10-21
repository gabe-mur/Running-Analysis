"""Manual lifting receipts, independent of running loads and issued guidance."""
from datetime import date, datetime, timezone
import json
from zoneinfo import ZoneInfo

from .web.schemas import StrengthCompletion, StrengthSuggestion

KEY = 'strength_completions'


def load_completions(connection):
    row = connection.execute('SELECT value_json FROM app_state WHERE key=?', (KEY,)).fetchone()
    return {date.fromisoformat(day): StrengthCompletion.model_validate(value)
            for day, value in json.loads(row[0] if row else '{}').items()}


def _suggestion_for_day(connection, config, day):
    from .strength_schedule import load_strength_rotation, _suggestion, _settings
    for key in ('weekly_schedule_internal', 'weekly_schedule'):
        row = connection.execute('SELECT value_json FROM app_state WHERE key=?', (key,)).fetchone()
        if row:
            for item in json.loads(row[0]).get('days', []):
                if item['date'] == day.isoformat() and item.get('strength_suggestion'):
                    return StrengthSuggestion.model_validate(item['strength_suggestion'])
    kind = load_strength_rotation(connection).get(day)
    return _suggestion(kind, _settings(config), 8) if kind else None


def save_completion(connection, config, day, kind, completed, *, as_of=None):
    """Idempotent checkbox/undo. Never trigger search or manufacture load."""
    now = as_of or datetime.now(timezone.utc)
    if day > now.astimezone(ZoneInfo(config['timezone_default'])).date():
        raise ValueError('Future lifting sessions cannot be marked completed.')
    connection.execute('BEGIN IMMEDIATE')
    try:
        receipts = load_completions(connection)
        existing = receipts.get(day)
        if existing and existing.session.session_type != kind:
            raise ValueError('This date has a different completed lifting session. Undo that completion first.')
        if completed:
            suggestion = existing.session if existing else _suggestion_for_day(connection, config, day)
            if suggestion is None or suggestion.session_type != kind:
                raise ValueError('This lifting suggestion changed. Reload the plan before marking it complete.')
            receipts[day] = existing or StrengthCompletion(session=suggestion, recorded_at=now)
        else:
            receipts.pop(day, None)
        connection.execute('''INSERT INTO app_state VALUES (?,?,?) ON CONFLICT(key)
            DO UPDATE SET value_json=excluded.value_json,updated_at_utc=excluded.updated_at_utc''',
            (KEY, json.dumps({d.isoformat(): r.model_dump(mode='json') for d, r in receipts.items()}), now.isoformat()))
        connection.commit()
        return receipts.get(day)
    except Exception:
        connection.rollback()
        raise


def with_strength_records(connection, schedule, config=None):
    """Overlay receipts and recent issued routines without mutating the plan."""
    from .strength_schedule import load_strength_rotation, _suggestion, _settings
    receipts = load_completions(connection)
    prior = load_strength_rotation(connection)
    settings = _settings(config or {})
    days = [d.model_copy(update={'strength_completion': receipts.get(d.date)}) for d in schedule.days]
    trailing = []
    for d in schedule.trailing_days:
        receipt = receipts.get(d.date)
        kind = prior.get(d.date)
        suggestion = receipt.session if receipt else _suggestion(kind, settings, 8) if kind else d.strength_suggestion
        trailing.append(d.model_copy(update={'strength_suggestion': suggestion, 'strength_completion': receipt}))
    return schedule.model_copy(update={'days': days, 'trailing_days': trailing})
