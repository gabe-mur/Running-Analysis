"""Read-only Settings projection of the last durable adaptive decision.

Never resume a harness, create a database, reprice a schedule or substitute factory
defaults. The snapshot is independently identity-checked before displaying values.
"""
import json
from pathlib import Path
import sqlite3

from run_analysis.web.schemas import PlannerSettingsStatus
from .decision_snapshot import restore
from .live_bridge import ENGINE, enabled
from .planner_preferences import latest_request


def inspect_settings(config, project_root):
    if not enabled(config):
        return None
    unavailable = dict(engine=ENGINE, status='review_required',
        detail='No verified adopted policy is available. Settings have not started or reset the planner.')
    try:
        database = Path(config['paths']['database'])
        if not database.is_absolute():
            database = Path(project_root) / database
        with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True) as app:
            pointer = app.execute("SELECT value_json FROM app_state WHERE key='adaptive_runtime'").fetchone()
        if pointer is None:
            if config.get('beginner_start'):
                return PlannerSettingsStatus(engine=ENGINE,status='introductory',
                    detail='Your starter schedule builds from recorded easy outings. Full-planner preferences become available after the handoff.')
            return PlannerSettingsStatus(**unavailable)
        pointer = json.loads(pointer[0])
        if pointer['engine'] != ENGINE:
            return PlannerSettingsStatus(**unavailable)
        runtime = Path(pointer['root']) / 'sandbox.sqlite'
        with sqlite3.connect(runtime.resolve().as_uri()+'?mode=ro', uri=True) as journal:
            journal.execute('BEGIN')
            row = journal.execute('SELECT event,snapshot FROM sandbox_decisions ORDER BY sequence DESC LIMIT 1').fetchone()
            requested = latest_request(journal)
        if row is None:
            return PlannerSettingsStatus(**unavailable)
        event = json.loads(row[0])
        snapshot = restore(row[1], expected_fingerprint=event['snapshot_fingerprint'])
        stage = event['stage_runtime']['stage']
        if (stage['key'] != snapshot.stage_key or snapshot.frequency_contract is None
                or snapshot.frequency_contract.version != 'frequency_midpoint_v3'):
            return PlannerSettingsStatus(**unavailable)
        context = snapshot.context
        frequency = snapshot.frequency_contract
        target = frequency.targets[-1]
        applied = snapshot.planner_preferences
        latest = requested or applied
        goal = applied.goal_long_miles if applied else stage['goal_long_miles']
        mode = applied.frequency_mode if applied else 'balanced'
        weekday = applied.long_run_weekday if applied else None
        scenario_path=Path(pointer['root'])/'scenario.json'
        future_policy=(json.loads(scenario_path.read_text())['search_policy'].get('general_progression')
            if scenario_path.exists() else None)  # Old read-only snapshot fixtures remain bounded-only.
        progression_detail=('Further growth is reviewed after delivered training, adding easy volume before increasing the long run.'
            if future_policy in {'easy_first_coordinated_v1','easy_first_coordinated_v2'} else
            'This bounded phase does not implement automatic subsequent phases toward the long-run goal.')
        return PlannerSettingsStatus(engine=ENGINE, status='adopted',
            detail='Effective values from the last issued decision. Frequency and long-run day edits trigger re-evaluation on Weekly Plan refresh; destination-only edits join the next normal replan without forcing a new calendar search. '+progression_detail,
            issued_at=snapshot.as_of, snapshot_fingerprint=snapshot.fingerprint,
            long_goal_miles=goal, long_target_miles=stage['long_miles'],
            volume_start_miles_per_7=stage['weekly_equivalent_miles'],
            volume_destination_miles_per_7=stage['policy']['volume_goal_miles'],
            review_days=stage['policy']['review_days'], growth_fraction=stage['policy']['growth_fraction'],
            long_cadence_days=7/context.policy.long_per_7,
            quality_cadence_days=7/context.policy.quality_per_7,
            frequency_lower_per_7=target.lower_per_7, frequency_upper_per_7=target.upper_per_7,
            frequency_preference_fraction=frequency.preference.edge_fraction,
            minimum_running_minutes=context.easy_duration_policy.minimum_minutes,
            penalty_free_running_minutes=context.easy_duration_policy.penalty_free_minutes,
            demand_memory_days=context.scenario.memory_days,
            demand_delivery_exchange=context.exchange, run_frequency=mode,
            requested_long_goal_miles=latest.goal_long_miles if latest else goal,
            requested_run_frequency=latest.frequency_mode if latest else mode,
            long_run_weekday=weekday,
            requested_long_run_weekday=latest.long_run_weekday if latest else weekday,
            long_run_not_before=applied.long_run_not_before if applied else None,
            requested_long_run_not_before=latest.long_run_not_before if latest else None,
            preferences_revision=latest.key if latest else 'adopted:'+snapshot.fingerprint,
            preferences_pending=requested is not None and (requested.goal_long_miles, requested.frequency_mode,
                requested.long_run_weekday,requested.long_run_not_before) != (goal, mode,weekday,
                    applied.long_run_not_before if applied else None),
            preferences_requested_at=latest.available_at if latest else None)
    except (OSError, sqlite3.Error, ValueError, KeyError, TypeError, AttributeError, ZeroDivisionError):
        return PlannerSettingsStatus(**unavailable)


def validate_settings_change(before, after, *, connection=None):
    """Reject unsupported edits BEFORE writing or reprocessing physical evidence."""
    if not enabled(before):
        return
    if before.get('moving_time') != after.get('moving_time'):
        raise ValueError('The active adaptive planner requires a reviewed evidence migration before changing movement detection. No settings were saved.')
    if any(before.get(k) != after.get(k) for k in ('max_hr','resting_hr','zones')):
        from .hr_reference import frozen_live_config
        try:
            if connection is None:
                raise ValueError('No verified adopted HR accounting reference')
            if (connection.execute("SELECT 1 FROM app_state WHERE key='adaptive_runtime'").fetchone()
                    or not (before.get('training_background') or before.get('beginner_start'))):
                frozen_live_config(after, connection=connection)
        except (OSError, sqlite3.Error, ValueError, KeyError, TypeError) as exc:
            import re
            detail=re.sub(r'(?i)(?:[.;]\s*)?no settings were saved\.?\s*$','',str(exc)).rstrip('.; ')
            raise ValueError(f'Could not apply HR settings to the saved planner: {detail}. No settings were saved.') from exc
    from run_analysis.settings_service import settings_response
    old = settings_response(before).coaching.model_dump(mode='python')
    new = settings_response(after).coaching.model_dump(mode='python')
    # Race goals still feed analysis, and strength is a real materializer input.
    supported = {'training_goal','goal_date','goal_pace_min_mile','strength_training','quality_sessions','quality_focus'}
    if any(old.get(k) != new.get(k) for k in (old.keys() | new.keys()) - supported):
        raise ValueError('Legacy planner weights do not control the active adaptive scheduler. No settings were saved; use a versioned adaptive policy transition.')


def save_planner_preferences(connection, config, patch, *, as_of):
    try:
        return _save_planner_preferences(connection, config, patch, as_of=as_of)
    except (OSError, sqlite3.Error, KeyError, TypeError, AttributeError) as error:
        raise ValueError('The adopted preference journal needs review. Reload Settings before trying again; the issued plan was not changed.') from error


def _save_planner_preferences(connection, config, patch, *, as_of):
    """Queue within the SAME runtime lock used for search/upload/publication."""
    import fcntl
    from .live_bridge import _LOCK
    from .planner_preferences import queue_request
    if not enabled(config):
        raise ValueError('Planner preferences require the active adaptive scheduler')
    row = connection.execute("SELECT value_json FROM app_state WHERE key='adaptive_runtime'").fetchone()
    if row is None:
        raise ValueError('No adopted adaptive runtime; no preferences saved')
    pointer = json.loads(row[0])
    if pointer['engine'] != ENGINE:
        raise ValueError('Adaptive runtime engine mismatch; no preferences saved')
    root = Path(pointer['root'])
    if not (root/'sandbox.sqlite').exists():
        raise ValueError('Adopted adaptive runtime is missing; no preferences saved')
    with _LOCK, (root/'live-bridge.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        with sqlite3.connect((root/'sandbox.sqlite').resolve().as_uri()+'?mode=rw',uri=True) as db:
            row = db.execute('SELECT event,snapshot FROM sandbox_decisions ORDER BY sequence DESC LIMIT 1').fetchone()
            if row is None:
                raise ValueError('No issued adaptive decision; no preferences saved')
            event = json.loads(row[0])
            state = restore(row[1],expected_fingerprint=event['snapshot_fingerprint'])
            if state.frequency_contract is None or state.frequency_contract.version != 'frequency_midpoint_v3':
                raise ValueError('Unsupported adaptive policy; no preferences saved')
            return queue_request(db,state=state,stage=event['stage_runtime']['stage'],patch=patch,as_of=as_of,
                prior_program=event.get('program',()))
