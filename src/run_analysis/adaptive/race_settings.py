"""Read-only guidance and an explicitly applied durable race request.

Draft saves never change the objective. Applying queues a dated request, not a
calendar or race clearance. Both source and committed-runtime release gates apply.
"""
from datetime import datetime, timedelta
import json
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfo

from .decision_snapshot import restore
from .live_bridge import ENGINE, enabled
from ..race_preparation import (RacePreparationGoal, marathon_preparation_options,
    short_preparation_options)
from ..staged_progression import DevelopmentStage, ProgressionPolicy

# AS156: three audited21-day daily/upload screens and all-distance phase/outcome
# seams. Not a global optimum or physiological clearance. Older runtime search
# policies still cannot Apply a race even with this source release enabled.
RACE_ACTIVATION_READY = True


def _released_race_search(event,runtime):
    """Do not enable race Apply on an older or newly cold-adopted search.

    The application release flag alone is insufficient: the latest committed
    decision must have used the shared setup covered by current daily screens.
    """
    try:
        policy=json.loads((runtime/'scenario.json').read_text())['search_policy']
    except (OSError,ValueError,KeyError,TypeError):
        return False
    refinement=(event.get('refinement') or {}).get('calendar_merge_doses') or {}
    supported=policy.get('general_progression') in {'easy_first_coordinated_v2','continuous_general_v7'}
    return supported and refinement.get('max_full_evaluations')==32 and all(policy.get(key)==value for key,value in dict(
        continuation_distribution='c1_v1',
        evaluations=10000,complete_first_search=True,final_winner_evaluations=1000,
        coordinated_final_evaluations=64,calendar_merge_evaluations=32).items())


def preview_preparation(config, project_root, draft, *, as_of):
    if as_of.tzinfo is None:
        raise ValueError('Aware assessment timestamp required')
    local = as_of.astimezone(ZoneInfo(config['timezone_default']))
    shared = dict(assessed_at=local, accepted=False, may_publish_calendar=False,
                  calendar_changed=False, draft_only=True)
    if draft.distance == 'general_fitness':
        return dict(shared, status='general_fitness', options=[],
                    detail='General fitness uses your running preferences. No race preparation is selected.')
    if draft.race_date <= local.date():
        raise ValueError('Choose a future race date')
    unavailable = dict(shared, status='evidence_review_required', options=[],
        detail='Preparation guidance needs a verified current training stage. No plan was changed.')
    if not enabled(config):
        return unavailable
    try:
        database = Path(config['paths']['database'])
        if not database.is_absolute():
            database = Path(project_root) / database
        with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True) as app:
            row = app.execute("SELECT value_json FROM app_state WHERE key='adaptive_runtime'").fetchone()
        if row is None:
            return unavailable
        pointer = json.loads(row[0])
        if pointer['engine'] != ENGINE:
            return unavailable
        runtime = Path(pointer['root']) / 'sandbox.sqlite'
        with sqlite3.connect(runtime.resolve().as_uri()+'?mode=ro', uri=True) as journal:
            journal.execute('BEGIN')
            row = journal.execute('SELECT event,snapshot FROM sandbox_decisions ORDER BY sequence DESC LIMIT 1').fetchone()
        if row is None:
            return unavailable
        event = json.loads(row[0])
        state = restore(row[1], expected_fingerprint=event['snapshot_fingerprint'])
        value = dict(event['stage_runtime']['stage'])
        value['issued_at'] = datetime.fromisoformat(value['issued_at'])
        value['policy'] = ProgressionPolicy(**value['policy'])
        stage = DevelopmentStage(**value)
        if (stage.key != state.stage_key or stage.issued_at != state.stage_issued_at
                or state.as_of > as_of or state.timezone != config['timezone_default']):
            return unavailable
        from .race_event import physical_keys
        race_keys=physical_keys(state.context)
        capacity={r.key:r.capacity_eligible for r in state.athlete.sessions}
        prior = [r for r in state.history if r.source == 'observed' and r.available_at <= as_of
                 and capacity.get(r.key,True) and r.key not in race_keys
                 and r.end <= as_of and local.date()-timedelta(days=30)
                 <= r.start.astimezone(ZoneInfo(state.timezone)).date() < local.date()]
        reference = max((r.miles for r in prior), default=None)
        shared.update(evidence_as_of=state.as_of, snapshot_fingerprint=state.fingerprint,
            longest_prior_30d_miles=reference,
            detail='Conditional guidance from the last issued plan and recorded runs. Not a race-ready verdict or an active race calendar.')
        # Unknown is not clearance. Current health must be checked by the real
        # planner at admission, not inferred from a stale snapshot or good run.
        if draft.distance == 'marathon':
            result = marathon_preparation_options(stage, race_date=draft.race_date,
                as_of=local, longest_prior_30d_miles=reference, response_state='unknown')
            return dict(result, **shared, selected_mode=draft.preparation_mode)
        goal = RacePreparationGoal(draft.distance, draft.race_date)
        result = short_preparation_options(stage, goal, as_of=local,
            longest_prior_30d_miles=reference, response_state='unknown')
        return dict(result, **shared)
    except (OSError, sqlite3.Error, ValueError, KeyError, TypeError, AttributeError):
        return unavailable


def _runtime(config, project_root):
    if not enabled(config):
        raise ValueError('Race planning requires the active adaptive scheduler')
    database = Path(config['paths']['database'])
    if not database.is_absolute():
        database = Path(project_root)/database
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as app:
        row=app.execute("SELECT value_json FROM app_state WHERE key='adaptive_runtime'").fetchone()
    if row is None:
        raise ValueError('Finish training setup before applying a race goal')
    pointer=json.loads(row[0])
    if pointer['engine'] != ENGINE:
        raise ValueError('Unsupported planner runtime')
    return Path(pointer['root'])


def goal_status(config, project_root):
    from .race_goal import latest_request
    try:
        runtime=_runtime(config,project_root)
        with sqlite3.connect((runtime/'sandbox.sqlite').resolve().as_uri()+'?mode=ro',uri=True) as db:
            db.execute('BEGIN')
            row=db.execute('SELECT event,snapshot FROM sandbox_decisions ORDER BY sequence DESC LIMIT 1').fetchone()
            if row is None:
                raise ValueError('No issued decision')
            event=json.loads(row[0])
            state=restore(row[1],expected_fingerprint=event['snapshot_fingerprint'])
            requested=latest_request(db)
            from .race_event import latest_resolution
            resolution=latest_resolution(db,state.race_goal.key) if state.race_goal and state.race_goal.goal else None
            from .program_evidence_adapter import read_accounted
            recorded={r.key:r for r in state.history}
            completions=db.execute('SELECT accounted,physical_fingerprint,available_at FROM sandbox_completions') if db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sandbox_completions'").fetchone() else ()
            from hashlib import sha256
            for physical,fingerprint,available_at in completions:
                available=datetime.fromisoformat(available_at)
                if available>datetime.now(available.tzinfo):
                    continue
                if physical is None or sha256(physical.encode()).hexdigest()!=fingerprint:
                    raise ValueError('Persisted physical evidence missing or changed')
                r=read_accounted(json.loads(physical))
                if r.available_at!=available:
                    raise ValueError('Persisted completion availability mismatch')
                recorded[r.key]=r
        latest=requested or state.race_goal
        outcome=None
        if state.race_goal and state.race_goal.goal:
            outcome=dict(expected_revision=resolution.key if resolution else state.race_goal.key,
                status=resolution.status if resolution else 'unknown',
                physical_key=resolution.physical_key if resolution else None,
                race_date=state.race_goal.race_date.isoformat(),
                report_available=datetime.now(ZoneInfo(state.timezone)).date()>=state.race_goal.race_date,
                activities=[dict(key=r.key,date=r.start.astimezone(ZoneInfo(state.timezone)).date().isoformat(),miles=r.miles)
                    for r in recorded.values() if r.source=='observed' and
                    r.start.astimezone(ZoneInfo(state.timezone)).date()==state.race_goal.race_date])
        activation=RACE_ACTIVATION_READY and _released_race_search(event,runtime)
        return dict(status='available', expected_revision=latest.key if latest else 'adopted:'+state.fingerprint,
            race_activation_available=activation,
            activation_note='Race planning is being validated; preparation preview is available, but applying a race is temporarily disabled.'
                if not RACE_ACTIVATION_READY else
                'Race Apply requires the validated shared-search runtime. Preparation previews remain available.'
                if not activation else None,
            applied=state.race_goal.payload if state.race_goal else None,
            requested=requested.payload if requested else None,
            pending=requested is not None and requested != state.race_goal,
            preparation=event.get('race_preparation'), outcome=outcome,participation_cleared=False)
    except (OSError,sqlite3.Error,ValueError,KeyError,TypeError) as error:
        return dict(status='review_required',expected_revision=None,detail=str(error),pending=False,
                    participation_cleared=False)


def save_race_request(config, project_root, draft, *, expected_revision, as_of,planner_preferences=None):
    """Explicit apply action, under the same publication lock, separate from drafts."""
    import fcntl
    from .live_bridge import _LOCK
    from .race_goal import queue_request
    if planner_preferences is not None and draft.distance!='general_fitness':
        raise ValueError('The weekly long-run target applies only to general fitness')
    preference_revision=None
    if draft.distance != 'general_fitness' and not RACE_ACTIVATION_READY:
        raise ValueError('Race activation is paused after a near-term date change in validation. Preparation previews remain available; no goal or calendar was changed.')
    runtime=_runtime(config,project_root)
    with _LOCK,(runtime/'live-bridge.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        # Review against the latest locked, identity-checked stage/evidence.
        if draft.distance != 'general_fitness':
            guidance=preview_preparation(config,project_root,draft,as_of=as_of)
            option=next((o for o in guidance.get('options',())
                         if o['mode']==draft.preparation_mode),None)
            if option is None or not option['requested_date_fits']:
                raise ValueError('This race date does not fit the conditional preparation build. Choose a later date or review preparation; no goal was applied.')
        with sqlite3.connect((runtime/'sandbox.sqlite').resolve().as_uri()+'?mode=rw',uri=True) as db:
            row=db.execute('SELECT event,snapshot FROM sandbox_decisions ORDER BY sequence DESC LIMIT 1').fetchone()
            if row is None:
                raise ValueError('No issued adaptive decision')
            event=json.loads(row[0])
            state=restore(row[1],expected_fingerprint=event['snapshot_fingerprint'])
            if draft.distance != 'general_fitness' and not _released_race_search(event,runtime):
                raise ValueError('Race activation requires the validated shared-search release. Preparation preview remains available; no goal was changed.')
            if state.frequency_contract is None or state.frequency_contract.version != 'frequency_midpoint_v3':
                raise ValueError('Unsupported adaptive policy')
            if draft.distance != 'general_fitness' and any(
                    r.start <= draft.race_date < r.end and r.available_at <= as_of for r in state.restrictions):
                raise ValueError('The race date is protected as a rest day. Resolve that conflict before applying the goal.')
            # Apply the general goal and its destination atomically. A stale
            # goal or invalid target cannot leave either request half-saved.
            with db:
                db.execute('BEGIN IMMEDIATE')
                if planner_preferences is not None:
                    from .planner_preferences import queue_request as queue_preferences
                    preference_revision=queue_preferences(db,state=state,stage=event['stage_runtime']['stage'],
                        patch=planner_preferences,as_of=as_of,prior_program=event.get('program',()),manage_transaction=False)
                revision=queue_request(db,state=state,draft=draft,expected_revision=expected_revision,as_of=as_of,
                    general_long_goal_miles=event['stage_runtime']['stage']['goal_long_miles'],manage_transaction=False,
                    general_long_goal_override=planner_preferences.long_run_goal_miles if planner_preferences else None)
    saved=revision is not None or preference_revision is not None
    from .settings_view import inspect_settings
    return dict(goal_status(config,project_root), saved=saved,
        **({'planner':inspect_settings(config,project_root).model_dump(mode='json')} if planner_preferences is not None else {}),
        calendar_changed=False,
        detail='Goal saved. It applies on the next successful Weekly Plan update; no new calendar or race clearance has been issued.'
            if saved else 'Goal unchanged; no new request or search.')


def save_race_outcome(config, project_root, *, status, physical_key, expected_revision, as_of):
    import fcntl
    from .live_bridge import _LOCK
    from .new_model_daily_harness import DailyHarness
    from .race_event import record_resolution
    runtime=_runtime(config,project_root)
    with _LOCK,(runtime/'live-bridge.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        h=DailyHarness.resume(runtime,config,as_of=as_of)
        try:
            # Includes uploaded originals even if the post-event refresh was
            # held pending this explicit report. No synthetic completion.
            h.sync_completions(as_of)
            row=record_resolution(h.sandbox.db,revision=h.race_goal,history=h.history,
                timezone_name=str(h.zone),status=status,physical_key=physical_key,
                expected_revision=expected_revision,as_of=as_of)
        finally:
            h.sandbox.close()
    return dict(goal_status(config,project_root),saved=True,calendar_changed=False,
        detail='Race outcome saved. Refresh Weekly Plan to apply recorded load and recovery.')
