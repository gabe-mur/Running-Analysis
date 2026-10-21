"""Versioned app bridge for the accepted adaptive model, never a legacy allocator.

Cold adoption owns a new durable ledger derived only from real application
evidence. An unissued comparator freezes the athlete's frequency-cost reference;
that reference is not recomputed on uploads or daily refreshes. Activation and
route wiring remain separate from these constructors.
"""
from datetime import datetime, timedelta, timezone
from dataclasses import asdict
from hashlib import sha256
import json
import sqlite3
from pathlib import Path
from threading import RLock
from zoneinfo import ZoneInfo

from .adaptive_candidate_pipeline import load_athlete, compile_easy_volume_phase
from .decision_snapshot import CapturedAppInputs, capture_app_inputs, resolve_current, evaluate_snapshot
from .factory import build, resume_service
from .frequency_range_trial import FrequencyContract
from .service import AdaptivePlannerService


ENGINE = 'adaptive_frequency_v3'
PLANNER_VERSION = 202
_LOCK = RLock()


def continuation_release_policy(policy):
    """Explicit v202 migration; generic/archived harness defaults stay linear."""
    from dataclasses import replace
    if (not policy.continuation_boundary or not policy.soft_horizon_taper
            or policy.delivery_contract != 'sustained_v1'
            or not policy.purpose_continuation
            or policy.purpose_continuation_phase != 'known_full_phase_v1'):
        raise LivePlannerReviewRequired('Continuation release requires the accepted live accounting contracts')
    return replace(policy, continuation_distribution='c1_v1', evaluations=10000,
        complete_first_search=True, final_winner_evaluations=1000,
        coordinated_final_evaluations=64)


def runtime_version(harness):
    if harness.policy.general_progression=='continuous_general_v7':
        return 217
    if harness.policy.general_progression=='continuous_general_v6':
        return 216
    if harness.policy.general_progression=='continuous_general_v5':
        return 215
    if harness.policy.general_progression=='continuous_general_v4':
        return 214
    if harness.policy.general_progression=='continuous_general_v3':
        return 213
    if harness.policy.general_progression=='continuous_general_v2':
        return 212
    if getattr(harness.policy,'volume_shortfall_policy',None)=='forgone_issued_v2':
        return 211
    if getattr(harness.policy,'volume_shortfall_policy',None):
        return 210
    if harness.policy.general_progression == 'continuous_general_v1':
        return 209
    if harness.policy.progression_credit == 'actual_distance_v2':
        return 208
    if harness.policy.progression_admission == 'observed_rolling_v2':
        return 207
    if getattr(harness,'coaching_base',None) is not None:
        return 206
    if harness.policy.continuation_distribution != 'c1_v1':
        return 201
    if harness.policy.general_progression == 'easy_first_coordinated_v2':
        return 205 if harness.policy.calendar_merge_evaluations==32 else 204
    return 203 if harness.policy.general_progression is not None else PLANNER_VERSION


def general_progression_release_policy(policy, stage):
    """Future lifecycle release only: do not restart today's accepted ramp.

    The current scorer, signature and immutable snapshots stay unchanged.
    Generic archived harnesses retain their explicit bounded-only policy.
    """
    from dataclasses import replace
    from run_analysis.general_progression import RATE_VERSION
    if (policy.continuation_distribution != 'c1_v1'
            or policy != continuation_release_policy(policy)
            or stage is None or stage.policy.volume_path != 'gradual'
            or stage.policy.growth_focus != 'easy_volume' or stage.policy.balance is not None):
        raise LivePlannerReviewRequired('General progression release requires the accepted bounded C1 stage')
    # An explicitly adopted continuous runtime must not be downgraded by the
    # ordinary bridge refresh. This does NOT opt existing live runtimes in.
    return policy if policy.general_progression in {'continuous_general_v1','continuous_general_v2','continuous_general_v3','continuous_general_v4','continuous_general_v5','continuous_general_v6','continuous_general_v7'} else replace(policy, general_progression=RATE_VERSION)


def calendar_merge_release_policy(policy, stage):
    """Explicit shared-search migration; never a race-only weight or default.

    The bounded refinement spends at most32 extra full evaluations on coupled
    calendar/dose alternatives. It changes search coverage, not the objective,
    accepted exposure, normalization or the current progression anchor.
    """
    from dataclasses import replace
    if policy != general_progression_release_policy(policy,stage):
        raise LivePlannerReviewRequired('Calendar refinement requires the accepted C1 progression contract')
    return replace(policy,calendar_merge_evaluations=32)


def observed_progression_release_policy(policy, stage):
    """Remove ordinary-admission deadlines, not rolling delivery pressure.

    Actual completions still own progression. Targets, growth rates, physical
    accounting, volume losses and continuation are deliberately unchanged.
    Generic archived harnesses retain their explicit checkpoint admission.
    """
    from dataclasses import replace
    if (stage is None or stage.policy.volume_path != 'gradual'
            or policy.delivery_contract != 'sustained_v1'
            or policy.continuation_distribution != 'c1_v1'
            or not policy.soft_horizon_taper or not policy.purpose_continuation
            or policy != continuation_release_policy(policy)):
        raise LivePlannerReviewRequired('Observed progression requires the accepted rolling-volume runtime')
    return replace(policy, progression_admission='observed_rolling_v2')


def actual_progression_credit_release_policy(policy, stage):
    """Count completed extra mileage; never rebase exposure or increase growth rates."""
    from dataclasses import replace
    if policy.progression_admission != 'observed_rolling_v2':
        raise LivePlannerReviewRequired('Actual progression credit requires observed rolling admission')
    observed_progression_release_policy(policy, stage)
    return replace(policy, progression_credit='actual_distance_v2')


def persist_release_policy(harness):
    """Persist search setup only after an authoritative C1 decision committed.

    Historical decision/snapshot bytes are never rewritten. Resume continues
    to read the scoring contract from the latest committed snapshot.
    """
    if harness.policy != continuation_release_policy(harness.policy):
        raise LivePlannerReviewRequired('Release search policy differs from the accepted setup')
    event = harness.events[-1]
    from .decision_snapshot import restore
    state = restore(
        (harness.sandbox.root/event['snapshot_file']).read_text(),
        expected_fingerprint=event['snapshot_fingerprint'])
    if state.context.continuation_distribution != 'c1_v1':
        raise LivePlannerReviewRequired('Release requires a committed C1 decision')
    path = harness.sandbox.root/'scenario.json'
    scenario = json.loads(path.read_text())
    scenario['search_policy'] = asdict(harness.policy)
    temporary = path.with_suffix('.release.tmp')
    temporary.write_text(json.dumps(scenario, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


class LivePlannerReviewRequired(ValueError):
    """An explicit integration/evidence failure, never legacy fallback or rest scoring."""


def enabled(config):
    # Internal release/rollback selection is not a user coaching preference.
    return (config.get('adaptive_planner',{}).get('engine') == ENGINE
            or bool(config.get('training_background')) or bool(config.get('beginner_start')))


def app_sources(connection):
    return {str(r['id']):dict(r) for r in connection.execute('''
        SELECT a.id,a.start_time_utc,a.updated_at_utc,s.path,s.sha256,s.imported_at_utc
        FROM activities a JOIN activity_sources l ON l.activity_id=a.id AND l.is_primary=1
        JOIN source_files s ON s.id=l.source_file_id WHERE a.sport='Running'
        ORDER BY a.start_time_utc,a.id''')}


def input_signature(h, captured, as_of):
    value=dict(day=as_of.astimezone(h.zone).date(),decision=h.decision(as_of),
        contexts=captured.contexts,restrictions=captured.restrictions,responses=captured.responses,
        metadata=captured.metadata,physical=[(r.key,r.available_at,r.miles) for r in h.history])
    from .quality_choices import VERSION,enabled_types,quality_focus
    value['quality_choices']=dict(version=VERSION,enabled=enabled_types(h.config))
    # Existing decisions keep their frozen cues. A prospective zone update
    # changes requested HR bands, not physical history or load normalization.
    # Preserve old cache signatures when the adopted zones are unchanged.
    runtime=getattr(h,'quality_runtime',None)
    if runtime and h.config['zones'] != runtime['evidence']['config']['zones']:
        value['quality_choices']['zones']=h.config['zones']
    # Preserve cached default signatures; a deliberate focus edit must replan.
    if quality_focus(h.config) != 'variety':
        value['quality_choices']['focus']=quality_focus(h.config)
    if getattr(h,'coaching_base',None) is not None:
        epochs=getattr(h,'coaching_epochs',(h.coaching_base,))
        value['coached_release']=dict(version=206,base=h.coaching_base.key,epoch=epochs[-1].key)
    if h.policy.progression_admission != 'checkpoint_v1':
        value['progression_admission'] = h.policy.progression_admission
    if h.policy.progression_credit != 'issued_nominal_v1':
        value['progression_credit'] = h.policy.progression_credit
    if h.policy.general_progression in {'continuous_general_v1','continuous_general_v2','continuous_general_v3','continuous_general_v4','continuous_general_v5','continuous_general_v6','continuous_general_v7'}:
        value['general_progression'] = h.policy.general_progression
    if getattr(h.policy,'volume_shortfall_policy',None):
        value['volume_shortfall_policy']=h.policy.volume_shortfall_policy
    # Destination-only edits do not change this bounded phase's objective.
    # Apply them at the next normal decision instead of spending search budget
    # on an unchanged calendar problem. A frequency-center edit DOES reprice.
    frequency=getattr(h,'frequency_contract',None)
    if frequency is not None and frequency.preference.center_fraction != .5:
        value['frequency_center_fraction']=frequency.preference.center_fraction
    preferences=getattr(h,'planner_preferences',None)
    if preferences is not None and preferences.long_run_weekday is not None:
        value['long_run_day']=dict(weekday=preferences.long_run_weekday,
            not_before=preferences.long_run_not_before.isoformat())
    if h.policy.continuation_distribution == 'c1_v1':
        value['continuation_release'] = dict(version=202,
            distribution=h.policy.continuation_distribution,
            evaluations=h.policy.evaluations, complete_first=h.policy.complete_first_search,
            final_winner=h.policy.final_winner_evaluations,
            coordinated_final=h.policy.coordinated_final_evaluations)
        if h.policy.calendar_merge_evaluations:
            value['continuation_release']['calendar_merge']=h.policy.calendar_merge_evaluations
    from .race_goal import requested_revision
    race = requested_revision(h,as_of)
    if race is not None:
        value['race_goal_revision']=race.key
        from .race_event import known_resolution
        resolution=known_resolution(h,race,as_of)
        if resolution is not None:
            value['race_event_revision']=resolution.key
    from .race_event import resolution_history
    outcomes=resolution_history(h,as_of)
    if outcomes:
        value['race_event_history']=tuple(r.key for r in outcomes)
    return sha256(json.dumps(value,sort_keys=True,default=lambda v:
        asdict(v) if hasattr(v,'__dataclass_fields__') else
        v.astimezone(timezone.utc).isoformat() if isinstance(v,datetime) else v.isoformat()).encode()).hexdigest()


def initialize_tracking(harness, connection):
    """One bridge identity per app activity; the adaptive journal owns load once."""
    sources=app_sources(connection)
    harness.sandbox.db.execute('''CREATE TABLE IF NOT EXISTS adaptive_app_sources(
        activity_id TEXT PRIMARY KEY,physical_key TEXT UNIQUE NOT NULL,source_sha256 TEXT NOT NULL)''')
    harness.sandbox.db.execute('''CREATE TABLE IF NOT EXISTS adaptive_bridge_state(
        key TEXT PRIMARY KEY,value_json TEXT NOT NULL)''')
    with harness.sandbox.db:
        for key in sorted(harness.seed_keys):
            if key not in sources:
                raise LivePlannerReviewRequired('An initial activity no longer exists in the application database')
            harness.sandbox.db.execute('INSERT OR IGNORE INTO adaptive_app_sources VALUES (?,?,?)',
                (key,key,sources[key]['sha256']))
        settings={k:harness.config[k] for k in ('max_hr','resting_hr','timezone_default')}
        harness.sandbox.db.execute('INSERT OR IGNORE INTO adaptive_bridge_state VALUES (?,?)',
            ('normalization',json.dumps(settings,sort_keys=True)))


def attach_runtime(connection, runtime_root, config, as_of):
    """Attach only a real-evidence, completed v3 adoption; never replay artifacts."""
    root=Path(runtime_root).resolve()
    summary=json.loads((root/'live-adoption-summary.json').read_text())
    if (summary.get('engine')!=ENGINE or summary.get('imported_replay_completions') is not False
            or summary.get('app_database_modified') is not False):
        raise LivePlannerReviewRequired('A real-evidence live adoption summary is required')
    service=resume_service(root,config,as_of=as_of)
    try:
        h=service.harness
        if h.sandbox.completed_accounted(as_of) or h.frequency_contract.version!='frequency_midpoint_v3':
            raise LivePlannerReviewRequired('Cold activation cannot import sandbox completions or an older frequency contract')
        current=app_sources(connection)
        original=summary.get('source_set')
        if original is None or any(key not in current or current[key]['sha256']!=digest
                                   for key,digest in original.items()):
            raise LivePlannerReviewRequired('The verified original source set changed or was not archived')
        initialize_tracking(h,connection)
        connection.execute('''INSERT INTO app_state(key,value_json,updated_at_utc) VALUES (?,?,?)
            ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json,updated_at_utc=excluded.updated_at_utc''',
            ('adaptive_runtime',json.dumps(dict(engine=ENGINE,root=str(root))),as_of.isoformat()))
        connection.commit()
    finally:
        service.harness.sandbox.close()


def sync_app_activity(service, connection, as_of):
    """Parse newly imported original bytes independently of the assigned workout.

    Same-file retries never add a second physical completion. Verified richer
    original FITs append dated corrections; arbitrary source changes or changed
    manual purpose still require review. Immutable issuance is never rewritten.
    """
    from run_analysis.web.upload_service import UploadPayload
    h=service.harness
    sources=app_sources(connection)
    known={r[0]:(r[1],r[2]) for r in h.sandbox.db.execute('SELECT * FROM adaptive_app_sources')}
    from .source_reconciliation import reconcile_sources
    try:
        reconcile_sources(service,connection,sources,known,as_of)
    except (OSError,ValueError) as error:
        raise LivePlannerReviewRequired(f'Existing activity source reconciliation needs attention: {error}') from error
    for activity_id,row in sources.items():
        if activity_id in known:
            continue
        start=datetime.fromisoformat(row['start_time_utc'])
        if start>=as_of or start<h.origin-timedelta(days=85):
            continue
        path=Path(row['path'])
        try:
            payload=UploadPayload(path.name,path.read_bytes())
            _,activity=h.sandbox.prepare_upload(payload)
        except (OSError, ValueError) as error:
            raise LivePlannerReviewRequired(f'Cannot reconcile original activity {activity_id}: {error}') from error
        if activity.start_time_utc!=start:
            raise LivePlannerReviewRequired('Source/activity identity mismatch; no completion imported')
        day=start.astimezone(h.zone).date().isoformat()
        earlier=[e for e in h.events if datetime.fromisoformat(e['as_of'])<=start]
        active=max(earlier,key=lambda e:datetime.fromisoformat(e['as_of']))['active'] if earlier else {}
        try:
            result=service.upload(payload,day=day if day in active else None,available_at=as_of,replan=False)
        except ValueError as error:
            raise LivePlannerReviewRequired(f'Activity {activity_id} remains imported in the app, but adaptive reconciliation requires review: {error}') from error
        physical_key=result['upload']['comparison']['actual_key']
        with h.sandbox.db:
            h.sandbox.db.execute('INSERT INTO adaptive_app_sources VALUES (?,?,?)',
                (activity_id,physical_key,row['sha256']))
    return {int(r[0]):r[1] for r in h.sandbox.db.execute('SELECT * FROM adaptive_app_sources')}


def _prior_capture(h):
    return CapturedAppInputs(h.last_as_of,(() if h.current_context is None else (h.current_context,)),
        h.restrictions,h.responses,tuple((r.key,h.capacity_eligibility.get(r.key,True),None) for r in h.history))


def weather_inputs(connection,config,root,h,as_of):
    """Existing provider/cache and absolute safety guards, not new fatigue weights.

    A flexible date is blocked only when all known still-available configured
    times are extreme. Missing forecasts never manufacture a no-run date.
    Issued contracts retain their nominal accounting clock. Weather-based
    suggestions are display-only, never a new date/dose or recovery gate.
    """
    from datetime import time
    from run_analysis.forecast import planned_forecast_options
    from run_analysis.environmental_stress import assess_training_weather
    from run_analysis.program_constraints import DateRestriction
    start=h.decision(as_of)
    hours=sorted({18,*(int(hour) for hour in config.get('weather',{}).get('automatic_run_time_hours_local',[7,12,19]))})
    groups=[]
    for offset in range(h.policy.hard_days):
        day=start+timedelta(days=offset)
        slots=[datetime.combine(day,time(hour),h.zone) for hour in hours]
        if day==as_of.astimezone(h.zone).date():
            slots=[at for at in slots if at>as_of+timedelta(minutes=10)] or [as_of]
        groups.append(slots)
    options=planned_forecast_options(connection,config,root,groups)
    blocked=[];display={};suggestions={}
    prior={r.key:r for r in h.restrictions if r.kind=='unsafe_weather'}
    for offset,rows in enumerate(options):
        day=start+timedelta(days=offset)
        # Provider results are already ranked by material weather stress with
        # stable configured-time ties. Do not prefer the 18:00 accounting clock.
        known=[(at,weather) for at,weather in rows if weather is not None]
        display[day.isoformat()]=known[0][1] if known else None
        if known:
            suggestions[day.isoformat()]=known[0][0]
        if rows and all(weather is not None and assess_training_weather(weather).extreme for _,weather in rows):
            key='app:extreme-weather:'+day.isoformat()
            blocked.append(DateRestriction(key,day,day+timedelta(days=1),
                prior[key].available_at if key in prior else as_of,'unsafe_weather','existing forecast absolute safety guard'))
        elif not rows or (any(weather is None for _,weather in rows)
                and not any(weather is not None and not assess_training_weather(weather).extreme for _,weather in rows)):
            # Provider failure is not evidence that a known hazard cleared.
            # Retain it only over its existing dated interval, never indefinitely.
            blocked.extend(r for r in prior.values() if r.start<=day<r.end)
    return tuple(blocked),display,suggestions


def schedule(connection, config, project_root, *, as_of=None, _upgrade_continuation=False,
             _upgrade_calendar_merge=False, _upgrade_coaching=None, _upgrade_progression=False):
    """All live GET/POST/upload refreshes share one durable adaptive owner."""
    from run_analysis.prescription_matching import archive_weekly_prescriptions
    from run_analysis.run_feedback import list_runs
    from .web_materialization import render
    from run_analysis.strength_schedule import load_strength_rotation, strength_rotation_payload
    from .continuous_search_adapter import NoEligibleProgram
    from datetime import timezone
    as_of=as_of or datetime.now(timezone.utc)
    pointer=connection.execute("SELECT value_json FROM app_state WHERE key='adaptive_runtime'").fetchone()
    if pointer is None:
        raise LivePlannerReviewRequired('Adaptive engine has no verified real-evidence runtime attached')
    pointer=json.loads(pointer[0])
    if pointer['engine']!=ENGINE:
        raise LivePlannerReviewRequired('Adaptive runtime engine differs from the configured engine')
    root=Path(pointer['root'])
    # Serialize across server threads AND process restarts/workers. The lock
    # guards the entire decision/completion publication, not just JSON exports.
    import fcntl
    from run_analysis.planner_observation import phase, follow_job, planner_job
    phase('waiting')
    follow_job(root)
    with _LOCK, planner_job(root), (root/'live-bridge.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        phase('history')
        try:
            from .hr_reference import frozen_live_config
            accounting_config = frozen_live_config(config, root=root)
            service=resume_service(root,accounting_config,as_of=as_of)
        except (OSError,ValueError,sqlite3.Error,KeyError,TypeError) as error:
            raise LivePlannerReviewRequired(f'Adaptive runtime requires reconciliation: {error}') from error
        try:
            h=service.harness
            if _upgrade_coaching is not None:
                # Release preparation on a coherent copy only. Activation is a
                # separately guarded pointer change, never a route parameter.
                live_db=Path(config['paths']['database'])
                from run_analysis.config import resolve_project_path
                live_db=resolve_project_path(project_root,live_db)
                with sqlite3.connect(live_db.resolve().as_uri()+'?mode=ro',uri=True) as check:
                    original=json.loads(check.execute("SELECT value_json FROM app_state WHERE key='adaptive_runtime'").fetchone()[0])
                if Path(original['root']).resolve()==root.resolve():
                    raise LivePlannerReviewRequired('Coaching release must be prepared on a copy before attachment')
                _upgrade_coaching(h,as_of)
            if _upgrade_continuation:
                # Used only on a preserved runtime/app copy before attachment.
                h.policy=continuation_release_policy(h.policy)
            elif (h.policy.continuation_distribution == 'c1_v1'
                    and h.policy != continuation_release_policy(h.policy)):
                raise LivePlannerReviewRequired('C1 live runtime search setup requires reconciliation')
            prior_policy=h.policy
            if h.policy.continuation_distribution == 'c1_v1':
                h.policy=general_progression_release_policy(h.policy,h.stage)
            if _upgrade_calendar_merge:
                # Preparation on a coherent app/runtime COPY only. Live roots
                # retain their previous policy until guarded pointer attachment.
                h.policy=calendar_merge_release_policy(h.policy,h.stage)
            if _upgrade_progression:
                h.policy=observed_progression_release_policy(h.policy,h.stage)
            from .planner_preferences import apply_request
            apply_request(h,as_of=as_of)
            stored=h.sandbox.db.execute("SELECT value_json FROM adaptive_bridge_state WHERE key='normalization'").fetchone()
            settings={k:h.config[k] for k in ('max_hr','resting_hr','timezone_default')}
            if stored is None or json.loads(stored[0])!=settings:
                raise LivePlannerReviewRequired('HR normalization/timezone changed; a reviewed evidence migration is required')
            if (h.frequency_contract is None or h.frequency_contract.version!='frequency_midpoint_v3'
                    or not h.policy.continuation_boundary or not h.policy.soft_horizon_taper
                    or not h.policy.purpose_continuation or h.policy.purpose_continuation_phase!='known_full_phase_v1'
                    or h.training_policy.purpose_weights[1]!=0):
                raise LivePlannerReviewRequired('Runtime does not use the accepted v3 model contracts; no legacy/default substitution')
            keys=sync_app_activity(service,connection,as_of)
            try:
                captured=capture_app_inputs(connection,captured_at=as_of,timezone=config['timezone_default'],
                    activity_keys=keys,previous=_prior_capture(h),
                    expectations={r.key:(r.expected_at,r.expectation) for r in h.responses})
            except ValueError as error:
                raise LivePlannerReviewRequired(f'Application evidence requires reconciliation: {error}') from error
            phase('weather')
            weather_restrictions,display_weather,suggested_times=weather_inputs(connection,config,project_root,h,as_of)
            try:
                apply_app_inputs(h,captured)
            except ValueError as error:
                raise LivePlannerReviewRequired(str(error)) from error
            h.restrictions=(*[r for r in h.restrictions if r.kind!='unsafe_weather'],*weather_restrictions)
            from dataclasses import replace
            captured=replace(captured,restrictions=h.restrictions)
            signature=input_signature(h,captured,as_of)
            old=h.sandbox.db.execute("SELECT value_json FROM adaptive_bridge_state WHERE key='last_input'").fetchone()
            review_reason=None
            current=h.current_context
            if current and current.state in {'health_concern','conflicting','unverified'}:
                review_reason='A current health/readiness report is needed before issuing a running program.'
            elif current and current.state=='little_tired' and current.applies_on<as_of.astimezone(h.zone).date():
                review_reason='The previous fatigue report is stale; report how you feel today.'
            elif old is None or old[0]!=signature:
                try:
                    phase('search')
                    decision=service.refresh(as_of,reason='live_app_refresh')
                except (NoEligibleProgram,ValueError) as error:
                    # Do not disguise missing readiness/funding/feasibility as
                    # an optimized rest week, or prevent the UI from accepting
                    # a fresh health report. The last issued journal stays intact.
                    review_reason=str(error)
                h=service.harness
                if review_reason is None:
                    if _upgrade_continuation or _upgrade_calendar_merge:
                        persist_release_policy(h)
                    # Menu revisions may acquire a new durable epoch during
                    # refresh. Store its resulting signature, not the old one,
                    # so a harmless next GET does not run a second search.
                    signature=input_signature(h,captured,as_of)
                    with h.sandbox.db:
                        h.sandbox.db.execute('INSERT INTO adaptive_bridge_state VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json',
                            ('last_input',signature))
            event=h.events[-1]
            if review_reason is None and h.policy != prior_policy:
                # Enabling the future reviewer is not evidence for changing
                # today's phase/calendar. Persist without forcing a search.
                persist_release_policy(h)
            phase('publish')
            prior_strength_plan=load_strength_rotation(connection,config,start_date=h.decision(as_of))
            result=render(h,event,config,list_runs(connection,limit=5000),
                review_reason=review_reason,weather_by_day=display_weather,
                suggested_times=suggested_times, display_as_of=as_of,
                prior_strength_plan=prior_strength_plan,
                review_as_of=as_of if review_reason else None)
            from run_analysis.strength_completion import with_strength_records
            result=with_strength_records(connection,result,config)
            # Preserve all21 real issued days for delayed upload/run-analysis
            # matching; only seven days serialize to the visible UI.
            if not review_reason:
                archive_weekly_prescriptions(connection,result.model_copy(update={'days':result.planning_days}))
            with connection:
                if not review_reason:
                    connection.execute('''INSERT INTO app_state VALUES (?,?,?)
                        ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json,updated_at_utc=excluded.updated_at_utc''',
                        ('strength_rotation',strength_rotation_payload(result.planning_days,prior_strength_plan,config),as_of.isoformat()))
                for key,value in (('weekly_schedule',result.model_dump_json()),
                        ('weekly_schedule_internal',json.dumps(dict(planner_version=runtime_version(h),
                            start_date=result.start_date.isoformat(),days=[d.model_dump(mode='json') for d in result.planning_days])))):
                    connection.execute('''INSERT INTO app_state VALUES (?,?,?)
                        ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json,updated_at_utc=excluded.updated_at_utc''',
                        (key,value,as_of.isoformat()))
            return result
        finally:
            service.harness.sandbox.close()


def apply_app_inputs(harness, captured):
    """Resolve current app inputs without inventing report times or compliance."""
    harness.current_context = resolve_current(captured.contexts,
        as_of=captured.captured_at, timezone=harness.config['timezone_default'])
    harness.restrictions = captured.restrictions
    # Captured app feedback is the source-resolved view, including manual RPE.
    # Retain standalone completions not yet associated with an app identity.
    keys = {r.key for r in captured.responses}
    harness.responses = tuple(r for r in harness.responses if r.key not in keys) + captured.responses
    for key, eligible, role in captured.metadata:
        physical = next((r for r in harness.history if r.key == key), None)
        # Match the original-file loader's explicit taxonomy, including retained
        # unknown/run-walk evidence. An unknown purpose still contributes its
        # measured running/cardio load; it is not a new quality prescription.
        roles = {'easy':'easy', 'recovery':'easy', 'medium_long':'easy', 'long':'long',
                 'intervals':'quality', 'tempo_threshold':'quality'}
        if physical is None or role is not None and roles.get(role, 'unknown') != physical.role:
            detail=(f'activity {key}: recorded {physical.role}, app {roles.get(role, "unknown")}'
                if physical is not None else f'activity {key}: physical history missing')
            raise ValueError('Manual role change needs physical-evidence reconciliation before planning '
                f'({detail})')
        harness.capacity_eligibility[key] = eligible


def cold_adopt(root, connection, config, database, as_of, *, evaluations=10000):
    """Create and issue a REAL-evidence ledger under the tested contracts.

    No replay donors, transformed completions, old-version issued calendars,
    fixed Gabe-specific fatigue number, or default/legacy factory mode.
    """
    inputs = load_athlete(Path(database), config, as_of)
    original_sources=app_sources(connection)
    baseline, candidate, proposal = compile_easy_volume_phase(inputs, as_of,
        timezone=config['timezone_default'])
    captured = capture_app_inputs(connection, captured_at=as_of,
        timezone=config['timezone_default'],
        activity_keys={int(r.key):r.key for r in inputs['history']})
    harness = build(root, config, inputs, candidate, as_of,
        baseline_candidate=baseline, evaluations=evaluations, tail_days=14,
        soft_horizon_taper=True, continuation_boundary=True, single_volume=True,
        volume_linear_coefficient=.1, prospective_review_hour=4,
        purpose_continuation=True, purpose_continuation_phase='known_full_phase_v1')
    try:
        apply_app_inputs(harness, captured)
        # Comparator is NOT issued to the athlete. Two cold searches, one
        # frequency adoption; later refreshes retain this fixed reference.
        preview = harness.replan(as_of, reason='unissued live adoption comparator', publish=False)
        checked = evaluate_snapshot(preview['state'], preview['planned'],
            preview['contract_keys'], reference=harness.ref)
        fatigue = checked['evaluation'].weighted_demand_cost
        harness.frequency_contract = FrequencyContract.freeze(adopted_at=as_of,
            stage_key=harness.stage.key, effective_on=harness.decision(as_of),
            volume_per_7=harness.stage.weekly_equivalent_miles,
            reference_fatigue=fatigue,
            reference_days=(preview['state'].context.end-preview['state'].context.decision).days,
            edge_fraction=.02)
        harness.previous = preview['program']
        event = harness.replan(as_of, reason='real-evidence frequency-v3 adoption')
        initialize_tracking(harness,connection)
        with harness.sandbox.db:
            harness.sandbox.db.execute('INSERT INTO adaptive_bridge_state VALUES (?,?)',
                ('last_input',input_signature(harness,captured,as_of)))
        return AdaptivePlannerService(harness, inputs), dict(
            engine=ENGINE, planner_version=PLANNER_VERSION,
            comparator_seconds=preview['search']['seconds'],
            comparator_cost=checked['evaluation'].total_cost,
            reference_fatigue=fatigue, reference_days=harness.frequency_contract.preference.reference_days,
            adoption_seconds=event['seconds'], proposal_status=proposal['status'],
            source_set={r.key:original_sources[r.key]['sha256'] for r in inputs['history']},
            imported_replay_completions=False, app_database_modified=False,
            stage=harness.stage, selected_event=event,
            limitations=('One bounded easy-volume phase; automatic next-phase progression remains open',
                         'Conditional model recommendations, not physiological recovery clearance'))
    except Exception:
        harness.sandbox.close()
        raise
