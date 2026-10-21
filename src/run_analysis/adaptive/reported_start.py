"""Cold-start distance anchors from a dated self-report and REAL HR calibration.

No reported session is added to history, load, progress charts or receipts.
Only an unadopted athlete can enter this path; an accepted runtime never rebases.
"""
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import json
import tempfile

from run_analysis.training_background import reported_background, calibration_activity
from run_analysis.staged_progression import DevelopmentStage, ProgressionPolicy, tenth


def candidate_from_report(inputs, config, as_of):
    from .adaptive_candidate_pipeline import compile_athlete
    report = reported_background(config)
    if report is None or not report.maintaining_training:
        raise ValueError('Enter your current maintained training level before starting a full plan')
    stage = DevelopmentStage('self-reported-start:'+config['training_background_reported_at'], as_of,
        tenth(report.weekly_miles), report.ordinary_miles, tenth(report.regular_long_miles),
        tenth(report.regular_long_miles), None,
        ProgressionPolicy(14, (1.04**2)-1, 'easy_volume', tenth(report.weekly_miles), 'gradual'),
        'Self-reported maintained exposure; not measured capacity or historical completions')
    return compile_athlete(inputs, as_of, stage=stage)


def frequency_policy(report):
    """Anchor the existing volume-linked envelope to current reported frequency.

    This is a starting coaching prior, not a fixed run-day mandate. The existing
    monotone volume relationship and future preference control remain intact.
    """
    from .frequency_range_trial import FrequencyRangePolicy
    ordinary = FrequencyRangePolicy()
    lo, hi = ordinary.bounds(report.weekly_miles)
    shift = report.runs_per_week-(lo+hi)/2
    return replace(ordinary, anchors=tuple((v,max(1.,min(7.,a+shift)),max(1.,min(7.,b+shift)))
        for v,a,b in ordinary.anchors), source='Self-reported starting frequency offset; existing volume-linked envelope')


def candidate_from_measured_start(inputs, observed, seed, config, as_of):
    """Same normal contracts; separate measured running from walking pace.

    All physical running/walking/HR load stays in inputs/history. Only the
    duration evidence used to compile running prescriptions is separated here.
    No completions or self-reported history are manufactured.
    """
    from .adaptive_candidate_pipeline import compile_athlete
    from run_analysis.staged_progression import MILE
    by_key = {str(s['id']):s for s in observed}
    duration_evidence = []
    for s in inputs['sessions']:
        measured = by_key.get(s.key)
        usable = bool(measured and measured['confirmed'] and not measured['needs_times']
            and measured['bout_s'] >= .95*measured['running_s'] and measured['running_miles'] > 0)
        duration_evidence.append(replace(s, capacity_eligible=usable,
            role='easy' if usable else s.role,
            moving_seconds=measured['running_s'] if usable else s.moving_seconds,
            meters=measured['running_miles']*MILE if usable else s.meters))
    stage = DevelopmentStage('measured-starter:'+','.join(map(str,seed['source_ids'])), as_of,
        tenth(seed['weekly_miles']), tenth(seed['ordinary_miles']), tenth(seed['regular_long_miles']),
        tenth(seed['regular_long_miles']), None,
        ProgressionPolicy(14, (1.04**2)-1, 'easy_volume', tenth(seed['weekly_miles']), 'gradual'),
        'Completed comfortable continuous running; measured starter handoff, no synthetic history')
    measured_keys = {s.key for s in duration_evidence if s.capacity_eligible}
    return compile_athlete(dict(inputs, sessions=tuple(duration_evidence),
        history=tuple(r for r in inputs['history'] if r.key in measured_keys)), as_of, stage=stage)


def measured_phase_inputs(connection, inputs, observed, config):
    """Calibrate running-rate predictions from measured running, not walk rests.

    The normal predictor is unchanged. It receives ordinary observed phase
    evidence; every source session still contributes its FULL physical load.
    """
    from run_analysis.processing import _load_points
    from run_analysis.movement import classify_movement
    from run_analysis.models import Activity
    from run_analysis.training_signals import observed_cardio_trace, HeartRateReference, integrate_cardio_trace
    from run_analysis.phase_calibration import clip_trace
    from .direct_quality_load import PhaseDose
    by_key={str(s['id']):s for s in observed if s['confirmed'] and not s['needs_times']
            and s['bout_s']>=.95*s['running_s']}
    phases=[]
    for history in inputs['phase_history']:
        measured=by_key.get(history.key)
        if measured is None:
            continue
        points=_load_points(connection,int(history.key))
        activity=Activity(activity_id=history.key,sport='Running',notes=None,creator=None,namespaces=[],laps=[],
            trackpoints=points,start_time_utc=datetime.fromisoformat(measured['start']))
        intervals=classify_movement(points,config['moving_time']).intervals
        trace=observed_cardio_trace(activity,intervals,
            HeartRateReference(config['resting_hr'],config['max_hr'],'measured starter running'),
            inputs['reference'].policy,include_spans=True,compute_density=False)
        rows=[]
        for a,b in measured['run_windows']:
            spans=clip_trace(trace['normalized_spans'],a,b)
            known=sum(s.end-s.start for s in spans)
            load=integrate_cardio_trace(spans,b,compute_density=False)['observed_load']
            rows.append(PhaseDose('warmup',b-a,None,known,load))
        phases.append(replace(history,phases=tuple(rows)))
    return dict(inputs,phase_history=tuple(phases))


def baseline_schedule(connection, config, as_of, request=None):
    from run_analysis.web.schemas import (WeeklyScheduleResponse, WeeklyScheduleDay, WeeklyTargetEvidence,
        RecommendationResponse, WorkoutStep, WorkoutType, ReadinessFlag, ConfidenceLevel)
    from run_analysis.recommendation_service import load_current_status, load_forced_rest_dates
    report=reported_background(config)
    zone=ZoneInfo(config['timezone_default']);today=as_of.astimezone(zone).date()
    status=request or load_current_status(connection, as_of=as_of)
    ready=calibration_settings_ready(connection,config)
    blocked=(not ready or not report.maintaining_training or status.health_status in {'sick_or_recovering','pain_or_injury_concern'})
    why=('Finish About you and heart-rate setup before your first recorded easy run.' if not ready else
         'Enter a comfortable current training level after a break; your former volume is not today’s starting prescription.'
         if not report.maintaining_training else
         'Running is paused for your current health report.' if blocked else
         'Record one normal easy run with GPS and heart rate. Your reported weekly volume and regular long run remain the starting anchors.')
    rests=load_forced_rest_dates(connection)
    completed_today=connection.execute("SELECT 1 FROM activities WHERE sport='Running' AND start_time_local LIKE ? LIMIT 1",
        (today.isoformat()+'%',)).fetchone()
    first=next((today+timedelta(days=i) for i in range(1 if completed_today or report.last_run_date==today else 0,7)
        if today+timedelta(days=i) not in rests),None)
    days=[]
    for offset in range(7):
        day=today+timedelta(days=offset)
        run=None
        if day==first and not blocked:
            run=RecommendationResponse(generated_at=as_of,fitness_state_as_of=as_of,
                workout_type=WorkoutType.EASY,planning_role='easy',title='Easy calibration run',
                distance_target_miles=report.ordinary_miles,
                distance_range_miles=(report.ordinary_miles,report.ordinary_miles),
                target_zones=['Z1','Z2'],structure=[WorkoutStep(
                    instruction=f'Run {report.ordinary_miles:.1f} miles at conversational Z1/Z2 effort. Record GPS and heart rate.',
                    distance_miles=report.ordinary_miles,prescription_basis='distance',target_zones=['Z1','Z2'])],
                reasons=[why],warnings=['Pace and recovery estimates are not calibrated yet.'],
                modification_rules=['Stop for pain or feeling unwell; do not force a pace.'],
                confidence=ConfidenceLevel.UNAVAILABLE,readiness=ReadinessFlag.UNKNOWN)
        days.append(WeeklyScheduleDay(date=day,recommendation=run,
            day_role='easy' if run else 'awaiting_calibration',forced_rest=day in rests,
            rationale=why if run or blocked else 'Full schedule follows your recorded easy run.'))
    result=WeeklyScheduleResponse(generated_at=as_of,start_date=today,end_date=today+timedelta(days=6),
        target_run_count=report.runs_per_week,target_distance_range_miles=(report.weekly_miles,report.weekly_miles),
        run_count=sum(d.recommendation is not None for d in days),
        projected_distance_range_miles=(report.ordinary_miles,report.ordinary_miles) if not blocked and first else (0.,0.),
        target_evidence=WeeklyTargetEvidence(recent_7d_miles=0,chronic_42d_weekly_miles=0,
            best_sustained_28d_weekly_miles=0,peak_7d_miles=0,demonstrated_run_days_per_week=0,
            current_run_days_per_week=report.runs_per_week,capacity_reference_miles=report.weekly_miles,
            planning_mode='baseline_building',baseline_session_miles=report.ordinary_miles,
            rationale='Self-reported starting exposure; not measured training history.'),
        summary=why,days=days)
    connection.execute('''INSERT INTO app_state VALUES (?,?,?) ON CONFLICT(key) DO UPDATE
        SET value_json=excluded.value_json,updated_at_utc=excluded.updated_at_utc''',
        ('weekly_schedule', result.model_dump_json(), as_of.isoformat()))
    connection.commit()
    return result


def calibration_settings_ready(connection,config):
    from run_analysis.onboarding import setup_state
    ready={str(s.step) for s in setup_state(connection,config).steps if s.complete}
    return {'profile','heart_rate','zones'} <= ready


def start_if_needed(connection, config, project_root, *, as_of=None, request=None):
    """Called only on a planner request, never from upload/settings/page reads."""
    report=reported_background(config)
    if connection.execute("SELECT 1 FROM app_state WHERE key='adaptive_runtime'").fetchone():
        return None
    if report is None:
        from .beginner_start import start_if_needed as beginner_start
        return beginner_start(connection, config, project_root, as_of=as_of, request=request)
    as_of=as_of or datetime.now(timezone.utc)
    reported_at=datetime.fromisoformat(config['training_background_reported_at'])
    from run_analysis.recommendation_service import load_current_status
    status=request or load_current_status(connection,as_of=as_of)
    if (not calibration_settings_ready(connection,config) or not report.maintaining_training
            or status.health_status in {'sick_or_recovering','pain_or_injury_concern'}
            or calibration_activity(connection,config,reported_at=reported_at,as_of=as_of) is None):
        return baseline_schedule(connection,config,as_of,request)
    from .live_bridge import _LOCK
    import fcntl
    parent=Path(project_root)/'data';parent.mkdir(parents=True,exist_ok=True)
    with _LOCK, (parent/'reported-start.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if connection.execute("SELECT 1 FROM app_state WHERE key='adaptive_runtime'").fetchone():
            return None
        return adopt(connection,config,Path(project_root),as_of)


def adopt(connection, config, project_root, as_of, *, measured_start=None):
    from run_analysis.config import resolve_project_path
    from .adaptive_candidate_pipeline import load_athlete
    from .factory import build
    from .frequency_range_trial import FrequencyContract
    from .live_bridge import (ENGINE, initialize_tracking, input_signature, attach_runtime,
        schedule, continuation_release_policy, apply_app_inputs)
    from .decision_snapshot import capture_app_inputs, evaluate_snapshot
    from .continuous_general_progression import propose
    from .quality_choices import install
    database=resolve_project_path(project_root,config['paths']['database'])
    report=reported_background(config)
    inputs=load_athlete(database,config,as_of)
    if measured_start is None:
        candidate=candidate_from_report(inputs,config,as_of)
        basis='self_report'
        starting_key='self-report:'+config['training_background_reported_at']
        provenance=dict(report=report.model_dump(mode='json'), reported_at=config['training_background_reported_at'],
            source=basis, fake_completions_added=0, measured_fitness=False)
    else:
        from types import SimpleNamespace
        observed, seed = measured_start
        inputs=measured_phase_inputs(connection,inputs,observed,config)
        candidate=candidate_from_measured_start(inputs,observed,seed,config,as_of)
        report=SimpleNamespace(**seed)
        basis='completed_starter_running'
        starting_key=candidate['stage'].key
        provenance=dict(source=basis, source_ids=seed['source_ids'], fake_completions_added=0,
            measured_start=seed, optimizer_weights_changed=False)
    parent=project_root/'data';parent.mkdir(parents=True,exist_ok=True)
    root=Path(tempfile.mkdtemp(prefix='reported-start-',dir=parent))/'runtime'
    h=build(root,config,inputs,candidate,as_of,soft_horizon_taper=True,
        continuation_boundary=True,single_volume=True,volume_linear_coefficient=.1,
        prospective_review_hour=4,purpose_continuation=True,purpose_continuation_phase='known_full_phase_v1')
    try:
        from run_analysis.program_constraints import ReportedDistanceReference
        h.reported_distance=ReportedDistanceReference(starting_key,
            as_of,h.decision(as_of),tenth(report.regular_long_miles))
        captured=capture_app_inputs(connection,captured_at=as_of,timezone=config['timezone_default'],
            activity_keys={int(r.key):r.key for r in inputs['history']})
        apply_app_inputs(h,captured)
        h.policy=replace(continuation_release_policy(h.policy),general_progression='continuous_general_v7',
            calendar_merge_evaluations=32,progression_admission='observed_rolling_v2',
            progression_credit='actual_distance_v2',volume_shortfall_policy='forgone_issued_v2')
        # The current reported level is the starting point, not a speculative
        # growth ramp or a measured recovery claim.
        h.general_progression_state=propose(h,as_of,(),longest=report.regular_long_miles)['continuous_state']
        install(h)
        preview=h.replan(as_of,reason='unissued starting-level comparator: '+basis,publish=False)
        checked=evaluate_snapshot(preview['state'],preview['planned'],preview['contract_keys'],reference=h.ref)
        h.frequency_contract=FrequencyContract.freeze(adopted_at=as_of,stage_key=h.stage.key,
            effective_on=h.decision(as_of),volume_per_7=report.weekly_miles,
            reference_fatigue=checked['evaluation'].weighted_demand_cost,
            reference_days=(preview['state'].context.end-preview['state'].context.decision).days,
            edge_fraction=.03,policy=frequency_policy(report))
        h.previous=preview['program']
        h.replan(as_of,reason='starting exposure: '+basis+'; measured easy-run calibration')
        initialize_tracking(h,connection)
        with h.sandbox.db:
            h.sandbox.db.execute('INSERT INTO adaptive_bridge_state VALUES (?,?)',('last_input',input_signature(h,captured,as_of)))
        scenario=json.loads((root/'scenario.json').read_text())
        scenario['search_policy']=asdict(h.policy)
        scenario['training_background' if measured_start is None else 'beginner_handoff']=provenance
        (root/'scenario.json').write_text(json.dumps(scenario,indent=2,allow_nan=False))
        sources={r.key:dict(connection.execute('''SELECT s.sha256 FROM source_files s
            JOIN activity_sources l ON l.source_file_id=s.id WHERE l.activity_id=? AND l.is_primary=1''',
            (int(r.key),)).fetchone())['sha256'] for r in inputs['history']}
        (root/'live-adoption-summary.json').write_text(json.dumps(dict(engine=ENGINE,
            imported_replay_completions=False,app_database_modified=False,source_set=sources,
            starting_basis=basis,measured_calibration_keys=[r.key for r in inputs['history']])))
    finally:
        h.sandbox.close()
    from run_analysis.config import load_config
    latest=load_config(project_root/'config.yaml')
    if any(latest.get(k)!=config.get(k) for k in
            ('training_background','training_background_reported_at','beginner_start','max_hr','resting_hr','zones','setup','profile')):
        raise ValueError('Setup changed while your starting plan was calculated. Refresh the Weekly Plan to use the saved settings.')
    if measured_start is not None:
        # A check-in/source/health edit during a long search invalidates the
        # measured starting decision. Do not attach a now-stale runtime.
        from run_analysis.beginner import sessions
        connection.execute('BEGIN IMMEDIATE')
        try:
            from run_analysis.recommendation_service import load_current_status
            if load_current_status(connection,as_of=as_of).health_status in {'sick_or_recovering','pain_or_injury_concern'}:
                raise ValueError('Running is paused for your current health report; the starter handoff was not applied.')
            current=sessions(connection,config,as_of)
            signature=lambda values:[(s['id'],s['fingerprint'],s['feedback']) for s in values]
            if signature(current)!=signature(measured_start[0]):
                raise ValueError('Your starter recordings or check-ins changed while planning. Refresh Weekly Plan to use the current evidence.')
            attach_runtime(connection,root,config,as_of)
        except Exception:
            connection.rollback()
            raise
    else:
        attach_runtime(connection,root,config,as_of)
    return schedule(connection,config,project_root,as_of=as_of)
