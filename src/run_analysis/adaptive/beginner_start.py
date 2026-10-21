"""Planner-request-only starter lifecycle and measured normal-planner handoff."""
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import json
from pathlib import Path

from run_analysis.beginner import StarterRecipe, sessions, progress


KEY = 'beginner_training_v1'


def starter_steps(recipe):
    from run_analysis.web.schemas import WorkoutStep
    def clock(seconds):
        return f'{seconds//60}:{seconds%60:02d}' if seconds%60 else f'{seconds//60} min'
    if recipe.running_s==recipe.bout_s:
        instruction=clock(recipe.running_s)+' · Run at conversational effort.'
    else:
        full, tail=divmod(recipe.running_s,recipe.bout_s)
        instruction=f'{full} × {clock(recipe.bout_s)} of easy running'
        if tail:
            instruction+=f', then {clock(tail)} of easy running'
        instruction+=f'. Walk {clock(recipe.walk_s)} between running stretches.'
    return [WorkoutStep(instruction='5 min · Walk easily.',prescription_basis='time',duration_minutes=5,phase='warmup'),
        WorkoutStep(instruction=instruction,prescription_basis='time',
            duration_minutes=(recipe.outing_s-600)/60,phase='easy',target_zones=['Z1','Z2']),
        WorkoutStep(instruction='5 min · Walk easily.',prescription_basis='time',duration_minutes=5,phase='cooldown')]


def _save(connection, key, value, as_of):
    connection.execute('''INSERT INTO app_state VALUES (?,?,?) ON CONFLICT(key) DO UPDATE
        SET value_json=excluded.value_json,updated_at_utc=excluded.updated_at_utc''',
        (key, json.dumps(value), as_of.isoformat()))


def start_if_needed(connection, config, project_root, *, as_of=None, request=None):
    if not config.get('beginner_start') or connection.execute("SELECT 1 FROM app_state WHERE key='adaptive_runtime'").fetchone():
        return None
    from .live_bridge import _LOCK
    import fcntl
    parent = Path(project_root)/'data'
    parent.mkdir(parents=True, exist_ok=True)
    with _LOCK, (parent/'beginner-start.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if connection.execute("SELECT 1 FROM app_state WHERE key='adaptive_runtime'").fetchone():
            return None
        return _schedule(connection, config, Path(project_root), as_of or datetime.now(timezone.utc), request)


def _schedule(connection, config, project_root, as_of, request):
    from .reported_start import calibration_settings_ready
    from run_analysis.recommendation_service import load_current_status, load_forced_rest_dates
    from run_analysis.web.schemas import (WeeklyScheduleResponse, WeeklyScheduleDay, WeeklyTargetEvidence,
        RecommendationResponse, WorkoutStep, WorkoutType, ConfidenceLevel, ReadinessFlag,
        TrailingCalendarDay, TrailingDayActivity)
    row = connection.execute('SELECT value_json FROM app_state WHERE key=?', (KEY,)).fetchone()
    saved = json.loads(row[0]) if row else dict(issues={}, version=1,
        started_at=(as_of-timedelta(days=7)).isoformat())
    observed = sessions(connection, config, as_of)
    state = progress(observed, saved['issues'], as_of)
    status = request or load_current_status(connection, as_of=as_of)
    ready = calibration_settings_ready(connection, config)
    blocked = not ready or status.health_status in {'sick_or_recovering','pain_or_injury_concern'}
    if state['handoff'] and not blocked:
        # Compilation is a real contract check, not a calendar/age checkbox.
        # Keep the starter if the normal command catalogue does not fit yet.
        from .reported_start import adopt, candidate_from_measured_start
        from .adaptive_candidate_pipeline import load_athlete
        from run_analysis.config import resolve_project_path
        inputs = load_athlete(resolve_project_path(project_root, config['paths']['database']), config, as_of)
        try:
            candidate_from_measured_start(inputs, observed, state['handoff'], config, as_of)
        except ValueError as error:
            state['handoff_wait_reason'] = str(error)
        else:
            # Any publication/reconciliation error propagates; never pretend
            # a failed normal search was a successful handoff.
            return adopt(connection, config, project_root, as_of,
                         measured_start=(observed, state['handoff']))
    zone = ZoneInfo(config['timezone_default'])
    today = as_of.astimezone(zone).date()
    recipe = StarterRecipe(**state['recipe']) if state['recipe'] else None
    latest = observed[-1] if observed else None
    pending = latest if latest and latest['healthy'] and not latest['recording_issue'] and latest['feedback'] is None else None
    if not ready:
        summary = 'Finish About you and heart-rate setup before your first recorded outing.'
    elif blocked:
        summary = 'Running is paused for your current health report.'
    elif pending:
        summary = 'Check in on your last outing before the next session is adjusted.'
    elif recipe:
        summary = ('Build comfortable running time, with a rest day between outings. '
                   'Your next increase follows two completed, comfortable sessions.')
    else:
        summary = 'Record 10–30 easy minutes, running continuously or taking walking breaks as needed.'
    if latest and latest['recording_issue'] and not blocked:
        summary = latest['recording_issue']
    if latest and latest['hard'] and not blocked:
        summary = 'Your last outing was above easy effort. Slow down or take more walking breaks; repeat before progressing.'
    if state.get('handoff_wait_reason'):
        summary = 'Keep building easy consistency; your recorded level does not yet fit the full planner’s workout menu.'
    rests = load_forced_rest_dates(connection)
    occupied = {date.fromisoformat(s['date']) for s in observed}
    last_day = max(occupied, default=today-timedelta(days=3))
    # Keep uncompleted published dates on refresh, unless today's constraints or
    # actual recordings make them impossible. No catch-up streak after a miss.
    selected = []
    candidates = sorted({date.fromisoformat(d) for d in saved['issues'] if today <= date.fromisoformat(d) <= today+timedelta(days=6)})
    candidates += [today+timedelta(days=i) for i in range(7) if today+timedelta(days=i) not in candidates]
    limit = 1 if recipe is None else 3
    for day in candidates:
        if blocked or len(selected) >= limit or day in rests or day in occupied:
            continue
        if (day-last_day).days < 2 or any(abs((day-d).days)<2 for d in selected):
            continue
        combined = occupied | set(selected) | {day}
        if any(sum(edge-timedelta(days=6) <= d <= edge for d in combined)>3
               for edge in set(selected) | {day}):
            continue
        selected.append(day)
    days = []
    for offset in range(7):
        day = today+timedelta(days=offset)
        run = None
        done = [s for s in observed if s['date'] == day.isoformat()]
        if day in selected:
            steps = []
            if recipe:
                steps = starter_steps(recipe)
                duration = (recipe.outing_s/60,)*2
            else:
                steps.append(WorkoutStep(instruction='Run easily or alternate running and walking. Record GPS and heart rate. Stop if you feel unwell or develop pain.',
                    prescription_basis='time', duration_minutes=10, target_zones=['Z1','Z2']))
                duration = (10,30)
            run = RecommendationResponse(generated_at=as_of, fitness_state_as_of=as_of,
                workout_type=WorkoutType.RUN_WALK if recipe and recipe.bout_s<recipe.running_s else WorkoutType.EASY,
                planning_role='easy', title='Easy run/walk' if recipe and recipe.bout_s<recipe.running_s else 'Easy starter run',
                duration_range_minutes=duration, target_zones=['Z1','Z2'], structure=steps,
                confidence=ConfidenceLevel.LOW, readiness=ReadinessFlag.UNKNOWN,
                reasons=[summary], modification_rules=['Keep it conversational. Walk or stop if needed; missed sessions are not made up.'])
            # A refresh never retroactively changes what a completed run owed.
            command=asdict(recipe) if recipe else None
            prior=saved['issues'].get(day.isoformat())
            if prior is None or prior.get('recipe')!=command:
                saved['issues'][day.isoformat()] = dict(issued_at=as_of.isoformat(), recipe=command)
        activities = [TrailingDayActivity(activity_id=s['id'], start_time=datetime.fromisoformat(s['start']),
            distance_miles=s['miles'], workout_type=s['workout_type'], health_tag=s['health_tag']) for s in done]
        days.append(WeeklyScheduleDay(date=day, recommendation=run, day_role='easy' if run else 'run_day' if done else 'rest_recovery_day',
            rationale='' if run or done else 'Rest between running sessions.', forced_rest=day in rests, completed_activities=activities))
    # Delete only future replaced commands; preserve dated past receipts.
    saved['issues'] = {d:v for d,v in saved['issues'].items() if date.fromisoformat(d)<today
        or date.fromisoformat(d) in occupied or date.fromisoformat(d) in selected}
    saved.update(state, as_of=as_of.isoformat())
    recent_miles = sum(s['miles'] for s in observed if datetime.fromisoformat(s['start'])>as_of-timedelta(days=7))
    feedback_view = None if not pending else dict(activity_id=pending['id'], needs_times=pending['needs_times'],
        running_minutes=round(pending['running_s']/60,1), longest_bout_minutes=round(pending['bout_s']/60,1))
    trailing = []
    for n in range(7,0,-1):
        d = today-timedelta(days=n)
        runs = [s for s in observed if s['date']==d.isoformat()]
        trailing.append(TrailingCalendarDay(date=d, day_role='run_day' if runs else 'rest_recovery_day',
            total_distance_miles=sum(s['miles'] for s in runs), activities=[TrailingDayActivity(
                activity_id=s['id'], start_time=datetime.fromisoformat(s['start']), distance_miles=s['miles'],
                workout_type=s['workout_type'], health_tag=s['health_tag']) for s in runs]))
    result = WeeklyScheduleResponse(generated_at=as_of, start_date=today, end_date=today+timedelta(days=6),
        target_run_count=limit, run_count=len(selected), target_distance_range_miles=(0,0), projected_distance_range_miles=(0,0),
        target_evidence=WeeklyTargetEvidence(recent_7d_miles=recent_miles, chronic_42d_weekly_miles=0,
            best_sustained_28d_weekly_miles=0, peak_7d_miles=recent_miles, demonstrated_run_days_per_week=0,
            capacity_reference_miles=0, planning_mode='baseline_building' if recipe else 'baseline_required',
            history_run_count=len(observed), baseline_session_minutes=recipe.running_s/60 if recipe else None,
            rationale='Measured introductory running; no inferred weekly capacity.'),
        summary=summary, days=days, planning_days=days, trailing_days=trailing,
        beginner_training=dict(status='building' if recipe else 'calibrating', summary=summary,
            running_minutes=recipe.running_s/60 if recipe else None, comfortable_sessions=state['successes'],
            next_feedback=feedback_view))
    from run_analysis.strength_schedule import (add_strength_suggestions, load_strength_rotation,
        strength_rotation_payload)
    from run_analysis.strength_completion import with_strength_records
    from run_analysis.prescription_matching import archive_weekly_prescriptions
    if not blocked:
        prior_strength=load_strength_rotation(connection,config,start_date=today)
        annotated=add_strength_suggestions(result.days,config,prior_plan=prior_strength)
        result=result.model_copy(update={'days':annotated,'planning_days':annotated})
        _save(connection,'strength_rotation',json.loads(strength_rotation_payload(annotated,prior_strength,config)),as_of)
    result=with_strength_records(connection,result,config)
    archive_weekly_prescriptions(connection,result)
    _save(connection, KEY, saved, as_of)
    _save(connection, 'weekly_schedule', result.model_dump(mode='json'), as_of)
    connection.commit()
    return result
