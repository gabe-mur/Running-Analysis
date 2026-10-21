"""Read-only Progress presentation prototype sharing released calculations.

No fitting, import, migrations, planner evaluation or journal writes happen here.
The existing pace estimator is reused, not presented as a new validated model.
"""
from dataclasses import asdict
from contextlib import closing
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfo

from .analytics import _window_estimate, build_fitness_analytics
from .progress import AVAILABLE_WINDOWS, prepare_progress_data, build_progress, _quality_performance, _trend_series
from .race_goals import goal_progress
from .processing import _load_points
from .quality_phases import detect_continuous_quality_phase
from .movement import classify_movement
from .run_feedback import assess_cardiac_drift, get_run_feedback
from .session_feedback import with_session_feedback
from .progress_comparisons import quality_comparisons, endurance_comparisons
from .web.schemas import WorkoutType, FitnessTrend, FitnessTrendPoint, PaceValue
from .vo2_estimation import estimate_local_vo2, vo2_series


def _date(value):
    return datetime.fromisoformat(value) if isinstance(value, str) else value


def _selected(items, start, end):
    return [s for s in items if start < s.start_time <= end]


def _effort_history(sessions, details, end, window_days):
    """Last-20-run purpose balance, separate from trailing-28-day HR time.

    Session share uses recorded roles; HR share reuses physical HR minutes.
    Unknown HR stays unknown. The display window changes history length only.
    """
    daily, session_daily = [], []
    quality_roles = (WorkoutType.INTERVALS, WorkoutType.TEMPO_THRESHOLD, WorkoutType.RACE)
    easy_roles = (WorkoutType.EASY, WorkoutType.RECOVERY, WorkoutType.LONG, WorkoutType.RUN_WALK)
    unknown_roles = (WorkoutType.UNKNOWN, WorkoutType.OTHER)
    # Select before excluding unknown roles from the denominator: they occupy a
    # real run slot, not permission to reach further back for classified runs.
    # Explicit non-running activities never consume one of the twenty slots.
    run_history = sorted((s for s in sessions
        if details[s.activity_id].get('workout', WorkoutType.UNKNOWN)
        in quality_roles + easy_roles + unknown_roles),
        key=lambda s: (s.start_time, s.activity_id))
    for offset in range(window_days + 1):
        at = end - timedelta(days=window_days - offset)
        selected = _selected(sessions, at - timedelta(days=28), at)
        loads = [details[s.activity_id]['load'] for s in selected]
        # Session purpose is independent of HR-zone minutes. Incidental Z3 on
        # an easy run is not quality; easy longs remain low-intensity sessions.
        recent_runs = [s for s in run_history if s.start_time <= at][-20:]
        roles = [details[s.activity_id].get('workout', WorkoutType.UNKNOWN) for s in recent_runs]
        quality = sum(role in quality_roles for role in roles)
        easy_sessions = sum(role in easy_roles for role in roles)
        unknown = sum(role in unknown_roles for role in roles)
        classified = quality + easy_sessions
        session_daily.append(dict(as_of=at.isoformat(), quality_sessions=quality,
            easy_sessions=easy_sessions, classified_runs=classified, unknown_runs=unknown,
            completed_runs=len(recent_runs),
            quality_percent=quality / classified * 100 if classified else None))
        easy = sum(r.easy_minutes for r in loads)
        moderate = sum(r.moderate_minutes for r in loads)
        hard = sum(r.hard_minutes for r in loads)
        missing = sum(r.unknown_hr_minutes for r in loads)
        known = easy + moderate + hard
        daily.append(dict(as_of=at.isoformat(), easy_minutes=easy,
            moderate_minutes=moderate, hard_minutes=hard,
            known_hr_minutes=known, missing_hr_minutes=missing,
            hr_coverage=known / (known + missing) if known + missing else None,
            easy_percent=easy / known * 100 if known else None,
            above_easy_percent=(moderate + hard) / known * 100 if known else None))
    return dict(trailing_days=28, current=daily[-1], daily=daily,
        session_balance=dict(trailing_runs=20, current=session_daily[-1], daily=session_daily,
            reference=dict(low=20., high=25., basis='Approximate coaching reference; not physiological thresholds')))


def _exposure(sessions, start, end, zone):
    selected = _selected(sessions, start, end)
    dates = sorted({s.start_time.astimezone(zone).date() for s in selected})
    streak = longest = 0
    last = None
    for day in dates:
        streak = streak + 1 if last and (day - last).days == 1 else 1
        longest = max(longest, streak)
        last = day
    return dict(run_count=len(selected), running_days=len(dates),
        distance_miles=sum(s.distance_miles for s in selected),
        moving_minutes=sum(s.moving_minutes for s in selected),
        longest_miles=max((s.distance_miles for s in selected), default=None),
        longest_minutes=max((s.moving_minutes for s in selected), default=None),
        longest_streak_days=longest,
        longest_internal_no_run_gap_days=max(((b-a).days-1 for a,b in zip(dates,dates[1:])), default=None))


def _canonical_load(connection, root, sessions, end, *, details=None):
    """Use verified committed observed accounting, never silently rebuild it.

    Missing, stale or identity-mismatched rows remain missing. Snapshot coverage
    is intentionally not treated as coverage of the whole imported database.
    """
    empty = dict(status='unavailable', reason='No committed adaptive load ledger.', rows=[], snapshot_as_of=None, goal_long_miles=None)
    row = connection.execute("SELECT value_json FROM app_state WHERE key='adaptive_runtime'").fetchone()
    if not row:
        return empty
    try:
        pointer = json.loads(row[0])
        runtime_root = Path(pointer['root'])
        if not runtime_root.is_absolute():
            runtime_root = root / runtime_root
        with closing(sqlite3.connect(f'{(runtime_root / "sandbox.sqlite").as_uri()}?mode=ro', uri=True)) as runtime:
            runtime.execute('BEGIN')
            committed = runtime.execute('SELECT event,snapshot FROM sandbox_decisions ORDER BY sequence DESC LIMIT 1').fetchone()
            from .adaptive.planner_preferences import latest_request
            requested = latest_request(runtime, as_of=end)
            # Newly synced runs have journal keys such as sandbox:1, not app IDs.
            tables = {r[0] for r in runtime.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            identities = {r[1]: str(r[0]) for r in runtime.execute(
                'SELECT activity_id,physical_key FROM adaptive_app_sources')} if 'adaptive_app_sources' in tables else {}
        if not committed:
            return empty
        from .adaptive.decision_snapshot import restore
        event = json.loads(committed[0])
        state = restore(committed[1], expected_fingerprint=event['snapshot_fingerprint'])
        by_id = {str(s.activity_id): s for s in sessions}
        rows = []
        for run in state.history:
            physical = by_id.get(identities.get(run.key, run.key))
            if (run.source != 'observed' or not physical or run.end > end
                or abs((run.start-physical.start_time).total_seconds()) > 1
                or abs(run.miles-physical.distance_miles) > .01):
                continue
            rows.append(dict(activity_id=physical.activity_id, start_time=run.start.isoformat(), end_time=run.end.isoformat(),
                role=run.role, running_minutes=asdict(run.demand[0]), cardio=asdict(run.demand[1]),
                work_minutes=asdict(run.demand[2]), work_basis=run.work_evidence_basis,
                accounting_key=run.accounting_key))
        stage = event.get('stage_runtime', {}).get('stage', {})
        goal = (requested.goal_long_miles if requested else
            state.planner_preferences.goal_long_miles if state.planner_preferences else
            stage.get('goal_long_miles') if stage.get('key') == state.stage_key else None)
        from .goal_journey import goal_journey
        excluded={row[0] for row in connection.execute('''SELECT a.id FROM activities a
            JOIN run_overrides o ON o.activity_id=COALESCE(NULLIF(a.activity_id,''),a.activity_uid)
            WHERE o.include_in_model=0 OR o.illness=1 OR o.health_tag NOT IN ('','normal')''')}
        details=details or {}
        completed=[dict(activity_id=s.activity_id,date=s.start_time.astimezone(ZoneInfo(state.timezone)).date().isoformat(),
            miles=s.distance_miles,role=str(details.get(s.activity_id,{}).get('workout','unknown'))) for s in sessions
            if s.activity_id not in excluded and details.get(s.activity_id,{}).get('health_tag','normal')=='normal'
            and details.get(s.activity_id,{}).get('workout') not in {WorkoutType.RUN_WALK,WorkoutType.HIKE,WorkoutType.BIKE}]
        dirty=connection.execute("SELECT 1 FROM app_state WHERE key='weekly_schedule_pending_refresh'").fetchone()
        saved=connection.execute("SELECT value_json FROM app_state WHERE key='weekly_schedule'").fetchone()
        review=bool(saved and json.loads(saved[0]).get('planner_status')=='review_required')
        from .health_reports import read_health_reports
        health=read_health_reports(connection,as_of=end)
        changed_health=bool(health and health[-1]['reported_at']>state.as_of)
        try:
            journey=goal_journey(state,event,as_of=end,completed=completed,
                needs_refresh=bool(dirty or changed_health),review_required=review,requested=requested)
        except (KeyError,TypeError,ValueError,AttributeError):
            journey=dict(available=False,reason='Goal progression is not available for this saved planner version.')
        return dict(status='available', reason=None, rows=rows, goal_long_miles=goal,
            goal_journey=journey,
            snapshot_as_of=state.as_of.isoformat(), fingerprint=event['snapshot_fingerprint'],
            fatigue_model=dict(scales=state.context.scales, timezone=state.timezone,
                accounting_key=state.context.demand_accounting_key,
                scenario=asdict(state.context.scenario),
                history_start=min((r.start for r in state.history)).isoformat() if state.history else None))
    except (OSError, sqlite3.Error, ValueError, KeyError, TypeError) as error:
        # A prototype should expose missing/incompatible evidence, not migrate it.
        return {**empty, 'reason': f'Committed ledger could not be verified ({type(error).__name__}).'}


def _load_total(rows, selected):
    ids = {s.activity_id for s in selected}
    found = [r for r in rows if r['activity_id'] in ids]
    keys = {r['accounting_key'] for r in found}
    complete = len(found) == len(ids) and len(keys) == 1
    totals = {}
    for channel in ('cardio', 'work_minutes'):
        totals[channel] = dict(low=sum(r[channel]['low'] for r in found),
            high=sum(r[channel]['high'] for r in found)) if found and len(keys) == 1 else None
    return dict(covered_runs=len(found), total_runs=len(ids), complete=complete,
        accounting_key=next(iter(keys)) if len(keys) == 1 else None, **totals)


def _drift(intervals, role):
    """Exploratory decoupling after five moving minutes, not a durability limit.

    Reuse existing steadiness/stop gates; require paired HR at both endpoints.
    Integrate exact moving-time overlaps, including an interval at a boundary.
    No terrain/heat correction, normative cutoff or tendon clearance is inferred.
    """
    legacy = assess_cardiac_drift(intervals, role)
    base = dict(available=False, decoupling_percent=None, warmup_minutes=5,
        legacy_decoupling_percent=legacy.decoupling_percent if legacy.valid else None,
        first_hr=None, second_hr=None, first_pace=None, second_pace=None)
    if not legacy.valid:
        return {**base, 'reason': legacy.reason}
    moving = [i for i in intervals if i.moving_time_s > 0 and i.distance_m > 0]
    total = sum(i.moving_time_s for i in moving)
    if total < 35*60:
        return {**base, 'reason': 'Less than 30 moving minutes remain after the 5-minute exclusion.'}
    bounds = (300., (total+300)/2, total)
    halves = []
    for a,b in zip(bounds,bounds[1:]):
        elapsed = known = distance = hr_sum = 0.
        cursor = 0.
        for i in moving:
            overlap_start = max(a, cursor)
            overlap_end = min(b, cursor+i.moving_time_s)
            overlap = max(0., overlap_end-overlap_start)
            elapsed += overlap
            hrs = (i.start.heart_rate_bpm, i.end.heart_rate_bpm)
            good = all(h is not None and h > 0 for h in hrs) and not any(f in i.flags for f in
                ('large_recording_gap','implausible_distance_speed','distance_reset','distance_unavailable'))
            if overlap and good:
                # Linear endpoint integration over only the overlapping slice.
                left = (overlap_start-cursor)/i.moving_time_s
                right = (overlap_end-cursor)/i.moving_time_s
                hr_sum += (hrs[0]+(hrs[1]-hrs[0])*(left+right)/2)*overlap
                distance += i.distance_m*overlap/i.moving_time_s
                known += overlap
            cursor += i.moving_time_s
        if known/max(elapsed,1) < .8 or distance <= 0:
            return {**base, 'reason': 'Paired HR/distance coverage is below 80% in a comparison half.'}
        halves.append(dict(hr=hr_sum/known, pace=known/60/(distance/1609.344), speed=distance/known))
    efficiency = [h['speed']/h['hr'] for h in halves]
    return {**base, 'available': True, 'decoupling_percent': (1-efficiency[1]/efficiency[0])*100,
        'first_hr': halves[0]['hr'], 'second_hr': halves[1]['hr'],
        'first_pace': halves[0]['pace'], 'second_pace': halves[1]['pace'],
        'reason': 'Warm-up excluded; raw speed/HR. Hills, heat, effort changes and HR kinetics remain confounders.'}


def _vo2(estimate, config, *, as_of, baseline, series=None):
    """Only adapt the smoothing input; never reimplement the VO2 estimator."""
    return estimate_local_vo2(
        current_pace=PaceValue(minutes_per_mile=estimate['pace_min_mile'], display='') if estimate else None,
        current_pace_uncertainty_95=estimate['uncertainty_95_min_mile'] if estimate else None,
        as_of=as_of, recent_load=baseline.current_load,
        # The selected history's pace verdict must not change the current VO2.
        fitness_trend=FitnessTrend.UNCERTAIN, config=config, series=series,
    )


def build_progress_wip(connection, config, root, window_days=28, *, as_of=None, include_exploratory=True, comparison_report=None):
    if window_days not in AVAILABLE_WINDOWS:
        raise ValueError('Unsupported Progress window')
    end = as_of or datetime.now(timezone.utc)
    if end.tzinfo is None:
        raise ValueError('Timezone-aware evaluation time required')
    zone = ZoneInfo(config.get('timezone_default','America/New_York'))
    start, prior_start = end-timedelta(days=window_days), end-timedelta(days=2*window_days)
    prepared = prepare_progress_data(connection)
    if comparison_report is not None:
        from .comparable_evidence import comparable_progress_data
        prepared = comparable_progress_data(prepared, comparison_report)
    # Keep the released view's observations, intensity mix, coverage and usual
    # volume context. Only the explicit WIP fields below use revised accounting.
    baseline = build_progress(connection, window_days, as_of=end, config=config, prepared=prepared,
        comparison_report=comparison_report)
    sessions = [s for s in prepared.sessions if s.start_time <= end]
    rows = [r for r in prepared.analytics_rows if _date(r['start_time_utc']) <= end]
    points = {p.activity_id:p for p in prepared.points if p.start_time <= end}
    raw_rows = [dict(start_time_utc=p.start_time.isoformat(), standardized_pace=p.raw_pace_min_mile,
        uncertainty_95=p.uncertainty_95_min_mile, trend_weight=p.trend_weight, distance_miles=p.distance_miles)
        for p in points.values() if p.included_in_trend and p.raw_pace_min_mile is not None]
    # Metadata uses the same last-scored-run anchor as released Progress.
    # Display verdicts/uncertainty come directly from baseline, not new gates.
    analysis = build_fitness_analytics(rows, window_days, config['target_hr'], evaluation_time=end)
    current = analysis.get('current')
    comparison = analysis.get('change_prior_window')
    prior = comparison.get('prior') if comparison else None
    slope = analysis.get('within_window_trend')
    last_observation = max((_date(r['start_time_utc']) for r in rows), default=None)
    # Fixed smoothing was requested for current VO2 only, not the pace headline.
    vo2_input = _window_estimate(rows,end,14,minimum_runs=2)
    selected = _selected(sessions,start,end)
    # Resolve lightweight reports without loading RR arrays.
    reports = with_session_feedback(connection,connection.execute('''SELECT a.id,o.perceived_exertion
        FROM activities a LEFT JOIN run_overrides o ON o.activity_id=a.activity_id''').fetchall())
    reports = {r['id']:r for r in reports}
    sensor_rows = connection.execute('''SELECT activity_id,source_file_id,
        json_extract(payload_json,'$.session_fields.total_training_effect') aerobic_te,
        json_extract(payload_json,'$.session_fields.total_anaerobic_training_effect') anaerobic_te,
        json_array_length(payload_json,'$.rr_batches') rr_batches,
        json_extract(payload_json,'$.dfa_status') dfa_status
        FROM activity_sensor_streams ORDER BY source_file_id DESC''').fetchall()
    sensors = {}
    for r in sensor_rows:
        sensors.setdefault(r['activity_id'],dict(r))
    ledger = _canonical_load(connection,Path(root),sessions,end,details=prepared.details)
    load_by_id = {r['activity_id']:r for r in ledger['rows']}
    contexts = {r['id']: dict(temperature_f=r['temperature_f'], dewpoint_f=r['dewpoint_f'],
        mean_abs_grade_percent=r['mean_abs_grade_percent']) for r in connection.execute('''
        SELECT a.id,w.temperature_f,w.dewpoint_f,
            (SELECT SUM(ABS(s.average_grade_percent)*s.distance_m)/NULLIF(SUM(s.distance_m),0)
             FROM segments s WHERE s.activity_id=a.id AND s.average_grade_percent IS NOT NULL
               AND s.distance_m > 0) mean_abs_grade_percent
        FROM activities a LEFT JOIN activity_weather w ON w.activity_id=a.id
        WHERE a.start_time_utc_epoch <= ?''', (end.timestamp(),))} if include_exploratory else {}
    activities, insight_activities, durability, quality = [], [], [], []
    insight_start = end-timedelta(days=180)
    recorded_quality = {q.activity_id: q.model_dump(mode='json') for q in
        _quality_performance(connection,end-timedelta(days=222),end,config)} if include_exploratory else {}
    # Include pre-display observations needed by 28/42-day summaries.
    # A display boundary must not manufacture zero work or a fresh median.
    for s in reversed(_selected(sessions,end-timedelta(days=max(window_days,222)),end)):
        role = prepared.details[s.activity_id]['workout']
        point = points.get(s.activity_id)
        report = reports.get(s.activity_id,{})
        item = dict(activity_id=s.activity_id,start_time=s.start_time.isoformat(),
            distance_miles=s.distance_miles,moving_minutes=s.moving_minutes,role=role.value,
            health_tag=prepared.details[s.activity_id]['health_tag'],
            adjusted_pace=point.standardized_pace_min_mile if point else None,
            raw_pace=point.raw_pace_min_mile if point else None,
            uncertainty_95=point.uncertainty_95_min_mile if point else None,
            aerobic_eligible=bool(point and point.included_in_trend),
            context_weight=point.trend_weight if point else 0.,
            rpe=report.get('perceived_exertion'),feeling=report.get('workout_feel'),
            effort_source=report.get('effort_source'),feeling_source=report.get('feeling_source'),
            feedback_conflict=report.get('feedback_conflict',False),
            sensor=sensors.get(s.activity_id),load=load_by_id.get(s.activity_id))
        if s.start_time > start:
            activities.append(item)
        # These removed panels still have an opt-in diagnostic payload. The
        # released page needs none of their raw-trackpoint or workout analyses.
        if not include_exploratory:
            continue
        if s.start_time <= end-timedelta(days=222):
            continue
        if s.start_time > insight_start:
            insight_activities.append(item)
        if role in (WorkoutType.INTERVALS,WorkoutType.TEMPO_THRESHOLD,WorkoutType.RACE):
            feedback = get_run_feedback(connection,config,s.activity_id)
            workout_analysis = feedback.workout_analysis if feedback else None
            phase = detect_continuous_quality_phase(connection,config,s.activity_id) if role == WorkoutType.TEMPO_THRESHOLD else None
            quality.append(dict(activity_id=s.activity_id,start_time=item['start_time'],role=role.value,
                context=contexts.get(s.activity_id,{}),health_tag=item['health_tag'],
                interval=workout_analysis.interval_analysis.model_dump(mode='json') if workout_analysis and workout_analysis.interval_analysis else None,
                execution=workout_analysis.execution.model_dump(mode='json') if workout_analysis else None,
                continuous=({**asdict(phase),'pace_min_mile':phase.pace_min_mile} if phase else None),
                recorded_performance=recorded_quality.get(s.activity_id),
                prescription=workout_analysis.prescription_match.model_dump(mode='json') if workout_analysis and workout_analysis.prescription_match else None,
                recorded_work=item['load']['work_minutes'] if item['load'] else None,
                work_basis=item['load']['work_basis'] if item['load'] else None))
        elif role in (WorkoutType.EASY,WorkoutType.RECOVERY,WorkoutType.LONG) and s.moving_minutes >= 30:
            movement = classify_movement(_load_points(connection,s.activity_id),config['moving_time'])
            durability.append(dict(activity_id=s.activity_id,start_time=item['start_time'],
                moving_minutes=s.moving_minutes,distance_miles=s.distance_miles,role=role.value,
                context=contexts.get(s.activity_id,{}),health_tag=item['health_tag'],
                **_drift(movement.intervals,role)))
    quality_history,durability_history = quality,durability
    quality = [q for q in quality_history if _date(q['start_time']) > insight_start]
    durability = [r for r in durability_history if _date(r['start_time']) > insight_start]
    daily = []
    for offset in range(window_days+1):
        at = end-timedelta(days=window_days-offset)
        seven = _selected(sessions,at-timedelta(days=7),at)
        four_weeks = _selected(sessions,at-timedelta(days=28),at)
        by_role = {'easy':0.,'long':0.,'quality':0.}
        for s in seven:
            role = prepared.details[s.activity_id]['workout']
            bucket = 'long' if role==WorkoutType.LONG else 'quality' if role in (WorkoutType.INTERVALS,WorkoutType.TEMPO_THRESHOLD,WorkoutType.RACE) else 'easy'
            by_role[bucket] += s.distance_miles
        daily.append(dict(as_of=at.isoformat(),distance_7d=sum(s.distance_miles for s in seven),
            distance_by_role=by_role, distance_28d_weekly=sum(s.distance_miles for s in four_weeks)/4,
            runs_7d=len(seven),load=_load_total(ledger['rows'],seven)))
    trend = [p.model_dump(mode='json') for p in baseline.trend_28d]
    raw_trend = [p.model_dump(mode='json') for p in _trend_series(raw_rows,28) if p.as_of > start]
    vo2_inputs = []
    for offset in range(window_days+1):
        at = end-timedelta(days=window_days-offset)
        estimate14 = _window_estimate(rows,at,14,minimum_runs=2)
        if estimate14:
            vo2_inputs.append(FitnessTrendPoint(as_of=at,
                **{k:estimate14[k] for k in ('pace_min_mile','uncertainty_95_min_mile','run_count')}))
    # Shared curve and headline preserve pooling, disagreement handling and
    # max-HR uncertainty. Missing current evidence does not erase valid history.
    curve = vo2_series(vo2_inputs,config=config,as_of=end)
    vo2 = _vo2(vo2_input,config,as_of=end,baseline=baseline,series=curve)
    role_summary = []
    for role in sorted({a['role'] for a in activities}):
        group = [a for a in activities if a['role']==role]
        role_summary.append(dict(role=role,runs=len(group),distance_miles=sum(a['distance_miles'] for a in group),
            minutes=sum(a['moving_minutes'] for a in group)))
    from .progress_fatigue import fatigue_history
    fatigue = fatigue_history(ledger, sessions, points, prepared.details, end, window_days)
    from .progress_training_load import recorded_cardio, training_load_history
    training_load = training_load_history(recorded_cardio(connection, config, end),
        sessions, points, prepared.details, end, window_days, str(zone))
    return dict(version='progress_wip_v8',as_of=end.isoformat(),timezone=str(zone),window_days=window_days,
        baseline=baseline.model_dump(mode='json'),race_goal=asdict(goal_progress(connection,config,as_of=end.astimezone(zone).date())),
        long_goal_miles=ledger['goal_long_miles'],
        goal_journey=ledger.get('goal_journey',dict(available=False,
            reason='Create or refresh your Weekly Plan to see goal progression.')),
        window_start=start.isoformat(),prior_window_start=prior_start.isoformat(),
        target_hr=config['target_hr'],reference_minutes=config['reference_conditions']['within_run_minutes'],
        reference_conditions=config['reference_conditions'],available_windows=list(AVAILABLE_WINDOWS),
        performance=dict(current=current,vo2_input=vo2_input,vo2_input_days=14,
            prior=prior,comparison=comparison,trajectory=slope,rolling_28d=trend,rolling_raw_28d=raw_trend,
            period_change=baseline.period_change.model_dump(mode='json') if baseline.period_change else None,
            within_window_trend=baseline.within_window_trend.model_dump(mode='json') if baseline.within_window_trend else None,
            fitness_trend=baseline.fitness_trend.value,confidence=baseline.fitness_confidence.value,
            last_observation=last_observation,vo2=vo2.model_dump(mode='json'),
            vo2_series=[p.model_dump(mode='json') for p in curve]),
        exposure=dict(current=_exposure(sessions,start,end,zone),prior=_exposure(sessions,prior_start,start,zone),
            roles=role_summary,daily=daily),
        effort=_effort_history(sessions, prepared.details, end, window_days),
        fatigue=fatigue,
        training_load=training_load,
        load=dict(status=ledger['status'],reason=ledger['reason'],snapshot_as_of=ledger['snapshot_as_of'],
            current=_load_total(ledger['rows'],selected),prior=_load_total(ledger['rows'],_selected(sessions,prior_start,start))),
        activities=activities,quality=quality,durability=durability,
        insights=dict(days=180,start=insight_start.isoformat(),activities=insight_activities,
            quality_history=quality_history,durability_history=durability_history,
            quality_comparisons=quality_comparisons([q for q in quality if q['health_tag']=='normal'],end),
            endurance_comparisons=endurance_comparisons(durability,end)),
        signals=dict(reports=sum(a['rpe'] is not None or a['feeling'] is not None for a in activities),
            rr_runs=sum(bool(a['sensor'] and a['sensor']['rr_batches']) for a in activities),
            dfa_status='deferred',fitness_model_refit=False,planner_influence=False))
