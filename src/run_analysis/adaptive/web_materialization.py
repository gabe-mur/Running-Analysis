"""Faithful API rendering of already scored and durably issued prescriptions.

No allocation, distance smoothing, workout RNG, recovery-hour estimate, or
secondary legacy recommendation occurs here. Soft-tail sessions are not shown
as promises; the API exposes seven of the full twenty-one issued days.
"""
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from run_analysis.strength_schedule import add_strength_suggestions
from run_analysis.web.schemas import (
    ConfidenceLevel, ReadinessFlag, RecommendationResponse, WorkoutStep,
    WeeklyScheduleDay, WeeklyScheduleResponse, WeeklyTargetEvidence,
    TrailingCalendarDay, TrailingDayActivity, WorkoutType, QualitySessionType,
)
from .live_bridge import runtime_version


def prescription_basis(role, quality_type):
    """Presentation contract for the existing workout definitions.

    Short distance repetitions and sustained timed efforts are distinct
    structures. Estimated metres/seconds do not choose the prescription unit.
    Authored workouts retain their per-step units and fractional instructions.
    """
    if role!='quality':
        return 'distance'
    return {
        QualitySessionType.SHORT_INTERVALS:'distance',
        QualitySessionType.LONG_INTERVALS:'time',
        QualitySessionType.THRESHOLD:'time',
        QualitySessionType.FARTLEK:'time',
        QualitySessionType.PROGRESSION:'portion',
        QualitySessionType.HILL_REPEATS:'time',
    }.get(quality_type,'time')


def phase_distances(harness, dose, intent):
    """Display estimates only; do not change the issued forecast or commands.

    Use the original work-bout pace when retained in the frozen contract, not
    whole-run pace for fast efforts. Allocate remaining mileage to easy phases.
    Never rebase the displayed breakdown on a later upload.
    """
    phases=intent['phases']
    total=intent['distance_miles']['target']
    seconds=sum(p['seconds']['target'] for p in phases)
    distances=[total*p['seconds']['target']/seconds for p in phases]
    contract=harness.coaching_contracts[dose.key]
    marker=';quality_structure:'
    key=contract.source.split(marker,1)[1].split(';',1)[0] if marker in contract.source else None
    original=next((r for r in harness.history if getattr(r,'key',None)==key
        and r.end<=contract.issued_at and r.available_at<=contract.issued_at),None) if key else None
    work=[i for i,p in enumerate(phases) if p['kind']=='work']
    if original and work and original.bouts:
        bouts=original.bouts
        if all(b.distance_meters is not None and b.distance_meters>0 for b in bouts):
            from run_analysis.staged_progression import MILE
            if len(bouts)>=len(work):
                measured=[b.distance_meters/MILE*p['seconds']['target']/b.work_seconds
                    for b,p in zip(bouts,(phases[i] for i in work))]
            else:
                speed=sum(b.distance_meters for b in bouts)/sum(b.work_seconds for b in bouts)
                measured=[speed*phases[i]['seconds']['target']/MILE for i in work]
            remaining=total-sum(measured)
            easy=[i for i,p in enumerate(phases) if p['kind']!='work']
            easy_seconds=sum(phases[i]['seconds']['target'] for i in easy)
            if remaining>0 and easy_seconds>0:
                for i,distance in zip(work,measured):
                    distances[i]=distance
                for i in easy:
                    distances[i]=remaining*phases[i]['seconds']['target']/easy_seconds
    return distances


def render(harness, event, config, run_history=(), *, review_reason=None, weather_by_day=None,
           review_as_of=None, suggested_times=None, display_as_of=None, prior_strength_plan=None):
    zone=ZoneInfo(config['timezone_default'])
    as_of=datetime.fromisoformat(event['as_of'])
    start=date.fromisoformat(event['decision'])
    hard_end=date.fromisoformat(event['hard_end'])
    if review_reason and review_as_of is not None:
        as_of=review_as_of
        start=harness.decision(as_of)
        hard_end=start+timedelta(days=harness.policy.hard_days)
    forced={d for r in harness.restrictions if r.kind=='reserved_rest'
        for d in (start+timedelta(days=i) for i in range((hard_end-start).days))
        if r.start<=d<r.end and r.available_at<=as_of}
    days=[]
    current=harness.current_context
    now=display_as_of or as_of
    today=now.astimezone(zone).date()
    fresh=current if current and current.available_at<=now and current.applies_on==today else None
    for offset in range((hard_end-start).days):
        day=start+timedelta(days=offset)
        issued=None if review_reason else event['active'].get(day.isoformat())
        recommendation=None
        rationale=review_reason or 'No run selected by the full projected training program.'
        role='planner_review_day' if review_reason else 'rest_recovery_day'
        if issued:
            readiness=ReadinessFlag.UNKNOWN
            readiness_label='Check on the day' if day>today else 'Check in today'
            readiness_reason=('Check how you feel on this run day; today’s check-in does not establish future recovery.'
                if day>today else 'No check-in recorded for today. Use “How do you feel now?” above; modeled training load is not verified current recovery.')
            if day==today and fresh:
                if fresh.state=='usual':
                    readiness=ReadinessFlag.READY
                    readiness_label='As planned'
                elif fresh.state=='little_tired':
                    readiness=ReadinessFlag.CAUTION
                    readiness_label='Ease if needed'
                elif fresh.state in {'health_concern','conflicting','unverified'}:
                    readiness=ReadinessFlag.NOT_READY
                    readiness_label='Check-in needs review'
                readiness_reason='Based on today’s check-in; modeled carryover is not verified current recovery.'
            dose=harness.catalog[issued['dose']]
            intent=harness.sandbox.record(issued['fingerprint']).payload['intent']
            suggested=(suggested_times or {}).get(day.isoformat())
            if suggested is not None and (suggested.tzinfo is None or suggested.astimezone(zone).date()!=day):
                raise ValueError('An advisory time cannot change the issued local run date')
            band=intent['distance_miles']
            phases=intent['phases']
            quality=dose.role=='quality'
            long=dose.role=='long'
            work=[p for p in phases if p['kind']=='work']
            contract=harness.coaching_contracts[dose.key]
            ordinary_bout=contract.ordinary.bout_seconds if quality else None
            quality_type=(QualitySessionType.LONG_INTERVALS if
                (ordinary_bout.high if ordinary_bout else max(p['seconds']['target'] for p in work))>=180
                else QualitySessionType.SHORT_INTERVALS
                ) if quality and work else None
            definition=dose.workout_definition
            if definition:
                quality_type=QualitySessionType(definition['family'])
            basis=prescription_basis(dose.role,quality_type)
            steps=[]
            distances=phase_distances(harness,dose,intent)
            for phase,distance in zip(phases,distances):
                kind=phase['kind']
                instruction=('Easy aerobic running' if not quality else
                    {'warmup':'Easy warm-up', 'cooldown':'Easy cool-down',
                     'recovery':'Easy jogging recovery',
                     'work':'Controlled hard effort; do not chase an instantaneous HR target'}[kind])
                instruction+=f" · {phase['seconds']['target']/60:g} min"
                steps.append(WorkoutStep(instruction=instruction,
                    prescription_basis=phase.get('prescription_basis',basis),
                    duration_minutes=phase['seconds']['target']/60,
                    distance_miles=distance,
                    target_zones=[] if kind=='work' else ['Z1','Z2'],phase=kind))
            if definition and definition['structure']:
                steps=[WorkoutStep.model_validate(s) for s in definition['structure']]
            recommendation=RecommendationResponse(
                generated_at=as_of,fitness_state_as_of=as_of,
                planned_for=suggested or datetime.fromisoformat(intent['suggested_start']),
                time_recommendation_basis='weather' if suggested else 'flexible',
                workout_type=WorkoutType(definition['workout_type']) if definition else WorkoutType.INTERVALS if quality else WorkoutType.LONG if long else WorkoutType.EASY,
                quality_session_type=quality_type,
                planning_role=dose.role,
                title=definition['title'] if definition else 'Controlled intervals' if quality else 'Easy long run' if long else 'Easy aerobic run',
                distance_target_miles=band['target'],
                distance_range_miles=(round(band['low'],1),round(band['high'],1)),
                duration_range_minutes=(sum(p['seconds']['low'] for p in phases)/60,
                    sum(p['seconds']['high'] for p in phases)/60),
                target_zones=definition['zones'] if definition else [] if quality else ['Z1','Z2'],structure=steps,
                confidence=ConfidenceLevel.LOW,readiness=readiness,
                readiness_reason=readiness_reason,
                readiness_label=readiness_label,
                reasons=['Selected with the complete projected running/cardio/work load, dated volume target and key-session purpose.',
                    'Distance and workout structure are the exact scored prescription, not a later substituted workout.'],
                warnings=['Suggested time is flexible. Recheck current readiness before running; stop for pain or illness.'])
            weather=(weather_by_day or {}).get(day.isoformat())
            if weather is not None:
                from run_analysis.environmental_stress import assess_training_weather
                stress=assess_training_weather(weather)
                updates=dict(planned_weather=weather)
                if stress.extreme:
                    updates.update(readiness=ReadinessFlag.NOT_READY,
                        readiness_label='Unsafe forecast',
                        readiness_reason='Unsafe weather at the nominal time; this is not an outdoor prescription for that time.',
                        warnings=[*recommendation.warnings,'Choose a safe alternate time or indoor setting; official warnings take precedence.'])
                elif stress.caution:
                    updates.update(readiness=ReadinessFlag.CAUTION,
                        readiness_label='Weather caution',
                        readiness_reason='Forecast conditions may increase effort; check how you feel on the day.',
                        warnings=[*recommendation.warnings,'Forecast conditions may increase effort. Do not chase normal-weather pace.'])
                recommendation=recommendation.model_copy(update=updates)
            role=dose.role+'_run_day'
            rationale='Selected by the adaptive whole-program evaluator.'
        race = getattr(harness,'race_goal',None)
        if not review_reason and race and race.goal and day == race.race_date:
            role='race_event_day'
            rationale=(f'{race.goal.profile.label} · {race.goal.profile.distance_miles:.1f} miles. '
                'Declared event, not an ordinary training prescription or clearance to race. '
                'Forecast race load is included; check current health and preparation before participating.')
            if event.get('race_preparation',{}).get('event_exceeds_training_session_limit',True):
                rationale+=' Event distance exceeds the ordinary training-session limit supported by recent runs; selecting this date does not clear that distance jump.'
        days.append(WeeklyScheduleDay(date=day,
            planned_at=recommendation.planned_for if recommendation else None,
            recommendation=recommendation,day_role=role,rationale=rationale,forced_rest=day in forced))
    def since(role):
        candidates=[r.start for r in harness.history if r.role==role and r.available_at<=as_of]
        return (as_of-max(candidates)).total_seconds()/86400 if candidates else None
    if not review_reason:
        completed_run_types={}
        for physical in harness.history:
            if physical.available_at > now:
                continue
            physical_day=physical.start.astimezone(zone).date()
            kind={'quality':WorkoutType.INTERVALS, 'long':WorkoutType.LONG}.get(physical.role,WorkoutType.EASY)
            completed_run_types.setdefault(physical_day,set()).add(kind)
        days=add_strength_suggestions(days,config,days_since_quality_run=since('quality'),days_since_long_run=since('long'),
            prior_plan=prior_strength_plan,completed_run_types=completed_run_types)
        days=[d.model_copy(update={'strength_suggestion':None}) if d.day_role=='race_event_day' else d for d in days]
    visible=days[:7]
    trailing=[]
    for offset in range(7,0,-1):
        day=start-timedelta(days=offset)
        runs=[r for r in run_history if r.start_time and r.start_time.astimezone(zone).date()==day]
        running=[r for r in runs if r.workout_type not in {WorkoutType.HIKE,WorkoutType.BIKE}]
        trailing.append(TrailingCalendarDay(date=day,
            day_role='run_day' if running else 'cross_training_day' if runs else 'rest_recovery_day',
            total_distance_miles=sum(r.distance_miles for r in running),
            activities=[TrailingDayActivity(activity_id=r.activity_id,start_time=r.start_time,
                distance_miles=r.distance_miles,workout_type=r.workout_type,health_tag=r.health_tag) for r in runs]))
    physical=[r for r in harness.history if r.available_at<=as_of]
    def volume(span):
        return sum(r.miles for r in physical if as_of-timedelta(days=span)<=r.start<as_of)*7/span
    def historic_peak(span):
        cutoffs=[as_of, *(r.start+timedelta(microseconds=1) for r in physical)]
        return max((sum(r.miles for r in physical if at-timedelta(days=span)<=r.start<at)*7/span
            for at in cutoffs if as_of-timedelta(days=84)<=at<=as_of),default=0.)
    volume_goal=harness.stage.policy.volume_goal_miles
    accepted=harness.stage.weekly_equivalent_miles
    target=event.get('progression_target',{}).get('planning_volume_per_7',accepted)
    count=sum(day.recommendation is not None for day in visible)
    miles=sum(day.recommendation.distance_target_miles for day in visible if day.recommendation)
    frequency=len({r.start.astimezone(zone).date() for r in physical if r.start>=as_of-timedelta(days=28)})/4
    evidence=WeeklyTargetEvidence(recent_7d_miles=volume(7),recent_14d_weekly_miles=volume(14),
        recent_21d_weekly_miles=volume(21),chronic_42d_weekly_miles=volume(42),
        best_sustained_28d_weekly_miles=historic_peak(28),peak_7d_miles=historic_peak(7),
        current_run_days_per_week=frequency,demonstrated_run_days_per_week=frequency,
        capacity_reference_miles=accepted,history_run_count=len(physical),
        baseline_session_miles=harness.stage.ordinary_miles,
        rationale='Accepted sustained stage and dated gradual target; not a calendar-week allocation or a recovery clearance.')
    summary=f'Adaptive v3: {len(days)} issued days, with a guarded soft projection. Current volume target {target:.1f} miles per seven days.'
    if event.get('progression_target',{}).get('mode')=='continuous_observed_volume':
        summary+=' Growth is reviewed against completed training, adding easy volume first.'
    elif volume_goal is not None:
        summary+=f' Phase destination {volume_goal:.1f}.'
        summary+=(' Further growth is reviewed after delivered training, adding easy volume first.'
            if harness.policy.general_progression is not None else
            ' Subsequent progression is not yet automatic.')
    race_info=event.get('race_preparation')
    if race_info:
        if race_info['phase']=='did_not_start':
            summary='Race not started. Returning to general-fitness training, using recorded load and current readiness.'
        elif race_info['phase']=='complete':
            summary='Race preparation complete. General-fitness progression remains conditional on delivered training and current readiness.'
        else:
            summary=(f"Race preparation · {race_info['distance'].replace('_',' ')} on {race_info['race_date']}. "
                f"{race_info['phase'].replace('_',' ').title()} phase. "
                'Growth remains conditional on delivered training and response review; race participation is not cleared.')
        if race_info['preparation_gap'] and race_info['phase'] in {'build','taper','race'}:
            summary+=' Current preparation remains below the selected training scenario; missed training will not trigger catch-up spikes.'
    if review_reason:
        summary='Planner review required: '+review_reason+' No replacement program has been issued.'
    return WeeklyScheduleResponse(planner_status='review_required' if review_reason else 'ready',
        planner_review_reason=review_reason,planner_version=runtime_version(harness),generated_at=as_of,
        start_date=start,end_date=start+timedelta(days=6),target_run_count=count,run_count=count,
        target_distance_range_miles=(target,target),projected_distance_range_miles=(miles,miles),
        visible_7d_scheduled_miles=miles,
        planned_14d_weekly_rate=sum(d.recommendation.distance_target_miles for d in days[:14] if d.recommendation)/2,
        target_evidence=evidence,trailing_days=trailing,summary=summary,days=visible,planning_days=days)
