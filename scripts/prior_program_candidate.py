"""Recover and causally translate a prior program for ordinary revalidation.

The incumbent is a candidate, never a lock. Same-boundary programs retain their
exact full sequence. On an advanced origin, consume only resolved elapsed dates,
promote newly hard prescriptions to native candidate times, preserve the remaining
physical program, and leave explicit terminal repair to the selector.
"""
from datetime import timedelta

from run_analysis import weekly_schedule as planner
from run_analysis.selection_objective import COMPLETED_SELECTION_HISTORY


def _saved_full(prior, saved):
    try:
        hard=list(prior.planning_days or prior.days)
        soft=[planner.WeeklyScheduleDay(date=r['date'],day_role=r['role'],
            recommendation=planner.RecommendationResponse.model_validate(dict(r['prescription'],
                generated_at=r['prescription']['fitness_state_as_of'])) if r['prescription'] else None,
            rationale='Prior continuation submitted for current-input revalidation.')
            for r in saved.get('soft_sessions',[])]
    except (KeyError,TypeError,ValueError) as error:
        return None,None,dict(status='rejected',reason='invalid_saved_continuation',detail=str(error))
    return hard,hard+soft,None


def _resolved_elapsed(elapsed, origin):
    evidence=COMPLETED_SELECTION_HISTORY.get()
    runs=tuple(evidence.get('runs',())) if evidence else ()
    for day in elapsed:
        rec=day.recommendation
        if rec is None or rec.workout_type==planner.WorkoutType.REST:
            continue
        matches=[run for run in runs if run.start_time<origin
            and run.start_time.astimezone(origin.tzinfo).date()==day.date]
        if not any((run.completed_prescribed_workout is True
                    and (run.prescribed_workout_type is None or run.prescribed_workout_type==rec.workout_type))
                   or run.start_time==rec.planned_for for run in matches):
            return False,day.date.isoformat()
    return True,None


def _promote_hard_times(hard,full,inputs):
    """Retain prescriptions but use only current-frame native hard times."""
    promoted=list(hard); promoted_full=list(full); changes=[]
    options=inputs.get('daily_state_options') or [[state] for state in inputs['daily_states']]
    for i,day in enumerate(hard):
        rec=day.recommendation
        if rec is None or rec.workout_type==planner.WorkoutType.REST:
            continue
        at=rec.planned_for or day.planned_at
        native=[state.as_of for state in options[i] if state.as_of.date()==day.date]
        if not native:
            return None,None,[],dict(status='not_applicable',reason='no_native_time',date=day.date.isoformat())
        selected=at if at in native else min(native,key=lambda value:(abs((value-at).total_seconds()),value))
        if selected==at:
            continue
        revised=rec.model_copy(update={'planned_for':selected,'fitness_state_as_of':selected})
        replacement=day.model_copy(update={'planned_at':selected,'recommendation':revised})
        promoted[i]=replacement; promoted_full[i]=replacement
        changes.append(dict(date=day.date.isoformat(),before=at.isoformat(),after=selected.isoformat()))
    return promoted,promoted_full,changes,None


def recover_prior_program(inputs,continuation_days):
    from scripts.continuation_replan import fingerprint
    prior=inputs.get('prior_schedule')
    diagnostics=prior.planner_diagnostics if prior is not None else None
    saved=diagnostics.continuation_selection if diagnostics is not None else None
    if not saved:
        return None,dict(status='unavailable',reason='no_saved_full_program')
    hard,full,error=_saved_full(prior,saved)
    if error:
        return None,error
    expected_old=[hard[0].date+timedelta(days=i) for i in range(len(hard)+continuation_days)]
    if [d.date for d in full]!=expected_old:
        return None,dict(status='not_applicable',reason='soft_boundary_changed_or_missing')
    if (fingerprint(hard)!=saved.get('selected_hard_fingerprint')
            or fingerprint(full)!=saved.get('selected_program_fingerprint')):
        return None,dict(status='rejected',reason='saved_program_identity_mismatch')
    origin_at=inputs['daily_states'][0].as_of; origin=origin_at.date()
    current_dates=[origin+timedelta(days=i) for i in range(len(inputs['daily_states']))]
    if [d.date for d in hard]==current_dates:
        offsets=[i for i,d in enumerate(hard) if d.recommendation
            and d.recommendation.workout_type!=planner.WorkoutType.REST]
        return dict(source='prior_program',days=hard,offsets=offsets,fixed_program=full),dict(
            status='pending_revalidation',translation='same_boundary',hard_fingerprint=fingerprint(hard),
            full_fingerprint=fingerprint(full),promoted_times=[])
    if origin<=full[0].date or origin>full[-1].date:
        return None,dict(status='not_applicable',reason='unsupported_boundary_shift')
    elapsed=[d for d in full if d.date<origin]
    resolved,unresolved=_resolved_elapsed(elapsed,origin_at)
    if not resolved:
        return None,dict(status='not_applicable',reason='elapsed_prescription_unresolved',date=unresolved)
    retained=[d for d in full if d.date>=origin]
    if len(retained)<len(inputs['daily_states']):
        return None,dict(status='not_applicable',reason='insufficient_retained_hard_program')
    translated_hard=retained[:len(inputs['daily_states'])]
    if [d.date for d in translated_hard]!=current_dates:
        return None,dict(status='not_applicable',reason='translated_hard_date_gap')
    translated_hard,retained,changes,error=_promote_hard_times(translated_hard,retained,inputs)
    if error:
        return None,error
    offsets=[i for i,d in enumerate(translated_hard) if d.recommendation
        and d.recommendation.workout_type!=planner.WorkoutType.REST]
    missing=len(inputs['daily_states'])+continuation_days-len(retained)
    if not 1<=missing<=continuation_days:
        return None,dict(status='not_applicable',reason='unsupported_terminal_extension',missing_days=missing)
    return dict(source='prior_program',days=translated_hard,offsets=offsets,
        fixed_prefix=retained,terminal_missing_days=missing),dict(status='pending_revalidation',
        translation='advanced_boundary',consumed_dates=[d.date.isoformat() for d in elapsed],
        promoted_times=changes,terminal_missing_days=missing,
        hard_fingerprint=fingerprint(translated_hard),retained_fingerprint=fingerprint(retained))
