"""Resolve original upload signals through the app's existing detectors.

No prescription is an input. Device workout titles and ordinary pace variation
cannot authorize interval classification. Manual metadata remains independent
user evidence; uncertain segmentation stays explicit.
"""
from datetime import timedelta
from types import SimpleNamespace

from .delivered_training import dose_from_profile,continuous_phase_from_activity
from .phase_partitions import phase_layout,detailed_layout
from .run_feedback import _infer_workout_type
from .session_feedback import with_session_feedback
from .work_stimulus_evidence import observed_work_signals
from .workload_profile import observed_profile
from .workout_detection import detect_structured_workout


def imported_feedback_response(comparison,actual):
    """Persisted FIT reports are evidence, not an invented readiness change."""
    from .session_decision import DatedResponse
    from .recovery_response import ResponseExpectation
    provenance=comparison.get('observed_evidence',{})
    if not provenance or all(provenance.get(k) is None for k in ('effort','feeling')):
        return None
    return DatedResponse(actual.key,None,actual.available_at,
        ResponseExpectation('no archived pre-run self-report expectation'),
        'original FIT feedback; effort='+str(provenance.get('effort_source'))+
        '; feeling='+str(provenance.get('feeling_source')),
        provenance.get('effort'),provenance.get('feeling'),provenance.get('feedback_conflict',False))


def resolve_original_upload(connection,activity_id,activity,config,*,available_at,
                            intervals,trace,long_threshold_miles):
    row=connection.execute('''SELECT a.id,o.workout_type,o.perceived_exertion FROM activities a
        LEFT JOIN run_overrides o ON o.activity_id=COALESCE(NULLIF(a.activity_id,''),a.activity_uid)
        WHERE a.id=?''',(activity_id,)).fetchone()
    if row is None:
        raise ValueError('Imported original activity identity required')
    detection=detect_structured_workout(connection,activity_id,intervals,z3_floor=config['zones']['z3'][0])
    workout=_infer_workout_type(row['workout_type'],detection.workout_type.value if detection else None,
        (activity.total_distance_m or 0)/1609.344,sum(i.moving_time_s for i in intervals)/60,{},long_threshold_miles)
    role=('quality' if workout.value in {'intervals','tempo_threshold'} else 'long' if workout.value=='long'
          else 'easy' if workout.value in {'easy','recovery','medium_long'} else 'unknown')
    profile=observed_profile(activity,config,workout_type=workout.value)
    end=activity.start_time_utc+timedelta(seconds=trace['elapsed_seconds'])
    dose=dose_from_profile(profile,key=f'sandbox:{activity_id}',start=activity.start_time_utc,end=end,
        available_at=available_at,continuous_phase=continuous_phase_from_activity(activity,config)
        if workout.value=='tempo_threshold' else None)
    physical=SimpleNamespace(source='observed',key=dose.key,start=dose.start,end=dose.end,
        available_at=available_at,bouts=dose.bouts,work_evidence_basis=dose.structure_basis)
    signals=observed_work_signals(physical,activity,config,intervals,trace['normalized_spans']) if dose.bouts else None
    phases=(tuple((k,b-a) for k,a,b in detailed_layout(phase_layout(signals.windows,trace['elapsed_seconds'])))
        if signals else (('warmup',trace['elapsed_seconds']),) if role in {'easy','long'} else None)
    feedback=with_session_feedback(connection,[row])[0]
    return dict(role=role,phases=phases,dose=dose,profile=profile,
        phase_evidence='explicit' if dose.structure_basis.startswith('recorded_') or role in {'easy','long'} else 'inferred',
        provenance=dict(workout_type=workout.value,
            role_source='manual_override' if row['workout_type'] else detection.source if detection else 'existing_app_distance_movement_classifier',
            structure_basis=dose.structure_basis,aligned_work=signals is not None,
            long_threshold_miles=long_threshold_miles,prescription_used_as_observation=False,
            effort=feedback['perceived_exertion'],feeling=feedback['workout_feel'],
            effort_source=feedback['effort_source'],feeling_source=feedback['feeling_source'],
            feedback_conflict=feedback['feedback_conflict']))
