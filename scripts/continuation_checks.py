"""Shared offline sequence checks; not a complete production acceptance gate."""

from run_analysis import weekly_schedule as planner
from run_analysis.selection_objective import RECOVERY_BACKEND


def check_sequence(days, inputs, state_at, edit_offsets):
    """Reproject all following sessions; distinguish guards from dose-menu coverage."""
    planned, checks, menu = [], [], []
    for i, day in enumerate(days):
        desired = day.recommendation
        if desired is None:
            continue
        role = ('long' if desired.workout_type == planner.WorkoutType.LONG else
                'quality' if desired.workout_type in planner.QUALITY_WORKOUT_TYPES else 'easy')
        at = desired.planned_for
        state, guarded = planner._select_budgeted_timed_recommendation(
            [state_at(at)], planned, inputs['request'], inputs['config'], weekly_role=role)
        actual_role = ('long' if guarded.workout_type == planner.WorkoutType.LONG else
                       'quality' if guarded.workout_type in planner.QUALITY_WORKOUT_TYPES else
                       'rest' if guarded.workout_type == planner.WorkoutType.REST else 'easy')
        reference = planner.projected_recovery_reference_miles(state)
        recovery = planner.estimate_recovery(state)
        residual = recovery.residual_load if recovery else 0.
        base_load = planner._recommendation_load_units(guarded, reference)
        allowed = max(0., 2. + max(0., residual + base_load - 2.) - residual)
        units = planner._recommendation_load_units(desired, reference)
        backend = RECOVERY_BACKEND.get()
        new_check = backend.check(desired, planned) if backend is not None else None
        checks.append({'offset': i, 'at': at.isoformat(), 'requested_role': role,
            'guarded_role': actual_role, 'opening_residual': residual,
            'upper_envelope_load': units, 'allowed_load': allowed,
            'role_ok': actual_role == role and guarded.readiness.value != 'not_ready',
            'load_ok': new_check['compatible'] if new_check is not None else units <= allowed + .05,
            **({'recovery_backend': new_check} if new_check is not None else {})})
        if i in edit_offsets:
            ordinary_low = planner.typical_easy_distance(state)[0]
            opening = inputs['daily_states'][0]
            target = inputs['_audit_targets'][i]
            upper = opening.typical_easy_run_miles * max(1., sum(target) / 2 /
                opening.recent_load.capacity_reference_miles)
            lower = planner._established_easy_midpoint_floor(opening)
            miles = planner._midpoint(desired)
            menu.append({'offset': i, 'target_miles': miles, 'intent_floor': lower,
                         'intent_upper': upper, 'ordinary_lower_reference': ordinary_low,
                         'within_intent_band': lower - 1e-9 <= miles <= upper + 1e-9})
        planned.append(desired)
    return {'checks': checks, 'menu': menu,
            'checks_pass': all(x['role_ok'] and x['load_ok'] for x in checks),
            'failed_offsets': [x['offset'] for x in checks if not x['role_ok'] or not x['load_ok']]}
