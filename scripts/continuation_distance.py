"""Offline cheap base policy using the detailed planner's distance machinery.

Shares intent menus, allocation and canonical finalization; does not pretend a
one-session base policy is the full multi-session distance optimization.
"""
from contextlib import contextmanager

from scripts.audit_easy_dose_program import (dose_trajectory, history_allocator,
    history_profile_from_state)
from run_analysis import weekly_schedule as planner
from run_analysis.selection_objective import RECOVERY_BACKEND
from run_analysis.prescription_distance import UnrepresentableDistanceTarget, rounded_distance_target
from scripts.easy_dose_prototype import generate_intents


def continuation_distance_intents(opening, upper_intent, config):
    """Reuse the exact history-led menu, not a new short-run floor or grid."""
    profile = history_profile_from_state(opening, config)
    menu = generate_intents(profile,
        useful_floor_miles=planner._established_easy_midpoint_floor(opening),
        progression_allowance_miles=max(0., upper_intent - profile.ordinary_planning_miles),
        progression_source='history_trajectory:candidate_independent_program_target')
    return tuple(option.midpoint_miles for option in menu.options)


@contextmanager
def continuation_distance_policy(opening, dates, targets, config):
    profile = history_profile_from_state(opening, config)
    trajectory = dose_trajectory(profile, dates, targets,
        opening.recent_load.capacity_reference_miles, mode='history_trajectory', fixed_fraction=0.)
    intentions = dict(zip(dates, trajectory.upper_miles, strict=True))
    with history_allocator(profile, trajectory, planner._established_easy_midpoint_floor(opening)):
        yield intentions


def finalize_continuation_distance(result, projected, config, target, upper_intent):
    """Finalize one base-policy dose without reallocating any committed prefix.

    Base policy attempts the existing history/progression intention for easy
    runs; quality/long start from their selected coaching dose. Full allocation
    elsewhere can prefer a smaller member of the SAME menu. No funding debt or
    extra rest is used to raise the intention. A failed expansion returns to the
    original guarded dose rather than weakening the recovery limit.
    """
    original = result
    def guarded_original():
        # Same non-enlarging canonical convention as the detailed allocator's
        # fixed/caution path, not nearest rounding above the guarded center.
        return original.model_copy(update={'distance_target_miles': rounded_distance_target(
            planner._midpoint(original), minimum=original.distance_range_miles[0],
            maximum=planner._midpoint(original))})
    intent = (upper_intent if result.workout_type == planner.WorkoutType.EASY
              else planner._midpoint(result))
    day = planner.WeeklyScheduleDay(date=result.planned_for.date(),
        planned_at=result.planned_for, recommendation=result,
        day_role=('long_run' if result.workout_type == planner.WorkoutType.LONG
                  else 'quality_run' if result.workout_type in planner.QUALITY_WORKOUT_TYPES
                  else 'easy_run'), rationale='Offline base-policy dose.')
    try:
        allocated = planner._allocate_visible_distance_ranges([day], [projected],
            (intent, intent), config, weekly_target_range=target,
            preserve_projected_recovery_caps=True,
            allocation_candidate_limit=planner.ALLOCATION_ASSIGNMENTS_PER_TOTAL)[0].recommendation
    except UnrepresentableDistanceTarget as error:
        # A singleton can have only a preferred quarter-mile option that is
        # not representable under its cap. Do not relax that cap. The base
        # policy may retain the original guarded recommendation instead.
        return guarded_original(), {'intent_miles': intent, 'guarded_dose_fallback': True,
                          'fallback_reason': str(error)}
    reference = planner.projected_recovery_reference_miles(projected)
    recovery = planner.estimate_recovery(projected)
    residual = recovery.residual_load if recovery else 0.
    base_load = planner._recommendation_load_units(original, reference)
    # Same ceiling as the detailed allocator's apply_recovery_caps. Its normal
    # loop handles preceding records; here their load is in projected already.
    allowed_load = max(0., 2. + max(0., residual + base_load - 2.) - residual)
    selected_load = planner._recommendation_load_units(allocated, reference)
    backend = RECOVERY_BACKEND.get()
    fallback = (not backend.check_state(projected, allocated)['compatible']
                if backend is not None else selected_load > allowed_load + .05)
    return (guarded_original() if fallback else allocated), {
        'intent_miles': intent, 'allocated_miles': planner._midpoint(allocated),
        'upper_envelope_load': selected_load, 'allowed_load': allowed_load,
        'guarded_dose_fallback': fallback,
    }
