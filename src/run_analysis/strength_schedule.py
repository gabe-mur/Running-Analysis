"""Attach non-binding strength suggestions to an already-finalized run plan.

This module deliberately runs after the running optimizer.  Strength guidance
may react to run placement, but it must never move a run or change its dose.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

from .web.schemas import (
    StrengthSessionType,
    StrengthSuggestion,
    WeeklyScheduleDay,
    WorkoutType,
)


TAXING_RUN_TYPES = frozenset(
    {
        WorkoutType.LONG,
        WorkoutType.INTERVALS,
        WorkoutType.TEMPO_THRESHOLD,
        WorkoutType.RACE,
    }
)
EASY_RUN_TYPES = frozenset({WorkoutType.EASY, WorkoutType.RECOVERY})

DEFAULT_MUSCLE_GROUPS: dict[StrengthSessionType, tuple[str, ...]] = {
    StrengthSessionType.FULL_UPPER: (
        "upper pecs",
        "lower pecs",
        "lats",
        "upper back",
        "side delts",
        "rear delts",
        "biceps",
        "triceps",
        "core/trunk",
    ),
    StrengthSessionType.PUSH: (
        "upper pecs",
        "lower pecs",
        "side delts",
        "triceps",
        "core/trunk",
    ),
    StrengthSessionType.PULL: (
        "lats",
        "upper back",
        "rear delts",
        "biceps",
    ),
    StrengthSessionType.LEGS: (
        "quads",
        "hamstrings",
        "glutes",
        "calves",
    ),
}


def _settings(config: dict[str, Any]) -> dict[str, Any]:
    value = config.get("coaching", {}).get("strength_training", {})
    return value if isinstance(value, dict) else {}


def _run_types(day: WeeklyScheduleDay) -> set[WorkoutType]:
    types = {activity.workout_type for activity in day.completed_activities}
    if day.recommendation is not None:
        types.add(day.recommendation.workout_type)
    return types


def _eligible_off_day(day: WeeklyScheduleDay) -> bool:
    # Completed running and explicit recovery/health guardrail prescriptions
    # are not quietly relabeled as lifting opportunities.
    return not day.completed_activities and day.recommendation is None


def _leg_candidate_cost(
    index: int,
    days: list[WeeklyScheduleDay],
    *,
    days_since_quality_run: float | None,
    days_since_long_run: float | None,
) -> float:
    taxing_distances = [
        abs(index - other)
        for other, day in enumerate(days)
        if _run_types(day) & TAXING_RUN_TYPES
    ]
    if days_since_quality_run is not None:
        taxing_distances.append(index + float(days_since_quality_run))
    if days_since_long_run is not None:
        taxing_distances.append(index + float(days_since_long_run))

    cost = 0.0
    for distance in taxing_distances:
        if distance <= 1.0:
            cost += 1_000.0
        elif distance <= 2.0:
            cost += 200.0
        elif distance <= 3.0:
            cost += 40.0

    easy_distances = [
        abs(index - other)
        for other, day in enumerate(days)
        if _run_types(day) & EASY_RUN_TYPES
    ]
    cost -= 25.0 * sum(distance == 1 for distance in easy_distances)
    cost -= 6.0 * sum(distance == 2 for distance in easy_distances)
    return cost


def _covers_horizon(combo: tuple[int, ...], length: int, interval: int) -> bool:
    if not combo:
        return False
    return (
        combo[0] < interval
        and length - 1 - combo[-1] < interval
        and all(right - left <= interval for left, right in zip(combo, combo[1:]))
    )


def _cadence_satisfied(
    exposures: set[int],
    length: int,
    interval: int,
) -> bool:
    minimum_count = max(1, (length + interval - 1) // interval)
    ordered = tuple(sorted(exposures))
    return len(ordered) >= minimum_count and _covers_horizon(
        ordered, length, interval
    )


def _planned_easy_day(day: WeeklyScheduleDay) -> bool:
    return (
        not day.completed_activities
        and day.recommendation is not None
        and day.recommendation.workout_type == WorkoutType.EASY
    )


def _minimum_upper_additions(
    exposure_sets: list[set[int]],
    candidates: list[int],
    days: list[WeeklyScheduleDay],
    interval: int,
) -> tuple[int, ...]:
    """Find the fewest easy-run pairings that close every supplied cadence gap."""

    if all(
        _cadence_satisfied(exposures, len(days), interval)
        for exposures in exposure_sets
    ):
        return ()
    taxing = [
        index
        for index, day in enumerate(days)
        if _run_types(day) & TAXING_RUN_TYPES
    ]
    for count in range(1, len(candidates) + 1):
        feasible = [
            combo
            for combo in combinations(candidates, count)
            if all(
                _cadence_satisfied(exposures | set(combo), len(days), interval)
                for exposures in exposure_sets
            )
        ]
        if feasible:
            def score(combo: tuple[int, ...]) -> tuple[float, tuple[int, ...]]:
                proximity_cost = sum(
                    10.0 if abs(candidate - hard_day) <= 1 else 2.0
                    if abs(candidate - hard_day) <= 2 else 0.0
                    for candidate in combo
                    for hard_day in taxing
                )
                return proximity_cost, combo

            return min(feasible, key=score)
    return ()


def _choose_leg_days(
    days: list[WeeklyScheduleDay],
    interval: int,
    *,
    days_since_quality_run: float | None,
    days_since_long_run: float | None,
) -> set[int]:
    eligible = [
        index
        for index, day in enumerate(days)
        if _eligible_off_day(day)
        # Missing the aspirational eight-day lifting cadence is preferable to
        # prescribing fatigued legs for the following day's primary long or
        # quality run. Strength suggestions never move the run plan.
        and not (
            index + 1 < len(days)
            and _run_types(days[index + 1]) & TAXING_RUN_TYPES
        )
    ]
    if not eligible:
        return set()
    costs = {
        index: _leg_candidate_cost(
            index,
            days,
            days_since_quality_run=days_since_quality_run,
            days_since_long_run=days_since_long_run,
        )
        for index in eligible
    }
    minimum_count = max(1, (len(days) + interval - 1) // interval)
    for count in range(minimum_count, len(eligible) + 1):
        feasible = [
            combo
            for combo in combinations(eligible, count)
            if _covers_horizon(combo, len(days), interval)
        ]
        if feasible:
            # Prefer safer run context, then regular spacing, then the stable
            # chronological tie-break supplied by the tuple itself.
            def score(combo: tuple[int, ...]) -> tuple[float, float, tuple[int, ...]]:
                spacing = sum(
                    abs(interval - (right - left))
                    for left, right in zip(combo, combo[1:])
                )
                return (sum(costs[index] for index in combo), spacing, combo)

            return set(min(feasible, key=score))

    # An extremely dense run plan can make an eight-day off-day cadence
    # impossible. Keep the safest available leg opportunity instead of moving
    # a run or misrepresenting a guardrail rest day as training.
    return {min(eligible, key=lambda index: (costs[index], index))}


def _muscle_groups(
    session_type: StrengthSessionType,
    settings: dict[str, Any],
) -> list[str]:
    configured_groups = settings.get("muscle_groups", {})
    configured = (
        configured_groups.get(session_type.value)
        if isinstance(configured_groups, dict)
        else None
    )
    if isinstance(configured, list) and all(
        isinstance(item, str) for item in configured
    ):
        return configured
    return list(DEFAULT_MUSCLE_GROUPS[session_type])


def _suggestion(
    session_type: StrengthSessionType,
    settings: dict[str, Any],
    leg_interval_days: int,
    *,
    paired_with_easy_run: bool = False,
) -> StrengthSuggestion:
    title = {
        StrengthSessionType.FULL_UPPER: "Upper body",
        StrengthSessionType.PUSH: "Upper-body push",
        StrengthSessionType.PULL: "Upper-body pull",
        StrengthSessionType.LEGS: "Leg strength",
    }[session_type]
    rationale = {
        StrengthSessionType.FULL_UPPER: (
            "An isolated no-run day can carry your usual upper-body muscle groups together."
        ),
        StrengthSessionType.PUSH: (
            "Adjacent no-run days allow upper-body work to be divided into push and pull groups."
        ),
        StrengthSessionType.PULL: (
            "Adjacent no-run days allow upper-body work to be divided into push and pull groups."
        ),
        StrengthSessionType.LEGS: (
            f"This is the best available no-run day for the {leg_interval_days}-day "
            "leg cadence while keeping lower-body work away from long and quality "
            "runs when the calendar allows."
        ),
    }[session_type]
    if paired_with_easy_run:
        rationale = (
            "The off-day pattern would otherwise leave too long between upper-body "
            "sessions, so this can share an easy-run day without changing the run."
        )
    return StrengthSuggestion(
        session_type=session_type,
        title=title,
        muscle_groups=_muscle_groups(session_type, settings),
        rationale=rationale,
    )


def add_strength_suggestions(
    days: list[WeeklyScheduleDay],
    config: dict[str, Any],
    *,
    days_since_quality_run: float | None = None,
    days_since_long_run: float | None = None,
) -> list[WeeklyScheduleDay]:
    """Return the same run plan with optional muscle-group suggestions added."""

    settings = _settings(config)
    if not bool(settings.get("enabled", False)) or not days:
        return days
    interval = max(1, int(settings.get("leg_frequency_days", 8)))
    upper_interval = max(
        1,
        int(settings.get("push_pull_frequency_days", interval)),
    )
    leg_days = _choose_leg_days(
        days,
        interval,
        days_since_quality_run=days_since_quality_run,
        days_since_long_run=days_since_long_run,
    )

    session_types: dict[int, StrengthSessionType] = {
        index: StrengthSessionType.LEGS for index in leg_days
    }
    remaining = [
        index
        for index, day in enumerate(days)
        if index not in leg_days and _eligible_off_day(day)
    ]
    position = 0
    while position < len(remaining):
        block = [remaining[position]]
        position += 1
        while position < len(remaining) and remaining[position] == block[-1] + 1:
            block.append(remaining[position])
            position += 1
        if len(block) == 1:
            session_types[block[0]] = StrengthSessionType.FULL_UPPER
        else:
            for block_position, index in enumerate(block):
                session_types[index] = (
                    StrengthSessionType.PUSH
                    if block_position % 2 == 0
                    else StrengthSessionType.PULL
                )

    def exposures_for(session_type: StrengthSessionType) -> set[int]:
        return {
            index
            for index, value in session_types.items()
            if value in {session_type, StrengthSessionType.FULL_UPPER}
        }

    paired_easy_days: set[int] = set()
    easy_candidates = [
        index
        for index, day in enumerate(days)
        if index not in session_types and _planned_easy_day(day)
    ]
    push_exposures = exposures_for(StrengthSessionType.PUSH)
    pull_exposures = exposures_for(StrengthSessionType.PULL)
    if not _cadence_satisfied(
        push_exposures, len(days), upper_interval
    ) and not _cadence_satisfied(pull_exposures, len(days), upper_interval):
        shared = _minimum_upper_additions(
            [push_exposures, pull_exposures],
            easy_candidates,
            days,
            upper_interval,
        )
        for index in shared:
            session_types[index] = StrengthSessionType.FULL_UPPER
            paired_easy_days.add(index)

    for session_type in (StrengthSessionType.PUSH, StrengthSessionType.PULL):
        exposures = exposures_for(session_type)
        if _cadence_satisfied(exposures, len(days), upper_interval):
            continue
        remaining_candidates = [
            index
            for index in easy_candidates
            if index not in session_types
        ]
        additions = _minimum_upper_additions(
            [exposures],
            remaining_candidates,
            days,
            upper_interval,
        )
        for index in additions:
            session_types[index] = session_type
            paired_easy_days.add(index)

    return [
        day.model_copy(
            update={
                "strength_suggestion": (
                    _suggestion(
                        session_types[index],
                        settings,
                        interval,
                        paired_with_easy_run=index in paired_easy_days,
                    )
                    if index in session_types
                    else None
                )
            }
        )
        for index, day in enumerate(days)
    ]
