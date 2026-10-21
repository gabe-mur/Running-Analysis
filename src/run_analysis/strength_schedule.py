"""Attach non-binding strength suggestions to an already-finalized run plan.

This module deliberately runs after the running optimizer.  Strength guidance
may react to run placement, but it must never move a run or change its dose.
"""

from __future__ import annotations

from datetime import date, timedelta
from itertools import combinations
import json
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
    if day.day_role == 'race_event_day':
        types.add(WorkoutType.RACE)
    if day.recommendation is not None:
        types.add(day.recommendation.workout_type)
    return types


def _eligible_off_day(day: WeeklyScheduleDay) -> bool:
    # Completed running and explicit recovery/health guardrail prescriptions
    # are not quietly relabeled as lifting opportunities.
    return not day.completed_activities and day.recommendation is None and day.day_role != 'race_event_day'


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


def _planned_easy_day(day: WeeklyScheduleDay) -> bool:
    return (
        not day.completed_activities
        and day.recommendation is not None
        and day.recommendation.workout_type == WorkoutType.EASY
    )


def _choose_leg_days(
    days: list[WeeklyScheduleDay],
    interval: int,
    *,
    days_since_quality_run: float | None,
    days_since_long_run: float | None,
    prior_plan: dict[date, StrengthSessionType] | None = None,
    completed_run_types: dict[date, set[WorkoutType]] | None = None,
) -> set[int]:
    prior_plan = prior_plan or {}
    completed_run_types = completed_run_types or {}
    eligible = [
        index
        for index, day in enumerate(days)
        if _eligible_off_day(day)
        and not completed_run_types.get(day.date)
        # Missing the aspirational eight-day lifting cadence is preferable to
        # prescribing fatigued legs for the following day's primary long or
        # quality run. Strength suggestions never move the run plan.
        # The last horizon day has no known following-day clearance.
        and index + 1 < len(days)
        and not ((_run_types(days[index + 1]) | completed_run_types.get(days[index + 1].date, set())) & TAXING_RUN_TYPES)
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
    previous = [d for d, kind in prior_plan.items() if d < days[0].date and kind == StrengthSessionType.LEGS]
    history = ((max(previous) - days[0].date).days,) if previous and (days[0].date - max(previous)).days <= interval else ()
    retained = {index for index in eligible if prior_plan.get(days[index].date) == StrengthSessionType.LEGS}
    for count in range(0 if history else 1, len(eligible) + 1):
        feasible = [
            combo
            for combo in combinations(eligible, count)
            if _covers_horizon(history + combo, len(days), interval)
            and retained.issubset(combo)
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
    if retained:
        return retained
    return {min(eligible, key=lambda index: (costs[index], index))}


def _upper_rotation(days, leg_days, interval, prior_plan, completed_run_types):
    """Small exact assignment of strength slots, not another running search.

    Cadence first; prefer no-run days, then separated push/pull and compatible
    prior suggestions. Full upper is an off-day coverage fallback; easy-day
    pairing is reserved for coverage that the no-run days cannot provide.
    History here is ISSUED guidance, never evidence of completed lifting.
    """
    lanes = (StrengthSessionType.PUSH, StrengthSessionType.PULL)
    ages = []
    mask = 0
    for bit, lane in enumerate(lanes):
        previous = [d for d, kind in prior_plan.items() if d < days[0].date
                    and kind in {lane, StrengthSessionType.FULL_UPPER}]
        # With no rotation history, cover both lanes in the first visible week.
        # This is a startup deadline, not an invented previous workout.
        ages.append(min(interval, (days[0].date - max(previous)).days - 1)
                    if previous else max(0, interval - min(interval, 7)))
        if previous:
            mask |= 1 << bit
    # State = push age, pull age, observed/scheduled lane mask.
    states = {(*ages, mask): ((0, 0, 0, 0, 0, 0), (), ())}
    codes = {StrengthSessionType.PUSH: 0, StrengthSessionType.PULL: 1,
             StrengthSessionType.FULL_UPPER: 2, None: 3}
    taxing = [i for i, day in enumerate(days) if _run_types(day) & TAXING_RUN_TYPES]
    prefer_off = any(index not in leg_days and _eligible_off_day(day)
                     and not completed_run_types.get(day.date)
                     for index, day in enumerate(days))
    for index, day in enumerate(days):
        previous = prior_plan.get(day.date)
        if previous not in {*lanes, StrengthSessionType.FULL_UPPER}:
            previous = None
        completed = completed_run_types.get(day.date, set()) | {a.workout_type for a in day.completed_activities}
        off = _eligible_off_day(day) and not completed
        easy = _planned_easy_day(day) and not completed
        # Completing the prescribed easy run does not cancel its already
        # assigned upper-body session. It is not a new leg/off-day opportunity.
        retained_easy = bool(completed) and completed.issubset(EASY_RUN_TYPES) and previous is not None
        if index in leg_days:
            options = (None,)
        elif retained_easy:
            options = (None, previous)
            easy = True
        else:
            options = (None, *lanes, StrengthSessionType.FULL_UPPER) if off or easy else (None,)
        advanced = {}
        for (push_age, pull_age, seen), (score, tie, path) in states.items():
            for kind in options:
                covered = tuple(kind in {lane, StrengthSessionType.FULL_UPPER} for lane in lanes)
                next_ages = tuple(0 if yes else min(interval, age + 1)
                                  for yes, age in zip(covered, (push_age, pull_age)))
                next_seen = seen | sum(1 << bit for bit, yes in enumerate(covered) if yes)
                # When off-day lifting is possible, use it before retaining an
                # old run/lift pair. Count covered lanes so a combined easy-day
                # session cannot hide the equivalent of two paired exposures.
                # With no off days, keep the established split/retention policy:
                # minimizing unavoidable pair counts across a rolling horizon
                # would needlessly move already-issued sessions.
                middle = ((sum(covered) if easy else 0,
                           int(kind == StrengthSessionType.FULL_UPPER),
                           int(previous is not None and kind != previous)) if prefer_off else
                          (int(kind == StrengthSessionType.FULL_UPPER),
                           int(previous is not None and kind != previous),
                           int(kind is not None and easy)))
                penalties = (sum(age >= interval for age in next_ages), *middle,
                    int(kind is not None),
                    sum(10 if abs(index - hard) <= 1 else 2 if abs(index - hard) <= 2 else 0
                        for hard in taxing) if kind is not None else 0)
                order = codes[kind]
                if prefer_off and kind in lanes and push_age != pull_age:
                    # Otherwise equal choices should serve the longer-waiting
                    # lane, not restart with Push at every new horizon.
                    order = int((kind == lanes[0]) != (push_age > pull_age))
                candidate = (tuple(a + b for a, b in zip(score, penalties)), tie + (order,), path + (kind,))
                key = (*next_ages, next_seen)
                if key not in advanced or candidate[:2] < advanced[key][:2]:
                    advanced[key] = candidate
        states = advanced
    def final_score(item):
        (_, _, seen), (score, tie, _) = item
        missing = 2 - seen.bit_count()
        return (score[0] + missing, *score[1:]), tie
    _, (_, _, path) = min(states.items(), key=final_score)
    return {index: kind for index, kind in enumerate(path) if kind is not None}


STRENGTH_ROTATION_VERSION = 2


def _rotation_preferences(config):
    settings = _settings(config)
    return dict(enabled=bool(settings.get('enabled', False)),
                legs=max(1, int(settings.get('leg_frequency_days', 8))),
                upper=max(1, int(settings.get('push_pull_frequency_days', settings.get('leg_frequency_days', 8)))))


def load_strength_rotation(connection, config=None, *, start_date=None) -> dict[date, StrengthSessionType]:
    """Separate suggestion memory; never a physical completion or run input."""
    row = connection.execute("SELECT value_json FROM app_state WHERE key='strength_rotation'").fetchone()
    if not row:
        return {}
    payload = json.loads(row[0])
    if payload.get('version') != STRENGTH_ROTATION_VERSION:
        return {}
    plan = {date.fromisoformat(d): StrengthSessionType(kind) for d, kind in payload['suggestions'].items()}
    if config is not None and payload.get('preferences') != _rotation_preferences(config):
        # A deliberate cadence/enable edit is permission to rebuild future
        # lifting, not to forget previously issued historical rotation slots.
        return {d: kind for d, kind in plan.items() if start_date is not None and d < start_date}
    return plan


def strength_rotation_payload(days, prior_plan, config) -> str:
    """Keep recent issued guidance across rollover without claiming adherence."""
    settings = _settings(config)
    interval = max(int(settings.get('leg_frequency_days', 8)), int(settings.get('push_pull_frequency_days', 8)))
    cutoff = days[0].date - timedelta(days=max(7, interval)) if days else date.max
    history = {d: kind for d, kind in prior_plan.items() if cutoff <= d < days[0].date} if days else {}
    history.update({day.date: day.strength_suggestion.session_type for day in days if day.strength_suggestion})
    return json.dumps(dict(version=STRENGTH_ROTATION_VERSION, provenance='issued_suggestions_not_completed_lifts',
                           preferences=_rotation_preferences(config),
                           suggestions={d.isoformat(): kind.value for d, kind in sorted(history.items())}))


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
            "Combined push and pull because separate sessions will not fit the rotation."
        ),
        StrengthSessionType.PUSH: (
            ""
        ),
        StrengthSessionType.PULL: (
            ""
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
    prior_plan: dict[date, StrengthSessionType] | None = None,
    completed_run_types: dict[date, set[WorkoutType]] | None = None,
) -> list[WeeklyScheduleDay]:
    """Return the same run plan with optional muscle-group suggestions added."""

    settings = _settings(config)
    if not bool(settings.get("enabled", False)) or not days:
        return [day.model_copy(update={'strength_suggestion': None}) for day in days]
    prior_plan = prior_plan or {}
    completed_run_types = completed_run_types or {}
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
        prior_plan=prior_plan,
        completed_run_types=completed_run_types,
    )

    session_types: dict[int, StrengthSessionType] = {
        index: StrengthSessionType.LEGS for index in leg_days
    }
    session_types.update(_upper_rotation(days, leg_days, upper_interval, prior_plan, completed_run_types))
    paired_easy_days = {index for index in session_types if _planned_easy_day(days[index])
                        or bool(completed_run_types.get(days[index].date, set()) & EASY_RUN_TYPES)
                        or any(a.workout_type in EASY_RUN_TYPES for a in days[index].completed_activities)}

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
