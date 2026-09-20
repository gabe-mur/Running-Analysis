from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from run_analysis.strength_schedule import add_strength_suggestions
from run_analysis.web.schemas import (
    ConfidenceLevel,
    ReadinessFlag,
    RecommendationResponse,
    StrengthSessionType,
    WeeklyScheduleDay,
    WorkoutType,
)


CONFIG = {
    "coaching": {
        "strength_training": {
            "enabled": True,
            "leg_frequency_days": 8,
        }
    }
}
START = date(2026, 9, 17)


def _day(offset: int, workout_type: WorkoutType | None = None) -> WeeklyScheduleDay:
    moment = datetime(2026, 9, 17, 12, tzinfo=timezone.utc) + timedelta(days=offset)
    recommendation = None
    if workout_type is not None:
        recommendation = RecommendationResponse(
            generated_at=moment,
            fitness_state_as_of=moment,
            planned_for=moment,
            workout_type=workout_type,
            title=workout_type.value,
            distance_range_miles=(3.0, 4.0) if workout_type != WorkoutType.REST else None,
            confidence=ConfidenceLevel.MODERATE,
            readiness=ReadinessFlag.READY,
        )
    return WeeklyScheduleDay(
        date=START + timedelta(days=offset),
        planned_at=moment if recommendation is not None else None,
        recommendation=recommendation,
        day_role="run" if recommendation is not None else "rest_day",
        rationale="Fixture.",
    )


def test_strength_layer_preserves_run_plan_and_covers_each_eight_day_span() -> None:
    run_types = {
        0: WorkoutType.EASY,
        2: WorkoutType.LONG,
        4: WorkoutType.EASY,
        6: WorkoutType.INTERVALS,
        8: WorkoutType.EASY,
        10: WorkoutType.EASY,
        12: WorkoutType.LONG,
        14: WorkoutType.EASY,
        16: WorkoutType.TEMPO_THRESHOLD,
        18: WorkoutType.EASY,
        20: WorkoutType.EASY,
    }
    original = [_day(index, run_types.get(index)) for index in range(21)]
    annotated = add_strength_suggestions(original, CONFIG)

    assert [day.recommendation for day in annotated] == [
        day.recommendation for day in original
    ]
    leg_days = [
        index
        for index, day in enumerate(annotated)
        if day.strength_suggestion
        and day.strength_suggestion.session_type == StrengthSessionType.LEGS
    ]
    assert leg_days[0] < 8
    assert 20 - leg_days[-1] < 8
    assert all(right - left <= 8 for left, right in zip(leg_days, leg_days[1:]))
    assert all(original[index].recommendation is None for index in leg_days)


def test_leg_day_prefers_easy_neighbors_over_long_or_quality_neighbors() -> None:
    days = [
        _day(0, WorkoutType.LONG),
        _day(1),
        _day(2, WorkoutType.EASY),
        _day(3),
        _day(4, WorkoutType.EASY),
        _day(5),
        _day(6, WorkoutType.INTERVALS),
        _day(7),
    ]
    annotated = add_strength_suggestions(days, CONFIG)
    leg_days = [
        index
        for index, day in enumerate(annotated)
        if day.strength_suggestion
        and day.strength_suggestion.session_type == StrengthSessionType.LEGS
    ]

    assert leg_days == [3]


def test_leg_day_never_uses_the_day_before_a_long_or_quality_run() -> None:
    days = [
        _day(0, WorkoutType.EASY),
        _day(1),
        _day(2, WorkoutType.LONG),
        _day(3, WorkoutType.EASY),
        _day(4),
        _day(5, WorkoutType.INTERVALS),
        _day(6, WorkoutType.EASY),
        _day(7),
    ]

    annotated = add_strength_suggestions(days, CONFIG)
    leg_days = [
        index
        for index, day in enumerate(annotated)
        if day.strength_suggestion
        and day.strength_suggestion.session_type == StrengthSessionType.LEGS
    ]

    assert leg_days == [7]


def test_adjacent_upper_days_split_push_pull_but_leg_pair_uses_full_upper() -> None:
    run_types = {
        0: WorkoutType.LONG,
        2: WorkoutType.EASY,
        4: WorkoutType.EASY,
        7: WorkoutType.INTERVALS,
    }
    days = [_day(index, run_types.get(index)) for index in range(8)]
    annotated = add_strength_suggestions(days, CONFIG)
    types = {
        index: day.strength_suggestion.session_type
        for index, day in enumerate(annotated)
        if day.strength_suggestion
    }

    assert types[1] == StrengthSessionType.FULL_UPPER
    assert types[3] == StrengthSessionType.LEGS
    assert types[5] == StrengthSessionType.PUSH
    assert types[6] == StrengthSessionType.PULL
    assert annotated[5].strength_suggestion.muscle_groups == [
        "upper pecs",
        "lower pecs",
        "side delts",
        "triceps",
        "core/trunk",
    ]
    assert annotated[6].strength_suggestion.muscle_groups == [
        "lats",
        "upper back",
        "rear delts",
        "biceps",
    ]


def test_guardrail_rest_stays_empty_and_easy_run_can_close_upper_cadence() -> None:
    days = [
        _day(0, WorkoutType.REST),
        _day(1, WorkoutType.EASY),
        _day(2),
    ]
    annotated = add_strength_suggestions(days, CONFIG)

    assert annotated[0].strength_suggestion is None
    assert (
        annotated[1].strength_suggestion.session_type
        == StrengthSessionType.FULL_UPPER
    )
    assert "share an easy-run day" in annotated[1].strength_suggestion.rationale
    assert annotated[2].strength_suggestion is not None
    assert annotated[2].strength_suggestion.muscle_groups == [
        "quads",
        "hamstrings",
        "glutes",
        "calves",
    ]
    assert all(
        "lb" not in group and "dumbbell" not in group
        for group in annotated[2].strength_suggestion.muscle_groups
    )


def test_dense_run_plan_uses_easy_days_to_keep_push_and_pull_inside_eight_days() -> None:
    days = [
        _day(index, None if index in {7, 15} else WorkoutType.EASY)
        for index in range(21)
    ]
    annotated = add_strength_suggestions(days, CONFIG)

    def exposures(session_type: StrengthSessionType) -> list[int]:
        return [
            index
            for index, day in enumerate(annotated)
            if day.strength_suggestion
            and day.strength_suggestion.session_type
            in {session_type, StrengthSessionType.FULL_UPPER}
        ]

    for session_type in (StrengthSessionType.PUSH, StrengthSessionType.PULL):
        values = exposures(session_type)
        assert len(values) >= 3
        assert values[0] < 8
        assert 20 - values[-1] < 8
        assert all(right - left <= 8 for left, right in zip(values, values[1:]))
    assert any(
        day.recommendation is not None and day.strength_suggestion is not None
        for day in annotated
    )
    assert all(
        day.recommendation is None
        or day.recommendation.workout_type == WorkoutType.EASY
        or day.strength_suggestion is None
        for day in annotated
    )
