from __future__ import annotations

from datetime import timedelta

from run_analysis.adherence_projection import ProjectionRun, simulate_adherence
from run_analysis.web.schemas import WorkoutType
from test_recommendation import CONFIG, _difficulty, _state


def test_adherence_projection_rolls_real_planner_forward_without_mutating_seed() -> None:
    template = _state(running_days_28d=12)
    seed = [
        ProjectionRun(
            start_time=template.as_of - timedelta(days=offset),
            distance_miles=4.0,
            moving_minutes=44.0,
            workout_type=WorkoutType.EASY,
            difficulty=_difficulty(miles=4.0),
        )
        for offset in range(2, 58, 2)
    ]
    original = list(seed)

    projection = simulate_adherence(
        template,
        seed,
        CONFIG,
        template.as_of,
        weeks=3,
    )

    assert len(projection) == 3
    assert seed == original
    assert all(week.run_count > 0 for week in projection)
    assert all(
        week.prescribed_low_miles
        <= week.assumed_completed_miles
        <= week.prescribed_high_miles
        for week in projection
    )
    assert all(week.workouts for week in projection)


def test_projection_does_not_invent_future_hr_load() -> None:
    template = _state(running_days_28d=10)
    seed = [
        ProjectionRun(
            start_time=template.as_of - timedelta(days=day),
            distance_miles=3.5,
            moving_minutes=38.5,
            workout_type=WorkoutType.EASY,
            difficulty=_difficulty(miles=3.5),
        )
        for day in (2, 4, 7, 10, 13, 16, 20, 24)
    ]

    projection = simulate_adherence(
        template,
        seed,
        CONFIG,
        template.as_of,
        weeks=2,
    )

    assert len(projection) == 2
    assert all(week.opening_acute_ratio is not None for week in projection)


def test_adherence_maintains_then_builds_long_run_instead_of_decaying_it() -> None:
    template = _state(running_days_28d=12)
    seed = [
        ProjectionRun(
            start_time=template.as_of - timedelta(days=1),
            distance_miles=6.4,
            moving_minutes=6.4 * 10.5,
            workout_type=WorkoutType.LONG,
            difficulty=_difficulty(miles=6.4, long=True),
        )
    ]
    seed.extend(
        ProjectionRun(
            start_time=template.as_of - timedelta(days=day),
            distance_miles=4.5,
            moving_minutes=4.5 * 10.5,
            workout_type=WorkoutType.EASY,
            difficulty=_difficulty(miles=4.5),
        )
        for day in range(4, 43, 2)
    )
    seed.extend(
        ProjectionRun(
            start_time=template.as_of - timedelta(days=day),
            distance_miles=miles,
            moving_minutes=miles * 10.5,
            workout_type=WorkoutType.LONG,
            difficulty=_difficulty(miles=miles, long=True),
        )
        for day, miles in ((50, 7.5), (69, 8.2), (96, 8.6))
    )

    projection = simulate_adherence(
        template,
        seed,
        CONFIG,
        template.as_of,
        weeks=13,
    )
    long_distances = [
        float(workout.rsplit(" ", 1)[-1])
        for week in projection
        for workout in week.workouts
        if " long " in workout
    ]

    assert long_distances
    assert long_distances[0] >= 6.4
    assert max(long_distances) > long_distances[0]
    assert max(long_distances) >= 8.0


def test_general_fitness_adherence_advances_weekly_capacity() -> None:
    template = _state(running_days_28d=12)
    seed = [
        ProjectionRun(
            start_time=template.as_of - timedelta(days=day),
            distance_miles=4.0,
            moving_minutes=44.0,
            workout_type=WorkoutType.EASY,
            difficulty=_difficulty(miles=4.0),
        )
        for day in range(1, 57, 2)
    ]

    projection = simulate_adherence(
        template,
        seed,
        CONFIG,
        template.as_of,
        weeks=13,
    )

    assert projection[-1].capacity_reference_miles > projection[0].capacity_reference_miles
    assert (
        sum((projection[-1].target_low_miles, projection[-1].target_high_miles)) / 2
        > sum((projection[0].target_low_miles, projection[0].target_high_miles)) / 2
    )

    # Perfect adherence is the clean-room check for a training-stimulus loop.
    # Real missed or shifted runs add enough noise to hide a planner that has
    # converged on the same calendar, roles, and workout content indefinitely.
    def signature(week) -> tuple[tuple[str, str, str], ...]:
        result = []
        for workout in week.workouts:
            day_name, workout_type = workout.split(" ", 2)[:2]
            role = (
                "quality"
                if workout_type in {"intervals", "tempo_threshold", "race"}
                else workout_type
            )
            title = workout.split("[", 1)[1].split("]", 1)[0]
            result.append((day_name, role, title))
        return tuple(result)

    signatures = [signature(week) for week in projection]
    longest_repeat = current_repeat = 1
    for previous, current in zip(signatures, signatures[1:]):
        current_repeat = current_repeat + 1 if current == previous else 1
        longest_repeat = max(longest_repeat, current_repeat)
    assert longest_repeat <= 2

    weekday = {
        "Mon": 0,
        "Tue": 1,
        "Wed": 2,
        "Thu": 3,
        "Fri": 4,
        "Sat": 5,
        "Sun": 6,
    }
    assert any(
        any(
            weekday[current[0]] - weekday[previous[0]] == 1
            for previous, current in zip(signature(week), signature(week)[1:])
        )
        for week in projection[4:]
    )

    quality_titles = {
        workout.split("[", 1)[1].split("]", 1)[0]
        for week in projection
        for workout in week.workouts
        if " intervals " in workout or " tempo_threshold " in workout
    }
    assert len(quality_titles) >= 3
