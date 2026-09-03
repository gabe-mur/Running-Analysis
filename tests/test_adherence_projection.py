from __future__ import annotations

from datetime import datetime, timedelta, timezone
from statistics import median_high

from run_analysis.adherence_projection import (
    HumanAdherenceProfile,
    ProjectionRun,
    _completed_activities_on_plan_date,
    _human_breaks,
    _observed_human_difficulty,
    _projected_difficulty,
    _state_at,
    simulate_adherence,
)
from run_analysis.web.schemas import (
    ConfidenceLevel,
    QualitySessionType,
    ReadinessFlag,
    RecommendationResponse,
    WorkoutStep,
    WorkoutType,
    WeeklyScheduleDay,
    WeeklyScheduleResponse,
)
from test_recommendation import CONFIG, _difficulty, _state


def test_human_break_calendar_has_bounded_trips_and_at_most_one_vacation() -> None:
    start = _state().as_of.date()
    pauses = _human_breaks(HumanAdherenceProfile(), start, 91)

    trips = [item for item in pauses if item.label.startswith("trip")]
    vacations = [item for item in pauses if item.label == "vacation"]
    assert 2 <= len(trips) <= 3
    assert all(2 <= (item.end_date - item.start_date).days <= 3 for item in trips)
    assert len(vacations) <= 1
    assert all(
        left.end_date + timedelta(days=2) <= right.start_date
        for left, right in zip(pauses, pauses[1:])
    )


def test_human_intensity_drift_becomes_observed_load_evidence() -> None:
    planned_for = _state().as_of
    recommendation = RecommendationResponse(
        generated_at=planned_for,
        fitness_state_as_of=planned_for,
        planned_for=planned_for,
        workout_type=WorkoutType.EASY,
        title="Easy aerobic run",
        distance_range_miles=(4.0, 4.0),
        confidence=ConfidenceLevel.MODERATE,
        readiness=ReadinessFlag.READY,
    )

    normal = _observed_human_difficulty(
        recommendation, 4.0, 10.0, WorkoutType.EASY, too_intense=False
    )
    drifted = _observed_human_difficulty(
        recommendation, 4.0, 10.0, WorkoutType.EASY, too_intense=True
    )

    assert normal.zone_load is not None
    assert drifted.zone_load > normal.zone_load
    assert drifted.zone_breakdown.hard_minutes > 0


def test_daily_reload_exposes_an_already_completed_run_on_the_same_date() -> None:
    template = _state()
    morning = template.as_of.replace(hour=7)
    evening_reload = template.as_of.replace(hour=19)
    prior_day = ProjectionRun(
        start_time=morning - timedelta(days=1),
        distance_miles=4.0,
        moving_minutes=44.0,
        workout_type=WorkoutType.EASY,
        difficulty=_difficulty(miles=4.0),
    )
    same_day = ProjectionRun(
        start_time=morning,
        distance_miles=3.5,
        moving_minutes=38.5,
        workout_type=WorkoutType.EASY,
        difficulty=_difficulty(miles=3.5),
        projected=True,
    )

    completed = _completed_activities_on_plan_date(
        [prior_day, same_day], evening_reload
    )

    assert len(completed) == 1
    assert completed[0].start_time == morning
    assert completed[0].distance_miles == 3.5


def test_projection_groups_runs_by_workout_date_and_excludes_end_boundary(
    monkeypatch,
) -> None:
    template = _state(
        as_of=datetime(2026, 9, 2, 12, tzinfo=timezone.utc),
        running_days_28d=12,
    )
    seed = [
        ProjectionRun(
            start_time=template.as_of - timedelta(days=offset),
            distance_miles=4.0,
            moving_minutes=44.0,
            workout_type=WorkoutType.EASY,
            difficulty=_difficulty(miles=4.0),
        )
        for offset in range(2, 30, 2)
    ]

    def next_morning_schedule(daily_states, *args, **kwargs):
        generated_at = daily_states[0].as_of
        planned_for = (generated_at + timedelta(days=1)).replace(
            hour=7,
            minute=0,
            second=0,
            microsecond=0,
        )
        recommendation = RecommendationResponse(
            generated_at=generated_at,
            fitness_state_as_of=generated_at,
            planned_for=planned_for,
            workout_type=WorkoutType.EASY,
            title="Boundary fixture",
            distance_range_miles=(4.0, 4.0),
            confidence=ConfidenceLevel.MODERATE,
            readiness=ReadinessFlag.READY,
        )
        day = WeeklyScheduleDay(
            date=planned_for.date(),
            planned_at=planned_for,
            recommendation=recommendation,
            day_role="easy_run",
            rationale="Boundary accounting fixture.",
        )
        target_range = kwargs["target_distance_range"]
        return WeeklyScheduleResponse(
            generated_at=generated_at,
            start_date=generated_at.date(),
            end_date=generated_at.date() + timedelta(days=6),
            target_run_count=1,
            target_distance_range_miles=target_range,
            target_evidence=kwargs["target_evidence"],
            run_count=1,
            projected_distance_range_miles=(4.0, 4.0),
            summary="Boundary accounting fixture.",
            days=[day],
            planning_days=[day],
        )

    monkeypatch.setattr(
        "run_analysis.adherence_projection.build_weekly_schedule",
        next_morning_schedule,
    )

    projection = simulate_adherence(
        template,
        seed,
        CONFIG,
        template.as_of,
        weeks=2,
        replan_interval_days=1,
    )

    assert [week.run_count for week in projection] == [6, 7]
    assert [week.assumed_completed_miles for week in projection] == [24.0, 28.0]


def test_human_projection_feeds_skipped_commitments_back_into_replans(
    monkeypatch,
) -> None:
    template = _state(
        as_of=datetime(2026, 9, 2, 12, tzinfo=timezone.utc),
        running_days_28d=12,
    )
    seed = [
        ProjectionRun(
            start_time=template.as_of - timedelta(days=offset),
            distance_miles=4.0,
            moving_minutes=44.0,
            workout_type=WorkoutType.EASY,
            difficulty=_difficulty(miles=4.0),
        )
        for offset in range(2, 30, 2)
    ]

    def next_morning_schedule(daily_states, *args, **kwargs):
        generated_at = daily_states[0].as_of
        planned_for = (generated_at + timedelta(days=1)).replace(
            hour=7, minute=0, second=0, microsecond=0
        )
        recommendation = RecommendationResponse(
            generated_at=generated_at,
            fitness_state_as_of=generated_at,
            planned_for=planned_for,
            workout_type=WorkoutType.EASY,
            title="Skip fixture",
            distance_range_miles=(4.0, 4.0),
            confidence=ConfidenceLevel.MODERATE,
            readiness=ReadinessFlag.READY,
        )
        day = WeeklyScheduleDay(
            date=planned_for.date(),
            planned_at=planned_for,
            recommendation=recommendation,
            day_role="easy_run",
            rationale="Skip fixture.",
        )
        return WeeklyScheduleResponse(
            generated_at=generated_at,
            start_date=generated_at.date(),
            end_date=generated_at.date() + timedelta(days=6),
            target_run_count=1,
            target_distance_range_miles=kwargs["target_distance_range"],
            target_evidence=kwargs["target_evidence"],
            run_count=1,
            projected_distance_range_miles=(4.0, 4.0),
            summary="Skip fixture.",
            days=[day],
            planning_days=[day],
        )

    monkeypatch.setattr(
        "run_analysis.adherence_projection.build_weekly_schedule",
        next_morning_schedule,
    )
    projection = simulate_adherence(
        template,
        seed,
        CONFIG,
        template.as_of,
        weeks=1,
        human_profile=HumanAdherenceProfile(
            seed=7,
            skip_probability=1.0,
            distance_variation_probability=0.0,
            intensity_drift_probability=0.0,
            substitution_probability=0.0,
            unscheduled_easy_probability_per_day=0.0,
            short_break_count_min=0,
            short_break_count_max=0,
            vacation_probability=0.0,
        ),
    )

    assert projection[0].planned_run_count > 0
    assert projection[0].run_count == 0
    assert projection[0].skipped_run_count == projection[0].planned_run_count
    assert projection[0].prescribed_low_miles > 0


def test_human_projection_can_add_an_unscheduled_rest_day_run(monkeypatch) -> None:
    template = _state(as_of=_state().as_of.replace(hour=12))

    def empty_schedule(daily_states, *args, **kwargs):
        generated_at = daily_states[0].as_of
        return WeeklyScheduleResponse(
            generated_at=generated_at,
            start_date=generated_at.date(),
            end_date=generated_at.date() + timedelta(days=6),
            target_run_count=0,
            target_distance_range_miles=kwargs["target_distance_range"],
            target_evidence=kwargs["target_evidence"],
            run_count=0,
            projected_distance_range_miles=(0.0, 0.0),
            summary="Rest fixture.",
            days=[],
            planning_days=[],
        )

    monkeypatch.setattr(
        "run_analysis.adherence_projection.build_weekly_schedule",
        empty_schedule,
    )
    projection = simulate_adherence(
        template,
        [],
        CONFIG,
        template.as_of,
        weeks=1,
        human_profile=HumanAdherenceProfile(
            seed=11,
            skip_probability=0.0,
            distance_variation_probability=0.0,
            intensity_drift_probability=0.0,
            substitution_probability=0.0,
            unscheduled_easy_probability_per_day=1.0,
            short_break_count_min=0,
            short_break_count_max=0,
            vacation_probability=0.0,
        ),
    )

    assert projection[0].planned_run_count == 0
    assert projection[0].unscheduled_run_count == 7
    assert projection[0].run_count == 7


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
        replan_interval_days=7,
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
        replan_interval_days=7,
    )

    assert len(projection) == 2
    assert all(week.opening_acute_ratio is not None for week in projection)


def test_short_support_runs_do_not_redefine_ordinary_easy_distance() -> None:
    template = _state(typical_easy_run_miles=4.0)
    ordinary = ProjectionRun(
        start_time=template.as_of - timedelta(days=35),
        distance_miles=4.0,
        moving_minutes=44.0,
        workout_type=WorkoutType.EASY,
        difficulty=_difficulty(miles=4.0),
        planning_role="ordinary_easy",
    )
    supports = [
        ProjectionRun(
            start_time=template.as_of - timedelta(days=day),
            distance_miles=2.0,
            moving_minutes=22.0,
            workout_type=WorkoutType.EASY,
            difficulty=_difficulty(miles=2.0),
            projected=True,
            planning_role="support_easy",
        )
        for day in (2, 5, 8, 11, 14)
    ]

    projected = _state_at(
        template,
        [ordinary, *supports],
        template.as_of,
        capacity_reference=16.0,
    )

    assert projected.typical_easy_run_miles == 4.0


def test_projection_uses_prescribed_quality_dose_not_a_fixed_fraction() -> None:
    planned_for = _state().as_of
    recommendation = RecommendationResponse(
        generated_at=planned_for,
        fitness_state_as_of=planned_for,
        planned_for=planned_for,
        workout_type=WorkoutType.INTERVALS,
        quality_session_type=QualitySessionType.SHORT_INTERVALS,
        title="Short controlled pickups",
        distance_range_miles=(4.0, 4.5),
        structure=[
            WorkoutStep(
                instruction="8 x 1 minute",
                phase="work",
                repetitions=8,
                work_duration_minutes=1,
                recovery_duration_minutes=1.5,
                target_zones=["Z4 effort"],
            )
        ],
        confidence=ConfidenceLevel.MODERATE,
        readiness=ReadinessFlag.READY,
    )

    difficulty = _projected_difficulty(recommendation, 5.0, 10.0)

    assert difficulty.moving_minutes == 50
    assert difficulty.zone_breakdown.hard_minutes == 8
    assert difficulty.zone_breakdown.moderate_minutes == 0
    assert difficulty.zone_breakdown.easy_minutes == 42


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
        replan_interval_days=7,
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
        replan_interval_days=7,
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

    # This fast fixture explicitly commits seven days at a time. It may reveal
    # boundary behavior, but cannot certify production-fidelity spacing; the
    # daily receding-horizon audit owns that assertion.

    quality_titles = {
        workout.split("[", 1)[1].split("]", 1)[0]
        for week in projection
        for workout in week.workouts
        if " intervals " in workout or " tempo_threshold " in workout
    }
    assert len(quality_titles) >= 3

    # The 10-minute calibration exemption belongs to baseline acquisition,
    # not established planning. Perfect adherence must not let plan-generated
    # short support runs pull the ordinary-easy reference into a self-reinforcing
    # sequence of calibration-sized filler sessions.
    ordinary_easy_by_week = [
        [
            float(workout.rsplit(" ", 1)[-1])
            for workout in week.workouts
            if " easy [Easy aerobic run] " in workout
        ]
        for week in projection
    ]
    opening_easy = next(
        (week for week in ordinary_easy_by_week if week),
        [],
    )
    assert opening_easy
    opening_reference = median_high(opening_easy)
    assert all(
        miles >= opening_reference * 0.5
        for week in ordinary_easy_by_week
        for miles in week
    )
