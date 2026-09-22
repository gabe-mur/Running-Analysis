from __future__ import annotations

from datetime import datetime, timedelta, timezone
from statistics import median_high

import pytest

from run_analysis.adherence_projection import (
    DeterministicAdherenceProfile,
    DeterministicAdherenceScenario,
    HumanAdherenceProfile,
    OverloadAdherenceProfile,
    OverloadScenario,
    ProjectionPlanSession,
    ProjectionReplan,
    ProjectionRun,
    _completed_activities_on_plan_date,
    _completed_planning_role,
    _consecutive_date_streaks,
    _human_breaks,
    _observed_human_difficulty,
    _projected_difficulty,
    _state_at,
    simulate_adherence,
    simulate_expected_policy_rollout,
    summarize_replan_churn,
)
from run_analysis.weekly_schedule import _project_state, derive_weekly_target
from run_analysis.web.schemas import (
    ConfidenceLevel,
    QualitySessionType,
    ReadinessFlag,
    RecommendationResponse,
    WorkoutStep,
    WorkoutType,
    WeeklyScheduleDay,
    WeeklyScheduleResponse,
    ZoneBreakdown,
)
from test_recommendation import CONFIG, _difficulty, _state


def test_replan_churn_separates_date_moves_from_distance_edits() -> None:
    opening = datetime(2026, 9, 1, 18, tzinfo=timezone.utc)

    def session(day: int, workout_type: WorkoutType, miles: float):
        return ProjectionPlanSession(
            planned_for=opening + timedelta(days=day),
            workout_type=workout_type,
            midpoint_miles=miles,
        )

    replans = (
        ProjectionReplan(
            generated_at=opening,
            opening_load_ratio=1.0,
            target_low_miles=17.0,
            target_high_miles=18.0,
            planned_sessions=(
                session(1, WorkoutType.EASY, 4.0),
                session(3, WorkoutType.EASY, 4.0),
                session(5, WorkoutType.LONG, 7.0),
            ),
            committed_sessions=(),
        ),
        ProjectionReplan(
            generated_at=opening + timedelta(days=1),
            opening_load_ratio=1.0,
            target_low_miles=17.0,
            target_high_miles=18.0,
            planned_sessions=(
                session(3, WorkoutType.EASY, 4.5),
                session(6, WorkoutType.LONG, 7.0),
            ),
            committed_sessions=(),
        ),
        ProjectionReplan(
            generated_at=opening + timedelta(days=2),
            opening_load_ratio=1.0,
            target_low_miles=17.0,
            target_high_miles=18.0,
            planned_sessions=(
                session(3, WorkoutType.EASY, 5.0),
                session(6, WorkoutType.LONG, 7.0),
            ),
            committed_sessions=(),
        ),
    )

    churn = summarize_replan_churn(replans, horizon_days=7)

    assert churn.comparison_count == 2
    assert churn.schedule_changed_comparisons == 1
    assert churn.distance_only_changed_comparisons == 1
    assert churn.date_slot_changes == 2
    assert churn.workout_type_changes == 0
    assert churn.distance_changes == 2
    # The day-one session disappeared because it was consumed before the
    # second refresh; it is deliberately absent from date-slot churn.


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


def test_simulated_in_range_edges_have_materially_same_recovery_residual() -> None:
    template = _state(
        as_of=datetime(2026, 9, 2, 12, tzinfo=timezone.utc),
        typical_easy_run_miles=4.0,
    )
    recommendation = RecommendationResponse(
        generated_at=template.as_of,
        fitness_state_as_of=template.as_of,
        planned_for=template.as_of,
        workout_type=WorkoutType.EASY,
        title="Range-equivalence fixture",
        distance_range_miles=(3.0, 4.0),
        confidence=ConfidenceLevel.MODERATE,
        readiness=ReadinessFlag.READY,
    )

    def projected_run(miles: float) -> ProjectionRun:
        return ProjectionRun(
            start_time=template.as_of,
            distance_miles=miles,
            moving_minutes=miles * 10.0,
            workout_type=WorkoutType.EASY,
            difficulty=_projected_difficulty(
                recommendation,
                miles,
                10.0,
            ),
            projected=True,
            planning_role="ordinary_easy",
            prescribed_workout_type=WorkoutType.EASY,
            completed_prescribed_workout=True,
            prescribed_low_miles=3.0,
            prescribed_high_miles=4.0,
        )

    prior_runs = [
        ProjectionRun(
            start_time=template.as_of - timedelta(days=day),
            distance_miles=4.0,
            moving_minutes=40.0,
            workout_type=WorkoutType.EASY,
            difficulty=_difficulty(miles=4.0),
            planning_role="ordinary_easy",
        )
        for day in (8, 6, 4, 2)
    ]
    observed_at = template.as_of + timedelta(hours=12)
    low_state = _state_at(
        template,
        [*prior_runs, projected_run(3.0)],
        observed_at,
        16.0,
    )
    high_state = _state_at(
        template,
        [*prior_runs, projected_run(4.0)],
        observed_at,
        16.0,
    )

    assert low_state.recovery_residual_load == pytest.approx(
        high_state.recovery_residual_load,
        abs=0.001,
    )


def test_midpoint_compliance_matches_the_planners_projected_future_state() -> None:
    template = _state(
        as_of=datetime(2026, 9, 19, 12, tzinfo=timezone.utc),
        typical_easy_run_miles=4.0,
        running_days_28d=12,
    )
    history = [
        ProjectionRun(
            start_time=template.as_of - timedelta(days=day),
            distance_miles=4.0,
            moving_minutes=44.0,
            workout_type=WorkoutType.EASY,
            difficulty=_difficulty(miles=4.0).model_copy(
                update={
                    "zone_breakdown": ZoneBreakdown(
                        easy_minutes=44.0,
                        moderate_minutes=0.0,
                        hard_minutes=0.0,
                    )
                }
            ),
            planning_role="ordinary_easy",
        )
        for day in range(2, 30, 2)
    ]
    planned_at = template.as_of + timedelta(days=2)
    future_at = planned_at + timedelta(days=1)
    recommendation = RecommendationResponse(
        generated_at=template.as_of,
        fitness_state_as_of=template.as_of,
        planned_for=planned_at,
        workout_type=WorkoutType.LONG,
        title="Projected long run",
        distance_range_miles=(7.5, 8.5),
        confidence=ConfidenceLevel.MODERATE,
        readiness=ReadinessFlag.READY,
        planning_role="long",
    )
    capacity = 16.0
    raw_future = _state_at(template, history, future_at, capacity)
    projected = _project_state(raw_future, [recommendation], CONFIG)
    midpoint = 8.0
    completed = ProjectionRun(
        start_time=planned_at,
        distance_miles=midpoint,
        moving_minutes=midpoint * 11.0,
        workout_type=WorkoutType.LONG,
        difficulty=_projected_difficulty(
            recommendation,
            midpoint,
            11.0,
        ),
        projected=True,
        planning_role="long",
        prescribed_workout_type=WorkoutType.LONG,
        completed_prescribed_workout=True,
        prescribed_low_miles=7.5,
        prescribed_high_miles=8.5,
    )
    observed = _state_at(
        template,
        [*history, completed],
        future_at,
        capacity,
    )

    assert projected.recovery_residual_load == pytest.approx(
        observed.recovery_residual_load,
        abs=0.001,
    )
    assert projected.recent_load.continuous_distance_miles == pytest.approx(
        observed.recent_load.continuous_distance_miles
    )
    assert (
        projected.recent_load.continuous_short_term_distance_miles
        == pytest.approx(
            observed.recent_load.continuous_short_term_distance_miles
        )
    )
    assert projected.recent_load.continuous_fatigue_miles == pytest.approx(
        observed.recent_load.continuous_fatigue_miles,
        abs=0.2,
    )
    assert projected.days_since_long_run == pytest.approx(
        observed.days_since_long_run
    )
    assert projected.recent_load.trailing_28d.distance_miles == pytest.approx(
        observed.recent_load.trailing_28d.distance_miles
    )


@pytest.mark.parametrize(
    ("planned_role", "actual_type", "expected_role"),
    [
        ("ordinary_easy", WorkoutType.EASY, "ordinary_easy"),
        ("support_easy", WorkoutType.EASY, "support_easy"),
        ("medium_long", WorkoutType.EASY, "medium_long"),
        (None, WorkoutType.EASY, "ordinary_easy"),
        ("ordinary_easy", WorkoutType.TEMPO_THRESHOLD, "quality"),
    ],
)
def test_projection_preserves_completed_session_purpose(
    planned_role: str | None,
    actual_type: WorkoutType,
    expected_role: str,
) -> None:
    planned_for = _state().as_of
    recommendation = RecommendationResponse(
        generated_at=planned_for,
        fitness_state_as_of=planned_for,
        planned_for=planned_for,
        workout_type=WorkoutType.EASY,
        title="Role fixture",
        distance_range_miles=(3.5, 4.0),
        confidence=ConfidenceLevel.MODERATE,
        readiness=ReadinessFlag.READY,
        planning_role=planned_role,
    )

    assert _completed_planning_role(recommendation, actual_type) == expected_role


def test_streak_accounting_crosses_the_recorded_to_projected_boundary() -> None:
    tuesday = datetime(2026, 9, 8, tzinfo=timezone.utc).date()
    wednesday = tuesday + timedelta(days=1)
    friday = tuesday + timedelta(days=3)

    assert _consecutive_date_streaks({tuesday, wednesday, friday}) == [
        {tuesday, wednesday},
        {friday},
    ]


def test_projection_can_limit_exact_daily_replans_to_scenario_boundary(
    monkeypatch,
) -> None:
    calls = 0

    def empty_schedule(daily_states, *args, **kwargs):
        nonlocal calls
        calls += 1
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
            summary="Short replay fixture.",
            days=[],
            planning_days=[],
        )

    monkeypatch.setattr(
        "run_analysis.adherence_projection.build_weekly_schedule",
        empty_schedule,
    )
    projection = simulate_adherence(
        _state(),
        [],
        CONFIG,
        _state().as_of,
        weeks=1,
        simulation_days=3,
    )

    assert calls == 3
    assert len(projection) == 1


def test_expected_policy_rollout_stitches_successive_daily_decisions(
    monkeypatch,
) -> None:
    template = _state(
        as_of=datetime(2026, 9, 2, 12, tzinfo=timezone.utc),
        running_days_28d=12,
    )
    calls = 0

    def changing_schedule(daily_states, *args, **kwargs):
        nonlocal calls
        generated_at = daily_states[0].as_of
        first_schedulable_at = kwargs["daily_state_options"][0][0].as_of
        current_type = WorkoutType.EASY if calls == 0 else WorkoutType.LONG
        current_miles = 4.0 if calls == 0 else 6.0
        prescriptions = [(first_schedulable_at, current_type, current_miles)]
        if calls == 0:
            prescriptions.append(
                (
                    first_schedulable_at + timedelta(days=1),
                    WorkoutType.INTERVALS,
                    3.0,
                )
            )
        calls += 1
        days = []
        for planned_for, workout_type, miles in prescriptions:
            recommendation = RecommendationResponse(
                generated_at=generated_at,
                fitness_state_as_of=generated_at,
                planned_for=planned_for,
                workout_type=workout_type,
                title="Policy rollout fixture",
                distance_range_miles=(miles, miles),
                confidence=ConfidenceLevel.MODERATE,
                readiness=ReadinessFlag.READY,
            )
            days.append(
                WeeklyScheduleDay(
                    date=planned_for.date(),
                    planned_at=planned_for,
                    recommendation=recommendation,
                    day_role="fixture_run",
                    rationale="Policy rollout fixture.",
                )
            )
        return WeeklyScheduleResponse(
            generated_at=generated_at,
            start_date=generated_at.date(),
            end_date=generated_at.date() + timedelta(days=20),
            target_run_count=len(days),
            target_distance_range_miles=kwargs["target_distance_range"],
            target_evidence=kwargs["target_evidence"],
            run_count=len(days),
            projected_distance_range_miles=(
                sum(item[2] for item in prescriptions),
                sum(item[2] for item in prescriptions),
            ),
            summary="Policy rollout fixture.",
            days=days,
            planning_days=days,
        )

    monkeypatch.setattr(
        "run_analysis.adherence_projection.build_weekly_schedule",
        changing_schedule,
    )

    rollout = simulate_expected_policy_rollout(
        template,
        [],
        CONFIG,
        template.as_of,
        horizon_days=2,
    )

    assert calls == 2
    assert len(rollout.replans) == 2
    assert len(rollout.steps) == 2
    assert rollout.steps[0].policy_sessions[0].workout_type == WorkoutType.EASY
    assert rollout.steps[1].opening_sessions[0].workout_type == WorkoutType.INTERVALS
    assert rollout.steps[1].policy_sessions[0].workout_type == WorkoutType.LONG
    assert [item.workout_type for item in rollout.opening_horizon_plan] == [
        WorkoutType.EASY,
        WorkoutType.INTERVALS,
    ]
    assert [item.workout_type for item in rollout.policy_plan] == [
        WorkoutType.EASY,
        WorkoutType.LONG,
    ]
    assert rollout.schedule_change_count == 1
    assert rollout.distance_change_count == 0


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


def test_projection_advances_boundary_after_a_same_day_completion(
    monkeypatch,
) -> None:
    template = _state(
        as_of=datetime(2026, 9, 2, 19, tzinfo=timezone.utc)
    )
    completed = ProjectionRun(
        start_time=template.as_of.replace(hour=7),
        distance_miles=4.0,
        moving_minutes=44.0,
        workout_type=WorkoutType.EASY,
        difficulty=_difficulty(miles=4.0),
    )
    captured: list[tuple[datetime, dict]] = []

    def empty_schedule(daily_states, *args, **kwargs):
        captured.append(
            (daily_states[0].as_of, kwargs["completed_activities_by_offset"])
        )
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
            summary="Forward-boundary fixture.",
            days=[],
            planning_days=[],
        )

    monkeypatch.setattr(
        "run_analysis.adherence_projection.build_weekly_schedule",
        empty_schedule,
    )
    simulate_adherence(
        template,
        [completed],
        CONFIG,
        template.as_of,
        weeks=1,
        simulation_days=1,
    )

    assert captured == [
        (template.as_of.replace(day=3, hour=0), {})
    ]


def test_projection_passes_saved_plan_to_first_replan(monkeypatch) -> None:
    template = _state()
    captured_prior_schedules = []
    initial = WeeklyScheduleResponse(
        generated_at=template.as_of - timedelta(hours=1),
        start_date=template.as_of.date(),
        end_date=template.as_of.date() + timedelta(days=6),
        target_run_count=0,
        target_distance_range_miles=(0.0, 0.0),
        target_evidence=derive_weekly_target([], template.as_of, CONFIG)[2],
        run_count=0,
        projected_distance_range_miles=(0.0, 0.0),
        summary="Saved-plan fixture.",
        days=[],
        planning_days=[],
    )

    def empty_schedule(daily_states, *args, **kwargs):
        captured_prior_schedules.append(kwargs.get("prior_schedule"))
        generated_at = daily_states[0].as_of
        return initial.model_copy(
            update={
                "generated_at": generated_at,
                "start_date": generated_at.date(),
                "end_date": generated_at.date() + timedelta(days=6),
            }
        )

    monkeypatch.setattr(
        "run_analysis.adherence_projection.build_weekly_schedule",
        empty_schedule,
    )

    simulate_adherence(
        template,
        [],
        CONFIG,
        template.as_of,
        weeks=1,
        initial_schedule=initial,
    )

    assert captured_prior_schedules[0] is initial
    assert captured_prior_schedules[1] is not initial


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

    def one_run_per_boundary_schedule(daily_states, *args, **kwargs):
        generated_at = daily_states[0].as_of
        planned_for = kwargs["daily_state_options"][0][0].as_of
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
        one_run_per_boundary_schedule,
    )

    projection = simulate_adherence(
        template,
        seed,
        CONFIG,
        template.as_of,
        weeks=2,
        replan_interval_days=1,
    )

    assert [week.run_count for week in projection] == [7, 7]
    assert [week.assumed_completed_miles for week in projection] == [28.0, 28.0]


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


def test_hidden_vacation_attempts_are_not_aggregated_into_one_live_plan(
    monkeypatch,
) -> None:
    template = _state(as_of=_state().as_of.replace(hour=12))

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
            title="Vacation attempt fixture",
            distance_range_miles=(4.0, 4.0),
            confidence=ConfidenceLevel.MODERATE,
            readiness=ReadinessFlag.READY,
        )
        day = WeeklyScheduleDay(
            date=planned_for.date(),
            planned_at=planned_for,
            recommendation=recommendation,
            day_role="easy_run",
            rationale="Vacation attempt fixture.",
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
            summary="Vacation attempt fixture.",
            days=[day],
            planning_days=[day],
        )

    monkeypatch.setattr(
        "run_analysis.adherence_projection.build_weekly_schedule",
        next_morning_schedule,
    )
    projection = simulate_adherence(
        template,
        [],
        CONFIG,
        template.as_of,
        weeks=2,
        human_profile=HumanAdherenceProfile(
            seed=17,
            skip_probability=0.0,
            distance_variation_probability=0.0,
            intensity_drift_probability=0.0,
            substitution_probability=0.0,
            unscheduled_easy_probability_per_day=0.0,
            short_break_count_min=0,
            short_break_count_max=0,
            vacation_probability=1.0,
            vacation_days=7,
        ),
    )

    snapshots = [
        snapshot
        for week in projection
        for snapshot in week.replan_snapshots
    ]
    attempted_dates = {
        session.planned_for.date()
        for snapshot in snapshots
        for session in snapshot.planned_sessions
    }

    assert len(attempted_dates) > 1
    assert max(len(snapshot.planned_sessions) for snapshot in snapshots) == 1
    assert sum(week.skipped_run_count for week in projection) == 7


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
    uploads = [
        snapshot
        for snapshot in projection[0].replan_snapshots
        if snapshot.trigger == "post_upload"
    ]
    assert len(uploads) == 7
    assert all(
        any(reason.startswith("unscheduled:") for reason in snapshot.material_evidence_reasons)
        for snapshot in uploads
    )


def test_human_unscheduled_run_does_not_duplicate_morning_completion(
    monkeypatch,
) -> None:
    template = _state(
        as_of=datetime(2026, 9, 2, 12, tzinfo=timezone.utc),
        typical_easy_run_miles=4.0,
    )
    monkeypatch.setattr(
        "run_analysis.adherence_projection.build_weekly_schedule",
        _single_then_empty_schedule(),
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

    # The scheduled Thursday morning run was committed on Wednesday's replan.
    # Thursday's reload must see it in history before considering an
    # unscheduled Thursday-evening run.
    assert projection[0].planned_run_count == 1
    assert projection[0].unscheduled_run_count == 6
    assert projection[0].run_count == 7


def _single_then_empty_schedule():
    calls = 0

    def schedule(daily_states, *args, **kwargs):
        nonlocal calls
        calls += 1
        generated_at = daily_states[0].as_of
        days = []
        if calls == 1:
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
                title="Overload fixture",
                distance_range_miles=(4.0, 4.0),
                confidence=ConfidenceLevel.MODERATE,
                readiness=ReadinessFlag.READY,
            )
            days.append(
                WeeklyScheduleDay(
                    date=planned_for.date(),
                    planned_at=planned_for,
                    recommendation=recommendation,
                    day_role="easy_run",
                    rationale="Overload fixture.",
                )
            )
        return WeeklyScheduleResponse(
            generated_at=generated_at,
            start_date=generated_at.date(),
            end_date=generated_at.date() + timedelta(days=6),
            target_run_count=1,
            target_distance_range_miles=kwargs["target_distance_range"],
            target_evidence=kwargs["target_evidence"],
            run_count=len(days),
            projected_distance_range_miles=(
                (4.0, 4.0) if days else (0.0, 0.0)
            ),
            summary="Overload fixture.",
            days=days,
            planning_days=days,
        )

    return schedule


def _repeating_range_schedule(
    distance_range: tuple[float, float] = (3.0, 4.0),
):
    def schedule(daily_states, *args, **kwargs):
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
            title="Deterministic scenario fixture",
            distance_range_miles=distance_range,
            confidence=ConfidenceLevel.MODERATE,
            readiness=ReadinessFlag.READY,
        )
        day = WeeklyScheduleDay(
            date=planned_for.date(),
            planned_at=planned_for,
            recommendation=recommendation,
            day_role="easy_run",
            rationale="Deterministic scenario fixture.",
        )
        return WeeklyScheduleResponse(
            generated_at=generated_at,
            start_date=generated_at.date(),
            end_date=generated_at.date() + timedelta(days=6),
            target_run_count=1,
            target_distance_range_miles=kwargs["target_distance_range"],
            target_evidence=kwargs["target_evidence"],
            run_count=1,
            projected_distance_range_miles=distance_range,
            summary="Deterministic scenario fixture.",
            days=[day],
            planning_days=[day],
        )

    return schedule


def test_projection_replans_immediately_after_upload_before_daily_refresh(
    monkeypatch,
) -> None:
    template = _state(
        as_of=datetime(2026, 9, 2, 12, tzinfo=timezone.utc),
        typical_easy_run_miles=4.0,
    )
    monkeypatch.setattr(
        "run_analysis.adherence_projection.build_weekly_schedule",
        _single_then_empty_schedule(),
    )

    projection = simulate_adherence(
        template,
        [],
        CONFIG,
        template.as_of,
        weeks=1,
        simulation_days=2,
    )

    snapshots = projection[0].replan_snapshots
    assert [snapshot.trigger for snapshot in snapshots] == [
        "scheduled_refresh",
        "post_upload",
        "scheduled_refresh",
    ]
    upload = snapshots[1]
    assert upload.source_activity_at is not None
    assert upload.generated_at > upload.source_activity_at
    assert upload.generated_at < snapshots[2].generated_at
    assert upload.decision_start_date == (
        upload.source_activity_at.date() + timedelta(days=1)
    )
    assert not upload.material_evidence_reasons


@pytest.mark.parametrize(
    ("scenario", "expected_miles"),
    [
        (DeterministicAdherenceScenario.IN_RANGE_LOW, 3.0),
        (DeterministicAdherenceScenario.IN_RANGE_HIGH, 4.0),
    ],
)
def test_in_range_edge_scenarios_execute_exact_prescription_edges(
    monkeypatch,
    scenario: DeterministicAdherenceScenario,
    expected_miles: float,
) -> None:
    template = _state(
        as_of=datetime(2026, 9, 2, 12, tzinfo=timezone.utc),
        typical_easy_run_miles=4.0,
    )
    monkeypatch.setattr(
        "run_analysis.adherence_projection.build_weekly_schedule",
        _repeating_range_schedule(),
    )
    seed = [
        ProjectionRun(
            start_time=template.as_of - timedelta(days=day),
            distance_miles=4.0,
            moving_minutes=40.0,
            workout_type=WorkoutType.EASY,
            difficulty=_difficulty(miles=4.0),
            planning_role="ordinary_easy",
        )
        for day in (8, 6, 4, 2)
    ]

    projection = simulate_adherence(
        template,
        seed,
        CONFIG,
        template.as_of,
        weeks=1,
        simulation_days=2,
        deterministic_profile=DeterministicAdherenceProfile(
            scenario=scenario
        ),
    )

    assert projection[0].assumed_completed_miles == expected_miles
    assert not projection[0].adherence_event_records
    comparison = next(
        snapshot
        for snapshot in projection[0].replan_snapshots
        if snapshot.trigger == "post_upload"
    )
    assert comparison.expected_opening_recovery_residual_load is not None
    assert comparison.recovery_surprise_units == pytest.approx(0.0, abs=1e-4)


@pytest.mark.parametrize(
    ("scenario", "expected_fragment"),
    [
        (
            DeterministicAdherenceScenario.BELOW_RANGE_EASY,
            "below prescribed range",
        ),
        (
            DeterministicAdherenceScenario.ABOVE_RANGE_OR_HARDER,
            "above prescribed range; more intense than prescribed",
        ),
    ],
)
def test_out_of_range_scenarios_emit_material_evidence(
    monkeypatch,
    scenario: DeterministicAdherenceScenario,
    expected_fragment: str,
) -> None:
    template = _state(
        as_of=datetime(2026, 9, 2, 12, tzinfo=timezone.utc),
        typical_easy_run_miles=4.0,
    )
    monkeypatch.setattr(
        "run_analysis.adherence_projection.build_weekly_schedule",
        _repeating_range_schedule(),
    )

    projection = simulate_adherence(
        template,
        [],
        CONFIG,
        template.as_of,
        weeks=1,
        simulation_days=2,
        deterministic_profile=DeterministicAdherenceProfile(
            scenario=scenario
        ),
    )

    events = projection[0].adherence_event_records
    assert len(events) == 1
    assert events[0].kind == "deviation"
    assert events[0].detail == expected_fragment
    post_upload = next(
        snapshot
        for snapshot in projection[0].replan_snapshots
        if snapshot.trigger == "post_upload"
    )
    assert post_upload.material_evidence_reasons == (
        f"deviation: {expected_fragment}",
    )
    following_refresh = [
        snapshot
        for snapshot in projection[0].replan_snapshots
        if snapshot.trigger == "scheduled_refresh"
    ][1]
    assert not following_refresh.material_evidence_reasons


def test_single_hidden_miss_skips_only_one_prescription(monkeypatch) -> None:
    template = _state(
        as_of=datetime(2026, 9, 2, 12, tzinfo=timezone.utc),
        typical_easy_run_miles=4.0,
    )
    monkeypatch.setattr(
        "run_analysis.adherence_projection.build_weekly_schedule",
        _repeating_range_schedule(),
    )

    projection = simulate_adherence(
        template,
        [],
        CONFIG,
        template.as_of,
        weeks=1,
        simulation_days=3,
        deterministic_profile=DeterministicAdherenceProfile(
            scenario=DeterministicAdherenceScenario.SINGLE_HIDDEN_MISS
        ),
    )

    assert projection[0].skipped_run_count == 1
    assert projection[0].planned_run_count == 2
    assert projection[0].run_count == 1


def test_known_forced_rest_scenario_supplies_absolute_block_to_planner(
    monkeypatch,
) -> None:
    template = _state(
        as_of=datetime(2026, 9, 2, 12, tzinfo=timezone.utc),
        typical_easy_run_miles=4.0,
    )
    observed_offsets: list[set[int]] = []

    def empty_schedule(daily_states, *args, **kwargs):
        observed_offsets.append(set(kwargs["forced_rest_offsets"]))
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
            summary="Known forced-rest fixture.",
            days=[],
            planning_days=[],
        )

    monkeypatch.setattr(
        "run_analysis.adherence_projection.build_weekly_schedule",
        empty_schedule,
    )
    simulate_adherence(
        template,
        [],
        CONFIG,
        template.as_of,
        weeks=1,
        simulation_days=1,
        deterministic_profile=DeterministicAdherenceProfile(
            scenario=DeterministicAdherenceScenario.KNOWN_FORCED_REST,
            block_start_day=5,
            block_days=3,
        ),
    )

    assert observed_offsets == [{5, 6, 7}]


def test_deterministic_distance_overload_is_observed_by_the_next_replan(
    monkeypatch,
) -> None:
    template = _state(
        as_of=datetime(2026, 9, 2, 12, tzinfo=timezone.utc),
        typical_easy_run_miles=4.0,
    )
    seed = [
        ProjectionRun(
            start_time=template.as_of - timedelta(days=day),
            distance_miles=4.0,
            moving_minutes=44.0,
            workout_type=WorkoutType.EASY,
            difficulty=_difficulty(miles=4.0),
        )
        for day in range(2, 30, 2)
    ]
    monkeypatch.setattr(
        "run_analysis.adherence_projection.build_weekly_schedule",
        _single_then_empty_schedule(),
    )

    projection = simulate_adherence(
        template,
        seed,
        CONFIG,
        template.as_of,
        weeks=1,
        overload_profile=OverloadAdherenceProfile(
            scenario=OverloadScenario.EXTRA_DISTANCE
        ),
    )

    events = projection[0].adherence_event_records
    assert len(events) == 1
    assert events[0].kind == "overload"
    assert "2.0 extra miles" in events[0].detail
    assert projection[0].assumed_completed_miles == 6.0
    snapshots = projection[0].replan_snapshots
    scheduled = [
        snapshot
        for snapshot in snapshots
        if snapshot.trigger == "scheduled_refresh"
    ]
    post_upload = [
        snapshot
        for snapshot in snapshots
        if snapshot.trigger == "post_upload"
    ]
    assert len(scheduled) == 7
    assert len(post_upload) == 1
    assert post_upload[0].opening_load_ratio > scheduled[0].opening_load_ratio


def test_dense_sequence_profile_injects_runs_only_after_daily_replans(
    monkeypatch,
) -> None:
    template = _state(
        as_of=datetime(2026, 9, 2, 12, tzinfo=timezone.utc),
        typical_easy_run_miles=4.0,
    )
    monkeypatch.setattr(
        "run_analysis.adherence_projection.build_weekly_schedule",
        _single_then_empty_schedule(),
    )

    projection = simulate_adherence(
        template,
        [],
        CONFIG,
        template.as_of,
        weeks=1,
        overload_profile=OverloadAdherenceProfile(
            scenario=OverloadScenario.DENSE_SEQUENCE
        ),
    )

    overloads = [
        event
        for event in projection[0].adherence_event_records
        if event.kind == "overload"
    ]
    assert len(overloads) == 2
    assert projection[0].run_count == 3
    assert projection[0].maximum_consecutive_run_days == 3
    assert all("unscheduled easy" in event.detail for event in overloads)


def test_unscheduled_easy_profile_injects_one_rest_day_run(monkeypatch) -> None:
    template = _state(
        as_of=datetime(2026, 9, 2, 12, tzinfo=timezone.utc),
        typical_easy_run_miles=4.0,
    )
    monkeypatch.setattr(
        "run_analysis.adherence_projection.build_weekly_schedule",
        _single_then_empty_schedule(),
    )

    projection = simulate_adherence(
        template,
        [],
        CONFIG,
        template.as_of,
        weeks=1,
        overload_profile=OverloadAdherenceProfile(
            scenario=OverloadScenario.UNSCHEDULED_EASY
        ),
    )

    overloads = [
        event
        for event in projection[0].adherence_event_records
        if event.kind == "overload"
    ]
    assert len(overloads) == 1
    assert projection[0].run_count == 2
    assert "unscheduled easy" in overloads[0].detail


def test_extra_intensity_profile_preserves_distance_and_records_overload(
    monkeypatch,
) -> None:
    template = _state(
        as_of=datetime(2026, 9, 2, 12, tzinfo=timezone.utc),
        typical_easy_run_miles=4.0,
    )
    monkeypatch.setattr(
        "run_analysis.adherence_projection.build_weekly_schedule",
        _single_then_empty_schedule(),
    )

    projection = simulate_adherence(
        template,
        [],
        CONFIG,
        template.as_of,
        weeks=1,
        overload_profile=OverloadAdherenceProfile(
            scenario=OverloadScenario.EXTRA_INTENSITY
        ),
    )

    assert projection[0].assumed_completed_miles == 4.0
    assert len(projection[0].adherence_event_records) == 1
    assert projection[0].adherence_event_records[0].detail == (
        "overload: extra intensity"
    )


def test_overload_diagnostics_reject_coarse_replanning() -> None:
    with pytest.raises(ValueError, match="daily replanning"):
        simulate_adherence(
            _state(),
            [],
            CONFIG,
            _state().as_of,
            weeks=1,
            replan_interval_days=7,
            overload_profile=OverloadAdherenceProfile(
                scenario=OverloadScenario.EXTRA_INTENSITY
            ),
        )


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


def test_first_projection_reload_preserves_latest_recorded_prescription_match() -> None:
    template = _state().model_copy(
        update={
            "last_run_prescribed_workout_type": WorkoutType.INTERVALS,
            "last_run_prescribed_distance_range_miles": (3.5, 4.0),
            "last_run_completed_prescribed_workout": True,
        }
    )
    latest_recorded = ProjectionRun(
        start_time=template.as_of - timedelta(hours=12),
        distance_miles=3.2,
        moving_minutes=38.0,
        workout_type=WorkoutType.INTERVALS,
        difficulty=_difficulty(miles=3.2, quality=True),
    )

    projected = _state_at(
        template,
        [latest_recorded],
        template.as_of,
        capacity_reference=16.0,
    )

    assert projected.last_run_prescribed_workout_type == WorkoutType.INTERVALS
    assert projected.last_run_prescribed_distance_range_miles == (3.5, 4.0)
    assert projected.last_run_completed_prescribed_workout is True


def test_projection_advances_exact_quality_variant_state() -> None:
    template = _state(
        last_completed_quality_session_type=QualitySessionType.SHORT_INTERVALS,
    )
    completed_long_intervals = ProjectionRun(
        start_time=template.as_of - timedelta(hours=12),
        distance_miles=4.7,
        moving_minutes=48.0,
        workout_type=WorkoutType.INTERVALS,
        difficulty=_difficulty(miles=4.7, quality=True),
        projected=True,
        quality_session_type=QualitySessionType.LONG_INTERVALS,
    )

    projected = _state_at(
        template,
        [completed_long_intervals],
        template.as_of,
        capacity_reference=16.0,
    )

    assert (
        projected.last_completed_quality_session_type
        == QualitySessionType.LONG_INTERVALS
    )


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


def test_planner_projected_state_matches_compliant_completed_state() -> None:
    planned_at = datetime(2026, 9, 25, 7, tzinfo=timezone.utc)
    as_of = planned_at + timedelta(days=1)
    template = _state(
        as_of=planned_at - timedelta(days=1),
        typical_easy_run_miles=4.0,
    )
    history = [
        ProjectionRun(
            start_time=planned_at - timedelta(days=day),
            distance_miles=4.0,
            moving_minutes=44.0,
            workout_type=WorkoutType.EASY,
            difficulty=_difficulty(miles=4.0).model_copy(
                update={
                    "zone_breakdown": ZoneBreakdown(
                        easy_minutes=44.0,
                        moderate_minutes=0.0,
                        hard_minutes=0.0,
                    )
                }
            ),
            planning_role="ordinary_easy",
        )
        for day in (9, 7, 4, 2)
    ]
    planned = RecommendationResponse(
        generated_at=planned_at - timedelta(days=1),
        fitness_state_as_of=planned_at - timedelta(days=1),
        planned_for=planned_at,
        workout_type=WorkoutType.INTERVALS,
        quality_session_type=QualitySessionType.SHORT_INTERVALS,
        title="Prescribed quality",
        distance_range_miles=(4.25, 4.75),
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
    raw_future = _state_at(
        template,
        history,
        as_of,
        capacity_reference=18.0,
    )
    projected = _project_state(raw_future, [planned], CONFIG)
    completed = ProjectionRun(
        start_time=planned_at,
        distance_miles=4.5,
        moving_minutes=49.5,
        workout_type=WorkoutType.INTERVALS,
        difficulty=_projected_difficulty(planned, 4.5, 11.0),
        projected=True,
        planning_role="quality",
        prescribed_workout_type=WorkoutType.INTERVALS,
        completed_prescribed_workout=True,
        prescribed_low_miles=4.25,
        prescribed_high_miles=4.75,
        quality_session_type=QualitySessionType.SHORT_INTERVALS,
    )
    actual = _state_at(
        template,
        [*history, completed],
        as_of,
        capacity_reference=18.0,
    )

    # Planning prices the top of the adherence range, so recovery/fatigue may
    # be slightly conservative. Distance-derived state and prescription
    # identity must otherwise agree with an exact midpoint completion.
    assert projected.recovery_residual_load >= actual.recovery_residual_load
    assert projected.recent_load.continuous_distance_miles == pytest.approx(
        actual.recent_load.continuous_distance_miles
    )
    assert (
        projected.recent_load.continuous_short_term_distance_miles
        == pytest.approx(
            actual.recent_load.continuous_short_term_distance_miles
        )
    )
    assert (
        projected.recent_load.continuous_fatigue_miles
        >= actual.recent_load.continuous_fatigue_miles
    )
    assert (
        projected.recent_load.acute_to_prior_ratio
        >= actual.recent_load.acute_to_prior_ratio
    )
    assert projected.last_run_distance_miles == actual.last_run_distance_miles
    assert (
        projected.last_run_prescribed_workout_type
        == actual.last_run_prescribed_workout_type
        == WorkoutType.INTERVALS
    )
    assert (
        projected.last_run_prescribed_distance_range_miles
        == actual.last_run_prescribed_distance_range_miles
        == (4.25, 4.75)
    )
    assert projected.last_run_completed_prescribed_workout is True
    assert projected.easy_fraction_14d == pytest.approx(
        actual.easy_fraction_14d
    )
    assert projected.moderate_fraction_14d == pytest.approx(
        actual.moderate_fraction_14d
    )
    assert projected.hard_fraction_14d == pytest.approx(
        actual.hard_fraction_14d
    )


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
