from __future__ import annotations

from datetime import datetime, timedelta, timezone
from math import log
from types import SimpleNamespace

import pytest
import run_analysis.weekly_schedule as weekly_schedule
from run_analysis.recommendation import recommend_next_run, typical_easy_distance
from run_analysis.weekly_schedule import (
    PlanningActivity,
    _adaptive_candidate_cost,
    _allocate_visible_distance_ranges,
    _decayed_recovery_load,
    _finalized_program_recovery_cost,
    _ordinary_easy_expansion_reference,
    _peak_projected_continuous_mileage_rate,
    _project_state,
    _rest_day_rationale,
    _select_joint_finalists,
    _select_timed_recommendation,
    _target_derived_bridge_reference,
    adaptive_run_day_offsets,
    automatic_run_day_offsets,
    build_weekly_schedule,
    derive_weekly_target,
    summarize_distance_alignment,
)
from run_analysis.web.schemas import (
    ActivityHealthTag,
    ConfidenceLevel,
    CurrentHealthStatus,
    LoadWindow,
    PlannedWeather,
    RecommendationRequest,
    RecommendationResponse,
    ReadinessFlag,
    TrailingDayActivity,
    WeatherExposureBaseline,
    WeeklyScheduleDay,
    WorkoutType,
)
from test_recommendation import CONFIG, _difficulty, _state


def test_automatic_week_uses_recent_frequency_and_allows_intentional_consecutive_days() -> None:
    disrupted = _state(running_days_28d=4)
    assert automatic_run_day_offsets(
        disrupted, CurrentHealthStatus.NORMAL, target_run_count=5
    ) == [0, 1, 3, 4, 6]


def test_yesterday_evening_run_delays_a_morning_slot_without_losing_target() -> None:
    base = _state(
        as_of=datetime(2026, 8, 25, 7, tzinfo=timezone.utc),
        days_since_last_run=0.5,
    )
    assert automatic_run_day_offsets(
        base, CurrentHealthStatus.NORMAL, CONFIG, target_run_count=4
    ) == [1, 3, 4, 6]


def test_yesterday_run_does_not_push_an_evening_slot_to_tomorrow() -> None:
    base = _state(
        as_of=datetime(2026, 8, 25, 19, tzinfo=timezone.utc),
        days_since_last_run=1.0,
    )
    assert automatic_run_day_offsets(
        base, CurrentHealthStatus.NORMAL, CONFIG, target_run_count=4
    ) == [0, 2, 4, 6]


def test_seven_day_fallback_never_restores_today_when_recovery_delays_it() -> None:
    base = _state(
        as_of=datetime(2026, 8, 25, 7, tzinfo=timezone.utc),
        days_since_last_run=0.5,
    )

    assert automatic_run_day_offsets(
        base, CurrentHealthStatus.NORMAL, CONFIG, target_run_count=7
    ) == [1, 2, 3, 4, 5, 6]


def test_fourteen_day_cadence_is_distributed_across_week_boundary() -> None:
    base = _state(days_since_last_run=3.0)

    assert automatic_run_day_offsets(
        base,
        CurrentHealthStatus.NORMAL,
        CONFIG,
        target_run_count=3,
        horizon_days=14,
    ) == [0, 2, 5, 7, 9, 12]


def test_twenty_one_day_cadence_spans_three_continuous_cycles() -> None:
    base = _state(days_since_last_run=3.0)

    assert automatic_run_day_offsets(
        base,
        CurrentHealthStatus.NORMAL,
        CONFIG,
        target_run_count=3,
        horizon_days=21,
    ) == [0, 2, 5, 7, 9, 12, 14, 16, 19]


def test_distance_grid_preserves_a_narrow_off_grid_feasible_band() -> None:
    assert weekly_schedule._distance_options(11.3461, 11.3461) == pytest.approx(
        [11.3461]
    )


def test_visible_continuous_mileage_peak_uses_actual_session_times() -> None:
    opening = datetime(2026, 9, 3, 12, tzinfo=timezone.utc)
    days = [
        SimpleNamespace(
            planned_at=opening,
            recommendation=SimpleNamespace(
                workout_type=WorkoutType.EASY,
                distance_range_miles=(3.5, 4.5),
            ),
        ),
        SimpleNamespace(
            planned_at=opening + timedelta(days=2),
            recommendation=SimpleNamespace(
                workout_type=WorkoutType.LONG,
                distance_range_miles=(6.0, 7.0),
            ),
        ),
    ]

    peak = _peak_projected_continuous_mileage_rate(
        10.0,
        opening,
        days,
        half_life_days=7.0,
    )

    first_rate = 10.0 + log(2.0) * 4.0
    second_rate = first_rate * 0.5 ** (2.0 / 7.0) + log(2.0) * 6.5
    assert peak == pytest.approx(max(first_rate, second_rate))
    assert weekly_schedule._distance_dp_units(11.3461) != (
        weekly_schedule._distance_dp_units(11.5)
    )


def test_candidate_work_reuse_preserves_exact_calendar_selection() -> None:
    base = _state(
        days_since_last_run=2.0,
        days_since_long_run=6.0,
        days_since_quality_run=5.0,
        running_days_28d=12,
    )
    states = [
        base.model_copy(
            update={
                "as_of": base.as_of + timedelta(days=offset),
                "days_since_last_run": 2.0 + offset,
                "days_since_long_run": 6.0 + offset,
                "days_since_quality_run": 5.0 + offset,
            }
        )
        for offset in range(7)
    ]
    request = RecommendationRequest(health_status=CurrentHealthStatus.NORMAL)

    optimized = adaptive_run_day_offsets(
        states,
        request,
        CONFIG,
        target_run_count=4,
        target_distance_range=(16.0, 18.0),
    )
    uncached = adaptive_run_day_offsets(
        states,
        request,
        CONFIG,
        target_run_count=4,
        target_distance_range=(16.0, 18.0),
        _reuse_candidate_work=False,
    )

    assert optimized == uncached


def test_exact_time_can_clear_taxing_run_guardrail() -> None:
    previous_at = datetime(2026, 8, 27, 7, tzinfo=timezone.utc)
    previous = recommend_next_run(
        _state(as_of=previous_at),
        RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
        CONFIG,
        weekly_role="easy",
    ).model_copy(
        update={
            "workout_type": WorkoutType.LONG,
            "planned_for": previous_at,
        }
    )
    morning_at = previous_at + timedelta(hours=24)
    evening_at = previous_at + timedelta(hours=36)
    morning = _state(
        as_of=morning_at,
        planned_weather=PlannedWeather(
            forecast_time=morning_at,
            apparent_temperature_f=60,
        ),
    )
    evening = _state(
        as_of=evening_at,
        planned_weather=PlannedWeather(
            forecast_time=evening_at,
            apparent_temperature_f=70,
        ),
    )

    selected_state, result = _select_timed_recommendation(
        [morning, evening],
        [previous],
        RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
        CONFIG,
        weekly_role="quality",
        allowed_candidates={"quality"},
    )

    assert selected_state.as_of == evening_at
    assert result.workout_type not in {WorkoutType.REST, WorkoutType.RECOVERY}
    recovery = next(
        item for item in result.rule_trace
        if item.rule_id == "recent_recovery_load"
    )
    assert recovery.facts["elapsed_hours"] == 36
    assert recovery.facts["hours_until_easy"] == 0


def test_extra_recovery_does_not_force_later_slot_after_both_are_ready() -> None:
    previous_at = datetime(2026, 8, 28, 19, tzinfo=timezone.utc)
    previous = recommend_next_run(
        _state(as_of=previous_at),
        RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
        CONFIG,
        weekly_role="easy",
    ).model_copy(
        update={
            "workout_type": WorkoutType.LONG,
            "planned_for": previous_at,
        }
    )
    morning_at = previous_at + timedelta(hours=36)
    evening_at = previous_at + timedelta(hours=48)
    options = [
        _state(
            as_of=planned_at,
            planned_weather=PlannedWeather(
                forecast_time=planned_at,
                apparent_temperature_f=65,
            ),
        )
        for planned_at in (morning_at, evening_at)
    ]

    selected_state, result = _select_timed_recommendation(
        options,
        [previous],
        RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
        CONFIG,
        weekly_role="quality",
    )

    assert selected_state.as_of == morning_at
    assert result.workout_type not in {WorkoutType.REST, WorkoutType.RECOVERY}


def test_projected_run_adds_to_completed_run_recovery_residue() -> None:
    planned_at = datetime(2026, 9, 2, 12, tzinfo=timezone.utc)
    planned = recommend_next_run(
        _state(as_of=planned_at),
        RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
        CONFIG,
        weekly_role="easy",
    ).model_copy(update={"planned_for": planned_at})
    future = _state(as_of=planned_at + timedelta(hours=12)).model_copy(
        update={"recovery_residual_load": 0.4}
    )

    projected = _project_state(future, [planned])

    planned_units = weekly_schedule._recommendation_load_units(
        planned,
        weekly_schedule.projected_recovery_reference_miles(future),
    )
    assert projected.recovery_residual_load == pytest.approx(
        0.4 + planned_units * 0.5
    )


def test_projected_support_runs_do_not_redefine_typical_easy_distance() -> None:
    planned_at = datetime(2026, 9, 2, 12, tzinfo=timezone.utc)
    future = _state(as_of=planned_at + timedelta(days=4))
    short_support = recommend_next_run(
        _state(as_of=planned_at),
        RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
        CONFIG,
        weekly_role="easy",
        allowed_candidates={"easy"},
    ).model_copy(
        update={
            "planned_for": planned_at,
            "distance_range_miles": (2.5, 3.0),
            "planning_role": "support_easy",
        }
    )

    projected = _project_state(future, [short_support], CONFIG)

    assert typical_easy_distance(projected) == typical_easy_distance(future)


def test_projected_quality_count_uses_only_the_trailing_fourteen_days() -> None:
    first_at = datetime(2026, 9, 2, 19, tzinfo=timezone.utc)
    base = _state(
        as_of=first_at,
        quality_sessions_14d=0,
        completed_quality_session_count=0,
    )
    quality = recommend_next_run(
        base,
        RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
        CONFIG,
        weekly_role="quality",
        allowed_candidates={"quality"},
    )
    older = quality.model_copy(update={"planned_for": first_at})
    recent = quality.model_copy(
        update={"planned_for": first_at + timedelta(days=18)}
    )
    future = base.model_copy(update={"as_of": first_at + timedelta(days=20)})

    projected = _project_state(future, [older, recent], CONFIG)

    assert projected.quality_sessions_14d == 1
    assert projected.completed_quality_session_count == 2


def test_due_quality_role_is_scaled_instead_of_silently_replaced_by_easy() -> None:
    planned_at = datetime(2026, 9, 2, 19, tzinfo=timezone.utc)
    base = _state(
        as_of=planned_at,
        days_since_last_run=3.0,
        days_since_quality_run=10.0,
        days_since_long_run=3.0,
        running_days_28d=14,
    )
    base = base.model_copy(
        update={
            "recent_load": base.recent_load.model_copy(
                update={
                    "capacity_reference_miles": 16.0,
                    "continuous_fatigue_miles": 18.0,
                    "continuous_fatigue_to_capacity_ratio": 1.125,
                    "continuous_distance_miles": 18.0,
                }
            )
        }
    )

    _, result = weekly_schedule._select_budgeted_timed_recommendation(
        [base],
        [],
        RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
        CONFIG,
        weekly_role="quality",
    )

    assert result.workout_type in {
        WorkoutType.INTERVALS,
        WorkoutType.TEMPO_THRESHOLD,
    }


def test_projected_run_adds_to_continuously_decaying_fatigue() -> None:
    planned_at = datetime(2026, 9, 2, 12, tzinfo=timezone.utc)
    planned = recommend_next_run(
        _state(as_of=planned_at),
        RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
        CONFIG,
        weekly_role="easy",
    ).model_copy(update={"planned_for": planned_at})
    base = _state(as_of=planned_at + timedelta(days=1))
    future = base.model_copy(
        update={
            "recent_load": base.recent_load.model_copy(
                update={
                    "continuous_fatigue_miles": 10.0,
                    "continuous_fatigue_to_capacity_ratio": 0.5,
                    "capacity_reference_miles": 20.0,
                }
            )
        }
    )

    projected = _project_state(future, [planned], CONFIG)

    easy_reference = sum(typical_easy_distance(future)) / 2
    planned_units = weekly_schedule._recommendation_load_units(
        planned, easy_reference
    )
    expected = (
        10.0
        + log(2.0)
        * planned_units
        * easy_reference
        * 0.5 ** (1.0 / 7.0)
    )
    assert projected.recent_load.continuous_fatigue_miles == pytest.approx(
        expected
    )
    assert projected.recent_load.continuous_fatigue_to_capacity_ratio == (
        pytest.approx(expected / 20.0)
    )


def test_today_rest_rationale_exposes_weather_and_recovery_tradeoff() -> None:
    tonight = datetime(2026, 8, 28, 19, tzinfo=timezone.utc)
    tomorrow = tonight + timedelta(hours=12)
    baseline = WeatherExposureBaseline(
        sample_count=6,
        warm_apparent_temperature_f=75,
        cold_apparent_temperature_f=60,
        humid_dewpoint_f=60,
    )
    current = _state(
        as_of=tonight,
        weather_exposure_baseline=baseline,
        planned_weather=PlannedWeather(
            forecast_time=tonight,
            apparent_temperature_f=84.2,
            dewpoint_f=71.3,
        ),
    )
    upcoming = _state(
        as_of=tomorrow,
        weather_exposure_baseline=baseline,
        planned_weather=PlannedWeather(
            forecast_time=tomorrow,
            apparent_temperature_f=60,
            dewpoint_f=56,
        ),
    )

    rationale = _rest_day_rationale(0, [1], [[current], [upcoming]])

    assert "Saturday" in rationale
    assert "moderate to none" in rationale
    assert "12 recovery hours" in rationale


def test_low_cost_run_yesterday_can_preserve_intentional_next_day_run() -> None:
    low_cost = _difficulty(miles=1.1, rpe=6).model_copy(
        update={"moving_minutes": 11.0, "elapsed_minutes": 12.0, "zone_load": 9.0}
    )
    base = _state(days_since_last_run=1.0, last_run=low_cost)

    assert automatic_run_day_offsets(
        base, CurrentHealthStatus.NORMAL, CONFIG, target_run_count=4
    ) == [0, 2, 4, 6]


def test_low_cost_run_does_not_override_cumulative_recovery_load() -> None:
    low_cost = _difficulty(miles=1.1, rpe=6).model_copy(
        update={"moving_minutes": 11.0, "elapsed_minutes": 12.0, "zone_load": 9.0}
    )
    base = _state(
        days_since_last_run=1.0,
        last_run=low_cost,
        recovery_residual_load=0.9,
    )

    assert automatic_run_day_offsets(
        base, CurrentHealthStatus.NORMAL, CONFIG, target_run_count=4
    ) == [1, 3, 4, 6]


def test_seven_day_fallback_retains_requested_work_inside_its_boundary() -> None:
    base = _state(
        as_of=datetime(2026, 8, 25, 7, tzinfo=timezone.utc),
        days_since_last_run=0.5,
        days_since_quality_run=20,
        days_since_long_run=20,
        running_days_28d=10,
    )
    states = [
        base.model_copy(
            update={
                "as_of": base.as_of + timedelta(days=offset),
                "days_since_last_run": 0.5 + offset,
                "days_since_quality_run": 20.0 + offset,
                "days_since_long_run": 20.0 + offset,
            }
        )
        for offset in range(7)
    ]
    schedule = build_weekly_schedule(
        states,
        RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
        CONFIG,
        target_run_count=4,
        target_distance_range=(15.5, 18.0),
    )
    assert [index for index, day in enumerate(schedule.days) if day.recommendation] == [
        1,
        3,
        4,
        6,
    ]
    assert schedule.run_count == 4
    assert schedule.target_run_count == 4


def test_final_visible_run_role_comes_from_elapsed_cadence_not_horizon() -> None:
    base = _state(
        days_since_last_run=2.0,
        days_since_quality_run=20,
        days_since_long_run=0,
        running_days_28d=10,
    )
    states = [
        base.model_copy(
            update={
                "as_of": base.as_of + timedelta(days=offset),
                "days_since_last_run": 2.0 + offset,
                "days_since_quality_run": 20.0 + offset,
                "days_since_long_run": float(offset),
            }
        )
        for offset in range(7)
    ]
    schedule = build_weekly_schedule(
        states,
        RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
        CONFIG,
        target_run_count=4,
        target_distance_range=(15.5, 18.0),
    )
    final = schedule.days[6].recommendation
    assert final is not None
    assert final.workout_type != WorkoutType.LONG
    sequence_trace = next(
        item for item in final.rule_trace if item.rule_id == "weekly_sequence_priority"
    )
    assert sequence_trace.facts["preferred_role"] != "long"


def test_quality_cadence_uses_last_session_instead_of_reload_week() -> None:
    start = datetime(2026, 8, 29, 12, tzinfo=timezone.utc)
    inferred_last_quality = start - timedelta(days=3)
    quality_types = {
        WorkoutType.INTERVALS,
        WorkoutType.TEMPO_THRESHOLD,
        WorkoutType.RACE,
    }

    for reload_days_later in (0, 1):
        base = _state(
            as_of=start + timedelta(days=reload_days_later),
            days_since_last_run=1.0 + reload_days_later,
            days_since_quality_run=3.0 + reload_days_later,
            days_since_long_run=1.0 + reload_days_later,
            running_days_28d=10,
        )
        base = base.model_copy(
            update={
                "recent_load": base.recent_load.model_copy(
                    update={
                        "capacity_reference_miles": 20.0,
                        "sustained_capacity_miles": 20.0,
                        "acute_distance_to_capacity_ratio": 0.6,
                    }
                )
            }
        )
        states = [
            base.model_copy(
                update={
                    "as_of": base.as_of + timedelta(days=offset),
                    "days_since_last_run": base.days_since_last_run + offset,
                    "days_since_quality_run": base.days_since_quality_run + offset,
                    "days_since_long_run": base.days_since_long_run + offset,
                }
            )
            for offset in range(14)
        ]

        schedule = build_weekly_schedule(
            states,
            RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
            CONFIG,
            target_run_count=4,
            target_distance_range=(15.5, 18.0),
        )
        quality = next(
            day.recommendation
            for day in schedule.days
            if day.recommendation
            and day.recommendation.workout_type in quality_types
        )
        assert quality.planned_for is not None
        elapsed_days = (
            quality.planned_for - inferred_last_quality
        ).total_seconds() / 86400
        # Seven days is the cadence center, not a block boundary or deadline.
        # The ordinary one-day grace remains a graded cost rather than a hard
        # prohibition, but an overdue occurrence must not erase its own delay.
        assert 7.0 <= elapsed_days <= 8.5


def test_long_cadence_uses_last_session_instead_of_reload_week() -> None:
    start = datetime(2026, 8, 29, 12, tzinfo=timezone.utc)
    inferred_last_long = start - timedelta(days=3)

    for reload_days_later in (0, 1):
        base = _state(
            as_of=start + timedelta(days=reload_days_later),
            days_since_last_run=1.0 + reload_days_later,
            days_since_quality_run=1.0 + reload_days_later,
            days_since_long_run=3.0 + reload_days_later,
            longest_run_30d_miles=6.5,
            retained_long_run_capacity_miles=6.5,
            running_days_28d=10,
        )
        base = base.model_copy(
            update={
                "recent_load": base.recent_load.model_copy(
                    update={
                        "capacity_reference_miles": 20.0,
                        "sustained_capacity_miles": 20.0,
                        "acute_distance_to_capacity_ratio": 0.6,
                    }
                )
            }
        )
        states = [
            base.model_copy(
                update={
                    "as_of": base.as_of + timedelta(days=offset),
                    "days_since_last_run": base.days_since_last_run + offset,
                    "days_since_quality_run": base.days_since_quality_run + offset,
                    "days_since_long_run": base.days_since_long_run + offset,
                }
            )
            for offset in range(14)
        ]

        schedule = build_weekly_schedule(
            states,
            RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
            CONFIG,
            target_run_count=4,
            target_distance_range=(15.5, 18.0),
        )
        long_run = next(
            day.recommendation
            for day in schedule.days
            if day.recommendation
            and day.recommendation.workout_type == WorkoutType.LONG
        )
        assert long_run.planned_for is not None
        elapsed_days = (
            long_run.planned_for - inferred_last_long
        ).total_seconds() / 86400
        assert 7.0 <= elapsed_days <= 8.5


def test_dynamic_target_uses_sustained_history_not_only_latest_week() -> None:
    base = _state()
    activities = [
        PlanningActivity(base.as_of - timedelta(days=day), 4.0)
        for day in range(0, 84, 2)
    ]
    runs, distance, evidence = derive_weekly_target(activities, base.as_of, CONFIG)
    assert runs >= 4
    assert distance[0] > 10
    assert evidence.best_sustained_28d_weekly_miles >= evidence.chronic_42d_weekly_miles


def test_general_fitness_target_builds_above_demonstrated_capacity() -> None:
    base = _state()
    activities = [
        PlanningActivity(base.as_of - timedelta(days=day), 4.0)
        for day in range(0, 84, 2)
    ]

    _, distance, evidence = derive_weekly_target(activities, base.as_of, CONFIG)

    assert sum(distance) / 2 > evidence.capacity_reference_miles
    assert distance[1] > evidence.capacity_reference_miles
    assert "successful training earns" in evidence.rationale


def test_continuous_mileage_path_penalizes_parking_load_late() -> None:
    steady = [
        0.0, 4.5, 0.0, 4.5, 0.0, 4.5, 0.0,
        4.5, 0.0, 4.5, 0.0, 4.5, 0.0, 4.5,
        0.0, 4.5, 0.0, 4.5, 0.0, 4.5, 0.0,
    ]
    parked = [0.0] * 14 + [9.0] * 7

    steady_cost = weekly_schedule._continuous_mileage_path_cost(
        steady,
        (17.0, 19.0),
        session_tolerance_miles=4.5,
    )
    parked_cost = weekly_schedule._continuous_mileage_path_cost(
        parked,
        (17.0, 19.0),
        session_tolerance_miles=4.5,
    )

    assert steady_cost < parked_cost


def test_continuous_mileage_path_has_no_day_seven_boundary() -> None:
    before_boundary = [0.0] * 5 + [6.0] + [0.0] * 15
    after_boundary = [0.0] * 7 + [6.0] + [0.0] * 13

    before_cost = weekly_schedule._continuous_mileage_path_cost(
        before_boundary,
        (17.0, 19.0),
        session_tolerance_miles=4.0,
    )
    after_cost = weekly_schedule._continuous_mileage_path_cost(
        after_boundary,
        (17.0, 19.0),
        session_tolerance_miles=4.0,
    )

    assert before_cost < after_cost
    assert before_cost > 0


def test_weekly_rate_range_is_centered_on_its_midpoint() -> None:
    target = (16.9, 18.0)

    center_cost = weekly_schedule._weekly_rate_alignment_cost(17.45, target)
    low_edge_cost = weekly_schedule._weekly_rate_alignment_cost(16.9, target)
    high_edge_cost = weekly_schedule._weekly_rate_alignment_cost(18.0, target)

    assert center_cost == 0
    assert center_cost < low_edge_cost
    assert center_cost < high_edge_cost


def test_continuous_path_is_invariant_when_a_rest_day_moves_the_origin() -> None:
    opening = 12.5
    target = (17.0, 18.0)
    future = [0.0, 4.5, 0.0, 4.5, 0.0, 4.5, 0.0]
    tolerance = 4.0 * log(2.0)

    original = weekly_schedule._continuous_mileage_path_cost(
        future,
        target,
        session_tolerance_miles=tolerance,
        opening_weekly_rate=opening,
        half_life_days=7,
    )
    reloaded = weekly_schedule._continuous_mileage_path_cost(
        future[1:],
        target,
        session_tolerance_miles=tolerance,
        opening_weekly_rate=opening * 0.5 ** (1 / 7),
        half_life_days=7,
    )
    assert original == pytest.approx(reloaded)


def test_prior_full_plan_is_only_a_soft_continuity_preference() -> None:
    prior = {1, 3, 5, 8}

    unchanged = weekly_schedule._plan_continuity_cost([1, 3, 5, 8], prior)
    shifted = weekly_schedule._plan_continuity_cost([1, 4, 6, 8], prior)

    assert unchanged == 0
    assert shifted > 0
    assert weekly_schedule._plan_continuity_cost([1, 3, 5, 20], prior) < shifted


def test_completed_prior_prescription_preserves_same_day_rest() -> None:
    as_of = _state().as_of
    planned_for = as_of - timedelta(days=1)
    prescription = RecommendationResponse(
        generated_at=planned_for - timedelta(days=1),
        fitness_state_as_of=planned_for - timedelta(days=1),
        planned_for=planned_for,
        workout_type=WorkoutType.EASY,
        title="Easy aerobic run",
        distance_range_miles=(3.5, 4.0),
        confidence=ConfidenceLevel.MODERATE,
        readiness=ReadinessFlag.READY,
    )
    prior_days = [
        WeeklyScheduleDay(
            date=planned_for.date(),
            planned_at=planned_for,
            recommendation=prescription,
            day_role="easy_run",
            rationale="Continuity fixture.",
        ),
        WeeklyScheduleDay(
            date=as_of.date(),
            planned_at=None,
            recommendation=None,
            day_role="rest_day",
            rationale="Expected post-workout rest.",
        ),
    ]
    matched = _state(
        as_of=as_of,
        days_since_last_run=1.0,
        last_run=_difficulty(miles=3.8),
        last_run_workout_type=WorkoutType.EASY,
    )
    too_long = matched.model_copy(
        update={"last_run": _difficulty(miles=4.8)}
    )

    assert weekly_schedule._latest_run_completed_prior_prescription(
        matched,
        prior_days,
    )
    assert not weekly_schedule._latest_run_completed_prior_prescription(
        too_long,
        prior_days,
    )
    assert weekly_schedule._post_adherence_rest_offset(
        matched,
        prior_days,
        21,
    ) == 0
    consecutive_plan = [
        prior_days[0],
        prior_days[1].model_copy(
            update={
                "planned_at": as_of,
                "recommendation": prescription.model_copy(
                    update={"planned_for": as_of}
                ),
                "day_role": "easy_run",
            }
        ),
    ]
    assert weekly_schedule._post_adherence_rest_offset(
        matched,
        consecutive_plan,
        21,
    ) is None
    reloaded = matched.model_copy(
        update={
            "last_run_prescribed_workout_type": WorkoutType.EASY,
            "last_run_prescribed_distance_range_miles": (3.5, 4.0),
        }
    )
    assert weekly_schedule._post_adherence_rest_offset(
        reloaded,
        [prior_days[1]],
        21,
    ) == 0
    completed_quality_dose = reloaded.model_copy(
        update={
            "last_run": _difficulty(miles=3.2, quality=True),
            "last_run_workout_type": WorkoutType.INTERVALS,
            "last_run_prescribed_workout_type": WorkoutType.INTERVALS,
            "last_run_prescribed_distance_range_miles": (3.5, 4.0),
            "last_run_completed_prescribed_workout": True,
        }
    )
    assert weekly_schedule._post_adherence_rest_offset(
        completed_quality_dose,
        [prior_days[1]],
        21,
    ) == 0


def test_full_model_calendar_choice_keeps_continuity_in_final_comparison(
    monkeypatch,
) -> None:
    base = _state(running_days_28d=10)
    states = [
        base.model_copy(update={"as_of": base.as_of + timedelta(days=offset)})
        for offset in range(3)
    ]
    monkeypatch.setattr(
        weekly_schedule,
        "_adaptive_candidate_cost",
        lambda *args, **kwargs: 0.0,
    )
    monkeypatch.setattr(
        weekly_schedule,
        "_joint_candidate_program_cost",
        lambda *args, **kwargs: (0.0, 0.0, 0.0, 0.0),
    )

    selected = weekly_schedule._adaptive_run_day_offsets_for_frequency(
        states,
        RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
        CONFIG,
        target_run_count=1,
        target_distance_range=(3.0, 4.0),
        horizon_run_count=1,
        joint_program_scoring=True,
        prior_run_offsets={1},
    )

    assert selected == [1]


def test_twenty_one_day_plan_is_retained_internally_but_not_serialized() -> None:
    base = _state(running_days_28d=10)
    states = [
        base.model_copy(update={"as_of": base.as_of + timedelta(days=offset)})
        for offset in range(21)
    ]

    schedule = build_weekly_schedule(
        states,
        RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
        CONFIG,
        target_run_count=4,
        target_distance_range=(15.5, 18.0),
    )

    assert len(schedule.days) == 7
    assert len(schedule.planning_days) == 21
    assert "planning_days" not in schedule.model_dump()


def test_no_history_requires_a_measured_baseline_instead_of_invented_mileage() -> None:
    base = _state()

    runs, distance, evidence = derive_weekly_target([], base.as_of, CONFIG)

    assert runs == 1
    assert distance == (0.0, 0.0)
    assert evidence.planning_mode == "baseline_required"
    assert evidence.capacity_reference_miles == 0
    assert "will not invent" in evidence.rationale


def test_sparse_history_repeats_observed_session_without_weekly_extrapolation() -> None:
    base = _state()
    activities = [
        PlanningActivity(
            base.as_of - timedelta(days=1),
            1.3,
            moving_minutes=20.0,
            easy_minutes=20.0,
        )
    ]

    runs, distance, evidence = derive_weekly_target(
        activities,
        base.as_of,
        CONFIG,
    )

    assert runs == 1
    assert distance == (0.0, 0.0)
    assert evidence.planning_mode == "baseline_building"
    assert evidence.baseline_session_miles == 1.3
    assert evidence.baseline_session_minutes == 20.0
    assert evidence.earned_progression_fraction == 0


def test_a_lone_long_or_hard_run_is_not_repeated_as_a_baseline() -> None:
    base = _state()
    activities = [
        PlanningActivity(
            base.as_of - timedelta(days=1),
            13.1,
            baseline_eligible=False,
        )
    ]

    runs, distance, evidence = derive_weekly_target(
        activities,
        base.as_of,
        CONFIG,
    )

    assert runs == 1
    assert distance == (0.0, 0.0)
    assert evidence.planning_mode == "baseline_required"
    assert evidence.baseline_session_miles is None


def test_baseline_builder_schedules_only_one_observed_easy_repeat() -> None:
    base = _state(days_since_last_run=2.0)
    activities = [PlanningActivity(base.as_of - timedelta(days=2), 1.3)]
    target_runs, target, evidence = derive_weekly_target(
        activities,
        base.as_of,
        CONFIG,
    )
    states = [
        base.model_copy(
            update={
                "as_of": base.as_of + timedelta(days=offset),
                "days_since_last_run": 2.0 + offset,
            }
        )
        for offset in range(21)
    ]

    schedule = build_weekly_schedule(
        states,
        RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
        CONFIG,
        target_run_count=target_runs,
        target_distance_range=target,
        target_evidence=evidence,
    )

    planned = [day.recommendation for day in schedule.days if day.recommendation]
    assert len(planned) == 1
    assert planned[0].workout_type == WorkoutType.EASY
    assert planned[0].distance_range_miles == (1.3, 1.3)
    assert "baseline" in planned[0].title.lower()


def test_no_history_schedules_one_time_based_conversational_baseline() -> None:
    base = _state(days_since_last_run=None)
    target_runs, target, evidence = derive_weekly_target([], base.as_of, CONFIG)
    states = [
        base.model_copy(update={"as_of": base.as_of + timedelta(days=offset)})
        for offset in range(21)
    ]

    schedule = build_weekly_schedule(
        states,
        RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
        CONFIG,
        target_run_count=target_runs,
        target_distance_range=target,
        target_evidence=evidence,
    )

    planned = [day.recommendation for day in schedule.days if day.recommendation]
    assert len(planned) == 1
    assert planned[0].title == "Conversational baseline run"
    assert planned[0].distance_range_miles is None
    assert planned[0].duration_range_minutes == (10.0, 30.0)
    assert planned[0].target_zones == ["Z2"]
    assert "Stop for pain, fatigue, and/or elevated heart rate" in planned[0].structure[1].instruction
    assert schedule.run_count == 1
    assert schedule.target_distance_range_miles == (0.0, 0.0)
    assert schedule.projected_distance_range_miles == (0.0, 0.0)


def test_off_week_scales_growth_bonus_without_erasing_retained_capacity() -> None:
    base = _state()
    sustained_history = [
        PlanningActivity(base.as_of - timedelta(days=day), 4.0)
        for day in range(28, 84, 2)
    ]
    recent_capacity_week = [
        PlanningActivity(base.as_of - timedelta(days=day), 4.0)
        for day in (0, 2, 4, 6)
    ]
    uninterrupted = [
        PlanningActivity(base.as_of - timedelta(days=day), 4.0)
        for day in range(0, 28, 2)
    ]

    _, continuous_target, continuous_evidence = derive_weekly_target(
        sustained_history + uninterrupted,
        base.as_of,
        CONFIG,
    )
    _, interrupted_target, interrupted_evidence = derive_weekly_target(
        sustained_history + recent_capacity_week,
        base.as_of,
        CONFIG,
    )

    assert (
        abs(
            interrupted_evidence.capacity_reference_miles
            - continuous_evidence.capacity_reference_miles
        )
        < 0.01
    )
    assert continuous_evidence.progression_continuity == 1.0
    assert 0 < interrupted_evidence.progression_continuity < 1.0
    assert (
        interrupted_evidence.earned_progression_fraction
        < continuous_evidence.earned_progression_fraction
    )
    interrupted_midpoint = sum(interrupted_target) / 2
    continuous_midpoint = sum(continuous_target) / 2
    assert interrupted_evidence.capacity_reference_miles <= interrupted_midpoint
    assert interrupted_midpoint < continuous_midpoint


def test_one_extra_run_day_does_not_ratchet_next_week_frequency() -> None:
    base = _state()
    ordinary_offsets = [
        0, 2, 4, 6,
        7, 9, 11, 13,
        14, 16, 18, 20,
        21, 23, 25, 27,
    ]
    activities = [
        PlanningActivity(base.as_of - timedelta(days=day), 4.0)
        for day in ordinary_offsets
    ]
    # Five distinct run days now appear in the latest seven days, but the
    # extra day has not established a five-day-per-week training pattern.
    activities.append(PlanningActivity(base.as_of - timedelta(days=1), 1.1))

    runs, _, evidence = derive_weekly_target(activities, base.as_of, CONFIG)

    assert runs == 4
    assert evidence.demonstrated_run_days_per_week == 4.25


def test_short_interruption_retains_capacity_without_forcing_old_frequency() -> None:
    base = _state()
    activities = [
        PlanningActivity(base.as_of - timedelta(days=35 + week * 7 + day), 4.0)
        for week in range(4)
        for day in (0, 2, 4, 6)
    ]
    config = {
        **CONFIG,
        "coaching": {
            **CONFIG["coaching"],
            "capacity_retention_grace_days": 28,
            "capacity_retention_half_life_days": 84,
        },
    }

    runs, distance, evidence = derive_weekly_target(activities, base.as_of, config)

    assert runs == 1
    assert distance[0] >= 14
    assert evidence.demonstrated_run_days_per_week > 3.5
    assert evidence.current_run_days_per_week == 0
    assert evidence.progression_continuity == 0
    assert abs(sum(distance) / 2 - evidence.capacity_reference_miles) <= 0.1


def test_recent_high_week_remains_capacity_evidence_after_trailing_window() -> None:
    base = _state()
    established = [
        PlanningActivity(base.as_of - timedelta(days=100 + week * 7 + day), 4.0)
        for week in range(4)
        for day in (0, 2, 4, 6)
    ]
    recent_high_week = [
        PlanningActivity(base.as_of - timedelta(days=day), 4.0)
        for day in (12, 13, 14, 16, 18)
    ]

    _, distance, evidence = derive_weekly_target(
        established + recent_high_week,
        base.as_of,
        CONFIG,
    )

    assert evidence.recent_7d_miles == 0
    assert evidence.capacity_reference_miles >= 15.5
    assert distance[0] >= 14.5


def test_longer_interruption_blends_old_frequency_toward_recent_routine() -> None:
    base = _state()
    historical = [
        PlanningActivity(base.as_of - timedelta(days=86 + week * 7 + day), 4.0)
        for week in range(4)
        for day in (0, 2, 4, 6)
    ]
    recent = [
        PlanningActivity(base.as_of - timedelta(days=day), 3.0)
        for day in (0, 3, 7, 10, 14, 17, 21, 24)
    ]
    config = {
        **CONFIG,
        "coaching": {
            **CONFIG["coaching"],
            "capacity_retention_grace_days": 28,
            "capacity_retention_half_life_days": 84,
        },
    }

    runs, _, evidence = derive_weekly_target(
        historical + recent, base.as_of, config
    )

    assert runs == 3
    assert 3.0 < evidence.demonstrated_run_days_per_week < 3.5


def test_current_cadence_limits_retained_four_day_routine_to_three_runs() -> None:
    base = _state()
    historical = [
        PlanningActivity(base.as_of - timedelta(days=60 + week * 7 + day), 4.0)
        for week in range(4)
        for day in (0, 2, 4, 6)
    ]
    current = [
        PlanningActivity(base.as_of - timedelta(days=day), 4.0)
        for day in (1, 3, 6, 9, 13, 17, 21, 24, 27)
    ]

    runs, _, evidence = derive_weekly_target(
        historical + current, base.as_of, CONFIG
    )

    assert evidence.current_run_days_per_week == 2.25
    assert evidence.demonstrated_run_days_per_week > 3.5
    assert runs == 3


def test_partial_range_overlap_is_not_described_as_a_full_fit() -> None:
    assert summarize_distance_alignment(
        (14.5, 16.5), (15.5, 18.0), 16.4
    ) == (
        "The lower end is below your usual range; the upper end reaches it if recovery remains normal."
    )


def test_weekly_schedule_coordinates_seven_days() -> None:
    base = _state(running_days_28d=10)
    states = [
        base.model_copy(
            update={
                "as_of": base.as_of + timedelta(days=offset),
                "days_since_last_run": (base.days_since_last_run or 0) + offset,
                "days_since_quality_run": (base.days_since_quality_run or 0) + offset,
                "days_since_long_run": (base.days_since_long_run or 0) + offset,
            }
        )
        for offset in range(7)
    ]
    schedule = build_weekly_schedule(
        states,
        RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
        CONFIG,
    )
    assert len(schedule.days) == 7
    assert schedule.run_count >= 2
    assert schedule.projected_distance_range_miles[1] >= schedule.projected_distance_range_miles[0]
    assert any(day.recommendation is None for day in schedule.days)
    assert {day.day_role for day in schedule.days} >= {"rest_day"}


def test_integrated_plan_spaces_taxing_work_and_allocates_weekly_mileage() -> None:
    base = _state(
        days_since_last_run=0.6,
        days_since_quality_run=20.0,
        days_since_long_run=20.0,
        running_days_28d=9,
        longest_run_30d_miles=5.9,
    )
    base = base.model_copy(
        update={
            "recent_load": base.recent_load.model_copy(
                update={
                    "capacity_reference_miles": 30.0,
                    "acute_distance_to_capacity_ratio": 0.4,
                }
            )
        }
    )
    states = [
        base.model_copy(
            update={
                "as_of": base.as_of + timedelta(days=offset),
                "days_since_last_run": 0.6 + offset,
                "days_since_quality_run": 20.0 + offset,
                "days_since_long_run": 20.0 + offset,
            }
        )
        # Production date selection owns one continuous 21-day lookahead. A
        # shorter fixture can create an artificial terminal cadence and no
        # longer exercises the integrated planner used by the app.
        for offset in range(21)
    ]

    schedule = build_weekly_schedule(
        states,
        RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
        CONFIG,
        target_run_count=3,
        target_distance_range=(15.5, 18.0),
    )

    planned = [
        (index, day.recommendation)
        for index, day in enumerate(schedule.days)
        if day.recommendation
    ]
    # The visible slice is not required to consume a weekly quota. The next
    # opportunity can fall just beyond day seven in the continuous plan.
    assert schedule.projected_distance_range_miles[0] >= 12.0
    assert schedule.projected_distance_range_miles[0] <= (
        schedule.target_distance_range_miles[1]
        + sum(typical_easy_distance(base)) / 2
    )
    adjacent_pairs = [
        (previous, current)
        for previous, current in zip(planned, planned[1:])
        if current[0] - previous[0] == 1
    ]
    # Consecutive days are legal but not a quota. Other tests cover cases
    # where recovery/load make a double the strongest plan; this fixture only
    # requires that any selected double avoid stacking taxing sessions.
    taxing_types = {
        WorkoutType.LONG,
        WorkoutType.INTERVALS,
        WorkoutType.TEMPO_THRESHOLD,
    }
    assert all(
        not (
            previous.workout_type in taxing_types
            and current.workout_type in taxing_types
        )
        for (_, previous), (_, current) in adjacent_pairs
    )
    taxing = [
        result for _, result in planned
        if result.workout_type
        in {WorkoutType.LONG, WorkoutType.INTERVALS, WorkoutType.TEMPO_THRESHOLD}
    ]
    assert 1 <= len(taxing) <= 2
    if len(taxing) == 2:
        recovery_trace = next(
            item
            for item in taxing[1].rule_trace
            if item.rule_id == "recent_recovery_load"
        )
        assert recovery_trace.facts["taxing_recovery_pressure"] == 0.0
    assert len({result.distance_range_miles for _, result in planned}) >= 2
    long_result = next(
        (
            result
            for _, result in planned
            if result.workout_type == WorkoutType.LONG
        ),
        None,
    )
    easy_results = [
        result for _, result in planned if result.workout_type == WorkoutType.EASY
    ]
    if long_result:
        assert all(
            easy.distance_range_miles[1] <= long_result.distance_range_miles[0]
            for easy in easy_results
        )


def test_yesterday_long_may_rest_or_run_easy_without_stacking_quality() -> None:
    base = _state(
        days_since_last_run=1.0,
        days_since_long_run=1.0,
        days_since_quality_run=8.0,
        last_run_workout_type=WorkoutType.LONG,
        last_run=_difficulty(miles=8.0, long=True),
        running_days_28d=14,
        longest_run_30d_miles=8.0,
    )
    states = [
        base.model_copy(
            update={
                "as_of": base.as_of + timedelta(days=offset),
                "days_since_last_run": 1.0 + offset,
                "days_since_long_run": 1.0 + offset,
                "days_since_quality_run": 8.0 + offset,
            }
        )
        for offset in range(21)
    ]

    schedule = build_weekly_schedule(
        states,
        RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
        CONFIG,
        target_run_count=4,
        target_distance_range=(18.0, 21.0),
    )
    planned = [
        (index, day.recommendation)
        for index, day in enumerate(schedule.days)
        if day.recommendation
    ]

    assert planned[0][0] <= 1
    if planned[0][1].workout_type in {
        WorkoutType.LONG,
        WorkoutType.INTERVALS,
        WorkoutType.TEMPO_THRESHOLD,
    }:
        recovery_trace = next(
            item
            for item in planned[0][1].rule_trace
            if item.rule_id == "recent_recovery_load"
        )
        assert recovery_trace.facts["taxing_recovery_pressure"] == 0.0
    first_quality_offset = next(
        index
        for index, result in planned
        if result.workout_type
        in {WorkoutType.INTERVALS, WorkoutType.TEMPO_THRESHOLD}
    )
    # The intervening easy run plus roughly 48 hours can make a mild quality
    # session viable; the load model, rather than a three-day calendar rule,
    # decides whether that transition is acceptable.
    assert first_quality_offset >= 1


def test_yesterday_quality_does_not_stack_another_taxing_run_today() -> None:
    base = _state(
        days_since_last_run=1.0,
        days_since_long_run=8.0,
        days_since_quality_run=1.0,
        last_run_workout_type=WorkoutType.TEMPO_THRESHOLD,
        last_run=_difficulty(miles=5.0, quality=True),
        running_days_28d=14,
        longest_run_30d_miles=8.0,
    )
    states = [
        base.model_copy(
            update={
                "as_of": base.as_of + timedelta(days=offset),
                "days_since_last_run": 1.0 + offset,
                "days_since_long_run": 8.0 + offset,
                "days_since_quality_run": 1.0 + offset,
            }
        )
        for offset in range(21)
    ]

    schedule = build_weekly_schedule(
        states,
        RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
        CONFIG,
        target_run_count=4,
        target_distance_range=(18.0, 21.0),
    )
    planned = [
        (index, day.recommendation)
        for index, day in enumerate(schedule.days)
        if day.recommendation
    ]

    assert not any(
        index == 0
        and result.workout_type
        in {
            WorkoutType.LONG,
            WorkoutType.INTERVALS,
            WorkoutType.TEMPO_THRESHOLD,
        }
        for index, result in planned
    )
    first_taxing_offset = next(
        index
        for index, result in planned
        if result.workout_type
        in {WorkoutType.LONG, WorkoutType.INTERVALS, WorkoutType.TEMPO_THRESHOLD}
    )
    assert first_taxing_offset >= 1


def test_mileage_funding_does_not_create_avoidable_multiday_streak() -> None:
    base = _state(
        days_since_last_run=1.0,
        days_since_long_run=8.0,
        days_since_quality_run=1.0,
        last_run_workout_type=WorkoutType.TEMPO_THRESHOLD,
        last_run=_difficulty(miles=5.0, quality=True),
        running_days_28d=14,
        longest_run_30d_miles=8.0,
    )
    states = [
        base.model_copy(
            update={
                "as_of": base.as_of + timedelta(days=offset),
                "days_since_last_run": 1.0 + offset,
                "days_since_long_run": 8.0 + offset,
                "days_since_quality_run": 1.0 + offset,
            }
        )
        for offset in range(21)
    ]

    schedule = build_weekly_schedule(
        states,
        RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
        CONFIG,
        target_run_count=4,
        target_distance_range=(18.0, 21.0),
    )
    offsets = [
        index
        for index, day in enumerate(schedule.days)
        if day.recommendation
    ]
    longest_streak = 0
    current_streak = 0
    previous = None
    for offset in offsets:
        current_streak = (
            current_streak + 1
            if previous is not None and offset == previous + 1
            else 1
        )
        longest_streak = max(longest_streak, current_streak)
        previous = offset

    # Intentional doubles remain legal. What must lose is an avoidable dense
    # block selected only because it fits the mileage path slightly better.
    assert longest_streak <= 2


def _role_days_for_allocation(base, roles):
    request = RecommendationRequest(health_status=CurrentHealthStatus.NORMAL)
    states = [
        base.model_copy(update={"as_of": base.as_of + timedelta(days=index)})
        for index in range(len(roles))
    ]
    days = []
    for index, role in enumerate(roles):
        result = recommend_next_run(
            states[index],
            request,
            CONFIG,
            weekly_role=role,
            allowed_candidates={role},
        )
        days.append(
            WeeklyScheduleDay(
                date=states[index].as_of.date(),
                planned_at=states[index].as_of,
                recommendation=result,
                day_role=f"{role}_run",
                rationale="Allocation fixture.",
            )
        )
    return states, days


def test_future_candidate_pressure_cannot_create_tiny_established_easy_run() -> None:
    base = _state(typical_easy_run_miles=4.0)
    states, days = _role_days_for_allocation(base, ["easy", "easy"])
    shortened = days[1].recommendation.model_copy(
        update={"distance_range_miles": (1.8, 2.2)}
    )
    days[1] = days[1].model_copy(update={"recommendation": shortened})

    allocated = _allocate_visible_distance_ranges(
        days,
        states,
        (7.5, 8.5),
        CONFIG,
    )

    second_midpoint = sum(
        allocated[1].recommendation.distance_range_miles
    ) / 2
    assert second_midpoint >= weekly_schedule._established_easy_midpoint_floor(
        base
    )


def test_future_first_session_does_not_inherit_same_day_recovery_exception() -> None:
    base = _state(typical_easy_run_miles=4.0)
    states, run_days = _role_days_for_allocation(base, ["easy", "easy"])
    shortened = run_days[0].recommendation.model_copy(
        update={"distance_range_miles": (1.8, 2.2)}
    )
    future = run_days[0].model_copy(
        update={
            "date": states[1].as_of.date(),
            "planned_at": states[1].as_of,
            "recommendation": shortened.model_copy(
                update={"planned_for": states[1].as_of}
            ),
        }
    )
    days = [
        WeeklyScheduleDay(
            date=states[0].as_of.date(),
            day_role="rest_day",
            rationale="Rest fixture.",
        ),
        future,
    ]

    allocated = _allocate_visible_distance_ranges(
        days,
        states,
        (3.5, 4.5),
        CONFIG,
    )

    midpoint = sum(allocated[1].recommendation.distance_range_miles) / 2
    assert midpoint >= weekly_schedule._established_easy_midpoint_floor(base)


def test_safe_meaningful_long_is_reserved_before_other_weekly_mileage() -> None:
    base = _state(
        longest_run_30d_miles=6.0,
        days_since_long_run=20.0,
        days_since_quality_run=20.0,
    )
    states, days = _role_days_for_allocation(
        base, ["long", "easy", "quality"]
    )

    allocated = _allocate_visible_distance_ranges(
        days, states, (9.0, 10.0), CONFIG
    )

    assert allocated[0].recommendation.workout_type == WorkoutType.LONG
    assert allocated[0].day_role == "long_run"
    assert (
        allocated[0].recommendation.distance_range_miles[0]
        > sum(typical_easy_distance(base)) / 2
    )


def test_full_horizon_preserves_each_supported_long_run() -> None:
    base = _state(
        longest_run_30d_miles=7.0,
        retained_long_run_capacity_miles=7.0,
        days_since_last_run=3.0,
        days_since_long_run=8.0,
        days_since_quality_run=8.0,
        typical_easy_run_miles=4.0,
    )
    states, days = _role_days_for_allocation(
        base, ["long", "easy", "quality", "easy", "long"]
    )
    prescribed_long_midpoints = [
        sum(day.recommendation.distance_range_miles) / 2
        for day in days
        if day.recommendation.workout_type == WorkoutType.LONG
    ]

    allocated = _allocate_visible_distance_ranges(
        days,
        states,
        (40.0, 45.0),
        CONFIG,
        weekly_target_range=(15.0, 18.0),
    )
    allocated_long_midpoints = [
        sum(day.recommendation.distance_range_miles) / 2
        for day in allocated
        if day.recommendation.workout_type == WorkoutType.LONG
    ]

    assert len(allocated_long_midpoints) == 2
    assert allocated_long_midpoints == prescribed_long_midpoints


def test_tight_week_preserves_quality_dose_and_shrinks_aerobic_mileage() -> None:
    base = _state(
        longest_run_30d_miles=6.0,
        days_since_long_run=20.0,
        days_since_quality_run=20.0,
    )
    states, days = _role_days_for_allocation(
        base, ["long", "quality", "easy", "easy"]
    )

    allocated = _allocate_visible_distance_ranges(
        days, states, (13.0, 14.0), CONFIG
    )

    quality = allocated[1].recommendation
    assert quality.workout_type in {
        WorkoutType.INTERVALS,
        WorkoutType.TEMPO_THRESHOLD,
    }
    assert sum(quality.distance_range_miles) / 2 == 4.5
    assert quality.distance_range_miles[1] - quality.distance_range_miles[0] >= 0.5
    assert sum(allocated[2].recommendation.distance_range_miles) / 2 < 4.0
    assert sum(allocated[3].recommendation.distance_range_miles) / 2 < 4.0


def test_dominated_allocation_pruning_preserves_exact_program() -> None:
    base = _state(
        longest_run_30d_miles=7.0,
        days_since_long_run=10.0,
        days_since_quality_run=8.0,
    )
    states, days = _role_days_for_allocation(
        base, ["long", "quality", "easy", "easy"]
    )

    optimized = _allocate_visible_distance_ranges(
        days, states, (15.0, 18.0), CONFIG
    )
    exhaustive = _allocate_visible_distance_ranges(
        days,
        states,
        (15.0, 18.0),
        CONFIG,
        _prune_dominated_allocations=False,
    )

    assert [day.model_dump() for day in optimized] == [
        day.model_dump() for day in exhaustive
    ]


def test_quality_keeps_its_full_dose_when_easy_mileage_can_absorb_budget() -> None:
    base = _state(
        longest_run_30d_miles=6.0,
        days_since_long_run=20.0,
        days_since_quality_run=20.0,
    )
    states, days = _role_days_for_allocation(
        base, ["long", "quality", "easy", "easy"]
    )
    prescribed_quality_midpoint = sum(
        days[1].recommendation.distance_range_miles
    ) / 2

    allocated = _allocate_visible_distance_ranges(
        days, states, (15.0, 16.0), CONFIG
    )

    quality = allocated[1].recommendation
    assert sum(quality.distance_range_miles) / 2 <= prescribed_quality_midpoint
    assert not any("instead of deleting" in reason for reason in quality.reasons)


def test_extra_recovery_lets_budget_pressure_use_more_long_run_headroom() -> None:
    allocated_longs = []
    for rest_days in (1.0, 3.0):
        base = _state(
            longest_run_30d_miles=6.0,
            retained_long_run_capacity_miles=6.0,
            days_since_last_run=rest_days,
            days_since_long_run=8.0,
            days_since_quality_run=2.0,
        )
        states, days = _role_days_for_allocation(base, ["long", "easy"])
        allocated = _allocate_visible_distance_ranges(
            days, states, (12.0, 13.0), CONFIG
        )
        allocated_longs.append(allocated[0].recommendation.distance_range_miles)

    assert sum(allocated_longs[1]) > sum(allocated_longs[0])
    assert sum(allocated_longs[1]) / 2 >= 6.5
    assert allocated_longs[1][1] - allocated_longs[1][0] >= 0.5


def test_spare_budget_does_not_make_long_run_guardrail_the_default() -> None:
    base = _state(
        longest_run_30d_miles=8.0,
        retained_long_run_capacity_miles=8.0,
        days_since_last_run=3.0,
        days_since_long_run=8.0,
        days_since_quality_run=8.0,
        typical_easy_run_miles=5.0,
        running_days_28d=12,
    )
    states, days = _role_days_for_allocation(
        base, ["long", "easy", "quality"]
    )

    allocated = _allocate_visible_distance_ranges(
        days,
        states,
        (22.0, 24.0),
        CONFIG,
        weekly_target_range=(22.0, 24.0),
    )
    long_midpoint = sum(
        allocated[0].recommendation.distance_range_miles
    ) / 2
    total_midpoint = sum(
        sum(day.recommendation.distance_range_miles) / 2
        for day in allocated
    )

    # This three-lane candidate is intentionally short of the whole-program
    # target. The outer optimizer can add a useful aerobic day; it must not
    # make the recovery-supported 10% long-run ceiling the default merely to
    # make this lower-frequency candidate balance arithmetically.
    assert long_midpoint == 8.5
    assert total_midpoint < 22.0


def test_joint_allocator_scales_long_distinction_for_higher_mileage_runner() -> None:
    base = _state(
        longest_run_30d_miles=12.0,
        days_since_long_run=20.0,
        days_since_quality_run=20.0,
        running_days_28d=20,
    )
    trailing_28d = base.recent_load.trailing_28d.model_copy(
        update={
            "distance_miles": 140.0,
            "moving_minutes": 1400.0,
            "zone_load": 2000.0,
            "activity_count": 20,
        }
    )
    base = base.model_copy(
        update={
            "recent_load": base.recent_load.model_copy(
                update={
                    "trailing_28d": trailing_28d,
                    "capacity_reference_miles": 40.0,
                    "acute_distance_to_capacity_ratio": 0.5,
                }
            )
        }
    )
    states, days = _role_days_for_allocation(
        base, ["long", "easy", "quality"]
    )

    allocated = _allocate_visible_distance_ranges(
        days, states, (28.0, 32.0), CONFIG
    )

    long_range = allocated[0].recommendation.distance_range_miles
    easy_range = allocated[1].recommendation.distance_range_miles
    quality_range = allocated[2].recommendation.distance_range_miles
    assert allocated[0].recommendation.workout_type == WorkoutType.LONG
    assert sum(long_range) / 2 >= (sum(easy_range) / 2) * 1.15
    assert quality_range[1] <= 5.0
    assert easy_range[1] > quality_range[1]


def test_weekly_shortfall_expands_easy_running_without_forcing_medium_long() -> None:
    base = _state(
        longest_run_30d_miles=9.0,
        retained_long_run_capacity_miles=9.0,
        days_since_last_run=3.0,
        days_since_long_run=8.0,
        days_since_quality_run=8.0,
        typical_easy_run_miles=4.0,
    )
    states, days = _role_days_for_allocation(
        base, ["long", "easy", "easy", "quality"]
    )

    allocated = _allocate_visible_distance_ranges(
        days, states, (22.0, 24.0), CONFIG
    )

    long_range = allocated[0].recommendation.distance_range_miles
    easy_ranges = [
        day.recommendation.distance_range_miles
        for day in allocated
        if day.recommendation.workout_type == WorkoutType.EASY
    ]
    assert long_range[1] <= 24.0 * 0.45
    assert all(distance[1] > 4.0 for distance in easy_ranges)
    medium_long_days = [
        day for day in allocated if day.day_role == "medium_long_run"
    ]
    assert len(medium_long_days) <= 1
    if medium_long_days:
        assert medium_long_days[0].recommendation.title == (
            "Medium-long aerobic run"
        )


def test_joint_finalists_preserve_delayed_and_low_cluster_calendars() -> None:
    scored = [
        (1.0, (0, 1, 3, 5, 7)),
        (1.1, (0, 2, 4, 6, 7)),
        (1.2, (1, 2, 4, 6, 8)),
        (1.3, (1, 3, 5, 7, 9)),
    ]

    finalists = _select_joint_finalists(scored)

    assert scored[0] in finalists
    assert any(offsets[0] == 1 for _, offsets in finalists)
    assert (1.3, (1, 3, 5, 7, 9)) in finalists


def test_selected_time_remains_visible_without_forecast_support() -> None:
    base = _state(running_days_28d=10)
    states = [
        base.model_copy(update={"as_of": base.as_of + timedelta(days=offset)})
        for offset in range(7)
    ]

    schedule = build_weekly_schedule(
        states,
        RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
        CONFIG,
        target_run_count=3,
    )

    planned_days = [day for day in schedule.days if day.recommendation]
    assert planned_days
    assert all(day.planned_at is not None for day in planned_days)


def test_fourteen_day_plan_can_use_more_short_runs_than_cadence_reference() -> None:
    base = _state(
        days_since_last_run=2.0,
        days_since_quality_run=20,
        days_since_long_run=20,
        running_days_28d=10,
    )
    states = [
        base.model_copy(
            update={
                "as_of": base.as_of + timedelta(days=offset),
                "days_since_last_run": 2.0 + offset,
                "days_since_quality_run": 20.0 + offset,
                "days_since_long_run": 20.0 + offset,
            }
        )
        for offset in range(14)
    ]

    schedule = build_weekly_schedule(
        states,
        RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
        CONFIG,
        target_run_count=3,
        target_distance_range=(9.0, 12.0),
    )

    assert len(schedule.days) == 7
    assert schedule.end_date == states[6].as_of.date()
    assert schedule.target_run_count == len(planned_days := [
        day for day in schedule.days if day.recommendation
    ])
    assert 2 <= len(planned_days) <= 4
    # A continuous horizon can place one more run inside this visible slice
    # than the adjacent slice. Do not alter the plan merely to satisfy the UI
    # boundary; bound the difference by one ordinary athlete-relative run.
    projected_low, projected_high = schedule.projected_distance_range_miles
    visible_midpoint = (projected_low + projected_high) / 2
    ordinary_run_midpoint = sum(typical_easy_distance(base)) / 2
    assert visible_midpoint <= (
        schedule.target_distance_range_miles[1] + ordinary_run_midpoint
    )
    visible_scheduled_midpoint = sum(
        sum(day.recommendation.distance_range_miles) / 2
        for day in schedule.days
        if day.recommendation and day.recommendation.distance_range_miles
    )
    planning_midpoint = sum(
        sum(day.recommendation.distance_range_miles) / 2
        for day in schedule.planning_days
        if day.recommendation and day.recommendation.distance_range_miles
    )
    assert schedule.visible_7d_scheduled_miles == pytest.approx(
        visible_scheduled_midpoint,
        abs=0.01,
    )
    assert schedule.planned_14d_weekly_rate == pytest.approx(
        planning_midpoint / 2,
        abs=0.01,
    )
    assert all(
        day.recommendation.distance_range_miles[1]
        - day.recommendation.distance_range_miles[0]
        >= 0.5
        for day in planned_days
    )
    assert all("accumulated load" in day.rationale for day in planned_days)


def test_adaptive_dates_move_away_from_a_higher_load_candidate() -> None:
    base = _state(
        days_since_last_run=2.0,
        days_since_quality_run=20,
        days_since_long_run=20,
        running_days_28d=10,
    )
    base_load = base.recent_load.model_copy(
        update={
            "capacity_reference_miles": 30.0,
            "acute_distance_to_capacity_ratio": 0.4,
        }
    )
    states = [
        base.model_copy(
            update={
                "as_of": base.as_of + timedelta(days=offset),
                "days_since_last_run": 2.0 + offset,
                "days_since_quality_run": 20.0 + offset,
                "days_since_long_run": 20.0 + offset,
                "recent_load": base_load,
            }
        )
        for offset in range(14)
    ]
    request = RecommendationRequest(health_status=CurrentHealthStatus.NORMAL)
    baseline = adaptive_run_day_offsets(
        states, request, CONFIG, 3, (9.0, 12.0)
    )
    penalized_offset = baseline[1]
    overloaded = base_load.model_copy(
        update={
            "capacity_reference_miles": 6.0,
            "acute_distance_to_capacity_ratio": 2.0,
            "trailing_7d": base_load.trailing_7d.model_copy(
                update={"distance_miles": 30.0}
            ),
        }
    )
    states[penalized_offset] = states[penalized_offset].model_copy(
        update={"recent_load": overloaded}
    )

    adjusted = adaptive_run_day_offsets(
        states, request, CONFIG, 3, (9.0, 12.0)
    )

    assert penalized_offset not in adjusted
    assert adjusted != baseline


def test_twenty_one_day_beam_ignores_supplied_run_count_target() -> None:
    base = _state(
        days_since_last_run=2.0,
        days_since_quality_run=20.0,
        days_since_long_run=20.0,
        running_days_28d=10,
    )
    base = base.model_copy(
        update={
            "recent_load": base.recent_load.model_copy(
                update={
                    # Established production plans always carry the retained
                    # distance denominator used by projected load. This test
                    # exercises frequency independence, not missing-baseline
                    # fallback behavior.
                    "capacity_reference_miles": 16.5,
                    "acute_distance_to_capacity_ratio": 12.0 / 16.5,
                }
            )
        }
    )
    states = [
        base.model_copy(
            update={
                "as_of": base.as_of + timedelta(days=offset),
                "days_since_last_run": 2.0 + offset,
                "days_since_quality_run": 20.0 + offset,
                "days_since_long_run": 20.0 + offset,
            }
        )
        for offset in range(21)
    ]
    request = RecommendationRequest(health_status=CurrentHealthStatus.NORMAL)

    low_count_input = adaptive_run_day_offsets(
        states, request, CONFIG, 1, (15.5, 18.0)
    )
    high_count_input = adaptive_run_day_offsets(
        states, request, CONFIG, 7, (15.5, 18.0)
    )

    assert low_count_input == high_count_input
    # Frequency emerges from mileage and recovery; neither supplied count
    # becomes a hard target or a terminal-horizon debt.
    assert low_count_input
    assert all(
        later > earlier
        for earlier, later in zip(low_count_input, low_count_input[1:])
    )


def test_twenty_one_day_beam_bounds_full_coaching_evaluations(
    monkeypatch,
) -> None:
    base = _state(days_since_last_run=2.0, running_days_28d=10)
    states = [
        base.model_copy(
            update={
                "as_of": base.as_of + timedelta(days=offset),
                "days_since_last_run": 2.0 + offset,
            }
        )
        for offset in range(21)
    ]
    calls = 0

    def count_cost(offsets, *args, **kwargs):
        nonlocal calls
        calls += 1
        return float(sum(offsets))

    monkeypatch.setattr(
        weekly_schedule,
        "_adaptive_candidate_cost",
        count_cost,
    )

    offsets = adaptive_run_day_offsets(
        states,
        RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
        CONFIG,
        3,
        (15.5, 18.0),
    )

    assert offsets
    # Each frequency evaluates the bounded beam, at most one earlier/later
    # prefix translation per finalist and prefix length, the final role-aware
    # shortlist, and the outer comparison. The work remains polynomial in the
    # 21-day horizon rather than approaching the raw combination count.
    maximum_calls_per_frequency = (
        weekly_schedule.MAX_ADAPTIVE_CANDIDATES
        + weekly_schedule.JOINT_DATE_FINALISTS
        * (2 * (len(states) - 1) + 1)
        + 1
    )
    assert calls <= (
        weekly_schedule.MAX_HORIZON_COUNT_OPTIONS
        * maximum_calls_per_frequency
    )


def test_count_search_includes_capacity_supported_fewer_run_plans(
    monkeypatch,
) -> None:
    base = _state(days_since_last_run=0.75, running_days_28d=9)
    base = base.model_copy(
        update={
            "recent_load": base.recent_load.model_copy(
                update={
                    "capacity_reference_miles": 16.4,
                    "acute_distance_to_capacity_ratio": 1.06,
                }
            )
        }
    )
    states = [
        base.model_copy(
            update={
                "as_of": base.as_of + timedelta(days=offset),
                "days_since_last_run": 0.75 + offset,
            }
        )
        for offset in range(21)
    ]
    evaluated_counts: list[int] = []

    def record_count(*args, horizon_run_count, **kwargs):
        evaluated_counts.append(horizon_run_count)
        return list(range(horizon_run_count))

    monkeypatch.setattr(
        weekly_schedule,
        "_adaptive_run_day_offsets_for_frequency",
        record_count,
    )
    monkeypatch.setattr(
        weekly_schedule,
        "_adaptive_candidate_cost",
        lambda *args, **kwargs: 0.0,
    )

    adaptive_run_day_offsets(
        states,
        RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
        CONFIG,
        3,
        (15.5, 18.0),
    )

    # The former central-minus-three floor started this current-like case at
    # twelve. Slightly longer but still useful sessions support ten or fewer.
    assert min(evaluated_counts) <= 10
    assert 10 in evaluated_counts
    assert len(evaluated_counts) <= weekly_schedule.MAX_HORIZON_COUNT_OPTIONS


@pytest.mark.parametrize(
    ("candidate_costs", "expected_count"),
    [
        ({10: 59.11, 11: 102.22}, 10),
        # A nearby but genuinely better higher-frequency program must not be
        # discarded by an implicit preference for fewer running days.
        ({10: 59.11, 11: 59.10}, 11),
    ],
)
def test_role_aware_distribution_uses_actual_best_program_cost(
    monkeypatch,
    candidate_costs: dict[int, float],
    expected_count: int,
) -> None:
    base = _state(
        days_since_last_run=0.75,
        days_since_quality_run=0.75,
        days_since_long_run=2.75,
        running_days_28d=9,
        longest_run_30d_miles=6.4,
    )
    base = base.model_copy(
        update={
            "recent_load": base.recent_load.model_copy(
                update={
                    "capacity_reference_miles": 16.4,
                    "acute_distance_to_capacity_ratio": 1.06,
                }
            )
        }
    )
    states = [
        base.model_copy(
            update={
                "as_of": base.as_of + timedelta(days=offset),
                "days_since_last_run": 0.75 + offset,
                "days_since_quality_run": 0.75 + offset,
                "days_since_long_run": 2.75 + offset,
            }
        )
        for offset in range(21)
    ]

    def evenly_spaced(*args, horizon_run_count, **kwargs):
        if horizon_run_count == 1:
            return [0]
        return [
            round(position * 20 / (horizon_run_count - 1))
            for position in range(horizon_run_count)
        ]

    def representative_coaching_cost(offsets, *args, **kwargs):
        # Exact-production diagnostic values for the formerly misranked pair.
        return candidate_costs.get(len(offsets), 1_000.0)

    monkeypatch.setattr(
        weekly_schedule,
        "_adaptive_run_day_offsets_for_frequency",
        evenly_spaced,
    )
    monkeypatch.setattr(
        weekly_schedule,
        "_adaptive_candidate_cost",
        representative_coaching_cost,
    )
    monkeypatch.setattr(
        weekly_schedule,
        "_joint_candidate_program_cost",
        lambda *args, **kwargs: (0.0, 0.0, 0.0, 0.0),
    )

    offsets = adaptive_run_day_offsets(
        states,
        RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
        CONFIG,
        3,
        (15.5, 18.0),
    )

    assert len(offsets) == expected_count


def test_high_mileage_adds_useful_days_before_oversizing_easy_runs() -> None:
    base = _state(
        days_since_last_run=2.0,
        days_since_quality_run=7.0,
        days_since_long_run=7.0,
        running_days_28d=24,
        longest_run_30d_miles=10.0,
        typical_easy_run_miles=4.5,
    )
    load = base.recent_load.model_copy(
        update={
            "trailing_7d": base.recent_load.trailing_7d.model_copy(
                update={
                    "distance_miles": 34.0,
                    "moving_minutes": 340.0,
                    "activity_count": 6,
                }
            ),
            "trailing_14d": base.recent_load.trailing_14d.model_copy(
                update={
                    "distance_miles": 68.0,
                    "moving_minutes": 680.0,
                    "activity_count": 12,
                }
            ),
            "trailing_28d": base.recent_load.trailing_28d.model_copy(
                update={
                    "distance_miles": 136.0,
                    "moving_minutes": 1360.0,
                    "activity_count": 24,
                }
            ),
            "capacity_reference_miles": 34.0,
            "sustained_capacity_miles": 34.0,
            "acute_distance_to_capacity_ratio": 1.0,
        }
    )
    base = base.model_copy(update={"recent_load": load})
    states = [
        base.model_copy(
            update={
                "as_of": base.as_of + timedelta(days=offset),
                "days_since_last_run": 2.0 + offset,
                "days_since_quality_run": 7.0 + offset,
                "days_since_long_run": 7.0 + offset,
            }
        )
        for offset in range(21)
    ]

    offsets = adaptive_run_day_offsets(
        states,
        RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
        CONFIG,
        6,
        (32.5, 34.0),
    )

    # Four weekly runs would require ordinary easy days well beyond the
    # athlete-relative expansion reference. Frequency emerges from useful
    # load distribution rather than the ignored supplied count of six.
    assert len(offsets) >= 13
    assert _ordinary_easy_expansion_reference(base) == 5.0


def test_twenty_one_day_sick_limit_does_not_empty_the_search() -> None:
    base = _state(days_since_last_run=3.0, running_days_28d=10)
    states = [
        base.model_copy(
            update={
                "as_of": base.as_of + timedelta(days=offset),
                "days_since_last_run": 3.0 + offset,
            }
        )
        for offset in range(21)
    ]

    offsets = adaptive_run_day_offsets(
        states,
        RecommendationRequest(
            health_status=CurrentHealthStatus.SICK_OR_RECOVERING
        ),
        CONFIG,
        7,
        (30.0, 40.0),
    )

    assert 1 <= len(offsets) <= 6


def test_exact_frequency_enumerator_has_no_seven_day_block_quota() -> None:
    base = _state(days_since_last_run=3.0, running_days_28d=10)
    states = [
        base.model_copy(
            update={
                "as_of": base.as_of + timedelta(days=offset),
                "days_since_last_run": 3.0 + offset,
            }
        )
        for offset in range(14)
    ]

    offsets = weekly_schedule._adaptive_run_day_offsets_for_frequency(
        states,
        RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
        CONFIG,
        3,
        (9.5, 11.0),
        forced_rest_offsets={1, 2, 3, 4, 5},
    )

    assert len(offsets) == 6
    assert offsets[0] == 0
    assert sum(offset < 7 for offset in offsets) == 2


def test_elapsed_gap_cost_crosses_display_boundary_only_while_underfunded() -> None:
    base = _state(
        days_since_last_run=3.0,
        days_since_quality_run=None,
        days_since_long_run=None,
        running_days_28d=10,
    )
    states = [
        base.model_copy(
            update={
                "as_of": base.as_of + timedelta(days=offset),
                "days_since_last_run": 3.0 + offset,
            }
        )
        for offset in range(21)
    ]
    state_options = [[state] for state in states]
    request = RecommendationRequest(health_status=CurrentHealthStatus.NORMAL)

    evenly_spaced = weekly_schedule._adaptive_candidate_cost(
        (0, 3, 6, 9, 12, 15, 18),
        states,
        state_options,
        request,
        CONFIG,
        (20.0, 22.0),
        role_loop_penalty=False,
        include_mileage_path=False,
        include_recovery_interactions=False,
    )
    boundary_hole = weekly_schedule._adaptive_candidate_cost(
        (0, 3, 6, 12, 15, 18, 20),
        states,
        state_options,
        request,
        CONFIG,
        (20.0, 22.0),
        role_loop_penalty=False,
        include_mileage_path=False,
        include_recovery_interactions=False,
    )

    assert boundary_hole > evenly_spaced


def test_cadence_idle_cost_uses_elapsed_hours_not_reporting_days() -> None:
    start = datetime(2026, 9, 2, 19, tzinfo=timezone.utc)

    assert weekly_schedule._elapsed_cadence_idle_cost(
        [start, start + timedelta(hours=24), start + timedelta(hours=72)],
        48.0,
    ) == 0.0
    assert weekly_schedule._elapsed_cadence_idle_cost(
        [start, start + timedelta(hours=72)],
        48.0,
    ) == pytest.approx(5.0)


def test_cadence_pressure_disappears_once_full_horizon_load_is_funded() -> None:
    """Cadence is an underfill aid, not an implicit frequency target."""

    assert weekly_schedule._cadence_underfill_pressure(
        51.0,
        (17.0, 18.0),
        21,
        4.0,
    ) == 0.0
    assert weekly_schedule._cadence_underfill_pressure(
        49.0,
        (17.0, 18.0),
        21,
        4.0,
    ) == pytest.approx(0.5)


def test_adaptive_dates_do_not_game_recovery_guardrail_for_lower_mileage() -> None:
    base = _state(
        days_since_last_run=2.0,
        days_since_quality_run=20,
        days_since_long_run=20,
        running_days_28d=10,
    )
    manageable_load = base.recent_load.model_copy(
        update={
            "capacity_reference_miles": 30.0,
            "acute_distance_to_capacity_ratio": 0.4,
        }
    )
    states = [
        base.model_copy(
            update={
                "as_of": base.as_of + timedelta(days=offset),
                "days_since_last_run": 2.0 + offset,
                "days_since_quality_run": 20.0 + offset,
                "days_since_long_run": 20.0 + offset,
                "recent_load": manageable_load,
            }
        )
        for offset in range(14)
    ]

    schedule = build_weekly_schedule(
        states,
        RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
        CONFIG,
        target_run_count=3,
        target_distance_range=(9.0, 12.0),
    )

    planned = [day.recommendation for day in schedule.days if day.recommendation]
    assert planned
    assert all(item.workout_type != WorkoutType.RECOVERY for item in planned)


def test_sick_or_recovering_week_limits_frequency_and_load() -> None:
    base = _state()
    states = [base.model_copy(update={"as_of": base.as_of + timedelta(days=offset)}) for offset in range(7)]
    schedule = build_weekly_schedule(
        states,
        RecommendationRequest(health_status=CurrentHealthStatus.SICK_OR_RECOVERING),
        CONFIG,
    )
    assert schedule.run_count <= 2
    assert schedule.run_count > 0
    recommendations = [day.recommendation for day in schedule.days if day.recommendation]
    assert all(item.workout_type == WorkoutType.RECOVERY for item in recommendations)
    assert all(item.distance_range_miles[1] <= 3 for item in recommendations)


def test_completed_run_today_replaces_prescription_and_reduces_remaining_plan() -> None:
    base = _state(running_days_28d=10)
    states = [base.model_copy(update={"as_of": base.as_of + timedelta(days=offset)}) for offset in range(7)]
    completed = TrailingDayActivity(
        activity_id=99,
        start_time=base.as_of,
        distance_miles=4.0,
        workout_type=WorkoutType.EASY,
        health_tag=ActivityHealthTag.NORMAL,
    )
    schedule = build_weekly_schedule(
        states,
        RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
        CONFIG,
        target_run_count=4,
        completed_activities_by_offset={0: [completed]},
    )
    assert schedule.start_date == base.as_of.date()
    assert schedule.completed_run_count == 1
    assert schedule.target_run_count == 4
    assert schedule.run_count == 3
    assert schedule.days[0].day_role == "completed_run"
    assert schedule.days[0].recommendation is None
    assert schedule.days[0].completed_activities[0].activity_id == 99


def test_completed_run_and_forced_rest_still_use_adaptive_plan() -> None:
    base = _state(days_since_last_run=0.0, running_days_28d=10)
    states = [
        base.model_copy(
            update={
                "as_of": base.as_of + timedelta(days=offset),
                "days_since_last_run": float(offset),
            }
        )
        for offset in range(14)
    ]
    completed = TrailingDayActivity(
        activity_id=100,
        start_time=base.as_of,
        distance_miles=3.0,
        workout_type=WorkoutType.EASY,
        health_tag=ActivityHealthTag.NORMAL,
    )
    initial = build_weekly_schedule(
        states,
        RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
        CONFIG,
        target_run_count=3,
        target_distance_range=(9.5, 11.0),
        completed_activities_by_offset={0: [completed]},
    )
    displaced_offset = next(
        index for index, day in enumerate(initial.days) if day.recommendation
    )

    replanned = build_weekly_schedule(
        states,
        RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
        CONFIG,
        target_run_count=3,
        target_distance_range=(9.5, 11.0),
        completed_activities_by_offset={0: [completed]},
        forced_rest_offsets={displaced_offset},
    )

    assert replanned.target_run_count == (
        replanned.completed_run_count + replanned.run_count
    )
    assert replanned.completed_run_count == 1
    assert 1 <= replanned.run_count <= 3
    projected_midpoint = sum(replanned.projected_distance_range_miles) / 2
    target_midpoint = sum(replanned.target_distance_range_miles) / 2
    assert projected_midpoint <= (
        target_midpoint + sum(typical_easy_distance(base)) / 2
    )
    assert replanned.days[displaced_offset].forced_rest is True
    assert replanned.projected_distance_range_miles[0] >= 9.5


def test_many_forced_rest_days_preserve_target_and_explain_shortfall() -> None:
    base = _state(days_since_last_run=3.0, running_days_28d=10)
    states = [
        base.model_copy(
            update={
                "as_of": base.as_of + timedelta(days=offset),
                "days_since_last_run": 3.0 + offset,
            }
        )
        for offset in range(14)
    ]

    schedule = build_weekly_schedule(
        states,
        RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
        CONFIG,
        target_run_count=3,
        target_distance_range=(9.5, 11.0),
        forced_rest_offsets={0, 1, 2, 3, 4, 5},
    )

    assert schedule.target_run_count == 1
    assert schedule.run_count == 1
    assert "rest-day constraints" in schedule.summary


def test_sequence_preferences_remain_secondary_to_workout_evidence() -> None:
    base = _state(
        days_since_last_run=2.0,
        days_since_quality_run=20,
        days_since_long_run=20,
        recent_illness_or_recovery=True,
        normal_runs_since_health_event=1,
        running_days_28d=10,
    )
    base = base.model_copy(
        update={
            "recent_load": base.recent_load.model_copy(
                update={
                    "acute_distance_to_capacity_ratio": 0.9,
                    "capacity_reference_miles": 20.0,
                    "sustained_capacity_miles": 20.0,
                    "prior_28d_weekly_miles": 9.0,
                }
            )
        }
    )
    states = [
        base.model_copy(
            update={
                "as_of": base.as_of + timedelta(days=offset),
                "days_since_last_run": 2.0 + offset,
                "days_since_quality_run": 20.0 + offset,
                "days_since_long_run": 20.0 + offset,
            }
        )
        for offset in range(7)
    ]
    schedule = build_weekly_schedule(
        states,
        RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
        CONFIG,
        target_run_count=4,
        target_distance_range=(15.5, 18.0),
    )
    recommendations = [
        day.recommendation for day in schedule.days if day.recommendation
    ]
    assert recommendations[0].workout_type == WorkoutType.LONG
    initial_scoring = next(
        item
        for item in recommendations[0].rule_trace
        if item.rule_id == "workout_scoring"
    )
    assert initial_scoring.facts["selected"] == "long"
    first_sequence = next(
        item
        for item in recommendations[0].rule_trace
        if item.rule_id == "weekly_sequence_priority"
    )
    assert first_sequence.facts["preferred_role"] == "long"
    assert first_sequence.facts["preference_points"] == 1.0
    taxing = [
        item for item in recommendations
        if item.workout_type
        in {WorkoutType.LONG, WorkoutType.INTERVALS, WorkoutType.TEMPO_THRESHOLD}
    ]
    assert len(taxing) <= 2
    if len(taxing) == 2:
        assert taxing[0].planned_for is not None
        assert taxing[1].planned_for is not None
        assert taxing[1].planned_for - taxing[0].planned_for >= timedelta(hours=36)


def test_completed_run_caution_does_not_repeat_across_projected_week() -> None:
    base = _state(
        days_since_last_run=2.0,
        days_since_quality_run=20,
        days_since_long_run=20,
        last_run_drift_percent=7.0,
        running_days_28d=10,
    )
    states = [
        base.model_copy(
            update={
                "as_of": base.as_of + timedelta(days=offset),
                "days_since_last_run": 2.0 + offset,
                "days_since_quality_run": 20.0 + offset,
                "days_since_long_run": 20.0 + offset,
            }
        )
        for offset in range(7)
    ]

    schedule = build_weekly_schedule(
        states,
        RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
        CONFIG,
        target_run_count=3,
        target_distance_range=(9.0, 11.0),
    )

    recommendations = [
        day.recommendation for day in schedule.days if day.recommendation
    ]
    assert recommendations[0].workout_type != WorkoutType.INTERVALS
    initial_drift_trace = next(
        item
        for item in recommendations[0].rule_trace
        if item.rule_id == "recent_drift_caution"
    )
    assert initial_drift_trace.fired is True
    assert 0 < initial_drift_trace.facts["volume_reduction_fraction"] < 0.30
    later_costly_trace = next(
        item
        for item in recommendations[1].rule_trace
        if item.rule_id == "recent_costly_response"
    )
    later_drift_trace = next(
        item
        for item in recommendations[1].rule_trace
        if item.rule_id == "recent_drift_caution"
    )
    assert later_costly_trace.fired is False
    assert later_costly_trace.facts["drift_percent"] is None
    assert later_drift_trace.fired is False


def test_planned_aerobic_run_clears_meaningful_moderate_leakage_for_later_days() -> None:
    base = _state(
        days_since_last_run=2.0,
        days_since_quality_run=20,
        days_since_long_run=20,
        moderate_fraction_14d=0.20,
        running_days_28d=10,
    )
    base = base.model_copy(
        update={
            "recent_load": base.recent_load.model_copy(
                update={
                    "capacity_reference_miles": 30.0,
                    "acute_distance_to_capacity_ratio": 0.4,
                }
            )
        }
    )
    states = [
        base.model_copy(
            update={
                "as_of": base.as_of + timedelta(days=offset),
                "days_since_last_run": 2.0 + offset,
                "days_since_quality_run": 20.0 + offset,
                "days_since_long_run": 20.0 + offset,
            }
        )
        for offset in range(7)
    ]

    schedule = build_weekly_schedule(
        states,
        RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
        CONFIG,
        target_run_count=3,
        target_distance_range=(9.0, 12.0),
    )

    recommendations = [
        day.recommendation for day in schedule.days if day.recommendation
    ]
    assert recommendations[0].workout_type in {
        WorkoutType.EASY,
        WorkoutType.LONG,
    }
    assert len({item.workout_type for item in recommendations}) > 1
    later_leakage = next(
        item
        for item in recommendations[1].rule_trace
        if item.rule_id == "moderate_leakage"
    )
    assert later_leakage.fired is False


def test_low_load_week_does_not_cluster_three_rest_days_then_taxing_runs() -> None:
    base = _state(
        days_since_last_run=0.95,
        days_since_quality_run=44.0,
        days_since_long_run=46.0,
        moderate_fraction_14d=0.175,
        running_days_28d=9,
    )
    base = base.model_copy(
        update={
            "recent_load": base.recent_load.model_copy(
                update={
                    "capacity_reference_miles": 14.3,
                    "acute_distance_to_capacity_ratio": 0.58,
                }
            )
        }
    )
    states = [
        base.model_copy(
            update={
                "as_of": base.as_of + timedelta(days=offset),
                "days_since_last_run": 0.95 + offset,
                "days_since_quality_run": 44.0 + offset,
                "days_since_long_run": 46.0 + offset,
            }
        )
        for offset in range(14)
    ]

    schedule = build_weekly_schedule(
        states,
        RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
        CONFIG,
        target_run_count=3,
        target_distance_range=(9.5, 11.0),
    )

    planned_offsets = [
        index for index, day in enumerate(schedule.days) if day.recommendation
    ]
    # The opening load is already above this plan's target high, and both
    # overdue key sessions carry real recovery cost. Do not impose a separate
    # maximum-rest-gap rule; mileage-path and recovery accounting own spacing.
    assert planned_offsets[0] <= 2
    # Both long and quality work are substantially overdue. Their minimum
    # useful doses may exceed the nominal range without being hard-clipped,
    # but should remain inside demonstrated capacity.
    assert schedule.projected_distance_range_miles[1] <= 14.3
    taxing = [
        day.recommendation
        for day in schedule.days
        if day.recommendation
        and day.recommendation.workout_type
        in {WorkoutType.LONG, WorkoutType.INTERVALS, WorkoutType.TEMPO_THRESHOLD}
    ]
    assert len(taxing) <= 2
    if len(taxing) == 2:
        assert taxing[0].planned_for is not None
        assert taxing[1].planned_for is not None
        assert (
            taxing[1].planned_for.date() - taxing[0].planned_for.date()
        ).days >= 2


def test_recovered_quality_can_follow_recent_completed_long_run() -> None:
    base = _state(
        days_since_last_run=0.65,
        days_since_quality_run=44.0,
        days_since_long_run=0.65,
        last_run=_difficulty(long=True, miles=6.4),
        last_run_workout_type=WorkoutType.LONG,
        recent_performance_response="stronger_than_recent",
        running_days_28d=9,
    )
    base = base.model_copy(
        update={
            "recent_load": base.recent_load.model_copy(
                update={
                    "capacity_reference_miles": 16.0,
                    "acute_distance_to_capacity_ratio": 1.05,
                }
            )
        }
    )
    states = [
        base.model_copy(
            update={
                "as_of": base.as_of + timedelta(days=offset),
                "days_since_last_run": 0.65 + offset,
                "days_since_quality_run": 44.0 + offset,
                "days_since_long_run": 0.65 + offset,
            }
        )
        for offset in range(14)
    ]

    schedule = build_weekly_schedule(
        states,
        RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
        CONFIG,
        target_run_count=3,
        target_distance_range=(15.5, 18.0),
    )

    planned = [
        (index, day.recommendation)
        for index, day in enumerate(schedule.days)
        if day.recommendation
    ]
    assert planned[0][0] <= 2
    quality_offset, quality = next(
        (offset, recommendation)
        for offset, recommendation in planned
        if recommendation.workout_type
        in {WorkoutType.INTERVALS, WorkoutType.TEMPO_THRESHOLD}
    )
    assert quality_offset <= 2
    assert all(
        current - previous <= 3
        for (previous, _), (current, _) in zip(planned, planned[1:])
    )


def test_normal_easy_yesterday_does_not_force_overdue_long_today() -> None:
    base = _state(
        as_of=datetime(2026, 9, 5, 7, tzinfo=timezone.utc),
        days_since_last_run=1.0,
        days_since_quality_run=5.0,
        days_since_long_run=8.0,
        last_run=_difficulty(miles=4.0),
        last_run_workout_type=WorkoutType.EASY,
        longest_run_30d_miles=6.5,
        retained_long_run_capacity_miles=8.0,
        running_days_28d=12,
    )
    base = base.model_copy(
        update={
            "recent_load": base.recent_load.model_copy(
                update={
                    "capacity_reference_miles": 18.0,
                    "sustained_capacity_miles": 18.0,
                    "acute_distance_to_capacity_ratio": 0.9,
                }
            )
        }
    )
    states = [
        base.model_copy(
            update={
                "as_of": base.as_of + timedelta(days=offset),
                "days_since_last_run": 1.0 + offset,
                "days_since_quality_run": 5.0 + offset,
                "days_since_long_run": 8.0 + offset,
            }
        )
        for offset in range(14)
    ]

    schedule = build_weekly_schedule(
        states,
        RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
        CONFIG,
        target_run_count=4,
        target_distance_range=(15.5, 18.0),
    )

    today = schedule.days[0].recommendation
    assert today is None or today.workout_type != WorkoutType.LONG
    long_offsets = [
        offset
        for offset, day in enumerate(schedule.days)
        if day.recommendation
        and day.recommendation.workout_type == WorkoutType.LONG
    ]
    assert long_offsets and long_offsets[0] in {1, 2}


def test_material_run_yesterday_leaves_today_for_full_horizon_comparison(
    monkeypatch,
) -> None:
    base = _state(
        as_of=datetime(2026, 8, 29, 19, tzinfo=timezone.utc),
        days_since_last_run=1.0,
        days_since_quality_run=44.0,
        days_since_long_run=1.0,
        last_run=_difficulty(long=True, miles=6.4),
        last_run_workout_type=WorkoutType.LONG,
        running_days_28d=9,
    )
    states = [
        base.model_copy(
            update={
                "as_of": base.as_of + timedelta(days=offset),
                "days_since_last_run": 1.0 + offset,
                "days_since_quality_run": 44.0 + offset,
                "days_since_long_run": 1.0 + offset,
            }
        )
        for offset in range(14)
    ]

    def prefer_recovery_day(offsets, *args, **kwargs):
        return 100.0 if 0 in offsets else 0.0

    monkeypatch.setattr(
        weekly_schedule,
        "_adaptive_candidate_cost",
        prefer_recovery_day,
    )
    offsets = weekly_schedule._adaptive_run_day_offsets_for_frequency(
        states,
        RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
        CONFIG,
        3,
        (15.5, 18.0),
    )

    assert offsets
    assert 0 not in offsets


def test_ready_today_still_competes_with_rest_in_full_horizon_comparison(
    monkeypatch,
) -> None:
    base = _state(
        as_of=datetime(2026, 8, 29, 19, tzinfo=timezone.utc),
        days_since_last_run=2.0,
        days_since_quality_run=8.0,
        days_since_long_run=5.0,
        last_run=_difficulty(miles=4.0),
        last_run_workout_type=WorkoutType.EASY,
        running_days_28d=10,
    )
    states = [
        base.model_copy(
            update={
                "as_of": base.as_of + timedelta(days=offset),
                "days_since_last_run": 2.0 + offset,
                "days_since_quality_run": 8.0 + offset,
                "days_since_long_run": 5.0 + offset,
            }
        )
        for offset in range(14)
    ]

    def prefer_rest_today(offsets, *args, **kwargs):
        return 100.0 if 0 in offsets else 0.0

    monkeypatch.setattr(
        weekly_schedule,
        "_adaptive_candidate_cost",
        prefer_rest_today,
    )
    offsets = weekly_schedule._adaptive_run_day_offsets_for_frequency(
        states,
        RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
        CONFIG,
        3,
        (15.5, 18.0),
    )

    assert offsets
    assert 0 not in offsets


def test_weekly_allocator_does_not_restore_recovery_scaled_easy_mileage() -> None:
    base = _state(
        as_of=datetime(2026, 8, 29, 19, tzinfo=timezone.utc),
        days_since_last_run=1.0,
        days_since_quality_run=44.0,
        days_since_long_run=1.0,
        last_run=_difficulty(long=True, miles=6.4),
        last_run_workout_type=WorkoutType.LONG,
        running_days_28d=9,
    )
    states = [
        base.model_copy(
            update={
                "as_of": base.as_of + timedelta(days=offset),
                "days_since_last_run": 1.0 + offset,
                "days_since_quality_run": 44.0 + offset,
                "days_since_long_run": 1.0 + offset,
            }
        )
        for offset in range(7)
    ]
    standalone = recommend_next_run(
        base,
        RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
        CONFIG,
        weekly_role="easy",
    )

    schedule = build_weekly_schedule(
        states,
        RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
        CONFIG,
        target_run_count=3,
        target_distance_range=(15.5, 18.0),
    )

    today = schedule.days[0].recommendation
    assert today is not None and today.workout_type == WorkoutType.EASY
    assert sum(today.distance_range_miles) / 2 <= sum(
        standalone.distance_range_miles
    ) / 2


def test_low_load_with_overdue_roles_keeps_the_next_run_from_drifting_late() -> None:
    start = datetime(2026, 8, 28, 19, tzinfo=timezone.utc)
    base = _state(
        as_of=start,
        days_since_last_run=1.97,
        days_since_quality_run=47.0,
        days_since_long_run=49.0,
        quality_sessions_14d=0,
        moderate_fraction_14d=0.18,
        running_days_28d=9,
        longest_run_30d_miles=5.9,
    )
    base = base.model_copy(
        update={
            "recent_load": base.recent_load.model_copy(
                update={
                    "capacity_reference_miles": 14.0,
                    "acute_distance_to_capacity_ratio": 0.75,
                }
            )
        }
    )
    states = [
        base.model_copy(
            update={
                "as_of": start + timedelta(days=offset),
                "days_since_last_run": 1.97 + offset,
                "days_since_quality_run": 47.0 + offset,
                "days_since_long_run": 49.0 + offset,
            }
        )
        for offset in range(14)
    ]

    schedule = build_weekly_schedule(
        states,
        RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
        CONFIG,
        target_run_count=3,
        target_distance_range=(15.5, 18.0),
    )
    planned = [
        (index, day.recommendation)
        for index, day in enumerate(schedule.days)
        if day.recommendation
    ]

    assert planned[0][0] == 0
    # Long and quality remain separated. Consecutive easy days are legal, but
    # this overdue-role fixture must not require one as a calendar quota.
    taxing_positions = [
        index
        for index, result in planned
        if result.workout_type
        in {WorkoutType.LONG, WorkoutType.INTERVALS, WorkoutType.TEMPO_THRESHOLD}
    ]
    assert all(
        current - previous >= 2
        for previous, current in zip(taxing_positions, taxing_positions[1:])
    )
    # Aerobic support belongs to the continuous plan, not necessarily the
    # first seven-day presentation slice.
    assert any(
        day.recommendation
        and day.recommendation.workout_type == WorkoutType.EASY
        for day in schedule.planning_days
    )


def test_forced_rest_date_is_excluded_and_week_is_replanned_around_it() -> None:
    base = _state(
        days_since_last_run=3.0,
        running_days_28d=10,
    )
    states = [
        base.model_copy(
            update={
                "as_of": base.as_of + timedelta(days=offset),
                "days_since_last_run": 3.0 + offset,
            }
        )
        for offset in range(14)
    ]
    request = RecommendationRequest(health_status=CurrentHealthStatus.NORMAL)
    baseline = build_weekly_schedule(
        states,
        request,
        CONFIG,
        target_run_count=3,
        target_distance_range=(9.5, 11.0),
    )
    baseline_offsets = [
        index for index, day in enumerate(baseline.days) if day.recommendation
    ]
    forced_offset = baseline_offsets[1]

    replanned = build_weekly_schedule(
        states,
        request,
        CONFIG,
        target_run_count=3,
        target_distance_range=(9.5, 11.0),
        forced_rest_offsets={forced_offset},
    )

    replanned_offsets = [
        index for index, day in enumerate(replanned.days) if day.recommendation
    ]
    assert replanned.days[forced_offset].forced_rest is True
    assert replanned.days[forced_offset].day_role == "forced_rest_day"
    assert replanned.days[forced_offset].recommendation is None
    assert forced_offset not in replanned_offsets
    # A run may move outside the seven-day display after the full continuous
    # horizon is rebuilt; preserving the visible count would reintroduce the
    # display-boundary bug.
    assert replanned_offsets
    assert replanned_offsets != baseline_offsets


def test_forcing_first_planned_day_rebalances_remaining_week() -> None:
    base = _state(
        days_since_last_run=2.0,
        days_since_quality_run=20.0,
        days_since_long_run=20.0,
        running_days_28d=10,
    )
    base = base.model_copy(
        update={
            "recent_load": base.recent_load.model_copy(
                update={
                    "capacity_reference_miles": 30.0,
                    "acute_distance_to_capacity_ratio": 0.4,
                }
            )
        }
    )
    states = [
        base.model_copy(
            update={
                "as_of": base.as_of + timedelta(days=offset),
                "days_since_last_run": 2.0 + offset,
                "days_since_quality_run": 20.0 + offset,
                "days_since_long_run": 20.0 + offset,
            }
        )
        for offset in range(14)
    ]
    request = RecommendationRequest(health_status=CurrentHealthStatus.NORMAL)
    baseline = build_weekly_schedule(
        states,
        request,
        CONFIG,
        target_run_count=3,
        target_distance_range=(9.0, 12.0),
    )
    baseline_offsets = [
        index for index, day in enumerate(baseline.days) if day.recommendation
    ]

    replanned = build_weekly_schedule(
        states,
        request,
        CONFIG,
        target_run_count=3,
        target_distance_range=(9.0, 12.0),
        forced_rest_offsets={baseline_offsets[0]},
    )

    replanned_offsets = [
        index for index, day in enumerate(replanned.days) if day.recommendation
    ]
    assert replanned.days[baseline_offsets[0]].forced_rest is True
    # A run may move outside the seven-day display after the full continuous
    # horizon is rebuilt; preserving the visible count would reintroduce the
    # display-boundary bug.
    assert replanned_offsets
    assert replanned_offsets != baseline_offsets
    # The whole horizon is rescored. It may independently retain later dates
    # when they remain the strongest choices; the constrained date itself must
    # disappear rather than being copied into a one-day substitution rule.
    assert baseline_offsets[0] not in replanned_offsets


def test_frequency_selection_prices_real_workouts_before_adding_run_days() -> None:
    """A low evidence floor must not make six normal workouts look cheap."""

    base = _state(
        days_since_last_run=1.92,
        days_since_quality_run=1.92,
        days_since_long_run=3.93,
        running_days_28d=8,
        longest_run_30d_miles=6.406,
        retained_long_run_capacity_miles=8.343,
        typical_easy_run_miles=3.917,
        last_run=_difficulty(quality=True, miles=4.407),
        last_run_workout_type=WorkoutType.TEMPO_THRESHOLD,
        quality_sessions_14d=1,
        completed_quality_session_count=3,
        easy_fraction_14d=0.7723,
        moderate_fraction_14d=0.1643,
        moderate_evidence_runs_14d=5,
        hard_fraction_14d=0.0641,
    )
    recent_load = base.recent_load.model_copy(
        update={
            "trailing_7d": LoadWindow(
                days=7, distance_miles=14.758, moving_minutes=154.03,
                zone_load=334.98, hard_minutes=15.08, activity_count=3,
                zone_load_activity_count=3,
            ),
            "trailing_14d": LoadWindow(
                days=14, distance_miles=23.13, moving_minutes=244.17,
                zone_load=509.98, hard_minutes=15.08, activity_count=6,
                zone_load_activity_count=6,
            ),
            "trailing_28d": LoadWindow(
                days=28, distance_miles=28.67, moving_minutes=303.95,
                zone_load=615.7, hard_minutes=15.08, activity_count=8,
                zone_load_activity_count=8,
            ),
            "acute_to_prior_ratio": 1.98,
            "acute_distance_to_capacity_ratio": 14.758 / 16.366,
            "capacity_reference_miles": 16.366,
            "sustained_capacity_miles": 16.366,
        }
    )
    base = base.model_copy(update={"recent_load": recent_load})
    states = [
        base.model_copy(
            update={
                "as_of": base.as_of + timedelta(days=offset),
                "days_since_last_run": 1.92 + offset,
                "days_since_quality_run": 1.92 + offset,
                "days_since_long_run": 3.93 + offset,
            }
        )
        for offset in range(21)
    ]

    schedule = build_weekly_schedule(
        states,
        RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
        CONFIG,
        target_run_count=3,
        target_distance_range=(16.8, 17.9),
    )

    planned_offsets = [
        index for index, day in enumerate(schedule.days) if day.recommendation
    ]
    assert schedule.run_count <= 4
    # A rolling seven-day view can include one additional ordinary session at
    # a boundary; the continuous 21-day plan, not the UI week, owns the load
    # budget. It still must not buy a second whole workout of overflow.
    assert schedule.projected_distance_range_miles[1] <= 17.9 + 3.917
    assert all(
        third - first > 2
        for first, third in zip(planned_offsets, planned_offsets[2:])
    )


def test_projected_load_uses_work_not_workout_label() -> None:
    ordinary = weekly_schedule._session_load_units(6.0, 4.0)
    same_work_with_intensity = weekly_schedule._session_load_units(
        6.0,
        4.0,
        estimated_duration_ratio=1.5,
        prescribed_intensity_factor=1.1,
    )

    assert ordinary == 1.5
    assert same_work_with_intensity == ordinary * 1.1


def test_transient_session_load_quarters_over_one_day() -> None:
    base = _state(
        days_since_last_run=3.0,
        running_days_28d=10,
    )
    planned = recommend_next_run(
        base,
        RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
        CONFIG,
    ).model_copy(update={"planned_for": base.as_of})
    one_day = base.model_copy(
        update={
            "as_of": base.as_of + timedelta(days=1),
            "days_since_last_run": 4.0,
        }
    )
    two_days = base.model_copy(
        update={
            "as_of": base.as_of + timedelta(days=2),
            "days_since_last_run": 5.0,
        }
    )

    after_one_day = _decayed_recovery_load(
        one_day, [planned], one_day.as_of, 3.0
    )
    after_two_days = _decayed_recovery_load(
        two_days, [planned], two_days.as_of, 3.0
    )
    assert after_two_days < after_one_day
    assert abs(after_one_day - after_two_days * 4) < 1e-9


def test_finalized_program_recovery_prices_allocated_distance() -> None:
    base = _state(days_since_last_run=2.0, running_days_28d=12)
    states = [
        base.model_copy(
            update={
                "as_of": base.as_of + timedelta(days=offset),
                "days_since_last_run": 2.0 + offset,
            }
        )
        for offset in range(2)
    ]
    ordinary = recommend_next_run(
        states[0],
        RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
        CONFIG,
        weekly_role="easy",
        allowed_candidates={"easy"},
    )

    def program(first_miles: float) -> list[WeeklyScheduleDay]:
        results = [
            ordinary.model_copy(
                update={
                    "planned_for": states[index].as_of,
                    "distance_range_miles": (miles, miles),
                }
            )
            for index, miles in enumerate((first_miles, 4.0))
        ]
        return [
            WeeklyScheduleDay(
                date=states[index].as_of.date(),
                planned_at=states[index].as_of,
                recommendation=result,
                day_role="easy_run",
                rationale="Finalized recovery fixture.",
            )
            for index, result in enumerate(results)
        ]

    ordinary_cost = _finalized_program_recovery_cost(
        program(4.0), states, CONFIG
    )
    enlarged_cost = _finalized_program_recovery_cost(
        program(6.0), states, CONFIG
    )

    assert enlarged_cost > ordinary_cost


def test_finalized_program_prices_committed_short_term_density() -> None:
    base = _state(days_since_last_run=4.0, running_days_28d=12)
    base = base.model_copy(
        update={
            "recent_load": base.recent_load.model_copy(
                update={
                    "continuous_distance_miles": 14.0,
                    "continuous_short_term_distance_miles": 10.0,
                }
            )
        }
    )
    states = [
        base.model_copy(
            update={
                "as_of": base.as_of + timedelta(days=offset),
                "days_since_last_run": 4.0 + offset,
            }
        )
        for offset in range(5)
    ]
    ordinary = recommend_next_run(
        states[0],
        RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
        CONFIG,
        weekly_role="easy",
        allowed_candidates={"easy"},
    )

    def program(
        offsets: tuple[int, ...],
        miles_by_offset: dict[int, float] | None = None,
    ) -> list[WeeklyScheduleDay]:
        miles_by_offset = miles_by_offset or {}
        results = {
            offset: ordinary.model_copy(
                update={
                    "planned_for": states[offset].as_of,
                    "distance_range_miles": (
                        miles_by_offset.get(offset, 4.0),
                        miles_by_offset.get(offset, 4.0),
                    ),
                }
            )
            for offset in offsets
        }
        return [
            WeeklyScheduleDay(
                date=state.as_of.date(),
                planned_at=(state.as_of if index in results else None),
                recommendation=results.get(index),
                day_role=("easy_run" if index in results else "rest_day"),
                rationale="Committed density fixture.",
            )
            for index, state in enumerate(states)
        ]

    dense = program((0, 1, 2))
    spaced = program((0, 2, 4))
    dense_recovery_only = _finalized_program_recovery_cost(
        dense, states, CONFIG
    )
    spaced_recovery_only = _finalized_program_recovery_cost(
        spaced, states, CONFIG
    )
    dense_with_density = _finalized_program_recovery_cost(
        dense, states, CONFIG, (14.0, 16.0)
    )
    spaced_with_density = _finalized_program_recovery_cost(
        spaced, states, CONFIG, (14.0, 16.0)
    )

    assert dense_with_density - dense_recovery_only > (
        spaced_with_density - spaced_recovery_only
    )

    clustered_short = program((0, 1, 2), {1: 2.5, 2: 2.5})
    isolated_short = program((0, 2, 4), {2: 2.5, 4: 2.5})
    clustered_short_recovery = _finalized_program_recovery_cost(
        clustered_short, states, CONFIG
    )
    isolated_short_recovery = _finalized_program_recovery_cost(
        isolated_short, states, CONFIG
    )
    # Bridge recovery compares each calendar with its own evenly distributed
    # reference. Clustering is costly, while an adequately spaced sequence may
    # legitimately reach the interaction's zero point.
    assert clustered_short_recovery > isolated_short_recovery >= 0


def test_bridge_density_reference_is_derived_from_target_and_session_evidence() -> None:
    base = _state(typical_easy_run_miles=4.0, running_days_28d=12)
    base = base.model_copy(
        update={
            "recent_load": base.recent_load.model_copy(
                update={
                    "trailing_28d": base.recent_load.trailing_28d.model_copy(
                        update={"distance_miles": 64.0, "activity_count": 16}
                    )
                }
            )
        }
    )

    reference_gap_hours, reference_load = _target_derived_bridge_reference(
        base, (18.0, 20.0), 4.0
    )
    short_history = base.model_copy(
        update={
            "recent_load": base.recent_load.model_copy(
                update={
                    "trailing_28d": base.recent_load.trailing_28d.model_copy(
                        update={"distance_miles": 16.0, "activity_count": 16}
                    )
                }
            )
        }
    )
    assert reference_gap_hours == pytest.approx(4.0 * 7.0 * 24.0 / 19.0)
    assert reference_load == pytest.approx(1.0)
    # Incidental short runs cannot shrink the protected ordinary-session unit
    # and thereby manufacture permission for a denser calendar.
    assert _target_derived_bridge_reference(
        short_history, (18.0, 20.0), 4.0
    ) == pytest.approx((reference_gap_hours, reference_load))


def test_post_long_catch_up_keeps_the_immediate_slot_aerobic() -> None:
    base = _state(
        days_since_last_run=1.0,
        days_since_long_run=1.0,
        days_since_quality_run=7.0,
        last_run_workout_type=WorkoutType.LONG,
        last_run=_difficulty(miles=7.0, long=True),
        running_days_28d=12,
        longest_run_30d_miles=7.0,
    )
    base = base.model_copy(
        update={
            "recent_load": base.recent_load.model_copy(
                update={
                    "capacity_reference_miles": 16.4,
                    "sustained_capacity_miles": 16.4,
                    "acute_distance_to_capacity_ratio": 1.20,
                    "continuous_fatigue_to_capacity_ratio": 1.20,
                    "continuous_fatigue_miles": 19.68,
                    "continuous_distance_miles": 18.0,
                }
            )
        }
    )
    states = [
        base.model_copy(
            update={
                "as_of": base.as_of + timedelta(days=offset),
                "days_since_last_run": 1.0 + offset,
                "days_since_long_run": 1.0 + offset,
                "days_since_quality_run": 7.0 + offset,
            }
        )
        for offset in range(21)
    ]
    sessions = weekly_schedule._materialize_candidate_sessions(
        (0, 2, 4, 6, 8, 10, 13, 16, 18),
        states,
        [[state] for state in states],
        RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
        CONFIG,
    )

    assert sessions[0].offset == 0
    assert sessions[0].recommendation.workout_type == WorkoutType.EASY


def test_future_candidate_only_caution_is_priced_at_allocator_size(monkeypatch) -> None:
    base = _state(typical_easy_run_miles=4.0)
    states = [
        base.model_copy(update={"as_of": base.as_of + timedelta(days=offset)})
        for offset in range(3)
    ]
    short = RecommendationResponse(
        generated_at=states[1].as_of,
        fitness_state_as_of=states[1].as_of,
        planned_for=states[1].as_of,
        workout_type=WorkoutType.EASY,
        title="Recovery-shortened candidate",
        distance_range_miles=(1.8, 2.2),
        confidence=ConfidenceLevel.MODERATE,
        readiness=ReadinessFlag.CAUTION,
    )
    monkeypatch.setattr(
        weekly_schedule,
        "_elapsed_workout_role",
        lambda *args, **kwargs: "easy",
    )
    monkeypatch.setattr(
        weekly_schedule,
        "_select_budgeted_timed_recommendation",
        lambda *args, **kwargs: (states[1], short),
    )

    sessions = weekly_schedule._materialize_candidate_sessions(
        (1,),
        states,
        [[state] for state in states],
        RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
        CONFIG,
    )

    midpoint = sum(sessions[0].recommendation.distance_range_miles) / 2
    assert midpoint == weekly_schedule._established_easy_midpoint_floor(
        states[1]
    )


def test_race_inside_horizon_is_scheduled_without_previous_day_compression() -> None:
    base = _state(running_days_28d=10)
    states = [
        base.model_copy(update={"as_of": base.as_of + timedelta(days=offset)})
        for offset in range(7)
    ]
    race_date = states[3].as_of.date().isoformat()
    config = {
        **CONFIG,
        "coaching": {
            **CONFIG["coaching"],
            "training_goal": "5k",
            "goal_date": race_date,
            "goal_pace_min_mile": 9.0,
        },
    }
    schedule = build_weekly_schedule(
        states,
        RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
        config,
        target_run_count=3,
    )

    assert schedule.days[2].recommendation is None
    assert schedule.days[3].recommendation is not None
    assert schedule.days[3].recommendation.workout_type == WorkoutType.RACE


def test_taper_applies_only_before_race_and_normal_roles_resume_after_recovery() -> None:
    base = _state(
        running_days_28d=16,
        days_since_last_run=2.0,
        days_since_long_run=7.0,
        days_since_quality_run=7.0,
    )
    states = [
        base.model_copy(
            update={
                "as_of": base.as_of + timedelta(days=offset),
                "days_since_last_run": 2.0 + offset,
                "days_since_long_run": 7.0 + offset,
                "days_since_quality_run": 7.0 + offset,
            }
        )
        for offset in range(21)
    ]
    race_offset = 4
    config = {
        **CONFIG,
        "coaching": {
            **CONFIG["coaching"],
            "training_goal": "half_marathon",
            "goal_date": states[race_offset].as_of.date().isoformat(),
            "goal_pace_min_mile": 9.0,
        },
    }

    schedule = build_weekly_schedule(
        states,
        RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
        config,
        target_run_count=4,
        target_distance_range=(16.9, 18.0),
    )
    taxing = {
        WorkoutType.LONG,
        WorkoutType.INTERVALS,
        WorkoutType.TEMPO_THRESHOLD,
    }

    assert schedule.planning_days[race_offset].recommendation is not None
    assert (
        schedule.planning_days[race_offset].recommendation.workout_type
        == WorkoutType.RACE
    )
    assert all(
        day.recommendation is None
        or day.recommendation.workout_type not in taxing
        for day in schedule.planning_days[:race_offset]
    )
    assert all(
        day.recommendation is None
        or day.recommendation.workout_type not in taxing
        for day in schedule.planning_days[race_offset + 1 : race_offset + 5]
    )
    assert any(
        day.recommendation
        and day.recommendation.workout_type in taxing
        for day in schedule.planning_days[race_offset + 5 :]
    )
