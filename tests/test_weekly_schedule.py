from __future__ import annotations

from datetime import timedelta

from run_analysis.weekly_schedule import (
    PlanningActivity,
    automatic_run_day_offsets,
    build_weekly_schedule,
    derive_weekly_target,
)
from run_analysis.web.schemas import (
    ActivityHealthTag,
    CurrentHealthStatus,
    RecommendationRequest,
    TrailingDayActivity,
    WorkoutType,
)
from test_recommendation import CONFIG, _difficulty, _state


def test_automatic_week_uses_recent_frequency_and_allows_intentional_consecutive_days() -> None:
    disrupted = _state(running_days_28d=4)
    assert automatic_run_day_offsets(
        disrupted, CurrentHealthStatus.NORMAL, target_run_count=5
    ) == [0, 1, 3, 4, 6]


def test_run_yesterday_makes_today_rest_without_losing_four_run_target() -> None:
    base = _state(days_since_last_run=1.0)
    assert automatic_run_day_offsets(
        base, CurrentHealthStatus.NORMAL, CONFIG, target_run_count=4
    ) == [1, 3, 5, 7]


def test_low_cost_run_yesterday_can_preserve_intentional_next_day_run() -> None:
    low_cost = _difficulty(miles=1.1, rpe=6).model_copy(
        update={"moving_minutes": 11.0, "elapsed_minutes": 12.0, "zone_load": 9.0}
    )
    base = _state(days_since_last_run=1.0, last_run=low_cost)

    assert automatic_run_day_offsets(
        base, CurrentHealthStatus.NORMAL, CONFIG, target_run_count=4
    ) == [0, 2, 4, 6]


def test_low_cost_run_does_not_override_high_accumulated_load() -> None:
    low_cost = _difficulty(miles=1.1, rpe=6).model_copy(
        update={"moving_minutes": 11.0, "elapsed_minutes": 12.0, "zone_load": 9.0}
    )
    loaded = _state().recent_load.model_copy(
        update={"acute_distance_to_capacity_ratio": 1.4}
    )
    base = _state(days_since_last_run=1.0, last_run=low_cost, recent_load=loaded)

    assert automatic_run_day_offsets(
        base, CurrentHealthStatus.NORMAL, CONFIG, target_run_count=4
    ) == [1, 3, 5, 7]


def test_horizon_never_pulls_long_run_earlier_to_fill_seven_days() -> None:
    base = _state(
        days_since_last_run=1.0,
        days_since_quality_run=20,
        days_since_long_run=20,
        running_days_28d=10,
    )
    states = [
        base.model_copy(
            update={
                "as_of": base.as_of + timedelta(days=offset),
                "days_since_last_run": 1.0 + offset,
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
    assert [index for index, day in enumerate(schedule.days) if day.recommendation] == [1, 3, 5]
    assert all(day.day_role != "long_run" for day in schedule.days)
    assert schedule.run_count == 3
    assert schedule.target_run_count == 4
    assert "additional session" in schedule.summary
    assert "not included in the planned mileage" in schedule.summary
    assert "capacity reference" in schedule.summary


def test_final_visible_run_is_not_forced_long_by_horizon_position() -> None:
    base = _state(
        days_since_last_run=2.0,
        days_since_quality_run=20,
        days_since_long_run=2,
        running_days_28d=10,
    )
    states = [
        base.model_copy(
            update={
                "as_of": base.as_of + timedelta(days=offset),
                "days_since_last_run": 2.0 + offset,
                "days_since_quality_run": 20.0 + offset,
                "days_since_long_run": 2.0 + offset,
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
    assert sequence_trace.fired is False
    assert sequence_trace.facts["preferred_role"] is None


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


def test_normalization_week_sequences_easy_quality_easy_long_after_run_yesterday() -> None:
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
    assert [day.day_role for day in schedule.days] == [
        "easy_run",
        "rest_day",
        "quality_run",
        "rest_day",
        "easy_run",
        "rest_day",
        "long_run",
    ]
