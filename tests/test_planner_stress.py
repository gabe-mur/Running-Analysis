from __future__ import annotations

from datetime import timedelta
from random import Random

from run_analysis.weekly_schedule import build_weekly_schedule
from run_analysis.web.schemas import (
    CurrentHealthStatus,
    RecommendationRequest,
    WorkoutType,
)
from test_recommendation import CONFIG, _difficulty, _state


def _random_state(rng: Random):
    trailing_28_miles = rng.uniform(8, 100)
    activity_count = rng.randint(2, 24)
    capacity = rng.uniform(6, 30)
    trailing_7_miles = rng.uniform(0, capacity * 1.8)
    zone_load_28 = trailing_28_miles * rng.uniform(12, 28)
    recent = _state().recent_load
    recent = recent.model_copy(
        update={
            "trailing_7d": recent.trailing_7d.model_copy(
                update={
                    "distance_miles": trailing_7_miles,
                    "moving_minutes": trailing_7_miles * rng.uniform(9, 15),
                    "zone_load": zone_load_28 / 4 * rng.uniform(0.3, 1.8),
                    "activity_count": rng.randint(0, 7),
                }
            ),
            "trailing_14d": recent.trailing_14d.model_copy(
                update={
                    "distance_miles": rng.uniform(trailing_7_miles, max(trailing_7_miles, trailing_28_miles)),
                    "moving_minutes": trailing_28_miles * 6,
                    "zone_load": zone_load_28 / 2,
                    "activity_count": rng.randint(1, 14),
                }
            ),
            "trailing_28d": recent.trailing_28d.model_copy(
                update={
                    "distance_miles": trailing_28_miles,
                    "moving_minutes": trailing_28_miles * rng.uniform(9, 15),
                    "zone_load": zone_load_28,
                    "activity_count": activity_count,
                }
            ),
            "acute_distance_to_capacity_ratio": trailing_7_miles / capacity,
            "acute_to_prior_ratio": rng.uniform(0.2, 2.5),
            "capacity_reference_miles": capacity,
            "sustained_capacity_miles": capacity,
            "prior_28d_weekly_miles": trailing_28_miles / 4,
        }
    )
    last_miles = rng.uniform(0.5, 12)
    last = _difficulty(
        miles=last_miles,
        long=rng.random() < 0.15,
        quality=rng.random() < 0.18,
        rpe=rng.choice([None, 3, 5, 6, 8, 9]),
    ).model_copy(
        update={
            "moving_minutes": last_miles * rng.uniform(8, 16),
            "elapsed_minutes": last_miles * rng.uniform(9, 18),
            "zone_load": rng.uniform(5, 260),
        }
    )
    return _state(
        recent_load=recent,
        days_since_last_run=rng.uniform(0.3, 8),
        days_since_quality_run=rng.uniform(0, 30),
        days_since_long_run=rng.uniform(0, 40),
        last_run=last,
        longest_run_30d_miles=rng.uniform(3, 13),
        quality_sessions_14d=rng.randint(0, 3),
        running_days_28d=rng.randint(1, 24),
        moderate_fraction_14d=rng.uniform(0, 0.4),
        moderate_evidence_runs_14d=rng.randint(0, 8),
        recent_performance_anomaly=rng.choice(
            ["within_recent_range", "unusually_costly", "unknown"]
        ),
    )


def test_randomized_weekly_plans_preserve_core_invariants() -> None:
    rng = Random(145)
    for _ in range(750):
        base = _random_state(rng)
        health = rng.choice(list(CurrentHealthStatus))
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
        target_runs = rng.randint(1, 7)
        target_low = rng.uniform(6, 25)
        schedule = build_weekly_schedule(
            states,
            RecommendationRequest(health_status=health),
            CONFIG,
            target_run_count=target_runs,
            target_distance_range=(target_low, target_low + rng.uniform(1, 5)),
        )

        assert len(schedule.days) == 7
        assert [day.date for day in schedule.days] == sorted(day.date for day in schedule.days)
        planned = [day.recommendation for day in schedule.days if day.recommendation]
        running = [item for item in planned if item.workout_type != WorkoutType.REST]
        assert schedule.run_count == len(running)
        assert schedule.projected_distance_range_miles[0] >= 0
        assert schedule.projected_distance_range_miles[1] >= schedule.projected_distance_range_miles[0]
        assert all(
            item.distance_range_miles is None
            or item.distance_range_miles[0] <= item.distance_range_miles[1]
            for item in planned
        )
        if health == CurrentHealthStatus.PAIN_OR_INJURY_CONCERN:
            assert not running
        if health == CurrentHealthStatus.SICK_OR_RECOVERING:
            assert len(running) <= 2
            assert all(item.workout_type == WorkoutType.RECOVERY for item in running)

        quality_dates = [
            item.planned_for
            for item in running
            if item.workout_type in {
                WorkoutType.INTERVALS,
                WorkoutType.TEMPO_THRESHOLD,
                WorkoutType.RACE,
            }
        ]
        assert all(
            (later - earlier).total_seconds() >= CONFIG["coaching"]["minimum_days_between_quality_sessions"] * 86400
            for earlier, later in zip(quality_dates, quality_dates[1:])
        )


def test_extreme_overload_does_not_prescribe_quality_or_long_run() -> None:
    base = _state()
    recent = base.recent_load.model_copy(
        update={
            "trailing_7d": base.recent_load.trailing_7d.model_copy(
                update={"distance_miles": 28, "moving_minutes": 310, "zone_load": 520}
            ),
            "acute_distance_to_capacity_ratio": 1.75,
            "acute_to_prior_ratio": 2.4,
            "capacity_reference_miles": 16,
        }
    )
    base = base.model_copy(update={"recent_load": recent})
    states = [
        base.model_copy(update={"as_of": base.as_of + timedelta(days=offset)})
        for offset in range(7)
    ]
    schedule = build_weekly_schedule(
        states,
        RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
        CONFIG,
        target_run_count=5,
        target_distance_range=(16, 19),
    )
    running = [day.recommendation for day in schedule.days if day.recommendation]
    assert running
    assert all(item.workout_type == WorkoutType.EASY for item in running)
    assert all(item.distance_range_miles[1] <= 3.5 for item in running)
    assert "below your usual range" in schedule.summary
