from __future__ import annotations

from datetime import datetime, timedelta, timezone

import run_analysis.weekly_schedule as weekly_schedule
from run_analysis.recommendation import recommend_next_run
from run_analysis.weekly_schedule import (
    PlanningActivity,
    _allocate_visible_distance_ranges,
    _decayed_recovery_load,
    _ordinary_easy_expansion_reference,
    _rest_day_rationale,
    _select_timed_recommendation,
    adaptive_run_day_offsets,
    automatic_run_day_offsets,
    build_weekly_schedule,
    derive_weekly_target,
    summarize_distance_alignment,
)
from run_analysis.web.schemas import (
    ActivityHealthTag,
    CurrentHealthStatus,
    PlannedWeather,
    RecommendationRequest,
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
    ) == [1, 3, 5, 7]


def test_yesterday_run_does_not_push_an_evening_slot_to_tomorrow() -> None:
    base = _state(
        as_of=datetime(2026, 8, 25, 19, tzinfo=timezone.utc),
        days_since_last_run=1.0,
    )
    assert automatic_run_day_offsets(
        base, CurrentHealthStatus.NORMAL, CONFIG, target_run_count=4
    ) == [0, 2, 4, 6]


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


def test_horizon_never_pulls_work_later_merely_to_fill_seven_days() -> None:
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
    assert [index for index, day in enumerate(schedule.days) if day.recommendation] == [1, 3, 5]
    assert schedule.run_count == 3
    assert schedule.target_run_count == 3
    assert "below your usual range" in schedule.summary


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
    assert sequence_trace.facts["preferred_role"] == "long"


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

    assert runs == 2
    assert distance[0] >= 14
    assert evidence.demonstrated_run_days_per_week > 3.5
    assert evidence.current_run_days_per_week == 0


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
    # The visible slice is not required to consume a weekly quota. The next
    # opportunity can fall just beyond day seven in the continuous plan.
    assert schedule.projected_distance_range_miles[0] >= 12.0
    assert schedule.projected_distance_range_miles[1] <= 18.0
    adjacent_pairs = [
        (previous, current)
        for previous, current in zip(planned, planned[1:])
        if current[0] - previous[0] == 1
    ]
    assert adjacent_pairs
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


def test_yesterday_long_allows_easy_today_without_stacking_quality() -> None:
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

    assert planned[0][0] == 0
    assert planned[0][1].workout_type == WorkoutType.EASY
    first_quality_offset = next(
        index
        for index, result in planned
        if result.workout_type
        in {WorkoutType.INTERVALS, WorkoutType.TEMPO_THRESHOLD}
    )
    assert first_quality_offset >= 2


def test_yesterday_quality_allows_easy_today_without_stacking_long() -> None:
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

    assert planned[0][0] == 0
    assert planned[0][1].workout_type == WorkoutType.EASY
    first_taxing_offset = next(
        index
        for index, result in planned
        if result.workout_type
        in {WorkoutType.LONG, WorkoutType.INTERVALS, WorkoutType.TEMPO_THRESHOLD}
    )
    assert first_taxing_offset >= 1


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
    assert allocated[0].recommendation.distance_range_miles[0] >= 5.0


def test_tight_week_shortens_quality_instead_of_deleting_its_stimulus() -> None:
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
    assert quality.distance_range_miles == (3.0, 3.5)
    assert any("instead of deleting" in reason for reason in quality.reasons)


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

    assert allocated_longs[1][0] > allocated_longs[0][0]
    assert allocated_longs[1][0] >= 6.75


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
    assert long_range[0] >= easy_range[1] + 1.0
    assert quality_range[1] <= 5.0
    assert easy_range[1] > quality_range[1]


def test_weekly_shortfall_expands_easy_running_before_maxing_long_run() -> None:
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
    # The weekly range is a soft density guide, not a clipping threshold.
    assert schedule.projected_distance_range_miles[1] <= 12.6
    assert all("balanced spacing" in day.rationale for day in planned_days)


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
    # Four weekly opportunities emerge from mileage distribution; neither
    # supplied count becomes a hard target. Useful doubles remain available,
    # but the optimizer does not create a three-day catch-up streak.
    assert len(low_count_input) == 12
    gaps = [
        current - previous
        for previous, current in zip(low_count_input, low_count_input[1:])
    ]
    assert 1 in gaps
    assert not any(
        current - two_back == 2
        for two_back, current in zip(low_count_input, low_count_input[2:])
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
    assert calls <= weekly_schedule.MAX_HORIZON_COUNT_OPTIONS * (
        weekly_schedule.MAX_ADAPTIVE_CANDIDATES + 1
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


def test_role_aware_distribution_prefers_recovered_ten_run_plan(
    monkeypatch,
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
        return {10: 59.11, 11: 102.22}.get(len(offsets), 1_000.0)

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

    offsets = adaptive_run_day_offsets(
        states,
        RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
        CONFIG,
        3,
        (15.5, 18.0),
    )

    assert len(offsets) == 10


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
    assert replanned.projected_distance_range_miles[1] <= 12.5
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


def test_planned_easy_run_clears_meaningful_moderate_leakage_for_later_days() -> None:
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
    assert recommendations[0].workout_type == WorkoutType.EASY
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
    # At only 22.8 hours since the latest run, either today or tomorrow can
    # win; the invariant is that low load does not manufacture a long gap.
    assert planned_offsets[0] in {0, 1}
    assert max(
        current - previous - 1
        for previous, current in zip(planned_offsets, planned_offsets[1:])
    ) <= 2
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
        recent_performance_anomaly="unusually_strong",
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
    assert today.distance_range_miles[1] <= standalone.distance_range_miles[1]


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
    # Long and quality remain separated. Short easy running can then form a
    # useful double rather than forcing a rigid every-other-day calendar.
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
    assert any(
        current_index - previous_index == 1
        and previous.workout_type == WorkoutType.EASY
        and current.workout_type == WorkoutType.EASY
        for (previous_index, previous), (current_index, current)
        in zip(planned, planned[1:])
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
    assert 1 <= len(replanned_offsets) <= len(baseline_offsets)
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
    assert 1 <= len(replanned_offsets) <= len(baseline_offsets)
    replanned_by_offset = {
        index: day.recommendation
        for index, day in enumerate(replanned.days)
        if day.recommendation
    }
    for previous, current in zip(replanned_offsets, replanned_offsets[1:]):
        if current - previous == 1:
            assert replanned_by_offset[previous].workout_type == WorkoutType.EASY
            assert replanned_by_offset[current].workout_type == WorkoutType.EASY
    # The whole horizon is rescored. It may independently retain later dates
    # when they remain the strongest choices; the constrained date itself must
    # disappear rather than being copied into a one-day substitution rule.
    assert replanned_offsets != baseline_offsets
    assert baseline_offsets[0] not in replanned_offsets


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
        one_day, [planned], one_day.as_of, 3.0, CONFIG
    )
    after_two_days = _decayed_recovery_load(
        two_days, [planned], two_days.as_of, 3.0, CONFIG
    )
    assert after_two_days < after_one_day
    assert abs(after_one_day - after_two_days * 4) < 1e-9


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
