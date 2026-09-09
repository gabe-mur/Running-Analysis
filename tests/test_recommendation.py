from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from run_analysis.environmental_stress import assess_training_weather
from run_analysis.recommendation import (
    recommend_next_run,
    scale_quality_session,
    structure_extended_quality_session,
    typical_easy_distance,
)
from run_analysis.web.schemas import (
    ConfidenceLevel,
    CurrentHealthStatus,
    FitnessState,
    FitnessTrend,
    LoadContext,
    LoadWindow,
    PaceValue,
    PlannedWeather,
    RecommendationRequest,
    SessionDifficulty,
    WeatherExposureBaseline,
    WeatherEmergencyAlert,
    WorkoutType,
    ZoneBreakdown,
)
from fastapi.testclient import TestClient
from pathlib import Path
from run_analysis.db import connect, initialize
from run_analysis.web.app import create_app
from test_web_phase1 import _write_config


CONFIG = {
    "coaching": {
        "long_run_progression_factor": 1.10,
        "high_load_ratio": 1.30,
        "moderate_intensity_leakage_fraction": 0.17,
        "minimum_running_days_28d_for_quality": 8,
        "long_run_recency_reference_days": 7,
        "reduced_volume_factor": 0.70,
    }
}


def _window(days: int, miles: float, load: float) -> LoadWindow:
    return LoadWindow(days=days, distance_miles=miles, moving_minutes=miles * 11, zone_load=load, hard_minutes=2, activity_count=max(1, int(miles / 4)))


def _difficulty(*, long: bool = False, quality: bool = False, miles: float = 5, rpe: int | None = None, flags: list[str] | None = None) -> SessionDifficulty:
    return SessionDifficulty(
        distance_miles=miles,
        moving_minutes=miles * 11,
        elapsed_minutes=miles * 11,
        stopped_minutes=0,
        zone_load=100,
        perceived_exertion=rpe,
        zone_breakdown=ZoneBreakdown(),
        is_long_run=long,
        is_quality_session=quality,
        difficulty_flags=flags or [],
    )


def _state(**changes) -> FitnessState:
    values = dict(
        as_of=datetime.now(timezone.utc),
        window_days=28,
        fitness_trend=FitnessTrend.STABLE,
        trend_confidence=ConfidenceLevel.MODERATE,
        recent_load=LoadContext(
            trailing_7d=_window(7, 12, 220),
            trailing_14d=_window(14, 24, 440),
            trailing_28d=_window(28, 48, 880),
            acute_to_prior_ratio=1.0,
            confidence=ConfidenceLevel.HIGH,
        ),
        days_since_last_run=3,
        days_since_quality_run=8,
        days_since_long_run=3,
        last_run=_difficulty(),
        last_run_workout_type=WorkoutType.EASY,
        longest_run_30d_miles=8,
        quality_sessions_14d=0,
        running_days_28d=10,
        easy_fraction_14d=0.82,
        moderate_fraction_14d=0.10,
        moderate_evidence_runs_14d=3,
        hard_fraction_14d=0.08,
        recent_performance_response="within_recent_range",
    )
    values.update(changes)
    return FitnessState(**values)


def _recommend(state: FitnessState, status=CurrentHealthStatus.NORMAL, config=None):
    return recommend_next_run(
        state, RecommendationRequest(health_status=status), config or CONFIG
    )


def test_typical_easy_distance_prefers_observed_easy_run_baseline() -> None:
    state = _state(typical_easy_run_miles=4.0)

    assert typical_easy_distance(state) == (3.5, 4.0)


def test_low_load_three_days_rest_and_no_recent_quality_can_be_quality_eligible() -> None:
    result = _recommend(_state(days_since_quality_run=12, days_since_long_run=3))
    assert result.workout_type == WorkoutType.INTERVALS
    assert result.planned_for is not None
    assert any(item.rule_id == "planned_timing" and item.fired for item in result.rule_trace)
    assert any(item.rule_id == "quality_eligible" and item.fired for item in result.rule_trace)


def _planned_weather(*, apparent: float, dewpoint: float) -> PlannedWeather:
    return PlannedWeather(
        forecast_time=datetime.now(timezone.utc),
        temperature_f=apparent - 5,
        apparent_temperature_f=apparent,
        dewpoint_f=dewpoint,
        wind_speed_mph=3,
        wind_gust_mph=7,
        precipitation_probability_percent=0,
        precipitation_in=0,
    )


def test_weather_stress_is_continuous_across_old_dewpoint_cutoff() -> None:
    below = assess_training_weather(_planned_weather(apparent=84.2, dewpoint=69.9))
    above = assess_training_weather(_planned_weather(apparent=84.2, dewpoint=70.1))

    assert below.band == above.band == "none"
    assert 0 < above.score - below.score < 0.02


def test_weather_spike_adapts_but_does_not_erase_due_long_run() -> None:
    baseline = WeatherExposureBaseline(
        sample_count=6,
        warm_apparent_temperature_f=75,
        cold_apparent_temperature_f=60,
        humid_dewpoint_f=60,
    )
    result = recommend_next_run(
        _state(
            planned_weather=_planned_weather(apparent=84.2, dewpoint=71.3),
            weather_exposure_baseline=baseline,
            days_since_long_run=12,
            days_since_quality_run=3,
            quality_sessions_14d=1,
        ),
        RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
        CONFIG,
        weekly_role="long",
    )
    weather_trace = next(
        item for item in result.rule_trace if item.rule_id == "planned_weather"
    )

    assert result.workout_type == WorkoutType.LONG
    assert result.readiness.value == "caution"
    assert weather_trace.fired is True
    assert weather_trace.facts["stress_band"] == "moderate"


def test_severe_combined_heat_and_humidity_can_redirect_long_run() -> None:
    result = recommend_next_run(
        _state(
            planned_weather=_planned_weather(apparent=105, dewpoint=80),
            days_since_long_run=12,
            days_since_quality_run=3,
            quality_sessions_14d=1,
        ),
        RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
        CONFIG,
        weekly_role="long",
    )

    assert result.workout_type == WorkoutType.REST
    assert result.readiness.value == "not_ready"


def test_conditions_within_recent_exposure_do_not_penalize_training() -> None:
    baseline = WeatherExposureBaseline(
        sample_count=9,
        warm_apparent_temperature_f=87,
        cold_apparent_temperature_f=72,
        humid_dewpoint_f=73,
    )
    assessed = assess_training_weather(
        _planned_weather(apparent=84.2, dewpoint=71.3), baseline
    )

    assert assessed.score == 0
    assert assessed.band == "none"


def test_relative_weather_does_not_penalize_absolute_comfort_zone() -> None:
    cool_month = WeatherExposureBaseline(
        sample_count=8,
        warm_apparent_temperature_f=55,
        cold_apparent_temperature_f=45,
        humid_dewpoint_f=40,
    )
    hot_month = WeatherExposureBaseline(
        sample_count=8,
        warm_apparent_temperature_f=88,
        cold_apparent_temperature_f=80,
        humid_dewpoint_f=72,
    )

    pleasant_warm = assess_training_weather(
        _planned_weather(apparent=75, dewpoint=45), cool_month
    )
    pleasant_cool = assess_training_weather(
        _planned_weather(apparent=60, dewpoint=45), hot_month
    )

    assert pleasant_warm.band == "none"
    assert pleasant_cool.band == "none"


def test_ordinary_rain_is_neutral_training_context() -> None:
    dry = _planned_weather(apparent=68, dewpoint=58)
    rain = dry.model_copy(
        update={
            "precipitation_probability_percent": 100,
            "precipitation_in": 1.5,
            "weather_code": 65,
        }
    )

    assert assess_training_weather(rain).score == assess_training_weather(dry).score
    assert assess_training_weather(rain).extreme is False


def test_thunderstorm_is_an_absolute_extreme_weather_guardrail() -> None:
    storm = _planned_weather(apparent=75, dewpoint=65).model_copy(
        update={"weather_code": 95}
    )
    result = _recommend(_state(planned_weather=storm))

    assert result.workout_type == WorkoutType.REST
    assert result.readiness.value == "not_ready"
    assert "thunderstorm" in result.reasons[0]


def test_blizzard_like_conditions_are_an_absolute_guardrail() -> None:
    blizzard = _planned_weather(apparent=15, dewpoint=10).model_copy(
        update={
            "weather_code": 75,
            "wind_gust_mph": 40,
            "visibility_miles": 0.2,
            "snowfall_in": 1.0,
        }
    )
    result = _recommend(_state(planned_weather=blizzard))

    assert result.workout_type == WorkoutType.REST
    assert "blizzard-like" in result.reasons[0]


def test_hurricane_force_wind_is_an_absolute_guardrail() -> None:
    hurricane = _planned_weather(apparent=78, dewpoint=72).model_copy(
        update={"wind_speed_mph": 75, "wind_gust_mph": 90}
    )
    result = _recommend(_state(planned_weather=hurricane))

    assert result.workout_type == WorkoutType.REST
    assert "hurricane-force" in result.reasons[0]


def test_official_dangerous_warning_blocks_only_its_outdoor_window() -> None:
    warning = WeatherEmergencyAlert(
        alert_id="https://api.weather.gov/alerts/example",
        event="Tornado Warning",
        headline="Tornado Warning issued for the planned route area",
        severity="Extreme",
        urgency="Immediate",
        certainty="Observed",
        blocks_outdoor_run=True,
    )
    weather = _planned_weather(apparent=75, dewpoint=62).model_copy(
        update={
            "emergency_alerts_checked": True,
            "emergency_alerts": [warning],
        }
    )

    result = _recommend(_state(planned_weather=weather))
    trace = next(
        item for item in result.rule_trace if item.rule_id == "planned_weather"
    )

    assert result.workout_type == WorkoutType.REST
    assert result.title == "Extreme-weather no-run window"
    assert "official NWS Tornado Warning" in result.reasons[0]
    assert trace.facts["emergency_alerts_checked"] is True
    assert trace.facts["emergency_alerts"] == "Tornado Warning"


def test_quality_cadence_is_recovery_and_dose_driven_not_a_calendar_gate() -> None:
    state = _state(
        days_since_quality_run=5,
        quality_sessions_14d=1,
        days_since_long_run=3,
    )
    ordinary = _recommend(state)
    result = recommend_next_run(
        state,
        RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
        CONFIG,
        weekly_role="quality",
    )
    assert ordinary.workout_type == WorkoutType.INTERVALS
    assert result.workout_type == WorkoutType.INTERVALS
    quality = next(item for item in result.rule_trace if item.rule_id == "quality_eligible")
    assert quality.facts["quality_recency_reference_days"] == 7
    assert quality.facts["recent_quality_saturation"] < 1
    assert quality.facts["quality_need"] > 0


def test_long_run_recovery_load_suppresses_quality_without_forcing_rest() -> None:
    result = _recommend(_state(days_since_last_run=0.8, last_run=_difficulty(long=True, miles=8)))
    assert result.workout_type == WorkoutType.EASY
    recovery = next(
        item for item in result.rule_trace
        if item.rule_id == "recent_recovery_load"
    )
    assert recovery.fired
    assert 0 < recovery.facts["hours_until_easy"] < recovery.facts["hours_until_taxing"]


def test_high_z3_leakage_forces_easy_z1_z2() -> None:
    result = _recommend(_state(moderate_fraction_14d=0.24))
    assert result.workout_type == WorkoutType.EASY
    assert result.target_zones == ["Z1", "Z2"]
    assert result.readiness.value == "caution"
    assert "24%" in result.readiness_reason
    assert "17%" in result.readiness_reason
    assert "Keep this run truly easy" in result.readiness_reason


def test_marginal_z3_excess_does_not_trigger_full_caution() -> None:
    result = _recommend(
        _state(
            moderate_fraction_14d=0.175,
            moderate_evidence_runs_14d=3,
        )
    )
    trace = next(
        item for item in result.rule_trace if item.rule_id == "moderate_leakage"
    )

    assert trace.fired is False
    assert 0 < trace.facts["evidence_strength"] < 0.5
    assert "above the 17%" not in (result.readiness_reason or "")


def test_higher_cost_response_is_not_a_second_direct_distance_penalty() -> None:
    baseline = _recommend(_state(recent_performance_response="within_recent_range"))
    result = _recommend(
        _state(recent_performance_response="higher_cost_than_recent")
    )
    trace = next(
        item for item in result.rule_trace
        if item.rule_id == "recent_costly_response"
    )

    assert trace.facts["performance_response_accounted_in_recovery"] is True
    assert trace.facts["response_stress"] == 0
    assert result.distance_range_miles == baseline.distance_range_miles


def test_strong_response_is_not_overridden_by_modest_drift() -> None:
    state = _state(
        recent_performance_response="stronger_than_recent",
        last_run_drift_percent=7.1,
        days_since_quality_run=3,
        quality_sessions_14d=1,
    )
    baseline = _recommend(
        state.model_copy(update={"last_run_drift_percent": None})
    )
    result = _recommend(
        state
    )
    costly = next(
        item for item in result.rule_trace
        if item.rule_id == "recent_costly_response"
    )
    drift = next(
        item for item in result.rule_trace
        if item.rule_id == "recent_drift_caution"
    )

    assert costly.fired is False
    assert costly.facts["response_stress"] < 0.1
    assert drift.fired is True
    assert 0 < drift.facts["volume_reduction_fraction"] < 0.05
    assert result.distance_range_miles == baseline.distance_range_miles


def test_moderate_standalone_drift_scales_volume_instead_of_using_full_penalty() -> None:
    state = _state(
        recent_performance_response="within_recent_range",
        last_run_drift_percent=7.1,
        days_since_quality_run=3,
        quality_sessions_14d=1,
    )
    result = _recommend(state)
    drift = next(
        item for item in result.rule_trace
        if item.rule_id == "recent_drift_caution"
    )

    assert drift.fired is True
    assert 0 < drift.facts["response_stress"] < 0.5
    assert 0 < drift.facts["volume_reduction_fraction"] < 0.30
    assert result.workout_type == WorkoutType.EASY
    assert result.distance_range_miles[0] < typical_easy_distance(state)[0]
    assert result.distance_range_miles[1] < typical_easy_distance(state)[1]


def test_older_comparable_anomaly_does_not_penalize_completed_quality_run_again() -> None:
    state = _state(
        last_run=_difficulty(miles=3.2, quality=True),
        last_run_workout_type=WorkoutType.INTERVALS,
        last_run_activity_id=102,
        recent_performance_response="higher_cost_than_recent",
        recent_performance_response_activity_id=101,
        last_run_drift_percent=None,
        days_since_last_run=1.5,
        days_since_quality_run=1.5,
    )
    baseline = _recommend(
        state.model_copy(
            update={
                "recent_performance_response": "within_recent_range",
                "recent_performance_response_activity_id": None,
            }
        )
    )
    result = _recommend(state)
    response = next(
        item for item in result.rule_trace
        if item.rule_id == "recent_costly_response"
    )

    assert response.facts["response_stress"] == 0
    assert result.distance_range_miles == baseline.distance_range_miles


def test_several_normal_runs_after_illness_can_restore_normal_eligibility() -> None:
    result = _recommend(_state(recent_illness_or_recovery=True, normal_runs_since_health_event=4, days_since_long_run=3))
    assert result.workout_type == WorkoutType.INTERVALS


def test_illness_blocks_until_current_self_report_returns_to_normal() -> None:
    state = _state(recent_illness_or_recovery=True, normal_runs_since_health_event=0)
    reduced = _recommend(state, status=CurrentHealthStatus.SICK_OR_RECOVERING)
    normal = _recommend(state, status=CurrentHealthStatus.NORMAL)
    assert reduced.workout_type == WorkoutType.RECOVERY
    assert reduced.distance_range_miles[1] <= 3
    assert normal.workout_type == WorkoutType.INTERVALS
    check = next(item for item in normal.rule_trace if item.rule_id == "post_illness_quality_check")
    assert check.fired is False
    assert check.facts["hidden_normal_run_requirement"] == 0


def test_a_check_in_carries_no_free_text_field() -> None:
    """The note went to one overwritten slot that nothing read. Rejecting it
    outright beats accepting input the app silently discards."""
    with pytest.raises(ValidationError):
        RecommendationRequest(health_status=CurrentHealthStatus.NORMAL, notes="my knee hurts")


def test_reporting_illness_with_the_buttons_does_change_the_prescription() -> None:
    result = _recommend(_state(), status=CurrentHealthStatus.SICK_OR_RECOVERING)
    assert result.workout_type == WorkoutType.RECOVERY
    assert any("not medical clearance" in warning for warning in result.warnings)


def test_high_rpe_on_easy_run_adds_recovery_caution() -> None:
    result = _recommend(
        _state(
            days_since_last_run=1,
            last_run=_difficulty(rpe=7),
            last_run_workout_type=WorkoutType.EASY,
        )
    )
    assert result.workout_type == WorkoutType.EASY
    trace = next(item for item in result.rule_trace if item.rule_id == "recent_high_rpe")
    assert trace.fired


def test_ready_next_day_easy_distance_recovers_continuously_after_long_run() -> None:
    state = _state(
        as_of=datetime(2026, 8, 29, 19, tzinfo=timezone.utc),
        days_since_last_run=1.0,
        last_run=_difficulty(long=True, miles=6.4),
        last_run_workout_type=WorkoutType.LONG,
    )

    next_day = recommend_next_run(
        state,
        RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
        CONFIG,
        weekly_role="easy",
    )
    recovered = recommend_next_run(
        state.model_copy(update={"days_since_last_run": 3.0}),
        RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
        CONFIG,
        weekly_role="easy",
    )
    recovery_trace = next(
        item
        for item in next_day.rule_trace
        if item.rule_id == "recent_recovery_load"
    )

    assert next_day.readiness.value == "ready"
    assert recovery_trace.facts["easy_recovery_pressure"] == 0
    assert recovery_trace.facts["easy_volume_recovery_pressure"] > 0
    # The latest run is removed from its own comparison baseline. Recovery is
    # then driven by its actual distance, duration, and HR load—not by the
    # long-run label—so the next-day adjustment remains continuous.
    assert next_day.distance_range_miles == (3.0, 3.5)
    assert recovered.distance_range_miles == (4.0, 5.0)
    assert any("remaining recovery load reduces" in reason for reason in next_day.reasons)


def test_recent_hilly_run_counts_as_mechanical_load_without_inventing_hr_points() -> None:
    result = _recommend(
        _state(
            days_since_last_run=1.8,
            last_run=_difficulty(flags=["hilly_session"]),
        )
    )
    trace = next(item for item in result.rule_trace if item.rule_id == "mechanical_load")
    assert trace.fired
    assert result.workout_type == WorkoutType.EASY


def test_high_acute_load_avoids_added_volume_orquality() -> None:
    state = _state(recent_load=_state().recent_load.model_copy(update={"acute_to_prior_ratio": 1.5}))
    result = _recommend(state)
    assert result.workout_type == WorkoutType.EASY
    assert any(item.rule_id == "high_recent_load" and item.fired for item in result.rule_trace)


def test_continuous_fatigue_replaces_boxcar_overload_when_available() -> None:
    load = _state().recent_load.model_copy(
        update={
            "acute_distance_to_capacity_ratio": 1.45,
            "continuous_fatigue_miles": 16.0,
            "continuous_fatigue_to_capacity_ratio": 1.0,
            "capacity_reference_miles": 16.0,
        }
    )

    result = _recommend(_state(recent_load=load))
    high_load = next(
        item for item in result.rule_trace if item.rule_id == "high_recent_load"
    )

    assert high_load.fired is False
    assert high_load.facts["effective_load_ratio"] == 1.0


def test_short_term_distance_density_can_add_proportional_load_caution() -> None:
    load = _state().recent_load.model_copy(
        update={
            "acute_distance_to_capacity_ratio": 1.45,
            "continuous_fatigue_to_capacity_ratio": 1.0,
            "continuous_short_term_distance_miles": 20.0,
            "capacity_reference_miles": 16.0,
        }
    )

    result = _recommend(_state(recent_load=load))
    high_load = next(
        item for item in result.rule_trace if item.rule_id == "high_recent_load"
    )

    assert high_load.facts["effective_load_ratio"] == 1.25


def test_high_confidence_hr_load_disagreement_adds_caution_to_mileage_capacity() -> None:
    load = _state().recent_load.model_copy(
        update={
            "acute_to_prior_ratio": 2.2,
            "acute_distance_to_capacity_ratio": 1.1,
            "capacity_reference_miles": 16.0,
        }
    )
    result = _recommend(_state(recent_load=load))
    high_load = next(item for item in result.rule_trace if item.rule_id == "high_recent_load")
    assert high_load.fired is True
    assert high_load.facts["effective_load_ratio"] == 1.35
    assert 0 < high_load.facts["surplus_strength"] < 0.5
    assert result.workout_type != WorkoutType.REST


def test_low_confidence_hr_load_disagreement_cannot_take_over_mileage_capacity() -> None:
    load = _state().recent_load.model_copy(
        update={
            "acute_to_prior_ratio": 2.2,
            "acute_distance_to_capacity_ratio": 1.1,
            "capacity_reference_miles": 16.0,
            "confidence": ConfidenceLevel.LOW,
        }
    )

    result = _recommend(_state(recent_load=load))
    high_load = next(
        item for item in result.rule_trace if item.rule_id == "high_recent_load"
    )

    assert high_load.fired is False
    assert high_load.facts["effective_load_ratio"] == 1.21


def test_synthetic_hr_load_does_not_penalize_planned_week_twice() -> None:
    load = _state().recent_load.model_copy(
        update={
            "acute_to_prior_ratio": 2.2,
            "acute_distance_to_capacity_ratio": 1.1,
            "capacity_reference_miles": 16.0,
            "flags": ["includes_planned_sessions"],
        }
    )

    result = _recommend(_state(recent_load=load))
    high_load = next(
        item for item in result.rule_trace if item.rule_id == "high_recent_load"
    )

    assert high_load.fired is False
    assert high_load.facts["effective_load_ratio"] == 1.1


def test_long_run_uses_rough_110_percent_reference_with_practical_rounding() -> None:
    load = _state().recent_load.model_copy(
        update={
            "trailing_28d": _window(28, 120, 1600),
            "capacity_reference_miles": 30,
        }
    )
    result = _recommend(_state(days_since_long_run=12, longest_run_30d_miles=8, recent_load=load))
    assert result.workout_type == WorkoutType.LONG
    # Progression targets 5% while the separate 10% value remains a ceiling.
    assert result.distance_range_miles == (8.25, 8.75)
    assert "rounded" in result.warnings[0]


def test_small_single_run_base_is_not_forced_into_a_long_run_label() -> None:
    """No global distance convention may manufacture a long-run role."""
    load = _state().recent_load.model_copy(
        update={
            "trailing_28d": _window(28, 40, 500),
            "capacity_reference_miles": 10,
        }
    )
    result = _recommend(_state(days_since_long_run=12, longest_run_30d_miles=3, recent_load=load))
    assert result.workout_type != WorkoutType.LONG
    long_trace = next(
        item for item in result.rule_trace if item.rule_id == "long_run_eligible"
    )
    assert long_trace.fired is False
    assert long_trace.facts["progression_ceiling_miles"] < (
        long_trace.facts["meaningful_long_threshold_miles"]
    )


def test_historical_long_capacity_accelerates_return_without_replacing_guardrail() -> None:
    load = _state().recent_load.model_copy(
        update={"capacity_reference_miles": 18.0}
    )
    result = _recommend(
        _state(
            days_since_long_run=12,
            longest_run_30d_miles=6.4,
            retained_long_run_capacity_miles=8.2,
            recent_load=load,
        )
    )

    assert result.workout_type == WorkoutType.LONG
    assert result.distance_range_miles == (7.0, 7.5)


def test_completed_long_progression_can_grow_instead_of_following_decay_down() -> None:
    load = _state().recent_load.model_copy(
        update={"capacity_reference_miles": 30.0}
    )
    first = _recommend(
        _state(
            days_since_long_run=12,
            longest_run_30d_miles=8.0,
            retained_long_run_capacity_miles=8.2,
            recent_load=load,
        )
    )
    assert first.distance_range_miles is not None
    second = _recommend(
        _state(
            days_since_long_run=12,
            longest_run_30d_miles=first.distance_range_miles[1],
            retained_long_run_capacity_miles=first.distance_range_miles[1],
            recent_load=load,
        )
    )

    assert first.distance_range_miles == (8.25, 8.75)
    assert second.distance_range_miles == (9.0, 9.5)


def test_long_run_needs_capacity_to_exceed_ordinary_easy_distance() -> None:
    result = _recommend(
        _state(
            longest_run_30d_miles=2.0,
            days_since_long_run=30.0,
            days_since_quality_run=2.0,
        )
    )

    trace = next(
        item for item in result.rule_trace if item.rule_id == "long_run_eligible"
    )
    assert trace.facts["progression_ceiling_miles"] < trace.facts[
        "meaningful_long_threshold_miles"
    ]
    assert result.workout_type != WorkoutType.LONG


def test_easy_run_zone_instruction_follows_configured_zones() -> None:
    config = {**CONFIG, "zones": {"z2": [132, 147]}}
    result = _recommend(_state(days_since_quality_run=1, days_since_long_run=1), config=config)
    assert result.workout_type == WorkoutType.EASY
    assert any("132–147 bpm" in step.instruction for step in result.structure)


def test_quality_session_types_rotate_and_respect_disabled_settings() -> None:
    settings = {
        **CONFIG,
        "coaching": {
            **CONFIG["coaching"],
            "quality_sessions": {
                "fartlek": False,
                "short_intervals": False,
                "long_intervals": False,
                "threshold": False,
                "progression": True,
                "hill_repeats": False,
            },
        },
    }
    result = recommend_next_run(
        _state(days_since_quality_run=8),
        RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
        settings,
    )
    assert result.quality_session_type == "progression"
    assert result.workout_type == WorkoutType.TEMPO_THRESHOLD


def test_long_quality_gap_favors_adaptable_fartlek_not_forced_intervals() -> None:
    result = _recommend(
        _state(
            days_since_quality_run=30,
            completed_quality_session_count=0,
            days_since_long_run=3,
        )
    )

    assert result.quality_session_type == "fartlek"
    assert result.title == "Fartlek by feel"
    assert any(
        "landmarks" in step.instruction and "8 controlled surges" in step.instruction
        for step in result.structure
    )
    assert any("single prescribed structure" in rule for rule in result.modification_rules)


def test_quality_workout_recommends_one_structure_not_a_menu() -> None:
    result = _recommend(
        _state(
            days_since_quality_run=8,
            completed_quality_session_count=2,
            days_since_long_run=3,
        )
    )

    assert result.quality_session_type == "threshold"
    instructions = " ".join(step.instruction for step in result.structure)
    assert "Run 18 minutes continuously" in instructions
    assert "Choose" not in instructions
    assert " or " not in instructions


def test_shortened_quality_scales_work_dose_instead_of_deleting_quality() -> None:
    settings = {
        **CONFIG,
        "coaching": {
            **CONFIG["coaching"],
            "quality_sessions": {
                "fartlek": False,
                "short_intervals": False,
                "long_intervals": False,
                "threshold": True,
                "progression": False,
                "hill_repeats": False,
            },
        },
    }
    original = _recommend(
        _state(days_since_quality_run=12, days_since_long_run=3),
        config=settings,
    )

    shortened = scale_quality_session(
        original,
        original.distance_range_miles,
        (3.0, 3.5),
    )

    assert shortened.workout_type == WorkoutType.TEMPO_THRESHOLD
    assert shortened.quality_session_type == "threshold"
    assert shortened.structure[1].duration_minutes == 16
    assert "Run 16 minutes continuously" in shortened.structure[1].instruction
    assert any("instead of deleting" in reason for reason in shortened.reasons)


def test_only_two_hour_quality_sessions_expand_to_multi_part_structure() -> None:
    ordinary = _recommend(
        _state(
            days_since_quality_run=8,
            completed_quality_session_count=2,
            days_since_long_run=3,
        )
    )

    assert structure_extended_quality_session(ordinary, 45.0) == ordinary

    allocated = structure_extended_quality_session(ordinary, 60.0)
    assert allocated == ordinary
    assert allocated.distance_range_miles == (3.5, 4.0)
    assert not any(
        "full prescribed distance" in step.instruction
        for step in allocated.structure
    )

    extended = structure_extended_quality_session(ordinary, 150.0)
    instructions = " ".join(step.instruction for step in extended.structure)
    assert extended.title == "Extended aerobic session with threshold blocks"
    assert "3 × 12 minutes" in instructions
    assert "150 minutes total" in instructions
    assert "Do not add more quality work" in instructions
    assert any("quality dose is capped" in reason for reason in extended.reasons)


def test_fixed_time_quality_distance_uses_athlete_pace_without_padding() -> None:
    settings = {
        **CONFIG,
        "coaching": {
            **CONFIG["coaching"],
            "quality_sessions": {
                "fartlek": False,
                "short_intervals": True,
                "long_intervals": False,
                "threshold": False,
                "progression": False,
                "hill_repeats": False,
            },
        },
    }
    slower = _recommend(
        _state(
            days_since_quality_run=8,
            standardized_pace_at_target_hr=PaceValue(
                minutes_per_mile=12.0,
                display="12:00/mi",
            ),
        ),
        config=settings,
    )
    faster = _recommend(
        _state(
            days_since_quality_run=8,
            standardized_pace_at_target_hr=PaceValue(
                minutes_per_mile=9.0,
                display="9:00/mi",
            ),
        ),
        config=settings,
    )

    assert sum(faster.distance_range_miles) > sum(slower.distance_range_miles)
    assert not any(
        "full prescribed distance" in step.instruction
        for result in (slower, faster)
        for step in result.structure
    )


def test_general_fitness_quality_stimuli_rotate_without_random_plan_churn() -> None:
    first = _recommend(
        _state(
            days_since_quality_run=8,
            completed_quality_session_count=0,
            days_since_long_run=3,
        )
    )
    second = _recommend(
        _state(
            days_since_quality_run=8,
            completed_quality_session_count=1,
            days_since_long_run=3,
        )
    )

    assert first.quality_session_type == "fartlek"
    assert second.quality_session_type == "progression"
    assert _recommend(
        _state(
            days_since_quality_run=8,
            completed_quality_session_count=0,
            days_since_long_run=3,
        )
    ).quality_session_type == first.quality_session_type


def test_long_run_recency_is_outweighed_by_high_load() -> None:
    high = _state().recent_load.model_copy(update={"acute_to_prior_ratio": 1.5})
    result = _recommend(_state(days_since_long_run=14, recent_load=high))
    assert result.workout_type == WorkoutType.EASY
    scoring = next(item for item in result.rule_trace if item.rule_id == "workout_scoring")
    assert scoring.facts["easy_score"] > scoring.facts["long_score"]


def test_interval_readiness_does_not_require_five_mile_long_run() -> None:
    result = _recommend(_state(longest_run_30d_miles=3.5, days_since_long_run=3))
    assert result.workout_type == WorkoutType.INTERVALS
    quality = next(item for item in result.rule_trace if item.rule_id == "quality_eligible")
    assert quality.facts["longest_run_is_gate"] is False


def test_missing_gps_keeps_load_but_lowers_default_confidence() -> None:
    result = _recommend(_state(data_quality_flags=["latest_run_pace_quality_low"], days_since_quality_run=1, days_since_long_run=3))
    assert result.workout_type == WorkoutType.EASY
    assert result.confidence == ConfidenceLevel.LOW


def test_recent_hard_workout_is_not_followed_byquality() -> None:
    result = _recommend(_state(days_since_quality_run=1, days_since_long_run=3))
    assert result.workout_type != WorkoutType.INTERVALS


def test_reported_pain_overrides_training_state() -> None:
    pain = _recommend(_state(), CurrentHealthStatus.PAIN_OR_INJURY_CONCERN)
    assert pain.workout_type == WorkoutType.REST


def test_recommendation_api_persists_request_state_and_rule_trace(tmp_path: Path) -> None:
    _write_config(tmp_path)
    database = tmp_path / "data" / "test.sqlite"
    with connect(database) as connection:
        initialize(connection)
    response = TestClient(create_app(tmp_path)).post(
        "/api/recommendation",
        json={
            "health_status": "normal",
            "planned_at": (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat(),
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["rule_trace"]
    assert payload["workout_type"] == "easy"
    with connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM recommendation_history").fetchone()[0] == 1


def test_recommendation_api_requires_a_planned_time(tmp_path: Path) -> None:
    _write_config(tmp_path)
    database = tmp_path / "data" / "test.sqlite"
    with connect(database) as connection:
        initialize(connection)
    response = TestClient(create_app(tmp_path)).post(
        "/api/recommendation", json={"health_status": "normal"}
    )
    assert response.status_code == 422
    assert "planned date and time" in response.json()["detail"]


def test_goal_progress_never_raises_where_the_validator_would() -> None:
    """A read-out that throws when the goal is unset is useless in exactly the
    situation an athlete most wants an answer."""
    from run_analysis.race_goals import GoalStatus, goal_progress

    assert goal_progress(None, {}).status == GoalStatus.NO_GOAL
    selected_but_undated = {"coaching": {"training_goal": "marathon"}}
    result = goal_progress(None, selected_but_undated)
    assert result.status == GoalStatus.NO_GOAL
    assert "no date set" in result.headline.lower()


def test_goal_progress_reports_thin_history_rather_than_guessing() -> None:
    from datetime import date as _date

    from run_analysis.race_goals import GoalStatus, goal_progress

    config = {
        "coaching": {
            "training_goal": "10k",
            "goal_date": "2027-01-01",
            "goal_pace_min_mile": 9.0,
        }
    }
    result = goal_progress(None, config, as_of=_date(2026, 8, 8))
    assert result.status == GoalStatus.INSUFFICIENT_EVIDENCE
    assert result.weeks_remaining is not None


def test_the_progress_read_out_and_the_validator_agree_on_supported_pace() -> None:
    """Two implementations of "what your running supports" would eventually
    disagree in front of the athlete."""
    from statistics import median

    from run_analysis.race_goals import RACE_GOALS, supported_goal_pace

    profile = RACE_GOALS["10k"]
    performances = [(9.0 + index * 0.1, 4.0) for index in range(10)]
    expected = median(
        sorted(
            (pace * distance * (profile.distance_miles / distance) ** 1.06
             + profile.prediction_penalty_minutes) / profile.distance_miles
            for pace, distance in performances
        )[:3]
    )
    assert supported_goal_pace(performances, profile) == pytest.approx(expected)


def test_a_zero_distance_run_cannot_crash_the_goal_read_out() -> None:
    """goal_progress promises never to raise; Riegel divides by distance."""
    from run_analysis.race_goals import RACE_GOALS, supported_goal_pace

    profile = RACE_GOALS["5k"]
    with pytest.raises(ValueError):
        supported_goal_pace([(9.0, 0.0)], profile)
    # A usable run alongside a bad one still produces an answer.
    assert supported_goal_pace([(9.0, 0.0), (9.0, 3.0)], profile) > 0
