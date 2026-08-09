from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from run_analysis.recommendation import recommend_next_run
from run_analysis.web.schemas import (
    ConfidenceLevel,
    CurrentHealthStatus,
    FitnessState,
    FitnessTrend,
    LoadContext,
    LoadWindow,
    RecommendationRequest,
    SessionDifficulty,
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
        "minimum_days_between_quality_sessions": 4,
        "minimum_running_days_28d_for_quality": 8,
        "long_run_recency_reference_days": 10,
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
        recent_performance_anomaly="within_recent_range",
    )
    values.update(changes)
    return FitnessState(**values)


def _recommend(state: FitnessState, status=CurrentHealthStatus.NORMAL, config=None):
    return recommend_next_run(
        state, RecommendationRequest(health_status=status), config or CONFIG
    )


def test_low_load_three_days_rest_and_no_recent_quality_can_be_quality_eligible() -> None:
    result = _recommend(_state(days_since_quality_run=12, days_since_long_run=3))
    assert result.workout_type == WorkoutType.INTERVALS
    assert result.planned_for is not None
    assert any(item.rule_id == "planned_timing" and item.fired for item in result.rule_trace)
    assert any(item.rule_id == "quality_eligible" and item.fired for item in result.rule_trace)


def test_quality_is_prioritized_weekly_not_every_four_days() -> None:
    result = _recommend(
        _state(
            days_since_quality_run=5,
            quality_sessions_14d=1,
            days_since_long_run=3,
        )
    )
    assert result.workout_type != WorkoutType.INTERVALS
    quality = next(item for item in result.rule_trace if item.rule_id == "quality_eligible")
    assert quality.facts["quality_recency_reference_days"] == 7


def test_long_run_yesterday_produces_rest_or_short_recovery() -> None:
    result = _recommend(_state(days_since_last_run=0.8, last_run=_difficulty(long=True, miles=8)))
    assert result.workout_type in {WorkoutType.REST, WorkoutType.RECOVERY}
    assert any(item.rule_id == "long_or_hard_yesterday" and item.fired for item in result.rule_trace)


def test_high_z3_leakage_forces_easy_z1_z2() -> None:
    result = _recommend(_state(moderate_fraction_14d=0.24))
    assert result.workout_type == WorkoutType.EASY
    assert result.target_zones == ["Z1", "Z2"]
    assert result.readiness.value == "caution"
    assert "24%" in result.readiness_reason
    assert "17%" in result.readiness_reason
    assert "Keep this run truly easy" in result.readiness_reason


def test_recent_illness_and_poor_response_reduce_easy_volume() -> None:
    result = _recommend(_state(recent_illness_or_recovery=True, normal_runs_since_health_event=1, recent_performance_anomaly="unusually_costly"))
    assert result.workout_type == WorkoutType.EASY
    assert result.distance_range_miles[1] < 5


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


def test_depressed_raw_hr_norm_does_not_override_normal_retained_mileage_capacity() -> None:
    load = _state().recent_load.model_copy(
        update={
            "acute_to_prior_ratio": 2.2,
            "acute_distance_to_capacity_ratio": 1.1,
            "capacity_reference_miles": 16.0,
        }
    )
    result = _recommend(_state(recent_load=load))
    high_load = next(item for item in result.rule_trace if item.rule_id == "high_recent_load")
    assert high_load.fired is False
    assert result.workout_type == WorkoutType.INTERVALS


def test_long_run_uses_rough_110_percent_reference_with_practical_rounding() -> None:
    load = _state().recent_load.model_copy(
        update={
            "trailing_28d": _window(28, 120, 1600),
            "capacity_reference_miles": 30,
        }
    )
    result = _recommend(_state(days_since_long_run=12, longest_run_30d_miles=8, recent_load=load))
    assert result.workout_type == WorkoutType.LONG
    assert result.distance_range_miles[1] == 9.0
    assert "rounded" in result.warnings[0]


def test_long_run_progression_cap_outranks_the_conventional_five_mile_floor() -> None:
    """A three-mile base must never be prescribed a five-mile long run."""
    load = _state().recent_load.model_copy(
        update={
            "trailing_28d": _window(28, 40, 500),
            "capacity_reference_miles": 10,
        }
    )
    result = _recommend(_state(days_since_long_run=12, longest_run_30d_miles=3, recent_load=load))
    assert result.workout_type == WorkoutType.LONG
    # 3 miles * 1.10 = 3.3, rounded to a practical half mile.
    assert result.distance_range_miles == (3.0, 3.5)
    assert any("progression limit" in warning for warning in result.warnings)


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
