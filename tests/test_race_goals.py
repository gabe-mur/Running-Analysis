from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import sqlite3

import pytest

from run_analysis.db import connect, initialize
from run_analysis.race_goals import assess_race_goal
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


def _insert_runs(connection: sqlite3.Connection, count: int = 10, distance_miles: float = 5) -> None:
    start_index = int(connection.execute("SELECT COUNT(*) FROM activities").fetchone()[0])
    for index in range(count):
        run_index = start_index + index
        cursor = connection.execute(
            """
            INSERT INTO activities(
                activity_uid,activity_id,sport,start_time_utc,start_time_utc_epoch,
                total_distance_m,lap_count,trackpoint_count,gps_quality,hr_quality,
                elevation_quality,cadence_quality,distance_source,namespaces_json,
                data_quality_json,created_at_utc,updated_at_utc
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                f"uid-{run_index}", f"activity-{run_index}", "Running",
                f"2026-07-{run_index + 1:02d}T12:00:00+00:00", 1_750_000_000 + run_index,
                distance_miles * 1609.344, 1, 100, "gps_complete", "hr_complete",
                "elevation_complete", "cadence_complete", "device", "{}", "{}", "now", "now",
            ),
        )
        connection.execute(
            """
            INSERT INTO activity_metrics(
                activity_id,metrics_json,calculated_at_utc,moving_pace_min_mile,
                analysis_distance_m,calculated_moving_time_s
            ) VALUES (?,?,?,?,?,?)
            """,
            (
                cursor.lastrowid, "{}", "now", 9.0 + index * 0.1,
                distance_miles * 1609.344, (9.0 + index * 0.1) * distance_miles * 60,
            ),
        )
    connection.commit()


def _config(goal: str, goal_date: date, pace: float) -> dict:
    return {
        "coaching": {
            "training_goal": goal,
            "goal_date": goal_date.isoformat(),
            "goal_pace_min_mile": pace,
        }
    }


def test_goal_uses_exactly_ten_recent_performances(tmp_path) -> None:
    today = date(2026, 8, 7)
    with connect(tmp_path / "runs.sqlite") as connection:
        initialize(connection)
        _insert_runs(connection)
        result = assess_race_goal(
            connection,
            _config("10k", today + timedelta(weeks=12), 9.0),
            as_of=today,
        )
    assert result is not None
    assert result.evidence_runs == 10
    assert result.recent_fast_training_pace == pytest.approx(9.1)
    expected = ((9.1 * 5) * (6.21371 / 5) ** 1.06) / 6.21371
    assert result.supported_goal_pace == pytest.approx(expected)
    assert result.fastest_allowed_goal_pace == pytest.approx(expected * 0.97)


def test_goal_rejects_too_few_runs_unreasonable_pace_and_short_timeline(tmp_path) -> None:
    today = date(2026, 8, 7)
    with connect(tmp_path / "runs.sqlite") as connection:
        initialize(connection)
        _insert_runs(connection, 9)
        with pytest.raises(ValueError, match="requires 10 usable"):
            assess_race_goal(
                connection, _config("5k", today + timedelta(weeks=12), 8.0), as_of=today
            )
        _insert_runs(connection, 1)
        with pytest.raises(ValueError, match="choose .* or slower"):
            assess_race_goal(
                connection, _config("10k", today + timedelta(weeks=12), 6.0), as_of=today
            )
        with pytest.raises(ValueError, match="move the race to .* or later"):
            assess_race_goal(
                connection, _config("10k", today + timedelta(weeks=12), 8.8), as_of=today
            )
        with pytest.raises(ValueError, match="at least 12 weeks"):
            assess_race_goal(
                connection, _config("10k", today + timedelta(weeks=2), 9.0), as_of=today
            )


def _state(as_of: datetime) -> FitnessState:
    window = lambda days: LoadWindow(days=days, distance_miles=16, moving_minutes=170, zone_load=300, hard_minutes=4, activity_count=4)
    return FitnessState(
        as_of=as_of,
        window_days=28,
        fitness_trend=FitnessTrend.STABLE,
        trend_confidence=ConfidenceLevel.MODERATE,
        recent_load=LoadContext(
            trailing_7d=window(7), trailing_14d=window(14), trailing_28d=window(28),
            acute_to_prior_ratio=1.0, confidence=ConfidenceLevel.HIGH,
        ),
        days_since_last_run=3,
        days_since_quality_run=8,
        days_since_long_run=3,
        last_run=SessionDifficulty(
            distance_miles=4, moving_minutes=44, elapsed_minutes=44, stopped_minutes=0,
            zone_load=70, zone_breakdown=ZoneBreakdown(),
        ),
        longest_run_30d_miles=8,
        running_days_28d=10,
        moderate_fraction_14d=0.1,
        moderate_evidence_runs_14d=3,
        recent_performance_anomaly="within_recent_range",
    )


def test_goal_changes_quality_variant_and_race_day_prescription() -> None:
    as_of = datetime(2026, 8, 7, 12, tzinfo=timezone.utc)
    config = {
        "coaching": {
            "training_goal": "10k",
            "goal_date": "2026-10-30",
            "goal_pace_min_mile": 8.0,
            "quality_sessions": {
                "short_intervals": True, "long_intervals": True, "threshold": True,
                "progression": True, "hill_repeats": False,
            },
        }
    }
    result = recommend_next_run(
        _state(as_of), RecommendationRequest(health_status=CurrentHealthStatus.NORMAL), config
    )
    assert result.quality_session_type is not None
    assert result.quality_session_type.value == "long_intervals"
    assert any("8:00/mi" in step.instruction for step in result.structure)
    assert next(item for item in result.rule_trace if item.rule_id == "race_goal").fired

    race_config = {
        "coaching": {
            **config["coaching"],
            "goal_date": as_of.date().isoformat(),
        }
    }
    race = recommend_next_run(
        _state(as_of), RecommendationRequest(health_status=CurrentHealthStatus.NORMAL), race_config
    )
    assert race.workout_type == WorkoutType.RACE
    assert race.distance_range_miles == pytest.approx((6.21371, 6.21371))
