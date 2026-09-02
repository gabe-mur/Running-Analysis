from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import json

import pytest

from run_analysis.db import connect, initialize
from run_analysis.fitness_state import _performance_anomaly
from run_analysis.progress import _trend_evidence_weight, build_progress
from run_analysis.web.schemas import FitnessPoint, WorkoutType


def _insert_run(connection, index: int, start: datetime, miles: float, minutes: float, zone_load: float) -> int:
    cursor = connection.execute(
        """
        INSERT INTO activities(
            activity_uid,activity_id,sport,start_time_utc,start_time_utc_epoch,total_distance_m,
            lap_count,trackpoint_count,gps_quality,hr_quality,elevation_quality,cadence_quality,
            distance_source,namespaces_json,data_quality_json,created_at_utc,updated_at_utc
        ) VALUES (?,?, 'Running',?,?,?,0,0,'gps_complete','hr_complete','complete','missing',
                  'device','{}','{}','now','now')
        """,
        (f"uid-{index}", f"external-{index}", start.isoformat(), start.timestamp(), miles * 1609.344),
    )
    activity_id = int(cursor.lastrowid)
    zone_seconds = {"z1": minutes * 30, "z2": minutes * 30, "z3": 0, "z4": 0, "z5": 0, "unknown": 0}
    connection.execute(
        """
        INSERT INTO activity_metrics(
            activity_id,metrics_json,calculated_at_utc,calculated_moving_time_s,
            hr_zone_seconds_json,diagnostics_json,session_zone_load,easy_minutes,
            moderate_minutes,hard_minutes,hr_load_coverage
        ) VALUES (?, '{}','now',?,?, '{}',?,?,0,0,1)
        """,
        (activity_id, minutes * 60, json.dumps(zone_seconds), zone_load, minutes),
    )
    return activity_id


def test_illness_recovery_runs_have_65_percent_fitness_trend_weight() -> None:
    assert _trend_evidence_weight("illness_recovery", WorkoutType.EASY) == pytest.approx(0.65)
    assert _trend_evidence_weight("illness", WorkoutType.EASY) == pytest.approx(0.25)


def test_quality_runs_do_not_influence_the_steady_aerobic_trend() -> None:
    strict = {
        "effective_window_count": 6,
        "reference_time_support": "interpolation",
        "steady_aerobic_benchmark": {
            "selection_quality": "strict_observed",
            "window_evidence_weight": 1.0,
        },
    }
    estimated = {
        "effective_window_count": 6,
        "reference_time_support": "interpolation",
        "steady_aerobic_benchmark": {
            "selection_quality": "estimated_fixed_time",
            "window_evidence_weight": 0.1,
        },
    }

    assert _trend_evidence_weight("normal", WorkoutType.TEMPO_THRESHOLD, strict) == 0
    assert _trend_evidence_weight("normal", WorkoutType.TEMPO_THRESHOLD, estimated) == 0
    assert _trend_evidence_weight("normal", WorkoutType.INTERVALS, estimated) == 0
    assert _trend_evidence_weight("normal", WorkoutType.RACE, strict) == 0
    assert _trend_evidence_weight("normal", WorkoutType.HIKE, strict) == 0


def test_progress_shows_a_scored_tempo_run_as_context_only(tmp_path: Path) -> None:
    anchor = datetime.now(timezone.utc)
    with connect(tmp_path / "tempo-progress.sqlite") as connection:
        initialize(connection)
        activity_id = _insert_run(connection, 1, anchor - timedelta(days=1), 4.4, 44, 100)
        connection.execute(
            "INSERT INTO run_overrides(activity_id,workout_type,health_tag) "
            "VALUES ('external-1','tempo_threshold','normal')"
        )
        connection.execute(
            "INSERT INTO model_runs(activity_id,model_name,model_version,result_json) VALUES (?,?,?,?)",
            (
                activity_id,
                "standardized_pace_at_target_hr",
                "test",
                json.dumps(
                    {
                        "raw_pace_at_target_hr_min_mile": 10.8,
                        "standardized_pace_at_target_hr_min_mile": 10.6,
                        "uncertainty_95_min_mile": 0.65,
                        "effective_window_count": 7.5,
                        "reference_time_support": "interpolation",
                        "steady_aerobic_benchmark": {
                            "selection_quality": "estimated_fixed_time",
                            "window_evidence_weight": 0.1,
                            "raw_pace_at_target_hr_min_mile": 10.7,
                            "standardized_pace_at_target_hr_min_mile": 10.5,
                            "uncertainty_95_min_mile": 1.0,
                        },
                    }
                ),
            ),
        )
        connection.commit()
        progress = build_progress(connection, 28)

    point = progress.series[0]
    coverage = progress.activity_coverage[0]
    assert point.included_in_trend is False
    assert point.trend_weight == 0
    assert coverage.score_status == "context_only"
    assert coverage.trend_weight == pytest.approx(point.trend_weight)
    assert "0% influence" in coverage.reason
    assert progress.current_pace is None


def test_quality_progress_does_not_depend_on_an_aerobic_model_result(tmp_path: Path) -> None:
    anchor = datetime.now(timezone.utc)
    with connect(tmp_path / "quality-progress.sqlite") as connection:
        initialize(connection)
        activity_id = _insert_run(connection, 1, anchor - timedelta(days=1), 4.4, 44, 100)
        connection.execute(
            "INSERT INTO run_overrides(activity_id,workout_type,health_tag) "
            "VALUES ('external-1','tempo_threshold','normal')"
        )
        for lap_index, (seconds, distance, average_hr) in enumerate(
            ((720, 1600, 140), (1080, 2900, 168), (840, 1800, 150))
        ):
            connection.execute(
                """
                INSERT INTO laps(
                    activity_id,lap_index,total_time_s,distance_m,
                    average_hr_bpm,maximum_hr_bpm
                ) VALUES (?,?,?,?,?,?)
                """,
                (activity_id, lap_index, seconds, distance, average_hr, average_hr + 5),
            )
        connection.commit()

        progress = build_progress(
            connection,
            28,
            config={"target_hr": 145, "zones": {"z3": [151, 166]}},
        )

    assert len(progress.quality_performance) == 1
    quality = progress.quality_performance[0]
    assert quality.duration_minutes == 18
    assert quality.source == "recorded_lap_2"
    assert len(progress.series) == 1
    context = progress.series[0]
    assert context.activity_id == activity_id
    assert context.standardized_pace_min_mile is None
    assert context.context_pace_min_mile == pytest.approx(10.0)
    assert context.trend_weight == 0
    assert context.included_in_trend is False
    coverage = progress.activity_coverage[0]
    assert coverage.score_status == "context_only"
    assert "0% influence" in coverage.reason
    assert progress.current_pace is None


def test_manually_excluded_run_walk_remains_visible_at_zero_weight(
    tmp_path: Path,
) -> None:
    anchor = datetime.now(timezone.utc)
    with connect(tmp_path / "excluded-run-walk-progress.sqlite") as connection:
        initialize(connection)
        activity_id = _insert_run(
            connection,
            1,
            anchor - timedelta(days=1),
            2.5,
            35,
            100,
        )
        connection.execute(
            "INSERT INTO run_overrides("
            "activity_id,include_in_model,workout_type,health_tag"
            ") VALUES ('external-1',0,'run_walk','other_abnormal')"
        )
        connection.commit()

        progress = build_progress(connection, 28)

    assert len(progress.series) == 1
    point = progress.series[0]
    assert point.activity_id == activity_id
    assert point.workout_type == WorkoutType.RUN_WALK
    assert point.standardized_pace_min_mile is None
    assert point.context_pace_min_mile == pytest.approx(14.0)
    assert point.trend_weight == 0
    assert point.included_in_trend is False
    coverage = progress.activity_coverage[0]
    assert coverage.score_status == "context_only"
    assert "0% influence" in coverage.reason
    assert progress.current_pace is None


def test_progress_uses_run_analysis_fallback_workout_types(tmp_path: Path) -> None:
    anchor = datetime.now(timezone.utc)
    with connect(tmp_path / "progress-workout-types.sqlite") as connection:
        initialize(connection)
        easy_id = _insert_run(
            connection,
            1,
            anchor - timedelta(days=2),
            3.5,
            38,
            90,
        )
        long_id = _insert_run(
            connection,
            2,
            anchor - timedelta(days=1),
            8.0,
            88,
            190,
        )
        connection.commit()

        progress = build_progress(connection, 28)

    coverage_types = {
        item.activity_id: item.workout_type for item in progress.activity_coverage
    }
    series_types = {item.activity_id: item.workout_type for item in progress.series}
    assert coverage_types[easy_id] == WorkoutType.EASY
    assert coverage_types[long_id] == WorkoutType.LONG
    assert series_types[easy_id] == WorkoutType.EASY
    assert series_types[long_id] == WorkoutType.LONG


def test_graph_only_context_point_cannot_break_performance_anomaly() -> None:
    anchor = datetime.now(timezone.utc)
    scored = [
        FitnessPoint(
            activity_id=index,
            start_time=anchor - timedelta(days=8 - index),
            raw_pace_min_mile=10.0,
            standardized_pace_min_mile=10.0,
            uncertainty_95_min_mile=0.2,
            distance_miles=4.0,
            workout_type=WorkoutType.EASY,
        )
        for index in range(1, 5)
    ]
    context = FitnessPoint(
        activity_id=5,
        start_time=anchor,
        raw_pace_min_mile=14.0,
        context_pace_min_mile=14.0,
        standardized_pace_min_mile=None,
        uncertainty_95_min_mile=0.0,
        distance_miles=2.5,
        workout_type=WorkoutType.RUN_WALK,
        included_in_trend=False,
        trend_weight=0.0,
    )

    assert _performance_anomaly([*scored, context]) == "within_recent_range"


def test_progress_keeps_pace_volume_and_intensity_as_separate_dimensions(tmp_path: Path) -> None:
    anchor = datetime.now(timezone.utc)
    with connect(tmp_path / "progress.sqlite") as connection:
        initialize(connection)
        ids = [
            _insert_run(connection, index, anchor - timedelta(days=day), miles, minutes, load)
            for index, (day, miles, minutes, load) in enumerate(
                [(5, 8, 90, 150), (12, 2, 20, 40), (19, 5, 55, 100), (26, 4, 45, 80)],
                start=1,
            )
        ]
        for activity_id, pace in zip(ids, (10.5, 10.2, 10.4, 10.3)):
            connection.execute(
                "INSERT INTO model_runs(activity_id,model_name,model_version,result_json) VALUES (?,?,?,?)",
                (
                    activity_id,
                    "standardized_pace_at_target_hr",
                    "test",
                    json.dumps(
                        {
                            "raw_pace_at_target_hr_min_mile": pace + 0.2,
                            "standardized_pace_at_target_hr_min_mile": pace,
                            "uncertainty_95_min_mile": 0.15,
                        }
                    ),
                ),
            )
        connection.commit()
        progress = build_progress(connection, 28)
        short_progress = build_progress(connection, 14)
    assert progress.period_comparison.current.distance_miles == pytest.approx(19)
    assert progress.current_load.trailing_28d.zone_load == pytest.approx(370)
    assert progress.current_load.capacity_reference_miles is not None
    assert progress.current_pace is not None
    assert len(progress.series) == 4
    assert len(short_progress.series) == 2
    assert all(
        point.start_time > short_progress.as_of - timedelta(days=14)
        for point in short_progress.series
    )
    assert progress.trend_28d
    assert progress.consistency.longest_run_miles == 8
    assert progress.intensity.easy_percent == 100


def test_progress_reports_unknown_hr_instead_of_imputing_load(tmp_path: Path) -> None:
    anchor = datetime.now(timezone.utc)
    with connect(tmp_path / "progress.sqlite") as connection:
        initialize(connection)
        activity_id = _insert_run(connection, 1, anchor - timedelta(days=2), 5, 50, 100)
        connection.execute(
            "UPDATE activity_metrics SET session_zone_load=NULL,hr_zone_seconds_json=? WHERE activity_id=?",
            (json.dumps({"z2": 300, "unknown": 2700}), activity_id),
        )
        connection.commit()
        progress = build_progress(connection, 28)
    assert progress.current_load.trailing_28d.zone_load is None
    assert progress.current_load.confidence.value == "low"
    assert progress.intensity.missing_hr_minutes > 0


def test_device_distance_fitness_estimate_is_not_labeled_unscored(tmp_path: Path) -> None:
    anchor = datetime.now(timezone.utc)
    with connect(tmp_path / "progress.sqlite") as connection:
        initialize(connection)
        activity_id = _insert_run(connection, 1, anchor - timedelta(days=2), 5, 50, 100)
        connection.execute(
            "INSERT INTO run_overrides(activity_id,workout_type,health_tag) VALUES ('external-1','easy','illness_recovery')"
        )
        connection.execute(
            "INSERT INTO model_runs(activity_id,model_name,model_version,result_json) VALUES (?,?,?,?)",
            (
                activity_id,
                "standardized_pace_at_target_hr",
                "test",
                json.dumps(
                    {
                        "raw_pace_at_target_hr_min_mile": 11.2,
                        "standardized_pace_at_target_hr_min_mile": 11.0,
                        "uncertainty_95_min_mile": 0.8,
                        "estimate_quality": "device_distance_fallback",
                        "gps_coverage_fraction": 0,
                        "fallback_uncertainty_95_min_mile": 0.5,
                        "steady_aerobic_benchmark": {
                            "raw_pace_at_target_hr_min_mile": 11.1,
                            "standardized_pace_at_target_hr_min_mile": 10.9,
                            "uncertainty_95_min_mile": 1.0,
                            "selection_quality": "estimated_fixed_time",
                            "estimate_quality": "device_distance_fallback",
                        },
                    }
                ),
            ),
        )
        connection.commit()
        progress = build_progress(connection, 28)

    coverage = progress.activity_coverage[0]
    assert coverage.score_status == "uncertain_estimate"
    assert coverage.standardized_pace_min_mile == 11.0
    assert coverage.trend_weight == pytest.approx(0.65)
    assert "Estimated from Garmin distance" in coverage.reason
    assert progress.steady_aerobic.series[0].benchmark_quality == "estimated_fixed_time"
    assert progress.steady_aerobic.series[0].measurement_quality == "device_distance_fallback"


def test_progress_copy_follows_the_configured_comparison_heart_rate(tmp_path: Path) -> None:
    """A changed comparison HR must show up in what the app tells the athlete."""
    anchor = datetime.now(timezone.utc)
    with connect(tmp_path / "target_hr.sqlite") as connection:
        initialize(connection)
        ids = [
            _insert_run(connection, index, anchor - timedelta(days=day), 5, 55, 100)
            for index, day in enumerate([3, 8, 14, 21], start=1)
        ]
        for activity_id, pace in zip(ids, (10.5, 10.2, 10.4, 10.3)):
            connection.execute(
                "INSERT INTO model_runs(activity_id,model_name,model_version,result_json) VALUES (?,?,?,?)",
                (
                    activity_id,
                    "standardized_pace_at_target_hr",
                    "test",
                    json.dumps(
                        {
                            "raw_pace_at_target_hr_min_mile": pace + 0.2,
                            "standardized_pace_at_target_hr_min_mile": pace,
                            "uncertainty_95_min_mile": 0.15,
                        }
                    ),
                ),
            )
        connection.commit()
        progress = build_progress(
            connection,
            28,
            config={"target_hr": 147, "reference_conditions": {"within_run_minutes": 18}},
        )
    assert progress.target_hr_bpm == 147
    assert "147 bpm" in progress.definition
    assert "145" not in progress.definition
    assert "147 bpm" in progress.steady_aerobic.definition
    assert "minute 18" in progress.steady_aerobic.definition


def test_a_pre_rename_database_upgrades_without_losing_any_scored_run(tmp_path: Path) -> None:
    """The 145-to-target-HR rename touches a column, two model_name values, and
    the stored JSON keys. Miss any one and every run silently reads as
    'no reliable aerobic windows' rather than failing loudly."""
    from run_analysis.db import SCHEMA_VERSION

    path = tmp_path / "legacy.sqlite"
    anchor = datetime.now(timezone.utc)
    with connect(path) as connection:
        initialize(connection)
        activity_ids = [
            _insert_run(connection, index, anchor - timedelta(days=day), 5, 55, 100)
            for index, day in enumerate([3, 8, 14, 21], start=1)
        ]
        # Rewind the schema to what it looked like before the rename.
        connection.execute(
            "ALTER TABLE activity_metrics "
            "RENAME COLUMN standardized_pace_at_target_hr_min_mile TO standardized_pace_145_min_mile"
        )
        for activity_id, pace in zip(activity_ids, (10.5, 10.2, 10.4, 10.3)):
            connection.execute(
                "INSERT INTO model_runs(activity_id,model_name,model_version,result_json) VALUES (?,?,?,?)",
                (
                    activity_id,
                    "standardized_pace_145",
                    "legacy",
                    json.dumps(
                        {
                            "raw_pace_145_min_mile": pace + 0.2,
                            "standardized_pace_145_min_mile": pace,
                            "uncertainty_95_min_mile": 0.15,
                            "steady_aerobic_benchmark": {
                                "standardized_pace_145_min_mile": pace,
                                "raw_pace_145_min_mile": pace + 0.2,
                                "uncertainty_95_min_mile": 0.2,
                                "selection_quality": "strict_observed",
                            },
                        }
                    ),
                ),
            )
        connection.execute(
            "INSERT INTO model_metadata(model_name,model_version,fitted_at_utc,metadata_json) VALUES (?,?,?,?)",
            ("standardized_pace_145", "legacy", anchor.isoformat(), "{}"),
        )
        connection.commit()

    # Reopening runs the migrations, exactly as every endpoint does per request.
    with connect(path) as connection:
        initialize(connection)
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(activity_metrics)")}
        assert "standardized_pace_at_target_hr_min_mile" in columns
        assert "standardized_pace_145_min_mile" not in columns
        assert connection.execute(
            "SELECT COUNT(*) FROM model_runs WHERE model_name='standardized_pace_145'"
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM model_runs WHERE result_json LIKE '%pace_145_min_mile%'"
        ).fetchone()[0] == 0
        assert int(
            connection.execute(
                "SELECT value FROM schema_metadata WHERE key='schema_version'"
            ).fetchone()[0]
        ) == SCHEMA_VERSION

        progress = build_progress(connection, 28, config={"target_hr": 145})

    # All four runs still score, and the coverage table does not claim they are
    # unusable.
    assert len(progress.series) == 4
    assert progress.current_pace is not None
    assert progress.steady_aerobic.series
    statuses = {item.score_status for item in progress.activity_coverage}
    assert "unscored" not in statuses, statuses
    assert all(
        item.standardized_pace_min_mile is not None for item in progress.activity_coverage
    )
