from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import json

import pytest

from run_analysis.db import connect, initialize
from run_analysis.progress import _trend_evidence_weight, build_progress
from run_analysis.web.schemas import WorkoutType


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
                    "standardized_pace_145",
                    "test",
                    json.dumps(
                        {
                            "raw_pace_145_min_mile": pace + 0.2,
                            "standardized_pace_145_min_mile": pace,
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
                "standardized_pace_145",
                "test",
                json.dumps(
                    {
                        "raw_pace_145_min_mile": 11.2,
                        "standardized_pace_145_min_mile": 11.0,
                        "uncertainty_95_min_mile": 0.8,
                        "estimate_quality": "device_distance_fallback",
                        "gps_coverage_fraction": 0,
                        "fallback_uncertainty_95_min_mile": 0.5,
                        "steady_aerobic_benchmark": {
                            "raw_pace_145_min_mile": 11.1,
                            "standardized_pace_145_min_mile": 10.9,
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
    assert "inverse-variance" in coverage.reason
    assert progress.steady_aerobic.series[0].benchmark_quality == "estimated_fixed_time"
    assert progress.steady_aerobic.series[0].measurement_quality == "device_distance_fallback"
