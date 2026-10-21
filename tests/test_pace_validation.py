from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest
import yaml

from run_analysis import objective_modeling as model
from run_analysis.db import connect, initialize
from run_analysis.pace_validation import validate_pace_normalization
from test_progress import _insert_run


def synthetic_rows(runs=30):
    rows = []
    for run in range(runs):
        for index, hr in enumerate((142, 150, 144, 148, 141, 152)):
            minute = 10 + index * 5
            speed = 2.3 + run * 0.01 + 0.02 * (hr - 145)
            rows.append(dict(activity_id=run + 1, start_time_utc=(datetime(2026, 1, 1, tzinfo=timezone.utc)+timedelta(days=run)).isoformat(),
                             moving_minutes_into_run=minute, average_hr_bpm=hr,
                             moving_pace_min_mile=model.speed_to_pace_min_mile(speed), grade_energy_ratio=1.0))
    return rows


def test_validation_recovers_known_hr_relationship_without_between_run_fitness_credit():
    result = validate_pace_normalization(synthetic_rows())
    assert result["held_out_runs"] == 15
    assert result["mean_absolute_error_seconds_per_mile"]["hr"] < 1e-8
    assert result["comparisons"]["hr"]["run_bootstrap_95"][1] < 0
    assert result == validate_pace_normalization(synthetic_rows())


def test_validation_requires_run_level_support_and_independent_variation():
    assert validate_pace_normalization([])["status"] == "insufficient_evidence"
    rows = synthetic_rows()
    for row in rows:
        row["average_hr_bpm"] = 145
    assert validate_pace_normalization(rows)["status"] == "insufficient_evidence"


@pytest.mark.parametrize("grade,temperature,weather_estimated", [(0.5, 30, False), (2.0, 100, True)])
def test_primary_score_has_no_hidden_credits_or_calibration_dependency(monkeypatch, tmp_path, grade, temperature, weather_estimated):
    cfg = yaml.safe_load((Path(__file__).parents[1]/"config.example.yaml").read_text())
    rows = []
    with connect(tmp_path/"observed.sqlite") as connection:
        initialize(connection)
        for run in range(10):
            start = datetime(2026, 1, 1, tzinfo=timezone.utc)+timedelta(days=run)
            activity_id = _insert_run(connection, run, start, 4, 40, 100)
            for index in range(6):
                minute = 10 + index * 5
                rows.append(dict(activity_id=activity_id, external_activity_id=f"external-{run}", start_time_utc=start.isoformat(),
                                 moving_minutes_into_run=minute, average_hr_bpm=145, moving_pace_min_mile=10.0,
                                 heart_rate_change_bpm=0, heart_rate_range_bpm=0, speed_change_fraction=0,
                                 stopped_time_s=0, elapsed_time_s=120, grade_energy_ratio=grade,
                                 temperature_f=temperature, weather_location_estimated=weather_estimated,
                                 gps_complete_fraction=1.0, workout_type="easy"))
        monkeypatch.setattr(model, "load_model_window_sets", lambda c, conf, sizes: ({s:rows for s in sizes}, {}))
        monkeypatch.setattr(model, "load_overlapping_model_windows", lambda *a, **kw: (rows, {}))
        monkeypatch.setattr(model, "_prior_corrected_speed", lambda *a: pytest.fail("hidden calibration correction"))
        monkeypatch.setattr(model, "_estimate_heat_posterior", lambda *a: pytest.fail("hidden heat credit"))
        metadata = model.fit_published_reference_model(connection, cfg, tmp_path/"model.json")
        assert metadata["method_kind"] == "observed_comparable"
        assert not metadata["grade_supported"] and not metadata["weather_supported"]
        assert metadata["heat_response"] == []
        results = [json.loads(r[0]) for r in connection.execute("SELECT result_json FROM model_runs")]
        assert len(results) == 10
        for result in results:
            assert result["standardized_pace_at_target_hr_min_mile"] == pytest.approx(10)
            assert result["observed_average_hr_bpm"] == 145
            assert not any(result["contributions_min_mile"].values())
            assert all(e["confidence"] == "unavailable" for e in result["adjustment_evidence"].values())
            assert result["steady_aerobic_benchmark"]["standardized_pace_at_target_hr_min_mile"] == pytest.approx(10)


def test_pre_quality_filter_handles_empty_aerobic_selection():
    assert model._pre_quality_indexes([], [], object()) == []
