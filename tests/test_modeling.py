from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from run_analysis.modeling import FeatureBuilder, _cross_validate, fit_ridge, grouped_folds


def config() -> dict:
    return yaml.safe_load((Path(__file__).parents[1] / "config.example.yaml").read_text())


def test_grouped_folds_never_split_a_run() -> None:
    groups = [1, 1, 1, 2, 2, 3, 3, 4, 4, 5]
    for train, test in grouped_folds(groups, 3):
        train_groups = {groups[index] for index in train}
        test_groups = {groups[index] for index in test}
        assert train_groups.isdisjoint(test_groups)


def test_ridge_recovers_simple_standardized_prediction() -> None:
    matrix = np.asarray([[1, -1], [1, 0], [1, 1], [1, 2]], dtype=float)
    response = np.asarray([11, 10, 9, 8], dtype=float)
    model = fit_ridge(matrix, response, alpha=0)
    assert model.predict(np.asarray([[1, 0]], dtype=float))[0] == pytest.approx(10)


def test_weather_model_improves_grouped_prediction_when_signal_is_real() -> None:
    settings = config()
    settings["model"]["ridge_alpha"] = 0.01
    rows = []
    for run in range(20):
        temperature = 50 + run * 2
        for segment in range(4):
            hr = 140 + segment * 3
            pace = 9.0 - (hr - 145) * 0.02 + max(0, temperature - 60) * 0.03
            rows.append(
                {
                    "activity_id": run,
                    "moving_pace_min_mile": pace,
                    "average_hr_bpm": hr,
                    "moving_minutes_into_run": 10 + segment * 10,
                    "average_grade_percent": 0,
                    "temperature_f": temperature,
                    "dewpoint_f": 45,
                    "wind_speed_mph": 3,
                    "headwind_signed_mph": 0,
                    "crosswind_mph": 0,
                    "previous_7d_miles": 10,
                    "previous_28d_miles": 40,
                    "days_since_previous_run": 2,
                    "days_since_previous_hard_run": 5,
                }
            )
    simple = _cross_validate(rows, "C", settings)
    weather = _cross_validate(rows, "D", settings)
    assert weather["mae_seconds_per_mile"] < simple["mae_seconds_per_mile"] - 2


def test_reference_features_match_standardized_conditions() -> None:
    settings = config()
    builder = FeatureBuilder("D", settings).fit([])
    reference = {
        "average_hr_bpm": 145,
        "moving_minutes_into_run": settings["reference_conditions"]["within_run_minutes"],
        "average_grade_percent": 0,
        "temperature_f": 55,
        "dewpoint_f": 45,
    }
    vector = builder.transform_row(reference)
    assert vector[0] == 1
    assert all(value == 0 for value in vector[1:])
