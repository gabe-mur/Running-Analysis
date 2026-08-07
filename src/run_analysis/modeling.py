"""Published-method HR/speed modeling with grouped and temporal validation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any
import hashlib
import json
import math
import sqlite3

import numpy as np

from .physiology import estimated_shade_wbgt_f, grade_energy_ratio
from .model_windows import load_model_window_sets

MODEL_VERSION = "aerobic-v2-science-anchored"
MODEL_ORDER = ("A", "B", "C", "D", "D+grade", "E", "E+grade")
POSITION_MODELS = {"B", "C", "D", "D+grade", "E", "E+grade"}
GRADE_MODELS = {"C", "D+grade", "E+grade"}
WEATHER_MODELS = {"D", "D+grade", "E", "E+grade"}
WIND_LOAD_MODELS = {"E", "E+grade"}
METERS_PER_MILE = 1609.344


class InsufficientModelDataError(ValueError):
    """The local history cannot yet support a defensible fitted score."""


def pace_to_speed_mps(pace_min_mile: float) -> float:
    return METERS_PER_MILE / (float(pace_min_mile) * 60.0)


def speed_to_pace_min_mile(speed_mps: float) -> float:
    if speed_mps <= 0:
        return float("inf")
    return METERS_PER_MILE / (float(speed_mps) * 60.0)


@dataclass(slots=True)
class RidgeModel:
    coefficients: np.ndarray
    covariance: np.ndarray
    residual_variance: float
    alpha: float

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        return matrix @ self.coefficients


@dataclass(slots=True)
class ModelFitSummary:
    window_count: int
    run_count: int
    selected_model: str
    weather_supported: bool
    wind_load_supported: bool
    output_path: str


def grouped_folds(groups: list[int], fold_count: int) -> list[tuple[np.ndarray, np.ndarray]]:
    unique, counts = np.unique(np.asarray(groups), return_counts=True)
    if len(unique) < 2:
        raise ValueError("Grouped validation requires at least two runs")
    folds = min(fold_count, len(unique))
    assignments: list[list[int]] = [[] for _ in range(folds)]
    loads = [0] * folds
    for group, count in sorted(zip(unique.tolist(), counts.tolist()), key=lambda item: (-item[1], item[0])):
        target = min(range(folds), key=lambda index: (loads[index], index))
        assignments[target].append(group)
        loads[target] += count
    group_array = np.asarray(groups)
    return [
        (np.flatnonzero(~np.isin(group_array, test_groups)), np.flatnonzero(np.isin(group_array, test_groups)))
        for test_groups in assignments
    ]


def chronological_folds(rows: list[dict[str, Any]], fold_count: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Hold out contiguous blocks of complete runs as a seasonal/time sensitivity test."""
    first_date: dict[int, str] = {}
    for row in rows:
        group = int(row["activity_id"])
        first_date[group] = min(first_date.get(group, row["start_time_utc"]), row["start_time_utc"])
    ordered = [group for group, _ in sorted(first_date.items(), key=lambda item: item[1])]
    blocks = [block.tolist() for block in np.array_split(np.asarray(ordered), min(fold_count, len(ordered))) if len(block)]
    groups = np.asarray([int(row["activity_id"]) for row in rows])
    return [
        (np.flatnonzero(~np.isin(groups, block)), np.flatnonzero(np.isin(groups, block)))
        for block in blocks
    ]


def fit_ridge(matrix: np.ndarray, response: np.ndarray, alpha: float) -> RidgeModel:
    penalty = np.eye(matrix.shape[1]) * alpha
    penalty[0, 0] = 0.0
    gram = matrix.T @ matrix
    inverse = np.linalg.pinv(gram + penalty)
    coefficients = inverse @ matrix.T @ response
    residuals = response - matrix @ coefficients
    degrees = max(1, len(response) - matrix.shape[1])
    variance = float(residuals @ residuals / degrees)
    covariance = variance * inverse @ gram @ inverse
    return RidgeModel(coefficients, covariance, variance, alpha)


class FeatureBuilder:
    """Nested features plus the Minetti physical grade transform for C-E."""

    def __init__(self, model_name: str, config: dict, medians: dict[str, float] | None = None):
        self.model_name = model_name
        self.config = config
        self.medians = medians or {}

    def fit(self, rows: list[dict[str, Any]]) -> "FeatureBuilder":
        for name in (
            "previous_7d_miles",
            "previous_28d_miles",
            "days_since_previous_run",
            "days_since_previous_hard_run",
            "headwind_signed_mph",
            "crosswind_mph",
        ):
            values = [float(row[name]) for row in rows if row.get(name) is not None]
            self.medians[name] = median(values) if values else 0.0
        return self

    @property
    def uses_grade_energy_transform(self) -> bool:
        return self.model_name in GRADE_MODELS

    @property
    def names(self) -> list[str]:
        names = ["intercept", "hr_delta_10bpm"]
        if self.model_name in POSITION_MODELS:
            names += ["position_delta_10min"]
        if self.model_name in WEATHER_MODELS:
            names += ["shade_wbgt_delta_10f"]
        if self.model_name in WIND_LOAD_MODELS:
            names += [
                "wind_speed_delta_5mph",
                "signed_headwind_5mph",
                "crosswind_5mph",
                "previous_7d_miles_10",
                "previous_28d_miles_30",
                "days_since_run_7",
                "days_since_hard_run_14",
            ]
        return names

    def _value(self, row: dict[str, Any], name: str) -> float:
        value = row.get(name)
        return float(self.medians.get(name, 0.0) if value is None else value)

    def _shade_wbgt(self, row: dict[str, Any]) -> float:
        return estimated_shade_wbgt_f(
            float(row["temperature_f"]),
            relative_humidity_percent=float(row["relative_humidity_percent"])
            if row.get("relative_humidity_percent") is not None
            else None,
            dewpoint_f=float(row["dewpoint_f"]) if row.get("dewpoint_f") is not None else None,
        )

    def transform_row(self, row: dict[str, Any]) -> list[float]:
        target_hr = float(self.config["target_hr"])
        reference = self.config["reference_conditions"]
        hr = (float(row["average_hr_bpm"]) - target_hr) / 10.0
        features = [1.0, hr]
        if self.model_name in POSITION_MODELS:
            position = (float(row["moving_minutes_into_run"]) - float(reference["within_run_minutes"])) / 10.0
            features += [position]
        if self.model_name in WEATHER_MODELS:
            reference_wbgt = estimated_shade_wbgt_f(
                float(reference["temperature_f"]), dewpoint_f=float(reference["dewpoint_f"])
            )
            wbgt_delta = (self._shade_wbgt(row) - reference_wbgt) / 10.0
            features += [wbgt_delta]
        if self.model_name in WIND_LOAD_MODELS:
            features += [
                (float(row["wind_speed_mph"]) - float(reference["wind_mph"])) / 5.0,
                self._value(row, "headwind_signed_mph") / 5.0,
                self._value(row, "crosswind_mph") / 5.0,
                self._value(row, "previous_7d_miles") / 10.0,
                self._value(row, "previous_28d_miles") / 30.0,
                self._value(row, "days_since_previous_run") / 7.0,
                self._value(row, "days_since_previous_hard_run") / 14.0,
            ]
        return features

    def transform(self, rows: list[dict[str, Any]]) -> np.ndarray:
        return np.asarray([self.transform_row(row) for row in rows], dtype=float)

    def response_row(self, row: dict[str, Any]) -> float:
        speed = pace_to_speed_mps(float(row["moving_pace_min_mile"]))
        if self.uses_grade_energy_transform:
            ratio = row.get("grade_energy_ratio")
            speed *= float(ratio) if ratio is not None else grade_energy_ratio(
                float(row["average_grade_percent"]),
                float(self.config["elevation"]["maximum_plausible_grade_percent"]),
            )
        return speed

    def response(self, rows: list[dict[str, Any]]) -> np.ndarray:
        return np.asarray([self.response_row(row) for row in rows], dtype=float)

    def predicted_raw_paces(self, model: RidgeModel, rows: list[dict[str, Any]]) -> np.ndarray:
        modeled_speeds = model.predict(self.transform(rows))
        paces = []
        for speed, row in zip(modeled_speeds, rows):
            raw_speed = float(speed)
            if self.uses_grade_energy_transform:
                ratio = row.get("grade_energy_ratio")
                raw_speed /= float(ratio) if ratio is not None else grade_energy_ratio(
                    float(row["average_grade_percent"]),
                    float(self.config["elevation"]["maximum_plausible_grade_percent"]),
                )
            paces.append(speed_to_pace_min_mile(raw_speed))
        return np.asarray(paces, dtype=float)


def _load_model_rows(
    connection: sqlite3.Connection, config: dict
) -> tuple[list[dict[str, Any]], dict[str, int], dict[int, list[dict[str, Any]]]]:
    sizes = tuple(int(value) for value in config["model"]["window_sensitivity_seconds"])
    window_sets, diagnostics = load_model_window_sets(connection, config, sizes)
    primary = int(config["model"]["window_seconds"])
    return window_sets[primary], diagnostics[primary], window_sets


def _cross_validate(
    rows: list[dict[str, Any]], model_name: str, config: dict, strategy: str = "grouped"
) -> dict:
    groups = [int(row["activity_id"]) for row in rows]
    actual_paces = np.asarray([float(row["moving_pace_min_mile"]) for row in rows], dtype=float)
    predictions = np.empty_like(actual_paces)
    folds = (
        grouped_folds(groups, int(config["model"]["grouped_cv_folds"]))
        if strategy == "grouped"
        else chronological_folds(rows, int(config["model"]["grouped_cv_folds"]))
    )
    folds_output = []
    for fold_index, (train, test) in enumerate(folds):
        train_rows = [rows[index] for index in train]
        test_rows = [rows[index] for index in test]
        builder = FeatureBuilder(model_name, config).fit(train_rows)
        model = fit_ridge(
            builder.transform(train_rows), builder.response(train_rows), float(config["model"]["ridge_alpha"])
        )
        predicted = builder.predicted_raw_paces(model, test_rows)
        predictions[test] = predicted
        errors = actual_paces[test] - predicted
        folds_output.append(
            {
                "fold": fold_index + 1,
                "train_runs": len(set(groups[index] for index in train)),
                "test_runs": len(set(groups[index] for index in test)),
                "mae_seconds_per_mile": float(np.mean(np.abs(errors)) * 60),
                "rmse_seconds_per_mile": float(np.sqrt(np.mean(errors**2)) * 60),
            }
        )
    errors = actual_paces - predictions
    return {
        "mae_seconds_per_mile": float(np.mean(np.abs(errors)) * 60),
        "rmse_seconds_per_mile": float(np.sqrt(np.mean(errors**2)) * 60),
        "folds": folds_output,
        "residual_mean_seconds_per_mile": float(np.mean(errors) * 60),
        "residual_std_seconds_per_mile": float(np.std(errors) * 60),
    }


def _accepted(
    baseline: str,
    candidate: str,
    grouped: dict[str, dict],
    temporal: dict[str, dict],
    threshold: float,
) -> tuple[bool, float, float]:
    grouped_gain = grouped[baseline]["mae_seconds_per_mile"] - grouped[candidate]["mae_seconds_per_mile"]
    temporal_gain = temporal[baseline]["mae_seconds_per_mile"] - temporal[candidate]["mae_seconds_per_mile"]
    accepted = (
        grouped_gain >= threshold
        and temporal_gain >= threshold
        and grouped[candidate]["rmse_seconds_per_mile"] <= grouped[baseline]["rmse_seconds_per_mile"]
        and temporal[candidate]["rmse_seconds_per_mile"] <= temporal[baseline]["rmse_seconds_per_mile"]
    )
    return accepted, grouped_gain, temporal_gain


def _select_model(
    grouped: dict[str, dict], temporal: dict[str, dict], config: dict
) -> tuple[str, bool, bool, bool, list[str]]:
    threshold = float(config["model"]["minimum_cv_improvement_seconds_per_mile"])
    selected = "A"
    decisions = []
    accepted, group_gain, time_gain = _accepted("A", "B", grouped, temporal, threshold)
    decisions.append(
        f"B {'accepted' if accepted else 'rejected'}: drift vs A grouped/time-blocked MAE gains "
        f"{group_gain:.2f}/{time_gain:.2f} s/mi"
    )
    if accepted:
        selected = "B"
    grade_supported, group_gain, time_gain = _accepted(selected, "C", grouped, temporal, threshold)
    decisions.append(
        f"C {'accepted' if grade_supported else 'rejected'}: Minetti grade vs {selected} grouped/time-blocked "
        f"MAE gains {group_gain:.2f}/{time_gain:.2f} s/mi"
    )
    if grade_supported:
        selected = "C"
    weather_candidate = "D+grade" if grade_supported else "D"
    weather_supported, group_gain, time_gain = _accepted(
        selected, weather_candidate, grouped, temporal, threshold
    )
    decisions.append(
        f"{weather_candidate} {'accepted' if weather_supported else 'rejected'}: shade-WBGT weather vs "
        f"{selected} grouped/time-blocked "
        f"MAE gains {group_gain:.2f}/{time_gain:.2f} s/mi"
    )
    if weather_supported:
        selected = weather_candidate
    wind_candidate = "E+grade" if grade_supported else "E"
    wind_load_supported, group_gain, time_gain = _accepted(
        selected, wind_candidate, grouped, temporal, threshold
    )
    wind_load_supported = weather_supported and wind_load_supported
    decisions.append(
        f"{wind_candidate} {'accepted' if wind_load_supported else 'rejected'}: wind/load grouped/time-blocked "
        f"MAE gains {group_gain:.2f}/{time_gain:.2f} s/mi"
    )
    if wind_load_supported:
        selected = wind_candidate
    return selected, grade_supported, weather_supported, wind_load_supported, decisions


def _reference_row(builder: FeatureBuilder, config: dict) -> dict[str, float]:
    reference = config["reference_conditions"]
    reference_humidity = None
    return {
        **builder.medians,
        "average_hr_bpm": float(config["target_hr"]),
        "moving_minutes_into_run": float(reference["within_run_minutes"]),
        "average_grade_percent": float(reference["grade_percent"]),
        "temperature_f": float(reference["temperature_f"]),
        "dewpoint_f": float(reference["dewpoint_f"]),
        "relative_humidity_percent": reference_humidity,
        "wind_speed_mph": float(reference["wind_mph"]),
        "headwind_signed_mph": 0.0,
        "crosswind_mph": 0.0,
        "moving_pace_min_mile": 10.0,
    }


def _mean_predicted_pace(model: RidgeModel, builder: FeatureBuilder, rows: list[dict]) -> float:
    return float(np.mean(builder.predicted_raw_paces(model, rows)))


def _counterfactual_contributions(
    model: RidgeModel, builder: FeatureBuilder, rows: list[dict], reference: dict
) -> dict[str, float]:
    current = [dict(row) for row in rows]
    prior = _mean_predicted_pace(model, builder, current)
    contributions: dict[str, float] = {}
    steps = [
        ("hr_normalization", {"average_hr_bpm": reference["average_hr_bpm"]}),
        (
            "weather_adjustment",
            {
                "temperature_f": reference["temperature_f"],
                "dewpoint_f": reference["dewpoint_f"],
                "relative_humidity_percent": None,
            },
        ),
        (
            "wind_adjustment",
            {"wind_speed_mph": reference["wind_speed_mph"], "headwind_signed_mph": 0.0, "crosswind_mph": 0.0},
        ),
        ("grade_adjustment", {"average_grade_percent": reference["average_grade_percent"]}),
        ("within_run_adjustment", {"moving_minutes_into_run": reference["moving_minutes_into_run"]}),
        (
            "workload_adjustment",
            {
                name: value
                for name, value in builder.medians.items()
                if name.startswith("previous_") or name.startswith("days_since_")
            },
        ),
    ]
    for name, replacements in steps:
        for row in current:
            row.update(replacements)
        prediction = _mean_predicted_pace(model, builder, current)
        contributions[name] = prediction - prior
        prior = prediction
    return contributions


def _heat_table(model: RidgeModel, builder: FeatureBuilder, reference: dict, rows: list[dict]) -> list[dict]:
    base_vector = np.asarray(builder.transform_row(reference))
    base_speed = float(base_vector @ model.coefficients)
    base_pace = speed_to_pace_min_mile(base_speed)
    output = []
    for temperature in (55, 60, 65, 70, 75, 80, 85, 90):
        for dewpoint in (45, 60, 70):
            scenario = dict(
                reference,
                temperature_f=float(temperature),
                dewpoint_f=float(dewpoint),
                relative_humidity_percent=None,
            )
            vector = np.asarray(builder.transform_row(scenario))
            scenario_speed = float(vector @ model.coefficients)
            penalty = speed_to_pace_min_mile(scenario_speed) - base_pace
            delta = vector - base_vector
            speed_se = math.sqrt(max(0.0, float(delta @ model.covariance @ delta)))
            pace_derivative = METERS_PER_MILE / (60.0 * max(0.1, scenario_speed) ** 2)
            supporting_runs = len(
                {
                    row["activity_id"]
                    for row in rows
                    if abs(row["temperature_f"] - temperature) <= 3
                    and abs(row["dewpoint_f"] - dewpoint) <= 5
                }
            )
            output.append(
                {
                    "temperature_f": temperature,
                    "dewpoint_f": dewpoint,
                    "pace_penalty_seconds_per_mile": penalty * 60,
                    "uncertainty_95_seconds_per_mile": 1.96 * speed_se * pace_derivative * 60,
                    "supporting_runs_near_conditions": supporting_runs,
                    "coverage": "weak" if supporting_runs < 5 else "moderate" if supporting_runs < 15 else "good",
                }
            )
    return output


def _pace_uncertainty_95(speed: float, standard_error: float) -> float:
    low_speed = max(0.1, speed - 1.96 * standard_error)
    high_speed = max(0.1, speed + 1.96 * standard_error)
    return (speed_to_pace_min_mile(low_speed) - speed_to_pace_min_mile(high_speed)) / 2.0


def _fit_data_selected_models(
    connection: sqlite3.Connection, config: dict, output_path: str | Path
) -> ModelFitSummary:
    rows, reliability, window_sets = _load_model_rows(connection, config)
    if len({row["activity_id"] for row in rows}) < 10:
        raise ValueError("At least 10 eligible runs with reliable steady-state segments are required")
    grouped_validation = {name: _cross_validate(rows, name, config, "grouped") for name in MODEL_ORDER}
    temporal_validation = {name: _cross_validate(rows, name, config, "temporal") for name in MODEL_ORDER}
    window_sensitivity = {}
    for seconds, sensitivity_rows in sorted(window_sets.items()):
        if len({row["activity_id"] for row in sensitivity_rows}) < 10:
            continue
        window_sensitivity[str(seconds)] = {
            "window_count": len(sensitivity_rows),
            "run_count": len({row["activity_id"] for row in sensitivity_rows}),
            "grouped_model_b": _cross_validate(sensitivity_rows, "B", config, "grouped"),
            "time_blocked_model_b": _cross_validate(sensitivity_rows, "B", config, "temporal"),
        }
    selected, grade_supported, weather_supported, wind_load_supported, decisions = _select_model(
        grouped_validation, temporal_validation, config
    )
    builder = FeatureBuilder(selected, config).fit(rows)
    matrix = builder.transform(rows)
    response = builder.response(rows)
    model = fit_ridge(matrix, response, float(config["model"]["ridge_alpha"]))
    modeled_predictions = model.predict(matrix)
    speed_residuals = response - modeled_predictions
    predicted_paces = builder.predicted_raw_paces(model, rows)
    actual_paces = np.asarray([float(row["moving_pace_min_mile"]) for row in rows])
    pace_residuals = actual_paces - predicted_paces
    reference = _reference_row(builder, config)
    reference_vector = np.asarray(builder.transform_row(reference))
    reference_speed = float(reference_vector @ model.coefficients)
    reference_variance = float(reference_vector @ model.covariance @ reference_vector)

    grouped_rows: dict[int, list[int]] = {}
    for index, row in enumerate(rows):
        grouped_rows.setdefault(int(row["activity_id"]), []).append(index)
    raw_run_effects = [float(np.mean(speed_residuals[indexes])) for indexes in grouped_rows.values()]
    maximum_effective = float(config["model"]["run_effect_max_effective_segments"])
    effective_sizes = [min(maximum_effective, len(indexes)) for indexes in grouped_rows.values()]
    sigma2 = model.residual_variance
    between_variance = float(np.var(raw_run_effects, ddof=1)) if len(raw_run_effects) > 1 else 0.0
    tau2 = max(1e-8, between_variance - float(np.mean([sigma2 / size for size in effective_sizes])))

    version_payload = {
        "model_version": MODEL_VERSION,
        "selected": selected,
        "features": builder.names,
        "medians": builder.medians,
        "config": config["model"],
    }
    version = hashlib.sha256(json.dumps(version_payload, sort_keys=True).encode()).hexdigest()[:16]
    connection.execute("DELETE FROM model_runs WHERE model_name='standardized_pace_145'")
    connection.execute(
        """UPDATE activity_metrics SET standardized_pace_145_min_mile=NULL,
           standardized_pace_uncertainty_min_mile=NULL,raw_aerobic_efficiency_min_mile=NULL,
           environmental_adjustment_min_mile=NULL,selected_model_name=NULL,
           selected_model_version=NULL"""
    )
    for activity_id, indexes in grouped_rows.items():
        run_rows = [rows[index] for index in indexes]
        raw_effect = float(np.mean(speed_residuals[indexes]))
        effective_n = min(maximum_effective, len(indexes))
        shrinkage = tau2 / (tau2 + sigma2 / effective_n)
        effect = raw_effect * shrinkage
        posterior_variance = 1.0 / (1.0 / tau2 + effective_n / sigma2)
        speed_uncertainty = math.sqrt(max(0.0, posterior_variance + reference_variance))
        standardized_speed = max(0.1, reference_speed + effect)
        standardized_pace = speed_to_pace_min_mile(standardized_speed)
        pace_uncertainty = _pace_uncertainty_95(standardized_speed, speed_uncertainty)
        contributions = _counterfactual_contributions(model, builder, run_rows, reference)
        observed = float(np.mean(actual_paces[indexes]))
        average_hr = float(np.mean([row["average_hr_bpm"] for row in run_rows]))
        raw_efficiency = observed * average_hr / float(config["target_hr"])
        environmental = contributions["weather_adjustment"] + contributions["wind_adjustment"]
        result = {
            "activity_id": run_rows[0]["external_activity_id"],
            "observed_segment_pace_min_mile": observed,
            "standardized_pace_145_min_mile": standardized_pace,
            "uncertainty_95_min_mile": pace_uncertainty,
            "raw_run_effect_mps": raw_effect,
            "shrunk_run_effect_mps": effect,
            "shrinkage": shrinkage,
            "segment_count": len(indexes),
            "contributions_min_mile": contributions,
            "environmental_adjustment_min_mile": environmental,
            "interpretation": "observational statistical adjustment anchored to published transforms; not causal",
        }
        connection.execute(
            "INSERT INTO model_runs(activity_id,model_name,model_version,result_json) VALUES (?,?,?,?)",
            (activity_id, "standardized_pace_145", version, json.dumps(result)),
        )
        connection.execute(
            """
            UPDATE activity_metrics SET standardized_pace_145_min_mile=?,
                standardized_pace_uncertainty_min_mile=?,raw_aerobic_efficiency_min_mile=?,
                environmental_adjustment_min_mile=?,selected_model_name=?,selected_model_version=?
            WHERE activity_id=?
            """,
            (
                standardized_pace,
                pace_uncertainty,
                raw_efficiency,
                environmental,
                selected,
                version,
                activity_id,
            ),
        )

    correlations = {}
    for name in (
        "average_hr_bpm",
        "temperature_f",
        "dewpoint_f",
        "average_grade_percent",
        "moving_minutes_into_run",
    ):
        values = np.asarray([row[name] for row in rows], dtype=float)
        correlations[name] = float(np.corrcoef(values, pace_residuals)[0, 1])
    candidate_models = {}
    for name in MODEL_ORDER:
        candidate_builder = FeatureBuilder(name, config).fit(rows)
        candidate_model = fit_ridge(
            candidate_builder.transform(rows),
            candidate_builder.response(rows),
            float(config["model"]["ridge_alpha"]),
        )
        candidate_models[name] = {
            "feature_names": candidate_builder.names,
            "coefficients_mps": dict(zip(candidate_builder.names, candidate_model.coefficients.tolist())),
            "grade_transform": "Minetti 2002" if candidate_builder.uses_grade_energy_transform else None,
        }
    heat_model_name = "D+grade" if grade_supported else "D"
    heat_builder = FeatureBuilder(heat_model_name, config).fit(rows)
    heat_model = fit_ridge(
        heat_builder.transform(rows), heat_builder.response(rows), float(config["model"]["ridge_alpha"])
    )
    heat_reference = _reference_row(heat_builder, config)
    metadata = {
        "model_version": MODEL_VERSION,
        "version": version,
        "selected_model": selected,
        "grade_supported": grade_supported,
        "weather_supported": weather_supported,
        "wind_load_supported": wind_load_supported,
        "selection_decisions": decisions,
        "window_count": len(rows),
        "run_count": len(grouped_rows),
        "reliable_window_filter": reliability,
        "model_observation_unit": f"{int(config['model']['window_seconds'])}-second non-overlapping windows from raw trackpoints",
        "window_resolution_sensitivity": window_sensitivity,
        "grouped_cross_validation": grouped_validation,
        "time_blocked_cross_validation": temporal_validation,
        "feature_names": builder.names,
        "coefficients_mps": dict(zip(builder.names, model.coefficients.tolist())),
        "candidate_models": candidate_models,
        "residual_standard_deviation_seconds_per_mile": float(np.std(pace_residuals) * 60),
        "residual_correlations": correlations,
        "run_effect_variance_mps2": tau2,
        "reference_conditions": reference,
        "reference_population_pace_min_mile": speed_to_pace_min_mile(reference_speed),
        "heat_response": _heat_table(heat_model, heat_builder, heat_reference, rows),
        "heat_response_model": heat_model_name,
        "heat_response_trusted_for_primary_score": weather_supported,
        "scientific_basis": {
            "hr_speed": "Firstbeat reliable HR/speed segment principle; linear submaximal HR-speed relationship",
            "grade": "Minetti et al. 2002 measured metabolic running-cost polynomial",
            "humidity": "Stull 2011 wet-bulb approximation",
            "heat_index": "estimated shade WBGT = 0.7 wet-bulb + 0.3 air temperature; no solar-load claim",
            "drift": "linear within-run time term after excluding the first five minutes and durations beyond configured limit",
        },
        "limitations": [
            "One-athlete observational model; adjustments are associative, not causal.",
            "Estimated shade WBGT omits direct solar radiation and is not measured outdoor WBGT.",
            "Run effects use conservative empirical-Bayes shrinkage with at most four effective segments per run.",
            "Heat/dew-point scenarios with fewer than five nearby runs are weakly supported extrapolations.",
            "Weather is excluded from the primary score unless it improves both whole-run and time-blocked validation.",
        ],
    }
    connection.execute(
        "INSERT OR REPLACE INTO model_metadata(model_name,model_version,fitted_at_utc,metadata_json) VALUES (?,?,?,?)",
        ("standardized_pace_145", version, datetime.now(timezone.utc).isoformat(), json.dumps(metadata)),
    )
    connection.commit()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return ModelFitSummary(
        window_count=len(rows),
        run_count=len(grouped_rows),
        selected_model=selected,
        weather_supported=weather_supported,
        wind_load_supported=wind_load_supported,
        output_path=str(output),
    )


def fit_models(connection: sqlite3.Connection, config: dict, output_path: str | Path) -> ModelFitSummary:
    """Fit the prespecified published-reference score.

    The older data-selected candidate implementation remains above for
    reproducibility and unit-level sensitivity tests, but it no longer drives
    the primary longitudinal score because fitness and season are confounded.
    """
    from .objective_modeling import fit_published_reference_model

    try:
        metadata = fit_published_reference_model(connection, config, output_path)
    except ValueError as exc:
        message = str(exc)
        if message.startswith("At least 10 runs") or message.startswith(
            "Insufficient within-run HR/time variation"
        ):
            raise InsufficientModelDataError(message) from exc
        raise
    return ModelFitSummary(
        window_count=int(metadata["window_count"]),
        run_count=int(metadata["run_count"]),
        selected_model=str(metadata["selected_model"]),
        weather_supported=bool(metadata["weather_supported"]),
        wind_load_supported=bool(metadata["wind_load_supported"]),
        output_path=str(output_path),
    )
