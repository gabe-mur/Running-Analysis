"""Prior-anchored, auditable aerobic-efficiency scoring.

Published evidence supplies environmental priors. Personal evidence may update
them only through local hot/cool matches, so a changing multi-year fitness trend
cannot freely determine an environmental coefficient.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any
import hashlib
import json
import math
import sqlite3

import numpy as np

from .model_windows import load_model_window_sets, load_overlapping_model_windows
from .physiology import estimated_shade_wbgt_f

METERS_PER_MILE = 1609.344
MODEL_VERSION = "aerobic-v9-shared-time-effect"


def pace_to_speed_mps(pace_min_mile: float) -> float:
    return METERS_PER_MILE / (float(pace_min_mile) * 60.0)


def speed_to_pace_min_mile(speed_mps: float) -> float:
    return METERS_PER_MILE / (max(0.1, float(speed_mps)) * 60.0)


def _wbgt_c(row: dict[str, Any]) -> float:
    wbgt_f = estimated_shade_wbgt_f(
        float(row["temperature_f"]),
        relative_humidity_percent=(
            float(row["relative_humidity_percent"])
            if row.get("relative_humidity_percent") is not None
            else None
        ),
        dewpoint_f=float(row["dewpoint_f"]) if row.get("dewpoint_f") is not None else None,
    )
    return (wbgt_f - 32.0) * 5.0 / 9.0


def _reference_wbgt_c(config: dict) -> float:
    reference = config["reference_conditions"]
    return _wbgt_c(
        {
            "temperature_f": reference["temperature_f"],
            "dewpoint_f": reference["dewpoint_f"],
            "relative_humidity_percent": None,
        }
    )


def _heat_exposure_c(row: dict[str, Any], config: dict) -> float:
    return max(0.0, _wbgt_c(row) - _reference_wbgt_c(config))


def _prior_corrected_speed(row: dict[str, Any], config: dict) -> float:
    """Initial correction used only to estimate the within-run HR slope."""
    raw = pace_to_speed_mps(float(row["moving_pace_min_mile"]))
    grade_ratio = row.get("grade_energy_ratio")
    after_grade = raw * (float(grade_ratio) if grade_ratio is not None else 1.0)
    prior = float(config["model"]["fixed_heat_loss_fraction_per_c"])
    return after_grade * math.exp(prior * _heat_exposure_c(row, config))


def _within_run_hr_time_slopes(
    rows: list[dict[str, Any]], speeds: list[float]
) -> tuple[float, float, dict]:
    """Estimate shared HR/time effects with a fixed performance offset per run."""
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[int(row["activity_id"])].append(index)
    predictors: list[tuple[float, float]] = []
    y: list[float] = []
    for indexes in grouped.values():
        if len(indexes) < 2:
            continue
        mean_hr = float(np.mean([float(rows[index]["average_hr_bpm"]) for index in indexes]))
        mean_time = float(
            np.mean([float(rows[index]["moving_minutes_into_run"]) for index in indexes])
        )
        mean_speed = float(np.mean([speeds[index] for index in indexes]))
        for index in indexes:
            predictors.append(
                (
                    float(rows[index]["average_hr_bpm"]) - mean_hr,
                    float(rows[index]["moving_minutes_into_run"]) - mean_time,
                )
            )
            y.append(speeds[index] - mean_speed)
    x_array = np.asarray(predictors, dtype=float)
    y_array = np.asarray(y, dtype=float)
    if len(x_array) < 20 or np.linalg.matrix_rank(x_array) < 2:
        raise ValueError("Insufficient within-run HR/time variation for calibration")
    weights = np.ones(len(x_array), dtype=float)
    slopes = np.linalg.lstsq(x_array, y_array, rcond=None)[0]
    for _ in range(8):
        residuals = y_array - x_array @ slopes
        scale = 1.4826 * float(np.median(np.abs(residuals - np.median(residuals))))
        if scale <= 1e-9:
            break
        cutoff = 1.345 * scale
        weights = np.minimum(1.0, cutoff / np.maximum(np.abs(residuals), 1e-12))
        weighted_x = x_array * np.sqrt(weights)[:, None]
        weighted_y = y_array * np.sqrt(weights)
        slopes = np.linalg.lstsq(weighted_x, weighted_y, rcond=None)[0]
    hr_slope = float(slopes[0])
    time_slope = float(slopes[1])
    return hr_slope, time_slope, {
        "method": "Huber robust joint regression on run-centered HR and time windows",
        "window_observations": len(x_array),
        "contributing_runs": sum(len(indexes) >= 2 for indexes in grouped.values()),
        "speed_mps_per_bpm": hr_slope,
        "speed_mps_per_minute": time_slope,
        "seconds_per_mile_per_10bpm_at_10min_mile": (
            speed_to_pace_min_mile(pace_to_speed_mps(10.0) + hr_slope * 10.0) - 10.0
        )
        * 60.0,
        "seconds_per_mile_per_10min_at_10min_mile": (
            speed_to_pace_min_mile(pace_to_speed_mps(10.0) + time_slope * 10.0) - 10.0
        )
        * 60.0,
    }


def _pace_delta(before_speed: float, after_speed: float) -> float:
    return speed_to_pace_min_mile(after_speed) - speed_to_pace_min_mile(before_speed)


def _primary_window_weight(row: dict[str, Any], config: dict) -> float:
    """Continuous relevance/stability weight; no best-window selection."""

    settings = config["model"]
    target_hr = float(config["target_hr"])
    target_minutes = float(config["reference_conditions"]["within_run_minutes"])
    hr_scale = float(settings["primary_hr_weight_scale_bpm"])
    time_scale = float(settings["primary_time_weight_scale_minutes"])
    hr_change_scale = float(settings["primary_hr_change_weight_scale_bpm"])
    speed_change_scale = float(settings["primary_speed_change_weight_scale_fraction"])
    hr_distance = (float(row["average_hr_bpm"]) - target_hr) / hr_scale
    time_distance = (float(row["moving_minutes_into_run"]) - target_minutes) / time_scale
    hr_change = abs(float(row.get("heart_rate_change_bpm") or 0.0)) / hr_change_scale
    speed_change = abs(float(row.get("speed_change_fraction") or 0.0)) / speed_change_scale
    stop_fraction = float(row.get("stopped_time_s") or 0.0) / max(
        1.0, float(row.get("elapsed_time_s") or 0.0)
    )
    return max(
        1e-4,
        math.exp(-0.5 * hr_distance**2)
        * math.exp(-0.5 * time_distance**2)
        * math.exp(-hr_change)
        * math.exp(-speed_change)
        * max(0.05, (1.0 - stop_fraction) ** 2),
    )


def _robust_weighted_speed(values: list[float], weights: list[float]) -> float:
    if not values or len(values) != len(weights):
        raise ValueError("Weighted speed aggregation requires aligned observations")
    estimate = sum(value * weight for value, weight in zip(values, weights)) / sum(weights)
    robust = list(weights)
    for _ in range(8):
        residuals = [value - estimate for value in values]
        center = median(residuals)
        scale = max(1e-4, 1.4826 * median(abs(value - center) for value in residuals))
        cutoff = 1.345 * scale
        robust = [
            weight * min(1.0, cutoff / max(abs(residual), 1e-12))
            for weight, residual in zip(weights, residuals)
        ]
        estimate = sum(value * weight for value, weight in zip(values, robust)) / sum(robust)
    return estimate


def _reference_time_support(
    minutes_into_run: list[float], reference_minutes: float, config: dict
) -> tuple[bool, float, float, str]:
    """Classify interpolation/extrapolation support without using total duration."""

    if not minutes_into_run:
        return False, float("inf"), float("inf"), "unavailable"
    low, high = min(minutes_into_run), max(minutes_into_run)
    if low <= reference_minutes <= high:
        return True, 1.0, 0.0, "interpolation"
    gap = min(abs(reference_minutes - low), abs(reference_minutes - high))
    settings = config["model"]
    maximum_gap = float(settings.get("maximum_reference_extrapolation_minutes", 5.0))
    if gap > maximum_gap:
        return False, float("inf"), gap, "unsupported_extrapolation"
    scale = float(
        settings.get(
            "reference_extrapolation_uncertainty_scale_minutes",
            max(0.5, float(settings["primary_window_seconds"]) / 120.0),
        )
    )
    return True, math.sqrt(1.0 + (gap / scale) ** 2), gap, "limited_extrapolation"


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    return sum(value * weight for value, weight in zip(values, weights)) / sum(weights)


def _shared_parameter_eligible(row: dict[str, Any]) -> bool:
    return (
        bool(row.get("gps_sufficient_for_shared_parameters", True))
        and not bool(row.get("weather_location_estimated"))
        and str(row.get("health_tag") or "normal") == "normal"
        and str(row.get("workout_type") or "unknown")
        not in {"intervals", "tempo_threshold", "race", "run_walk", "hike", "bike"}
    )


def _fit_pair_slope(pairs: list[dict[str, float]], omitted_run: int | None = None) -> float | None:
    selected = [
        pair
        for pair in pairs
        if omitted_run is None
        or (int(pair["first_id"]) != omitted_run and int(pair["second_id"]) != omitted_run)
    ]
    denominator = sum(pair["weight"] * pair["x"] ** 2 for pair in selected)
    if denominator <= 0:
        return None
    return sum(pair["weight"] * pair["x"] * pair["y"] for pair in selected) / denominator


def _estimate_heat_posterior(run_rows: list[dict[str, Any]], config: dict) -> dict[str, Any]:
    """Normal prior updated by local run-pair contrasts.

    Pairs are close in calendar time but meaningfully different in WBGT. The
    contrast limits, rather than eliminates, confounding from moving fitness.
    """
    settings = config["model"]
    prior_mean = float(settings["fixed_heat_loss_fraction_per_c"])
    prior_sd = float(settings["fixed_heat_loss_uncertainty_fraction_per_c"])
    max_days = float(settings.get("heat_personal_match_max_days", 56))
    minimum_delta = float(settings.get("heat_personal_match_min_wbgt_delta_c", 3))
    pairs: list[dict[str, float]] = []
    support: dict[int, int] = defaultdict(int)
    for first_index, first in enumerate(run_rows):
        first_date = datetime.fromisoformat(str(first["start_time_utc"]))
        for second in run_rows[first_index + 1 :]:
            second_date = datetime.fromisoformat(str(second["start_time_utc"]))
            days = abs((second_date - first_date).total_seconds()) / 86400.0
            x = float(first["heat_exposure_c"]) - float(second["heat_exposure_c"])
            if days > max_days or abs(x) < minimum_delta:
                continue
            y = math.log(float(first["pace_before_heat"])) - math.log(
                float(second["pace_before_heat"])
            )
            weight = math.exp(-days / max_days)
            pairs.append(
                {
                    "first_id": float(first["activity_id"]),
                    "second_id": float(second["activity_id"]),
                    "x": x,
                    "y": y,
                    "weight": weight,
                }
            )
            support[int(first["activity_id"])] += 1
            support[int(second["activity_id"])] += 1

    personal_mean = _fit_pair_slope(pairs)
    matched_ids = sorted(support)
    personal_se = None
    if personal_mean is not None and len(pairs) >= 3 and len(matched_ids) >= 4:
        residual_numerator = sum(
            pair["weight"] * (pair["y"] - personal_mean * pair["x"]) ** 2
            for pair in pairs
        )
        denominator = sum(pair["weight"] * pair["x"] ** 2 for pair in pairs)
        residual_se = math.sqrt(
            max(0.0, residual_numerator / max(1, len(pairs) - 1) / denominator)
        )
        leave_one_out = [
            value
            for run_id in matched_ids
            if (value := _fit_pair_slope(pairs, run_id)) is not None
        ]
        if len(leave_one_out) >= 3:
            center = sum(leave_one_out) / len(leave_one_out)
            jackknife_se = math.sqrt(
                (len(leave_one_out) - 1)
                / len(leave_one_out)
                * sum((value - center) ** 2 for value in leave_one_out)
            )
            personal_se = max(residual_se, jackknife_se)
        else:
            personal_se = residual_se
        personal_se = max(
            personal_se,
            float(settings.get("heat_personal_minimum_se_fraction_per_c", 0.0005)),
        )

    posterior_mean = prior_mean
    posterior_sd = prior_sd
    data_weight = 0.0
    if personal_mean is not None and personal_se is not None and personal_se > 0:
        prior_precision = 1.0 / prior_sd**2
        data_precision = 1.0 / personal_se**2
        posterior_mean = max(
            0.0,
            (prior_mean * prior_precision + personal_mean * data_precision)
            / (prior_precision + data_precision),
        )
        posterior_sd = math.sqrt(1.0 / (prior_precision + data_precision))
        data_weight = data_precision / (prior_precision + data_precision)
    if len(matched_ids) >= 20 and data_weight >= 0.65:
        confidence = "high"
    elif len(matched_ids) >= 8 and data_weight >= 0.25:
        confidence = "moderate"
    else:
        confidence = "low"
    return {
        "prior_mean_fraction_per_c": prior_mean,
        "prior_sd_fraction_per_c": prior_sd,
        "personal_likelihood_mean_fraction_per_c": personal_mean,
        "personal_likelihood_se_fraction_per_c": personal_se,
        "posterior_mean_fraction_per_c": posterior_mean,
        "posterior_sd_fraction_per_c": posterior_sd,
        "personal_data_weight": data_weight,
        "matched_pair_count": len(pairs),
        "matched_run_count": len(matched_ids),
        "match_window_days": max_days,
        "minimum_wbgt_contrast_c": minimum_delta,
        "confidence": confidence,
        "run_support_counts": {str(key): value for key, value in support.items()},
        "interpretation": "normal literature prior updated by calendar-local hot/cool run contrasts",
    }


def _temperature_dew_log_factors(
    row: dict[str, Any], config: dict, coefficient: float
) -> tuple[float, float]:
    reference = config["reference_conditions"]
    ref = {
        "temperature_f": float(reference["temperature_f"]),
        "dewpoint_f": float(reference["dewpoint_f"]),
        "relative_humidity_percent": None,
    }
    observed = {
        "temperature_f": float(row["temperature_f"]),
        "dewpoint_f": (
            float(row["dewpoint_f"])
            if row.get("dewpoint_f") is not None
            else ref["dewpoint_f"]
        ),
        "relative_humidity_percent": None,
    }
    temp_only = {**observed, "dewpoint_f": ref["dewpoint_f"]}
    dew_only = {**ref, "dewpoint_f": observed["dewpoint_f"]}
    l00 = coefficient * _heat_exposure_c(ref, config)
    l10 = coefficient * _heat_exposure_c(temp_only, config)
    l01 = coefficient * _heat_exposure_c(dew_only, config)
    l11 = coefficient * _heat_exposure_c(observed, config)
    # Two-order Shapley decomposition; the two components sum to total log factor.
    temperature = 0.5 * ((l10 - l00) + (l11 - l01))
    dewpoint = 0.5 * ((l01 - l00) + (l11 - l10))
    return temperature, dewpoint


def _heat_table(
    config: dict, reference_pace: float, rows: list[dict[str, Any]], posterior: dict[str, Any]
) -> list[dict]:
    coefficient = float(posterior["posterior_mean_fraction_per_c"])
    coefficient_uncertainty = float(posterior["posterior_sd_fraction_per_c"])
    reference_wbgt = _reference_wbgt_c(config)
    output = []
    for temperature in (55, 60, 65, 70, 75, 80, 85, 90):
        for dewpoint in (45, 60, 70):
            scenario = {
                "temperature_f": float(temperature),
                "dewpoint_f": float(dewpoint),
                "relative_humidity_percent": None,
            }
            delta_c = max(0.0, _wbgt_c(scenario) - reference_wbgt)
            penalty = reference_pace * (math.exp(coefficient * delta_c) - 1.0) * 60.0
            supporting_runs = len(
                {
                    int(row["activity_id"])
                    for row in rows
                    if abs(float(row["temperature_f"]) - temperature) <= 3
                    and row.get("dewpoint_f") is not None
                    and abs(float(row["dewpoint_f"]) - dewpoint) <= 5
                }
            )
            output.append(
                {
                    "temperature_f": temperature,
                    "dewpoint_f": dewpoint,
                    "pace_penalty_seconds_per_mile": penalty,
                    "uncertainty_95_seconds_per_mile": (
                        1.96 * reference_pace * coefficient_uncertainty * delta_c * 60.0
                    ),
                    "supporting_runs_near_conditions": supporting_runs,
                    "coverage": "weak" if supporting_runs < 5 else "moderate" if supporting_runs < 15 else "good",
                    "evidence_confidence": posterior["confidence"],
                    "personal_data_weight": posterior["personal_data_weight"],
                }
            )
    return output


def select_comparable_run_windows(
    rows: list[dict[str, Any]],
    indexes: list[int],
    reference_minutes: float,
    maximum_windows: int,
) -> list[int]:
    """Return fixed-position run evidence without duration-dependent weighting."""

    return sorted(
        indexes,
        key=lambda index: (
            abs(float(rows[index]["moving_minutes_into_run"]) - reference_minutes),
            float(rows[index]["moving_minutes_into_run"]),
        ),
    )[:maximum_windows]


def select_steady_aerobic_window(
    rows: list[dict[str, Any]], indexes: list[int], config: dict
) -> list[int]:
    """Select one fixed-time aerobic window with inspectable stability rules.

    This is a deliberately narrower corroboration signal. It does not replace
    the broader per-run score, and a run with no qualifying window is missing
    rather than coerced into comparability.
    """

    settings = config["model"]
    reference = float(settings["steady_benchmark_reference_minutes"])
    maximum_reference_offset = float(
        settings["steady_benchmark_maximum_reference_offset_minutes"]
    )
    minimum_minutes = float(settings["steady_benchmark_minimum_minutes"])
    maximum_minutes = float(settings["steady_benchmark_maximum_minutes"])
    minimum_hr = float(settings["steady_benchmark_minimum_hr_bpm"])
    maximum_hr = float(settings["steady_benchmark_maximum_hr_bpm"])
    maximum_hr_change = float(settings["steady_benchmark_maximum_hr_change_bpm"])
    maximum_hr_range = float(settings["steady_benchmark_maximum_hr_range_bpm"])
    maximum_speed_change = float(settings["steady_benchmark_maximum_speed_change_fraction"])
    maximum_stopped = float(settings["steady_benchmark_maximum_stopped_seconds"])
    eligible = []
    for index in indexes:
        row = rows[index]
        position = float(row["moving_minutes_into_run"])
        hr = float(row["average_hr_bpm"])
        hr_change = row.get("heart_rate_change_bpm")
        hr_range = row.get("heart_rate_range_bpm")
        speed_change = row.get("speed_change_fraction")
        if not (
            minimum_minutes <= position <= maximum_minutes
            and abs(position - reference) <= maximum_reference_offset
            and minimum_hr <= hr <= maximum_hr
        ):
            continue
        if float(row.get("stopped_time_s") or 0.0) > maximum_stopped:
            continue
        if hr_change is None or abs(float(hr_change)) > maximum_hr_change:
            continue
        if hr_range is None or float(hr_range) > maximum_hr_range:
            continue
        if speed_change is None or abs(float(speed_change)) > maximum_speed_change:
            continue
        eligible.append(index)
    return sorted(
        eligible,
        key=lambda index: (
            abs(float(rows[index]["moving_minutes_into_run"]) - reference),
            abs(float(rows[index]["average_hr_bpm"]) - float(config["target_hr"])),
        ),
    )[:1]


def select_fixed_time_benchmark_fallback(
    rows: list[dict[str, Any]], indexes: list[int], config: dict
) -> list[int]:
    """Choose an inspectable fixed-time estimate when no strict window exists.

    The underlying model-window loader has already enforced reliable HR,
    distance, weather, stop, and submaximal-range requirements.  This fallback
    therefore relaxes only the benchmark's narrow minute/HR/stability gates.
    Its relevance/stability weight is converted into additional uncertainty
    rather than pretending the window was a strict observation.
    """

    if not indexes:
        return []
    settings = config["model"]
    reference = float(settings["steady_benchmark_reference_minutes"])
    minimum_minutes = float(settings["steady_benchmark_minimum_minutes"])
    maximum_minutes = float(settings["steady_benchmark_maximum_minutes"])
    target_hr = float(config["target_hr"])
    preferred = [
        index
        for index in indexes
        if minimum_minutes <= float(rows[index]["moving_minutes_into_run"]) <= maximum_minutes
    ]
    candidates = preferred or indexes
    return sorted(
        candidates,
        key=lambda index: (
            abs(float(rows[index]["moving_minutes_into_run"]) - reference),
            abs(float(rows[index]["average_hr_bpm"]) - target_hr),
            -_primary_window_weight(rows[index], config),
        ),
    )[:1]


def _distance_fallback_penalty_seconds(row: dict[str, Any], config: dict) -> float:
    """Return the same continuous sensor/location uncertainty used by the primary score."""

    minimum_gps = float(config["model"]["minimum_gps_coverage"])
    gps_coverage = float(row.get("gps_complete_fraction") or 0.0)
    shortfall = max(0.0, 1.0 - gps_coverage / max(minimum_gps, 1e-9))
    return math.sqrt(
        (
            float(
                config["model"].get(
                    "device_distance_fallback_uncertainty_seconds_per_mile", 30
                )
            )
            * shortfall
        )
        ** 2
        + (
            float(
                config["model"].get(
                    "estimated_weather_location_uncertainty_seconds_per_mile", 10
                )
            )
            if bool(row.get("weather_location_estimated"))
            else 0.0
        )
        ** 2
    )


def fit_published_reference_model(
    connection: sqlite3.Connection, config: dict, output_path: str | Path
) -> dict[str, Any]:
    sizes = tuple(int(value) for value in config["model"]["window_sensitivity_seconds"])
    benchmark_seconds = int(config["model"]["steady_benchmark_window_seconds"])
    if benchmark_seconds not in sizes:
        sizes = (*sizes, benchmark_seconds)
    window_sets, diagnostics = load_model_window_sets(connection, config, sizes)
    calibration_seconds = int(config["model"]["window_seconds"])
    calibration_rows = [
        row for row in window_sets[calibration_seconds] if _shared_parameter_eligible(row)
    ]
    primary_seconds = int(config["model"]["primary_window_seconds"])
    primary_stride = int(config["model"]["primary_window_stride_seconds"])
    rows, primary_diagnostics = load_overlapping_model_windows(
        connection,
        config,
        window_seconds=primary_seconds,
        stride_seconds=primary_stride,
    )
    if len({int(row["activity_id"]) for row in rows}) < 10:
        raise ValueError("At least 10 runs with reliable steady-state windows are required")

    calibration_speeds = [_prior_corrected_speed(row, config) for row in calibration_rows]
    hr_slope, time_slope, slope_diagnostics = _within_run_hr_time_slopes(
        calibration_rows, calibration_speeds
    )
    target_hr = float(config["target_hr"])
    reference_minutes = float(config["reference_conditions"]["within_run_minutes"])
    raw_window_speeds = [pace_to_speed_mps(float(row["moving_pace_min_mile"])) for row in rows]
    raw_hr_window_speeds = [
        speed + hr_slope * (target_hr - float(row["average_hr_bpm"]))
        for speed, row in zip(raw_window_speeds, rows)
    ]
    raw_hr_time_window_speeds = [
        speed
        + time_slope
        * (reference_minutes - float(row["moving_minutes_into_run"]))
        for speed, row in zip(raw_hr_window_speeds, rows)
    ]
    grade_window_speeds = [
        speed * (float(row["grade_energy_ratio"]) if row.get("grade_energy_ratio") is not None else 1.0)
        for speed, row in zip(raw_hr_time_window_speeds, rows)
    ]

    grouped: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[int(row["activity_id"])].append(index)
    # Use every reliable overlapping window. Relevance and transition quality
    # affect weight continuously; no best-section selection determines the run.
    comparable_grouped = grouped
    window_weights = [_primary_window_weight(row, config) for row in rows]
    heat_evidence_rows = []
    for activity_id, indexes in comparable_grouped.items():
        if not _shared_parameter_eligible(rows[indexes[0]]):
            continue
        weights = [window_weights[index] for index in indexes]
        heat_evidence_rows.append(
            {
                "activity_id": activity_id,
                "start_time_utc": rows[indexes[0]]["start_time_utc"],
                "pace_before_heat": speed_to_pace_min_mile(
                    _robust_weighted_speed(
                        [grade_window_speeds[index] for index in indexes], weights
                    )
                ),
                "heat_exposure_c": _weighted_mean(
                    [_heat_exposure_c(rows[index], config) for index in indexes], weights
                ),
            }
        )
    heat_posterior = _estimate_heat_posterior(heat_evidence_rows, config)
    heat_coefficient = float(heat_posterior["posterior_mean_fraction_per_c"])
    temperature_log_factors = []
    dewpoint_log_factors = []
    for row in rows:
        temperature_factor, dewpoint_factor = _temperature_dew_log_factors(
            row, config, heat_coefficient
        )
        temperature_log_factors.append(temperature_factor)
        dewpoint_log_factors.append(dewpoint_factor)
    standardized_window_speeds = [
        speed * math.exp(temperature + dewpoint)
        for speed, temperature, dewpoint in zip(
            grade_window_speeds, temperature_log_factors, dewpoint_log_factors
        )
    ]
    temperature_window_speeds = [
        speed * math.exp(temperature)
        for speed, temperature in zip(grade_window_speeds, temperature_log_factors)
    ]
    benchmark_rows = rows if benchmark_seconds == primary_seconds else window_sets[benchmark_seconds]
    benchmark_raw_speeds = [
        pace_to_speed_mps(float(row["moving_pace_min_mile"])) for row in benchmark_rows
    ]
    benchmark_hr_speeds = [
        speed + hr_slope * (target_hr - float(row["average_hr_bpm"]))
        for speed, row in zip(benchmark_raw_speeds, benchmark_rows)
    ]
    benchmark_grade_speeds = [
        speed * (float(row["grade_energy_ratio"]) if row.get("grade_energy_ratio") is not None else 1.0)
        for speed, row in zip(benchmark_hr_speeds, benchmark_rows)
    ]
    benchmark_standardized_speeds = [
        speed * math.exp(sum(_temperature_dew_log_factors(row, config, heat_coefficient)))
        for speed, row in zip(benchmark_grade_speeds, benchmark_rows)
    ]
    benchmark_grouped: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(benchmark_rows):
        benchmark_grouped[int(row["activity_id"])].append(index)
    selected_benchmark = {
        activity_id: select_steady_aerobic_window(benchmark_rows, indexes, config)
        for activity_id, indexes in benchmark_grouped.items()
    }
    fallback_benchmark = {
        activity_id: select_fixed_time_benchmark_fallback(benchmark_rows, indexes, config)
        for activity_id, indexes in benchmark_grouped.items()
    }
    centered_residuals = []
    for indexes in grouped.values():
        if not _shared_parameter_eligible(rows[indexes[0]]):
            continue
        weights = [window_weights[index] for index in indexes]
        center = _robust_weighted_speed(
            [standardized_window_speeds[index] for index in indexes], weights
        )
        centered_residuals.extend(standardized_window_speeds[index] - center for index in indexes)
    residual_sigma = 1.4826 * median(abs(value - median(centered_residuals)) for value in centered_residuals)
    residual_sigma = max(residual_sigma, 1e-6)

    version_payload = {
        "model_version": MODEL_VERSION,
        "window_seconds": primary_seconds,
        "heat_prior_fraction_per_c": config["model"]["fixed_heat_loss_fraction_per_c"],
        "heat_posterior_fraction_per_c": heat_posterior[
            "posterior_mean_fraction_per_c"
        ],
        "target_hr": target_hr,
        "hr_slope_method": slope_diagnostics["method"],
        "shared_time_slope_mps_per_minute": time_slope,
        "comparable_reference_minutes": reference_minutes,
        "window_stride_seconds": primary_stride,
        "aggregation": "shared HR/time fixed effects plus robust run offset",
    }
    version = hashlib.sha256(json.dumps(version_payload, sort_keys=True).encode()).hexdigest()[:16]
    selected_name = "literature-prior + locally matched personal evidence"
    connection.execute("DELETE FROM model_runs WHERE model_name='standardized_pace_at_target_hr'")
    connection.execute(
        """UPDATE activity_metrics SET standardized_pace_at_target_hr_min_mile=NULL,
           standardized_pace_uncertainty_min_mile=NULL,raw_aerobic_efficiency_min_mile=NULL,
           environmental_adjustment_min_mile=NULL,selected_model_name=NULL,
           selected_model_version=NULL"""
    )
    connection.execute(
        "UPDATE activity_metrics SET exclusion_reason=NULL "
        "WHERE exclusion_reason='reference_time_unsupported_by_run_windows'"
    )

    run_paces = []
    for activity_id, indexes in comparable_grouped.items():
        run_rows = [rows[index] for index in indexes]
        weights = [window_weights[index] for index in indexes]
        run_minutes = [float(rows[index]["moving_minutes_into_run"]) for index in indexes]
        (
            reference_supported,
            time_support_multiplier,
            extrapolation_minutes,
            reference_support_kind,
        ) = _reference_time_support(run_minutes, reference_minutes, config)
        if not reference_supported:
            connection.execute(
                "UPDATE activity_metrics SET exclusion_reason=? WHERE activity_id=?",
                ("reference_time_unsupported_by_run_windows", activity_id),
            )
            continue
        observed_speed = _robust_weighted_speed(
            [raw_window_speeds[index] for index in indexes], weights
        )
        raw_hr_speed = _robust_weighted_speed(
            [raw_hr_window_speeds[index] for index in indexes], weights
        )
        time_speed = _robust_weighted_speed(
            [raw_hr_time_window_speeds[index] for index in indexes], weights
        )
        grade_speed = _robust_weighted_speed(
            [grade_window_speeds[index] for index in indexes], weights
        )
        temperature_speed = _robust_weighted_speed(
            [temperature_window_speeds[index] for index in indexes], weights
        )
        final_speed = _robust_weighted_speed(
            [standardized_window_speeds[index] for index in indexes], weights
        )
        standardized_pace = speed_to_pace_min_mile(final_speed)
        kish_n = sum(weights) ** 2 / sum(weight**2 for weight in weights)
        overlap_fraction = min(1.0, primary_stride / primary_seconds)
        effective_n = max(
            1.0,
            min(
                float(config["model"]["primary_maximum_effective_windows"]),
                kish_n * overlap_fraction,
            ),
        )
        speed_uncertainty = 1.96 * residual_sigma / math.sqrt(effective_n)
        low = speed_to_pace_min_mile(final_speed + speed_uncertainty)
        high = speed_to_pace_min_mile(max(0.1, final_speed - speed_uncertainty))
        measurement_uncertainty = (high - low) / 2.0 * time_support_multiplier
        run_heat_exposure = _weighted_mean(
            [_heat_exposure_c(row, config) for row in run_rows], weights
        )
        heat_coefficient_uncertainty = (
            1.96
            * standardized_pace
            * float(heat_posterior["posterior_sd_fraction_per_c"])
            * run_heat_exposure
        )
        uncertainty = math.sqrt(
            measurement_uncertainty**2 + heat_coefficient_uncertainty**2
        )
        run_gps_coverage = _weighted_mean(
            [float(row.get("gps_complete_fraction") or 0.0) for row in run_rows],
            weights,
        )
        minimum_gps_coverage = float(config["model"]["minimum_gps_coverage"])
        uses_device_fallback = run_gps_coverage < minimum_gps_coverage
        uses_estimated_weather_location = any(
            bool(row.get("weather_location_estimated")) for row in run_rows
        )
        gps_coverage_shortfall = max(
            0.0,
            1.0 - run_gps_coverage / max(minimum_gps_coverage, 1e-9),
        )
        fallback_penalty_seconds = math.sqrt(
            (
                float(config["model"].get("device_distance_fallback_uncertainty_seconds_per_mile", 30))
                * gps_coverage_shortfall
            ) ** 2
            + (
                float(config["model"].get("estimated_weather_location_uncertainty_seconds_per_mile", 10))
                if uses_estimated_weather_location
                else 0.0
            )
            ** 2
        )
        uncertainty = math.sqrt(uncertainty**2 + (fallback_penalty_seconds / 60.0) ** 2)

        observed_pace = speed_to_pace_min_mile(observed_speed)
        raw_pace_at_target_hr = speed_to_pace_min_mile(time_speed)
        contributions = {
            "hr_normalization": _pace_delta(observed_speed, raw_hr_speed),
            "time_normalization": _pace_delta(raw_hr_speed, time_speed),
            "grade_adjustment": _pace_delta(time_speed, grade_speed),
            "temperature_adjustment": _pace_delta(grade_speed, temperature_speed),
            "dewpoint_adjustment": _pace_delta(temperature_speed, final_speed),
            "wind_adjustment": 0.0,
            "drift_adjustment": 0.0,
        }
        environmental_adjustment = sum(
            contributions[name]
            for name in (
                "grade_adjustment",
                "temperature_adjustment",
                "dewpoint_adjustment",
                "wind_adjustment",
                "drift_adjustment",
            )
        )
        grade_adjusted_count = sum(
            row.get("grade_energy_ratio") is not None for row in run_rows
        )
        if grade_adjusted_count == len(run_rows):
            grade_confidence = "high"
        elif grade_adjusted_count:
            grade_confidence = "moderate"
        else:
            grade_confidence = "unavailable"
        support_count = int(
            heat_posterior["run_support_counts"].get(str(activity_id), 0)
        )
        if heat_posterior["confidence"] == "high" and support_count >= 8:
            run_heat_confidence = "high"
        elif heat_posterior["confidence"] in {"high", "moderate"} and support_count >= 3:
            run_heat_confidence = "moderate"
        else:
            run_heat_confidence = "low"
        adjustment_evidence = {
            "heart_rate": {
                "confidence": "moderate",
                "basis": "within-run centered personal calibration",
                "personal_data_weight": 1.0,
            },
            "time": {
                "confidence": "moderate",
                "basis": "shared run-centered time effect across normal non-quality runs",
                "personal_data_weight": 1.0,
            },
            "grade": {
                "confidence": grade_confidence,
                "basis": "published Minetti transform plus recorded altitude coverage",
                "personal_data_weight": 0.0,
            },
            "temperature": {
                "confidence": run_heat_confidence,
                "basis": "literature prior updated by local hot/cool matches",
                "personal_data_weight": heat_posterior["personal_data_weight"],
                "local_match_count": support_count,
            },
            "dew_point": {
                "confidence": run_heat_confidence,
                "basis": "WBGT Shapley decomposition of the same heat posterior",
                "personal_data_weight": heat_posterior["personal_data_weight"],
                "local_match_count": support_count,
            },
            "wind": {
                "confidence": "unavailable",
                "basis": "reported but not adjusted",
                "personal_data_weight": 0.0,
            },
            "drift": {
                "confidence": "unavailable",
                "basis": "filtered and reported, not adjusted",
                "personal_data_weight": 0.0,
            },
        }
        result = {
            "activity_id": run_rows[0]["external_activity_id"],
            # The heart rate every "at target HR" figure in this result refers
            # to. Stored per run so a later configuration change is visible
            # instead of silently relabeling old estimates.
            "target_hr_bpm": float(config["target_hr"]),
            "observed_segment_pace_min_mile": observed_pace,
            "raw_pace_at_target_hr_min_mile": raw_pace_at_target_hr,
            "environmental_adjustment_min_mile": environmental_adjustment,
            "standardized_pace_at_target_hr_min_mile": standardized_pace,
            "uncertainty_95_min_mile": uncertainty,
            "measurement_uncertainty_95_min_mile": measurement_uncertainty,
            "heat_coefficient_uncertainty_95_min_mile": heat_coefficient_uncertainty,
            "segment_count": len(indexes),
            "available_window_count": len(grouped[activity_id]),
            "selected_window_count": len(indexes),
            "effective_window_count": effective_n,
            "estimate_quality": (
                "device_distance_fallback"
                if run_gps_coverage <= 1e-9
                else "partial_gps_device_distance"
                if uses_device_fallback
                else "full_sensor"
            ),
            "gps_coverage_fraction": run_gps_coverage,
            "fallback_uncertainty_95_min_mile": fallback_penalty_seconds / 60.0,
            "weather_location_estimated": uses_estimated_weather_location,
            "comparable_window_center_minutes": _weighted_mean(
                [float(row["moving_minutes_into_run"]) for row in run_rows], weights
            ),
            "comparable_reference_minutes": reference_minutes,
            "shared_time_slope_mps_per_minute": time_slope,
            "reference_time_support_multiplier": time_support_multiplier,
            "reference_time_support": reference_support_kind,
            "reference_time_extrapolation_minutes": extrapolation_minutes,
            "grade_adjusted_windows": grade_adjusted_count,
            "grade_unavailable_windows": len(run_rows) - grade_adjusted_count,
            "contributions_min_mile": contributions,
            "adjustment_evidence": adjustment_evidence,
            "interpretation": (
                "higher-uncertainty estimate from Garmin device-distance windows at reference HR/time/conditions"
                if uses_device_fallback
                else "shared HR/time effects plus robust run-specific performance offset at reference conditions"
            ),
        }
        strict_benchmark_indexes = selected_benchmark.get(activity_id, [])
        benchmark_indexes = strict_benchmark_indexes or fallback_benchmark.get(activity_id, [])
        if benchmark_indexes:
            benchmark_index = benchmark_indexes[0]
            benchmark_row = benchmark_rows[benchmark_index]
            benchmark_speed = benchmark_standardized_speeds[benchmark_index]
            strict_observation = bool(strict_benchmark_indexes)
            benchmark_quality_weight = (
                1.0 if strict_observation else _primary_window_weight(benchmark_row, config)
            )
            benchmark_pace = speed_to_pace_min_mile(benchmark_speed)
            benchmark_speed_uncertainty = (
                1.96
                * residual_sigma
                / math.sqrt(max(0.10, benchmark_quality_weight))
            )
            benchmark_low = speed_to_pace_min_mile(benchmark_speed + benchmark_speed_uncertainty)
            benchmark_high = speed_to_pace_min_mile(max(0.1, benchmark_speed - benchmark_speed_uncertainty))
            benchmark_measurement_uncertainty = (benchmark_high - benchmark_low) / 2.0
            benchmark_fallback_penalty = _distance_fallback_penalty_seconds(
                benchmark_row, config
            )
            benchmark_uncertainty = math.sqrt(
                benchmark_measurement_uncertainty**2
                + (benchmark_fallback_penalty / 60.0) ** 2
            )
            result["steady_aerobic_benchmark"] = {
                "standardized_pace_at_target_hr_min_mile": benchmark_pace,
                "raw_pace_at_target_hr_min_mile": speed_to_pace_min_mile(benchmark_hr_speeds[benchmark_index]),
                "uncertainty_95_min_mile": benchmark_uncertainty,
                "measurement_uncertainty_95_min_mile": benchmark_measurement_uncertainty,
                "fallback_uncertainty_95_min_mile": benchmark_fallback_penalty / 60.0,
                "selection_quality": "strict_observed" if strict_observation else "estimated_fixed_time",
                "estimate_quality": (
                    "device_distance_fallback"
                    if float(benchmark_row.get("gps_complete_fraction") or 0.0) <= 1e-9
                    else "partial_gps_device_distance"
                    if float(benchmark_row.get("gps_complete_fraction") or 0.0)
                    < float(config["model"]["minimum_gps_coverage"])
                    else "full_sensor"
                ),
                "window_evidence_weight": benchmark_quality_weight,
                "gps_coverage_fraction": float(benchmark_row.get("gps_complete_fraction") or 0.0),
                "window_center_minutes": float(benchmark_row["moving_minutes_into_run"]),
                "average_hr_bpm": float(benchmark_row["average_hr_bpm"]),
                "heart_rate_change_bpm": float(benchmark_row["heart_rate_change_bpm"]),
                "heart_rate_range_bpm": float(benchmark_row["heart_rate_range_bpm"]),
                "speed_change_fraction": float(benchmark_row["speed_change_fraction"]),
                "window_seconds": benchmark_seconds,
                "definition": (
                    "one continuous, stable aerobic window nearest minute 20"
                    if strict_observation
                    else "nearest reliable fixed-time window, HR-normalized and uncertainty-weighted"
                ),
            }
        else:
            result["steady_aerobic_benchmark"] = None
        connection.execute(
            "INSERT INTO model_runs(activity_id,model_name,model_version,result_json) VALUES (?,?,?,?)",
            (activity_id, "standardized_pace_at_target_hr", version, json.dumps(result)),
        )
        connection.execute(
            """UPDATE activity_metrics SET standardized_pace_at_target_hr_min_mile=?,
                standardized_pace_uncertainty_min_mile=?,raw_aerobic_efficiency_min_mile=?,
                environmental_adjustment_min_mile=?,selected_model_name=?,selected_model_version=?
            WHERE activity_id=?""",
            (
                standardized_pace,
                uncertainty,
                raw_pace_at_target_hr,
                environmental_adjustment,
                selected_name,
                version,
                activity_id,
            ),
        )
        run_paces.append(standardized_pace)

    sensitivity = {
        str(seconds): {
            "window_count": len(sensitivity_rows),
            "run_count": len({int(row["activity_id"]) for row in sensitivity_rows}),
        }
        for seconds, sensitivity_rows in sorted(window_sets.items())
    }
    reference_pace = float(median(run_paces))
    reliability = dict(primary_diagnostics)
    reliability["grade_unavailable_retained"] = sum(
        row.get("grade_energy_ratio") is None for row in rows
    )
    metadata = {
        "model_version": MODEL_VERSION,
        "version": version,
        "method_kind": "prior_anchored",
        "selected_model": selected_name,
        "grade_supported": True,
        "weather_supported": True,
        "weather_basis": "literature prior updated by calendar-local hot/cool contrasts",
        "wind_load_supported": False,
        "selection_decisions": [
            "No environmental coefficient is freely fitted against the multi-year fitness trend.",
            "Minetti grade cost is applied only to windows with adequate altitude coverage; other windows remain scoreable.",
            f"Heat starts at a 0.2%/C WBGT literature prior; matched personal evidence currently has {heat_posterior['personal_data_weight'] * 100:.1f}% posterior weight.",
            "Wind remains diagnostic because an objective drag correction needs runner drag area and reliable local wind exposure.",
            "All reliable overlapping windows contribute in speed space; HR distance, target-time distance, stops, HR change, and speed change alter weight continuously.",
            "Low-GPS runs may be scored from reliable Garmin device-distance windows; they cannot calibrate shared parameters, and their uncertainty grows continuously as GPS coverage falls. Trend aggregation then downweights them by inverse variance.",
        ],
        "window_count": len(rows),
        "run_count": len(grouped),
        "shared_parameter_run_count": len(
            {
                int(row["activity_id"])
                for row in rows
                if _shared_parameter_eligible(row)
            }
        ),
        "reliable_window_filter": reliability,
        "model_observation_unit": (
            f"{primary_seconds}-second windows every {primary_stride} seconds from raw trackpoints"
        ),
        "run_aggregation": (
            f"shared run-centered HR/time effects evaluated at HR {target_hr:g} and "
            f"{reference_minutes:g} moving minutes, followed by a robust run-specific performance offset; "
            "total run duration is not a predictor"
        ),
        "steady_aerobic_corroboration": {
            "window_seconds": benchmark_seconds,
            "reference_minutes": config["model"]["steady_benchmark_reference_minutes"],
            "maximum_reference_offset_minutes": config["model"][
                "steady_benchmark_maximum_reference_offset_minutes"
            ],
            "heart_rate_range_bpm": [
                config["model"]["steady_benchmark_minimum_hr_bpm"],
                config["model"]["steady_benchmark_maximum_hr_bpm"],
            ],
            "eligible_run_count": sum(bool(indexes) for indexes in selected_benchmark.values()),
            "estimated_run_count": sum(
                not bool(selected_benchmark.get(activity_id)) and bool(indexes)
                for activity_id, indexes in fallback_benchmark.items()
            ),
            "run_count_considered": len(selected_benchmark),
            "selection": "single nearest window passing HR, continuity, HR-stability, and pace-stability rules",
        },
        "window_resolution_sensitivity": sensitivity,
        "grouped_cross_validation": {},
        "time_blocked_cross_validation": {},
        "candidate_models": {},
        "hr_calibration": slope_diagnostics,
        "heat_posterior": heat_posterior,
        "residual_standard_deviation_seconds_per_mile": (
            abs(speed_to_pace_min_mile(pace_to_speed_mps(reference_pace) - residual_sigma) - reference_pace)
            * 60.0
        ),
        "reference_conditions": {
            **config["reference_conditions"],
            "shade_wbgt_c": _reference_wbgt_c(config),
        },
        "reference_population_pace_min_mile": reference_pace,
        "heat_response": _heat_table(config, reference_pace, rows, heat_posterior),
        "heat_response_model": "literature prior plus locally matched personal evidence",
        "heat_response_trusted_for_primary_score": True,
        "scientific_basis": {
            "hr_speed": "personal slope from within-run centered steady-state windows; fitness level cancels within each run",
            "grade": "Minetti et al. 2002 measured metabolic running-cost polynomial",
            "heat": "conservative marathon population prior updated by calendar-local matched run contrasts",
            "humidity": "Stull 2011 wet-bulb approximation",
            "drift": "not fitted or corrected; early HR-lag and post-60-minute windows are excluded",
        },
        "limitations": [
            "The heat prior comes from population race evidence and local matching cannot remove every training/fatigue confounder.",
            "Shade WBGT omits direct solar radiation and is not measured outdoor WBGT.",
            "Grade is left unadjusted—not imputed—where altitude coverage is inadequate.",
            "Wind is reported but not corrected without defensible drag-area and street-level exposure inputs.",
            "Individual points are per-run estimates; 7- and 28-day trends are descriptive smoothers, not fitted fitness states.",
        ],
    }
    connection.execute(
        "INSERT OR REPLACE INTO model_metadata(model_name,model_version,fitted_at_utc,metadata_json) VALUES (?,?,?,?)",
        (
            "standardized_pace_at_target_hr",
            version,
            datetime.now(timezone.utc).isoformat(),
            json.dumps(metadata),
        ),
    )
    connection.commit()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata
