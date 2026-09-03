"""Interpret independent per-run scores as a changing fitness time series."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from math import erf, sqrt
from statistics import median
from typing import Any


MIN_MEASUREMENT_SIGMA_MIN_MILE = 10.0 / 60.0
LIKELY_DIRECTION_PROBABILITY = 0.80
CLEAR_DIRECTION_Z = 1.96


def _date(row: dict[str, Any]) -> datetime:
    return datetime.fromisoformat(row["start_time_utc"])


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def _directional_evidence(
    change_min_mile: float,
    standard_error_min_mile: float,
) -> dict[str, Any]:
    """Separate a likely direction from a clear two-sided result.

    The old classifier required the change to clear a 95% two-sided interval
    before it even consulted an 80% one-sided probability. That made the 80%
    branch unreachable. Both statements are useful, but they are not the same
    strength of claim.
    """

    standard_error = max(standard_error_min_mile, 1e-12)
    uncertainty_95 = CLEAR_DIRECTION_Z * standard_error
    probability_faster = _normal_cdf(-change_min_mile / standard_error)
    if abs(change_min_mile) > uncertainty_95:
        evidence_strength = "clear"
        directional_interpretation = (
            "improving" if change_min_mile < 0 else "declining"
        )
    elif probability_faster >= LIKELY_DIRECTION_PROBABILITY:
        evidence_strength = "likely"
        directional_interpretation = "improving"
    elif probability_faster <= 1.0 - LIKELY_DIRECTION_PROBABILITY:
        evidence_strength = "likely"
        directional_interpretation = "declining"
    else:
        evidence_strength = "inconclusive"
        directional_interpretation = "stable_or_uncertain"
    return {
        "uncertainty_95_min_mile": uncertainty_95,
        "probability_faster": probability_faster,
        "directional_interpretation": directional_interpretation,
        "evidence_strength": evidence_strength,
    }


def _weighted_line(
    x_values: list[float],
    y_values: list[float],
    weights: list[float],
) -> tuple[float, float, float, float] | None:
    weight_sum = sum(weights)
    if weight_sum <= 0:
        return None
    x_mean = sum(weight * value for weight, value in zip(weights, x_values)) / weight_sum
    y_mean = sum(weight * value for weight, value in zip(weights, y_values)) / weight_sum
    x_variance_sum = sum(
        weight * (value - x_mean) ** 2
        for weight, value in zip(weights, x_values)
    )
    if x_variance_sum <= 1e-12:
        return None
    slope = sum(
        weight * (x_value - x_mean) * (y_value - y_mean)
        for weight, x_value, y_value in zip(weights, x_values, y_values)
    ) / x_variance_sum
    intercept = y_mean - slope * x_mean
    return intercept, slope, x_mean, x_variance_sum


def _window_trend(
    scored: list[dict[str, Any]],
    end: datetime,
    days: int,
    *,
    robust: bool = True,
) -> dict[str, Any] | None:
    """Fit a measurement-weighted pace trajectory inside one window.

    This complements adjacent block averages. A gradual change can leave the
    two block means close even while a consistent within-window slope is
    visible. The slope uses the same uncertainty and contextual weights as the
    level estimate, with the same Huber protection against an isolated bad run.
    """

    start = end - timedelta(days=days)
    selected = [row for row in scored if start < _date(row) <= end]
    if len(selected) < 3:
        return None
    x_values = [(_date(row) - end).total_seconds() / 86400.0 for row in selected]
    if max(x_values) - min(x_values) <= 0:
        return None
    y_values = [float(row["standardized_pace"]) for row in selected]
    measurement_sigmas = [
        max(
            MIN_MEASUREMENT_SIGMA_MIN_MILE,
            float(row["uncertainty_95"] or 0.0) / CLEAR_DIRECTION_Z,
        )
        for row in selected
    ]
    context_weights = [
        max(0.0, min(1.0, float(row.get("trend_weight", 1.0))))
        for row in selected
    ]
    base_weights = [
        context / sigma**2
        for context, sigma in zip(context_weights, measurement_sigmas)
    ]
    weights = list(base_weights)
    fitted = _weighted_line(x_values, y_values, weights)
    if fitted is None:
        return None
    for _ in range(8 if robust else 0):
        intercept, slope, _, _ = fitted
        residuals = [
            value - (intercept + slope * x_value)
            for x_value, value in zip(x_values, y_values)
        ]
        center = median(residuals)
        scale = max(
            5.0 / 60.0,
            1.4826 * median(abs(residual - center) for residual in residuals),
        )
        cutoff = 1.345 * scale
        huber_factors = [
            min(1.0, cutoff / max(abs(residual), 1e-12))
            for residual in residuals
        ]
        weights = [
            base * factor for base, factor in zip(base_weights, huber_factors)
        ]
        fitted = _weighted_line(x_values, y_values, weights)
        if fitted is None:
            return None

    intercept, slope, x_mean, x_variance_sum = fitted
    residuals = [
        value - (intercept + slope * x_value)
        for x_value, value in zip(x_values, y_values)
    ]
    weight_sum = sum(weights)
    effective_n = weight_sum**2 / sum(weight**2 for weight in weights)
    weighted_variance = sum(
        weight * residual**2 for weight, residual in zip(weights, residuals)
    ) / weight_sum
    normalized_x_variance = x_variance_sum / weight_sum
    measurement_slope_se = sqrt(1.0 / x_variance_sum)
    between_run_slope_se = sqrt(
        weighted_variance
        / max(1e-12, effective_n * normalized_x_variance)
    )
    slope_se = sqrt(measurement_slope_se**2 + between_run_slope_se**2)
    window_change = slope * days
    window_change_se = slope_se * days
    directional = _directional_evidence(window_change, window_change_se)
    coverage_span_days = max(x_values) - min(x_values)
    return {
        "basis": f"weighted slope within the last {days} days",
        "pace_change_min_mile": window_change,
        "pace_change_seconds_per_mile": window_change * 60.0,
        "uncertainty_95_seconds_per_mile": (
            directional["uncertainty_95_min_mile"] * 60.0
        ),
        "probability_faster": directional["probability_faster"],
        "directional_interpretation": directional[
            "directional_interpretation"
        ],
        "evidence_strength": directional["evidence_strength"],
        "run_count": len(selected),
        "full_weight_run_equivalents": sum(context_weights),
        "effective_run_count": effective_n,
        "coverage_span_days": coverage_span_days,
        "coverage_fraction": min(1.0, coverage_span_days / float(days)),
        "slope_seconds_per_mile_per_day": slope * 60.0,
        "centered_at_utc": (
            end + timedelta(days=x_mean)
        ).isoformat(),
    }


def _window_estimate(
    scored: list[dict[str, Any]],
    end: datetime,
    days: int,
    minimum_runs: int = 1,
    *,
    robust: bool = True,
) -> dict[str, Any] | None:
    """Weighted trailing estimate.

    ``robust=False`` disables the Huber residual reweighting and is used to
    check that the robust layer is not masking genuine step changes; see
    ``scripts/huber_sensitivity.py``.
    """
    start = end - timedelta(days=days)
    selected = [row for row in scored if start < _date(row) <= end]
    if len(selected) < minimum_runs:
        return None
    values = [float(row["standardized_pace"]) for row in selected]
    measurement_sigmas = [
        max(
            MIN_MEASUREMENT_SIGMA_MIN_MILE,
            float(row["uncertainty_95"] or 0.0) / 1.96,
        )
        for row in selected
    ]
    context_weights = [
        max(0.0, min(1.0, float(row.get("trend_weight", 1.0))))
        for row in selected
    ]
    base_weights = [
        context / sigma**2
        for context, sigma in zip(context_weights, measurement_sigmas)
    ]
    weights = list(base_weights)
    estimate = sum(weight * value for weight, value in zip(weights, values)) / sum(weights)
    for _ in range(8 if robust else 0):
        residuals = [value - estimate for value in values]
        center = median(residuals)
        scale = max(
            5.0 / 60.0,
            1.4826 * median(abs(residual - center) for residual in residuals),
        )
        cutoff = 1.345 * scale
        huber_factors = [min(1.0, cutoff / max(abs(residual), 1e-12)) for residual in residuals]
        weights = [base * factor for base, factor in zip(base_weights, huber_factors)]
        estimate = sum(weight * value for weight, value in zip(weights, values)) / sum(weights)

    weight_sum = sum(weights)
    effective_n = weight_sum**2 / sum(weight**2 for weight in weights)
    weighted_variance = sum(
        weight * (value - estimate) ** 2 for weight, value in zip(weights, values)
    ) / weight_sum
    measurement_se = sqrt(1.0 / weight_sum)
    between_run_se = sqrt(weighted_variance / max(1.0, effective_n))
    uncertainty_95 = 1.96 * sqrt(measurement_se**2 + between_run_se**2)
    coverage_span_days = (
        max(_date(row) for row in selected) - min(_date(row) for row in selected)
    ).total_seconds() / 86400.0
    return {
        "pace_min_mile": estimate,
        "uncertainty_95_min_mile": uncertainty_95,
        "run_count": len(selected),
        "full_weight_run_equivalents": sum(context_weights),
        "effective_run_count": effective_n,
        "between_run_spread_min_mile": sqrt(weighted_variance),
        "coverage_span_days": coverage_span_days,
        "coverage_fraction": min(1.0, coverage_span_days / float(days)),
        "start_time_utc": min(_date(row) for row in selected).isoformat(),
        "end_time_utc": max(_date(row) for row in selected).isoformat(),
    }


def _comparison(
    current: dict[str, Any], prior: dict[str, Any] | None, label: str
) -> dict[str, Any] | None:
    if prior is None:
        return None
    delta = float(current["pace_min_mile"]) - float(prior["pace_min_mile"])
    delta_se = sqrt(
        (float(current["uncertainty_95_min_mile"]) / 1.96) ** 2
        + (float(prior["uncertainty_95_min_mile"]) / 1.96) ** 2
    )
    directional = _directional_evidence(delta, delta_se)
    # Preserve the original strong-direction field for coaching consumers.
    # Display clients use ``directional_interpretation`` plus
    # ``evidence_strength`` to distinguish likely from clear change.
    direction = (
        directional["directional_interpretation"]
        if directional["evidence_strength"] == "clear"
        else "stable_or_uncertain"
    )
    return {
        "comparison": label,
        "pace_change_min_mile": delta,
        "pace_change_seconds_per_mile": delta * 60.0,
        "uncertainty_95_seconds_per_mile": directional[
            "uncertainty_95_min_mile"
        ]
        * 60.0,
        "probability_faster": directional["probability_faster"],
        "direction": direction,
        "directional_interpretation": directional[
            "directional_interpretation"
        ],
        "evidence_strength": directional["evidence_strength"],
        "prior": prior,
    }


def build_fitness_analytics(
    runs: list[dict[str, Any]],
    window_days: int = 28,
    target_hr_bpm: float | None = None,
    *,
    robust: bool = True,
    evaluation_time: datetime | None = None,
) -> dict[str, Any]:
    """Build a descriptive fitness state without refitting environmental effects."""
    scored = sorted(
        [row for row in runs if row.get("standardized_pace") is not None],
        key=_date,
    )
    if not scored:
        return {"available": False, "reason": "No scored runs"}

    historical = []
    for row in scored:
        estimate = _window_estimate(scored, _date(row), window_days, minimum_runs=3, robust=robust)
        if estimate is None:
            continue
        estimate = {**estimate, "as_of_utc": _date(row).isoformat()}
        historical.append(estimate)

    anchor = _date(scored[-1])
    current = _window_estimate(scored, anchor, window_days, minimum_runs=1, robust=robust)
    assert current is not None
    prior_window = _window_estimate(
        scored, anchor - timedelta(days=window_days), window_days, minimum_runs=1, robust=robust
    )
    prior_90 = _window_estimate(
        scored, anchor - timedelta(days=90), window_days, minimum_runs=1, robust=robust
    )
    change_prior = _comparison(current, prior_window, f"preceding {window_days} days")
    change_90 = _comparison(current, prior_90, f"{window_days}-day fitness 90 days earlier")
    within_window_trend = _window_trend(
        scored,
        anchor,
        window_days,
        robust=robust,
    )

    comparable = [
        item
        for item in historical
        if item["run_count"] >= 3 and item["coverage_fraction"] >= 0.5
    ]
    percentile = None
    best = None
    if comparable:
        percentile = 100.0 * sum(
            float(item["pace_min_mile"]) >= float(current["pace_min_mile"])
            for item in comparable
        ) / len(comparable)
        sustained = [item for item in comparable if item["run_count"] >= 5]
        best = min(sustained, key=lambda item: float(item["pace_min_mile"])) if sustained else None

    if evaluation_time is None:
        evaluation_time = datetime.now(timezone.utc)
    elif evaluation_time.tzinfo is None:
        raise ValueError("evaluation_time must include a timezone")
    freshness_days = max(
        0.0,
        (
            evaluation_time.astimezone(timezone.utc)
            - anchor.astimezone(timezone.utc)
        ).total_seconds()
        / 86400.0,
    )
    if current["run_count"] >= 6 and freshness_days <= 7:
        evidence = "good"
    elif current["run_count"] >= 3 and freshness_days <= 21:
        evidence = "moderate"
    else:
        evidence = "limited"

    # A directional comparison is only as strong as its weaker time window.
    # Current-run freshness alone must not turn a sparse prior period into
    # high-confidence evidence of improvement or decline.
    comparison_evidence = "limited"
    if prior_window is not None:
        if (
            current["run_count"] >= 6
            and prior_window["run_count"] >= 6
            and current["coverage_fraction"] >= 0.5
            and prior_window["coverage_fraction"] >= 0.5
            and freshness_days <= 7
        ):
            comparison_evidence = "good"
        elif (
            current["run_count"] >= 3
            and prior_window["run_count"] >= 3
            and current["coverage_fraction"] >= 0.25
            and prior_window["coverage_fraction"] >= 0.25
            and freshness_days <= 21
        ):
            comparison_evidence = "moderate"

    status = change_prior["direction"] if change_prior else "insufficient_comparison"
    best_gap = (
        (float(current["pace_min_mile"]) - float(best["pace_min_mile"])) * 60.0
        if best is not None
        else None
    )
    return {
        "available": True,
        "window_days": window_days,
        "target_hr_bpm": target_hr_bpm,
        "definition": (
            f"Robust trailing {window_days}-day estimate of reference-condition pace"
            + (f" at {target_hr_bpm:g} bpm" if target_hr_bpm else " at the comparison heart rate")
        ),
        "as_of_utc": anchor.isoformat(),
        "current": current,
        "change_prior_window": change_prior,
        "change_90d": change_90,
        "within_window_trend": within_window_trend,
        "status": status,
        "personal_history_percentile": percentile,
        "percentile_definition": "100 means fastest sustained 28-day level in the scored history; this is not a population percentile",
        "best_sustained": best,
        "seconds_per_mile_from_best": best_gap,
        "evidence_quality": evidence,
        "comparison_evidence_quality": comparison_evidence,
        "days_since_latest_scored_run": freshness_days,
        "historical_estimate_count": len(comparable),
        "historical": historical,
    }


def build_fitness_analytics_set(
    runs: list[dict[str, Any]],
    windows: tuple[int, ...] = (14, 28, 42, 56, 90),
    target_hr_bpm: float | None = None,
) -> dict[str, Any]:
    return {
        "default_window_days": 28,
        "available_windows": list(windows),
        "by_window": {
            str(days): build_fitness_analytics(runs, days, target_hr_bpm) for days in windows
        },
    }
