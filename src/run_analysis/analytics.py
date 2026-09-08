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


def _weighted_plane(
    x_values: list[float],
    y_values: list[float],
    context_values: list[float],
    weights: list[float],
) -> tuple[float, float, float, float, float] | None:
    """Fit ``y = intercept + x + context`` and retain x's unique support.

    The final value is the weighted variance in ``x`` left after projecting it
    on the context covariate.  It approaches zero when run distance and period
    (or run distance and date) are inseparable, which is exactly when a
    distance-adjusted fitness claim should be withheld.
    """

    weight_sum = sum(weights)
    if weight_sum <= 0:
        return None
    x_mean = sum(w * x for w, x in zip(weights, x_values)) / weight_sum
    y_mean = sum(w * y for w, y in zip(weights, y_values)) / weight_sum
    context_mean = (
        sum(w * value for w, value in zip(weights, context_values)) / weight_sum
    )
    centered_x = [value - x_mean for value in x_values]
    centered_y = [value - y_mean for value in y_values]
    centered_context = [value - context_mean for value in context_values]
    sxx = sum(w * value**2 for w, value in zip(weights, centered_x))
    scc = sum(w * value**2 for w, value in zip(weights, centered_context))
    sxc = sum(
        w * x * context
        for w, x, context in zip(weights, centered_x, centered_context)
    )
    sxy = sum(
        w * x * y for w, x, y in zip(weights, centered_x, centered_y)
    )
    scy = sum(
        w * context * y
        for w, context, y in zip(weights, centered_context, centered_y)
    )
    if sxx <= 1e-12 or scc <= 1e-12:
        return None
    determinant = sxx * scc - sxc**2
    scale = max(sxx * scc, 1.0)
    if determinant <= 1e-12 * scale:
        return None
    x_slope = (sxy * scc - scy * sxc) / determinant
    context_slope = (scy * sxx - sxy * sxc) / determinant
    intercept = y_mean - x_slope * x_mean - context_slope * context_mean
    residual_x_variance_sum = sxx - sxc**2 / scc
    if residual_x_variance_sum <= 1e-12:
        return None
    return (
        intercept,
        x_slope,
        context_slope,
        x_mean,
        residual_x_variance_sum,
    )


def _robust_plane(
    x_values: list[float],
    y_values: list[float],
    context_values: list[float],
    base_weights: list[float],
    *,
    robust: bool,
) -> tuple[tuple[float, float, float, float, float], list[float]] | None:
    weights = list(base_weights)
    fitted = _weighted_plane(x_values, y_values, context_values, weights)
    if fitted is None:
        return None
    for _ in range(8 if robust else 0):
        intercept, x_slope, context_slope, _, _ = fitted
        residuals = [
            y - (intercept + x_slope * x + context_slope * context)
            for x, y, context in zip(x_values, y_values, context_values)
        ]
        center = median(residuals)
        scale = max(
            5.0 / 60.0,
            1.4826 * median(abs(residual - center) for residual in residuals),
        )
        cutoff = 1.345 * scale
        weights = [
            base * min(1.0, cutoff / max(abs(residual), 1e-12))
            for base, residual in zip(base_weights, residuals)
        ]
        fitted = _weighted_plane(x_values, y_values, context_values, weights)
        if fitted is None:
            return None
    return fitted, weights


def _window_trend(
    scored: list[dict[str, Any]],
    end: datetime,
    days: int,
    *,
    robust: bool = True,
    distance_effect_min_mile_per_added_mile: float | None = None,
    distance_effect_se_min_mile_per_added_mile: float | None = None,
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
    raw_y_values = [float(row["standardized_pace"]) for row in selected]
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
    distance_values = [row.get("distance_miles") for row in selected]
    distance_adjusted = bool(
        distance_effect_min_mile_per_added_mile is not None
        and all(value is not None and float(value) > 0 for value in distance_values)
    )
    distance_slope = (
        float(distance_effect_min_mile_per_added_mile)
        if distance_adjusted
        else None
    )
    y_values = [
        pace - (distance_slope or 0.0) * float(distance or 0.0)
        for pace, distance in zip(raw_y_values, distance_values)
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
        weights = [
            base * min(1.0, cutoff / max(abs(residual), 1e-12))
            for base, residual in zip(base_weights, residuals)
        ]
        fitted = _weighted_line(x_values, y_values, weights)
        if fitted is None:
            return None
    intercept, slope, x_mean, x_variance_sum = fitted
    fitted_values = [intercept + slope * x for x in x_values]
    residuals = [
        value - fitted_value for value, fitted_value in zip(y_values, fitted_values)
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
    distance_calibration_slope_se = 0.0
    if distance_adjusted and distance_effect_se_min_mile_per_added_mile:
        distance_line = _weighted_line(
            x_values,
            [float(value) for value in distance_values],
            weights,
        )
        if distance_line is not None:
            distance_per_day = distance_line[1]
            distance_calibration_slope_se = abs(distance_per_day) * float(
                distance_effect_se_min_mile_per_added_mile
            )
    slope_se = sqrt(
        measurement_slope_se**2
        + between_run_slope_se**2
        + distance_calibration_slope_se**2
    )
    # Describe change only across the span actually supported by observations.
    # Multiplying a 17-day evidence span by a nominal 28-day window exaggerates
    # the displayed magnitude and creates a hard-window boundary jump after a
    # break in training. Scaling the estimate and its uncertainty together
    # preserves the directional probability without extrapolating beyond data.
    coverage_span_days = max(x_values) - min(x_values)
    window_change = slope * coverage_span_days
    window_change_se = slope_se * coverage_span_days
    directional = _directional_evidence(window_change, window_change_se)
    return {
        "basis": (
            f"weighted slope across the observed {coverage_span_days:.1f}-day "
            f"span within the last {days} days"
            + (", controlling for run distance" if distance_adjusted else "")
        ),
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
        "distance_adjusted": distance_adjusted,
        "distance_effect_seconds_per_mile_per_added_mile": (
            distance_slope * 60.0 if distance_slope is not None else None
        ),
        "distance_effect_uncertainty_95_seconds_per_mile_per_added_mile": (
            CLEAR_DIRECTION_Z
            * float(distance_effect_se_min_mile_per_added_mile)
            * 60.0
            if distance_adjusted
            and distance_effect_se_min_mile_per_added_mile is not None
            else None
        ),
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


def _distance_adjusted_comparison(
    scored: list[dict[str, Any]],
    *,
    current_end: datetime,
    prior_end: datetime,
    days: int,
    current: dict[str, Any],
    prior: dict[str, Any] | None,
    label: str,
    robust: bool,
) -> dict[str, Any] | None:
    """Compare two periods while estimating their run-distance effect jointly.

    This is an ANCOVA-style comparison. The period coefficient answers how the
    two periods differ at the same run distance. If distance and period are
    inseparable, the fit is rank deficient and no directional claim is made.
    """

    if prior is None:
        return None
    current_start = current_end - timedelta(days=days)
    prior_start = prior_end - timedelta(days=days)
    current_rows = [
        row for row in scored if current_start < _date(row) <= current_end
    ]
    prior_rows = [row for row in scored if prior_start < _date(row) <= prior_end]
    combined = prior_rows + current_rows
    if (
        len(current_rows) < 2
        or len(prior_rows) < 2
        or not all(
            row.get("distance_miles") is not None
            and float(row["distance_miles"]) > 0
            for row in combined
        )
    ):
        # Legacy/tooling rows may not carry distance. Preserve their existing
        # aggregate while production data graduate to the stronger comparison.
        return _comparison(current, prior, label)

    period_values = [0.0] * len(prior_rows) + [1.0] * len(current_rows)
    pace_values = [float(row["standardized_pace"]) for row in combined]
    distance_values = [float(row["distance_miles"]) for row in combined]
    measurement_sigmas = [
        max(
            MIN_MEASUREMENT_SIGMA_MIN_MILE,
            float(row["uncertainty_95"] or 0.0) / CLEAR_DIRECTION_Z,
        )
        for row in combined
    ]
    context_weights = [
        max(0.0, min(1.0, float(row.get("trend_weight", 1.0))))
        for row in combined
    ]
    base_weights = [
        context / sigma**2
        for context, sigma in zip(context_weights, measurement_sigmas)
    ]
    plane = _robust_plane(
        period_values,
        pace_values,
        distance_values,
        base_weights,
        robust=robust,
    )
    if plane is None:
        return None
    fitted, weights = plane
    intercept, period_change, distance_slope, _, period_variance_sum = fitted
    residuals = [
        pace - (intercept + period_change * period + distance_slope * distance)
        for pace, period, distance in zip(
            pace_values, period_values, distance_values
        )
    ]
    weight_sum = sum(weights)
    effective_n = weight_sum**2 / sum(weight**2 for weight in weights)
    weighted_variance = sum(
        weight * residual**2 for weight, residual in zip(weights, residuals)
    ) / weight_sum
    normalized_period_variance = period_variance_sum / weight_sum
    measurement_se = sqrt(1.0 / period_variance_sum)
    between_run_se = sqrt(
        weighted_variance
        / max(1e-12, effective_n * normalized_period_variance)
    )
    change_se = sqrt(measurement_se**2 + between_run_se**2)
    period_line_for_distance = _weighted_line(
        period_values, distance_values, weights
    )
    distance_effect_se = None
    if period_line_for_distance is not None:
        distance_intercept, distance_by_period, _, _ = period_line_for_distance
        distance_residual_variance_sum = sum(
            weight
            * (distance - (distance_intercept + distance_by_period * period)) ** 2
            for weight, distance, period in zip(
                weights, distance_values, period_values
            )
        )
        if distance_residual_variance_sum > 1e-12:
            distance_measurement_se = sqrt(1.0 / distance_residual_variance_sum)
            normalized_distance_variance = (
                distance_residual_variance_sum / weight_sum
            )
            distance_between_run_se = sqrt(
                weighted_variance
                / max(1e-12, effective_n * normalized_distance_variance)
            )
            distance_effect_se = sqrt(
                distance_measurement_se**2 + distance_between_run_se**2
            )
    directional = _directional_evidence(period_change, change_se)
    direction = (
        directional["directional_interpretation"]
        if directional["evidence_strength"] == "clear"
        else "stable_or_uncertain"
    )
    current_weight = sum(weights[len(prior_rows) :])
    prior_weight = sum(weights[: len(prior_rows)])
    return {
        "comparison": f"{label}, controlling for run distance",
        "pace_change_min_mile": period_change,
        "pace_change_seconds_per_mile": period_change * 60.0,
        "uncertainty_95_seconds_per_mile": (
            directional["uncertainty_95_min_mile"] * 60.0
        ),
        "probability_faster": directional["probability_faster"],
        "direction": direction,
        "directional_interpretation": directional[
            "directional_interpretation"
        ],
        "evidence_strength": directional["evidence_strength"],
        "prior": prior,
        "distance_adjusted": True,
        "distance_effect_seconds_per_mile_per_added_mile": distance_slope * 60.0,
        "distance_effect_uncertainty_95_seconds_per_mile_per_added_mile": (
            CLEAR_DIRECTION_Z * distance_effect_se * 60.0
            if distance_effect_se is not None
            else None
        ),
        "current_weighted_distance_miles": (
            sum(
                weight * distance
                for weight, distance in zip(
                    weights[len(prior_rows) :],
                    distance_values[len(prior_rows) :],
                )
            )
            / current_weight
        ),
        "prior_weighted_distance_miles": (
            sum(
                weight * distance
                for weight, distance in zip(
                    weights[: len(prior_rows)], distance_values[: len(prior_rows)]
                )
            )
            / prior_weight
        ),
        "effective_run_count": effective_n,
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
    change_prior = _distance_adjusted_comparison(
        scored,
        current_end=anchor,
        prior_end=anchor - timedelta(days=window_days),
        days=window_days,
        current=current,
        prior=prior_window,
        label=f"preceding {window_days} days",
        robust=robust,
    )
    change_90 = _distance_adjusted_comparison(
        scored,
        current_end=anchor,
        prior_end=anchor - timedelta(days=90),
        days=window_days,
        current=current,
        prior=prior_90,
        label=f"{window_days}-day fitness 90 days earlier",
        robust=robust,
    )
    within_window_trend = _window_trend(
        scored,
        anchor,
        window_days,
        robust=robust,
        distance_effect_min_mile_per_added_mile=(
            float(
                change_prior[
                    "distance_effect_seconds_per_mile_per_added_mile"
                ]
            )
            / 60.0
            if change_prior and change_prior.get("distance_adjusted")
            else None
        ),
        distance_effect_se_min_mile_per_added_mile=(
            float(
                change_prior[
                    "distance_effect_uncertainty_95_seconds_per_mile_per_added_mile"
                ]
            )
            / (CLEAR_DIRECTION_Z * 60.0)
            if change_prior
            and change_prior.get(
                "distance_effect_uncertainty_95_seconds_per_mile_per_added_mile"
            )
            is not None
            else None
        ),
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
