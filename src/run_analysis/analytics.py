"""Interpret independent per-run scores as a changing fitness time series."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from math import erf, sqrt
from statistics import median
from typing import Any


MIN_MEASUREMENT_SIGMA_MIN_MILE = 10.0 / 60.0


def _date(row: dict[str, Any]) -> datetime:
    return datetime.fromisoformat(row["start_time_utc"])


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def _window_estimate(
    scored: list[dict[str, Any]], end: datetime, days: int, minimum_runs: int = 1
) -> dict[str, Any] | None:
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
    for _ in range(8):
        residuals = [value - estimate for value in values]
        center = median(residuals)
        scale = max(
            5.0 / 60.0,
            1.4826 * median(abs(residual - center) for residual in residuals),
        )
        cutoff = 1.345 * scale
        robust = [min(1.0, cutoff / max(abs(residual), 1e-12)) for residual in residuals]
        weights = [base * factor for base, factor in zip(base_weights, robust)]
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
    delta_uncertainty_95 = 1.96 * delta_se
    probability_faster = _normal_cdf(-delta / max(delta_se, 1e-12))
    if abs(delta) <= delta_uncertainty_95:
        direction = "stable_or_uncertain"
    elif probability_faster >= 0.80:
        direction = "improving"
    elif probability_faster <= 0.20:
        direction = "declining"
    else:
        direction = "stable_or_uncertain"
    return {
        "comparison": label,
        "pace_change_min_mile": delta,
        "pace_change_seconds_per_mile": delta * 60.0,
        "uncertainty_95_seconds_per_mile": delta_uncertainty_95 * 60.0,
        "probability_faster": probability_faster,
        "direction": direction,
        "prior": prior,
    }


def build_fitness_analytics(
    runs: list[dict[str, Any]], window_days: int = 28
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
        estimate = _window_estimate(scored, _date(row), window_days, minimum_runs=3)
        if estimate is None:
            continue
        estimate = {**estimate, "as_of_utc": _date(row).isoformat()}
        historical.append(estimate)

    anchor = _date(scored[-1])
    current = _window_estimate(scored, anchor, window_days, minimum_runs=1)
    assert current is not None
    prior_window = _window_estimate(
        scored, anchor - timedelta(days=window_days), window_days, minimum_runs=1
    )
    prior_90 = _window_estimate(scored, anchor - timedelta(days=90), window_days, minimum_runs=1)
    change_prior = _comparison(current, prior_window, f"preceding {window_days} days")
    change_90 = _comparison(current, prior_90, f"{window_days}-day fitness 90 days earlier")

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

    now = datetime.now(timezone.utc)
    freshness_days = max(0.0, (now - anchor.astimezone(timezone.utc)).total_seconds() / 86400.0)
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
        "definition": f"Robust trailing {window_days}-day estimate of reference-condition pace at 145 bpm",
        "as_of_utc": anchor.isoformat(),
        "current": current,
        "change_prior_window": change_prior,
        "change_90d": change_90,
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
    runs: list[dict[str, Any]], windows: tuple[int, ...] = (14, 28, 42, 56, 90)
) -> dict[str, Any]:
    return {
        "default_window_days": 28,
        "available_windows": list(windows),
        "by_window": {
            str(days): build_fitness_analytics(runs, days) for days in windows
        },
    }
