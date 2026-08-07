from __future__ import annotations

from datetime import datetime, timedelta, timezone

from run_analysis.analytics import build_fitness_analytics, build_fitness_analytics_set


def _run(day: int, pace: float) -> dict:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=day)
    return {
        "start_time_utc": start.isoformat(),
        "standardized_pace": pace,
        "uncertainty_95": 0.12,
    }


def test_recent_improvement_is_described_without_changing_run_scores() -> None:
    runs = [_run(day, 11.0) for day in range(0, 56, 7)]
    runs += [_run(day, 10.0) for day in range(56, 112, 7)]
    original = [row["standardized_pace"] for row in runs]
    analysis = build_fitness_analytics(runs, 28)
    assert analysis["available"]
    assert analysis["current"]["pace_min_mile"] == 10.0
    assert analysis["change_90d"]["direction"] == "improving"
    assert [row["standardized_pace"] for row in runs] == original


def test_adjustable_windows_are_precomputed() -> None:
    runs = [_run(day, 10.0 + day / 1000) for day in range(0, 120, 5)]
    analyses = build_fitness_analytics_set(runs)
    assert analyses["default_window_days"] == 28
    assert analyses["available_windows"] == [14, 28, 42, 56, 90]
    assert analyses["by_window"]["14"]["window_days"] == 14
    assert analyses["by_window"]["90"]["window_days"] == 90
    assert analyses["by_window"]["14"]["current"]["run_count"] < analyses["by_window"]["90"]["current"]["run_count"]


def test_health_context_can_contribute_at_reduced_weight() -> None:
    runs = [_run(day, 10.0) for day in (0, 7, 14)]
    illness = _run(20, 14.0)
    illness["trend_weight"] = 0.25
    runs.append(illness)
    analysis = build_fitness_analytics(runs, 28)
    current = analysis["current"]
    assert current["run_count"] == 4
    assert current["full_weight_run_equivalents"] == 3.25
    assert current["pace_min_mile"] < 11.0
    assert current["pace_min_mile"] > 10.0


def test_change_inside_combined_uncertainty_is_not_forced_directional() -> None:
    runs = [_run(day, 10.0) for day in (0, 7, 14, 21)]
    runs += [_run(day, 10.2) for day in (28, 35, 42, 49)]
    for row in runs:
        row["uncertainty_95"] = 0.5
    analysis = build_fitness_analytics(runs, 28)
    change = analysis["change_prior_window"]
    assert abs(change["pace_change_seconds_per_mile"]) < change["uncertainty_95_seconds_per_mile"]
    assert change["direction"] == "stable_or_uncertain"

