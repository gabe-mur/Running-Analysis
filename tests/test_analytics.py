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



def test_analytics_definition_names_the_configured_comparison_heart_rate() -> None:
    """Changing the comparison HR must change what the app says, not only what
    it calculates."""
    runs = [
        {"start_time_utc": f"2026-05-{day:02d}T12:00:00+00:00", "standardized_pace": 9.0, "uncertainty_95": 0.2}
        for day in range(1, 10)
    ]
    at_147 = build_fitness_analytics(runs, 28, target_hr_bpm=147)
    assert at_147["target_hr_bpm"] == 147
    assert "147 bpm" in at_147["definition"]
    assert "145" not in at_147["definition"]
    # With no configured value the copy stays honest rather than inventing one.
    assert "comparison heart rate" in build_fitness_analytics(runs, 28)["definition"]


def _series(paces: list[float]) -> list[dict]:
    from datetime import datetime, timedelta, timezone

    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return [
        {
            "start_time_utc": (start + timedelta(days=index * 2)).isoformat(),
            "standardized_pace": pace,
            "uncertainty_95": 0.25,
            "trend_weight": 1.0,
        }
        for index, pace in enumerate(paces)
    ]


def test_huber_layer_absorbs_a_corrupted_run_far_more_than_it_delays_a_real_step() -> None:
    """The robust layer's cost is a small lag; its benefit is large. See
    scripts/huber_sensitivity.py for the full comparison this pins down."""
    stable = [9.5] * 13
    corrupted = _series(stable + [12.5])
    clean = _series(stable + [9.5])

    def level(rows, robust):
        return build_fitness_analytics(rows, 28, robust=robust)["current"]["pace_min_mile"]

    robust_shift = level(corrupted, True) - level(clean, True)
    plain_shift = level(corrupted, False) - level(clean, False)
    assert robust_shift < plain_shift / 3

    # A genuine sustained step is still tracked, not suppressed.
    stepped = _series([9.5] * 13 + [9.0] * 8)
    assert level(stepped, True) < 9.35
