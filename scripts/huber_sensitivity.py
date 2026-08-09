"""Compare the trailing fitness trend with and without Huber reweighting.

The trend smoother already down-weights runs for measurement uncertainty and
for illness/workout context. On top of that it applies a Huber residual weight,
which shrinks the influence of runs far from the current estimate. That is the
right behaviour for a bad GPS day and the wrong behaviour for a genuine
step-down or step-up in fitness, which is exactly a run far from the current
estimate.

This script answers two questions:

1. On the athlete's real history, how much does the robust layer move the
   trailing estimate and the direction call?
2. On synthetic histories containing a known step change, how many runs does
   each version take to register it?

Run with the project's interpreter:

    .venv/bin/python scripts/huber_sensitivity.py
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_analysis.analytics import build_fitness_analytics  # noqa: E402
from run_analysis.config import load_config  # noqa: E402
from run_analysis.db import connect  # noqa: E402
from run_analysis.progress import _scored_runs, _sessions  # noqa: E402

WINDOWS = (14, 28, 56, 90)


def _load_real_rows() -> list[dict]:
    config = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    with connect(config["paths"]["database"]) as connection:
        _sessions_list, details = _sessions(connection)
        analytics_rows, _points, _steady, _steady_points = _scored_runs(connection, details)
    return analytics_rows


def _report_real(rows: list[dict]) -> None:
    print(f"REAL HISTORY  ({len(rows)} runs contributing to the trend)")
    print(f"  {'window':>7}  {'robust':>9}  {'plain':>9}  {'delta s/mi':>11}  direction")
    for days in WINDOWS:
        robust = build_fitness_analytics(rows, days, robust=True)
        plain = build_fitness_analytics(rows, days, robust=False)
        if not robust.get("available") or not plain.get("available"):
            print(f"  {days:>7}  unavailable")
            continue
        r_pace = robust["current"]["pace_min_mile"]
        p_pace = plain["current"]["pace_min_mile"]
        agree = "same" if robust["status"] == plain["status"] else "DIFFERS"
        print(
            f"  {days:>7}  {r_pace:>9.3f}  {p_pace:>9.3f}  {(p_pace - r_pace) * 60:>+11.1f}"
            f"  {robust['status']} / {plain['status']} ({agree})"
        )

    print()
    print("  Change vs the preceding window (the number the dashboard reports):")
    for days in WINDOWS:
        robust = build_fitness_analytics(rows, days, robust=True)
        plain = build_fitness_analytics(rows, days, robust=False)
        r_change = (robust.get("change_prior_window") or {}).get("pace_change_seconds_per_mile")
        p_change = (plain.get("change_prior_window") or {}).get("pace_change_seconds_per_mile")
        if r_change is None or p_change is None:
            print(f"    {days:>3}d: no comparable prior window")
            continue
        print(
            f"    {days:>3}d: robust {r_change:+6.1f} s/mi   plain {p_change:+6.1f} s/mi"
            f"   (robust reports {abs(r_change) / abs(p_change) * 100 if p_change else float('nan'):.0f}% of the plain magnitude)"
        )


def _synthetic(
    step_seconds_per_mile: float,
    runs_before: int = 20,
    runs_after: int = 12,
    every_days: int = 2,
    noise_seconds: float = 12.0,
    seed: int = 7,
) -> list[dict]:
    """A clean history that steps to a new level partway through."""
    import random

    rng = random.Random(seed)
    base = 9.5
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    for index in range(runs_before + runs_after):
        level = base + (step_seconds_per_mile / 60.0 if index >= runs_before else 0.0)
        rows.append(
            {
                "start_time_utc": (start + timedelta(days=index * every_days)).isoformat(),
                "standardized_pace": level + rng.gauss(0, noise_seconds / 60.0),
                "uncertainty_95": 0.25,
                "trend_weight": 1.0,
                "_is_after": index >= runs_before,
            }
        )
    return rows


def _runs_to_detect(rows: list[dict], step_seconds: float, days: int, robust: bool) -> int | None:
    """How many post-step runs before the trailing estimate has moved halfway."""
    before = [row for row in rows if not row["_is_after"]]
    after = [row for row in rows if row["_is_after"]]
    baseline = build_fitness_analytics(before, days, robust=robust)
    if not baseline.get("available"):
        return None
    start_level = baseline["current"]["pace_min_mile"]
    target = start_level + (step_seconds / 60.0) * 0.5
    for count in range(1, len(after) + 1):
        analysis = build_fitness_analytics(before + after[:count], days, robust=robust)
        level = analysis["current"]["pace_min_mile"]
        moved = level >= target if step_seconds > 0 else level <= target
        if moved:
            return count
    return None


def _report_synthetic() -> None:
    print()
    print("SYNTHETIC STEP CHANGES  (post-step runs needed to move the 28-day estimate halfway)")
    print(f"  {'step':>12}  {'robust':>8}  {'plain':>8}  penalty")
    for step in (-45.0, -30.0, -15.0, 15.0, 30.0, 45.0):
        rows = _synthetic(step)
        robust = _runs_to_detect(rows, step, 28, robust=True)
        plain = _runs_to_detect(rows, step, 28, robust=False)
        label = lambda value: "never" if value is None else str(value)  # noqa: E731
        penalty = (
            f"+{robust - plain} runs"
            if robust is not None and plain is not None and robust > plain
            else "none"
            if robust is not None and plain is not None
            else "robust never detects" if robust is None else "plain never detects"
        )
        print(f"  {step:>+8.0f} s/mi  {label(robust):>8}  {label(plain):>8}  {penalty}")


def _report_outlier_protection() -> None:
    """The reason the robust layer exists: one bad run should not move the level."""
    print()
    print("OUTLIER PROTECTION  (one 3 min/mi GPS-corrupted run added to a stable history)")
    clean = _synthetic(0.0, runs_before=24, runs_after=0)
    corrupted = list(clean)
    corrupted[-1] = {**corrupted[-1], "standardized_pace": corrupted[-1]["standardized_pace"] + 3.0}
    for robust in (True, False):
        good = build_fitness_analytics(clean, 28, robust=robust)["current"]["pace_min_mile"]
        bad = build_fitness_analytics(corrupted, 28, robust=robust)["current"]["pace_min_mile"]
        name = "robust" if robust else "plain "
        print(f"  {name}: {(bad - good) * 60:+.1f} s/mi shift from the single bad run")


def main() -> None:
    rows = _load_real_rows()
    if rows:
        _report_real(rows)
    else:
        print("REAL HISTORY: no scored runs in the database; skipping.")
    _report_synthetic()
    _report_outlier_protection()
    print()
    print(
        "Read together: the penalty column is the cost of the robust layer and the "
        "outlier shift is its benefit. Keep the layer only if the second is clearly "
        "larger than the first."
    )
    _ = statistics  # referenced for readers extending this script


if __name__ == "__main__":
    main()
