"""Print a read-only multi-week projection of perfect plan adherence."""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from run_analysis.adherence_projection import (
    HumanAdherenceProfile,
    runs_from_summaries,
    simulate_adherence,
)
from run_analysis.config import load_config, resolve_project_path
from run_analysis.db import connect, initialize
from run_analysis.recommendation_service import current_fitness_state
from run_analysis.run_feedback import list_runs


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--weeks", type=int, default=13)
    parser.add_argument(
        "--mode",
        choices=("perfect", "human"),
        default="perfect",
        help="Adherence behavior to simulate (default: perfect)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260902,
        help="Reproducible random seed for --mode human",
    )
    parser.add_argument(
        "--replan-days",
        type=int,
        default=1,
        help="Days between planner reloads (default: 1, matching normal app use)",
    )
    parser.add_argument(
        "--trace-replans",
        action="store_true",
        help="Print each receding-horizon plan and the work committed from it",
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = load_config(root / "config.yaml")
    database = resolve_project_path(root, config["paths"]["database"])
    zone = ZoneInfo(str(config["timezone_default"]))
    start_at = datetime.now(zone)
    with connect(database) as connection:
        initialize(connection)
        state = current_fitness_state(connection, config)
        runs = runs_from_summaries(list_runs(connection, limit=5000))
    weeks = simulate_adherence(
        state,
        runs,
        config,
        start_at,
        weeks=max(1, args.weeks),
        replan_interval_days=args.replan_days,
        human_profile=(
            HumanAdherenceProfile(seed=args.seed)
            if args.mode == "human"
            else None
        ),
    )
    print(
        "week | start | committed attempts/actual | target | prescribed | actual | "
        "capacity | opening continuous load | peak rolling 7d (diagnostic) | "
        "streak | workouts"
    )
    print("--- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---")
    for item in weeks:
        opening = (
            f"{item.opening_acute_ratio * 100:.0f}%"
            if item.opening_acute_ratio is not None
            else "—"
        )
        print(
            f"{item.week} | {item.start_date} | "
            f"{item.planned_run_count}/{item.run_count} | "
            f"{item.target_low_miles:.1f}–{item.target_high_miles:.1f} | "
            f"{item.prescribed_low_miles:.1f}–{item.prescribed_high_miles:.1f} | "
            f"{item.assumed_completed_miles:.1f} | "
            f"{item.capacity_reference_miles:.1f} | {opening} | "
            f"{item.peak_rolling_7d_miles:.1f} | "
            f"{item.maximum_consecutive_run_days} | "
            + "; ".join(item.workouts)
        )
        if args.trace_replans:
            for trace in item.replan_trace:
                print(f"  - {trace}")
        if item.adherence_events:
            for event in item.adherence_events:
                print(f"  * {event}")
    print(
        "\nSummary: "
        f"{sum(item.planned_run_count for item in weeks)} committed attempts, "
        f"{sum(item.run_count for item in weeks)} completed, "
        f"{sum(item.skipped_run_count for item in weeks)} skipped, "
        f"{sum(item.unscheduled_run_count for item in weeks)} unscheduled; "
        f"{sum(item.assumed_completed_miles for item in weeks):.1f} actual miles; "
        f"{sum(item.assumed_completed_miles for item in weeks) / len(weeks):.1f} "
        "miles per seven days over the full projection; "
        f"maximum streak {max(item.maximum_consecutive_run_days for item in weeks)} days; "
        "diagnostic peak rolling-7 load "
        f"{max(item.peak_rolling_7d_miles for item in weeks):.1f} miles."
    )
    print(
        "\nAssumptions: "
        + (
            "every run is completed at the distance midpoint with a normal response. "
            if args.mode == "perfect"
            else f"bounded human adherence with seed {args.seed}: occasional skips, "
            "distance and intensity drift, workout substitutions, unscheduled easy "
            "runs, 2–3 short trips, and a 50% chance of one seven-day vacation. "
        )
        + f"The planner is regenerated every {args.replan_days} day(s). "
        "Future pace adaptation, weather, sleep, soreness, illness, and injury are not invented."
    )
    print(
        "Report note: seven-day rows are presentation slices, not optimizer "
        "budgets. Interpret a sparse or dense row using adjacent dated runs, "
        "continuous opening load, and the full-projection average. Prescribed "
        "distance is the sum of daily committed attempts; after a miss, a "
        "replacement prescribed on a later replan is another attempt, not "
        "mileage that coexisted in one live plan."
    )


if __name__ == "__main__":
    main()
