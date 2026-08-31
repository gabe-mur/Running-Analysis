"""Print a read-only multi-week projection of perfect plan adherence."""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from run_analysis.adherence_projection import runs_from_summaries, simulate_adherence
from run_analysis.config import load_config, resolve_project_path
from run_analysis.db import connect, initialize
from run_analysis.recommendation_service import current_fitness_state
from run_analysis.run_feedback import list_runs


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--weeks", type=int, default=13)
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
        state, runs, config, start_at, weeks=max(1, args.weeks)
    )
    print("week | start | runs | target | prescribed | midpoint | capacity | opening load | workouts")
    print("--- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---")
    for item in weeks:
        opening = (
            f"{item.opening_acute_ratio * 100:.0f}%"
            if item.opening_acute_ratio is not None
            else "—"
        )
        print(
            f"{item.week} | {item.start_date} | {item.run_count} | "
            f"{item.target_low_miles:.1f}–{item.target_high_miles:.1f} | "
            f"{item.prescribed_low_miles:.1f}–{item.prescribed_high_miles:.1f} | "
            f"{item.assumed_completed_miles:.1f} | "
            f"{item.capacity_reference_miles:.1f} | {opening} | "
            + "; ".join(item.workouts)
        )
    print(
        "\nAssumptions: every run is completed at the distance midpoint with a normal response. "
        "Future HR, pace adaptation, weather, sleep, soreness, illness, injury, and missed runs are not invented."
    )


if __name__ == "__main__":
    main()
