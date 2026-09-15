"""Compare an open-loop plan with a bounded expected-policy rollout."""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from run_analysis.adherence_projection import (
    ProjectionPlanSession,
    runs_from_summaries,
    simulate_expected_policy_rollout,
)
from run_analysis.config import load_config, resolve_project_path
from run_analysis.db import connect
from run_analysis.recommendation_service import (
    current_fitness_state,
    load_latest_weekly_planning_days,
    load_latest_weekly_schedule,
)
from run_analysis.run_feedback import list_runs
from run_analysis.weekly_schedule import WEEKLY_PLANNER_VERSION


def _format_sessions(sessions: tuple[ProjectionPlanSession, ...]) -> str:
    if not sessions:
        return "rest"
    return "; ".join(
        f"{session.planned_for.strftime('%a %-m/%-d')} "
        f"{session.workout_type.value} {session.midpoint_miles:.2f} mi"
        for session in sessions
    )


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--days", type=int, default=4)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--trace-replans", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    config = load_config(root / "config.yaml")
    database = resolve_project_path(root, config["paths"]["database"])
    zone = ZoneInfo(str(config["timezone_default"]))
    start_at = datetime.now(zone)
    with connect(database) as connection:
        state = current_fitness_state(connection, config)
        runs = runs_from_summaries(list_runs(connection, limit=5000))
        initial_schedule = load_latest_weekly_schedule(connection)
        if (
            initial_schedule is not None
            and initial_schedule.planner_version == WEEKLY_PLANNER_VERSION
        ):
            initial_schedule = initial_schedule.model_copy(
                update={
                    "planning_days": load_latest_weekly_planning_days(connection)
                }
            )
        else:
            initial_schedule = None

    rollout = simulate_expected_policy_rollout(
        state,
        runs,
        config,
        start_at,
        horizon_days=args.days,
        initial_schedule=initial_schedule,
    )
    print(f"Opening bounded calendar: {_format_sessions(rollout.opening_horizon_plan)}")
    print(f"Expected-policy calendar: {_format_sessions(rollout.policy_plan)}\n")
    print("window | opening plan | expected-policy decision | change")
    print("--- | --- | --- | ---")
    for step in rollout.steps:
        changes = []
        if step.schedule_changed_from_opening:
            changes.append("schedule")
        if step.distance_changed_from_opening:
            changes.append("distance")
        print(
            f"{step.window_start.strftime('%a %-m/%-d')} | "
            f"{_format_sessions(step.opening_sessions)} | "
            f"{_format_sessions(step.policy_sessions)} | "
            f"{', '.join(changes) or 'none'}"
        )
    print(
        f"\n{rollout.schedule_change_count} schedule changes and "
        f"{rollout.distance_change_count} distance-only changes across "
        f"{rollout.horizon_days} daily decision windows."
    )
    if args.trace_replans:
        for replan in rollout.replans:
            print(
                f"\n{replan.generated_at.isoformat()} target "
                f"{replan.target_low_miles:.1f}–{replan.target_high_miles:.1f}"
            )
            print(f"  plan: {_format_sessions(replan.planned_sessions)}")
            print(f"  commit: {_format_sessions(replan.committed_sessions)}")


if __name__ == "__main__":
    main()
