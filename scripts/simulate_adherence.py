"""Print a read-only multi-week projection of perfect plan adherence."""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import run_analysis.weekly_schedule as weekly_schedule
from run_analysis.adherence_projection import (
    DeterministicAdherenceProfile,
    DeterministicAdherenceScenario,
    HumanAdherenceProfile,
    OverloadAdherenceProfile,
    OverloadScenario,
    runs_from_summaries,
    simulate_adherence,
    summarize_replan_churn,
)
from run_analysis.config import load_config, resolve_project_path
from run_analysis.db import connect, initialize
from run_analysis.recommendation_service import (
    current_fitness_state,
    load_latest_weekly_planning_days,
    load_latest_weekly_schedule,
)
from run_analysis.run_feedback import list_runs
from run_analysis.projection_gate import (
    ProjectionGate,
    ProjectionGateConfig,
    ProjectionGateTriggered,
    build_projection_report,
)
from run_analysis.weekly_schedule import WEEKLY_PLANNER_VERSION


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--weeks", type=int, default=13)
    parser.add_argument(
        "--simulation-days",
        type=int,
        default=None,
        help="Optionally cap the projection to this many daily decision windows.",
    )
    parser.add_argument(
        "--mode",
        choices=("perfect", "human"),
        default="perfect",
        help="Adherence behavior to simulate (default: perfect)",
    )
    parser.add_argument(
        "--scenario",
        choices=(
            "perfect",
            "mixed_human",
            *(item.value for item in DeterministicAdherenceScenario),
            *(
                item.value
                for item in OverloadScenario
                if item != OverloadScenario.CONTROL
            ),
        ),
        default=None,
        help=(
            "Named deterministic scenario. This supersedes --mode; "
            "--mode remains for compatibility."
        ),
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
    parser.add_argument(
        "--progress-replans",
        action="store_true",
        help="Print each completed planner decision immediately.",
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--start-at",
        type=str,
        default=None,
        help=(
            "Fix the simulation origin as an ISO-8601 timestamp instead of "
            "using the current wall clock."
        ),
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=None,
        help="Use a specific database snapshot instead of the configured database.",
    )
    parser.add_argument(
        "--ignore-saved-plan",
        action="store_true",
        help="Start from a fresh planner decision instead of a persisted schedule.",
    )
    parser.add_argument(
        "--planner-ablation",
        choices=(
            "none",
            "target20",
            "no-underload-debt",
            "target20-no-underload-debt",
        ),
        default="none",
        help="Temporarily disable selected planner changes for controlled audits.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Write daily replans, failures, churn, and timing as JSON.",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit nonzero when the projection violates a regression gate.",
    )
    parser.add_argument(
        "--stop-on-first-failure",
        action="store_true",
        help="Stop immediately after the first bad daily replan.",
    )
    args = parser.parse_args()
    if args.stop_on_first_failure and not args.fail_on_regression:
        parser.error("--stop-on-first-failure requires --fail-on-regression")
    if args.scenario is not None and args.mode != "perfect":
        parser.error("Use either --scenario or --mode human, not both")
    scenario = args.scenario or (
        "mixed_human" if args.mode == "human" else "perfect"
    )
    if args.planner_ablation in {"target20", "target20-no-underload-debt"}:
        weekly_schedule.TARGET_FIT_UNIT = weekly_schedule.PROGRAM_FIT_UNIT
    if args.planner_ablation in {
        "no-underload-debt",
        "target20-no-underload-debt",
    }:
        original_path_cost = weekly_schedule._continuous_mileage_path_cost

        def ablated_path_cost(*path_args, **path_kwargs):
            if path_kwargs.get("underload_only", False):
                return 0.0
            return original_path_cost(*path_args, **path_kwargs)

        weekly_schedule._continuous_mileage_path_cost = ablated_path_cost
    root = args.project_root.resolve()
    config = load_config(root / "config.yaml")
    database = (
        args.database.resolve()
        if args.database is not None
        else resolve_project_path(root, config["paths"]["database"])
    )
    zone = ZoneInfo(str(config["timezone_default"]))
    start_at = (
        datetime.fromisoformat(args.start_at).astimezone(zone)
        if args.start_at is not None
        else datetime.now(zone)
    )
    with connect(database) as connection:
        initialize(connection)
        state = current_fitness_state(connection, config)
        runs = runs_from_summaries(list_runs(connection, limit=5000))
        initial_schedule = (
            None if args.ignore_saved_plan else load_latest_weekly_schedule(connection)
        )
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
    gate = ProjectionGate(
        ProjectionGateConfig(
            enforce_stability=scenario
            in {"perfect", "in_range_low", "in_range_high"},
            enforce_key_session_cadence=scenario
            in {"perfect", "in_range_low", "in_range_high"},
            require_capacity_progression=(
                scenario == "perfect"
                and (
                    args.simulation_days is None
                    or args.simulation_days >= 91
                )
            ),
        ),
        fail_fast=args.stop_on_first_failure,
    )
    weeks = []
    stopped_early = False

    def observe_replan(snapshot) -> None:
        if args.progress_replans:
            print(
                f"Replan {snapshot.generated_at.isoformat()} "
                f"[{snapshot.trigger}] {snapshot.planning_seconds:.2f}s",
                flush=True,
            )
        gate.observe(snapshot)

    try:
        weeks = simulate_adherence(
            state,
            runs,
            config,
            start_at,
            weeks=max(1, args.weeks),
            simulation_days=(
                max(1, args.simulation_days)
                if args.simulation_days is not None
                else None
            ),
            replan_interval_days=args.replan_days,
            human_profile=(
                HumanAdherenceProfile(seed=args.seed)
                if scenario == "mixed_human"
                else None
            ),
            overload_profile=(
                OverloadAdherenceProfile(
                    scenario=OverloadScenario(scenario)
                )
                if scenario in {item.value for item in OverloadScenario}
                and scenario != OverloadScenario.CONTROL.value
                else None
            ),
            deterministic_profile=(
                DeterministicAdherenceProfile(
                    scenario=DeterministicAdherenceScenario(scenario)
                )
                if scenario
                in {item.value for item in DeterministicAdherenceScenario}
                else None
            ),
            initial_schedule=initial_schedule,
            replan_observer=observe_replan,
        )
        gate.finalize(weeks)
    except ProjectionGateTriggered as error:
        stopped_early = True
        print(
            f"Projection stopped at {error.failure.occurred_at.isoformat()}: "
            f"{error.failure.code}: {error.failure.detail}"
        )

    report = build_projection_report(
        mode=scenario,
        seed=args.seed,
        replan_interval_days=args.replan_days,
        weeks=weeks,
        gate=gate,
        stopped_early=stopped_early,
    )
    if args.output_json is not None:
        output_path = args.output_json.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Projection JSON: {output_path}")
    if stopped_early:
        if args.fail_on_regression:
            raise SystemExit(1)
        return

    print(f"Planner ablation: {args.planner_ablation}")
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
    replans = tuple(
        snapshot
        for item in weeks
        for snapshot in item.replan_snapshots
    )
    for trigger, label in (
        ("post_upload", "upload replans"),
        ("scheduled_refresh", "clock refreshes"),
    ):
        for horizon_days in (3, 4, 7):
            churn = summarize_replan_churn(
                replans,
                horizon_days=horizon_days,
                transition_trigger=trigger,
            )
            print(
                f"Plan stability after {label}, next {horizon_days} days: "
                f"{churn.schedule_changed_comparisons}/"
                f"{churn.comparison_count} transitions changed dates/types; "
                f"{churn.date_slot_changes} date-slot changes; "
                f"{churn.workout_type_changes} workout-type changes; "
                f"{churn.distance_only_changed_comparisons} transitions "
                f"changed distance only; {churn.distance_changes} same-date "
                "distance edits total."
            )
    print(
        "\nAssumptions: "
        + (
            "every run is completed at the distance midpoint with a normal response. "
            if scenario == "perfect"
            else (
                f"named deterministic scenario {scenario}. "
                if scenario != "mixed_human"
                else f"bounded human adherence with seed {args.seed}: occasional skips, "
                "distance and intensity drift, workout substitutions, unscheduled easy "
                "runs, 2–3 short trips, and a 50% chance of one seven-day vacation. "
            )
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
    if gate.failures:
        print("\nRegression gate failures:")
        for failure in gate.failures:
            print(
                f"- {failure.occurred_at.isoformat()} {failure.code}: "
                f"{failure.detail}"
            )
        if args.fail_on_regression:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
