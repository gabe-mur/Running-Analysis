"""Run deterministic daily-replan overload scenarios through the real planner."""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from run_analysis.adherence_projection import (
    OverloadAdherenceProfile,
    OverloadScenario,
    ProjectionReplan,
    runs_from_summaries,
    simulate_adherence,
)
from run_analysis.config import load_config, resolve_project_path
from run_analysis.db import connect, initialize
from run_analysis.recommendation_service import current_fitness_state
from run_analysis.run_feedback import list_runs
from run_analysis.web.schemas import WorkoutType


TAXING_TYPES = {
    WorkoutType.LONG,
    WorkoutType.INTERVALS,
    WorkoutType.TEMPO_THRESHOLD,
    WorkoutType.RACE,
}


def _snapshots(projection) -> list[ProjectionReplan]:
    return sorted(
        (
            snapshot
            for week in projection
            for snapshot in week.replan_snapshots
        ),
        key=lambda item: item.generated_at,
    )


def _forward(snapshot: ProjectionReplan, days: int) -> tuple[float, int]:
    end = snapshot.generated_at + timedelta(days=days)
    sessions = [
        item
        for item in snapshot.planned_sessions
        if snapshot.generated_at <= item.planned_for < end
    ]
    return sum(item.midpoint_miles for item in sessions), len(sessions)


def _next_session(snapshot: ProjectionReplan, *, taxing: bool = False) -> str:
    sessions = [
        item
        for item in snapshot.planned_sessions
        if item.planned_for >= snapshot.generated_at
        and (not taxing or item.workout_type in TAXING_TYPES)
    ]
    if not sessions:
        return "none"
    item = min(sessions, key=lambda value: value.planned_for)
    gap_hours = (item.planned_for - snapshot.generated_at).total_seconds() / 3600
    return f"{gap_hours:.0f}h {item.workout_type.value} {item.midpoint_miles:.1f}mi"


def _print_comparison(control, scenario, label: str) -> None:
    events = [
        event
        for week in scenario
        for event in week.adherence_event_records
        if event.kind == "overload"
    ]
    if not events:
        print(f"\n{label}: intervention did not occur inside the simulated period")
        return
    intervention = min(events, key=lambda item: item.occurred_at)
    print(f"\n{label}: {intervention.occurred_at.isoformat()} — {intervention.detail}")
    control_by_time = {
        item.generated_at: item for item in _snapshots(control)
    }
    post = [
        item
        for item in _snapshots(scenario)
        if item.generated_at > intervention.occurred_at
    ][:7]
    print(
        "replan | opening load | next run | next taxing | "
        "next 7d vs control | next 14d rate vs control"
    )
    for item in post:
        baseline = control_by_time.get(item.generated_at)
        miles_7, runs_7 = _forward(item, 7)
        miles_14, _ = _forward(item, 14)
        if baseline is not None:
            base_7, base_runs_7 = _forward(baseline, 7)
            base_14, _ = _forward(baseline, 14)
            comparison_7 = (
                f"{miles_7:.1f}/{runs_7} "
                f"({miles_7 - base_7:+.1f}mi, {runs_7 - base_runs_7:+d} runs)"
            )
            comparison_14 = (
                f"{miles_14 / 2:.1f} "
                f"({(miles_14 - base_14) / 2:+.1f}mi/wk)"
            )
        else:
            comparison_7 = f"{miles_7:.1f}/{runs_7}"
            comparison_14 = f"{miles_14 / 2:.1f}"
        opening = (
            f"{item.opening_load_ratio * 100:.0f}%"
            if item.opening_load_ratio is not None
            else "—"
        )
        next_run = _next_session(item)
        next_taxing = _next_session(item, taxing=True)
        if baseline is not None:
            if (
                item.opening_load_ratio is not None
                and baseline.opening_load_ratio is not None
            ):
                load_delta_points = (
                    item.opening_load_ratio - baseline.opening_load_ratio
                ) * 100
                opening += (
                    f" ({load_delta_points:+.0f}pp)"
                )
            baseline_next_run = _next_session(baseline)
            baseline_next_taxing = _next_session(baseline, taxing=True)
            if next_run != baseline_next_run:
                next_run += f" [control {baseline_next_run}]"
            if next_taxing != baseline_next_taxing:
                next_taxing += f" [control {baseline_next_taxing}]"
        print(
            f"{item.generated_at.strftime('%a %b %-d')} | {opening} | "
            f"{next_run} | {next_taxing} | "
            f"{comparison_7} | {comparison_14}"
        )


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--weeks", type=int, default=2)
    parser.add_argument(
        "--scenario",
        choices=("all",) + tuple(
            item.value
            for item in OverloadScenario
            if item != OverloadScenario.CONTROL
        ),
        default="all",
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = load_config(root / "config.yaml")
    database = resolve_project_path(root, config["paths"]["database"])
    zone = ZoneInfo(str(config["timezone_default"]))

    with connect(database) as connection:
        initialize(connection)
        state = current_fitness_state(connection, config)
        runs = runs_from_summaries(list_runs(connection, limit=5000))

    start_at = state.as_of.astimezone(zone)
    control = simulate_adherence(
        state,
        runs,
        config,
        start_at,
        weeks=max(1, args.weeks),
        replan_interval_days=1,
        overload_profile=OverloadAdherenceProfile(
            scenario=OverloadScenario.CONTROL
        ),
    )
    scenarios = (
        [
            item
            for item in OverloadScenario
            if item != OverloadScenario.CONTROL
        ]
        if args.scenario == "all"
        else [OverloadScenario(args.scenario)]
    )
    for scenario in scenarios:
        result = simulate_adherence(
            state,
            runs,
            config,
            start_at,
            weeks=max(1, args.weeks),
            replan_interval_days=1,
            overload_profile=OverloadAdherenceProfile(scenario=scenario),
        )
        _print_comparison(control, result, scenario.value)


if __name__ == "__main__":
    main()
