"""Audit committed strength suggestions across a daily-replanned run projection."""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import run_analysis.adherence_projection as adherence_projection
from run_analysis.adherence_projection import runs_from_summaries
from run_analysis.config import load_config, resolve_project_path
from run_analysis.db import connect, initialize
from run_analysis.recommendation_service import (
    current_fitness_state,
    load_latest_weekly_planning_days,
    load_latest_weekly_schedule,
)
from run_analysis.run_feedback import list_runs
from run_analysis.web.schemas import StrengthSessionType, WorkoutType
from run_analysis.weekly_schedule import WEEKLY_PLANNER_VERSION


def _lane_dates(
    events: list[tuple[date, StrengthSessionType, WorkoutType | None]],
    lane: StrengthSessionType,
) -> list[date]:
    matching_types = {lane}
    if lane in {StrengthSessionType.PUSH, StrengthSessionType.PULL}:
        matching_types.add(StrengthSessionType.FULL_UPPER)
    return [
        event_date
        for event_date, session_type, _ in events
        if session_type in matching_types
    ]


def _cadence_summary(
    values: list[date],
    start_date: date,
    end_date: date,
) -> str:
    if not values:
        return "no committed sessions"
    internal_gaps = [
        (right - left).days for left, right in zip(values, values[1:])
    ]
    opening_gap = (values[0] - start_date).days
    closing_gap = (end_date - values[-1]).days
    maximum_gap = max([opening_gap, closing_gap, *internal_gaps])
    violations = sum(gap > 8 for gap in internal_gaps)
    violations += opening_gap >= 8
    violations += closing_gap >= 8
    return (
        f"{len(values)} sessions; maximum gap {maximum_gap} days; "
        f"eight-day violations {violations}"
    )


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--days", type=int, default=90)
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

    committed_strength: dict[
        date, tuple[StrengthSessionType, WorkoutType | None]
    ] = {}
    revised_strength_dates: set[date] = set()
    daily_runs: dict[date, WorkoutType] = {}
    original_builder = adherence_projection.build_weekly_schedule
    builder_depth = 0

    def capture_builder(*builder_args, **builder_kwargs):
        nonlocal builder_depth
        builder_depth += 1
        try:
            schedule = original_builder(*builder_args, **builder_kwargs)
        finally:
            builder_depth -= 1
        if builder_depth != 0:
            return schedule
        decision_date = schedule.start_date
        today = next(day for day in schedule.days if day.date == decision_date)
        run_type = (
            today.recommendation.workout_type
            if today.recommendation is not None
            and today.recommendation.workout_type != WorkoutType.REST
            else None
        )
        if run_type is not None:
            daily_runs[decision_date] = run_type
        if today.strength_suggestion is not None:
            candidate = (today.strength_suggestion.session_type, run_type)
            previous = committed_strength.setdefault(decision_date, candidate)
            if previous != candidate:
                revised_strength_dates.add(decision_date)
        elif decision_date in committed_strength:
            revised_strength_dates.add(decision_date)
        return schedule

    adherence_projection.build_weekly_schedule = capture_builder
    try:
        weeks = adherence_projection.simulate_adherence(
            state,
            runs,
            config,
            start_at,
            weeks=max(1, (args.days + 6) // 7),
            replan_interval_days=1,
            initial_schedule=initial_schedule,
            simulation_days=args.days,
        )
    finally:
        adherence_projection.build_weekly_schedule = original_builder

    start_date = start_at.date()
    end_date = start_date + timedelta(days=args.days - 1)
    events = [
        (event_date, *details)
        for event_date, details in sorted(committed_strength.items())
    ]
    counts = Counter(session_type.value for _, session_type, _ in events)
    paired_easy = sum(run_type == WorkoutType.EASY for _, _, run_type in events)
    invalid_pairings = [
        item
        for item in events
        if item[2] is not None and item[2] != WorkoutType.EASY
    ]
    taxing_dates = [
        run_date
        for run_date, workout_type in daily_runs.items()
        if workout_type
        in {
            WorkoutType.LONG,
            WorkoutType.INTERVALS,
            WorkoutType.TEMPO_THRESHOLD,
            WorkoutType.RACE,
        }
    ]
    leg_dates = _lane_dates(events, StrengthSessionType.LEGS)
    leg_clearances = [
        min((abs((leg_date - taxing_date).days) for taxing_date in taxing_dates), default=99)
        for leg_date in leg_dates
    ]

    print(f"Projection: {start_date} through {end_date} ({args.days} days)")
    print(
        f"Runs: {sum(week.run_count for week in weeks)}; "
        f"strength sessions: {len(events)}; breakdown: {dict(sorted(counts.items()))}"
    )
    for label, lane in (
        ("Push", StrengthSessionType.PUSH),
        ("Pull", StrengthSessionType.PULL),
        ("Legs", StrengthSessionType.LEGS),
    ):
        print(
            f"{label}: "
            + _cadence_summary(_lane_dates(events, lane), start_date, end_date)
        )
    print(f"Strength paired with easy runs: {paired_easy}")
    print(f"Strength paired with prohibited run types: {len(invalid_pairings)}")
    print(f"Committed dates revised by a later refresh: {len(revised_strength_dates)}")
    print(
        "Leg clearance from nearest long/quality run: "
        f"minimum {min(leg_clearances, default=0)} days; "
        f"within 1 day {sum(value <= 1 for value in leg_clearances)}; "
        f"within 2 days {sum(value <= 2 for value in leg_clearances)}"
    )
    print("\nCommitted strength calendar:")
    for event_date, session_type, run_type in events:
        pairing = f" + {run_type.value} run" if run_type is not None else ""
        print(f"{event_date} {session_type.value}{pairing}")


if __name__ == "__main__":
    main()
