"""Stop at the first seven-refresh run drought and print horizon movement.

This diagnostic exercises the real daily adherence loop. It is intentionally
bounded: once seven consecutive replans decline to commit a same-day run, the
trace has enough evidence to distinguish an empty plan from a perpetually
receding first session.
"""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import run_analysis.adherence_projection as adherence_projection
from run_analysis.adherence_projection import (
    HumanAdherenceProfile,
    runs_from_summaries,
)
from run_analysis.config import load_config, resolve_project_path
from run_analysis.db import connect, initialize
from run_analysis.recommendation import effective_load_ratio
from run_analysis.recommendation_service import (
    current_fitness_state,
    load_latest_weekly_planning_days,
    load_latest_weekly_schedule,
)
from run_analysis.run_feedback import list_runs
from run_analysis.weekly_schedule import WEEKLY_PLANNER_VERSION
from run_analysis.web.schemas import WorkoutType


class ProcrastinationDetected(RuntimeError):
    pass


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--database", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--days", type=int, default=42)
    parser.add_argument("--drought-days", type=int, default=7)
    args = parser.parse_args()

    root = args.project_root.resolve()
    config = load_config(root / "config.yaml")
    database = (
        args.database.resolve()
        if args.database is not None
        else resolve_project_path(root, config["paths"]["database"])
    )
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
                    "planning_days": load_latest_weekly_planning_days(
                        connection
                    )
                }
            )
        else:
            initial_schedule = None

    original_builder = adherence_projection.build_weekly_schedule
    drought = 0

    def traced_builder(
        daily_states,
        request,
        planner_config,
        *builder_args,
        **builder_kwargs,
    ):
        nonlocal drought
        schedule = original_builder(
            daily_states,
            request,
            planner_config,
            *builder_args,
            **builder_kwargs,
        )
        origin = daily_states[0].as_of.date()
        planned = [
            day
            for day in (schedule.planning_days or schedule.days)
            if day.recommendation is not None
            and day.recommendation.workout_type != WorkoutType.REST
            and day.recommendation.planned_for is not None
        ]
        first_offset = (
            (planned[0].date - origin).days if planned else None
        )
        same_day = first_offset == 0
        drought = 0 if same_day else drought + 1
        opening = effective_load_ratio(daily_states[0].recent_load)
        target = builder_kwargs.get("target_distance_range")
        print(
            f"{origin} opening={opening!r} target={target!r} "
            f"first_offset={first_offset!r} drought={drought} "
            f"horizon="
            + ",".join(
                f"{(day.date - origin).days}:"
                f"{day.recommendation.workout_type.value}:"
                f"{sum(day.recommendation.distance_range_miles) / 2:.2f}"
                for day in planned
                if day.recommendation is not None
                and day.recommendation.distance_range_miles is not None
            ),
            flush=True,
        )
        if drought >= max(1, args.drought_days):
            raise ProcrastinationDetected(
                f"No same-day commitment for {drought} daily replans"
            )
        return schedule

    adherence_projection.build_weekly_schedule = traced_builder
    try:
        adherence_projection.simulate_adherence(
            state,
            runs,
            config,
            start_at,
            weeks=max(1, (args.days + 6) // 7),
            simulation_days=max(1, args.days),
            replan_interval_days=1,
            human_profile=HumanAdherenceProfile(seed=args.seed),
            initial_schedule=initial_schedule,
        )
    except ProcrastinationDetected as error:
        print(str(error), flush=True)
    finally:
        adherence_projection.build_weekly_schedule = original_builder


if __name__ == "__main__":
    main()
