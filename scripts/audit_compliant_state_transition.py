"""Compare a planned-prefix state with its compliant completed-state replay."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import run_analysis.adherence_projection as adherence_projection
import run_analysis.weekly_schedule as weekly_schedule
from run_analysis.adherence_projection import runs_from_summaries
from run_analysis.config import load_config, resolve_project_path
from run_analysis.db import connect, initialize
from run_analysis.recommendation_service import current_fitness_state
from run_analysis.run_feedback import list_runs
from run_analysis.web.schemas import CurrentHealthStatus, RecommendationRequest, WorkoutType


def main() -> None:
    root = Path.cwd()
    config = load_config(root / "config.yaml")
    database = resolve_project_path(root, config["paths"]["database"])
    zone = ZoneInfo(str(config["timezone_default"]))
    start_at = datetime.now(zone)
    with connect(database) as connection:
        initialize(connection)
        template = current_fitness_state(connection, config)
        runs = runs_from_summaries(list_runs(connection, limit=5000))

    captures: list[dict] = []
    original_builder = adherence_projection.build_weekly_schedule

    def capture_builder(daily_states, request, planner_config, *args, **kwargs):
        schedule = original_builder(
            daily_states,
            request,
            planner_config,
            *args,
            **kwargs,
        )
        captures.append(
            {
                "states": daily_states,
                "options": kwargs["daily_state_options"],
                "request": request,
                "schedule": schedule,
            }
        )
        return schedule

    adherence_projection.build_weekly_schedule = capture_builder
    try:
        adherence_projection.simulate_adherence(
            template,
            runs,
            config,
            start_at,
            weeks=1,
            simulation_days=7,
            replan_interval_days=1,
            initial_schedule=None,
        )
    finally:
        adherence_projection.build_weekly_schedule = original_builder

    for earlier, later in zip(captures, captures[1:]):
        earlier_runs = [
            day.recommendation
            for day in earlier["schedule"].planning_days
            if day.recommendation is not None
            and day.recommendation.workout_type != WorkoutType.REST
        ]
        if len(earlier_runs) < 2:
            continue
        first, second = earlier_runs[:2]
        if first.workout_type not in weekly_schedule.QUALITY_WORKOUT_TYPES:
            continue
        if second.planned_for.date() != first.planned_for.date().fromordinal(
            first.planned_for.date().toordinal() + 1
        ):
            continue
        if later["states"][0].as_of.date() != second.planned_for.date():
            continue
        offset = (second.planned_for.date() - earlier["states"][0].as_of.date()).days
        projected_state, projected_result = weekly_schedule._select_budgeted_timed_recommendation(
            earlier["options"][offset],
            [first],
            RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
            config,
            weekly_role="easy",
        )
        actual_state, actual_result = weekly_schedule._select_budgeted_timed_recommendation(
            later["options"][0],
            [],
            RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
            config,
            weekly_role="easy",
        )
        projected_payload = projected_state.model_dump()
        actual_payload = actual_state.model_dump()
        state_diff = {
            key: (projected_payload[key], actual_payload[key])
            for key in projected_payload
            if projected_payload[key] != actual_payload[key]
        }
        print(
            {
                "quality": first.model_dump(mode="json"),
                "saved_next": second.model_dump(mode="json"),
                "projected_raw_range": projected_result.distance_range_miles,
                "actual_raw_range": actual_result.distance_range_miles,
                "projected_time": projected_state.as_of.isoformat(),
                "actual_time": actual_state.as_of.isoformat(),
                "state_diff": state_diff,
            }
        )
        return
    raise RuntimeError("No quality-to-next-day compliant transition captured")


if __name__ == "__main__":
    main()
