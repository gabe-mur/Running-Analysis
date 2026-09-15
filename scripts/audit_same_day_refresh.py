"""Read-only clock sweep of a frozen saved weekly-planner snapshot."""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, time, timedelta
import json
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfo

from run_analysis.config import load_config, resolve_project_path
from run_analysis.recommendation_service import current_fitness_state
from run_analysis.web.schemas import (
    FitnessState,
    RecommendationRequest,
    TrailingDayActivity,
    WeeklyScheduleDay,
    WeeklyScheduleResponse,
    WeeklyTargetEvidence,
)
from run_analysis.weekly_schedule import (
    PlanningActivity,
    build_weekly_schedule,
    make_expected_target_projector,
)


def _shape(schedule: WeeklyScheduleResponse) -> str:
    return "; ".join(
        f"{day.date.strftime('%a')} "
        + (
            f"{day.recommendation.workout_type.value} "
            f"{day.recommendation.distance_range_miles}"
            if day.recommendation is not None
            else "rest"
        )
        for day in schedule.days[:7]
    )


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = load_config(root / "config.yaml")
    database = resolve_project_path(root, config["paths"]["database"])
    zone = ZoneInfo(str(config["timezone_default"]))
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute(
            "SELECT value_json FROM app_state WHERE key='weekly_planner_snapshot'"
        ).fetchone()
        if row is None:
            raise RuntimeError("No saved planner snapshot")
        payload = json.loads(row[0])
        saved = WeeklyScheduleResponse.model_validate(payload["result"])
        prior_payload = payload.get("prior_schedule")
        prior = (
            WeeklyScheduleResponse.model_validate(prior_payload)
            if prior_payload is not None
            else None
        )
        if prior is not None and payload.get("prior_planning_days") is not None:
            prior = prior.model_copy(
                update={
                    "planning_days": [
                        WeeklyScheduleDay.model_validate(day)
                        for day in payload["prior_planning_days"]
                    ]
                }
            )
        daily_states = [
            FitnessState.model_validate(item)
            for item in payload["daily_states"]
        ]
        options = [
            [FitnessState.model_validate(item) for item in group]
            for group in payload["daily_state_options"]
        ]
        target_evidence = WeeklyTargetEvidence.model_validate(
            payload["target_evidence"]
        )
        observed = [
            PlanningActivity(
                datetime.fromisoformat(item["start_time"]),
                float(item["distance_miles"]),
                moving_minutes=item.get("moving_minutes"),
                easy_minutes=item.get("easy_minutes"),
                baseline_eligible=bool(item.get("baseline_eligible", True)),
            )
            for item in payload["observed_planning_activities"]
        ]
        projector = make_expected_target_projector(
            observed,
            daily_states[0].as_of.date(),
            payload["config"],
            pace_min_mile=float(payload["target_projection_pace_min_mile"]),
            maximum_horizon_days=len(daily_states),
            opening_target_range=tuple(payload["target_distance_range"]),
            opening_evidence=target_evidence,
        )
        completed = {
            int(offset): [TrailingDayActivity.model_validate(item) for item in values]
            for offset, values in payload.get(
                "completed_activities_by_offset", {}
            ).items()
        }
        day = daily_states[0].as_of.date()
        frozen_weather = options[0][0].planned_weather
        capacity = max(
            target_evidence.capacity_reference_miles,
            options[0][0].recent_load.capacity_reference_miles or 0.0,
        )
        candidate_hours = sorted(
            int(hour)
            for hour in payload["config"].get("weather", {}).get(
                "automatic_run_time_hours_local", [7, 12, 19]
            )
        )
        time_states: dict[int, FitnessState] = {}
        for hour in candidate_hours:
            planned_at = datetime.combine(day, time(hour), tzinfo=zone)
            state = current_fitness_state(
                connection,
                payload["config"],
                RecommendationRequest(
                    health_status=payload["request"]["health_status"],
                    planned_at=planned_at,
                ),
            )
            recent = state.recent_load
            state = state.model_copy(
                update={
                    "planned_weather": frozen_weather,
                    "recent_load": recent.model_copy(
                        update={
                            "capacity_reference_miles": capacity,
                            "sustained_capacity_miles": max(
                                recent.sustained_capacity_miles or 0.0,
                                capacity,
                            ),
                            "acute_distance_to_capacity_ratio": (
                                recent.trailing_7d.distance_miles / capacity
                                if capacity > 0
                                else None
                            ),
                            "continuous_fatigue_to_capacity_ratio": (
                                recent.continuous_fatigue_miles / capacity
                                if recent.continuous_fatigue_miles is not None
                                and capacity > 0
                                else None
                            ),
                        }
                    ),
                }
            )
            time_states[hour] = state

        print(f"Saved snapshot: {saved.generated_at.isoformat()}")
        print(f"Saved plan: {_shape(saved)}")
        print(f"Prior plan: {_shape(prior) if prior is not None else 'none'}")
        if prior is not None and payload.get("prior_planning_days") is None:
            print(
                "Warning: this older snapshot omitted the prior 21-day warm "
                "start, so its replay cannot exactly reconstruct the saved "
                "optimizer input."
            )
        print(
            "Frozen day-0 weather; unchanged history, targets, future-day "
            "states, and prior warm start. Only today's feasible time set changes."
        )
        for clock_hour, clock_minute in ((1, 0), (7, 30), (11, 45), (12, 5), (17, 48)):
            clock = datetime.combine(
                day,
                time(clock_hour, clock_minute),
                tzinfo=zone,
            )
            feasible = [
                hour
                for hour in candidate_hours
                if datetime.combine(day, time(hour), tzinfo=zone)
                > clock + timedelta(minutes=10)
            ]
            if not feasible:
                feasible = [candidate_hours[-1]]
            sweep_options = [
                [time_states[hour] for hour in feasible],
                *options[1:],
            ]
            schedule = build_weekly_schedule(
                daily_states,
                RecommendationRequest.model_validate(payload["request"]),
                payload["config"],
                target_run_count=payload["target_run_count"],
                target_distance_range=tuple(payload["target_distance_range"]),
                target_evidence=target_evidence,
                completed_activities_by_offset=completed,
                daily_state_options=sweep_options,
                forced_rest_offsets=set(payload.get("forced_rest_offsets", [])),
                prior_schedule=prior,
                expected_target_projector=projector,
            )
            print(
                f"{clock.strftime('%H:%M')} slots {feasible}: {_shape(schedule)}",
                flush=True,
            )
    finally:
        connection.close()


if __name__ == "__main__":
    main()
