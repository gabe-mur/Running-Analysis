"""Compare a retained absolute calendar with a later daily replan.

This diagnostic replays through a selected daily decision boundary, then
scores the preceding plan's still-future absolute dates against the selected
dates using the later boundary's evidence. It does not persist a schedule or
modify run history.
"""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import run_analysis.adherence_projection as adherence_projection
from run_analysis.adherence_projection import HumanAdherenceProfile, runs_from_summaries
from run_analysis.config import load_config, resolve_project_path
from run_analysis.db import connect, initialize
from run_analysis.environmental_stress import assess_training_weather
from run_analysis.recommendation import effective_load_ratio
from run_analysis.recommendation_service import current_fitness_state
from run_analysis.run_feedback import list_runs
import run_analysis.weekly_schedule as weekly_schedule
from run_analysis.web.schemas import CurrentHealthStatus, WorkoutType

from audit_planner_frequency import _score_parts


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--database", type=Path, default=None)
    parser.add_argument(
        "--planning-horizon-days",
        type=int,
        default=adherence_projection.PLANNING_HORIZON_DAYS,
    )
    parser.add_argument(
        "--transition-day",
        type=int,
        default=1,
        help=(
            "Compare the preceding plan with this zero-based daily refresh. "
            "For example, 4 compares Monday's plan with Tuesday's refresh "
            "when the simulation begins on Friday."
        ),
    )
    parser.add_argument(
        "--perfect-adherence",
        action="store_true",
        help="Complete prescribed work at its midpoint before later refreshes.",
    )
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

    captured: list[dict] = []
    original_builder = adherence_projection.build_weekly_schedule
    original_horizon = adherence_projection.PLANNING_HORIZON_DAYS

    def capture_builder(daily_states, request, planner_config, *builder_args, **builder_kwargs):
        schedule = original_builder(
            daily_states,
            request,
            planner_config,
            *builder_args,
            **builder_kwargs,
        )
        captured.append(
            {
                "states": daily_states,
                "request": request,
                "config": planner_config,
                "target": tuple(builder_kwargs["target_distance_range"]),
                "projector": builder_kwargs.get("expected_target_projector"),
                "prior": builder_kwargs.get("prior_schedule"),
                "schedule": schedule,
            }
        )
        return schedule

    adherence_projection.build_weekly_schedule = capture_builder
    adherence_projection.PLANNING_HORIZON_DAYS = max(
        7, args.planning_horizon_days
    )
    try:
        transition_day = max(1, args.transition_day)
        adherence_projection.simulate_adherence(
            state,
            runs,
            config,
            start_at,
            weeks=1,
            simulation_days=transition_day + 1,
            replan_interval_days=1,
            human_profile=(
                None
                if args.perfect_adherence
                else HumanAdherenceProfile(
                    seed=1,
                    skip_probability=1.0,
                    distance_variation_probability=0.0,
                    intensity_drift_probability=0.0,
                    substitution_probability=0.0,
                    unscheduled_easy_probability_per_day=0.0,
                    short_break_count_min=0,
                    short_break_count_max=0,
                    vacation_probability=0.0,
                )
            ),
            initial_schedule=None,
        )
    finally:
        adherence_projection.build_weekly_schedule = original_builder
        adherence_projection.PLANNING_HORIZON_DAYS = original_horizon

    expected_replans = max(1, args.transition_day) + 1
    if len(captured) != expected_replans:
        raise RuntimeError(
            f"Expected {expected_replans} replans, captured {len(captured)}"
        )
    second = captured[-1]
    origin = second["states"][0].as_of.date()

    def offsets(schedule) -> tuple[int, ...]:
        if schedule is None:
            return ()
        days = schedule.planning_days or schedule.days
        return tuple(
            sorted(
                (day.date - origin).days
                for day in days
                if day.recommendation is not None
                and day.recommendation.workout_type != WorkoutType.REST
                and 0 <= (day.date - origin).days < len(second["states"])
            )
        )

    retained_offsets = offsets(second["prior"])
    selected_offsets = offsets(second["schedule"])
    common = dict(
        states=second["states"],
        options=[[state] for state in second["states"]],
        request=second["request"],
        config=second["config"],
        target=second["target"],
        completed_miles={},
        projector=second["projector"],
    )

    def calendar_base_breakdown(candidate_offsets: tuple[int, ...]) -> dict:
        sessions = weekly_schedule._materialize_candidate_sessions(
            candidate_offsets,
            second["states"],
            [[state] for state in second["states"]],
            second["request"],
            second["config"],
        )
        details = []
        accounted = 0.0
        for session in sessions:
            result = session.recommendation
            ratio = effective_load_ratio(session.state.recent_load)
            load_penalty = (
                max(0.0, ratio - 1.0) * 12.0
                if ratio is not None
                else 0.0
            )
            readiness_penalty = (
                100.0
                if result.readiness.value == "not_ready"
                else 1.5
                if result.readiness.value == "caution"
                else 0.0
            )
            recovery_substitution_penalty = (
                10.0
                if result.workout_type == WorkoutType.RECOVERY
                and second["request"].health_status
                == CurrentHealthStatus.NORMAL
                else 0.0
            )
            weather_penalty = assess_training_weather(
                session.state.planned_weather,
                session.state.weather_exposure_baseline,
            ).score * 3.0
            subtotal = (
                load_penalty
                + readiness_penalty
                + recovery_substitution_penalty
                + weather_penalty
            )
            accounted += subtotal
            details.append(
                {
                    "offset": session.offset,
                    "date": session.state.as_of.date().isoformat(),
                    "workout_type": result.workout_type.value,
                    "load_ratio": ratio,
                    "load_penalty": load_penalty,
                    "readiness_penalty": readiness_penalty,
                    "weather_penalty": weather_penalty,
                }
            )
        cadence_details = []
        cadence_total = 0.0
        cadence_specs = (
            (
                "long",
                {WorkoutType.LONG},
                second["states"][0].days_since_long_run,
                float(
                    second["config"].get("coaching", {}).get(
                        "long_run_recency_reference_days", 7
                    )
                ),
            ),
            (
                "quality",
                set(weekly_schedule.QUALITY_WORKOUT_TYPES),
                second["states"][0].days_since_quality_run,
                float(
                    second["config"].get("coaching", {}).get(
                        "quality_recency_reference_days", 7
                    )
                ),
            ),
        )
        for label, accepted, days_since, reference in cadence_specs:
            if days_since is None:
                continue
            occurrences = {
                item.recommendation.planned_for.date()
                for item in sessions
                if item.recommendation.workout_type in accepted
                and item.recommendation.planned_for is not None
            }
            age = days_since
            previous_date = second["states"][0].as_of.date()
            for state in second["states"]:
                age += max(0, (state.as_of.date() - previous_date).days)
                charge = max(0.0, age - (reference + 1.0)) * 8.0
                if charge > 0:
                    cadence_details.append(
                        {
                            "lane": label,
                            "date": state.as_of.date().isoformat(),
                            "age": age,
                            "occurrence": state.as_of.date() in occurrences,
                            "charge": charge,
                        }
                    )
                    cadence_total += charge
                if state.as_of.date() in occurrences:
                    age = 0.0
                previous_date = state.as_of.date()
        base = _score_parts(candidate_offsets, **common)["calendar_base"]
        return {
            "calendar_base_accounted": accounted,
            "calendar_base_other_including_key_cadence": base - accounted,
            "key_cadence_reproduced": cadence_total,
            "key_cadence_charges": cadence_details,
            "sessions": details,
        }

    retained_score = _score_parts(retained_offsets, **common)
    selected_score = _score_parts(selected_offsets, **common)
    print(
        json.dumps(
            {
                "generated_at": second["states"][0].as_of.isoformat(),
                "target": second["target"],
                "retained_absolute_calendar": {
                    "offsets": retained_offsets,
                    **retained_score,
                    **calendar_base_breakdown(retained_offsets),
                },
                "selected_calendar": {
                    "offsets": selected_offsets,
                    **selected_score,
                    **calendar_base_breakdown(selected_offsets),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
