"""Read-only decomposition of weekly-planner frequency choices.

Loads the exact persisted planner snapshot, re-scores every frequency option
through the production candidate search, and reports the terms that decide the
winner. It never writes to the application database.
"""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime
import json
from math import ceil
from pathlib import Path
import sqlite3

from run_analysis.config import load_config, resolve_project_path
import run_analysis.weekly_schedule as weekly_schedule
from run_analysis.web.schemas import (
    FitnessState,
    RecommendationRequest,
    TrailingDayActivity,
    WeeklyTargetEvidence,
)
from run_analysis.weekly_schedule import (
    MAX_HORIZON_COUNT_OPTIONS,
    PlanningActivity,
    _adaptive_candidate_cost,
    _adaptive_run_day_offsets_for_frequency,
    _joint_candidate_program_cost,
    _program_selection_cost,
    effective_load_ratio,
    make_expected_target_projector,
    typical_easy_distance,
)


def _load_snapshot(connection: sqlite3.Connection) -> dict:
    row = connection.execute(
        "SELECT value_json FROM app_state WHERE key='weekly_planner_snapshot'"
    ).fetchone()
    if row is None:
        raise RuntimeError("No saved planner snapshot")
    return json.loads(row[0])


def _score_parts(
    offsets: tuple[int, ...],
    *,
    states: list[FitnessState],
    options: list[list[FitnessState]],
    request: RecommendationRequest,
    config: dict,
    target: tuple[float, float],
    completed_miles: dict[int, float],
    projector,
) -> dict:
    kwargs = dict(
        completed_miles_by_offset=completed_miles,
        role_loop_penalty=False,
    )
    base = _adaptive_candidate_cost(
        offsets,
        states,
        options,
        request,
        config,
        target,
        include_mileage_path=False,
        include_recovery_interactions=False,
        include_cadence_pressure=False,
        **kwargs,
    )
    with_recovery = _adaptive_candidate_cost(
        offsets,
        states,
        options,
        request,
        config,
        target,
        include_mileage_path=False,
        include_recovery_interactions=True,
        include_cadence_pressure=False,
        **kwargs,
    )
    with_path = _adaptive_candidate_cost(
        offsets,
        states,
        options,
        request,
        config,
        target,
        include_mileage_path=True,
        include_recovery_interactions=False,
        include_cadence_pressure=False,
        **kwargs,
    )
    with_cadence = _adaptive_candidate_cost(
        offsets,
        states,
        options,
        request,
        config,
        target,
        include_mileage_path=False,
        include_recovery_interactions=False,
        include_cadence_pressure=True,
        **kwargs,
    )
    joint = _joint_candidate_program_cost(
        offsets,
        states,
        options,
        request,
        config,
        target,
        completed_miles_by_offset=completed_miles,
        expected_target_projector=projector,
    )
    ordinary = sum(typical_easy_distance(states[0])) / 2
    return {
        "selection": _program_selection_cost(joint, base, ordinary),
        "calendar_base": base,
        "preallocation_recovery": with_recovery - base,
        "preallocation_path": with_path - base,
        "cadence_pressure": with_cadence - base,
        "target_violation": joint[0],
        "aerobic_support_violation": joint[1],
        "shape_violation": joint[2],
        "finalized_recovery": joint[3],
    }


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--database",
        type=Path,
        default=None,
        help=(
            "Read a specific database instead of the project-configured one. "
            "Useful for auditing a regenerated disposable copy."
        ),
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=None,
        help="Override the production candidate beam width for sensitivity auditing.",
    )
    parser.add_argument(
        "--finalists",
        type=int,
        default=None,
        help="Override the production full-program finalist count.",
    )
    parser.add_argument(
        "--materialize-winner",
        action="store_true",
        help="Render the winning fixed-offset calendar through the production allocator.",
    )
    parser.add_argument(
        "--only-count",
        type=int,
        default=None,
        help="Audit one horizon run count instead of the complete frequency range.",
    )
    parser.add_argument(
        "--score-offsets",
        default=None,
        help="Additionally score a comma-separated fixed calendar, such as 3,5,7.",
    )
    parser.add_argument(
        "--use-current-prior",
        action="store_true",
        help="Supply the saved internal calendar as the unpreferred warm start.",
    )
    parser.add_argument(
        "--no-underload-debt",
        action="store_true",
        help="Diagnostic only: suppress the accumulated underload-debt term.",
    )
    args = parser.parse_args()
    if args.no_underload_debt:
        original_path_cost = weekly_schedule._continuous_mileage_path_cost

        def without_underload_debt(*path_args, **path_kwargs):
            if path_kwargs.get("underload_only", False):
                return 0.0
            return original_path_cost(*path_args, **path_kwargs)

        weekly_schedule._continuous_mileage_path_cost = without_underload_debt
    if args.beam_width is not None:
        weekly_schedule.MAX_ADAPTIVE_CANDIDATES = max(1, args.beam_width)
    if args.finalists is not None:
        weekly_schedule.JOINT_DATE_FINALISTS = max(1, args.finalists)
    root = args.project_root.resolve()
    config = load_config(root / "config.yaml")
    database = (
        args.database.resolve()
        if args.database is not None
        else resolve_project_path(root, config["paths"]["database"])
    )
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        payload = _load_snapshot(connection)
        states = [FitnessState.model_validate(item) for item in payload["daily_states"]]
        options = [
            [FitnessState.model_validate(item) for item in group]
            for group in payload["daily_state_options"]
        ]
        request = RecommendationRequest.model_validate(payload["request"])
        target = tuple(float(value) for value in payload["target_distance_range"])
        evidence = WeeklyTargetEvidence.model_validate(payload["target_evidence"])
        observed = [
            PlanningActivity(
                start_time=datetime.fromisoformat(item["start_time"]),
                distance_miles=float(item["distance_miles"]),
                moving_minutes=item.get("moving_minutes"),
                easy_minutes=item.get("easy_minutes"),
                baseline_eligible=bool(item.get("baseline_eligible", True)),
                target_distance_miles=item.get("target_distance_miles"),
            )
            for item in payload["observed_planning_activities"]
        ]
        projector = make_expected_target_projector(
            observed,
            states[0].as_of.date(),
            payload["config"],
            pace_min_mile=float(payload["target_projection_pace_min_mile"]),
            maximum_horizon_days=len(states),
            opening_target_range=target,
            opening_evidence=evidence,
        )
        completed = {
            int(offset): [
                TrailingDayActivity.model_validate(item) for item in values
            ]
            for offset, values in payload.get(
                "completed_activities_by_offset", {}
            ).items()
        }
        completed_offsets = set(completed)
        completed_miles = {
            offset: sum(item.distance_miles for item in activities)
            for offset, activities in completed.items()
        }
        forced = set(payload.get("forced_rest_offsets", []))
        internal_row = connection.execute(
            "SELECT value_json FROM app_state "
            "WHERE key='weekly_schedule_internal'"
        ).fetchone()
        internal = json.loads(internal_row[0]) if internal_row is not None else {}
        current_offsets = tuple(
            index
            for index, day in enumerate(internal.get("days", []))
            if day.get("recommendation") is not None
        )

        if args.score_offsets:
            fixed_offsets = tuple(
                int(value.strip())
                for value in args.score_offsets.split(",")
                if value.strip()
            )
            print(
                json.dumps(
                    {
                        "fixed_offsets": fixed_offsets,
                        **_score_parts(
                            fixed_offsets,
                            states=states,
                            options=options,
                            request=request,
                            config=payload["config"],
                            target=target,
                            completed_miles=completed_miles,
                            projector=projector,
                        ),
                    },
                    indent=2,
                )
            )

        ordinary = sum(typical_easy_distance(states[0])) / 2
        horizon_scale = len(states) / 7
        opening_ratio = effective_load_ratio(states[0].recent_load)
        if opening_ratio is None:
            target_fraction = 0.50
        else:
            load_position = min(1.0, max(0.0, (opening_ratio - 0.75) / 0.35))
            target_fraction = 0.75 - load_position * 0.40
        desired_weekly = target[0] + (target[1] - target[0]) * target_fraction
        desired_horizon = max(
            0.0,
            desired_weekly * horizon_scale - sum(completed_miles.values()),
        )
        central = max(1, round(desired_horizon / max(1.0, ordinary)))
        maximum = min(len(states) - len(forced), central + 3)
        minimum = max(1, maximum - MAX_HORIZON_COUNT_OPTIONS + 1)

        print(
            json.dumps(
                {
                    "generated_at": payload["generated_at"],
                    "opening_load_ratio": opening_ratio,
                    "ordinary_easy_midpoint": ordinary,
                    "target_weekly_range": target,
                    "target_fraction": target_fraction,
                    "desired_weekly_miles": desired_weekly,
                    "desired_horizon_miles": desired_horizon,
                    "central_horizon_count": central,
                    "frequency_options": list(range(minimum, maximum + 1)),
                    "beam_width": weekly_schedule.MAX_ADAPTIVE_CANDIDATES,
                    "joint_finalists": weekly_schedule.JOINT_DATE_FINALISTS,
                },
                indent=2,
            ),
            flush=True,
        )

        prefix_cache = {}
        load_cache = {}
        state_cache = {}
        rows: list[dict] = []
        counts = (
            [args.only_count]
            if args.only_count is not None
            else range(minimum, maximum + 1)
        )
        for count in counts:
            joint_cache = {}
            offsets = tuple(
                _adaptive_run_day_offsets_for_frequency(
                    states,
                    request,
                    payload["config"],
                    max(1, round(count / horizon_scale)),
                    target,
                    daily_state_options=options,
                    forced_rest_offsets=forced,
                    completed_run_offsets=completed_offsets,
                    completed_miles_by_offset=completed_miles,
                    horizon_run_count=count,
                    joint_program_scoring=True,
                    joint_cost_cache=joint_cache,
                    prefix_cache=prefix_cache,
                    recommendation_load_cache=load_cache,
                    projected_state_cache=state_cache,
                    prior_run_offsets=(
                        set(current_offsets) if args.use_current_prior else None
                    ),
                    expected_target_projector=projector,
                )
            )
            if not offsets:
                continue
            parts = _score_parts(
                offsets,
                states=states,
                options=options,
                request=request,
                config=payload["config"],
                target=target,
                completed_miles=completed_miles,
                projector=projector,
            )
            row = {
                "horizon_count": count,
                "offsets": offsets,
                "visible_offsets": [value for value in offsets if value < 7],
                **parts,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)

        old_visible_hybrid = tuple(
            sorted(
                {0, 3, 4, 6}
                | {offset for offset in current_offsets if offset >= 7}
            )
        )
        hybrid = {
            "label": "last-night-visible-plus-current-tail",
            "horizon_count": len(old_visible_hybrid),
            "offsets": old_visible_hybrid,
            "visible_offsets": [value for value in old_visible_hybrid if value < 7],
            **_score_parts(
                old_visible_hybrid,
                states=states,
                options=options,
                request=request,
                config=payload["config"],
                target=target,
                completed_miles=completed_miles,
                projector=projector,
            ),
        }
        print(json.dumps(hybrid), flush=True)
        winner = min(rows, key=lambda item: item["selection"])
        print(
            json.dumps(
                {
                    "winner": winner,
                    "current_offsets": current_offsets,
                },
                indent=2,
            )
        )
        if args.materialize_winner:
            original_selector = weekly_schedule.adaptive_run_day_offsets
            weekly_schedule.adaptive_run_day_offsets = (
                lambda *unused_args, **unused_kwargs: list(winner["offsets"])
            )
            try:
                rendered = weekly_schedule.build_weekly_schedule(
                    states,
                    request,
                    payload["config"],
                    target_run_count=int(payload["target_run_count"]),
                    target_distance_range=target,
                    target_evidence=evidence,
                    completed_activities_by_offset=completed,
                    daily_state_options=options,
                    forced_rest_offsets=forced,
                    prior_schedule=None,
                    expected_target_projector=projector,
                )
            finally:
                weekly_schedule.adaptive_run_day_offsets = original_selector
            print(
                json.dumps(
                    {
                        "materialized_winner": [
                            {
                                "offset": index,
                                "date": day.date.isoformat(),
                                "role": day.day_role,
                                "planned_at": (
                                    day.planned_at.isoformat()
                                    if day.planned_at is not None
                                    else None
                                ),
                                "title": (
                                    day.recommendation.title
                                    if day.recommendation is not None
                                    else None
                                ),
                                "distance": (
                                    day.recommendation.distance_range_miles
                                    if day.recommendation is not None
                                    else None
                                ),
                            }
                            for index, day in enumerate(rendered.planning_days)
                        ]
                    },
                    indent=2,
                    default=str,
                )
            )
    finally:
        connection.close()


if __name__ == "__main__":
    main()
