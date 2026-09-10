"""Database orchestration around the pure recommendation rules."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
import json
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfo

from .fitness_state import build_fitness_state
from .progress import PreparedProgressData, prepare_progress_data
from .forecast import (
    NWS_ALERT_CACHE_SECONDS,
    _active_nws_alerts,
    _alerts_for_time,
    get_planned_forecast,
    planned_forecast_options,
)
from .recommendation import recommend_next_run
from .recovery import decay_recovery_load
from .training_load import continuous_distance_rate, short_term_density_half_life_days
from .prescription_matching import archive_weekly_prescriptions
from .run_feedback import get_run_feedback, list_runs
from .weekly_schedule import (
    BASELINE_MINIMUM_AEROBIC_MINUTES,
    WEEKLY_PLANNER_VERSION,
    PLANNING_HORIZON_DAYS,
    PlanningActivity,
    _peak_projected_continuous_mileage_rate,
    build_weekly_schedule,
    derive_weekly_target,
)
from .web.schemas import (
    FitnessState,
    RecommendationRequest,
    RecommendationResponse,
    WeeklyScheduleRequest,
    WeeklyScheduleResponse,
    WeeklyTargetEvidence,
    WeeklyScheduleDay,
    TrailingCalendarDay,
    TrailingDayActivity,
    WorkoutType,
)


def current_fitness_state(
    connection: sqlite3.Connection,
    config: dict,
    request: RecommendationRequest | None = None,
    *,
    prepared_progress: PreparedProgressData | None = None,
    preloaded_runs=None,
    preloaded_latest_feedback=None,
) -> FitnessState:
    return build_fitness_state(
        connection,
        config,
        health_status=request.health_status if request else "normal",
        as_of=request.planned_at if request and request.planned_at else None,
        prepared_progress=prepared_progress,
        preloaded_runs=preloaded_runs,
        preloaded_latest_feedback=preloaded_latest_feedback,
    )


def generate_recommendation(
    connection: sqlite3.Connection,
    config: dict,
    request: RecommendationRequest,
    project_root: str | Path,
) -> tuple[FitnessState, RecommendationResponse]:
    if request.planned_at is None:
        raise ValueError("Choose a planned date and time before generating a recommendation.")
    now = datetime.now(timezone.utc)
    planned = request.planned_at.astimezone(timezone.utc)
    if planned < now - timedelta(hours=2):
        raise ValueError("The planned run time is in the past. Choose a current or future time.")
    if planned > now + timedelta(days=16):
        raise ValueError("Choose a time within the next 16 days; later training is not yet knowable.")
    state = current_fitness_state(connection, config, request)
    forecast = get_planned_forecast(connection, config, project_root, request.planned_at)
    state = state.model_copy(update={"planned_weather": forecast})
    result = recommend_next_run(state, request, config)
    connection.execute(
        """
        INSERT INTO recommendation_history(
            generated_at_utc,fitness_state_json,request_json,result_json
        ) VALUES (?,?,?,?)
        """,
        (
            datetime.now(timezone.utc).isoformat(),
            state.model_dump_json(),
            request.model_dump_json(),
            result.model_dump_json(),
        ),
    )
    connection.execute(
        """
        INSERT INTO app_state(key,value_json,updated_at_utc) VALUES ('current_health',?,?)
        ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json,updated_at_utc=excluded.updated_at_utc
        """,
        (request.model_dump_json(), datetime.now(timezone.utc).isoformat()),
    )
    connection.commit()
    return state, result


def load_current_status(connection: sqlite3.Connection) -> RecommendationRequest:
    row = connection.execute("SELECT value_json FROM app_state WHERE key='current_health'").fetchone()
    if not row:
        return RecommendationRequest(health_status="normal")
    return RecommendationRequest.model_validate(json.loads(row[0]))


def load_latest_recommendation(connection: sqlite3.Connection) -> RecommendationResponse | None:
    row = connection.execute(
        "SELECT result_json FROM recommendation_history ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return RecommendationResponse.model_validate_json(row[0]) if row else None


def load_forced_rest_dates(connection: sqlite3.Connection) -> set[date]:
    row = connection.execute(
        "SELECT value_json FROM app_state WHERE key='weekly_forced_rest_dates'"
    ).fetchone()
    if not row:
        return set()
    try:
        values = json.loads(row[0])
        return {date.fromisoformat(str(value)) for value in values}
    except (TypeError, ValueError, json.JSONDecodeError):
        return set()


def save_forced_rest_dates(
    connection: sqlite3.Connection,
    values: set[date],
) -> None:
    connection.execute(
        """
        INSERT INTO app_state(key,value_json,updated_at_utc)
        VALUES ('weekly_forced_rest_dates',?,?)
        ON CONFLICT(key) DO UPDATE SET
            value_json=excluded.value_json,
            updated_at_utc=excluded.updated_at_utc
        """,
        (
            json.dumps([value.isoformat() for value in sorted(values)]),
            datetime.now(timezone.utc).isoformat(),
        ),
    )


def generate_weekly_schedule(
    connection: sqlite3.Connection,
    config: dict,
    request: WeeklyScheduleRequest,
    project_root: str | Path,
    *,
    discard_prior_schedule: bool = False,
) -> WeeklyScheduleResponse:
    """Create and persist an automatic seven-day schedule starting today."""
    prior_schedule = load_latest_weekly_schedule(connection)
    if discard_prior_schedule:
        # A forced-rest schedule is a temporary constrained solution, not a
        # valid continuity baseline after the user removes that constraint.
        # Re-run the normal optimizer from current evidence so its empty slot
        # cannot perpetuate itself through candidate seeding or soft costs.
        prior_schedule = None
    if (
        prior_schedule is not None
        and prior_schedule.planner_version != WEEKLY_PLANNER_VERSION
    ):
        prior_schedule = None
    elif prior_schedule is not None:
        prior_schedule = prior_schedule.model_copy(
            update={
                "planning_days": load_latest_weekly_planning_days(connection)
            }
        )
    local_zone = ZoneInfo(str(config.get("timezone_default", "UTC")))
    local_now = datetime.now(timezone.utc).astimezone(local_zone)
    start_date = local_now.date()
    forced_rest_dates = {
        value for value in load_forced_rest_dates(connection) if value >= start_date
    }
    forced_rest_offsets = {
        (value - start_date).days
        for value in forced_rest_dates
        if 0 <= (value - start_date).days < PLANNING_HORIZON_DAYS
    }
    run_history = list_runs(connection, limit=5000)
    prepared_progress = prepare_progress_data(connection)
    latest_feedback = (
        get_run_feedback(connection, config, run_history[0].activity_id)
        if run_history
        else None
    )
    history = [
        PlanningActivity(
            run.start_time,
            run.distance_miles,
            moving_minutes=run.moving_minutes,
            easy_minutes=(
                run.session_difficulty.zone_breakdown.easy_minutes
                if run.session_difficulty
                else None
            ),
            baseline_eligible=(
                run.health_tag.value == "normal"
                and run.workout_type
                in {
                    WorkoutType.EASY,
                    WorkoutType.RECOVERY,
                    WorkoutType.RUN_WALK,
                    WorkoutType.UNKNOWN,
                }
                and run.moving_minutes is not None
                and run.moving_minutes >= BASELINE_MINIMUM_AEROBIC_MINUTES
                and run.session_difficulty is not None
                and run.session_difficulty.zone_breakdown.easy_minutes
                >= BASELINE_MINIMUM_AEROBIC_MINUTES
                and not (
                    run.session_difficulty
                    and (
                        run.session_difficulty.is_long_run
                        or run.session_difficulty.is_quality_session
                    )
                )
            ),
        )
        for run in run_history
        if run.start_time
        and run.workout_type not in {WorkoutType.HIKE, WorkoutType.BIKE}
    ]
    target_runs, target_distance, target_evidence = derive_weekly_target(
        history, local_now, config
    )
    # Every day receives timing/weather context before the planner compares
    # candidate date combinations. Date selection is therefore evidence-led,
    # rather than weather being fetched only after fixed offsets are chosen.
    # Keep all weather-backed time choices so projected recovery can prefer a
    # later slot when its continuously decaying load is materially lower.
    candidate_groups: list[list[datetime]] = []
    for offset in range(PLANNING_HORIZON_DAYS):
        candidate_hours = config.get("weather", {}).get(
            "automatic_run_time_hours_local", [7, 12, 19]
        )
        configured_candidates = [
            datetime.combine(
                start_date + timedelta(days=offset),
                time(int(hour), 0),
                tzinfo=local_zone,
            )
            for hour in candidate_hours
        ]
        candidates = configured_candidates
        if offset == 0:
            candidates = [item for item in candidates if item > local_now + timedelta(minutes=10)]
            # The current date remains part of the plan even after every
            # configured time has passed. Keep the final evening slot for the
            # rest of the local day rather than silently advancing to tomorrow.
            # Once that clock time has elapsed, represent the slot as now so
            # recovery and newly issued emergency alerts are evaluated at the
            # time the athlete could actually leave.
            if not candidates:
                candidates = [local_now]
        candidate_groups.append(candidates)
    forecast_options = planned_forecast_options(
        connection,
        config,
        project_root,
        candidate_groups,
    )
    daily_state_options: list[list[FitnessState]] = []
    for options in forecast_options:
        states: list[FitnessState] = []
        base_planned_at = options[0][0]
        base_request = RecommendationRequest(
            health_status=request.health_status,
            planned_at=base_planned_at,
        )
        base_state = current_fitness_state(
            connection,
            config,
            base_request,
            prepared_progress=prepared_progress,
            preloaded_runs=run_history,
            preloaded_latest_feedback=latest_feedback,
        )
        for planned_at, forecast in options:
            delta_days = (
                planned_at - base_planned_at
            ).total_seconds() / 86400

            def shifted(value: float | None) -> float | None:
                return None if value is None else max(0.0, value + delta_days)

            states.append(
                base_state.model_copy(
                    update={
                        "as_of": planned_at,
                        "days_since_last_run": shifted(
                            base_state.days_since_last_run
                        ),
                        "days_since_quality_run": shifted(
                            base_state.days_since_quality_run
                        ),
                        "days_since_long_run": shifted(
                            base_state.days_since_long_run
                        ),
                        "recovery_residual_load": (
                            decay_recovery_load(
                                base_state.recovery_residual_load,
                                max(0.0, delta_days * 24.0),
                            )
                            if base_state.recovery_residual_load is not None
                            else None
                        ),
                        "planned_weather": forecast,
                        # Weekly mileage and daily load checks must use the
                        # same retained capacity evidence. Otherwise the plan
                        # can target 16+ miles while a stale smaller denominator
                        # labels that exact target excessive on later days.
                        "recent_load": base_state.recent_load.model_copy(
                            update={
                                "capacity_reference_miles": max(
                                    base_state.recent_load.capacity_reference_miles
                                    or 0.0,
                                    target_evidence.capacity_reference_miles,
                                ),
                                "sustained_capacity_miles": max(
                                    base_state.recent_load.sustained_capacity_miles
                                    or 0.0,
                                    target_evidence.capacity_reference_miles,
                                ),
                                "acute_distance_to_capacity_ratio": (
                                    base_state.recent_load.trailing_7d.distance_miles
                                    / max(
                                        base_state.recent_load.capacity_reference_miles
                                        or 0.0,
                                        target_evidence.capacity_reference_miles,
                                    )
                                    if max(
                                        base_state.recent_load.capacity_reference_miles
                                        or 0.0,
                                        target_evidence.capacity_reference_miles,
                                    ) > 0
                                    else None
                                ),
                                "continuous_fatigue_miles": (
                                    base_state.recent_load.continuous_fatigue_miles
                                    * 0.5
                                    ** (
                                        max(0.0, delta_days)
                                        / max(
                                            0.1,
                                            float(
                                                config.get("coaching", {}).get(
                                                    "continuous_fatigue_half_life_days",
                                                    7,
                                                )
                                            ),
                                        )
                                    )
                                    if base_state.recent_load.continuous_fatigue_miles
                                    is not None
                                    else None
                                ),
                                "continuous_fatigue_to_capacity_ratio": (
                                    (
                                        base_state.recent_load.continuous_fatigue_miles
                                        * 0.5
                                        ** (
                                            max(0.0, delta_days)
                                            / max(
                                                0.1,
                                                float(
                                                    config.get("coaching", {}).get(
                                                        "continuous_fatigue_half_life_days",
                                                        7,
                                                    )
                                                ),
                                            )
                                        )
                                    )
                                    / max(
                                        base_state.recent_load.capacity_reference_miles
                                        or 0.0,
                                        target_evidence.capacity_reference_miles,
                                    )
                                    if base_state.recent_load.continuous_fatigue_miles
                                    is not None
                                    and max(
                                        base_state.recent_load.capacity_reference_miles
                                        or 0.0,
                                        target_evidence.capacity_reference_miles,
                                    )
                                    > 0
                                    else None
                                ),
                                "continuous_distance_miles": (
                                    base_state.recent_load.continuous_distance_miles
                                    * 0.5
                                    ** (
                                        max(0.0, delta_days)
                                        / max(
                                            0.1,
                                            float(
                                                config.get("coaching", {}).get(
                                                    "continuous_fatigue_half_life_days",
                                                    7,
                                                )
                                            ),
                                        )
                                    )
                                    if base_state.recent_load.continuous_distance_miles
                                    is not None
                                    else None
                                ),
                                "continuous_short_term_distance_miles": (
                                    base_state.recent_load.continuous_short_term_distance_miles
                                    * 0.5
                                    ** (
                                        max(0.0, delta_days)
                                        / max(
                                            0.1,
                                            short_term_density_half_life_days(
                                                float(
                                                    config.get("coaching", {}).get(
                                                        "continuous_fatigue_half_life_days",
                                                        7,
                                                    )
                                                )
                                            ),
                                        )
                                    )
                                    if base_state.recent_load.continuous_short_term_distance_miles
                                    is not None
                                    else None
                                ),
                            }
                        ),
                    }
                )
            )
        daily_state_options.append(states)
    # Forecast options are weather-ranked. The leading state remains the
    # neutral/default date state; the planner may select another exact time.
    daily_states = [options[0] for options in daily_state_options]
    shared_request = RecommendationRequest(
        health_status=request.health_status,
    )
    completed_today = [
        TrailingDayActivity(
            activity_id=run.activity_id,
            start_time=run.start_time,
            distance_miles=run.distance_miles,
            workout_type=run.workout_type,
            health_tag=run.health_tag,
        )
        for run in run_history
        if run.start_time
        and run.start_time.astimezone(local_zone).date() == start_date
        and run.workout_type not in {WorkoutType.HIKE, WorkoutType.BIKE}
    ]
    completed_by_offset = {0: completed_today} if completed_today else {}
    result = build_weekly_schedule(
        daily_states,
        shared_request,
        config,
        target_run_count=target_runs,
        target_distance_range=target_distance,
        target_evidence=target_evidence,
        completed_activities_by_offset=completed_by_offset,
        daily_state_options=daily_state_options,
        forced_rest_offsets=forced_rest_offsets,
        prior_schedule=prior_schedule,
    )
    # Recent training is a completed-calendar-day lookback.  Including today
    # before it is over makes "no activity yet" look like a completed rest day
    # and drops the actual seventh prior day from the strip.
    trailing_days: list[TrailingCalendarDay] = []
    for offset in range(7, 0, -1):
        calendar_date = local_now.date() - timedelta(days=offset)
        activities = [
            run for run in run_history
            if run.start_time and run.start_time.astimezone(local_zone).date() == calendar_date
        ]
        running = [
            run for run in activities
            if run.workout_type not in {WorkoutType.HIKE, WorkoutType.BIKE}
        ]
        if not activities:
            role = "rest_recovery_day"
        elif not running:
            role = "cross_training_day"
        elif any(run.workout_type == WorkoutType.RECOVERY for run in running):
            role = "recovery_run_day"
        elif any(run.workout_type in {WorkoutType.INTERVALS, WorkoutType.TEMPO_THRESHOLD, WorkoutType.RACE} for run in running):
            role = "quality_run_day"
        elif any(run.workout_type == WorkoutType.LONG for run in running):
            role = "long_run_day"
        else:
            role = "run_day"
        trailing_days.append(
            TrailingCalendarDay(
                date=calendar_date,
                day_role=role,
                total_distance_miles=sum(run.distance_miles for run in activities),
                activities=[
                    TrailingDayActivity(
                        activity_id=run.activity_id,
                        start_time=run.start_time,
                        distance_miles=run.distance_miles,
                        workout_type=run.workout_type,
                        health_tag=run.health_tag,
                    )
                    for run in activities
                ],
            )
        )
    result = result.model_copy(update={"trailing_days": trailing_days})
    archive_weekly_prescriptions(connection, result)
    connection.execute(
        """
        INSERT INTO app_state(key,value_json,updated_at_utc) VALUES ('weekly_schedule',?,?)
        ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json,updated_at_utc=excluded.updated_at_utc
        """,
        (result.model_dump_json(), datetime.now(timezone.utc).isoformat()),
    )
    connection.execute(
        """
        INSERT INTO app_state(key,value_json,updated_at_utc)
        VALUES ('weekly_planner_snapshot',?,?)
        ON CONFLICT(key) DO UPDATE SET
            value_json=excluded.value_json,
            updated_at_utc=excluded.updated_at_utc
        """,
        (
            json.dumps(
                {
                    "planner_version": WEEKLY_PLANNER_VERSION,
                    "generated_at": result.generated_at.isoformat(),
                    "request": shared_request.model_dump(mode="json"),
                    "config": config,
                    "daily_state_options": [
                        [state.model_dump(mode="json") for state in options]
                        for options in daily_state_options
                    ],
                    "target_run_count": target_runs,
                    "target_distance_range": list(target_distance),
                    "target_evidence": target_evidence.model_dump(mode="json"),
                    "completed_activities_by_offset": {
                        str(offset): [item.model_dump(mode="json") for item in items]
                        for offset, items in completed_by_offset.items()
                    },
                    "forced_rest_offsets": sorted(forced_rest_offsets),
                    "prior_schedule": (
                        prior_schedule.model_dump(mode="json")
                        if prior_schedule is not None
                        else None
                    ),
                    "result": result.model_dump(mode="json"),
                },
                sort_keys=True,
                default=str,
            ),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    connection.execute(
        """
        INSERT INTO app_state(key,value_json,updated_at_utc)
        VALUES ('weekly_schedule_internal',?,?)
        ON CONFLICT(key) DO UPDATE SET
            value_json=excluded.value_json,
            updated_at_utc=excluded.updated_at_utc
        """,
        (
            json.dumps(
                {
                    "planner_version": result.planner_version,
                    "start_date": result.start_date.isoformat(),
                    "days": [
                        day.model_dump(mode="json")
                        for day in result.planning_days
                    ],
                }
            ),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    current = RecommendationRequest(
        health_status=request.health_status,
        planned_at=None,
    )
    connection.execute(
        """
        INSERT INTO app_state(key,value_json,updated_at_utc) VALUES ('current_health',?,?)
        ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json,updated_at_utc=excluded.updated_at_utc
        """,
        (current.model_dump_json(), datetime.now(timezone.utc).isoformat()),
    )
    connection.commit()
    return result


def load_latest_weekly_schedule(connection: sqlite3.Connection) -> WeeklyScheduleResponse | None:
    row = connection.execute(
        "SELECT value_json FROM app_state WHERE key='weekly_schedule'"
    ).fetchone()
    if not row:
        return None
    payload = json.loads(row[0])
    # Schedules saved before automatic by-day timing carried this obsolete
    # global preference.  Ignore it during the one-time schema transition.
    payload.pop("preferred_time", None)
    result = WeeklyScheduleResponse.model_validate(payload)
    # The builder's summary may explain forced-rest constraints, deferred
    # sessions, or readiness substitutions in addition to mileage alignment.
    # Preserve that context when the saved plan is reloaded.
    return result


def load_latest_weekly_planning_days(
    connection: sqlite3.Connection,
) -> list[WeeklyScheduleDay]:
    """Load the prior full horizon used only for soft plan continuity."""

    row = connection.execute(
        "SELECT value_json FROM app_state WHERE key='weekly_schedule_internal'"
    ).fetchone()
    if not row:
        return []
    payload = json.loads(row[0])
    if payload.get("planner_version") != WEEKLY_PLANNER_VERSION:
        return []
    return [
        WeeklyScheduleDay.model_validate(day)
        for day in payload.get("days", [])
    ]


def replay_latest_weekly_schedule(
    connection: sqlite3.Connection,
) -> tuple[WeeklyScheduleResponse, WeeklyScheduleResponse] | None:
    """Re-run the optimizer from its exact persisted inputs.

    The first item is the saved result and the second is the replay. Callers
    can compare every coaching decision while deliberately ignoring the wall
    clock timestamp attached to the newly materialized response.
    """

    row = connection.execute(
        "SELECT value_json FROM app_state WHERE key='weekly_planner_snapshot'"
    ).fetchone()
    if row is None:
        return None
    payload = json.loads(row[0])
    if payload.get("planner_version") != WEEKLY_PLANNER_VERSION:
        return None
    options = [
        [FitnessState.model_validate(state) for state in states]
        for states in payload["daily_state_options"]
    ]
    completed = {
        int(offset): [TrailingDayActivity.model_validate(item) for item in items]
        for offset, items in payload.get("completed_activities_by_offset", {}).items()
    }
    prior_payload = payload.get("prior_schedule")
    replay = build_weekly_schedule(
        [states[0] for states in options],
        RecommendationRequest.model_validate(payload["request"]),
        payload["config"],
        target_run_count=payload["target_run_count"],
        target_distance_range=tuple(payload["target_distance_range"]),
        target_evidence=WeeklyTargetEvidence.model_validate(
            payload["target_evidence"]
        ),
        completed_activities_by_offset=completed,
        daily_state_options=options,
        forced_rest_offsets=set(payload.get("forced_rest_offsets", [])),
        prior_schedule=(
            WeeklyScheduleResponse.model_validate(prior_payload)
            if prior_payload is not None
            else None
        ),
    )
    saved = WeeklyScheduleResponse.model_validate(payload["result"])

    def preserve_materialization_times(
        replay_days: list[WeeklyScheduleDay],
        saved_days: list[WeeklyScheduleDay],
    ) -> list[WeeklyScheduleDay]:
        saved_by_date = {day.date: day for day in saved_days}
        output: list[WeeklyScheduleDay] = []
        for day in replay_days:
            saved_day = saved_by_date.get(day.date)
            if (
                day.recommendation is not None
                and saved_day is not None
                and saved_day.recommendation is not None
            ):
                day = day.model_copy(
                    update={
                        "recommendation": day.recommendation.model_copy(
                            update={
                                "generated_at": (
                                    saved_day.recommendation.generated_at
                                )
                            }
                        )
                    }
                )
            output.append(day)
        return output

    replay = replay.model_copy(
        update={
            "generated_at": saved.generated_at,
            "emergency_alerts_checked_at": saved.emergency_alerts_checked_at,
            "trailing_days": saved.trailing_days,
            "days": preserve_materialization_times(replay.days, saved.days),
            "planning_days": preserve_materialization_times(
                replay.planning_days, saved.planning_days
            ),
        }
    )
    return saved, replay


def _enrich_saved_schedule_load_context(
    connection: sqlite3.Connection,
    schedule: WeeklyScheduleResponse,
    config: dict,
    as_of: datetime,
) -> WeeklyScheduleResponse:
    """Add display load context without asking the optimizer for a new plan."""

    planning_days = load_latest_weekly_planning_days(connection)
    if not planning_days:
        planning_days = list(schedule.days)
    visible_scheduled = sum(
        sum(day.recommendation.distance_range_miles) / 2
        for day in schedule.days
        if day.recommendation is not None
        and day.recommendation.workout_type != WorkoutType.REST
        and day.recommendation.distance_range_miles is not None
    )
    context_days = planning_days[:14] if len(planning_days) >= 14 else []
    context_miles = sum(
        activity.distance_miles
        for day in context_days
        for activity in day.completed_activities
    ) + sum(
        sum(day.recommendation.distance_range_miles) / 2
        for day in context_days
        if day.recommendation is not None
        and day.recommendation.workout_type != WorkoutType.REST
        and day.recommendation.distance_range_miles is not None
    )
    rate_14d = (
        context_miles / 2.0
        if len(context_days) == 14
        else None
    )
    half_life_days = float(
        config.get("coaching", {}).get(
            "continuous_fatigue_half_life_days",
            7,
        )
    )
    # This is a cheap distance-only projection. It deliberately does not
    # rebuild fitness state or invoke any recommendation/optimizer work.
    sessions = prepare_progress_data(connection).sessions
    opening_rate = continuous_distance_rate(
        sessions,
        as_of,
        half_life_days=half_life_days,
    )
    peak = _peak_projected_continuous_mileage_rate(
        opening_rate,
        as_of,
        list(schedule.days),
        half_life_days=half_life_days,
    )
    summary = schedule.summary
    target_low, target_high = schedule.target_distance_range_miles
    visible_midpoint = sum(schedule.projected_distance_range_miles) / 2
    special_rest_context = bool(
        any(day.forced_rest for day in schedule.days)
        or "changed to rest" in summary
    )
    if (
        not special_rest_context
        and rate_14d is not None
        and target_low <= rate_14d <= target_high
    ):
        if visible_midpoint > target_high:
            summary = (
                "The first seven days are busier, but your 14-day average and "
                "rolling mileage load stay on target."
            )
        elif visible_midpoint < target_low:
            summary = (
                "The first seven days are lighter, but your 14-day average "
                "stays on target."
            )
    enriched = schedule.model_copy(
        update={
            "visible_7d_scheduled_miles": round(visible_scheduled, 2),
            "planned_14d_weekly_rate": (
                round(rate_14d, 2) if rate_14d is not None else None
            ),
            "peak_projected_continuous_mileage_rate": (
                round(peak, 2) if peak is not None else None
            ),
            "summary": summary,
            "planning_days": planning_days,
        }
    )
    connection.execute(
        """
        INSERT INTO app_state(key,value_json,updated_at_utc)
        VALUES ('weekly_schedule',?,?)
        ON CONFLICT(key) DO UPDATE SET
            value_json=excluded.value_json,
            updated_at_utc=excluded.updated_at_utc
        """,
        (
            enriched.model_dump_json(),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    connection.commit()
    return enriched


def _today_plan_time_is_stale(
    schedule: WeeklyScheduleResponse,
    local_now: datetime,
    config: dict,
) -> bool:
    """Refresh today when its feasible time set changes during the day."""
    today = next((day for day in schedule.days if day.date == local_now.date()), None)
    if today is None:
        return False
    configured_hours = config.get("weather", {}).get(
        "automatic_run_time_hours_local", [7, 12, 19]
    )
    if not configured_hours:
        return False
    hours = sorted({int(hour) for hour in configured_hours})

    if today.recommendation is None:
        # A constraint or an already-completed run is stable.  An ordinary
        # planner-created rest day is not: once an early candidate closes, run
        # the whole-week optimizer again around the remaining slots.  Exclude
        # the final slot because it intentionally remains available all day.
        if today.forced_rest or today.completed_activities:
            return False
        generated_local = schedule.generated_at.astimezone(local_now.tzinfo)
        return any(
            generated_local
            < datetime.combine(
                local_now.date(),
                time(hour, 0),
                tzinfo=local_now.tzinfo,
            )
            - timedelta(minutes=10)
            <= local_now
            for hour in hours[:-1]
        )

    if today.planned_at is None:
        return True
    final_slot = datetime.combine(
        local_now.date(),
        time(hours[-1], 0),
        tzinfo=local_now.tzinfo,
    )
    planned_at = today.planned_at.astimezone(local_now.tzinfo)
    return planned_at < local_now and planned_at < final_slot


def _weekly_plan_shape_is_stale(schedule: WeeklyScheduleResponse) -> bool:
    """Reject saved plans produced before continuous-horizon coordination."""
    return (
        getattr(schedule, "planner_version", 1) < WEEKLY_PLANNER_VERSION
        or len(schedule.trailing_days) != 7
        or schedule.trailing_days[-1].date
        != schedule.start_date - timedelta(days=1)
    )


def _weekly_emergency_alerts_are_stale(
    schedule: WeeklyScheduleResponse,
    local_now: datetime,
    config: dict,
) -> bool:
    """Refresh an actionable same-day plan when its official alert check ages."""
    if not bool(config.get("weather", {}).get("emergency_alerts_enabled", False)):
        return False
    today = next(
        (day for day in schedule.days if day.date == local_now.date()),
        None,
    )
    if (
        today is None
        or today.recommendation is None
    ):
        return False
    checked_at = (
        getattr(schedule, "emergency_alerts_checked_at", None)
        or schedule.generated_at
    )
    age_seconds = (
        local_now - checked_at.astimezone(local_now.tzinfo)
    ).total_seconds()
    return age_seconds >= NWS_ALERT_CACHE_SECONDS


def _refresh_saved_schedule_emergency_alerts(
    connection: sqlite3.Connection,
    schedule: WeeklyScheduleResponse,
    config: dict,
    project_root: str | Path,
    local_now: datetime,
) -> WeeklyScheduleResponse | None:
    """Patch unchanged alerts cheaply; return None when a replan is required."""
    today_index = next(
        (
            index
            for index, day in enumerate(schedule.days)
            if day.date == local_now.date() and day.recommendation is not None
        ),
        None,
    )
    if today_index is None:
        return schedule
    day = schedule.days[today_index]
    result = day.recommendation
    assert result is not None
    alerts, checked = _active_nws_alerts(
        connection,
        config,
        project_root,
    )
    planned_at = result.planned_for or day.planned_at
    alert_moment = (
        max(planned_at.astimezone(local_now.tzinfo), local_now)
        if planned_at
        else None
    )
    applicable = _alerts_for_time(alerts, alert_moment) if alert_moment else []
    existing = (
        result.planned_weather.emergency_alerts
        if result.planned_weather
        else []
    )
    if not checked:
        # A transient API failure must not erase a warning that the saved plan
        # already knew about. The next refresh can remove it once the official
        # check succeeds or its own expiry makes it inapplicable.
        applicable = existing
    old_blocking = {
        alert.alert_id for alert in existing if alert.blocks_outdoor_run
    }
    new_blocking = {
        alert.alert_id for alert in applicable if alert.blocks_outdoor_run
    }
    if old_blocking != new_blocking:
        return None

    updated_result = result
    if result.planned_weather is not None:
        updated_result = result.model_copy(
            update={
                "planned_weather": result.planned_weather.model_copy(
                    update={
                        "emergency_alerts_checked": checked,
                        "emergency_alerts": applicable,
                    }
                )
            }
        )
    updated_days = list(schedule.days)
    updated_days[today_index] = day.model_copy(
        update={"recommendation": updated_result}
    )
    refreshed = schedule.model_copy(
        update={
            "emergency_alerts_checked_at": local_now,
            "days": updated_days,
        }
    )
    connection.execute(
        """
        INSERT INTO app_state(key,value_json,updated_at_utc)
        VALUES ('weekly_schedule',?,?)
        ON CONFLICT(key) DO UPDATE SET
            value_json=excluded.value_json,
            updated_at_utc=excluded.updated_at_utc
        """,
        (
            refreshed.model_dump_json(),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    connection.commit()
    return refreshed


def ensure_current_weekly_schedule(
    connection: sqlite3.Connection,
    config: dict,
    project_root: str | Path,
) -> WeeklyScheduleResponse:
    """Return today's leading schedule, regenerating stale saved state."""

    current = load_latest_weekly_schedule(connection)
    local_now = datetime.now(timezone.utc).astimezone(
        ZoneInfo(str(config.get("timezone_default", "UTC")))
    )
    reusable = (
        current is not None
        and current.start_date == local_now.date()
        and not _today_plan_time_is_stale(current, local_now, config)
        and not _weekly_plan_shape_is_stale(current)
    )
    if reusable and current is not None:
        if (
            current.visible_7d_scheduled_miles is None
            or current.planned_14d_weekly_rate is None
            or current.peak_projected_continuous_mileage_rate is None
        ):
            current = _enrich_saved_schedule_load_context(
                connection,
                current,
                config,
                local_now,
            )
        if _weekly_emergency_alerts_are_stale(current, local_now, config):
            refreshed = _refresh_saved_schedule_emergency_alerts(
                connection,
                current,
                config,
                project_root,
                local_now,
            )
            if refreshed is not None:
                return refreshed
        else:
            return current
    saved = load_current_status(connection)
    return generate_weekly_schedule(
        connection,
        config,
        WeeklyScheduleRequest(health_status=saved.health_status),
        project_root,
    )
