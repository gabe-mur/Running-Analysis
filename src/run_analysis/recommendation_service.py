"""Database orchestration around the pure recommendation rules."""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
import json
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfo

from .fitness_state import build_fitness_state
from .forecast import choose_planned_forecast, get_planned_forecast
from .recommendation import recommend_next_run
from .run_feedback import list_runs
from .weekly_schedule import (
    PlanningActivity,
    automatic_run_day_offsets,
    build_weekly_schedule,
    derive_weekly_target,
)
from .web.schemas import (
    FitnessState,
    RecommendationRequest,
    RecommendationResponse,
    WeeklyScheduleRequest,
    WeeklyScheduleResponse,
    TrailingCalendarDay,
    TrailingDayActivity,
    WorkoutType,
)


def current_fitness_state(
    connection: sqlite3.Connection,
    config: dict,
    request: RecommendationRequest | None = None,
) -> FitnessState:
    return build_fitness_state(
        connection,
        config,
        health_status=request.health_status if request else "normal",
        as_of=request.planned_at if request and request.planned_at else None,
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


def generate_weekly_schedule(
    connection: sqlite3.Connection,
    config: dict,
    request: WeeklyScheduleRequest,
    project_root: str | Path,
) -> WeeklyScheduleResponse:
    """Create and persist an automatic seven-day schedule starting tomorrow."""
    local_zone = ZoneInfo(str(config.get("timezone_default", "UTC")))
    local_now = datetime.now(timezone.utc).astimezone(local_zone)
    start_date = local_now.date() + timedelta(days=1)
    run_history = list_runs(connection, limit=5000)
    history = [
        PlanningActivity(run.start_time, run.distance_miles)
        for run in run_history
        if run.start_time
        and run.workout_type not in {WorkoutType.HIKE, WorkoutType.BIKE}
    ]
    target_runs, target_distance, target_evidence = derive_weekly_target(
        history, local_now, config
    )
    daily_states: list[FitnessState] = []
    for offset in range(7):
        planned_at = datetime.combine(start_date + timedelta(days=offset), time(12, 0), tzinfo=local_zone)
        if offset == 0 and planned_at <= local_now:
            planned_at = local_now + timedelta(minutes=15)
        daily_request = RecommendationRequest(
            health_status=request.health_status,
            planned_at=planned_at,
            notes=request.notes,
        )
        daily_states.append(current_fitness_state(connection, config, daily_request))
    selected_offsets = automatic_run_day_offsets(
        daily_states[0],
        request.health_status,
        config,
        target_runs,
    )
    for offset in (item for item in selected_offsets if 0 <= item < len(daily_states)):
        candidate_hours = config.get("weather", {}).get(
            "automatic_run_time_hours_local", [7, 12, 19]
        )
        candidates = [
            datetime.combine(
                start_date + timedelta(days=offset),
                time(int(hour), 0),
                tzinfo=local_zone,
            )
            for hour in candidate_hours
        ]
        if offset == 0:
            candidates = [item for item in candidates if item > local_now + timedelta(minutes=10)]
        chosen_at, forecast = choose_planned_forecast(
            connection,
            config,
            project_root,
            candidates,
        )
        chosen_request = RecommendationRequest(
            health_status=request.health_status,
            planned_at=chosen_at,
            notes=request.notes,
        )
        daily_states[offset] = current_fitness_state(
            connection, config, chosen_request
        ).model_copy(update={"planned_weather": forecast})
    shared_request = RecommendationRequest(
        health_status=request.health_status,
        notes=request.notes,
    )
    result = build_weekly_schedule(
        daily_states,
        shared_request,
        config,
        target_run_count=target_runs,
        target_distance_range=target_distance,
        target_evidence=target_evidence,
    )
    trailing_days: list[TrailingCalendarDay] = []
    for offset in range(6, -1, -1):
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
    connection.execute(
        """
        INSERT INTO app_state(key,value_json,updated_at_utc) VALUES ('weekly_schedule',?,?)
        ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json,updated_at_utc=excluded.updated_at_utc
        """,
        (result.model_dump_json(), datetime.now(timezone.utc).isoformat()),
    )
    current = RecommendationRequest(
        health_status=request.health_status,
        planned_at=None,
        notes=request.notes,
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
    projected_low, projected_high = result.projected_distance_range_miles
    target_low, target_high = result.target_distance_range_miles
    if result.target_evidence.capacity_reference_miles <= 0:
        summary = "This is a starter plan until more runs are available."
    elif projected_high < target_low:
        summary = "This week stays below your usual range."
    elif projected_low > target_high:
        summary = "This week is above your usual range, so review each workout before following it."
    else:
        summary = "This fits your recent training."
    return result.model_copy(update={"summary": summary})


def ensure_current_weekly_schedule(
    connection: sqlite3.Connection,
    config: dict,
    project_root: str | Path,
) -> WeeklyScheduleResponse:
    """Return tomorrow's leading schedule, regenerating stale saved state."""

    current = load_latest_weekly_schedule(connection)
    local_today = datetime.now(timezone.utc).astimezone(
        ZoneInfo(str(config.get("timezone_default", "UTC")))
    ).date()
    if current is not None and current.start_date == local_today + timedelta(days=1):
        return current
    saved = load_current_status(connection)
    return generate_weekly_schedule(
        connection,
        config,
        WeeklyScheduleRequest(health_status=saved.health_status, notes=saved.notes),
        project_root,
    )
