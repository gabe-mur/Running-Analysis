"""Database orchestration around the pure recommendation rules."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
import json
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfo

from .fitness_state import build_fitness_state
from .forecast import (
    NWS_ALERT_CACHE_SECONDS,
    _active_nws_alerts,
    _alerts_for_time,
    get_planned_forecast,
    planned_forecast_options,
)
from .recommendation import recommend_next_run
from .prescription_matching import archive_weekly_prescriptions
from .run_feedback import list_runs
from .weekly_schedule import (
    WEEKLY_PLANNER_VERSION,
    PLANNING_HORIZON_DAYS,
    PlanningActivity,
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
) -> WeeklyScheduleResponse:
    """Create and persist an automatic seven-day schedule starting today."""
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
    history = [
        PlanningActivity(run.start_time, run.distance_miles)
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
            connection, config, base_request
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
    result = build_weekly_schedule(
        daily_states,
        shared_request,
        config,
        target_run_count=target_runs,
        target_distance_range=target_distance,
        target_evidence=target_evidence,
        completed_activities_by_offset=({0: completed_today} if completed_today else None),
        daily_state_options=daily_state_options,
        forced_rest_offsets=forced_rest_offsets,
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
