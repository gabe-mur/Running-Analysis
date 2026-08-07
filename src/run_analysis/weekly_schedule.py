"""Pure seven-day schedule construction from daily fitness states.

The planner chooses run days automatically, then evaluates the existing
inspectable recommendation rules on those days. Planned sessions are projected
into later daily states so consecutive days, load, and workout recency are
intentional rather than independent recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from math import exp, floor
from zoneinfo import ZoneInfo

from .recommendation import recommend_next_run, typical_easy_distance
from .web.schemas import (
    CurrentHealthStatus,
    FitnessState,
    LoadContext,
    LoadWindow,
    RecommendationRequest,
    RecommendationResponse,
    SessionDifficulty,
    TrailingDayActivity,
    WeeklyScheduleDay,
    WeeklyScheduleResponse,
    WeeklyTargetEvidence,
    WorkoutType,
    ZoneBreakdown,
)


@dataclass(frozen=True, slots=True)
class PlanningActivity:
    start_time: datetime
    distance_miles: float


def _half_mile(value: float) -> float:
    return round(value * 2) / 2


def derive_weekly_target(
    activities: list[PlanningActivity],
    as_of: datetime,
    config: dict,
) -> tuple[int, tuple[float, float], WeeklyTargetEvidence]:
    """Infer a leading-week target from acute, chronic, and sustained history."""
    zone = ZoneInfo(str(config.get("timezone_default", "UTC")))
    end = as_of.astimezone(zone).date()
    daily: dict = {}
    for activity in activities:
        day = activity.start_time.astimezone(zone).date()
        if day <= end:
            daily[day] = daily.get(day, 0.0) + activity.distance_miles
    if not daily:
        evidence = WeeklyTargetEvidence(
            recent_7d_miles=0,
            chronic_42d_weekly_miles=0,
            best_sustained_28d_weekly_miles=0,
            peak_7d_miles=0,
            demonstrated_run_days_per_week=0,
            capacity_reference_miles=0,
            rationale="No running history is available; conservative starter defaults apply.",
        )
        return 2, (6.0, 8.0), evidence

    first = min(daily)
    alpha = 1.0 - exp(-1.0 / 42.0)
    chronic_daily = 0.0
    day = first
    while day <= end:
        chronic_daily += alpha * (daily.get(day, 0.0) - chronic_daily)
        day += timedelta(days=1)
    chronic_weekly = chronic_daily * 7.0

    history_start = max(first, end - timedelta(days=364))
    windows: list[tuple] = []
    day = history_start
    while day <= end:
        seven_start = day - timedelta(days=6)
        twenty_eight_start = day - timedelta(days=27)
        seven = sum(miles for date, miles in daily.items() if seven_start <= date <= day)
        twenty_eight = sum(miles for date, miles in daily.items() if twenty_eight_start <= date <= day) / 4.0
        run_days = sum(1 for date in daily if twenty_eight_start <= date <= day) / 4.0
        windows.append((day, seven, twenty_eight, run_days))
        day += timedelta(days=1)
    peak_seven = max(item[1] for item in windows)
    best_sustained = max(windows, key=lambda item: (item[2], item[0]))
    recent_seven = windows[-1][1]
    half_life = float(config.get("coaching", {}).get("capacity_retention_half_life_days", 42))
    days_since_sustained_peak = max(0, (end - best_sustained[0]).days)
    retention_grace = int(config.get("coaching", {}).get("capacity_retention_grace_days", 28))
    decay_days = max(0, days_since_sustained_peak - retention_grace)
    retained_sustained = best_sustained[2] * 0.5 ** (decay_days / half_life)
    capacity_reference = max(
        chronic_weekly,
        retained_sustained,
        min(recent_seven, best_sustained[2]),
    )
    target_low = max(6.0, _half_mile(capacity_reference * 0.95))
    target_high = max(target_low + 0.5, _half_mile(min(peak_seven, capacity_reference * 1.10)))

    # Planned frequency must be earned over a sustained window. Using the
    # trailing seven-day count directly creates a self-reinforcing ratchet: one
    # incidental short run can raise next week's target, and completing the
    # extra prescribed day then appears to validate the higher frequency. A
    # 28-day rate still adapts to a genuine change in training pattern without
    # letting one unusually busy week rewrite the schedule.
    recent_28d_run_days_per_week = (
        sum(1 for date in daily if end - timedelta(days=27) <= date <= end) / 4.0
    )
    demonstrated = max(float(recent_28d_run_days_per_week), float(best_sustained[3]))
    target_runs = max(2, min(7, floor(demonstrated + 0.5)))
    evidence = WeeklyTargetEvidence(
        recent_7d_miles=recent_seven,
        chronic_42d_weekly_miles=chronic_weekly,
        best_sustained_28d_weekly_miles=best_sustained[2],
        peak_7d_miles=peak_seven,
        demonstrated_run_days_per_week=demonstrated,
        capacity_reference_miles=capacity_reference,
        rationale=(
            "Mileage target blends the current seven days, a slowly decaying 42-day chronic level, "
            "and the best sustained 28-day level from the trailing year. Demonstrated capacity is "
            "retained through a short disruption, then decays gradually. Frequency advances only "
            "after the higher pattern is demonstrated across roughly four weeks. One rest day "
            "between runs is the default cadence; "
            "consecutive days are used only when a higher demonstrated frequency calls for them."
        ),
    )
    return target_runs, (target_low, target_high), evidence


def automatic_run_day_offsets(
    state: FitnessState,
    health: CurrentHealthStatus,
    config: dict | None = None,
    target_run_count: int | None = None,
) -> list[int]:
    """Choose run days from the explicit target, not a disrupted recent period."""
    if health == CurrentHealthStatus.PAIN_OR_INJURY_CONCERN:
        return []
    target = target_run_count
    if target is None:
        observed = max(state.recent_load.trailing_7d.activity_count, floor(state.running_days_28d / 4 + 0.5))
        target = observed
    target = max(1, min(7, int(target)))
    if health == CurrentHealthStatus.SICK_OR_RECOVERING:
        target = min(2, target)
    if health == CurrentHealthStatus.LITTLE_TIRED:
        target = max(2, target - 1)
    ordinary_patterns = {
        1: [0],
        2: [0, 2],
        3: [0, 2, 4],
        4: [0, 2, 4, 6],
        5: [0, 1, 3, 4, 6],
        6: [0, 1, 2, 4, 5, 6],
        7: [0, 1, 2, 3, 4, 5, 6],
    }
    # A normal-cost run on the preceding calendar day makes today the default
    # rest day. A genuinely low-cost run can preserve the ordinary cadence
    # when its relative distance/duration/HR load and the accumulated load all
    # support an intentional consecutive day. Offsets beyond day six remain
    # intentional: the builder never pulls work earlier merely to fill the
    # displayed horizon.
    recent_patterns = {
        1: [1],
        2: [1, 3],
        3: [1, 3, 5],
        4: [1, 3, 5, 7],
        5: [1, 2, 4, 5, 7],
        6: [1, 2, 3, 5, 6, 7],
        7: [1, 2, 3, 4, 5, 6, 7],
    }
    typical_rest_days = int((config or {}).get("coaching", {}).get("typical_rest_days_between_runs", 1))
    ran_today_or_yesterday = False
    if typical_rest_days >= 1 and state.days_since_last_run is not None:
        inferred_last_date = (state.as_of - timedelta(days=state.days_since_last_run)).date()
        ran_today_or_yesterday = inferred_last_date >= state.as_of.date() - timedelta(days=1)
    low_cost_consecutive, _ = consecutive_day_evidence(state, config)
    low_cost_consecutive = low_cost_consecutive and health == CurrentHealthStatus.NORMAL
    return (
        recent_patterns[target]
        if ran_today_or_yesterday and not low_cost_consecutive
        else ordinary_patterns[target]
    )


def consecutive_day_evidence(
    state: FitnessState,
    config: dict | None = None,
) -> tuple[bool, dict[str, float | bool | None]]:
    """Decide whether the latest run is cheap enough to preserve a next-day run.

    The latest activity is never ignored: its distance, duration, HR load,
    workout type, RPE, and accumulated seven-day load all participate. Ratios
    are athlete-relative so an incidental run is not treated like a normal
    training session, while a short but genuinely hard workout still earns
    recovery.
    """

    latest = state.last_run
    window = state.recent_load.trailing_28d
    settings = (config or {}).get("coaching", {})
    high_load_threshold = float(settings.get("high_load_ratio", 1.30))
    if latest is None or window.activity_count <= 0:
        return False, {
            "latest_available": latest is not None,
            "typical_history_available": window.activity_count > 0,
        }

    typical_distance = window.distance_miles / window.activity_count
    typical_minutes = window.moving_minutes / window.activity_count
    typical_zone_load = (
        window.zone_load / window.activity_count
        if window.zone_load is not None and window.zone_load > 0
        else None
    )
    distance_ratio = latest.distance_miles / typical_distance if typical_distance else None
    duration_ratio = latest.moving_minutes / typical_minutes if typical_minutes else None
    zone_load_ratio = (
        latest.zone_load / typical_zone_load
        if latest.zone_load is not None and typical_zone_load
        else None
    )
    observed_cost_ratios = [
        value for value in (distance_ratio, duration_ratio, zone_load_ratio) if value is not None
    ]
    relative_cost = max(observed_cost_ratios) if observed_cost_ratios else None
    acute_capacity_ratio = state.recent_load.acute_distance_to_capacity_ratio
    accumulated_load_manageable = (
        acute_capacity_ratio is None or acute_capacity_ratio < high_load_threshold
    )
    high_rpe = latest.perceived_exertion is not None and latest.perceived_exertion >= 8
    meaningful_hard_work = latest.zone_breakdown.hard_minutes >= 3
    taxing_type = latest.is_long_run or latest.is_quality_session
    low_cost = bool(
        relative_cost is not None
        and relative_cost <= 0.60
        and not taxing_type
        and not high_rpe
        and not meaningful_hard_work
        and accumulated_load_manageable
    )
    return low_cost, {
        "latest_available": True,
        "typical_history_available": True,
        "distance_ratio": distance_ratio,
        "duration_ratio": duration_ratio,
        "zone_load_ratio": zone_load_ratio,
        "relative_cost": relative_cost,
        "latest_is_long_or_quality": taxing_type,
        "latest_rpe": latest.perceived_exertion,
        "latest_hard_minutes": latest.zone_breakdown.hard_minutes,
        "acute_capacity_ratio": acute_capacity_ratio,
        "high_load_threshold": high_load_threshold,
        "accumulated_load_manageable": accumulated_load_manageable,
    }


def _midpoint(result: RecommendationResponse) -> float:
    if result.distance_range_miles:
        return sum(result.distance_range_miles) / 2
    if result.duration_range_minutes:
        pace = 11.0
        return sum(result.duration_range_minutes) / 2 / pace
    return 0.0


def _project_window(window: LoadWindow, additions: list[RecommendationResponse], as_of: datetime) -> LoadWindow:
    recent = [
        result for result in additions
        if result.planned_for and 0 <= (as_of - result.planned_for).total_seconds() <= window.days * 86400
    ]
    miles = sum(_midpoint(result) for result in recent)
    moving = miles * 11.0
    base_load_per_mile = (window.zone_load or 0) / window.distance_miles if window.distance_miles else 18.0
    added_load = sum(
        _midpoint(result)
        * base_load_per_mile
        * (1.35 if result.workout_type in {WorkoutType.INTERVALS, WorkoutType.TEMPO_THRESHOLD, WorkoutType.RACE} else 1.05 if result.workout_type == WorkoutType.LONG else 1.0)
        for result in recent
    )
    return window.model_copy(
        update={
            "distance_miles": window.distance_miles + miles,
            "moving_minutes": window.moving_minutes + moving,
            "zone_load": (window.zone_load + added_load) if window.zone_load is not None else None,
            "activity_count": window.activity_count + len(recent),
        }
    )


def _project_state(state: FitnessState, planned: list[RecommendationResponse]) -> FitnessState:
    if not planned:
        return state
    prior_runs = [item for item in planned if item.planned_for and item.planned_for < state.as_of and item.workout_type != WorkoutType.REST]
    if not prior_runs:
        return state
    last = prior_runs[-1]
    quality = [item for item in prior_runs if item.workout_type in {WorkoutType.INTERVALS, WorkoutType.TEMPO_THRESHOLD, WorkoutType.RACE}]
    long_runs = [item for item in prior_runs if item.workout_type == WorkoutType.LONG]
    windows = {
        days: _project_window(getattr(state.recent_load, f"trailing_{days}d"), prior_runs, state.as_of)
        for days in (7, 14, 28)
    }
    base_7 = state.recent_load.trailing_7d.zone_load
    projected_7 = windows[7].zone_load
    ratio = state.recent_load.acute_to_prior_ratio
    if ratio is not None and base_7 and projected_7 is not None:
        ratio *= projected_7 / base_7
    capacity_reference = state.recent_load.capacity_reference_miles
    distance_ratio = (
        windows[7].distance_miles / capacity_reference
        if capacity_reference
        else state.recent_load.acute_distance_to_capacity_ratio
    )
    distance = _midpoint(last)
    moving = distance * 11.0
    last_quality = last.workout_type in {WorkoutType.INTERVALS, WorkoutType.TEMPO_THRESHOLD, WorkoutType.RACE}
    last_long = last.workout_type == WorkoutType.LONG
    difficulty = SessionDifficulty(
        distance_miles=distance,
        moving_minutes=moving,
        elapsed_minutes=moving,
        stopped_minutes=0,
        zone_load=None,
        zone_breakdown=ZoneBreakdown(),
        is_long_run=last_long,
        is_quality_session=last_quality,
        difficulty_flags=["planned_session_projection"],
    )
    return state.model_copy(
        update={
            "recent_load": LoadContext(
                trailing_7d=windows[7],
                trailing_14d=windows[14],
                trailing_28d=windows[28],
                acute_to_prior_ratio=ratio,
                acute_distance_to_capacity_ratio=distance_ratio,
                prior_28d_weekly_miles=state.recent_load.prior_28d_weekly_miles,
                sustained_capacity_miles=state.recent_load.sustained_capacity_miles,
                capacity_reference_miles=capacity_reference,
                confidence=state.recent_load.confidence,
                flags=[*state.recent_load.flags, "includes_planned_sessions"],
            ),
            "days_since_last_run": max(0.0, (state.as_of - last.planned_for).total_seconds() / 86400),
            "days_since_quality_run": (
                max(0.0, (state.as_of - quality[-1].planned_for).total_seconds() / 86400)
                if quality else state.days_since_quality_run
            ),
            "days_since_long_run": (
                max(0.0, (state.as_of - long_runs[-1].planned_for).total_seconds() / 86400)
                if long_runs else state.days_since_long_run
            ),
            "last_run": difficulty,
            "last_run_workout_type": last.workout_type,
            "longest_run_30d_miles": max(state.longest_run_30d_miles, *( _midpoint(item) for item in long_runs), 0),
            "quality_sessions_14d": state.quality_sessions_14d + len(quality),
            "completed_quality_session_count": state.completed_quality_session_count + len(quality),
            "running_days_28d": state.running_days_28d + len(prior_runs),
            "normal_runs_since_health_event": state.normal_runs_since_health_event + len(prior_runs),
        }
    )


def build_weekly_schedule(
    daily_states: list[FitnessState],
    request: RecommendationRequest,
    config: dict,
    target_run_count: int | None = None,
    target_distance_range: tuple[float, float] | None = None,
    target_evidence: WeeklyTargetEvidence | None = None,
    completed_activities_by_offset: dict[int, list[TrailingDayActivity]] | None = None,
) -> WeeklyScheduleResponse:
    """Generate a coordinated seven-day schedule from prebuilt daily states."""
    if len(daily_states) != 7:
        raise ValueError("Weekly planning requires exactly seven daily states")
    completed_activities_by_offset = completed_activities_by_offset or {}
    completed_run_count = sum(bool(items) for items in completed_activities_by_offset.values())
    remaining_target = (
        max(0, int(target_run_count) - completed_run_count)
        if target_run_count is not None
        else None
    )
    offset_state = (
        daily_states[0].model_copy(update={"days_since_last_run": 0.0})
        if completed_activities_by_offset.get(0)
        else daily_states[0]
    )
    desired_offsets = (
        automatic_run_day_offsets(
            offset_state, request.health_status, config, remaining_target
        )
        if remaining_target is None or remaining_target > 0
        else []
    )
    offsets = [offset for offset in desired_offsets if not completed_activities_by_offset.get(offset)]
    # Coordinate the buildup, but leave the final target session unassigned.
    # Its workout type must win the ordinary evidence score; being the last
    # visible day (or the end of the target cadence) is not long-run evidence.
    role_preferences = {offset: "easy" for offset in offsets[:-1]}
    if len(offsets) >= 3:
        role_preferences[offsets[1]] = "quality"
    planned: list[RecommendationResponse] = []
    days: list[WeeklyScheduleDay] = []
    stable_easy_distance = typical_easy_distance(daily_states[0])
    low_cost_consecutive, consecutive_facts = consecutive_day_evidence(
        daily_states[0], config
    )
    for offset, raw_state in enumerate(daily_states):
        state = _project_state(raw_state, planned)
        completed = completed_activities_by_offset.get(offset, [])
        if completed:
            workout_types = {item.workout_type for item in completed}
            role = (
                "completed_quality_run"
                if workout_types & {WorkoutType.INTERVALS, WorkoutType.TEMPO_THRESHOLD, WorkoutType.RACE}
                else "completed_long_run"
                if WorkoutType.LONG in workout_types
                else "completed_recovery_run"
                if WorkoutType.RECOVERY in workout_types
                else "completed_run"
            )
            days.append(
                WeeklyScheduleDay(
                    date=raw_state.as_of.date(),
                    planned_at=None,
                    recommendation=None,
                    day_role=role,
                    rationale="Recorded activity completed; no second workout is prescribed for this day.",
                    completed_activities=completed,
                )
            )
            continue
        if offset not in offsets:
            days.append(
                WeeklyScheduleDay(
                    date=raw_state.as_of.date(),
                    planned_at=None,
                    recommendation=None,
                    day_role="rest_day",
                    rationale="Planned non-running day between the week's selected training opportunities.",
                )
            )
            continue
        result = recommend_next_run(
            state,
            request,
            config,
            weekly_role=role_preferences.get(offset),
        )
        if (
            result.workout_type == WorkoutType.EASY
            and result.readiness.value == "ready"
        ):
            result = result.model_copy(
                update={"distance_range_miles": stable_easy_distance}
            )
        if result.workout_type != WorkoutType.REST:
            planned.append(result)
        role = {
            WorkoutType.RECOVERY: "recovery_run",
            WorkoutType.EASY: "easy_run",
            WorkoutType.LONG: "long_run",
            WorkoutType.INTERVALS: "quality_run",
            WorkoutType.TEMPO_THRESHOLD: "quality_run",
            WorkoutType.RACE: "quality_run",
            WorkoutType.REST: "guardrail_rest_day",
        }.get(result.workout_type, "scheduled_run")
        days.append(
            WeeklyScheduleDay(
                date=raw_state.as_of.date(),
                planned_at=raw_state.as_of,
                recommendation=result,
                day_role=role,
                rationale=(
                    (
                        "Consecutive day selected intentionally because the latest run was "
                        f"{float(consecutive_facts.get('relative_cost') or 0) * 100:.0f}% of a typical recent session at its highest observed cost ratio, "
                        "was neither long nor quality, and accumulated distance remained below the high-load guardrail."
                    )
                    if offset == 0 and low_cost_consecutive
                    else
                    "Consecutive run day selected intentionally and evaluated with the prior planned session included."
                    if planned and len(planned) >= 2 and planned[-2].planned_for and (raw_state.as_of.date() - planned[-2].planned_for.date()).days == 1
                    else "Automatically selected from recent frequency, projected load, workout recency, and recovery spacing."
                ),
            )
        )
    run_results = [day.recommendation for day in days if day.recommendation and day.recommendation.workout_type != WorkoutType.REST]
    deferred_count = sum(offset >= len(daily_states) for offset in offsets)
    distance_low = sum((item.distance_range_miles or (0.0, 0.0))[0] for item in run_results)
    distance_high = sum((item.distance_range_miles or (0.0, 0.0))[1] for item in run_results)
    if target_distance_range is None:
        capacity = max(
            daily_states[0].recent_load.trailing_28d.distance_miles / 4.0,
            daily_states[0].recent_load.trailing_7d.distance_miles,
        )
        target_distance_range = (_half_mile(capacity * 0.95), _half_mile(capacity * 1.10))
    if target_evidence is None:
        target_evidence = WeeklyTargetEvidence(
            recent_7d_miles=daily_states[0].recent_load.trailing_7d.distance_miles,
            chronic_42d_weekly_miles=daily_states[0].recent_load.trailing_28d.distance_miles / 4.0,
            best_sustained_28d_weekly_miles=daily_states[0].recent_load.trailing_28d.distance_miles / 4.0,
            peak_7d_miles=daily_states[0].recent_load.trailing_7d.distance_miles,
            demonstrated_run_days_per_week=daily_states[0].running_days_28d / 4.0,
            capacity_reference_miles=sum(target_distance_range) / 2,
            rationale="Fallback target derived from the supplied fitness state.",
        )
    guardrail_rest_count = sum(
        bool(day.recommendation and day.recommendation.workout_type == WorkoutType.REST)
        for day in days
    )
    if target_evidence.capacity_reference_miles <= 0:
        horizon_explanation = (
            "No training history is available, so this is a conservative starter placeholder rather than a personalized capacity estimate. "
        )
    elif deferred_count:
        horizon_explanation = (
            f"{deferred_count} additional session from the sustained-frequency reference falls after {daily_states[-1].as_of.date().isoformat()} because recovery spacing takes precedence; it is not included in the planned mileage. "
        )
    elif guardrail_rest_count:
        horizon_explanation = (
            f"{guardrail_rest_count} selected training opportunit{'y was' if guardrail_rest_count == 1 else 'ies were'} converted to rest after load, health, recovery, or weather guardrails; this is why the visible plan is below the capacity reference. "
        )
    elif distance_high < target_distance_range[0]:
        horizon_explanation = (
            "All selected run days fit inside the horizon, but session-level guardrails keep planned mileage below the capacity reference. "
        )
    elif distance_low > target_distance_range[1]:
        horizon_explanation = (
            "The planned mileage is above the capacity reference; inspect the individual rule traces before treating the schedule as appropriate. "
        )
    else:
        horizon_explanation = (
            "The visible planned-mileage range overlaps the demonstrated-capacity reference. "
        )
    return WeeklyScheduleResponse(
        generated_at=datetime.now(daily_states[0].as_of.tzinfo),
        start_date=daily_states[0].as_of.date(),
        end_date=daily_states[-1].as_of.date(),
        target_run_count=len(desired_offsets) + completed_run_count,
        target_distance_range_miles=target_distance_range,
        target_evidence=target_evidence,
        completed_run_count=completed_run_count,
        run_count=len(run_results),
        projected_distance_range_miles=(round(distance_low, 1), round(distance_high, 1)),
        summary=(
            f"The visible horizon contains {completed_run_count} completed and {len(run_results)} planned runs totaling {distance_low:.1f}–{distance_high:.1f} miles. "
            + horizon_explanation
            + (
                f"The {target_distance_range[0]:g}–{target_distance_range[1]:g} mile range is a demonstrated-capacity reference, not a requirement to compress mileage into these dates. "
            )
            + "Quality is considered in the scoring but remains subject to load, health, recovery, and weather guardrails."
        ),
        days=days,
    )
