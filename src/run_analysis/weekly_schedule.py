"""Continuous rolling-lookahead schedule construction from daily fitness states.

The planner chooses run days automatically, then evaluates the existing
inspectable recommendation rules on those days. Planned sessions are projected
into later daily states so consecutive days, load, and workout recency are
intentional rather than independent recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import combinations
from math import ceil, comb, exp, floor
from zoneinfo import ZoneInfo

from .environmental_stress import assess_training_weather
from .recommendation import (
    effective_load_ratio,
    long_run_reference_miles,
    recommend_next_run,
    scale_quality_session,
    single_session_progression_reference_miles,
    structure_extended_quality_session,
    typical_easy_distance,
)
from .race_goals import (
    build_weeks_before_taper,
    configured_race_goal,
    required_compound_progression,
)
from .recovery import (
    EASY_RUN_RESIDUAL_LIMIT,
    EASY_VOLUME_RESIDUAL_FLOOR,
    LONG_RUN_LOAD_FACTOR,
    QUALITY_SESSION_LOAD_FACTOR,
    RECOVERY_RUN_LOAD_FACTOR,
    RECOVERY_HALF_LIFE_HOURS,
    TAXING_RUN_RESIDUAL_LIMIT,
    athlete_relative_session_load,
    decay_recovery_load,
    estimate_recovery,
)
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


VISIBLE_HORIZON_DAYS = 7
PLANNING_HORIZON_DAYS = 21
WEEKLY_PLANNER_VERSION = 49
MAX_ADAPTIVE_CANDIDATES = 64
MAX_HORIZON_COUNT_OPTIONS = 14
MINIMUM_QUALITY_RANGE_MILES = (3.0, 3.5)
MAXIMUM_ORDINARY_QUALITY_MILES = 5.0


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
        seven_run_days = sum(1 for date in daily if seven_start <= date <= day)
        windows.append((day, seven, twenty_eight, run_days, seven_run_days))
        day += timedelta(days=1)
    peak_seven = max(item[1] for item in windows)
    best_sustained = max(windows, key=lambda item: (item[2], item[0]))
    recent_seven = windows[-1][1]
    half_life = float(config.get("coaching", {}).get("capacity_retention_half_life_days", 84))
    retention_grace = int(config.get("coaching", {}).get("capacity_retention_grace_days", 28))

    def retained(value: float, window_end) -> float:
        age_days = max(0, (end - window_end).days)
        decay_days = max(0, age_days - retention_grace)
        return value * 0.5 ** (decay_days / half_life)

    # Decay every demonstrated window from its own date before choosing the
    # strongest remaining evidence. Decaying only the all-time raw maximum can
    # ignore a slightly lower but much newer training block.
    retained_sustained, retained_sustained_window = max(
        (retained(item[2], item[0]), item) for item in windows
    )
    # A recent high-mileage week remains evidence after it exits the trailing
    # seven days, but is capped by demonstrated sustained capacity so one
    # isolated spike cannot establish an unlimited weekly target.
    retained_recent_peak = max(
        retained(min(item[1], best_sustained[2]), item[0])
        for item in windows
    )
    capacity_reference = max(
        chronic_weekly,
        retained_sustained,
        retained_recent_peak,
        min(recent_seven, best_sustained[2]),
    )
    general_progression = min(
        0.10,
        max(
            0.0,
            float(
                config.get("coaching", {}).get(
                    "general_fitness_progression_fraction", 0.08
                )
            ),
        ),
    )
    goal = configured_race_goal(config, on_date=end)
    if goal:
        # Goal mode starts from demonstrated training and is subsequently
        # raised only as much as the backward-planned race trajectory needs.
        target_low = max(6.0, _half_mile(capacity_reference * 0.95))
        target_high = max(
            target_low + 0.5,
            _half_mile(min(peak_seven, capacity_reference * 1.10)),
        )
        goal_trajectory_detail = ""
    else:
        # General fitness is a productive mode, not a request to preserve an
        # old mileage number indefinitely. Put the center of the planning
        # range a modest percentage above demonstrated capacity. The adaptive
        # scheduler still chooses lower in this range when opening load is
        # high and can underfill it when recovery does not support the work.
        # Because projected mileage only becomes capacity after completion,
        # missed or costly weeks do not create automatic compounding.
        productive_midpoint = capacity_reference * (1.0 + general_progression)
        target_low = max(6.0, _half_mile(productive_midpoint * 0.97))
        target_high = max(
            target_low + 0.5,
            _half_mile(productive_midpoint * 1.03),
        )
        goal_trajectory_detail = (
            " In general-fitness mode, successful training earns a modest "
            "percentage-based build above demonstrated capacity; recovery and "
            "actual completion determine whether that target becomes the next baseline."
        )
    if goal:
        goal_profile, race_date, _ = goal
        build_weeks = build_weeks_before_taper(goal_profile, race_date, end)
        if build_weeks > 0 and capacity_reference > 0:
            required_weekly_rate = required_compound_progression(
                capacity_reference,
                goal_profile.peak_weekly_miles,
                build_weeks,
            )
            # Goal mode can use the required trajectory up to an explicit
            # eight-percent weekly planning ceiling. It does not silently
            # promise an infeasible build or override later recovery checks.
            # A roomy race calendar must not make training less productive
            # than general-fitness mode. A demanding calendar can raise the
            # rate, up to the explicit goal-mode ceiling.
            planned_weekly_rate = max(
                general_progression,
                min(0.08, required_weekly_rate),
            )
            goal_midpoint = min(
                goal_profile.peak_weekly_miles,
                capacity_reference * (1.0 + planned_weekly_rate),
            )
            if goal_midpoint > sum((target_low, target_high)) / 2:
                target_low = max(target_low, _half_mile(goal_midpoint * 0.97))
                target_high = max(
                    target_low + 0.5,
                    _half_mile(goal_midpoint * 1.03),
                )
                goal_trajectory_detail = (
                    f" {goal_profile.label} preparation requires about "
                    f"{goal_profile.peak_weekly_miles:.0f} peak weekly miles before taper; "
                    f"this week advances along that backward-planned trajectory."
                )

    # Planned frequency must be earned over a sustained window. Using the
    # trailing seven-day count directly creates a self-reinforcing ratchet: one
    # incidental short run can raise next week's target, and completing the
    # extra prescribed day then appears to validate the higher frequency. A
    # 28-day rate still adapts to a genuine change in training pattern without
    # letting one unusually busy week rewrite the schedule.
    recent_28d_run_days_per_week = (
        sum(1 for date in daily if end - timedelta(days=27) <= date <= end) / 4.0
    )
    # Frequency needs the same time-aware retention as mileage.  Keeping the
    # historical maximum forever can pair an old four-day routine with a much
    # smaller current mileage target, producing a week of implausibly short
    # runs.  Blend from the current routine toward the historical routine by
    # the retained fraction: short interruptions barely change frequency,
    # while longer breaks ease it down exponentially instead of resetting it.
    retained_window_age = max(
        0, (end - retained_sustained_window[0]).days
    )
    retained_window_decay_days = max(
        0, retained_window_age - retention_grace
    )
    retention = 0.5 ** (
        retained_window_decay_days / max(1.0, half_life)
    )
    recent_frequency = float(recent_28d_run_days_per_week)
    historical_frequency = float(retained_sustained_window[3])
    sustained_frequency = recent_frequency + max(
        0.0, historical_frequency - recent_frequency
    ) * retention
    retained_recent_frequency = max(
        retained(
            min(float(item[4]), float(best_sustained[3])),
            item[0],
        )
        for item in windows
    )
    demonstrated = max(sustained_frequency, retained_recent_frequency)
    # Capacity and prescription answer different questions. Retained history
    # keeps a short interruption from erasing mileage durability, but current
    # 28-day cadence determines how many days should carry that mileage now.
    # At most three quarters of a day per week is borrowed from retained
    # frequency, preventing an old four-day routine from forcing four current
    # runs (and an avoidable consecutive pair) after cadence has fallen.
    prescribed_frequency = min(
        demonstrated,
        recent_frequency + min(0.75, max(0.0, demonstrated - recent_frequency)),
    )
    target_runs = max(2, min(7, floor(prescribed_frequency + 0.5)))
    evidence = WeeklyTargetEvidence(
        recent_7d_miles=recent_seven,
        chronic_42d_weekly_miles=chronic_weekly,
        best_sustained_28d_weekly_miles=best_sustained[2],
        peak_7d_miles=peak_seven,
        current_run_days_per_week=recent_frequency,
        demonstrated_run_days_per_week=demonstrated,
        capacity_reference_miles=capacity_reference,
        rationale=(
            "Mileage uses current plus recency-decayed capacity evidence. Recent and demonstrated cadence remain descriptive "
            "evidence, while the continuous load/recovery optimizer chooses how many useful run days carry that mileage. "
            "A short break preserves useful capacity without forcing an outdated frequency."
            f"{goal_trajectory_detail}"
        ),
    )
    return target_runs, (target_low, target_high), evidence


def automatic_run_day_offsets(
    state: FitnessState,
    health: CurrentHealthStatus,
    config: dict | None = None,
    target_run_count: int | None = None,
    *,
    horizon_days: int = VISIBLE_HORIZON_DAYS,
) -> list[int]:
    """Choose run days from the explicit target, not a disrupted recent period."""
    if horizon_days < 1:
        raise ValueError("Planning horizon must contain at least one day")
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
    recovery_too_short_for_today = False
    if typical_rest_days >= 1 and state.days_since_last_run is not None:
        inferred_last_date = (state.as_of - timedelta(days=state.days_since_last_run)).date()
        ran_today = inferred_last_date >= state.as_of.date()
        load_ratio = effective_load_ratio(state.recent_load)
        high_load = (
            load_ratio is not None
            and load_ratio
            >= float((config or {}).get("coaching", {}).get("high_load_ratio", 1.30))
        )
        recovery = estimate_recovery(state)
        ran_yesterday_with_material_recovery_load = (
            inferred_last_date == state.as_of.date() - timedelta(days=1)
            and (
                (recovery is not None and recovery.hours_until_easy > 0)
                or high_load
            )
        )
        recovery_too_short_for_today = (
            ran_today or ran_yesterday_with_material_recovery_load
        )
    low_cost_consecutive, _ = consecutive_day_evidence(state, config)
    low_cost_consecutive = low_cost_consecutive and health == CurrentHealthStatus.NORMAL
    delay_for_recovery = recovery_too_short_for_today and not low_cost_consecutive
    if horizon_days == VISIBLE_HORIZON_DAYS:
        return recent_patterns[target] if delay_for_recovery else ordinary_patterns[target]

    # Spread the earned weekly frequency over the entire rolling horizon.
    # This avoids resetting the cadence at an arbitrary Sunday boundary: at
    # three days per week, for example, six runs land near 0/2/5/7/9/12
    # instead of 0/2/4 followed by two empty end-of-week days.
    first_offset = 1 if delay_for_recovery else 0
    available_days = max(0, horizon_days - first_offset)
    total_runs = min(
        available_days,
        max(1, floor(target * horizon_days / VISIBLE_HORIZON_DAYS + 0.5)),
    )
    if total_runs <= 0:
        return []
    spacing = available_days / total_runs
    return [
        first_offset + floor(index * spacing + 0.5)
        for index in range(total_runs)
    ]


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


def _ordinary_easy_expansion_reference(state: FitnessState) -> float:
    """Mark where another run day should compete with more easy mileage.

    An ordinary easy run can be somewhat longer than the athlete's standalone
    template. Beyond this athlete-relative reference, expansion becomes
    progressively more expensive so another useful easy day usually wins.
    This is deliberately not a cap: constrained schedules can still allocate
    a longer aerobic run when that is the strongest overall plan.
    """

    easy_low, easy_high = typical_easy_distance(state)
    easy_midpoint = (easy_low + easy_high) / 2
    return max(
        easy_high,
        floor(easy_midpoint * 1.25 * 2 + 1e-9) / 2,
    )


def _session_load_units(
    distance_miles: float,
    easy_reference_miles: float,
    *,
    long_run: bool = False,
    quality_session: bool = False,
    recovery_run: bool = False,
) -> float:
    """Express one session relative to the athlete's ordinary easy run."""
    intensity_factor = (
        QUALITY_SESSION_LOAD_FACTOR
        if quality_session
        else LONG_RUN_LOAD_FACTOR
        if long_run
        else RECOVERY_RUN_LOAD_FACTOR
        if recovery_run
        else 1.0
    )
    return distance_miles / max(1.0, easy_reference_miles) * intensity_factor


def _recommendation_load_units(
    result: RecommendationResponse,
    easy_reference_miles: float,
) -> float:
    return _session_load_units(
        _midpoint(result),
        easy_reference_miles,
        long_run=result.workout_type == WorkoutType.LONG,
        quality_session=result.workout_type
        in {WorkoutType.INTERVALS, WorkoutType.TEMPO_THRESHOLD, WorkoutType.RACE},
        recovery_run=result.workout_type == WorkoutType.RECOVERY,
    )


def _recovery_spacing_cost(
    state: FitnessState,
    result: RecommendationResponse | None = None,
) -> float:
    """Price candidate timing from the same athlete-relative recovery signal."""
    recovery = estimate_recovery(state)
    if recovery is None:
        return 0.0
    proposed_load = (
        _recommendation_load_units(
            result,
            sum(typical_easy_distance(state)) / 2,
        )
        if result is not None
        else 1.0
    )
    taxing_next = bool(
        result
        and result.workout_type
        in {
            WorkoutType.LONG,
            WorkoutType.INTERVALS,
            WorkoutType.TEMPO_THRESHOLD,
            WorkoutType.RACE,
        }
    )
    readiness_limit = (
        TAXING_RUN_RESIDUAL_LIMIT
        if taxing_next
        else EASY_RUN_RESIDUAL_LIMIT
    )
    # Extra recovery is useful only while a candidate is outside the relevant
    # readiness range. Once two slots are both ready, residual numerical load
    # must not manufacture a permanent preference for the latest time of day;
    # weather order (or the ordinary earliest slot without weather) wins.
    excess = max(0.0, recovery.residual_load - readiness_limit)
    interaction = excess * proposed_load * 8.0
    if taxing_next:
        interaction += excess * 4.0
    return interaction


def _decayed_recovery_load(
    raw_state: FitnessState,
    planned: list[RecommendationResponse],
    as_of: datetime,
    easy_reference_miles: float,
    config: dict,
) -> float:
    """Project transient session load, allowing it to decay during rest."""
    half_life_hours = RECOVERY_HALF_LIFE_HOURS
    residual = 0.0
    if raw_state.last_run and raw_state.days_since_last_run is not None:
        completed_units, _ = athlete_relative_session_load(
            raw_state.last_run,
            raw_state.recent_load.trailing_28d,
            performance_anomaly=raw_state.recent_performance_anomaly,
            drift_percent=raw_state.last_run_drift_percent,
        )
        residual += decay_recovery_load(
            completed_units,
            raw_state.days_since_last_run * 24,
            half_life_hours=half_life_hours,
        )
    for item in planned:
        if (
            item.workout_type == WorkoutType.REST
            or item.planned_for is None
            or item.planned_for >= as_of
        ):
            continue
        elapsed_hours = (as_of - item.planned_for).total_seconds() / 3600
        residual += decay_recovery_load(
            _recommendation_load_units(item, easy_reference_miles),
            elapsed_hours,
            half_life_hours=half_life_hours,
        )
    return residual


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
    moderate_known_minutes = state.recent_load.trailing_14d.moving_minutes
    projected_easy_minutes = sum(
        _midpoint(item) * 11.0
        for item in prior_runs
        if item.workout_type
        not in {WorkoutType.INTERVALS, WorkoutType.TEMPO_THRESHOLD, WorkoutType.RACE}
    )
    projected_moderate_fraction = state.moderate_fraction_14d
    if (
        projected_moderate_fraction is not None
        and moderate_known_minutes > 0
        and projected_easy_minutes > 0
    ):
        # Planned Z1/Z2 running adds known easy time to the same rolling
        # denominator used by the leakage guardrail. Without this projection,
        # one marginally high historical fraction is copied unchanged onto
        # every future workout and can make the entire plan identical.
        projected_moderate_fraction = (
            projected_moderate_fraction
            * moderate_known_minutes
            / (moderate_known_minutes + projected_easy_minutes)
        )
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
            # These measurements belong to the latest completed run.  Once a
            # planned session intervenes, its response is unknown; repeating
            # the old completed-run caution across every later workout makes
            # the whole prospective week identical.  Later sessions are
            # therefore conditional on the intervening workout going as
            # prescribed, rather than assumed to inherit stale evidence.
            "last_run_drift_percent": None,
            "recent_performance_anomaly": "unknown",
            "longest_run_30d_miles": max(
                state.longest_run_30d_miles,
                *(_midpoint(item) for item in prior_runs),
                0,
            ),
            "retained_long_run_capacity_miles": max(
                state.retained_long_run_capacity_miles,
                *(_midpoint(item) for item in prior_runs),
                0,
            ),
            "quality_sessions_14d": state.quality_sessions_14d + len(quality),
            "completed_quality_session_count": state.completed_quality_session_count + len(quality),
            "running_days_28d": state.running_days_28d + len(prior_runs),
            "moderate_fraction_14d": projected_moderate_fraction,
            "normal_runs_since_health_event": state.normal_runs_since_health_event + len(prior_runs),
        }
    )


def _elapsed_workout_role(
    state_options: list[FitnessState],
    planned: list[RecommendationResponse],
    config: dict,
    *,
    taper_horizon: bool,
) -> str:
    """Prefer workout composition from elapsed recency, never block position."""

    if not state_options:
        return "easy"
    projected = _project_state(state_options[0], planned)
    settings = config.get("coaching", {})
    long_reference = float(
        settings.get("long_run_recency_reference_days", 7)
    )
    quality_reference = float(
        settings.get("quality_recency_reference_days", 7)
    )
    long_ratio = (
        projected.days_since_long_run / max(1.0, long_reference)
        if projected.days_since_long_run is not None
        else 1.0
    )
    quality_ratio = (
        projected.days_since_quality_run / max(1.0, quality_reference)
        if projected.days_since_quality_run is not None
        else 1.0
    )
    if long_ratio >= 1.0 and (
        taper_horizon or long_ratio >= quality_ratio
    ):
        return "long"
    if not taper_horizon and quality_ratio >= 1.0:
        return "quality"
    return "easy"


def _return_to_retained_capacity_due(
    state: FitnessState,
    config: dict,
) -> bool:
    reference = single_session_progression_reference_miles(state)
    if reference <= 0 or state.days_since_long_run is None:
        return False
    ordinary_fraction = float(
        config.get("coaching", {}).get(
            "long_run_target_progression_fraction", 0.05
        )
    )
    retained_gap = state.retained_long_run_capacity_miles / reference - 1.0
    return state.days_since_long_run >= 7.0 and retained_gap >= ordinary_fraction


def _select_timed_recommendation(
    state_options: list[FitnessState],
    planned: list[RecommendationResponse],
    request: RecommendationRequest,
    config: dict,
    *,
    weekly_role: str | None,
    allowed_candidates: set[str] | None = None,
) -> tuple[FitnessState, RecommendationResponse]:
    """Choose weather-ranked timing unless a later slot clears a guardrail.

    ``state_options`` arrive in weather-preference order. Recovery and
    readiness outcomes take precedence over that order, which lets an exact
    timestamp accumulate meaningfully more recovery without treating a named
    hour as a cadence target.
    """
    choices: list[
        tuple[tuple[int, int, float, int], FitnessState, RecommendationResponse]
    ] = []
    for weather_order, raw_state in enumerate(state_options):
        state = _project_state(raw_state, planned)
        result = recommend_next_run(
            state,
            request,
            config,
            weekly_role=weekly_role,
            allowed_candidates=allowed_candidates,
        )
        guardrail_rank = (
            2
            if result.workout_type == WorkoutType.REST
            else 1
            if (
                result.workout_type == WorkoutType.RECOVERY
                and request.health_status == CurrentHealthStatus.NORMAL
            )
            else 0
        )
        readiness_rank = {
            "ready": 0,
            "caution": 1,
            "not_ready": 2,
        }[result.readiness.value]
        # A date is not represented faithfully by its earliest viable hour.
        # When two slots have the same guardrail/readiness outcome, let the
        # continuous recovery and weather costs choose between them. This is
        # especially important after an evening long run: Sunday evening may
        # be a sound aerobic slot even when Sunday morning is unnecessarily
        # close, and the whole-day optimizer should see that better option.
        timing_cost = _recovery_spacing_cost(state, result) + assess_training_weather(
            state.planned_weather, state.weather_exposure_baseline
        ).score * 3.0
        choices.append(
            (
                (guardrail_rank, readiness_rank, timing_cost, weather_order),
                state,
                result,
            )
        )
    _, state, result = min(choices, key=lambda item: item[0])
    return state, result


def _select_budgeted_timed_recommendation(
    state_options: list[FitnessState],
    planned: list[RecommendationResponse],
    request: RecommendationRequest,
    config: dict,
    *,
    weekly_role: str | None,
    projected_week_miles: float,
    remaining_week_runs: int,
    target_week_high: float,
    reserve_easy_miles: float,
) -> tuple[FitnessState, RecommendationResponse]:
    """Choose the strongest workout that still leaves room for the week."""
    return_long_due = False
    if state_options:
        role_state = _project_state(state_options[0], planned)
        return_long_due = _return_to_retained_capacity_due(role_state, config)
        if return_long_due:
            weekly_role = "long"
    allowed = {"easy", "long", "quality"}
    taxing_spacing_substitution = False
    while True:
        state, result = _select_timed_recommendation(
            state_options,
            planned,
            request,
            config,
            weekly_role=weekly_role,
            allowed_candidates=allowed,
        )
        projected_total = (
            projected_week_miles
            + _midpoint(result)
            + remaining_week_runs * reserve_easy_miles
        )
        projected_with_short_easy_runs = projected_total - (
            remaining_week_runs * max(0.0, reserve_easy_miles - 2.0)
        )
        candidate = (
            "long"
            if result.workout_type == WorkoutType.LONG
            else "quality"
            if result.workout_type
            in {WorkoutType.INTERVALS, WorkoutType.TEMPO_THRESHOLD, WorkoutType.RACE}
            else "easy"
        )
        latest_taxing = next(
            (
                item
                for item in reversed(planned)
                if item.planned_for
                and item.workout_type
                in {
                    WorkoutType.LONG,
                    WorkoutType.INTERVALS,
                    WorkoutType.TEMPO_THRESHOLD,
                    WorkoutType.RACE,
                }
            ),
            None,
        )
        taxing_residual = 0.0
        consecutive_taxing_dates = False
        if result.planned_for and latest_taxing and latest_taxing.planned_for:
            easy_reference = sum(typical_easy_distance(state)) / 2
            consecutive_taxing_dates = (
                result.planned_for.date() - latest_taxing.planned_for.date()
            ).days <= 1
            taxing_residual = decay_recovery_load(
                _recommendation_load_units(latest_taxing, easy_reference),
                (result.planned_for - latest_taxing.planned_for).total_seconds()
                / 3600,
            )
        elif (recovery := estimate_recovery(state)) is not None:
            taxing_residual = recovery.residual_load
            if (
                state.last_run_workout_type
                in {
                    WorkoutType.LONG,
                    WorkoutType.INTERVALS,
                    WorkoutType.TEMPO_THRESHOLD,
                    WorkoutType.RACE,
                }
                and state.days_since_last_run is not None
            ):
                inferred_last_date = (
                    state.as_of - timedelta(days=state.days_since_last_run)
                ).date()
                consecutive_taxing_dates = (
                    state.as_of.date() - inferred_last_date
                ).days <= 1
        taxing_recovery_incomplete = bool(
            candidate in {"long", "quality"}
            and result.workout_type != WorkoutType.RACE
            and (
                taxing_residual > TAXING_RUN_RESIDUAL_LIMIT
                or consecutive_taxing_dates
            )
        )
        if taxing_recovery_incomplete:
            # Preserve the run date when it is otherwise a good fit, but make
            # the session aerobic. This applies equally to a taxing activity
            # completed before the new plan and one already placed inside it;
            # intensity spacing is a composition constraint, not a reason to
            # manufacture another rest day and later compress run frequency.
            allowed.discard(candidate)
            taxing_spacing_substitution = True
            continue
        projected_with_scaled_quality = projected_with_short_easy_runs
        if candidate == "quality":
            projected_with_scaled_quality -= max(
                0.0,
                _midpoint(result) - sum(MINIMUM_QUALITY_RANGE_MILES) / 2,
            )
        if (
            candidate == "easy"
            or projected_total <= target_week_high
            or (
                candidate == "long"
                and projected_with_short_easy_runs <= target_week_high
            )
            or (
                candidate == "quality"
                and projected_with_scaled_quality <= target_week_high
            )
            or (candidate == "long" and return_long_due)
        ):
            if taxing_spacing_substitution:
                spacing_reason = (
                    "A second taxing workout was replaced with aerobic running because long and quality sessions need an intervening non-taxing day."
                    if consecutive_taxing_dates
                    else
                    "A second taxing workout was replaced with aerobic running because athlete-relative recovery load was still above the taxing-session reference."
                )
                result = result.model_copy(
                    update={
                        "reasons": [
                            *result.reasons,
                            spacing_reason,
                        ]
                    }
                )
            return state, result
        allowed.discard(candidate)


def _adaptive_candidate_cost(
    offsets: tuple[int, ...],
    daily_states: list[FitnessState],
    daily_state_options: list[list[FitnessState]],
    request: RecommendationRequest,
    config: dict,
    target_distance_range: tuple[float, float],
    *,
    taper_horizon: bool,
    completed_miles_by_offset: dict[int, float] | None = None,
) -> float:
    """Score one continuous run-date combination across the entire horizon.

    Day seven is not a reset point: frequency, workout composition, recovery,
    and mileage are evaluated across the supplied horizon before the UI takes
    a slice.
    """
    planned: list[RecommendationResponse] = []
    completed_miles_by_offset = completed_miles_by_offset or {}
    completed_miles = sum(completed_miles_by_offset.values())
    projected_miles = completed_miles
    projected_miles_by_offset = dict(completed_miles_by_offset)
    # Completed mileage is already represented in the opening trailing load.
    # This map contains only proposed work so the reload boundary is not
    # double-counted.
    minimum_density_miles_by_offset: dict[int, float] = {}
    allocatable_headroom_by_offset: dict[int, float] = {}
    horizon_scale = len(daily_states) / VISIBLE_HORIZON_DAYS
    horizon_target_low = target_distance_range[0] * horizon_scale
    horizon_target_high = target_distance_range[1] * horizon_scale
    cost = 0.0
    for offset in offsets:
        role = _elapsed_workout_role(
            daily_state_options[offset],
            planned,
            config,
            taper_horizon=taper_horizon,
        )
        remaining_horizon_runs = sum(offset < later for later in offsets)
        reserve_easy_miles = typical_easy_distance(daily_states[0])[0]
        state, result = _select_budgeted_timed_recommendation(
            daily_state_options[offset],
            planned,
            request,
            config,
            weekly_role=role,
            projected_week_miles=projected_miles,
            remaining_week_runs=remaining_horizon_runs,
            target_week_high=horizon_target_high,
            reserve_easy_miles=reserve_easy_miles,
        )
        if result.workout_type == WorkoutType.REST:
            cost += 100.0
            continue
        if (
            result.workout_type == WorkoutType.RECOVERY
            and request.health_status == CurrentHealthStatus.NORMAL
        ):
            # A candidate date that forces a recovery substitution is worse
            # than an adjacent date that can support the intended training.
            # Without this, the optimizer can game the mileage target by
            # intentionally provoking a short guardrail workout.
            cost += 10.0

        result_midpoint = _midpoint(result)
        projected_miles += result_midpoint
        projected_miles_by_offset[offset] = (
            projected_miles_by_offset.get(offset, 0.0) + result_midpoint
        )
        if (
            result.readiness.value != "ready"
            or result.workout_type in {WorkoutType.RECOVERY, WorkoutType.RACE}
            or not result.distance_range_miles
        ):
            minimum_density_miles = result_midpoint
        elif result.workout_type == WorkoutType.LONG:
            minimum_density_miles = result.distance_range_miles[0]
        elif result.workout_type in {
            WorkoutType.INTERVALS,
            WorkoutType.TEMPO_THRESHOLD,
        }:
            minimum_density_miles = sum(MINIMUM_QUALITY_RANGE_MILES) / 2
        else:
            # An extra aerobic day can be a useful 2.0–2.5 mile run. Date
            # selection must not price it as a full standalone easy template
            # before the joint allocator has distributed the mileage.
            minimum_density_miles = 2.25
        minimum_density_miles_by_offset[offset] = (
            minimum_density_miles_by_offset.get(offset, 0.0)
            + minimum_density_miles
        )
        if (
            result.readiness.value == "ready"
            and result.distance_range_miles
            and result.workout_type not in {WorkoutType.RECOVERY, WorkoutType.RACE}
        ):
            easy_low, easy_high = typical_easy_distance(state)
            current_midpoint = _midpoint(result)
            if result.workout_type == WorkoutType.LONG:
                progression_factor = float(
                    config.get("coaching", {}).get(
                        "long_run_progression_factor", 1.10
                    )
                )
                reachable = min(
                    single_session_progression_reference_miles(state)
                    * progression_factor,
                    sum(target_distance_range) / 2 * 0.40,
                )
            elif result.workout_type in {
                WorkoutType.INTERVALS,
                WorkoutType.TEMPO_THRESHOLD,
            }:
                reachable = easy_high + 1.0
            else:
                reachable = max(
                    easy_high,
                    min(
                        easy_high * 1.75,
                        long_run_reference_miles(state) * 1.05,
                    ),
                )
                recovery_trace = next(
                    (
                        item
                        for item in result.rule_trace
                        if item.rule_id == "recent_recovery_load"
                    ),
                    None,
                )
                recovery_residual = (
                    recovery_trace.facts.get("residual_load")
                    if recovery_trace
                    and isinstance(
                        recovery_trace.facts.get("residual_load"),
                        (int, float),
                    )
                    else None
                )
                if (
                    recovery_residual is not None
                    and recovery_residual > EASY_VOLUME_RESIDUAL_FLOOR
                ):
                    # The allocator will preserve the recommendation's
                    # recovery-scaled ceiling. Frequency selection must use
                    # the same reachable mileage or it can choose too few run
                    # days based on headroom that later disappears.
                    reachable = min(
                        reachable,
                        result.distance_range_miles[1],
                    )
            allocatable_headroom_by_offset[offset] = max(
                0.0, reachable - current_midpoint
            )
        planned.append(result)
        if result.readiness.value == "not_ready":
            cost += 100.0
        elif result.readiness.value == "caution":
            cost += 1.5

        selected_raw_state = next(
            (
                option
                for option in daily_state_options[offset]
                if option.as_of == state.as_of
            ),
            daily_states[offset],
        )
        # Keep the athlete-relative load unit stable across the planning
        # horizon. Recomputing it from every future raw state lets an older run
        # rolling out of the lookback make the exact same proposed workout
        # abruptly more expensive one day later, which is not recovery decay.
        easy_reference_miles = sum(typical_easy_distance(daily_states[0])) / 2
        residual_load = _decayed_recovery_load(
            selected_raw_state,
            planned[:-1],
            state.as_of,
            easy_reference_miles,
            config,
        )
        proposed_load = _recommendation_load_units(
            result, easy_reference_miles
        )
        # Two load units are the recoverable envelope of roughly two ordinary
        # easy sessions. Rest replenishes that envelope exponentially, so a
        # larger workout can fit after more recovery while the same workout
        # becomes unattractive when prior session load is still present.
        cost += max(
            0.0, residual_load + proposed_load - 2.0
        ) * 20.0
        # Residual stress matters even below the hard two-unit envelope. This
        # interaction decays continuously with time, so an extra rest day can
        # create room for more work instead of spacing being only a calendar
        # penalty or later dates being punished for accumulated weekly miles.
        cost += residual_load * proposed_load * 6.0
        # Do not compare candidate dates by their raw trailing-seven-day ratio.
        # A run rolling out on day eight creates a calendar cliff, not a sudden
        # physiological recovery. The recommendation guardrail still sees the
        # rolling load, while date selection uses the continuously decaying
        # completed/planned session load above.

        # Weather changes date preference continuously. A marginal forecast
        # change should not create a four-point cliff that silently deletes a
        # long run from the week.
        cost += assess_training_weather(
            state.planned_weather, state.weather_exposure_baseline
        ).score * 3.0
        candidate_load_ratio = effective_load_ratio(state.recent_load)
        candidate_high_load = float(
            config.get("coaching", {}).get("high_load_ratio", 1.30)
        )
        if candidate_load_ratio is not None:
            # Only material overload is date evidence. Ordinary changes in a
            # trailing window are handled by continuous recovery decay and the
            # soft density model instead of creating a calendar cliff.
            cost += max(
                0.0, candidate_load_ratio - candidate_high_load
            ) * 12.0

        # Date scoring above already uses residual-load × proposed-load. The
        # shared spacing function is reserved for choosing among same-day time
        # options so the same recovery evidence is not charged twice here.

    reachable_miles = projected_miles + sum(
        allocatable_headroom_by_offset.values()
    )
    if reachable_miles < horizon_target_low:
        cost += (horizon_target_low - reachable_miles) * 3.0
    elif projected_miles > horizon_target_high:
        cost += (projected_miles - horizon_target_high) * 3.0

    # Weekly mileage is a load-density guide, not a hard calendar wall. Price
    # only meaningful overages across every rolling seven-day span in the
    # 21-day lookahead,
    # and make the cost grow smoothly. This can retain a justified long or
    # quality session just above the range, but increasingly favors spreading
    # clustered mileage instead of deleting or truncating a workout because it
    # crossed one particular display boundary.
    if len(daily_states) >= VISIBLE_HORIZON_DAYS:
        soft_overage_allowance = max(
            0.5,
            min(1.0, target_distance_range[1] * 0.05),
        )
        soft_density_high = (
            target_distance_range[1] + soft_overage_allowance
        )
        opening_recent_miles = (
            daily_states[0].recent_load.trailing_7d.distance_miles
        )
        for end in range(len(daily_states)):
            start = max(0, end - VISIBLE_HORIZON_DAYS + 1)
            # The daily state has aggregate trailing mileage, not the exact
            # age of each prior run. Decay that opening load linearly instead
            # of pretending it vanishes at day seven; future proposed runs
            # then fill the same continuous load window.
            retained_opening_miles = opening_recent_miles * max(
                0.0,
                1.0 - end / VISIBLE_HORIZON_DAYS,
            )
            window_miles = sum(
                miles
                for offset, miles in minimum_density_miles_by_offset.items()
                if start <= offset <= end
            ) + retained_opening_miles
            density_excess = max(0.0, window_miles - soft_density_high)
            cost += density_excess**2 * 1.5

    typical_rest_days = int(
        config.get("coaching", {}).get("typical_rest_days_between_runs", 1)
    )
    maximum_unforced_rest_days = max(2, typical_rest_days + 1)
    opening_load_ratio = effective_load_ratio(daily_states[0].recent_load)
    high_load_threshold = float(
        config.get("coaching", {}).get("high_load_ratio", 1.30)
    )
    if (
        request.health_status != CurrentHealthStatus.NORMAL
        or (
            opening_load_ratio is not None
            and opening_load_ratio >= high_load_threshold
        )
    ):
        maximum_unforced_rest_days += 1
    # This is a modest cadence term, not a hard spacing rule. At ordinary
    # load it prevents a small second-week scoring advantage from manufacturing
    # three empty days; genuine load/readiness costs can still outweigh it,
    # and high opening load explicitly earns another unpenalized rest day.
    cost += 4.0 * sum(
        max(0, (current - previous - 1) - maximum_unforced_rest_days)
        for previous, current in zip(offsets, offsets[1:])
    )

    # Consecutive dates are not categorically blocked. Their cost is mostly
    # athlete-relative above: the prior session's decayed load is
    # multiplied by the proposed session load. Retain only a small transition
    # cost so an equivalent spaced plan wins, while mileage coverage can still
    # justify a useful double. The former twelve-point charge made consecutive
    # dates practically impossible and manufactured an every-other-day plan.
    cost += 3.0 * sum(
        current - previous == 1
        for previous, current in zip(offsets, offsets[1:])
    )
    # A second adjacent link forms a three-day cluster. Price that
    # progressively so separate useful doubles usually win, without making a
    # short low-load three-day block categorically illegal.
    cost += 6.0 * sum(
        current - two_back == 2
        for two_back, current in zip(offsets, offsets[2:])
    )
    taxing_types = {
        WorkoutType.LONG,
        WorkoutType.INTERVALS,
        WorkoutType.TEMPO_THRESHOLD,
        WorkoutType.RACE,
    }
    for previous, current in zip(planned, planned[1:]):
        if not previous.planned_for or not current.planned_for:
            continue
        gap_days = (
            current.planned_for.date() - previous.planned_for.date()
        ).days
        if (
            previous.workout_type in taxing_types
            and current.workout_type in taxing_types
            and gap_days < 3
        ):
            cost += 12.0 * (3 - gap_days)
    # A completed run immediately before the lookahead is included by
    # ``_decayed_recovery_load``. Apply the same small transition cost used
    # inside the lookahead so a history/plan seam is not a free calendar reset.
    if offsets and offsets[0] == 0 and daily_states[0].days_since_last_run is not None:
        inferred_last_date = (
            daily_states[0].as_of
            - timedelta(days=daily_states[0].days_since_last_run)
        ).date()
        if inferred_last_date == daily_states[0].as_of.date() - timedelta(days=1):
            cost += 3.0
    if (
        offsets
        and planned
        and planned[0].planned_for
        and daily_states[0].days_since_last_run is not None
        and daily_states[0].last_run_workout_type in taxing_types
        and planned[0].workout_type in taxing_types
    ):
        inferred_last_date = (
            daily_states[0].as_of
            - timedelta(days=daily_states[0].days_since_last_run)
        ).date()
        boundary_gap_days = (
            planned[0].planned_for.date() - inferred_last_date
        ).days
        if boundary_gap_days < 3:
            cost += 12.0 * (3 - boundary_gap_days)

    if offsets:
        opening_load_ratio = effective_load_ratio(
            daily_states[0].recent_load
        )
        high_load_threshold = float(
            config.get("coaching", {}).get("high_load_ratio", 1.30)
        )
        if (
            request.health_status == CurrentHealthStatus.NORMAL
            and (
                opening_load_ratio is None
                or opening_load_ratio < high_load_threshold
            )
        ):
            # Waiting several usable days and then paying for consecutive
            # catch-up runs is not recovery optimization. It is deferred load.
            cost += 12.0 * max(0, offsets[0] - 1)

    # Long and quality recurrence are elapsed-time objectives, not positions
    # in a weekly sequence. Allow recovery to move them, but make an
    # unexplained drift beyond the ordinary 7–8 day window progressively
    # expensive across the continuous lookahead.
    cadence_specs = [
        (
            WorkoutType.LONG,
            daily_states[0].days_since_long_run,
            float(
                config.get("coaching", {}).get(
                    "long_run_recency_reference_days", 7
                )
            ),
        ),
    ]
    if not taper_horizon:
        cadence_specs.append(
            (
                {WorkoutType.INTERVALS, WorkoutType.TEMPO_THRESHOLD},
                daily_states[0].days_since_quality_run,
                float(
                    config.get("coaching", {}).get(
                        "quality_recency_reference_days", 7
                    )
                ),
            )
        )
    for workout_types, days_since, reference in cadence_specs:
        if days_since is None:
            continue
        accepted_types = (
            workout_types
            if isinstance(workout_types, set)
            else {workout_types}
        )
        first = next(
            (
                item
                for item in planned
                if item.workout_type in accepted_types and item.planned_for
            ),
            None,
        )
        if first is None or first.planned_for is None:
            cost += max(0.0, days_since + len(daily_states) - reference) * 8.0
            continue
        elapsed_days = (
            first.planned_for - daily_states[0].as_of
        ).total_seconds() / 86400
        cost += max(
            0.0,
            days_since + elapsed_days - (reference + 1.0),
        ) * 8.0

    return cost


def _adaptive_run_day_offsets_for_frequency(
    daily_states: list[FitnessState],
    request: RecommendationRequest,
    config: dict,
    target_run_count: int,
    target_distance_range: tuple[float, float],
    *,
    taper_horizon: bool = False,
    daily_state_options: list[list[FitnessState]] | None = None,
    forced_rest_offsets: set[int] | None = None,
    completed_run_offsets: set[int] | None = None,
    completed_miles_by_offset: dict[int, float] | None = None,
    horizon_run_count: int | None = None,
) -> list[int]:
    """Choose dates for one candidate frequency across a continuous horizon.

    The frequency is an enumeration detail used by the outer optimizer, not
    an athlete target. No seven-day sub-window receives its own run quota.
    """
    if request.health_status == CurrentHealthStatus.PAIN_OR_INJURY_CONCERN:
        return []
    daily_state_options = daily_state_options or [
        [state] for state in daily_states
    ]
    if len(daily_state_options) != len(daily_states) or any(
        not options for options in daily_state_options
    ):
        raise ValueError("Each planning day needs at least one timing option")
    forced_rest_offsets = {
        offset
        for offset in (forced_rest_offsets or set())
        if 0 <= offset < len(daily_states)
    }
    completed_run_offsets = {
        offset
        for offset in (completed_run_offsets or set())
        if 0 <= offset < len(daily_states)
    }
    # If the beginning of the plan is explicitly unavailable, plan the normal
    # cadence from the first usable day. Reusing the ordinary planner here is
    # important: keeping an unavailable day zero as the cadence anchor turns a
    # rest-day constraint into a one-day substitution instead of a fresh plan.
    planning_start_offset = next(
        (
            offset
            for offset in range(len(daily_states))
            if offset not in forced_rest_offsets
        ),
        len(daily_states),
    )
    if planning_start_offset >= len(daily_states):
        return []
    if horizon_run_count is None:
        relative_seed_offsets = automatic_run_day_offsets(
            daily_states[planning_start_offset],
            request.health_status,
            config,
            target_run_count,
            horizon_days=len(daily_states) - planning_start_offset,
        )
        base_seed_offsets = [
            planning_start_offset + offset for offset in relative_seed_offsets
        ]
        desired_total_runs = len(base_seed_offsets)
    else:
        desired_total_runs = min(
            max(0, horizon_run_count),
            len(daily_states) - planning_start_offset,
        )
        if desired_total_runs <= 1:
            base_seed_offsets = (
                [planning_start_offset] if desired_total_runs else []
            )
        else:
            usable_span = len(daily_states) - planning_start_offset - 1
            base_seed_offsets = [
                planning_start_offset
                + round(position * usable_span / (desired_total_runs - 1))
                for position in range(desired_total_runs)
            ]
    if desired_total_runs <= 0:
        return []
    allowed_offsets = [
        offset
        for offset in range(len(daily_states))
        if offset not in forced_rest_offsets
        and offset not in completed_run_offsets
    ]
    total_runs = min(
        max(0, desired_total_runs - len(completed_run_offsets)),
        len(allowed_offsets),
    )
    if total_runs <= 0:
        return []
    # Completed runs satisfy the nearest nominal opportunity. This seed is
    # used only to make the combinatorial prefilter efficient; the full load
    # model below does not score adherence to it.
    seed_offsets = list(base_seed_offsets)
    for completed_offset in sorted(completed_run_offsets):
        if seed_offsets:
            seed_offsets.remove(
                min(
                    seed_offsets,
                    key=lambda value: (abs(value - completed_offset), value),
                )
            )
    seed_offsets = seed_offsets[:total_runs]
    best_offsets: tuple[int, ...] | None = None
    best_cost: float | None = None
    today_is_materially_available = False
    if 0 in allowed_offsets:
        _, today_result = _select_timed_recommendation(
            daily_state_options[0],
            [],
            request,
            config,
            weekly_role=_elapsed_workout_role(
                daily_state_options[0],
                [],
                config,
                taper_horizon=taper_horizon,
            ),
        )
        inferred_last_run_date = (
            (
                daily_states[0].as_of
                - timedelta(days=daily_states[0].days_since_last_run)
            ).date()
            if daily_states[0].days_since_last_run is not None
            else None
        )
        ran_yesterday = inferred_last_run_date == (
            daily_states[0].as_of.date() - timedelta(days=1)
        )
        low_cost_consecutive, _ = consecutive_day_evidence(
            daily_states[0], config
        )
        today_is_materially_available = bool(
            today_result.workout_type != WorkoutType.REST
            and today_result.readiness.value == "ready"
            and not (ran_yesterday and not low_cost_consecutive)
        )
    weather_costs: dict[int, float] = {}
    high_load_threshold = float(
        config.get("coaching", {}).get("high_load_ratio", 1.30)
    )
    for offset in allowed_offsets:
        state = daily_states[offset]
        weather = assess_training_weather(
            state.planned_weather,
            state.weather_exposure_baseline,
        )
        load_ratio = effective_load_ratio(state.recent_load)
        weather_costs[offset] = weather.score * 3.0 + (
            100.0 if weather.extreme else 0.0
        ) + (
            max(0.0, load_ratio - high_load_threshold) * 12.0
            if load_ratio is not None
            else 0.0
        )

    def incremental_prefilter_cost(
        prefix: tuple[int, ...],
        offset: int,
    ) -> float:
        position = len(prefix)
        score = weather_costs[offset]
        if position < len(seed_offsets):
            # The seed makes the beam search efficient; it is not a desired
            # calendar. A full-strength penalty here turned even spacing into
            # an accidental every-other-day prescription before the coaching
            # model could evaluate recovery and session load.
            score += abs(offset - seed_offsets[position]) * 0.25
        if prefix:
            gap = offset - prefix[-1]
            if gap == 1:
                # Keep doubles in the beam without letting a dense calendar
                # crowd out every spaced alternative before full load scoring.
                score += 3.0
                if len(prefix) >= 2 and prefix[-1] - prefix[-2] == 1:
                    score += 6.0
            elif gap > 3:
                score += 2.0 * (gap - 3)
        return score

    def prefilter_cost(
        offsets: tuple[int, ...],
    ) -> tuple[float, tuple[int, ...]]:
        score = 0.0
        prefix: tuple[int, ...] = ()
        for offset in offsets:
            score += incremental_prefilter_cost(prefix, offset)
            prefix = (*prefix, offset)
        return score, offsets

    # Fourteen days was still small enough to enumerate before applying the
    # full coaching model. At 21 days, common frequencies create hundreds of
    # thousands or millions of combinations. Build the same kind of
    # whole-horizon candidates incrementally and retain the strongest partial
    # calendars at each depth. The display boundary never participates.
    combination_count = comb(len(allowed_offsets), total_runs)
    if combination_count <= 5_000:
        candidate_offsets = list(combinations(allowed_offsets, total_runs))
    else:
        beam: list[tuple[float, tuple[int, ...], int]] = [(0.0, (), -1)]
        for position in range(total_runs):
            remaining = total_runs - position - 1
            expanded: list[tuple[float, tuple[int, ...], int]] = []
            for score, prefix, previous_index in beam:
                first_index = previous_index + 1
                final_index = len(allowed_offsets) - remaining
                for allowed_index in range(first_index, final_index):
                    offset = allowed_offsets[allowed_index]
                    if (
                        today_is_materially_available
                        and position == 0
                        and offset != 0
                    ):
                        continue
                    expanded.append(
                        (
                            score + incremental_prefilter_cost(prefix, offset),
                            (*prefix, offset),
                            allowed_index,
                        )
                    )
            beam = sorted(
                expanded,
                key=lambda item: (item[0], item[1]),
            )[:MAX_ADAPTIVE_CANDIDATES]
            if not beam:
                break
        candidate_offsets = [prefix for _, prefix, _ in beam]

    if today_is_materially_available:
        # Preserve a genuinely available current-day slot unless yesterday
        # carried material cost. Rest otherwise competes normally in the full
        # load/recovery model.
        candidate_offsets = [
            offsets for offsets in candidate_offsets if offsets and offsets[0] == 0
        ]
    if len(candidate_offsets) > MAX_ADAPTIVE_CANDIDATES:
        candidate_offsets = sorted(
            candidate_offsets,
            key=prefilter_cost,
        )[:MAX_ADAPTIVE_CANDIDATES]

    for offsets in candidate_offsets:
        cost = _adaptive_candidate_cost(
            offsets,
            daily_states,
            daily_state_options,
            request,
            config,
            target_distance_range,
            taper_horizon=taper_horizon,
            completed_miles_by_offset=completed_miles_by_offset,
        )
        if best_cost is None or (cost, offsets) < (best_cost, best_offsets):
            best_cost = cost
            best_offsets = offsets
    return list(best_offsets or ())


def adaptive_run_day_offsets(
    daily_states: list[FitnessState],
    request: RecommendationRequest,
    config: dict,
    target_run_count: int,
    target_distance_range: tuple[float, float],
    *,
    taper_horizon: bool = False,
    daily_state_options: list[list[FitnessState]] | None = None,
    forced_rest_offsets: set[int] | None = None,
    completed_run_offsets: set[int] | None = None,
    completed_miles_by_offset: dict[int, float] | None = None,
) -> list[int]:
    """Select both frequency and dates from one continuous-horizon model.

    ``target_run_count`` is retained for API compatibility but deliberately
    ignored. Mileage coverage rewards enough useful sessions; recovery/load
    interactions, minimum useful distance, and spacing reject excess ones.
    """
    if request.health_status == CurrentHealthStatus.PAIN_OR_INJURY_CONCERN:
        return []
    del target_run_count
    daily_state_options = daily_state_options or [
        [state] for state in daily_states
    ]
    forced_rest_offsets = forced_rest_offsets or set()
    completed_run_offsets = completed_run_offsets or set()
    completed_miles_by_offset = completed_miles_by_offset or {}
    ordinary_easy_midpoint = sum(typical_easy_distance(daily_states[0])) / 2
    horizon_scale = len(daily_states) / VISIBLE_HORIZON_DAYS
    opening_load_ratio = effective_load_ratio(daily_states[0].recent_load)
    if opening_load_ratio is None:
        target_fraction = 0.50
    else:
        # Low accumulated load creates room to progress toward the upper part
        # of the safe range, often through another useful easy run. As acute
        # load rises, distribute only the lower/middle target.
        load_position = min(
            1.0,
            max(0.0, (opening_load_ratio - 0.75) / 0.35),
        )
        target_fraction = 0.75 - load_position * 0.40
    desired_weekly_miles = (
        target_distance_range[0]
        + (target_distance_range[1] - target_distance_range[0])
        * target_fraction
    )
    desired_horizon_miles = max(
        0.0,
        desired_weekly_miles * horizon_scale
        - sum(completed_miles_by_offset.values()),
    )
    central_horizon_count = max(
        1,
        round(desired_horizon_miles / max(1.0, ordinary_easy_midpoint)),
    )
    maximum_horizon_count = len(daily_states) - len(forced_rest_offsets)
    if request.health_status == CurrentHealthStatus.SICK_OR_RECOVERING:
        maximum_horizon_count = min(
            maximum_horizon_count,
            max(1, ceil(2 * horizon_scale)),
        )
    # Search useful counts directly. The old central-count-minus-three floor
    # could exclude a lower-frequency plan even when slightly longer sessions
    # stayed inside the allocator's established aerobic range. Derive a
    # capacity floor from that same 1.75x useful-session ceiling, while keeping
    # the upper side near the ordinary-distance center. This is a search bound,
    # not a prescribed run count; the full recovery/load model still chooses.
    maximum_useful_average = max(2.5, ordinary_easy_midpoint * 1.75)
    capacity_supported_minimum = max(
        1,
        ceil(desired_horizon_miles / maximum_useful_average),
    )
    maximum_horizon_count = min(
        maximum_horizon_count,
        central_horizon_count + 3,
    )
    minimum_horizon_count = max(
        1,
        min(
            central_horizon_count - 3,
            capacity_supported_minimum,
            maximum_horizon_count,
        ),
    )
    # The broad capacity floor adds only a bounded number of whole-horizon
    # searches. Lower-count alternatives are retained first because they were
    # the cases the old window omitted; the ordinary center remains inside
    # this cap for a 21-day horizon.
    maximum_horizon_count = min(
        maximum_horizon_count,
        minimum_horizon_count + MAX_HORIZON_COUNT_OPTIONS - 1,
    )
    frequency_options = range(
        minimum_horizon_count,
        maximum_horizon_count + 1,
    )
    choices: list[tuple[float, int, list[int]]] = []
    for horizon_run_count in frequency_options:
        offsets = _adaptive_run_day_offsets_for_frequency(
            daily_states,
            request,
            config,
            max(1, round(horizon_run_count / horizon_scale)),
            target_distance_range,
            taper_horizon=taper_horizon,
            daily_state_options=daily_state_options,
            forced_rest_offsets=forced_rest_offsets,
            completed_run_offsets=completed_run_offsets,
            completed_miles_by_offset=completed_miles_by_offset,
            horizon_run_count=horizon_run_count,
        )
        if not offsets:
            continue
        horizon_offsets = tuple(offsets)
        coaching_cost = _adaptive_candidate_cost(
            horizon_offsets,
            daily_states,
            daily_state_options,
            request,
            config,
            target_distance_range,
            taper_horizon=taper_horizon,
            completed_miles_by_offset=completed_miles_by_offset,
        )
        planned_count = max(1, len(horizon_offsets))
        remaining_target_miles = max(
            0.0,
            desired_horizon_miles,
        )
        # Reserve realistic long and quality mileage before judging how much
        # work the easy days must carry. Comparing the all-session average to
        # a typical easy run falsely labels a normal mixed plan as
        # concentrated and can reward an unnecessary extra run day.
        cycle_count = max(1, round(horizon_scale))
        long_slots = min(planned_count, cycle_count)
        quality_slots = min(
            max(0, planned_count - long_slots),
            0 if taper_horizon else cycle_count,
        )
        easy_slots = max(0, planned_count - long_slots - quality_slots)
        progression_factor = float(
            config.get("coaching", {}).get(
                "long_run_progression_factor", 1.10
            )
        )
        long_distribution_reference = max(
            ordinary_easy_midpoint,
            min(
                single_session_progression_reference_miles(daily_states[0])
                * progression_factor,
                desired_weekly_miles * 0.45,
            ),
        )
        quality_distribution_reference = min(
            MAXIMUM_ORDINARY_QUALITY_MILES,
            ordinary_easy_midpoint + 1.0,
        )
        reserved_role_miles = (
            long_slots * long_distribution_reference
            + quality_slots * quality_distribution_reference
        )
        remaining_easy_miles = max(
            0.0,
            remaining_target_miles - reserved_role_miles,
        )
        required_easy_average = (
            remaining_easy_miles / easy_slots
            if easy_slots
            else remaining_target_miles / planned_count
        )
        easy_expansion_reference = _ordinary_easy_expansion_reference(
            daily_states[0]
        )
        concentration = max(
            0.0,
            required_easy_average / max(1.0, ordinary_easy_midpoint) - 1.0,
        )
        # Once the normal expansion ceiling is crossed, another useful easy
        # day should become substantially more attractive. Normalize by the
        # available expansion band so this remains athlete-relative instead
        # of becoming a hidden mileage or run-count threshold.
        oversize_easy = max(
            0.0,
            (required_easy_average - easy_expansion_reference)
            / max(0.5, easy_expansion_reference - ordinary_easy_midpoint),
        )
        required_average = remaining_target_miles / planned_count
        too_small = max(0.0, (2.0 - required_average) / 2.0)
        distribution_cost = 70.0 * (
            concentration * concentration
            + oversize_easy * oversize_easy
            + too_small * too_small
        )
        horizon_gaps = [
            current - previous
            for previous, current in zip(
                horizon_offsets, horizon_offsets[1:]
            )
        ]
        spacing_distribution_cost = (
            3.0 * sum(gap == 1 for gap in horizon_gaps)
            + 6.0 * sum(
                current - two_back == 2
                for two_back, current in zip(
                    horizon_offsets, horizon_offsets[2:]
                )
            )
            + 2.0 * sum(max(0, gap - 3) for gap in horizon_gaps)
        ) / max(1.0, horizon_scale / 2.0)
        comparable_cost = (
            coaching_cost / planned_count
            + distribution_cost
            + spacing_distribution_cost
        )
        # No historical run-count penalty belongs here. Mileage distribution
        # rewards useful extra easy sessions; recovery/load interactions and
        # minimum useful distance reject gratuitous ones.
        choices.append((comparable_cost, horizon_run_count, offsets))
    return min(choices, key=lambda item: (item[0], item[1], item[2]))[2] if choices else []


def _cap_visible_distance_high(
    days: list[WeeklyScheduleDay],
    target_high: float,
) -> list[WeeklyScheduleDay]:
    """Trim optional range headroom before a visible week exceeds its target."""
    updated = list(days)

    def total_high() -> float:
        planned_high = sum(
            day.recommendation.distance_range_miles[1]
            for day in updated
            if day.recommendation
            and day.recommendation.workout_type != WorkoutType.REST
            and day.recommendation.distance_range_miles
        )
        completed_miles = sum(
            activity.distance_miles
            for day in updated
            for activity in day.completed_activities
        )
        return planned_high + completed_miles

    candidates = sorted(
        (
            (index, day)
            for index, day in enumerate(updated)
            if day.recommendation and day.recommendation.distance_range_miles
        ),
        key=lambda item: (
            0
            if item[1].recommendation.workout_type
            in {WorkoutType.EASY, WorkoutType.RECOVERY}
            else 1,
            -item[0],
        ),
    )
    for index, day in candidates:
        excess = round(max(0.0, total_high() - target_high), 1)
        if excess <= 0:
            break
        result = day.recommendation
        assert result is not None and result.distance_range_miles is not None
        lower, upper = result.distance_range_miles
        reduction = min(excess, upper - lower)
        if reduction <= 0:
            continue
        capped = result.model_copy(
            update={
                "distance_range_miles": (lower, round(upper - reduction, 1)),
                "modification_rules": [
                    *result.modification_rules,
                    "The upper end is capped so the visible week remains inside its mileage target.",
                ],
                "reasons": [
                    *result.reasons,
                    "The upper distance is capped so the seven-day plan stays within its mileage target.",
                ],
            }
        )
        updated[index] = day.model_copy(
            update={
                "recommendation": capped,
                "rationale": (
                    f"{day.rationale} Optional distance was capped to keep the week within its target."
                ),
            }
        )
    # A continuous multi-week allocation can intentionally move one run across
    # the visible boundary. If completed mileage or rounding still leaves the
    # displayed slice above its own ceiling after optional range headroom is
    # removed, reduce ordinary aerobic mileage before touching coached long
    # or quality work.
    for index, day in candidates:
        excess = round(max(0.0, total_high() - target_high), 1)
        if excess <= 0:
            break
        result = day.recommendation
        assert result is not None and result.distance_range_miles is not None
        if result.workout_type not in {
            WorkoutType.EASY,
            WorkoutType.RECOVERY,
            WorkoutType.INTERVALS,
            WorkoutType.TEMPO_THRESHOLD,
        }:
            continue
        lower, upper = result.distance_range_miles
        minimum_lower = (
            MINIMUM_QUALITY_RANGE_MILES[0]
            if result.workout_type
            in {WorkoutType.INTERVALS, WorkoutType.TEMPO_THRESHOLD}
            else 2.0
        )
        reduction = min(excess, max(0.0, lower - minimum_lower))
        if reduction <= 0:
            continue
        reduced_range = (
            round(lower - reduction, 1),
            round(max(lower - reduction, upper - reduction), 1),
        )
        capped = result.model_copy(
            update={
                "distance_range_miles": reduced_range,
                "reasons": [
                    *result.reasons,
                    "Aerobic mileage was reduced because the continuous allocation placed more load inside this seven-day window.",
                ],
            }
        )
        if result.workout_type in {
            WorkoutType.INTERVALS,
            WorkoutType.TEMPO_THRESHOLD,
        }:
            capped = scale_quality_session(
                capped,
                result.distance_range_miles,
                reduced_range,
            )
        updated[index] = day.model_copy(update={"recommendation": capped})
    return updated


def _allocate_visible_distance_ranges(
    days: list[WeeklyScheduleDay],
    daily_states: list[FitnessState],
    target_range: tuple[float, float],
    config: dict,
    *,
    weekly_target_range: tuple[float, float] | None = None,
) -> list[WeeklyScheduleDay]:
    """Jointly allocate session distance and retain only meaningful roles.

    The solver compares a semantically distinct long-run plan with an aerobic
    endurance alternative. It optimizes the supplied horizon on a half-mile
    grid, balancing target coverage, local load density, per-session load,
    progression limits, and role meaning. Recovery/caution sessions remain
    fixed and are never enlarged to make a mileage number work.
    """
    updated = list(days)
    completed_miles = sum(
        activity.distance_miles
        for day in updated
        for activity in day.completed_activities
    )
    fixed_midpoint = 0.0
    records: list[dict] = []
    previous_run_index: int | None = None
    progression_factor = float(
        config.get("coaching", {}).get("long_run_progression_factor", 1.10)
    )
    target_midpoint = sum(target_range) / 2
    for index, day in enumerate(updated):
        result = day.recommendation
        if not result or result.workout_type == WorkoutType.REST:
            continue
        if result.distance_range_miles is None:
            previous_run_index = index
            continue
        lower, upper = result.distance_range_miles
        if (
            result.readiness.value != "ready"
            or result.workout_type in {WorkoutType.RECOVERY, WorkoutType.RACE}
        ):
            fixed_midpoint += (lower + upper) / 2
            previous_run_index = index
            continue
        gap = index - previous_run_index if previous_run_index is not None else 3
        role = (
            "long"
            if result.workout_type == WorkoutType.LONG
            else "quality"
            if result.workout_type
            in {WorkoutType.INTERVALS, WorkoutType.TEMPO_THRESHOLD}
            else "easy"
        )
        easy_reference = typical_easy_distance(daily_states[index])
        recent_durability = long_run_reference_miles(daily_states[index]) * 1.05
        aerobic_maximum = max(
            easy_reference[1],
            min(easy_reference[1] * 1.75, recent_durability),
        )
        if role == "long":
            progression_ceiling = (
                # Recommendation ranges are centered on a quarter mile and
                # may extend another quarter beyond the raw warning ceiling.
                # Preserve that disclosed rounding instead of truncating the
                # return-to-capacity target during weekly allocation.
                floor(
                    single_session_progression_reference_miles(
                        daily_states[index]
                    )
                    * progression_factor
                    * 4
                    + 1e-9
                )
                / 4
                + 0.5
            )
            minimum = max(2.0, floor(lower * 4 + 1e-9) / 4)
            recovery_trace = next(
                (
                    item
                    for item in result.rule_trace
                    if item.rule_id == "recent_recovery_load"
                ),
                None,
            )
            residual = (
                recovery_trace.facts.get("residual_load")
                if recovery_trace
                else None
            )
            if isinstance(residual, (int, float)):
                recovery_headroom = (
                    1.0
                    if residual <= 0.05
                    else max(
                        0.0,
                        min(1.0, 1.0 - residual / TAXING_RUN_RESIDUAL_LIMIT),
                    )
                )
            else:
                recovery_headroom = (
                    1.0
                    if daily_states[index].days_since_last_run is None
                    else 0.0
                )
            # Expose progressively more of the safe single-run ceiling as
            # athlete-relative recovery load decays. The weekly solver uses
            # that headroom only when ordinary session sizes leave a mileage
            # shortfall, so extra rest permits a larger long run without
            # making maximum progression automatic.
            recovery_ceiling = minimum + (
                progression_ceiling - minimum
            ) * recovery_headroom
            maximum = max(
                minimum,
                floor(recovery_ceiling * 4 + 1e-9) / 4,
            )
            # Extra recovery may expose the safe single-session progression
            # ceiling, but the long run must not become the default sink for
            # every unallocated mile. At low frequency a long run can
            # reasonably carry a large share of the week; beyond roughly 45%
            # the surrounding aerobic and quality work is too small to form a
            # balanced program. This scales with weekly volume rather than
            # imposing an athlete-independent mileage cap.
            weekly_budget_midpoint = sum(
                weekly_target_range or target_range
            ) / 2
            # Low-volume weeks can legitimately devote a larger fraction to
            # one long run because every useful session has a minimum size.
            # As weekly volume grows, converge smoothly to the 45% ceiling.
            maximum_long_share = max(
                0.45,
                0.60 - max(0.0, weekly_budget_midpoint - 12.5) * 0.015,
            )
            weekly_share_ceiling = (
                floor(
                    weekly_budget_midpoint
                    * maximum_long_share
                    * 4
                    + 1e-9
                )
                / 4
            )
            maximum = max(
                minimum,
                min(maximum, weekly_share_ceiling),
            )
            preferred = minimum
            weight = 1.20
        elif role == "quality":
            # Preserve a useful quality stimulus before deleting the role.
            # At this center the displayed range is 3.0–3.5 miles, which is
            # enough for the shortened work-dose templates plus an easy
            # warm-up and cool-down. The scaling pass below reduces reps or
            # sustained work rather than merely relabeling a shorter run.
            minimum = MINIMUM_QUALITY_RANGE_MILES[1]
            # Quality distance is set by the workout's warm-up, work dose,
            # recoveries, and cool-down—not by whatever weekly mileage remains.
            # It may scale down below the original range when load requires,
            # but ordinary budget allocation never enlarges the session.
            original_midpoint = (lower + upper) / 2
            quality_center = floor(
                (original_midpoint + 0.25) * 2 + 1e-9
            ) / 2
            preferred = min(
                quality_center,
                MAXIMUM_ORDINARY_QUALITY_MILES,
            )
            maximum = preferred
            weight = 1.05
        else:
            # Ordinary easy distance is a default, not a floor. When weekly
            # mileage is better distributed across more run days, allow a
            # genuinely short aerobic session instead of forcing every slot
            # to carry the full standalone template.
            minimum = 2.0
            maximum = max(upper, aerobic_maximum)
            preferred = min(
                maximum,
                ceil((sum(easy_reference) / 2) * 2 - 1e-9) / 2,
            )
            weight = 1.0 + min(3, gap) * 0.05
            recovery_trace = next(
                (
                    item
                    for item in result.rule_trace
                    if item.rule_id == "recent_recovery_load"
                ),
                None,
            )
            recovery_residual = (
                recovery_trace.facts.get("residual_load")
                if recovery_trace
                and isinstance(
                    recovery_trace.facts.get("residual_load"), (int, float)
                )
                else None
            )
            if (
                recovery_residual is not None
                and recovery_residual > EASY_VOLUME_RESIDUAL_FLOOR
            ):
                # A ready easy run can coexist with residual load that makes a
                # taxing session unattractive. Preserve the recommendation's
                # continuously recovery-scaled ceiling rather than enlarging
                # it back to the ordinary template merely to chase miles.
                maximum = min(maximum, upper)
                preferred = min(preferred, maximum)
        option_scale = 4 if role == "long" else 2
        maximum = max(
            minimum,
            floor(maximum * option_scale + 1e-9) / option_scale,
        )
        options = [
            step / option_scale
            for step in range(
                int(minimum * option_scale),
                int(maximum * option_scale) + 1,
            )
        ]
        records.append(
            {
                "index": index,
                "role": role,
                "weight": weight,
                "minimum": minimum,
                "maximum": maximum,
                "preferred": preferred,
                "options": options,
                "aerobic_options": [
                    step / 2
                    for step in range(
                        int(easy_reference[1] * 2),
                        int(
                            max(easy_reference[1], floor(aerobic_maximum * 2) / 2)
                            * 2
                        )
                        + 1,
                    )
                ],
            }
        )
        previous_run_index = index

    if not records:
        return updated

    primary_long = next(
        (record for record in records if record["role"] == "long"),
        None,
    )
    ordinary_easy_midpoint = sum(typical_easy_distance(daily_states[0])) / 2
    long_margin = max(
        0.5,
        round(ordinary_easy_midpoint * 0.15 * 2) / 2,
    )

    def ranges_for(
        assignment: dict[int, float],
        retain_long: bool,
    ) -> dict[int, tuple[float, float]]:
        ranges: dict[int, tuple[float, float]] = {}
        for record in records:
            center = assignment[record["index"]]
            role = record["role"]
            if role == "long" and retain_long:
                ranges[record["index"]] = (
                    center,
                    min(record["maximum"], center + 0.5),
                )
            elif role == "quality":
                ranges[record["index"]] = (max(2.0, center - 0.5), center)
            else:
                ranges[record["index"]] = (max(2.0, center - 0.5), center)
        return ranges

    def solve(retain_long: bool) -> tuple[float, dict[int, float]] | None:
        long_options: list[float | None]
        if retain_long:
            if primary_long is None:
                return None
            minimum_meaningful = ordinary_easy_midpoint + long_margin
            long_options = [
                option
                for option in primary_long["options"]
                if option + 1e-9
                >= minimum_meaningful
            ]
            if not long_options:
                return None
        else:
            long_options = [None]

        best: tuple[float, dict[int, float]] | None = None
        for long_option in long_options:
            scenario_records = [
                record
                for record in records
                if record is not primary_long
            ]
            role_weights = {
                record["index"]: (
                    1.0 if record is primary_long and not retain_long else record["weight"]
                )
                for record in records
            }
            remaining_target = max(
                0.0,
                target_midpoint - completed_miles - fixed_midpoint,
            )
            total_weight = sum(role_weights.values())
            ideals = {
                index: remaining_target * weight / max(0.1, total_weight)
                for index, weight in role_weights.items()
            }
            initial_assignment = (
                {primary_long["index"]: long_option}
                if long_option is not None and primary_long is not None
                else {}
            )
            initial_units = int((long_option or 0) * 4)
            initial_cost = (
                (long_option - ideals[primary_long["index"]]) ** 2
                if long_option is not None and primary_long is not None
                else 0.0
            )
            dp: dict[int, tuple[float, dict[int, float]]] = {
                initial_units: (initial_cost, initial_assignment)
            }
            if primary_long is not None and not retain_long:
                scenario_records = records
            preserve_quality_dose = (
                completed_miles
                + fixed_midpoint
                + (long_option or 0.0)
                + sum(
                    record["preferred"]
                    if record["role"] == "quality"
                    else record["minimum"]
                    for record in scenario_records
                )
                <= target_range[1] + 1e-9
            )
            for record in scenario_records:
                options = list(
                    record["aerobic_options"]
                    if record is primary_long and not retain_long
                    else record["options"]
                )
                if (
                    retain_long
                    and long_option is not None
                    and record["role"] == "easy"
                ):
                    options = [
                        option
                        for option in options
                        if option <= long_option - long_margin + 1e-9
                    ]
                if preserve_quality_dose and record["role"] == "quality":
                    options = [
                        option
                        for option in options
                        if option + 1e-9 >= record["preferred"]
                    ]
                if not options:
                    dp = {}
                    break
                next_dp: dict[int, tuple[float, dict[int, float]]] = {}
                for units, (cost, assignment) in dp.items():
                    for option in options:
                        next_units = units + int(option * 4)
                        next_cost = cost + (
                            option - ideals[record["index"]]
                        ) ** 2
                        candidate = (next_cost, {**assignment, record["index"]: option})
                        if (
                            next_units not in next_dp
                            or candidate[0] < next_dp[next_units][0]
                        ):
                            next_dp[next_units] = candidate
                dp = next_dp
            for balance_cost, assignment in dp.values():
                allocated_ranges = ranges_for(assignment, retain_long)
                projected_low = completed_miles + sum(
                    (
                        allocated_ranges[index][0]
                        if index in allocated_ranges
                        else day.recommendation.distance_range_miles[0]
                    )
                    for index, day in enumerate(updated)
                    if day.recommendation
                    and day.recommendation.workout_type != WorkoutType.REST
                    and day.recommendation.distance_range_miles
                )
                projected_high = completed_miles + sum(
                    (
                        allocated_ranges[index][1]
                        if index in allocated_ranges
                        else day.recommendation.distance_range_miles[1]
                    )
                    for index, day in enumerate(updated)
                    if day.recommendation
                    and day.recommendation.workout_type != WorkoutType.REST
                    and day.recommendation.distance_range_miles
                )
                projected_mid = (projected_low + projected_high) / 2
                total_shortfall = max(0.0, target_range[0] - projected_high)
                total_excess = max(0.0, projected_low - target_range[1])
                density_score = 0.0
                if (
                    weekly_target_range is not None
                    and len(updated) >= VISIBLE_HORIZON_DAYS
                ):
                    allowance = max(
                        0.5,
                        min(1.0, weekly_target_range[1] * 0.05),
                    )
                    soft_low = max(0.0, weekly_target_range[0] - allowance)
                    soft_high = weekly_target_range[1] + allowance
                    opening_recent_miles = (
                        daily_states[0]
                        .recent_load.trailing_7d.distance_miles
                    )
                    for end in range(len(updated)):
                        start = max(
                            0,
                            end - VISIBLE_HORIZON_DAYS + 1,
                        )
                        retained_opening_miles = opening_recent_miles * max(
                            0.0,
                            1.0 - end / VISIBLE_HORIZON_DAYS,
                        )
                        cumulative_low = retained_opening_miles
                        cumulative_high = retained_opening_miles
                        for index in range(start, end + 1):
                            day = updated[index]
                            result = day.recommendation
                            if (
                                not result
                                or result.workout_type == WorkoutType.REST
                                or not result.distance_range_miles
                            ):
                                continue
                            lower, upper = allocated_ranges.get(
                                index, result.distance_range_miles
                            )
                            cumulative_low += lower
                            cumulative_high += upper
                        cumulative_midpoint = (
                            cumulative_low + cumulative_high
                        ) / 2
                        window_excess = max(
                            0.0, cumulative_midpoint - soft_high
                        )
                        window_shortfall = (
                            max(0.0, soft_low - cumulative_midpoint)
                            if end >= VISIBLE_HORIZON_DAYS - 1
                            else 0.0
                        )
                        # Overload is the stronger concern. Underload remains
                        # a lighter preference because recovery, weather, or
                        # forced rest may justify a locally quieter span.
                        density_score += window_excess**2 * 4.0
                        density_score += window_shortfall**2 * 0.75
                score = (
                    balance_cost
                    + (projected_mid - target_midpoint) ** 2
                    + total_shortfall**2 * 30.0
                    + total_excess**2 * 40.0
                    + density_score
                    - (0.25 if retain_long else 0.0)
                )
                if best is None or score < best[0]:
                    best = (score, assignment)
        return best

    long_solution = solve(True)
    aerobic_solution = solve(False)
    # Workout selection has already evaluated recovery, load, recency, and
    # whether the proposed long is meaningfully distinct. If that safe long
    # can fit, reserve it before allocating quality and easy mileage. The
    # distance allocator distributes the week; it must not silently overrule
    # the coaching decision merely because equal-sized aerobic runs produce a
    # tidier numerical balance.
    retain_long = long_solution is not None
    selected = long_solution if retain_long else aerobic_solution
    if selected is None:
        return updated
    allocated_ranges = ranges_for(selected[1], retain_long)
    recovery_capped_indices: set[int] = set()
    allocated_planned: list[RecommendationResponse] = []
    stable_easy_reference = sum(typical_easy_distance(daily_states[0])) / 2
    for record in sorted(records, key=lambda item: item["index"]):
        index = record["index"]
        day = updated[index]
        result = day.recommendation
        assert result is not None and result.distance_range_miles is not None
        allocated = allocated_ranges[index]
        demoted_long = record is primary_long and not retain_long
        provisional_type = WorkoutType.EASY if demoted_long else result.workout_type
        provisional = result.model_copy(
            update={
                "workout_type": provisional_type,
                "distance_range_miles": allocated,
            }
        )
        timing_trace = next(
            (
                item
                for item in result.rule_trace
                if item.rule_id == "planned_timing"
            ),
            None,
        )
        elapsed_from_prior = (
            (
                (result.planned_for - allocated_planned[-1].planned_for)
                .total_seconds()
                / 3600
            )
            if allocated_planned
            and result.planned_for
            and allocated_planned[-1].planned_for
            else None
        )
        projected_prior_matches = bool(
            timing_trace
            and elapsed_from_prior is not None
            and isinstance(timing_trace.facts.get("hours_since_last_run"), (int, float))
            and abs(
                float(timing_trace.facts["hours_since_last_run"])
                - elapsed_from_prior
            )
            <= 1.0
        )
        if allocated_planned and projected_prior_matches:
            planned_at = result.planned_for or daily_states[index].as_of
            residual = _decayed_recovery_load(
                daily_states[index],
                allocated_planned,
                planned_at,
                stable_easy_reference,
                config,
            )
            base_units = _recommendation_load_units(
                result,
                stable_easy_reference,
            )
            base_overflow = max(0.0, residual + base_units - 2.0)
            allocated_units = _recommendation_load_units(
                provisional,
                stable_easy_reference,
            )
            allowed_units = max(0.0, 2.0 + base_overflow - residual)
            if allocated_units > allowed_units + 0.05:
                intensity_factor = (
                    QUALITY_SESSION_LOAD_FACTOR
                    if provisional_type
                    in {
                        WorkoutType.INTERVALS,
                        WorkoutType.TEMPO_THRESHOLD,
                        WorkoutType.RACE,
                    }
                    else LONG_RUN_LOAD_FACTOR
                    if provisional_type == WorkoutType.LONG
                    else RECOVERY_RUN_LOAD_FACTOR
                    if provisional_type == WorkoutType.RECOVERY
                    else 1.0
                )
                allowed_midpoint = (
                    allowed_units
                    * max(1.0, stable_easy_reference)
                    / intensity_factor
                )
                original_upper = result.distance_range_miles[1]
                capped_upper = floor(
                    (allowed_midpoint + 0.25) * 2 + 1e-9
                ) / 2
                capped_upper = min(
                    allocated[1],
                    max(original_upper, capped_upper),
                )
                capped_lower = max(
                    2.0,
                    min(allocated[0], capped_upper - 0.5),
                )
                allocated = (capped_lower, capped_upper)
                allocated_ranges[index] = allocated
                provisional = provisional.model_copy(
                    update={"distance_range_miles": allocated}
                )
                recovery_capped_indices.add(index)
        allocated_planned.append(provisional)

    for record in records:
        index = record["index"]
        day = updated[index]
        result = day.recommendation
        assert result is not None and result.distance_range_miles is not None
        allocated = tuple(round(value, 1) for value in allocated_ranges[index])
        demoted_long = record is primary_long and not retain_long
        revised = result.model_copy(
            update={
                "workout_type": (
                    WorkoutType.EASY if demoted_long else result.workout_type
                ),
                "title": (
                    "Easy aerobic run" if demoted_long else result.title
                ),
                "distance_range_miles": allocated,
                "reasons": (
                    [
                        "A safe long-run prescription was not distinct enough from the other aerobic runs to justify a separate label.",
                        "The weekly allocator favored balanced load over forcing a workout label.",
                    ]
                    if demoted_long
                    else [
                        *result.reasons,
                        "Distance was jointly allocated across the full week under recovery, progression, and workout-role constraints.",
                    ]
                ),
                "modification_rules": [
                    *result.modification_rules,
                    "Shorten toward the lower end if recovery or effort is worse than projected; do not make the mileage up later.",
                    *(
                        [
                            "Optional added mileage was capped after the finalized earlier workouts were projected through this recovery window."
                        ]
                        if index in recovery_capped_indices
                        else []
                    ),
                ],
            }
        )
        if record["role"] == "quality" and not demoted_long:
            revised = scale_quality_session(
                revised,
                result.distance_range_miles,
                allocated,
            )
        recent_distance = daily_states[index].recent_load.trailing_28d.distance_miles
        recent_minutes = daily_states[index].recent_load.trailing_28d.moving_minutes
        estimated_pace = (
            recent_minutes / recent_distance
            if recent_distance > 0 and recent_minutes > 0
            else 11.0
        )
        estimated_pace = min(20.0, max(6.0, estimated_pace))
        estimated_duration = sum(allocated) / 2 * estimated_pace
        revised = structure_extended_quality_session(
            revised,
            estimated_duration,
        )
        updated[index] = day.model_copy(
            update={
                "recommendation": revised,
                "day_role": (
                    "easy_run" if demoted_long else day.day_role
                ),
                "rationale": (
                    f"{day.rationale} The long-run label was removed because a distinct long run would worsen weekly load balance."
                    if demoted_long
                    else f"{day.rationale} Mileage was jointly allocated across the full week."
                ),
            }
        )
    return updated


def summarize_distance_alignment(
    projected: tuple[float, float],
    target: tuple[float, float],
    capacity_reference_miles: float,
) -> str:
    """Describe full range containment instead of treating overlap as a fit."""
    projected_low, projected_high = projected
    target_low, target_high = target
    if capacity_reference_miles <= 0:
        return "This is a starter plan until more runs are available."
    if projected_high < target_low:
        return "This week stays below your usual range."
    if projected_low < target_low:
        return "The lower end is below your usual range; the upper end reaches it if recovery remains normal."
    if projected_low > target_high:
        return "This week is above your usual range, so review each workout before following it."
    if projected_high > target_high:
        return "The upper end is above your usual range; stay near the lower end unless recovery remains normal."
    return "This fits your recent training."


def _rest_day_rationale(
    offset: int,
    run_offsets: list[int],
    daily_state_options: list[list[FitnessState]],
) -> str:
    """Explain today's optimizer-created rest without inventing a guardrail."""
    if offset != 0:
        return (
            "No run is scheduled between these workouts. Other training is not "
            "currently recorded, so only running load is projected."
        )
    next_offset = next((item for item in run_offsets if item > offset), None)
    if next_offset is None:
        return "No run fits the current health and recovery guardrails."
    current = daily_state_options[offset][0]
    upcoming = daily_state_options[next_offset][0]
    current_weather = assess_training_weather(
        current.planned_weather, current.weather_exposure_baseline
    )
    upcoming_weather = assess_training_weather(
        upcoming.planned_weather, upcoming.weather_exposure_baseline
    )
    added_hours = max(0.0, (upcoming.as_of - current.as_of).total_seconds() / 3600)
    day_name = upcoming.as_of.strftime("%A")
    if current_weather.extreme:
        return (
            f"The next run is placed {day_name} because "
            f"{current_weather.extreme_reasons[0]} blocks outdoor running at today's available time."
        )
    if current_weather.score >= upcoming_weather.score + 0.10:
        return (
            f"The next run is placed {day_name} because forecast training stress "
            f"improves from {current_weather.band} to {upcoming_weather.band}, and "
            f"the later start adds about {added_hours:.0f} recovery hours."
        )
    return (
        f"The full-week recovery and projected-load score places the next run "
        f"{day_name}, about {added_hours:.0f} hours after today's best time option."
    )


def build_weekly_schedule(
    daily_states: list[FitnessState],
    request: RecommendationRequest,
    config: dict,
    target_run_count: int | None = None,
    target_distance_range: tuple[float, float] | None = None,
    target_evidence: WeeklyTargetEvidence | None = None,
    completed_activities_by_offset: dict[int, list[TrailingDayActivity]] | None = None,
    daily_state_options: list[list[FitnessState]] | None = None,
    forced_rest_offsets: set[int] | None = None,
) -> WeeklyScheduleResponse:
    """Plan across the supplied horizon and expose its leading seven days."""
    if len(daily_states) < VISIBLE_HORIZON_DAYS:
        raise ValueError("Weekly planning requires at least seven daily states")
    planning_horizon_days = len(daily_states)
    daily_state_options = daily_state_options or [
        [state] for state in daily_states
    ]
    if len(daily_state_options) != planning_horizon_days or any(
        not options for options in daily_state_options
    ):
        raise ValueError("Each planning day needs at least one timing option")
    completed_activities_by_offset = completed_activities_by_offset or {}
    forced_rest_offsets = {
        offset
        for offset in (forced_rest_offsets or set())
        if 0 <= offset < planning_horizon_days
    }
    completed_run_count = sum(
        bool(items)
        for offset, items in completed_activities_by_offset.items()
        if offset < VISIBLE_HORIZON_DAYS
    )
    completed_run_offsets = {
        offset
        for offset, items in completed_activities_by_offset.items()
        if items and 0 <= offset < planning_horizon_days
    }
    completed_miles_by_offset: dict[int, float] = {}
    for offset, items in completed_activities_by_offset.items():
        if not items or not 0 <= offset < planning_horizon_days:
            continue
        completed_miles = sum(item.distance_miles for item in items)
        completed_miles_by_offset[offset] = completed_miles
    if target_distance_range is None:
        capacity = max(
            daily_states[0].recent_load.trailing_28d.distance_miles / 4.0,
            daily_states[0].recent_load.trailing_7d.distance_miles,
        )
        target_distance_range = (_half_mile(capacity * 0.95), _half_mile(capacity * 1.10))
    offset_state = (
        daily_states[0].model_copy(update={"days_since_last_run": 0.0})
        if completed_activities_by_offset.get(0)
        else daily_states[0]
    )
    goal = configured_race_goal(config, on_date=daily_states[0].as_of.date())
    race_offset: int | None = None
    taper_horizon = False
    if goal:
        goal_profile, race_date, _ = goal
        race_offset = (race_date - daily_states[0].as_of.date()).days
        taper_horizon = 0 <= race_offset <= goal_profile.taper_days
    desired_offsets = (
        adaptive_run_day_offsets(
            daily_states,
            request,
            config,
            int(target_run_count or 0),
            target_distance_range,
            taper_horizon=taper_horizon,
            daily_state_options=daily_state_options,
            forced_rest_offsets=forced_rest_offsets,
            completed_run_offsets=completed_run_offsets,
            completed_miles_by_offset=completed_miles_by_offset,
        )
        if planning_horizon_days > VISIBLE_HORIZON_DAYS
        else automatic_run_day_offsets(
            offset_state,
            request.health_status,
            config,
            target_run_count,
            horizon_days=planning_horizon_days,
        )
    )
    offsets = [
        offset
        for offset in desired_offsets
        if offset not in forced_rest_offsets
        and not completed_activities_by_offset.get(offset)
    ]
    if goal:
        if (
            0 <= race_offset < len(daily_states)
            and race_offset not in forced_rest_offsets
            and not completed_activities_by_offset.get(race_offset)
        ):
            if race_offset not in offsets:
                if offsets:
                    replace = min(offsets, key=lambda value: (abs(value - race_offset), -value))
                    offsets.remove(replace)
                offsets.append(race_offset)
                offsets.sort()
            # Do not compress an ordinary scheduled workout onto the calendar
            # day immediately before a race merely to preserve run count.
            if race_offset - 1 in offsets:
                offsets.remove(race_offset - 1)
    planned: list[RecommendationResponse] = []
    days: list[WeeklyScheduleDay] = []
    projected_horizon_miles = sum(completed_miles_by_offset.values())
    horizon_target_high = target_distance_range[1] * (
        planning_horizon_days / VISIBLE_HORIZON_DAYS
    )
    stable_easy_distance = typical_easy_distance(daily_states[0])
    low_cost_consecutive, consecutive_facts = consecutive_day_evidence(
        daily_states[0], config
    )
    for offset, raw_state in enumerate(daily_states):
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
                    rationale="Your uploaded run replaces the planned workout for this day.",
                    completed_activities=completed,
                )
            )
            continue
        if offset in forced_rest_offsets:
            days.append(
                WeeklyScheduleDay(
                    date=raw_state.as_of.date(),
                    planned_at=None,
                    recommendation=None,
                    day_role="forced_rest_day",
                    rationale="You marked this date as unavailable for running; the remaining plan was recalculated around it.",
                    forced_rest=True,
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
                    rationale=_rest_day_rationale(
                        offset, offsets, daily_state_options
                    ),
                )
            )
            continue
        remaining_horizon_runs = sum(offset < later for later in offsets)
        state, result = _select_budgeted_timed_recommendation(
            daily_state_options[offset],
            planned,
            request,
            config,
            weekly_role=_elapsed_workout_role(
                daily_state_options[offset],
                planned,
                config,
                taper_horizon=taper_horizon,
            ),
            projected_week_miles=projected_horizon_miles,
            remaining_week_runs=remaining_horizon_runs,
            target_week_high=horizon_target_high,
            reserve_easy_miles=stable_easy_distance[0],
        )
        timing_adjusted_for_recovery = (
            state.as_of != daily_state_options[offset][0].as_of
        )
        if timing_adjusted_for_recovery:
            result = result.model_copy(
                update={
                    "reasons": [
                        *result.reasons,
                        "This time clears a recovery/readiness guardrail that applied to the weather-preferred earlier option.",
                    ]
                }
            )
        if result.workout_type != WorkoutType.REST:
            planned.append(result)
            projected_horizon_miles += _midpoint(result)
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
                # Timing remains useful coaching advice even when weather is
                # unavailable. In particular, today's evening slot must stay
                # visible for the entire local day.
                planned_at=state.as_of,
                recommendation=result,
                day_role=role,
                rationale=(
                    "Selected from the continuous rolling 21-day plan using projected recovery, rolling mileage, and balanced spacing. "
                    + (
                        "A back-to-back run is reasonable because the previous run was short and easy enough."
                        if offset == 0 and low_cost_consecutive
                        else
                        "The run time was shifted to preserve recovery spacing."
                        if timing_adjusted_for_recovery
                        else
                        "A consecutive run day was evaluated with the prior planned session included."
                        if planned and len(planned) >= 2 and planned[-2].planned_for and (state.as_of.date() - planned[-2].planned_for.date()).days == 1
                        else "The selected date had the strongest available fit."
                    )
                    if planning_horizon_days > VISIBLE_HORIZON_DAYS
                    else
                    "A back-to-back run is reasonable because the previous run was short and easy enough."
                    if offset == 0 and low_cost_consecutive
                    else
                    "Consecutive run day selected intentionally and evaluated with the prior planned session included."
                    if planned and len(planned) >= 2 and planned[-2].planned_for and (state.as_of.date() - planned[-2].planned_for.date()).days == 1
                    else "Fits your recent training and recovery."
                ),
            )
        )
    if planning_horizon_days > VISIBLE_HORIZON_DAYS:
        horizon_scale = planning_horizon_days / VISIBLE_HORIZON_DAYS
        allocated_horizon = _allocate_visible_distance_ranges(
            days,
            daily_states,
            (
                target_distance_range[0] * horizon_scale,
                target_distance_range[1] * horizon_scale,
            ),
            config,
            weekly_target_range=target_distance_range,
        )
        visible_days = allocated_horizon[:VISIBLE_HORIZON_DAYS]
    else:
        visible_days = _allocate_visible_distance_ranges(
            days[:VISIBLE_HORIZON_DAYS],
            daily_states[:VISIBLE_HORIZON_DAYS],
            target_distance_range,
            config,
        )
        visible_days = _cap_visible_distance_high(
            visible_days,
            target_distance_range[1],
        )
    run_results = [day.recommendation for day in visible_days if day.recommendation and day.recommendation.workout_type != WorkoutType.REST]
    response_target_run_count = len(run_results) + completed_run_count
    completed_visible_miles = sum(
        activity.distance_miles
        for day in visible_days
        for activity in day.completed_activities
    )
    distance_low = completed_visible_miles + sum(
        (item.distance_range_miles or (0.0, 0.0))[0] for item in run_results
    )
    distance_high = completed_visible_miles + sum(
        (item.distance_range_miles or (0.0, 0.0))[1] for item in run_results
    )
    if target_evidence is None:
        target_evidence = WeeklyTargetEvidence(
            recent_7d_miles=daily_states[0].recent_load.trailing_7d.distance_miles,
            chronic_42d_weekly_miles=daily_states[0].recent_load.trailing_28d.distance_miles / 4.0,
            best_sustained_28d_weekly_miles=daily_states[0].recent_load.trailing_28d.distance_miles / 4.0,
            peak_7d_miles=daily_states[0].recent_load.trailing_7d.distance_miles,
            current_run_days_per_week=daily_states[0].running_days_28d / 4.0,
            demonstrated_run_days_per_week=daily_states[0].running_days_28d / 4.0,
            capacity_reference_miles=sum(target_distance_range) / 2,
            rationale="Fallback target derived from the supplied fitness state.",
        )
    guardrail_rest_count = sum(
        bool(day.recommendation and day.recommendation.workout_type == WorkoutType.REST)
        for day in visible_days
    )
    forced_rest_count = sum(day.forced_rest for day in visible_days)
    projected_range = (round(distance_low, 1), round(distance_high, 1))
    if guardrail_rest_count:
        horizon_explanation = f"{guardrail_rest_count} planned day{'s were' if guardrail_rest_count != 1 else ' was'} changed to rest based on recovery, health, load, or weather."
    elif forced_rest_count:
        horizon_explanation = (
            "The continuous 21-day plan was recalculated around your selected "
            "rest-day constraints."
        )
    else:
        horizon_explanation = summarize_distance_alignment(
            projected_range,
            target_distance_range,
            target_evidence.capacity_reference_miles,
        )
    generated_at = datetime.now(daily_states[0].as_of.tzinfo)
    return WeeklyScheduleResponse(
        planner_version=WEEKLY_PLANNER_VERSION,
        generated_at=generated_at,
        emergency_alerts_checked_at=generated_at,
        start_date=daily_states[0].as_of.date(),
        end_date=daily_states[VISIBLE_HORIZON_DAYS - 1].as_of.date(),
        target_run_count=response_target_run_count,
        target_distance_range_miles=target_distance_range,
        target_evidence=target_evidence,
        completed_run_count=completed_run_count,
        run_count=len(run_results),
        projected_distance_range_miles=projected_range,
        summary=horizon_explanation,
        days=visible_days,
    )
