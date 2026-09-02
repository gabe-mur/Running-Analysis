"""Read-only closed-loop projection of consistent plan adherence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from random import Random

from .durability import supported_long_run_capacity
from .easy_baseline import recency_weighted_easy_distance
from .recovery import athlete_relative_session_load, cumulative_recovery_load
from .training_load import (
    TrainingSession,
    acute_to_prior_weekly_ratio,
    continuous_fatigue_load,
    continuous_distance_rate,
    short_term_density_half_life_days,
    rolling_load,
)
from .weekly_schedule import (
    PLANNING_HORIZON_DAYS,
    PlanningActivity,
    build_weekly_schedule,
    derive_weekly_target,
)
from .web.schemas import (
    ActivityHealthTag,
    ConfidenceLevel,
    CurrentHealthStatus,
    FitnessState,
    LoadContext,
    LoadWindow,
    ReadinessFlag,
    RecommendationRequest,
    RecommendationResponse,
    RunSummary,
    SessionDifficulty,
    TrailingDayActivity,
    WorkoutType,
    ZoneBreakdown,
)


QUALITY_TYPES = {
    WorkoutType.INTERVALS,
    WorkoutType.TEMPO_THRESHOLD,
    WorkoutType.RACE,
}


@dataclass(frozen=True, slots=True)
class ProjectionRun:
    start_time: datetime
    distance_miles: float
    moving_minutes: float
    workout_type: WorkoutType
    difficulty: SessionDifficulty
    projected: bool = False
    planning_role: str | None = None
    prescribed_low_miles: float | None = None
    prescribed_high_miles: float | None = None
    prescription_title: str | None = None
    adherence_note: str | None = None


@dataclass(frozen=True, slots=True)
class HumanAdherenceProfile:
    """Bounded, reproducible deviations for planner stress testing.

    These are scenario assumptions rather than coaching thresholds. The
    defaults create occasional ordinary human noise without modeling illness,
    injury, or deliberately reckless training.
    """

    seed: int = 20260902
    skip_probability: float = 0.08
    distance_variation_probability: float = 0.35
    minimum_distance_variation_fraction: float = 0.08
    maximum_distance_variation_fraction: float = 0.18
    intensity_drift_probability: float = 0.10
    substitution_probability: float = 0.12
    unscheduled_easy_probability_per_day: float = 0.08
    short_break_count_min: int = 2
    short_break_count_max: int = 3
    short_break_days_min: int = 2
    short_break_days_max: int = 3
    vacation_probability: float = 0.50
    vacation_days: int = 7


@dataclass(frozen=True, slots=True)
class ProjectionWeek:
    week: int
    start_date: date
    target_low_miles: float
    target_high_miles: float
    prescribed_low_miles: float
    prescribed_high_miles: float
    assumed_completed_miles: float
    run_count: int
    capacity_reference_miles: float
    opening_acute_ratio: float | None
    workouts: tuple[str, ...]
    peak_rolling_7d_miles: float = 0.0
    maximum_consecutive_run_days: int = 0
    replan_trace: tuple[str, ...] = ()
    planned_run_count: int = 0
    skipped_run_count: int = 0
    unscheduled_run_count: int = 0
    adherence_events: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _HumanBreak:
    start_date: date
    end_date: date
    label: str

    def contains(self, value: date) -> bool:
        return self.start_date <= value < self.end_date


def runs_from_summaries(values: list[RunSummary]) -> list[ProjectionRun]:
    """Convert recorded run summaries into the simulator's compact history."""

    result: list[ProjectionRun] = []
    for run in values:
        if (
            run.start_time is None
            or run.workout_type in {WorkoutType.HIKE, WorkoutType.BIKE}
        ):
            continue
        moving = run.moving_minutes or max(1.0, run.distance_miles * 11.0)
        workout = (
            WorkoutType.EASY
            if run.workout_type == WorkoutType.UNKNOWN
            else run.workout_type
        )
        difficulty = run.session_difficulty or SessionDifficulty(
            distance_miles=run.distance_miles,
            moving_minutes=moving,
            elapsed_minutes=moving,
            stopped_minutes=0.0,
            zone_load=None,
            zone_breakdown=ZoneBreakdown(),
            is_long_run=workout == WorkoutType.LONG,
            is_quality_session=workout in QUALITY_TYPES,
            difficulty_flags=["projection_seed_without_difficulty"],
        )
        result.append(
            ProjectionRun(
                start_time=run.start_time,
                distance_miles=run.distance_miles,
                moving_minutes=moving,
                workout_type=workout,
                difficulty=difficulty,
                planning_role=run.prescribed_planning_role,
                prescribed_low_miles=(
                    run.prescribed_distance_range_miles[0]
                    if run.prescribed_distance_range_miles is not None
                    else None
                ),
                prescribed_high_miles=(
                    run.prescribed_distance_range_miles[1]
                    if run.prescribed_distance_range_miles is not None
                    else None
                ),
            )
        )
    return sorted(result, key=lambda item: item.start_time)


def _completed_activities_on_plan_date(
    history: list[ProjectionRun],
    plan_start: datetime,
) -> list[TrailingDayActivity]:
    """Expose already-committed same-day runs to each simulated reload."""

    return [
        TrailingDayActivity(
            activity_id=-(index + 1),
            start_time=run.start_time,
            distance_miles=run.distance_miles,
            workout_type=run.workout_type,
            health_tag=ActivityHealthTag.NORMAL,
        )
        for index, run in enumerate(history)
        if run.start_time <= plan_start
        and run.start_time.astimezone(plan_start.tzinfo).date()
        == plan_start.date()
    ]


def _load_window(runs: list[ProjectionRun], as_of: datetime, days: int) -> LoadWindow:
    load = rolling_load(
        [
            TrainingSession(
                activity_id=index,
                start_time=run.start_time,
                distance_miles=run.distance_miles,
                moving_minutes=run.moving_minutes,
                zone_load=run.difficulty.zone_load,
                hard_minutes=run.difficulty.zone_breakdown.hard_minutes,
            )
            for index, run in enumerate(runs)
        ],
        as_of,
        days,
    )
    return LoadWindow(
        days=days,
        distance_miles=load.distance_miles,
        moving_minutes=load.moving_minutes,
        zone_load=load.zone_load,
        hard_minutes=load.hard_minutes,
        activity_count=load.activity_count,
        zone_load_activity_count=load.zone_load_activity_count,
    )


def _state_at(
    template: FitnessState,
    runs: list[ProjectionRun],
    as_of: datetime,
    capacity_reference: float,
    easy_baseline_half_life_days: float = 84.0,
    fatigue_half_life_days: float = 7.0,
) -> FitnessState:
    completed = [run for run in runs if run.start_time <= as_of]
    recent_7 = _load_window(completed, as_of, 7)
    recent_14 = _load_window(completed, as_of, 14)
    recent_28 = _load_window(completed, as_of, 28)
    training_sessions = [
        TrainingSession(
            activity_id=index,
            start_time=run.start_time,
            distance_miles=run.distance_miles,
            moving_minutes=run.moving_minutes,
            zone_load=run.difficulty.zone_load,
            hard_minutes=run.difficulty.zone_breakdown.hard_minutes,
        )
        for index, run in enumerate(completed)
    ]
    acute_ratio = acute_to_prior_weekly_ratio(training_sessions, as_of)
    continuous_fatigue = continuous_fatigue_load(
        training_sessions,
        as_of,
        half_life_days=fatigue_half_life_days,
    )
    last = completed[-1] if completed else None
    recovery_residual_load = cumulative_recovery_load(
        (
            (
                athlete_relative_session_load(
                    run.difficulty,
                    recent_28,
                    performance_anomaly=(
                        template.recent_performance_anomaly
                        if run is last and not run.projected
                        else "within_recent_range"
                    ),
                    drift_percent=(
                        template.last_run_drift_percent
                        if run is last and not run.projected
                        else None
                    ),
                )[0],
                max(0.0, (as_of - run.start_time).total_seconds() / 3600.0),
            )
            for run in completed
        )
    )

    def days_since(predicate) -> float | None:
        matches = [run for run in completed if predicate(run)]
        return (
            max(0.0, (as_of - matches[-1].start_time).total_seconds() / 86400)
            if matches
            else None
        )

    recent_14_runs = [
        run for run in completed
        if as_of - timedelta(days=14) < run.start_time <= as_of
    ]
    known_minutes = sum(
        run.difficulty.zone_breakdown.easy_minutes
        + run.difficulty.zone_breakdown.moderate_minutes
        + run.difficulty.zone_breakdown.hard_minutes
        for run in recent_14_runs
    )
    moderate_minutes = sum(
        run.difficulty.zone_breakdown.moderate_minutes for run in recent_14_runs
    )
    hard_minutes = sum(
        run.difficulty.zone_breakdown.hard_minutes for run in recent_14_runs
    )
    recent_30 = [
        run for run in completed
        if as_of - timedelta(days=30) < run.start_time <= as_of
    ]
    coaching_long_capacity = supported_long_run_capacity(
        (
            (
                run.start_time,
                run.distance_miles,
                run.prescribed_high_miles,
            )
            for run in completed
        ),
        as_of,
    )
    recent_28_runs = [
        run for run in completed
        if as_of - timedelta(days=28) < run.start_time <= as_of
    ]
    ordinary_easy_samples = [
        (run.start_time, run.distance_miles)
        for run in completed
        if run.workout_type == WorkoutType.EASY
        and run.planning_role != "support_easy"
        and run.distance_miles > 0
    ]
    typical_easy_run_miles = recency_weighted_easy_distance(
        ordinary_easy_samples,
        as_of,
        half_life_days=easy_baseline_half_life_days,
    )
    prior_28 = [
        run for run in completed
        if as_of - timedelta(days=35) < run.start_time <= as_of - timedelta(days=7)
    ]
    prior_weekly_miles = sum(run.distance_miles for run in prior_28) / 4.0
    distance_ratio = (
        recent_7.distance_miles / capacity_reference
        if capacity_reference > 0
        else None
    )
    return template.model_copy(
        update={
            "as_of": as_of,
            "recent_load": LoadContext(
                trailing_7d=recent_7,
                trailing_14d=recent_14,
                trailing_28d=recent_28,
                acute_to_prior_ratio=acute_ratio,
                acute_distance_to_capacity_ratio=distance_ratio,
                continuous_fatigue_miles=(
                    continuous_fatigue.equivalent_weekly_miles
                ),
                continuous_fatigue_to_capacity_ratio=(
                    continuous_fatigue.equivalent_weekly_miles
                    / capacity_reference
                    if capacity_reference > 0
                    else None
                ),
                continuous_distance_miles=continuous_distance_rate(
                    training_sessions,
                    as_of,
                    half_life_days=fatigue_half_life_days,
                ),
                continuous_short_term_distance_miles=continuous_distance_rate(
                    training_sessions,
                    as_of,
                    half_life_days=short_term_density_half_life_days(
                        fatigue_half_life_days
                    ),
                ),
                prior_28d_weekly_miles=prior_weekly_miles,
                sustained_capacity_miles=capacity_reference,
                capacity_reference_miles=capacity_reference,
                confidence=ConfidenceLevel.MODERATE,
                flags=["adherence_projection"],
            ),
            "days_since_last_run": (
                max(0.0, (as_of - last.start_time).total_seconds() / 86400)
                if last
                else None
            ),
            "days_since_quality_run": days_since(
                lambda run: run.workout_type in QUALITY_TYPES
            ),
            "days_since_long_run": days_since(
                lambda run: run.workout_type == WorkoutType.LONG
            ),
            "last_run": last.difficulty if last else None,
            "last_run_workout_type": last.workout_type if last else None,
            "last_run_drift_percent": None,
            "recovery_residual_load": recovery_residual_load,
            "longest_run_30d_miles": supported_long_run_capacity(
                (
                    (
                        run.start_time,
                        run.distance_miles,
                        run.prescribed_high_miles,
                    )
                    for run in recent_30
                ),
                as_of,
            ),
            "retained_long_run_capacity_miles": coaching_long_capacity,
            "quality_sessions_14d": sum(
                run.workout_type in QUALITY_TYPES for run in recent_14_runs
            ),
            "completed_quality_session_count": sum(
                run.workout_type in QUALITY_TYPES for run in completed
            ),
            "running_days_28d": len(
                {run.start_time.date() for run in recent_28_runs}
            ),
            "typical_easy_run_miles": (
                typical_easy_run_miles
                if typical_easy_run_miles is not None
                else template.typical_easy_run_miles
            ),
            "easy_fraction_14d": (
                max(0.0, 1.0 - (moderate_minutes + hard_minutes) / known_minutes)
                if known_minutes > 0
                else None
            ),
            "moderate_fraction_14d": (
                moderate_minutes / known_minutes if known_minutes > 0 else None
            ),
            "moderate_evidence_runs_14d": sum(
                run.difficulty.zone_breakdown.moderate_minutes > 0
                for run in recent_14_runs
            ),
            "hard_fraction_14d": (
                hard_minutes / known_minutes if known_minutes > 0 else None
            ),
            "recent_performance_anomaly": "within_recent_range",
            "recent_illness_or_recovery": False,
            "normal_runs_since_health_event": 0,
            "current_health_status": CurrentHealthStatus.NORMAL,
            "anomaly_flags": [],
            "planned_weather": None,
        }
    )


def _projected_difficulty(
    recommendation: RecommendationResponse,
    miles: float,
    pace_min_mile: float,
) -> SessionDifficulty:
    minutes = miles * pace_min_mile
    workout_type = recommendation.workout_type
    if workout_type in QUALITY_TYPES:
        moderate = hard = 0.0
        for step in recommendation.structure:
            zones = " ".join(step.target_zones).casefold()
            if step.repetitions and step.work_duration_minutes:
                work = step.repetitions * step.work_duration_minutes
            elif step.repetitions and step.work_duration_range_minutes:
                low, high = step.work_duration_range_minutes
                work = step.repetitions * ((low + high) / 2)
            elif step.phase == "work" and step.duration_minutes:
                work = step.duration_minutes
            elif step.phase == "work":
                # A distance-defined progression has no fixed clock dose. The
                # final quarter is the best machine-readable approximation of
                # its controlled Z3 finish; the opening progression remains
                # aerobic rather than being mislabeled as hard work.
                work = minutes * 0.25
            else:
                continue
            if any(marker in zones for marker in ("z4", "z5", "strong")):
                if "z3" in zones or "threshold" in zones:
                    moderate += work * 0.60
                    hard += work * 0.40
                else:
                    hard += work
            elif "z3" in zones or "threshold" in zones:
                moderate += work
        quality_minutes = min(minutes, moderate + hard)
        easy = max(0.0, minutes - quality_minutes)
    else:
        easy, moderate, hard = minutes, 0.0, 0.0
    known = max(1.0, easy + moderate + hard)
    return SessionDifficulty(
        distance_miles=miles,
        moving_minutes=minutes,
        elapsed_minutes=minutes,
        stopped_minutes=0.0,
        # Future HR is intentionally unknown. Duration supplies rolling-load
        # continuity, while prescribed zone fractions support workout balance.
        zone_load=None,
        perceived_exertion=None,
        session_rpe_load=None,
        zone_breakdown=ZoneBreakdown(
            zone_seconds={
                "z2": easy * 60,
                "z3": moderate * 60,
                "z4": hard * 60,
            },
            zone_fractions={
                "z2": easy / known,
                "z3": moderate / known,
                "z4": hard / known,
            },
            easy_minutes=easy,
            moderate_minutes=moderate,
            hard_minutes=hard,
        ),
        is_long_run=workout_type == WorkoutType.LONG,
        is_quality_session=workout_type in QUALITY_TYPES,
        difficulty_flags=["adherence_projection_no_observed_response"],
    )


def _human_breaks(
    profile: HumanAdherenceProfile,
    start_date: date,
    total_days: int,
) -> tuple[_HumanBreak, ...]:
    """Generate separated surprise trips without consulting the planner."""

    if total_days < 14:
        return ()
    rng = Random(profile.seed ^ 0xB4EA)
    requested: list[tuple[int, str]] = []
    short_low = max(0, profile.short_break_count_min)
    short_high = max(short_low, profile.short_break_count_max)
    for index in range(rng.randint(short_low, short_high)):
        requested.append(
            (
                rng.randint(
                    max(1, profile.short_break_days_min),
                    max(
                        max(1, profile.short_break_days_min),
                        profile.short_break_days_max,
                    ),
                ),
                f"trip {index + 1}",
            )
        )
    if rng.random() < min(1.0, max(0.0, profile.vacation_probability)):
        requested.append((max(1, profile.vacation_days), "vacation"))

    placed: list[_HumanBreak] = []
    for length, label in sorted(requested, reverse=True):
        latest_start = max(5, total_days - length - 5)
        candidates = list(range(5, latest_start + 1))
        rng.shuffle(candidates)
        for offset in candidates:
            candidate = _HumanBreak(
                start_date=start_date + timedelta(days=offset),
                end_date=start_date + timedelta(days=offset + length),
                label=label,
            )
            if any(
                candidate.start_date < existing.end_date + timedelta(days=2)
                and existing.start_date < candidate.end_date + timedelta(days=2)
                for existing in placed
            ):
                continue
            placed.append(candidate)
            break
    return tuple(sorted(placed, key=lambda item: item.start_date))


def _observed_human_difficulty(
    recommendation: RecommendationResponse,
    miles: float,
    pace_min_mile: float,
    actual_type: WorkoutType,
    *,
    too_intense: bool,
) -> SessionDifficulty:
    """Create plausible observed HR evidence for a deviated workout."""

    minutes = miles * pace_min_mile
    if actual_type in QUALITY_TYPES:
        shaped = _projected_difficulty(
            recommendation.model_copy(update={"workout_type": actual_type}),
            miles,
            pace_min_mile,
        )
        moderate = shaped.zone_breakdown.moderate_minutes
        hard = shaped.zone_breakdown.hard_minutes
        if moderate + hard <= 0:
            moderate = minutes * 0.18
            hard = minutes * 0.07
        easy = max(0.0, minutes - moderate - hard)
    else:
        easy, moderate, hard = minutes, 0.0, 0.0
    if too_intense:
        spill = min(easy, minutes * 0.20)
        easy -= spill
        moderate += spill * 0.75
        hard += spill * 0.25
    known = max(1.0, easy + moderate + hard)
    return SessionDifficulty(
        distance_miles=miles,
        moving_minutes=minutes,
        elapsed_minutes=minutes,
        stopped_minutes=0.0,
        zone_load=easy * 2.0 + moderate * 3.0 + hard * 4.0,
        perceived_exertion=None,
        session_rpe_load=None,
        zone_breakdown=ZoneBreakdown(
            zone_seconds={
                "z2": easy * 60,
                "z3": moderate * 60,
                "z4": hard * 60,
            },
            zone_fractions={
                "z2": easy / known,
                "z3": moderate / known,
                "z4": hard / known,
            },
            easy_minutes=easy,
            moderate_minutes=moderate,
            hard_minutes=hard,
        ),
        is_long_run=actual_type == WorkoutType.LONG,
        is_quality_session=actual_type in QUALITY_TYPES,
        difficulty_flags=[
            "human_adherence_observed_hr",
            *(["human_intensity_overshoot"] if too_intense else []),
        ],
    )


def simulate_adherence(
    template: FitnessState,
    recorded_runs: list[ProjectionRun],
    config: dict,
    start_at: datetime,
    *,
    weeks: int = 13,
    replan_interval_days: int = 1,
    human_profile: HumanAdherenceProfile | None = None,
) -> list[ProjectionWeek]:
    """Roll the real planner forward under perfect or bounded human adherence.

    The simulator projects scheduling and training load only. It assumes a
    normal response and does not invent future pace, HR, sleep, soreness,
    illness, injury, weather, missed sessions, or fitness improvement.

    Daily replanning is the production-fidelity default: only work scheduled
    before the next reload is committed. Longer intervals are an explicit
    coarse-test mode and must not be used to draw calendar-spacing conclusions.
    """

    if not 1 <= replan_interval_days <= 7:
        raise ValueError("replan_interval_days must be between 1 and 7")

    history = list(recorded_runs)
    summaries: dict[int, dict] = {}
    commitments: dict[date, tuple[datetime, float, float]] = {}
    adherence_events: list[tuple[datetime, str, str]] = []
    behavior_rng = Random(human_profile.seed) if human_profile else None
    candidate_hours = sorted(
        int(value)
        for value in config.get("weather", {}).get(
            "automatic_run_time_hours_local", [7, 12, 19]
        )
    )
    default_hour = candidate_hours[0] if candidate_hours else 7
    total_days = max(1, weeks) * 7
    simulation_start_date = start_at.date()
    breaks = (
        _human_breaks(human_profile, simulation_start_date, total_days)
        if human_profile
        else ()
    )
    for pause in breaks:
        adherence_events.append(
            (
                datetime.combine(
                    pause.start_date, time.min, tzinfo=start_at.tzinfo
                ),
                "break",
                f"{pause.label}: {pause.start_date} through "
                f"{pause.end_date - timedelta(days=1)}",
            )
        )
    simulation_end_at = datetime.combine(
        simulation_start_date + timedelta(days=total_days),
        time.min,
        tzinfo=start_at.tzinfo,
    )
    previous_schedule = None
    for plan_offset in range(0, total_days, replan_interval_days):
        plan_start = start_at + timedelta(days=plan_offset)
        commit_end = min(
            plan_start + timedelta(days=replan_interval_days),
            simulation_end_at,
        )
        week_index = min(weeks - 1, plan_offset // 7)
        activities = [
            PlanningActivity(run.start_time, run.distance_miles) for run in history
        ]
        target_runs, target_range, evidence = derive_weekly_target(
            activities, plan_start, config
        )
        capacity = evidence.capacity_reference_miles
        daily_states: list[FitnessState] = []
        for offset in range(PLANNING_HORIZON_DAYS):
            day = (plan_start + timedelta(days=offset)).date()
            hour = default_hour
            if offset == 0 and plan_offset == 0:
                future_hours = [value for value in candidate_hours if value > start_at.hour]
                hour = future_hours[0] if future_hours else start_at.hour
            planned_at = datetime.combine(day, time(hour), tzinfo=start_at.tzinfo)
            if planned_at < plan_start:
                planned_at = plan_start
            daily_states.append(
                _state_at(
                    template,
                    history,
                    planned_at,
                    capacity,
                    easy_baseline_half_life_days=float(
                        config.get("coaching", {}).get(
                            "capacity_retention_half_life_days", 84
                        )
                    ),
                    fatigue_half_life_days=float(
                        config.get("coaching", {}).get(
                            "continuous_fatigue_half_life_days", 7
                        )
                    ),
                )
            )
        schedule = build_weekly_schedule(
            daily_states,
            RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
            config,
            target_run_count=target_runs,
            target_distance_range=target_range,
            target_evidence=evidence,
            completed_activities_by_offset={
                0: _completed_activities_on_plan_date(history, plan_start)
            },
            prior_schedule=previous_schedule,
        )
        previous_schedule = schedule
        committed = [
            day.recommendation
            for day in schedule.days
            if day.recommendation
            and day.recommendation.workout_type != WorkoutType.REST
            and day.recommendation.planned_for
            and plan_start <= day.recommendation.planned_for < commit_end
        ]
        pace_window = _load_window(history, plan_start, 28)
        pace = (
            pace_window.moving_minutes / pace_window.distance_miles
            if pace_window.distance_miles > 0
            else 11.0
        )
        pace = min(15.0, max(7.0, pace))
        opening = (
            daily_states[0].recent_load.continuous_fatigue_to_capacity_ratio
            if daily_states[0].recent_load.continuous_fatigue_to_capacity_ratio
            is not None
            else daily_states[0].recent_load.acute_distance_to_capacity_ratio
        )
        summary = summaries.setdefault(
            week_index,
            {
                "replans": [],
            },
        )
        visible_plan = [
            day.recommendation
            for day in (schedule.planning_days or schedule.days)
            if day.recommendation
            and day.recommendation.workout_type != WorkoutType.REST
            and day.recommendation.planned_for
        ]
        summary["replans"].append(
            f"{plan_start.strftime('%a %b %-d')}: "
            f"opening {opening * 100:.0f}%" if opening is not None else
            f"{plan_start.strftime('%a %b %-d')}: opening —"
        )
        summary["replans"][-1] += (
            f"; target {target_range[0]:.1f}–{target_range[1]:.1f}; horizon "
            + ", ".join(
                f"{item.planned_for.strftime('%a')} {item.workout_type.value} "
                f"{sum(item.distance_range_miles) / 2:.1f}"
                for item in visible_plan
                if item.planned_for and item.distance_range_miles
            )
            + "; committed "
            + (
                ", ".join(
                    f"{item.planned_for.strftime('%a')} {item.workout_type.value}"
                    for item in committed
                    if item.planned_for
                )
                or "none"
            )
        )
        for item in committed:
            if not item.distance_range_miles or not item.planned_for:
                continue
            commitments[item.planned_for.date()] = (
                item.planned_for,
                item.distance_range_miles[0],
                item.distance_range_miles[1],
            )
            active_break = next(
                (
                    pause
                    for pause in breaks
                    if pause.contains(item.planned_for.date())
                ),
                None,
            )
            if active_break is not None:
                adherence_events.append(
                    (
                        item.planned_for,
                        "skip",
                        f"missed scheduled {item.workout_type.value} "
                        f"during {active_break.label}",
                    )
                )
                continue
            if (
                behavior_rng is not None
                and behavior_rng.random() < human_profile.skip_probability
            ):
                adherence_events.append(
                    (
                        item.planned_for,
                        "skip",
                        f"skipped scheduled {item.workout_type.value}",
                    )
                )
                continue
            miles = sum(item.distance_range_miles) / 2.0
            notes: list[str] = []
            if (
                behavior_rng is not None
                and behavior_rng.random()
                < human_profile.distance_variation_probability
            ):
                lower = max(
                    0.0, human_profile.minimum_distance_variation_fraction
                )
                upper = max(
                    lower, human_profile.maximum_distance_variation_fraction
                )
                variation = behavior_rng.uniform(lower, upper)
                direction = -1.0 if behavior_rng.random() < 0.5 else 1.0
                miles = max(0.5, miles * (1.0 + direction * variation))
                notes.append("too short" if direction < 0 else "too long")
            actual_type = item.workout_type
            if (
                behavior_rng is not None
                and behavior_rng.random() < human_profile.substitution_probability
            ):
                if item.workout_type in QUALITY_TYPES:
                    actual_type = WorkoutType.EASY
                elif item.workout_type in {
                    WorkoutType.EASY,
                    WorkoutType.RECOVERY,
                }:
                    actual_type = WorkoutType.TEMPO_THRESHOLD
                if actual_type != item.workout_type:
                    notes.append(
                        f"substituted {actual_type.value} for "
                        f"{item.workout_type.value}"
                    )
            too_intense = bool(
                behavior_rng is not None
                and behavior_rng.random()
                < human_profile.intensity_drift_probability
            )
            if too_intense:
                notes.append("more intense than prescribed")
            adherence_note = "; ".join(notes) or "within prescribed margin"
            if notes:
                adherence_events.append(
                    (item.planned_for, "deviation", adherence_note)
                )
            history.append(
                ProjectionRun(
                    start_time=item.planned_for,
                    distance_miles=miles,
                    moving_minutes=miles * pace,
                    workout_type=actual_type,
                    difficulty=(
                        _observed_human_difficulty(
                            item,
                            miles,
                            pace,
                            actual_type,
                            too_intense=too_intense,
                        )
                        if human_profile
                        else _projected_difficulty(item, miles, pace)
                    ),
                    projected=True,
                    planning_role=(
                        "quality"
                        if actual_type in QUALITY_TYPES
                        else "ordinary_easy"
                        if actual_type == WorkoutType.EASY
                        else item.planning_role
                    ),
                    prescribed_low_miles=item.distance_range_miles[0],
                    prescribed_high_miles=item.distance_range_miles[1],
                    prescription_title=item.title,
                    adherence_note=(adherence_note if human_profile else None),
                )
            )
        if (
            human_profile
            and behavior_rng is not None
            and not any(
                item.planned_for
                and item.planned_for.date() == plan_start.date()
                for item in committed
            )
            and behavior_rng.random()
            < human_profile.unscheduled_easy_probability_per_day
        ):
            unscheduled_at = datetime.combine(
                plan_start.date(), time(19), tzinfo=start_at.tzinfo
            )
            if unscheduled_at <= plan_start:
                unscheduled_at += timedelta(days=1)
            active_break = next(
                (
                    pause
                    for pause in breaks
                    if pause.contains(unscheduled_at.date())
                ),
                None,
            )
            if unscheduled_at < commit_end and active_break is None:
                ordinary = (
                    _state_at(
                        template,
                        history,
                        unscheduled_at,
                        capacity,
                    ).typical_easy_run_miles
                    or 3.0
                )
                miles = max(1.0, ordinary * behavior_rng.uniform(0.60, 0.90))
                synthetic = RecommendationResponse(
                    generated_at=plan_start,
                    fitness_state_as_of=plan_start,
                    planned_for=unscheduled_at,
                    workout_type=WorkoutType.EASY,
                    planning_role="support_easy",
                    title="Unscheduled easy run",
                    distance_range_miles=(miles, miles),
                    confidence=ConfidenceLevel.MODERATE,
                    readiness=ReadinessFlag.READY,
                )
                too_intense = (
                    behavior_rng.random()
                    < human_profile.intensity_drift_probability
                )
                history.append(
                    ProjectionRun(
                        start_time=unscheduled_at,
                        distance_miles=miles,
                        moving_minutes=miles * pace,
                        workout_type=WorkoutType.EASY,
                        difficulty=_observed_human_difficulty(
                            synthetic,
                            miles,
                            pace,
                            WorkoutType.EASY,
                            too_intense=too_intense,
                        ),
                        projected=True,
                        planning_role="support_easy",
                        prescription_title="Unscheduled easy run",
                        adherence_note="unscheduled easy run",
                    )
                )
                adherence_events.append(
                    (
                        unscheduled_at,
                        "unscheduled",
                        f"unscheduled easy {miles:.1f} mi"
                        + (" with intensity drift" if too_intense else ""),
                    )
                )
        history.sort(key=lambda item: item.start_time)

    projected_dates = sorted(
        {
            run.start_time.date()
            for run in history
            if run.projected
            and simulation_start_date
            <= run.start_time.date()
            < simulation_start_date + timedelta(days=total_days)
        }
    )
    streaks: list[set[date]] = []
    current_streak: set[date] = set()
    previous_date: date | None = None
    for run_date in projected_dates:
        if previous_date is None or run_date != previous_date + timedelta(days=1):
            if current_streak:
                streaks.append(current_streak)
            current_streak = set()
        current_streak.add(run_date)
        previous_date = run_date
    if current_streak:
        streaks.append(current_streak)

    results: list[ProjectionWeek] = []
    for week_index in range(weeks):
        week_start_date = simulation_start_date + timedelta(days=week_index * 7)
        week_end_date = week_start_date + timedelta(days=7)
        week_start = (
            start_at
            if week_index == 0
            else datetime.combine(
                week_start_date,
                time.min,
                tzinfo=start_at.tzinfo,
            )
        )
        week_end = datetime.combine(
            week_end_date,
            time.min,
            tzinfo=start_at.tzinfo,
        )
        history_at_open = [run for run in history if run.start_time <= week_start]
        _, target_range, evidence = derive_weekly_target(
            [
                PlanningActivity(run.start_time, run.distance_miles)
                for run in history_at_open
            ],
            week_start,
            config,
        )
        opening_state = _state_at(
            template,
            history_at_open,
            week_start,
            evidence.capacity_reference_miles,
            easy_baseline_half_life_days=float(
                config.get("coaching", {}).get(
                    "capacity_retention_half_life_days", 84
                )
            ),
            fatigue_half_life_days=float(
                config.get("coaching", {}).get(
                    "continuous_fatigue_half_life_days", 7
                )
            ),
        )
        opening = (
            opening_state.recent_load.continuous_fatigue_to_capacity_ratio
            if opening_state.recent_load.continuous_fatigue_to_capacity_ratio
            is not None
            else opening_state.recent_load.acute_distance_to_capacity_ratio
        )
        projected_runs = sorted(
            (
                run
                for run in history
                if run.projected
                and week_start_date <= run.start_time.date() < week_end_date
            ),
            key=lambda run: run.start_time,
        )
        evaluation_times = [
            week_start + timedelta(days=offset) for offset in range(8)
        ] + [
            run.start_time
            for run in history
            if week_start <= run.start_time <= week_end
        ]
        peak_rolling = max(
            (_load_window(history, at, 7).distance_miles for at in evaluation_times),
            default=0.0,
        )
        week_dates = {
            week_start_date + timedelta(days=offset) for offset in range(7)
        }
        longest_streak = max(
            (
                len(streak)
                for streak in streaks
                if streak & week_dates
            ),
            default=0,
        )
        week_commitments = [
            item
            for item in commitments.values()
            if week_start_date <= item[0].date() < week_end_date
        ]
        prescribed_low = sum(item[1] for item in week_commitments)
        prescribed_high = sum(item[2] for item in week_commitments)
        week_events = [
            item
            for item in adherence_events
            if week_start_date <= item[0].date() < week_end_date
        ]
        summary = summaries.get(week_index, {"replans": []})
        results.append(
            ProjectionWeek(
            week=week_index + 1,
            start_date=week_start_date,
            target_low_miles=target_range[0],
            target_high_miles=target_range[1],
            prescribed_low_miles=round(prescribed_low, 2),
            prescribed_high_miles=round(prescribed_high, 2),
            assumed_completed_miles=sum(
                run.distance_miles for run in projected_runs
            ),
            run_count=len(projected_runs),
            capacity_reference_miles=evidence.capacity_reference_miles,
            opening_acute_ratio=opening,
            workouts=tuple(
                f"{run.start_time.strftime('%a')} {run.workout_type.value} "
                f"[{run.prescription_title or run.workout_type.value}] "
                f"{run.distance_miles:.1f}"
                + (
                    f" ({run.adherence_note})"
                    if run.adherence_note
                    and run.adherence_note != "within prescribed margin"
                    else ""
                )
                for run in projected_runs
            ),
            peak_rolling_7d_miles=peak_rolling,
            maximum_consecutive_run_days=longest_streak,
            replan_trace=tuple(summary["replans"]),
            planned_run_count=len(week_commitments),
            skipped_run_count=len(
                {
                    at.date()
                    for at, kind, _ in week_events
                    if kind == "skip"
                }
                - {run.start_time.date() for run in projected_runs}
            ),
            unscheduled_run_count=sum(
                kind == "unscheduled" for _, kind, _ in week_events
            ),
            adherence_events=tuple(
                dict.fromkeys(
                    f"{at.strftime('%a %b %-d')}: {message}"
                    for at, _, message in week_events
                )
            ),
        )
        )
    return results
