"""Read-only closed-loop projection of consistent plan adherence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

from .durability import retained_long_run_capacity
from .training_load import (
    TrainingSession,
    acute_to_prior_weekly_ratio,
    rolling_load,
)
from .weekly_schedule import (
    PLANNING_HORIZON_DAYS,
    PlanningActivity,
    build_weekly_schedule,
    derive_weekly_target,
)
from .web.schemas import (
    ConfidenceLevel,
    CurrentHealthStatus,
    FitnessState,
    LoadContext,
    LoadWindow,
    RecommendationRequest,
    RunSummary,
    SessionDifficulty,
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
            )
        )
    return sorted(result, key=lambda item: item.start_time)


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
    )


def _state_at(
    template: FitnessState,
    runs: list[ProjectionRun],
    as_of: datetime,
    capacity_reference: float,
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
    last = completed[-1] if completed else None

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
    coaching_long_capacity = retained_long_run_capacity(
        ((run.start_time, run.distance_miles) for run in completed),
        as_of,
    )
    recent_28_runs = [
        run for run in completed
        if as_of - timedelta(days=28) < run.start_time <= as_of
    ]
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
            "longest_run_30d_miles": max(
                (run.distance_miles for run in recent_30), default=0.0
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
    workout_type: WorkoutType,
    miles: float,
    pace_min_mile: float,
) -> SessionDifficulty:
    minutes = miles * pace_min_mile
    if workout_type in QUALITY_TYPES:
        easy, moderate, hard = minutes * 0.65, minutes * 0.25, minutes * 0.10
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


def simulate_adherence(
    template: FitnessState,
    recorded_runs: list[ProjectionRun],
    config: dict,
    start_at: datetime,
    *,
    weeks: int = 13,
) -> list[ProjectionWeek]:
    """Roll the real weekly planner forward under midpoint adherence.

    The simulator projects scheduling and training load only. It assumes a
    normal response and does not invent future pace, HR, sleep, soreness,
    illness, injury, weather, missed sessions, or fitness improvement.
    """

    history = list(recorded_runs)
    results: list[ProjectionWeek] = []
    candidate_hours = sorted(
        int(value)
        for value in config.get("weather", {}).get(
            "automatic_run_time_hours_local", [7, 12, 19]
        )
    )
    default_hour = candidate_hours[0] if candidate_hours else 7
    for week_index in range(weeks):
        week_start = start_at + timedelta(days=7 * week_index)
        activities = [
            PlanningActivity(run.start_time, run.distance_miles) for run in history
        ]
        target_runs, target_range, evidence = derive_weekly_target(
            activities, week_start, config
        )
        capacity = evidence.capacity_reference_miles
        daily_states: list[FitnessState] = []
        for offset in range(PLANNING_HORIZON_DAYS):
            day = (week_start + timedelta(days=offset)).date()
            hour = default_hour
            if offset == 0 and week_index == 0:
                future_hours = [value for value in candidate_hours if value > start_at.hour]
                hour = future_hours[0] if future_hours else start_at.hour
            planned_at = datetime.combine(day, time(hour), tzinfo=start_at.tzinfo)
            if planned_at < week_start:
                planned_at = week_start
            daily_states.append(
                _state_at(template, history, planned_at, capacity)
            )
        schedule = build_weekly_schedule(
            daily_states,
            RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
            config,
            target_run_count=target_runs,
            target_distance_range=target_range,
            target_evidence=evidence,
        )
        prescribed = [
            day.recommendation for day in schedule.days if day.recommendation
        ]
        pace_window = _load_window(history, week_start, 28)
        pace = (
            pace_window.moving_minutes / pace_window.distance_miles
            if pace_window.distance_miles > 0
            else 11.0
        )
        pace = min(15.0, max(7.0, pace))
        assumed_total = 0.0
        workout_labels: list[str] = []
        for item in prescribed:
            if not item.distance_range_miles or not item.planned_for:
                continue
            miles = sum(item.distance_range_miles) / 2.0
            assumed_total += miles
            workout_labels.append(
                f"{item.planned_for.strftime('%a')} {item.workout_type.value} {miles:.1f}"
            )
            history.append(
                ProjectionRun(
                    start_time=item.planned_for,
                    distance_miles=miles,
                    moving_minutes=miles * pace,
                    workout_type=item.workout_type,
                    difficulty=_projected_difficulty(
                        item.workout_type, miles, pace
                    ),
                    projected=True,
                )
            )
        history.sort(key=lambda item: item.start_time)
        opening = daily_states[0].recent_load.acute_distance_to_capacity_ratio
        results.append(
            ProjectionWeek(
                week=week_index + 1,
                start_date=schedule.start_date,
                target_low_miles=target_range[0],
                target_high_miles=target_range[1],
                prescribed_low_miles=schedule.projected_distance_range_miles[0],
                prescribed_high_miles=schedule.projected_distance_range_miles[1],
                assumed_completed_miles=assumed_total,
                run_count=len(prescribed),
                capacity_reference_miles=capacity,
                opening_acute_ratio=opening,
                workouts=tuple(workout_labels),
            )
        )
    return results
