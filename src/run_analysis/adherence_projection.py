"""Read-only closed-loop projection of consistent plan adherence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from enum import Enum
from random import Random
from time import perf_counter
from typing import Any, Callable

from .durability import supported_long_run_capacity
from .easy_baseline import (
    ordinary_easy_sample_distance,
    recency_weighted_easy_distance,
)
from .recovery import (
    athlete_relative_session_load,
    cumulative_recovery_load,
    ordinary_session_reference,
)
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
    adherence_normalized_distance,
    build_weekly_schedule,
    derive_weekly_target,
    make_expected_target_projector,
)
from .web.schemas import (
    ActivityHealthTag,
    ConfidenceLevel,
    CurrentHealthStatus,
    FitnessState,
    LoadContext,
    LoadWindow,
    QualitySessionType,
    ReadinessFlag,
    RecommendationRequest,
    RecommendationResponse,
    RunSummary,
    SessionDifficulty,
    TrailingDayActivity,
    WorkoutType,
    WeeklyScheduleResponse,
    ZoneBreakdown,
)


QUALITY_TYPES = {
    WorkoutType.INTERVALS,
    WorkoutType.TEMPO_THRESHOLD,
    WorkoutType.RACE,
}


def _completed_planning_role(
    recommendation: RecommendationResponse,
    actual_type: WorkoutType,
) -> str | None:
    """Preserve the purpose of a completed projected aerobic session.

    A medium-long or support run is still that kind of evidence after it is
    completed. Support work must stay identifiable so it cannot shrink the
    ordinary-run reference; medium-long work retains its program semantics
    even though its completed aerobic distance can stabilize the broader
    session-scale baseline.
    """

    if actual_type in QUALITY_TYPES:
        return "quality"
    if actual_type == WorkoutType.EASY:
        return recommendation.planning_role or "ordinary_easy"
    return recommendation.planning_role


def _consecutive_date_streaks(run_dates: set[date]) -> list[set[date]]:
    """Group continuous run dates without treating the forecast edge as rest."""

    streaks: list[set[date]] = []
    current_streak: set[date] = set()
    previous_date: date | None = None
    for run_date in sorted(run_dates):
        if previous_date is None or run_date != previous_date + timedelta(days=1):
            if current_streak:
                streaks.append(current_streak)
            current_streak = set()
        current_streak.add(run_date)
        previous_date = run_date
    if current_streak:
        streaks.append(current_streak)
    return streaks


@dataclass(frozen=True, slots=True)
class ProjectionRun:
    start_time: datetime
    distance_miles: float
    moving_minutes: float
    workout_type: WorkoutType
    difficulty: SessionDifficulty
    projected: bool = False
    planning_role: str | None = None
    prescribed_workout_type: WorkoutType | None = None
    completed_prescribed_workout: bool | None = None
    prescribed_low_miles: float | None = None
    prescribed_high_miles: float | None = None
    prescription_title: str | None = None
    quality_session_type: QualitySessionType | None = None
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


class OverloadScenario(str, Enum):
    """Deterministic deviations used to inspect closed-loop load absorption."""

    CONTROL = "control"
    EXTRA_DISTANCE = "extra_distance"
    EXTRA_INTENSITY = "extra_intensity"
    UNSCHEDULED_EASY = "unscheduled_easy"
    DENSE_SEQUENCE = "dense_sequence"


class DeterministicAdherenceScenario(str, Enum):
    """Named single-purpose deviations for fast closed-loop regressions."""

    IN_RANGE_LOW = "in_range_low"
    IN_RANGE_HIGH = "in_range_high"
    BELOW_RANGE_EASY = "below_range_easy"
    ABOVE_RANGE_OR_HARDER = "above_range_or_harder"
    SINGLE_HIDDEN_MISS = "single_hidden_miss"
    KNOWN_FORCED_REST = "known_forced_rest"
    HIDDEN_VACATION = "hidden_vacation"


@dataclass(frozen=True, slots=True)
class DeterministicAdherenceProfile:
    scenario: DeterministicAdherenceScenario
    trigger_scheduled_run: int = 1
    below_range_fraction: float = 0.75
    above_range_easy_fraction: float = 0.50
    block_start_day: int = 5
    block_days: int = 7


@dataclass(frozen=True, slots=True)
class OverloadAdherenceProfile:
    """One athlete-relative overload introduced into an otherwise clean loop.

    These values define stress-test inputs, not coaching thresholds. Distance
    deviations scale from the athlete's ordinary easy session so the same
    scenario remains meaningful for runners with different mileage.
    """

    scenario: OverloadScenario
    trigger_scheduled_run: int = 1
    extra_distance_easy_fraction: float = 0.50
    unscheduled_easy_fraction: float = 0.75
    dense_sequence_additional_days: int = 2


@dataclass(frozen=True, slots=True)
class ProjectionPlanSession:
    planned_for: datetime
    workout_type: WorkoutType
    midpoint_miles: float
    distance_low_miles: float | None = None
    distance_high_miles: float | None = None
    planning_role: str | None = None
    readiness: str | None = None


@dataclass(frozen=True, slots=True)
class ProjectionReplan:
    """Structured daily planner output retained for absorption diagnostics."""

    generated_at: datetime
    opening_load_ratio: float | None
    target_low_miles: float
    target_high_miles: float
    planned_sessions: tuple[ProjectionPlanSession, ...]
    committed_sessions: tuple[ProjectionPlanSession, ...]
    capacity_reference_miles: float | None = None
    planning_mode: str | None = None
    ordinary_easy_midpoint_miles: float | None = None
    boundary_session_miles: float | None = None
    opening_continuous_distance_miles: float | None = None
    opening_short_term_distance_miles: float | None = None
    peak_projected_continuous_mileage_rate: float | None = None
    planning_seconds: float | None = None
    target_trajectory: tuple[tuple[float, float], ...] = ()
    decision_start_date: date | None = None
    planner_diagnostics: dict[str, Any] | None = None
    material_evidence_reasons: tuple[str, ...] = ()
    opening_recovery_residual_load: float | None = None
    expected_opening_recovery_residual_load: float | None = None
    expected_opening_state_at: datetime | None = None
    recovery_surprise_units: float | None = None
    days_since_long_run: float | None = None
    days_since_quality_run: float | None = None
    long_cadence_reference_days: float | None = None
    quality_cadence_reference_days: float | None = None
    cadence_exception_reasons: tuple[str, ...] = ()
    trigger: str = "scheduled_refresh"
    source_activity_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class ProjectionChurnSummary:
    """Changes between consecutive replans inside one fixed future window."""

    horizon_days: int
    comparison_count: int
    schedule_changed_comparisons: int
    distance_only_changed_comparisons: int
    date_slot_changes: int
    workout_type_changes: int
    distance_changes: int


def summarize_replan_churn(
    replans: list[ProjectionReplan] | tuple[ProjectionReplan, ...],
    *,
    horizon_days: int,
    transition_trigger: str | None = None,
) -> ProjectionChurnSummary:
    """Measure future-plan churn without counting already-consumed sessions.

    Each comparison uses the later refresh's next ``horizon_days`` calendar
    dates. The refresh date itself is excluded because a prescribed run may
    have been completed before that refresh; its disappearance is compliance,
    not instability. Date/type changes are kept separate from distance-only
    edits because moving a run is more disruptive than resizing it.
    """

    if horizon_days < 1:
        raise ValueError("horizon_days must be at least one")
    ordered = sorted(replans, key=lambda item: item.generated_at)
    schedule_changed_comparisons = 0
    distance_only_changed_comparisons = 0
    date_slot_changes = 0
    workout_type_changes = 0
    distance_changes = 0
    for previous, current in zip(ordered, ordered[1:]):
        if (
            transition_trigger is not None
            and current.trigger != transition_trigger
        ):
            continue
        window_start = (
            current.decision_start_date
            if current.trigger == "post_upload"
            and current.decision_start_date is not None
            else current.generated_at.date() + timedelta(days=1)
        )
        window_end = window_start + timedelta(days=horizon_days)

        def future_by_date(
            replan: ProjectionReplan,
        ) -> dict[date, ProjectionPlanSession]:
            return {
                session.planned_for.date(): session
                for session in replan.planned_sessions
                if window_start <= session.planned_for.date() < window_end
            }

        before = future_by_date(previous)
        after = future_by_date(current)
        changed_dates = set(before) ^ set(after)
        shared_dates = set(before) & set(after)
        changed_types = {
            value
            for value in shared_dates
            if before[value].workout_type != after[value].workout_type
        }
        changed_distances = {
            value
            for value in shared_dates - changed_types
            if abs(
                before[value].midpoint_miles
                - after[value].midpoint_miles
            )
            > 1e-9
        }
        schedule_changed = bool(changed_dates or changed_types)
        schedule_changed_comparisons += int(schedule_changed)
        distance_only_changed_comparisons += int(
            not schedule_changed and bool(changed_distances)
        )
        date_slot_changes += len(changed_dates)
        workout_type_changes += len(changed_types)
        distance_changes += len(changed_distances)

    return ProjectionChurnSummary(
        horizon_days=horizon_days,
        comparison_count=sum(
            1
            for _, current in zip(ordered, ordered[1:])
            if transition_trigger is None
            or current.trigger == transition_trigger
        ),
        schedule_changed_comparisons=schedule_changed_comparisons,
        distance_only_changed_comparisons=distance_only_changed_comparisons,
        date_slot_changes=date_slot_changes,
        workout_type_changes=workout_type_changes,
        distance_changes=distance_changes,
    )


@dataclass(frozen=True, slots=True)
class ExpectedPolicyRolloutStep:
    """One decision window in a bounded expected-compliance rollout."""

    window_start: datetime
    window_end: datetime
    source_replan: ProjectionReplan
    opening_sessions: tuple[ProjectionPlanSession, ...]
    policy_sessions: tuple[ProjectionPlanSession, ...]

    @property
    def schedule_changed_from_opening(self) -> bool:
        """Whether dates or workout types changed, ignoring distance edits."""

        def signature(
            sessions: tuple[ProjectionPlanSession, ...],
        ) -> tuple[tuple[date, WorkoutType], ...]:
            return tuple(
                (session.planned_for.date(), session.workout_type)
                for session in sessions
            )

        return signature(self.opening_sessions) != signature(self.policy_sessions)

    @property
    def distance_changed_from_opening(self) -> bool:
        """Whether an otherwise matching decision changed prescribed mileage."""

        if self.schedule_changed_from_opening:
            return False
        return any(
            abs(opening.midpoint_miles - policy.midpoint_miles) > 1e-9
            for opening, policy in zip(
                self.opening_sessions,
                self.policy_sessions,
                strict=True,
            )
        )


@dataclass(frozen=True, slots=True)
class ExpectedPolicyRollout:
    """A short calendar assembled from successive policy decisions.

    This is deliberately a diagnostic artifact. Each step retains the full
    planner snapshot that produced it, so the stitched calendar cannot conceal
    instability in the underlying daily plans.
    """

    generated_at: datetime
    horizon_days: int
    opening_plan: tuple[ProjectionPlanSession, ...]
    steps: tuple[ExpectedPolicyRolloutStep, ...]
    replans: tuple[ProjectionReplan, ...]

    @property
    def opening_horizon_plan(self) -> tuple[ProjectionPlanSession, ...]:
        return tuple(
            session
            for step in self.steps
            for session in step.opening_sessions
        )

    @property
    def policy_plan(self) -> tuple[ProjectionPlanSession, ...]:
        return tuple(
            session
            for step in self.steps
            for session in step.policy_sessions
        )

    @property
    def schedule_change_count(self) -> int:
        return sum(step.schedule_changed_from_opening for step in self.steps)

    @property
    def distance_change_count(self) -> int:
        return sum(step.distance_changed_from_opening for step in self.steps)


@dataclass(frozen=True, slots=True)
class ProjectionAdherenceEvent:
    occurred_at: datetime
    kind: str
    detail: str


@dataclass(frozen=True, slots=True)
class ProjectionActualSession:
    occurred_at: datetime
    workout_type: WorkoutType
    distance_miles: float
    was_prescribed: bool


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
    replan_snapshots: tuple[ProjectionReplan, ...] = ()
    adherence_event_records: tuple[ProjectionAdherenceEvent, ...] = ()
    actual_sessions: tuple[ProjectionActualSession, ...] = ()


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
                prescribed_workout_type=run.prescribed_workout_type,
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
    ordinary_easy_samples = []
    for run in completed:
        if run.workout_type != WorkoutType.EASY:
            continue
        prescribed_range = (
            (run.prescribed_low_miles, run.prescribed_high_miles)
            if run.prescribed_low_miles is not None
            and run.prescribed_high_miles is not None
            else None
        )
        sample_distance = ordinary_easy_sample_distance(
            run.distance_miles,
            run.planning_role,
            prescribed_range,
        )
        if sample_distance is not None:
            ordinary_easy_samples.append((run.start_time, sample_distance))
    typical_easy_run_miles = recency_weighted_easy_distance(
        ordinary_easy_samples,
        as_of,
        half_life_days=easy_baseline_half_life_days,
    )
    recovery_reference = ordinary_session_reference(
        recent_28,
        typical_easy_run_miles,
    )
    recovery_residual_load = cumulative_recovery_load(
        (
            (
                athlete_relative_session_load(
                    run.difficulty,
                    recovery_reference,
                    performance_response=(
                        template.recent_performance_response
                        if run is last and not run.projected
                        else "within_recent_range"
                    ),
                    drift_percent=(
                        template.last_run_drift_percent
                        if run is last and not run.projected
                        else None
                    ),
                    prescribed_intensity_factor=(
                        template.last_run_prescribed_intensity_factor
                        if run is last and not run.projected
                        else None
                    ),
                    prescribed_distance_range_miles=(
                        (
                            run.prescribed_low_miles,
                            run.prescribed_high_miles,
                        )
                        if run.prescribed_low_miles is not None
                        and run.prescribed_high_miles is not None
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
            "last_run_distance_miles": last.distance_miles if last else None,
            "last_run_workout_type": last.workout_type if last else None,
            "last_run_prescribed_workout_type": (
                (
                    last.prescribed_workout_type
                    if last.prescribed_workout_type is not None
                    else template.last_run_prescribed_workout_type
                )
                if last and not last.projected
                else last.prescribed_workout_type if last else None
            ),
            "last_run_prescribed_distance_range_miles": (
                (last.prescribed_low_miles, last.prescribed_high_miles)
                if last
                and last.prescribed_low_miles is not None
                and last.prescribed_high_miles is not None
                else template.last_run_prescribed_distance_range_miles
                if last and not last.projected
                else None
            ),
            "last_run_completed_prescribed_workout": (
                (
                    last.completed_prescribed_workout
                    if last.completed_prescribed_workout is not None
                    else template.last_run_completed_prescribed_workout
                )
                if last and not last.projected
                else last.completed_prescribed_workout if last else None
            ),
            "last_run_drift_percent": None,
            "last_run_prescribed_intensity_factor": None,
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
            "last_completed_quality_session_type": next(
                (
                    run.quality_session_type
                    for run in reversed(completed)
                    if run.workout_type in QUALITY_TYPES
                    and run.quality_session_type is not None
                ),
                template.last_completed_quality_session_type,
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
            "recent_performance_response": "within_recent_range",
            "recent_illness_or_recovery": False,
            "normal_runs_since_health_event": 0,
            "current_health_status": CurrentHealthStatus.NORMAL,
            "response_flags": [],
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
    overload_profile: OverloadAdherenceProfile | None = None,
    deterministic_profile: DeterministicAdherenceProfile | None = None,
    initial_schedule: WeeklyScheduleResponse | None = None,
    simulation_days: int | None = None,
    replan_observer: Callable[[ProjectionReplan], None] | None = None,
    emit_post_upload_replans: bool = True,
) -> list[ProjectionWeek]:
    """Roll the real planner forward under controlled adherence behavior.

    The simulator projects scheduling and training load only. It assumes a
    normal response and does not invent future pace, HR, sleep, soreness,
    illness, injury, weather, missed sessions, or fitness improvement.

    Daily replanning is the production-fidelity default: only work scheduled
    before the next reload is committed. Longer intervals are an explicit
    coarse-test mode and must not be used to draw calendar-spacing conclusions.
    """

    if not 1 <= replan_interval_days <= 7:
        raise ValueError("replan_interval_days must be between 1 and 7")
    if simulation_days is not None and simulation_days < 1:
        raise ValueError("simulation_days must be at least one")
    selected_profiles = sum(
        profile is not None
        for profile in (
            human_profile,
            overload_profile,
            deterministic_profile,
        )
    )
    if selected_profiles > 1:
        raise ValueError(
            "Human, overload, and deterministic profiles are mutually exclusive"
        )
    if overload_profile is not None and replan_interval_days != 1:
        raise ValueError("Closed-loop overload diagnostics require daily replanning")
    if (
        overload_profile is not None
        and overload_profile.trigger_scheduled_run < 1
    ):
        raise ValueError("trigger_scheduled_run must be at least one")
    if (
        deterministic_profile is not None
        and deterministic_profile.trigger_scheduled_run < 1
    ):
        raise ValueError("trigger_scheduled_run must be at least one")

    history = list(recorded_runs)
    summaries: dict[int, dict] = {}
    commitments: dict[date, tuple[datetime, float, float]] = {}
    adherence_events: list[tuple[datetime, str, str]] = []
    behavior_rng = Random(human_profile.seed) if human_profile else None
    scheduled_execution_count = 0
    deterministic_candidate_count = 0
    deterministic_trigger_complete = False
    expected_openings: dict[datetime, tuple[datetime, float | None]] = {}
    overload_trigger_date: date | None = None
    unscheduled_overload_complete = False
    candidate_hours = sorted(
        int(value)
        for value in config.get("weather", {}).get(
            "automatic_run_time_hours_local", [7, 12, 19]
        )
    )
    default_hour = candidate_hours[0] if candidate_hours else 7
    total_days = (
        int(simulation_days)
        if simulation_days is not None
        else max(1, weeks) * 7
    )
    simulation_start_date = start_at.date()
    breaks = (
        _human_breaks(human_profile, simulation_start_date, total_days)
        if human_profile
        else (
            (
                _HumanBreak(
                    start_date=(
                        simulation_start_date
                        + timedelta(
                            days=max(0, deterministic_profile.block_start_day)
                        )
                    ),
                    end_date=(
                        simulation_start_date
                        + timedelta(
                            days=max(0, deterministic_profile.block_start_day)
                            + max(1, deterministic_profile.block_days)
                        )
                    ),
                    label="hidden vacation",
                ),
            )
            if deterministic_profile is not None
            and deterministic_profile.scenario
            == DeterministicAdherenceScenario.HIDDEN_VACATION
            else ()
        )
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
    # The live app supplies its saved full-horizon schedule as a beam-search
    # warm start. It receives no scoring preference, but prevents an
    # approximate candidate search from overlooking a still-optimal translated
    # calendar after the horizon origin advances.
    previous_schedule = initial_schedule
    last_replan_at: datetime | None = None

    def planning_activities_for(
        values: list[ProjectionRun],
    ) -> list[PlanningActivity]:
        return [
            PlanningActivity(
                run.start_time,
                run.distance_miles,
                moving_minutes=run.moving_minutes,
                easy_minutes=run.difficulty.zone_breakdown.easy_minutes,
                baseline_eligible=(
                    run.workout_type
                    in {
                        WorkoutType.EASY,
                        WorkoutType.RECOVERY,
                        WorkoutType.RUN_WALK,
                        WorkoutType.UNKNOWN,
                    }
                    and not run.difficulty.is_long_run
                    and not run.difficulty.is_quality_session
                    and run.planning_role
                    not in {"support_easy", "medium_long"}
                ),
                target_distance_miles=adherence_normalized_distance(
                    run.distance_miles,
                    (
                        (run.prescribed_low_miles, run.prescribed_high_miles)
                        if run.prescribed_low_miles is not None
                        and run.prescribed_high_miles is not None
                        else None
                    ),
                ),
            )
            for run in values
        ]

    def build_post_upload_replan(
        *,
        upload_at: datetime,
        source_activity_at: datetime,
        prior_schedule: WeeklyScheduleResponse,
        expected_opening: tuple[datetime, float | None] | None,
    ) -> tuple[WeeklyScheduleResponse, ProjectionReplan]:
        """Regenerate from uploaded evidence before the next clock refresh.

        The production planner advances its decision boundary after a run is
        completed. The projection previously skipped this event and compared
        the pre-run plan directly with the following day's refresh, obscuring
        whether churn was caused by the upload or by an evidence-free reload.
        """

        completed_today = _completed_activities_on_plan_date(history, upload_at)
        decision_start_date = upload_at.date() + timedelta(
            days=1 if completed_today else 0
        )
        target_as_of = datetime.combine(
            decision_start_date,
            time.min,
            tzinfo=start_at.tzinfo,
        )
        activities = planning_activities_for(history)
        target_runs, target_range, evidence = derive_weekly_target(
            activities,
            target_as_of,
            config,
        )
        capacity = evidence.capacity_reference_miles
        daily_states: list[FitnessState] = []
        daily_state_options: list[list[FitnessState]] = []
        for offset in range(PLANNING_HORIZON_DAYS):
            day = decision_start_date + timedelta(days=offset)
            hours = list(candidate_hours)
            if offset == 0 and decision_start_date == upload_at.date():
                hours = [
                    hour
                    for hour in hours
                    if datetime.combine(
                        day,
                        time(hour),
                        tzinfo=start_at.tzinfo,
                    )
                    > upload_at + timedelta(minutes=10)
                ]
                if not hours:
                    hours = [upload_at.hour]
            options: list[FitnessState] = []
            for hour in hours:
                planned_at = datetime.combine(
                    day,
                    time(hour),
                    tzinfo=start_at.tzinfo,
                )
                if planned_at < upload_at:
                    planned_at = upload_at
                options.append(
                    _state_at(
                        template,
                        history,
                        planned_at,
                        capacity,
                        easy_baseline_half_life_days=float(
                            config.get("coaching", {}).get(
                                "capacity_retention_half_life_days",
                                84,
                            )
                        ),
                        fatigue_half_life_days=float(
                            config.get("coaching", {}).get(
                                "continuous_fatigue_half_life_days",
                                7,
                            )
                        ),
                    )
                )
            daily_state_options.append(options)
        planning_reference_state = _state_at(
            template,
            history,
            target_as_of,
            capacity,
            easy_baseline_half_life_days=float(
                config.get("coaching", {}).get(
                    "capacity_retention_half_life_days",
                    84,
                )
            ),
            fatigue_half_life_days=float(
                config.get("coaching", {}).get(
                    "continuous_fatigue_half_life_days",
                    7,
                )
            ),
        )
        daily_states = [
            planning_reference_state,
            *(options[0] for options in daily_state_options[1:]),
        ]
        pace_window = _load_window(history, upload_at, 28)
        pace = (
            pace_window.moving_minutes / pace_window.distance_miles
            if pace_window.distance_miles > 0
            else 11.0
        )
        pace = min(15.0, max(7.0, pace))
        expected_target_projector = make_expected_target_projector(
            activities,
            decision_start_date,
            config,
            pace_min_mile=pace,
            maximum_horizon_days=PLANNING_HORIZON_DAYS,
            opening_target_range=target_range,
            opening_evidence=evidence,
            ordinary_easy_midpoint_miles=(
                daily_states[0].typical_easy_run_miles
            ),
            prior_schedule=prior_schedule,
        )
        forced_rest_offsets: set[int] = set()
        if (
            deterministic_profile is not None
            and deterministic_profile.scenario
            == DeterministicAdherenceScenario.KNOWN_FORCED_REST
        ):
            block_start = simulation_start_date + timedelta(
                days=max(0, deterministic_profile.block_start_day)
            )
            block_end = block_start + timedelta(
                days=max(1, deterministic_profile.block_days)
            )
            forced_rest_offsets = {
                offset
                for offset, state in enumerate(daily_states)
                if block_start <= state.as_of.date() < block_end
            }
        planning_started = perf_counter()
        schedule = build_weekly_schedule(
            daily_states,
            RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
            config,
            target_run_count=target_runs,
            target_distance_range=target_range,
            target_evidence=evidence,
            completed_activities_by_offset={},
            daily_state_options=daily_state_options,
            forced_rest_offsets=forced_rest_offsets,
            prior_schedule=prior_schedule,
            expected_target_projector=expected_target_projector,
        )
        planning_seconds = perf_counter() - planning_started
        visible_plan = [
            day.recommendation
            for day in (schedule.planning_days or schedule.days)
            if day.recommendation
            and day.recommendation.workout_type != WorkoutType.REST
            and day.recommendation.planned_for
        ]
        opening = (
            daily_states[0].recent_load.continuous_fatigue_to_capacity_ratio
            if daily_states[0].recent_load.continuous_fatigue_to_capacity_ratio
            is not None
            else daily_states[0].recent_load.acute_distance_to_capacity_ratio
        )
        expected_state_at = (
            expected_opening[0] if expected_opening is not None else None
        )
        expected_recovery_residual = (
            expected_opening[1] if expected_opening is not None else None
        )
        actual_recovery_residual = daily_states[0].recovery_residual_load
        recovery_surprise = (
            actual_recovery_residual - expected_recovery_residual
            if actual_recovery_residual is not None
            and expected_recovery_residual is not None
            and expected_state_at == daily_states[0].as_of
            else None
        )
        material_evidence_reasons = tuple(
            f"{kind}: {detail}"
            for occurred_at, kind, detail in adherence_events
            if occurred_at == source_activity_at
            and kind in {"deviation", "overload", "unscheduled"}
        )
        snapshot = ProjectionReplan(
            generated_at=upload_at,
            opening_load_ratio=opening,
            target_low_miles=target_range[0],
            target_high_miles=target_range[1],
            planned_sessions=tuple(
                ProjectionPlanSession(
                    planned_for=item.planned_for,
                    workout_type=item.workout_type,
                    midpoint_miles=sum(item.distance_range_miles) / 2,
                    distance_low_miles=item.distance_range_miles[0],
                    distance_high_miles=item.distance_range_miles[1],
                    planning_role=item.planning_role,
                    readiness=item.readiness.value,
                )
                for item in visible_plan
                if item.planned_for and item.distance_range_miles
            ),
            committed_sessions=(),
            capacity_reference_miles=capacity,
            planning_mode=evidence.planning_mode.value,
            ordinary_easy_midpoint_miles=daily_states[0].typical_easy_run_miles,
            boundary_session_miles=(
                daily_states[0].last_run_distance_miles
                if daily_states[0].days_since_last_run is not None
                and daily_states[0].days_since_last_run <= 7
                else None
            ),
            opening_continuous_distance_miles=(
                daily_states[0].recent_load.continuous_distance_miles
            ),
            opening_short_term_distance_miles=(
                daily_states[0].recent_load.continuous_short_term_distance_miles
            ),
            peak_projected_continuous_mileage_rate=(
                schedule.peak_projected_continuous_mileage_rate
            ),
            planning_seconds=planning_seconds,
            target_trajectory=(
                expected_target_projector(
                    visible_plan,
                    PLANNING_HORIZON_DAYS,
                )
                if replan_observer is not None
                else ()
            ),
            decision_start_date=decision_start_date,
            planner_diagnostics=(
                schedule.planner_diagnostics.model_dump()
                if schedule.planner_diagnostics is not None
                else None
            ),
            material_evidence_reasons=material_evidence_reasons,
            opening_recovery_residual_load=actual_recovery_residual,
            expected_opening_recovery_residual_load=expected_recovery_residual,
            expected_opening_state_at=expected_state_at,
            recovery_surprise_units=recovery_surprise,
            days_since_long_run=daily_states[0].days_since_long_run,
            days_since_quality_run=daily_states[0].days_since_quality_run,
            long_cadence_reference_days=float(
                config.get("coaching", {}).get(
                    "long_run_recency_reference_days",
                    7,
                )
            ),
            quality_cadence_reference_days=float(
                config.get("coaching", {}).get(
                    "quality_recency_reference_days",
                    7,
                )
            ),
            cadence_exception_reasons=tuple(
                dict.fromkeys(
                    reason
                    for day in (schedule.planning_days or schedule.days)
                    if day.recommendation is not None
                    for reason in day.recommendation.reasons
                    if any(
                        marker in reason.casefold()
                        for marker in (
                            "replaced with aerobic",
                            "recovery guardrail",
                            "readiness guardrail",
                            "recovery is incomplete",
                        )
                    )
                )
            ),
            trigger="post_upload",
            source_activity_at=source_activity_at,
        )
        return schedule, snapshot

    for plan_offset in range(0, total_days, replan_interval_days):
        plan_start = start_at + timedelta(days=plan_offset)
        commit_end = min(
            plan_start + timedelta(days=replan_interval_days),
            simulation_end_at,
        )
        week_index = min(weeks - 1, plan_offset // 7)
        completed_today = _completed_activities_on_plan_date(
            history, plan_start
        )
        # Match the live planner's forward decision boundary. Once today's
        # prescribed work has been completed, it belongs to load/history and
        # the remaining-work optimizer starts tomorrow. This makes the
        # post-completion plan directly comparable with the next reload.
        decision_start_date = plan_start.date() + timedelta(
            days=1 if completed_today else 0
        )
        target_as_of = datetime.combine(
            decision_start_date,
            time.min,
            tzinfo=start_at.tzinfo,
        )
        activities = planning_activities_for(history)
        target_runs, target_range, evidence = derive_weekly_target(
            activities, target_as_of, config
        )
        capacity = evidence.capacity_reference_miles
        daily_state_options: list[list[FitnessState]] = []
        for offset in range(PLANNING_HORIZON_DAYS):
            day = decision_start_date + timedelta(days=offset)
            hours = list(candidate_hours)
            if offset == 0 and decision_start_date == plan_start.date():
                hours = [
                    hour
                    for hour in hours
                    if datetime.combine(
                        day,
                        time(hour),
                        tzinfo=start_at.tzinfo,
                    )
                    > plan_start + timedelta(minutes=10)
                ]
                if not hours:
                    hours = [plan_start.hour]
            options: list[FitnessState] = []
            for hour in hours:
                planned_at = datetime.combine(
                    day,
                    time(hour),
                    tzinfo=start_at.tzinfo,
                )
                if planned_at < plan_start:
                    planned_at = plan_start
                options.append(
                    _state_at(
                        template,
                        history,
                        planned_at,
                        capacity,
                        easy_baseline_half_life_days=float(
                            config.get("coaching", {}).get(
                                "capacity_retention_half_life_days",
                                84,
                            )
                        ),
                        fatigue_half_life_days=float(
                            config.get("coaching", {}).get(
                                "continuous_fatigue_half_life_days",
                                7,
                            )
                        ),
                    )
                )
            daily_state_options.append(options)
        planning_reference_state = _state_at(
            template,
            history,
            target_as_of,
            capacity,
            easy_baseline_half_life_days=float(
                config.get("coaching", {}).get(
                    "capacity_retention_half_life_days",
                    84,
                )
            ),
            fatigue_half_life_days=float(
                config.get("coaching", {}).get(
                    "continuous_fatigue_half_life_days",
                    7,
                )
            ),
        )
        daily_states = [
            planning_reference_state,
            *(options[0] for options in daily_state_options[1:]),
        ]
        pace_window = _load_window(history, plan_start, 28)
        pace = (
            pace_window.moving_minutes / pace_window.distance_miles
            if pace_window.distance_miles > 0
            else 11.0
        )
        pace = min(15.0, max(7.0, pace))
        expected_target_projector = make_expected_target_projector(
            activities,
            decision_start_date,
            config,
            pace_min_mile=pace,
            maximum_horizon_days=PLANNING_HORIZON_DAYS,
            opening_target_range=target_range,
            opening_evidence=evidence,
            ordinary_easy_midpoint_miles=(
                daily_states[0].typical_easy_run_miles
            ),
            prior_schedule=previous_schedule,
        )

        planning_started = perf_counter()
        forced_rest_offsets: set[int] = set()
        if (
            deterministic_profile is not None
            and deterministic_profile.scenario
            == DeterministicAdherenceScenario.KNOWN_FORCED_REST
        ):
            block_start = simulation_start_date + timedelta(
                days=max(0, deterministic_profile.block_start_day)
            )
            block_end = block_start + timedelta(
                days=max(1, deterministic_profile.block_days)
            )
            forced_rest_offsets = {
                offset
                for offset, state in enumerate(daily_states)
                if block_start <= state.as_of.date() < block_end
            }
        schedule = build_weekly_schedule(
            daily_states,
            RecommendationRequest(health_status=CurrentHealthStatus.NORMAL),
            config,
            target_run_count=target_runs,
            target_distance_range=target_range,
            target_evidence=evidence,
            completed_activities_by_offset={},
            daily_state_options=daily_state_options,
            forced_rest_offsets=forced_rest_offsets,
            prior_schedule=previous_schedule,
            expected_target_projector=expected_target_projector,
        )
        planning_seconds = perf_counter() - planning_started
        previous_schedule = schedule
        committed = [
            day.recommendation
            for day in schedule.days
            if day.recommendation
            and day.recommendation.workout_type != WorkoutType.REST
            and day.recommendation.planned_for
            and plan_start <= day.recommendation.planned_for < commit_end
        ]
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
                "snapshots": [],
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
        target_trajectory = (
            expected_target_projector(
                visible_plan,
                PLANNING_HORIZON_DAYS,
            )
            if replan_observer is not None
            else ()
        )
        expected_opening = expected_openings.pop(plan_start, None)
        expected_state_at = (
            expected_opening[0] if expected_opening is not None else None
        )
        expected_recovery_residual = (
            expected_opening[1] if expected_opening is not None else None
        )
        actual_recovery_residual = daily_states[0].recovery_residual_load
        recovery_surprise = (
            actual_recovery_residual - expected_recovery_residual
            if actual_recovery_residual is not None
            and expected_recovery_residual is not None
            and expected_state_at == daily_states[0].as_of
            else None
        )
        snapshot = ProjectionReplan(
            generated_at=plan_start,
            opening_load_ratio=opening,
            target_low_miles=target_range[0],
            target_high_miles=target_range[1],
            planned_sessions=tuple(
                ProjectionPlanSession(
                    planned_for=item.planned_for,
                    workout_type=item.workout_type,
                    midpoint_miles=sum(item.distance_range_miles) / 2,
                    distance_low_miles=item.distance_range_miles[0],
                    distance_high_miles=item.distance_range_miles[1],
                    planning_role=item.planning_role,
                    readiness=item.readiness.value,
                )
                for item in visible_plan
                if item.planned_for and item.distance_range_miles
            ),
            committed_sessions=tuple(
                ProjectionPlanSession(
                    planned_for=item.planned_for,
                    workout_type=item.workout_type,
                    midpoint_miles=sum(item.distance_range_miles) / 2,
                    distance_low_miles=item.distance_range_miles[0],
                    distance_high_miles=item.distance_range_miles[1],
                    planning_role=item.planning_role,
                    readiness=item.readiness.value,
                )
                for item in committed
                if item.planned_for and item.distance_range_miles
            ),
            capacity_reference_miles=capacity,
            planning_mode=evidence.planning_mode.value,
            ordinary_easy_midpoint_miles=(
                daily_states[0].typical_easy_run_miles
            ),
            boundary_session_miles=(
                daily_states[0].last_run_distance_miles
                if daily_states[0].days_since_last_run is not None
                and daily_states[0].days_since_last_run <= 7
                else None
            ),
            opening_continuous_distance_miles=(
                daily_states[0].recent_load.continuous_distance_miles
            ),
            opening_short_term_distance_miles=(
                daily_states[0]
                .recent_load
                .continuous_short_term_distance_miles
            ),
            peak_projected_continuous_mileage_rate=(
                schedule.peak_projected_continuous_mileage_rate
            ),
            planning_seconds=planning_seconds,
            target_trajectory=target_trajectory,
            decision_start_date=decision_start_date,
            planner_diagnostics=(
                schedule.planner_diagnostics.model_dump()
                if schedule.planner_diagnostics is not None
                else None
            ),
            material_evidence_reasons=tuple(
                f"{kind}: {detail}"
                for occurred_at, kind, detail in adherence_events
                if (last_replan_at is None or last_replan_at < occurred_at)
                and occurred_at <= plan_start
                and kind
                in {
                    "deviation",
                    "overload",
                    "skip",
                    "unscheduled",
                    "break",
                }
            ),
            opening_recovery_residual_load=actual_recovery_residual,
            expected_opening_recovery_residual_load=(
                expected_recovery_residual
            ),
            expected_opening_state_at=expected_state_at,
            recovery_surprise_units=recovery_surprise,
            days_since_long_run=daily_states[0].days_since_long_run,
            days_since_quality_run=daily_states[0].days_since_quality_run,
            long_cadence_reference_days=float(
                config.get("coaching", {}).get(
                    "long_run_recency_reference_days",
                    7,
                )
            ),
            quality_cadence_reference_days=float(
                config.get("coaching", {}).get(
                    "quality_recency_reference_days",
                    7,
                )
            ),
            cadence_exception_reasons=tuple(
                dict.fromkeys(
                    reason
                    for day in (schedule.planning_days or schedule.days)
                    if day.recommendation is not None
                    for reason in day.recommendation.reasons
                    if any(
                        marker in reason.casefold()
                        for marker in (
                            "replaced with aerobic",
                            "recovery guardrail",
                            "readiness guardrail",
                            "recovery is incomplete",
                        )
                    )
                )
            ),
        )
        summary["snapshots"].append(snapshot)
        if replan_observer is not None:
            replan_observer(snapshot)
        last_replan_at = snapshot.generated_at
        next_plan_start = plan_start + timedelta(
            days=replan_interval_days
        )
        if next_plan_start < simulation_end_at:
            expected_committed_runs = [
                ProjectionRun(
                    start_time=item.planned_for,
                    distance_miles=sum(item.distance_range_miles) / 2.0,
                    moving_minutes=(
                        sum(item.distance_range_miles) / 2.0 * pace
                    ),
                    workout_type=item.workout_type,
                    difficulty=_projected_difficulty(
                        item,
                        sum(item.distance_range_miles) / 2.0,
                        pace,
                    ),
                    projected=True,
                    planning_role=_completed_planning_role(
                        item,
                        item.workout_type,
                    ),
                    prescribed_workout_type=item.workout_type,
                    completed_prescribed_workout=True,
                    prescribed_low_miles=item.distance_range_miles[0],
                    prescribed_high_miles=item.distance_range_miles[1],
                    prescription_title=item.title,
                    quality_session_type=(
                        item.quality_session_type
                        if item.workout_type in QUALITY_TYPES
                        else None
                    ),
                )
                for item in committed
                if item.planned_for is not None
                and item.distance_range_miles is not None
            ]
            expected_history = sorted(
                [*history, *expected_committed_runs],
                key=lambda item: item.start_time,
            )
            expected_completed_today = _completed_activities_on_plan_date(
                expected_history,
                next_plan_start,
            )
            expected_decision_date = next_plan_start.date() + timedelta(
                days=1 if expected_completed_today else 0
            )
            expected_target_as_of = datetime.combine(
                expected_decision_date,
                time.min,
                tzinfo=start_at.tzinfo,
            )
            _, _, expected_evidence = derive_weekly_target(
                planning_activities_for(expected_history),
                expected_target_as_of,
                config,
            )
            expected_state = _state_at(
                template,
                expected_history,
                expected_target_as_of,
                expected_evidence.capacity_reference_miles,
                easy_baseline_half_life_days=float(
                    config.get("coaching", {}).get(
                        "capacity_retention_half_life_days",
                        84,
                    )
                ),
                fatigue_half_life_days=float(
                    config.get("coaching", {}).get(
                        "continuous_fatigue_half_life_days",
                        7,
                    )
                ),
            )
            expected_openings[next_plan_start] = (
                expected_state.as_of,
                expected_state.recovery_residual_load,
            )
        for item in committed:
            if not item.distance_range_miles or not item.planned_for:
                continue
            deterministic_triggered = False
            if (
                deterministic_profile is not None
                and not deterministic_trigger_complete
            ):
                deterministic_eligible = bool(
                    deterministic_profile.scenario
                    != DeterministicAdherenceScenario.BELOW_RANGE_EASY
                    or item.workout_type
                    in {WorkoutType.EASY, WorkoutType.RECOVERY}
                )
                if deterministic_eligible:
                    deterministic_candidate_count += 1
                    deterministic_triggered = bool(
                        deterministic_candidate_count
                        == deterministic_profile.trigger_scheduled_run
                    )
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
                deterministic_profile is not None
                and deterministic_profile.scenario
                == DeterministicAdherenceScenario.SINGLE_HIDDEN_MISS
                and deterministic_triggered
            ):
                deterministic_trigger_complete = True
                adherence_events.append(
                    (
                        item.planned_for,
                        "skip",
                        f"single hidden miss: {item.workout_type.value}",
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
            if deterministic_profile is not None:
                if (
                    deterministic_profile.scenario
                    == DeterministicAdherenceScenario.IN_RANGE_LOW
                ):
                    miles = item.distance_range_miles[0]
                elif (
                    deterministic_profile.scenario
                    == DeterministicAdherenceScenario.IN_RANGE_HIGH
                ):
                    miles = item.distance_range_miles[1]
                elif (
                    deterministic_profile.scenario
                    == DeterministicAdherenceScenario.BELOW_RANGE_EASY
                    and deterministic_triggered
                ):
                    miles = max(
                        0.5,
                        item.distance_range_miles[0]
                        * max(
                            0.0,
                            min(1.0, deterministic_profile.below_range_fraction),
                        ),
                    )
                    deterministic_trigger_complete = True
                    notes.append("below prescribed range")
                elif (
                    deterministic_profile.scenario
                    == DeterministicAdherenceScenario.ABOVE_RANGE_OR_HARDER
                    and deterministic_triggered
                ):
                    ordinary_miles = daily_states[0].typical_easy_run_miles
                    miles = item.distance_range_miles[1] + max(
                        0.1,
                        ordinary_miles
                        * deterministic_profile.above_range_easy_fraction,
                    )
                    deterministic_trigger_complete = True
                    notes.append("above prescribed range")
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
            if (
                deterministic_profile is not None
                and deterministic_profile.scenario
                == DeterministicAdherenceScenario.ABOVE_RANGE_OR_HARDER
                and deterministic_triggered
            ):
                too_intense = True
            if too_intense:
                notes.append("more intense than prescribed")
            if overload_profile is not None:
                scheduled_execution_count += 1
                if (
                    scheduled_execution_count
                    == overload_profile.trigger_scheduled_run
                ):
                    overload_trigger_date = item.planned_for.date()
                    ordinary_miles = (
                        _state_at(
                            template,
                            history,
                            item.planned_for,
                            capacity,
                        ).typical_easy_run_miles
                        or miles
                    )
                    if (
                        overload_profile.scenario
                        == OverloadScenario.EXTRA_DISTANCE
                    ):
                        added_miles = max(
                            0.0,
                            ordinary_miles
                            * overload_profile.extra_distance_easy_fraction,
                        )
                        miles += added_miles
                        notes.append(
                            f"overload: {added_miles:.1f} extra miles"
                        )
                    elif (
                        overload_profile.scenario
                        == OverloadScenario.EXTRA_INTENSITY
                    ):
                        too_intense = True
                        notes.append("overload: extra intensity")
            adherence_note = "; ".join(notes) or "within prescribed margin"
            observed_deviation = bool(
                human_profile
                or overload_profile
                or (
                    deterministic_profile is not None
                    and deterministic_profile.scenario
                    in {
                        DeterministicAdherenceScenario.BELOW_RANGE_EASY,
                        DeterministicAdherenceScenario.ABOVE_RANGE_OR_HARDER,
                    }
                )
            )
            if notes:
                adherence_events.append(
                    (
                        item.planned_for,
                        (
                            "overload"
                            if overload_profile is not None
                            and any(
                                note.startswith("overload:")
                                for note in notes
                            )
                            else "deviation"
                        ),
                        adherence_note,
                    )
                )
            history_before_run = list(history)
            completed_run = ProjectionRun(
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
                        if observed_deviation
                        else _projected_difficulty(item, miles, pace)
                    ),
                    projected=True,
                    planning_role=_completed_planning_role(item, actual_type),
                    prescribed_workout_type=item.workout_type,
                    completed_prescribed_workout=(
                        actual_type == item.workout_type
                        or (
                            actual_type in QUALITY_TYPES
                            and item.workout_type in QUALITY_TYPES
                        )
                    ),
                    prescribed_low_miles=item.distance_range_miles[0],
                    prescribed_high_miles=item.distance_range_miles[1],
                    prescription_title=item.title,
                    quality_session_type=(
                        item.quality_session_type
                        if actual_type in QUALITY_TYPES
                        else None
                    ),
                    adherence_note=(
                        adherence_note
                        if observed_deviation
                        else None
                    ),
                )
            history.append(completed_run)
            history.sort(key=lambda run: run.start_time)

            upload_at = item.planned_for + timedelta(
                minutes=max(1.0, completed_run.moving_minutes)
            )
            if emit_post_upload_replans and upload_at < simulation_end_at:
                expected_miles = sum(item.distance_range_miles) / 2.0
                expected_run = ProjectionRun(
                    start_time=item.planned_for,
                    distance_miles=expected_miles,
                    moving_minutes=expected_miles * pace,
                    workout_type=item.workout_type,
                    difficulty=_projected_difficulty(
                        item,
                        expected_miles,
                        pace,
                    ),
                    projected=True,
                    planning_role=_completed_planning_role(
                        item,
                        item.workout_type,
                    ),
                    prescribed_workout_type=item.workout_type,
                    completed_prescribed_workout=True,
                    prescribed_low_miles=item.distance_range_miles[0],
                    prescribed_high_miles=item.distance_range_miles[1],
                    prescription_title=item.title,
                    quality_session_type=(
                        item.quality_session_type
                        if item.workout_type in QUALITY_TYPES
                        else None
                    ),
                )
                expected_history = sorted(
                    [*history_before_run, expected_run],
                    key=lambda run: run.start_time,
                )
                expected_completed_today = (
                    _completed_activities_on_plan_date(
                        expected_history,
                        upload_at,
                    )
                )
                expected_decision_date = upload_at.date() + timedelta(
                    days=1 if expected_completed_today else 0
                )
                expected_state_at = datetime.combine(
                    expected_decision_date,
                    time.min,
                    tzinfo=start_at.tzinfo,
                )
                _, _, expected_evidence = derive_weekly_target(
                    planning_activities_for(expected_history),
                    expected_state_at,
                    config,
                )
                expected_state = _state_at(
                    template,
                    expected_history,
                    expected_state_at,
                    expected_evidence.capacity_reference_miles,
                    easy_baseline_half_life_days=float(
                        config.get("coaching", {}).get(
                            "capacity_retention_half_life_days",
                            84,
                        )
                    ),
                    fatigue_half_life_days=float(
                        config.get("coaching", {}).get(
                            "continuous_fatigue_half_life_days",
                            7,
                        )
                    ),
                )
                previous_schedule, post_upload_snapshot = (
                    build_post_upload_replan(
                        upload_at=upload_at,
                        source_activity_at=item.planned_for,
                        prior_schedule=previous_schedule,
                        expected_opening=(
                            expected_state.as_of,
                            expected_state.recovery_residual_load,
                        ),
                    )
                )
                summary["snapshots"].append(post_upload_snapshot)
                summary["replans"].append(
                    f"{upload_at.strftime('%a %b %-d %H:%M')}: "
                    "post-upload replan"
                )
                if replan_observer is not None:
                    replan_observer(post_upload_snapshot)
                last_replan_at = post_upload_snapshot.generated_at
        deterministic_unscheduled = False
        if (
            overload_profile is not None
            and overload_trigger_date is not None
            and not any(
                item.planned_for
                and item.planned_for.date() == plan_start.date()
                for item in committed
            )
        ):
            days_after_trigger = (
                plan_start.date() - overload_trigger_date
            ).days
            if (
                overload_profile.scenario
                == OverloadScenario.UNSCHEDULED_EASY
                and days_after_trigger >= 1
                and not unscheduled_overload_complete
            ):
                deterministic_unscheduled = True
                unscheduled_overload_complete = True
            elif (
                overload_profile.scenario
                == OverloadScenario.DENSE_SEQUENCE
                and 1
                <= days_after_trigger
                <= overload_profile.dense_sequence_additional_days
            ):
                deterministic_unscheduled = True
        if (
            deterministic_unscheduled
            or (
                human_profile
                and behavior_rng is not None
                and not any(
                    item.planned_for
                    and item.planned_for.date() == plan_start.date()
                    for item in committed
                )
                and behavior_rng.random()
                < human_profile.unscheduled_easy_probability_per_day
            )
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
            already_running_that_day = any(
                run.start_time.date() == unscheduled_at.date()
                for run in history
            ) or any(
                item.planned_for
                and item.planned_for.date() == unscheduled_at.date()
                for item in committed
            )
            if (
                unscheduled_at < commit_end
                and active_break is None
                and not already_running_that_day
            ):
                ordinary = (
                    _state_at(
                        template,
                        history,
                        unscheduled_at,
                        capacity,
                    ).typical_easy_run_miles
                    or 3.0
                )
                miles = (
                    max(
                        0.5,
                        ordinary
                        * overload_profile.unscheduled_easy_fraction,
                    )
                    if deterministic_unscheduled
                    and overload_profile is not None
                    else max(
                        1.0,
                        ordinary * behavior_rng.uniform(0.60, 0.90),
                    )
                )
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
                too_intense = bool(
                    human_profile is not None
                    and behavior_rng is not None
                    and behavior_rng.random()
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
                        adherence_note=(
                            "overload: unscheduled easy run"
                            if deterministic_unscheduled
                            else "unscheduled easy run"
                        ),
                    )
                )
                adherence_events.append(
                    (
                        unscheduled_at,
                        (
                            "overload"
                            if deterministic_unscheduled
                            else "unscheduled"
                        ),
                        (
                            "overload: "
                            if deterministic_unscheduled
                            else ""
                        )
                        + f"unscheduled easy {miles:.1f} mi"
                        + (" with intensity drift" if too_intense else ""),
                    )
                )
                upload_at = unscheduled_at + timedelta(
                    minutes=max(1.0, miles * pace)
                )
                if emit_post_upload_replans and upload_at < simulation_end_at:
                    history.sort(key=lambda run: run.start_time)
                    previous_schedule, post_upload_snapshot = (
                        build_post_upload_replan(
                            upload_at=upload_at,
                            source_activity_at=unscheduled_at,
                            prior_schedule=previous_schedule,
                            expected_opening=None,
                        )
                    )
                    summary["snapshots"].append(post_upload_snapshot)
                    summary["replans"].append(
                        f"{upload_at.strftime('%a %b %-d %H:%M')}: "
                        "post-upload replan (unscheduled run)"
                    )
                    if replan_observer is not None:
                        replan_observer(post_upload_snapshot)
                    last_replan_at = post_upload_snapshot.generated_at
        history.sort(key=lambda item: item.start_time)

    # Include recorded history when measuring streaks so a run on projection
    # day one is not treated as isolated merely because the preceding run sits
    # just outside the forecast boundary. Older streaks do not affect a row
    # unless they intersect that row's dates.
    streaks = _consecutive_date_streaks(
        {
            run.start_time.date()
            for run in history
            if run.start_time.date()
            < simulation_start_date + timedelta(days=total_days)
        }
    )

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
                PlanningActivity(
                    run.start_time,
                    run.distance_miles,
                    target_distance_miles=adherence_normalized_distance(
                        run.distance_miles,
                        (
                            (
                                run.prescribed_low_miles,
                                run.prescribed_high_miles,
                            )
                            if run.prescribed_low_miles is not None
                            and run.prescribed_high_miles is not None
                            else None
                        ),
                    ),
                )
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
        summary = summaries.get(
            week_index,
            {"replans": [], "snapshots": []},
        )
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
            replan_snapshots=tuple(summary["snapshots"]),
            adherence_event_records=tuple(
                ProjectionAdherenceEvent(
                    occurred_at=at,
                    kind=kind,
                    detail=message,
                )
                for at, kind, message in week_events
            ),
            actual_sessions=tuple(
                ProjectionActualSession(
                    occurred_at=run.start_time,
                    workout_type=run.workout_type,
                    distance_miles=run.distance_miles,
                    was_prescribed=run.prescribed_workout_type is not None,
                )
                for run in projected_runs
            ),
        )
        )
    return results


def simulate_expected_policy_rollout(
    template: FitnessState,
    recorded_runs: list[ProjectionRun],
    config: dict,
    start_at: datetime,
    *,
    horizon_days: int = 4,
    initial_schedule: WeeklyScheduleResponse | None = None,
) -> ExpectedPolicyRollout:
    """Prototype a short plan from successive exact-compliance decisions.

    The ordinary planner emits an open-loop 21-day plan. This prototype instead
    runs the same planner once per day, assumes each committed prescription is
    completed at its midpoint, and stitches together only the work committed in
    each next 24-hour window. It does not alter or replace the production plan.
    """

    if not 1 <= horizon_days <= 7:
        raise ValueError("horizon_days must be between 1 and 7")

    projection = simulate_adherence(
        template,
        recorded_runs,
        config,
        start_at,
        weeks=1,
        replan_interval_days=1,
        initial_schedule=initial_schedule,
        simulation_days=horizon_days,
        emit_post_upload_replans=False,
    )
    replans = tuple(
        snapshot
        for week in projection
        for snapshot in week.replan_snapshots
        if snapshot.trigger == "scheduled_refresh"
    )
    if len(replans) != horizon_days:
        raise RuntimeError(
            "Expected one daily replan per policy-rollout decision window"
        )

    opening_plan = replans[0].planned_sessions
    steps: list[ExpectedPolicyRolloutStep] = []
    for offset, replan in enumerate(replans):
        window_start = start_at + timedelta(days=offset)
        window_end = window_start + timedelta(days=1)
        opening_sessions = tuple(
            session
            for session in opening_plan
            if window_start <= session.planned_for < window_end
        )
        steps.append(
            ExpectedPolicyRolloutStep(
                window_start=window_start,
                window_end=window_end,
                source_replan=replan,
                opening_sessions=opening_sessions,
                policy_sessions=replan.committed_sessions,
            )
        )

    return ExpectedPolicyRollout(
        generated_at=start_at,
        horizon_days=horizon_days,
        opening_plan=opening_plan,
        steps=tuple(steps),
        replans=replans,
    )
