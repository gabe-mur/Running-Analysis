"""Continuous rolling-lookahead schedule construction from daily fitness states.

The planner chooses run days automatically, then evaluates the existing
inspectable recommendation rules on those days. Planned sessions are projected
into later daily states so consecutive days, load, and workout recency are
intentional rather than independent recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from itertools import combinations
from math import ceil, comb, exp, floor, log
from statistics import median
from typing import Callable
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
from .prescription_load import (
    prescribed_intensity_factor,
    prescribed_zone_minutes,
    structured_duration_minutes,
)
from .recovery import (
    EASY_RUN_RESIDUAL_LIMIT,
    RECOVERY_HALF_LIFE_HOURS,
    TAXING_RUN_RESIDUAL_LIMIT,
    athlete_relative_session_load,
    decay_recovery_load,
    estimate_recovery,
    prior_typical_load,
    projected_recovery_reference_miles,
)
from .training_load import (
    TrainingSession,
    distance_capacity,
    short_term_density_half_life_days,
)
from .strength_schedule import add_strength_suggestions
from .web.schemas import (
    CurrentHealthStatus,
    FitnessState,
    LoadContext,
    LoadWindow,
    PlannerDiagnostics,
    PlannerScoreBreakdown,
    RecommendationRequest,
    RecommendationResponse,
    SessionDifficulty,
    TrailingDayActivity,
    WeeklyScheduleDay,
    WeeklyScheduleResponse,
    WeeklyPlanningMode,
    WeeklyTargetEvidence,
    WorkoutStep,
    WorkoutType,
    ZoneBreakdown,
)


VISIBLE_HORIZON_DAYS = 7
PLANNING_HORIZON_DAYS = 21
WEEKLY_PLANNER_VERSION = 118
# Version 95 introduced the current continuous-horizon schedule schema. Plans
# from that version onward remain valid unpreferred warm starts across an
# optimizer-version bump even though they must be regenerated for display.
WEEKLY_WARM_START_MIN_VERSION = 95
MAX_ADAPTIVE_CANDIDATES = 64
MAX_HORIZON_COUNT_OPTIONS = 14
ALLOCATION_ASSIGNMENTS_PER_TOTAL = 16
JOINT_DATE_FINALISTS = 5
DISTANCE_DP_SCALE = 1000
BASELINE_MINIMUM_AEROBIC_MINUTES = 10.0
# Recovery and role-shape objectives use one common program-fit unit. Target
# funding has its own stronger unit below because a complete ordinary-session
# mileage miss is more consequential than one soft workout-shape preference.
PROGRAM_FIT_UNIT = 20.0
# One extra secondary-endurance exposure is a soft shape preference, not a
# missing primary long-run lane or a target-load failure.
SECONDARY_ENDURANCE_SHAPE_FRACTION = 0.30
# Replanning has a real switching cost for an athlete who has arranged the
# next several days. Treat small whole-program score improvements as
# insufficient to rewrite those dates, while capping the cost so materially
# better recovery/load programs still win. The fifth decision offset covers
# the next four *future* dates on a late same-day refresh; post-upload plans
# start tomorrow and therefore receive one extra protected boundary date.
NEAR_TERM_STABILITY_DECISION_OFFSETS = 5
# A near-term date rewrite must buy at least this much full-objective benefit.
# This equals the existing secondary-endurance shape unit and is intentionally
# independent of unrelated far-horizon penalties already in the raw score.
NEAR_TERM_STABILITY_RELATIVE_BAND = 0.0
NEAR_TERM_STABILITY_ABSOLUTE_CAP = (
    PROGRAM_FIT_UNIT * SECONDARY_ENDURANCE_SHAPE_FRACTION
)
NEAR_TERM_STABILITY_ABSOLUTE_FLOOR = (
    PROGRAM_FIT_UNIT * SECONDARY_ENDURANCE_SHAPE_FRACTION
)
# Quarter-mile midpoint movement is ordinary allocator resolution. Larger
# near-term rewrites pay a soft switching cost; the projection gate below uses
# a wider half-mile threshold before declaring a regression.
NEAR_TERM_DISTANCE_STABILITY_TOLERANCE_MILES = 0.25


@dataclass(slots=True)
class _PlannerSearchDiagnostics:
    candidate_count: int = 0
    full_score_count: int = 0
    winning_candidate: PlannerScoreBreakdown | None = None
    translated_prior_candidate: PlannerScoreBreakdown | None = None
    prior_continuation_candidate: PlannerScoreBreakdown | None = None
    near_term_stable_candidate: PlannerScoreBreakdown | None = None
    near_term_stable_offsets: list[tuple[int, ...]] = field(
        default_factory=list
    )
    prior_independent_frequency_candidates: list[
        PlannerScoreBreakdown
    ] = field(default_factory=list)
    unconstrained_winning_candidate: PlannerScoreBreakdown | None = None
    stability_tie_break_applied: bool = False
    stability_tolerance: float | None = None
    winning_role_overrides: dict[int, str] = field(default_factory=dict)


def _near_term_stability_tolerance(unconstrained_cost: float) -> float:
    """Return the fixed, capped cost of rewriting near-term run dates."""

    return min(
        NEAR_TERM_STABILITY_ABSOLUTE_CAP,
        max(
            NEAR_TERM_STABILITY_ABSOLUTE_FLOOR,
            max(0.0, unconstrained_cost)
            * NEAR_TERM_STABILITY_RELATIVE_BAND,
        ),
    )


def _score_breakdown(
    offsets: tuple[int, ...],
    joint_cost: tuple[float, float, float, float],
    coaching_cost: float,
    ordinary_easy_midpoint: float,
    diagnostic_details: dict[str, object] | None = None,
) -> PlannerScoreBreakdown:
    diagnostic_details = diagnostic_details or {}
    return PlannerScoreBreakdown(
        offsets=list(offsets),
        total_cost=_program_selection_cost(
            joint_cost,
            coaching_cost,
            ordinary_easy_midpoint,
        ),
        coaching_cost=coaching_cost,
        target_violation=joint_cost[0],
        support_violation=joint_cost[1],
        shape_violation=joint_cost[2],
        finalized_recovery_cost=joint_cost[3],
        fragmentation_violation=diagnostic_details.get(
            "fragmentation_violation"
        ),
        long_shape_violation=diagnostic_details.get(
            "long_shape_violation"
        ),
        medium_long_shape_violation=diagnostic_details.get(
            "medium_long_shape_violation"
        ),
        allocation_trace=list(
            diagnostic_details.get("allocation_trace", [])
        ),
        projected_state_trace=list(
            diagnostic_details.get("projected_state_trace", [])
        ),
        recovery_cost_components=dict(
            diagnostic_details.get("recovery_cost_components", {})
        ),
    )
# A whole ordinary-session funding miss is more consequential than one role-
# shape preference. Recovery still wins when a denser calendar is genuinely
# unsafe, but it cannot buy away a leading-week target with a cheap soft term.
TARGET_FIT_UNIT = 35.0
HARD_FUNDING_ROUNDING_TOLERANCE_MILES = 0.5
SOFT_CONTINUATION_DAYS = 28.0
# A shortened support run still supplies aerobic load, so a complete support
# shortfall is less consequential than missing an entire ordinary session.
AEROBIC_SUPPORT_FIT_FRACTION = 0.60
QUALITY_WORKOUT_TYPES = frozenset(
    {
        WorkoutType.INTERVALS,
        WorkoutType.TEMPO_THRESHOLD,
        WorkoutType.RACE,
    }
)


def _race_timing(
    config: dict,
    on_date: date,
) -> tuple[int, int] | None:
    """Return days to race and taper length without hiding post-race dates."""

    goal = configured_race_goal(config)
    if goal is None:
        return None
    profile, race_date, _ = goal
    return (race_date - on_date).days, profile.taper_days


def _is_taper_date(config: dict, on_date: date) -> bool:
    timing = _race_timing(config, on_date)
    return bool(timing and 0 < timing[0] <= timing[1])


def _is_taper_or_race_date(config: dict, on_date: date) -> bool:
    timing = _race_timing(config, on_date)
    return bool(timing and 0 <= timing[0] <= timing[1])


def _distance_options(
    minimum: float,
    maximum: float,
    *,
    step_miles: float = 0.25,
) -> list[float]:
    """Return an inclusive distance grid, preserving a narrow off-grid band."""

    if maximum < minimum - 1e-9:
        raise ValueError("Maximum distance cannot be below minimum distance")
    maximum = max(minimum, maximum)
    scale = round(1.0 / step_miles)
    first = ceil(minimum * scale - 1e-9)
    last = floor(maximum * scale + 1e-9)
    if first <= last:
        return [value / scale for value in range(first, last + 1)]
    return [round((minimum + maximum) / 2.0, 4)]


def _distance_dp_units(miles: float) -> int:
    return round(miles * DISTANCE_DP_SCALE)


@dataclass(frozen=True, slots=True)
class PlanningActivity:
    start_time: datetime
    distance_miles: float
    moving_minutes: float | None = None
    easy_minutes: float | None = None
    baseline_eligible: bool = True
    target_distance_miles: float | None = None


def adherence_normalized_distance(
    actual_miles: float,
    prescribed_range_miles: tuple[float, float] | None,
) -> float:
    """Return prescribed midpoint plus only out-of-range execution deviation.

    The actual distance remains available for observed easy-run baselines. This
    normalized value is only for the conditional mileage/capacity ledger, where
    every value inside a prescribed range must count as equally compliant.
    """

    actual = max(0.0, actual_miles)
    if prescribed_range_miles is None:
        return actual
    low, high = prescribed_range_miles
    low = max(0.0, low)
    high = max(low, high)
    midpoint = (low + high) / 2.0
    if actual < low:
        return max(0.0, midpoint - (low - actual))
    if actual > high:
        return midpoint + (actual - high)
    return midpoint


def _target_activity_miles(activity: PlanningActivity) -> float:
    return (
        activity.target_distance_miles
        if activity.target_distance_miles is not None
        else activity.distance_miles
    )


def expected_compliance_activity(
    recommendation: RecommendationResponse,
    *,
    pace_min_mile: float,
) -> PlanningActivity | None:
    """Convert one prescription into forecast-only target evidence.

    The midpoint is the neutral adherence assumption.  Projected work may
    update rolling continuity and conditional capacity at a later decision
    boundary, but it must not become an observed ordinary-easy sample: doing
    so would let the planner rewrite its own baseline before the athlete runs.
    """

    if (
        recommendation.workout_type == WorkoutType.REST
        or recommendation.planned_for is None
        or recommendation.distance_range_miles is None
    ):
        return None
    midpoint = sum(recommendation.distance_range_miles) / 2.0
    if midpoint <= 0:
        return None
    moving_minutes = max(
        structured_duration_minutes(recommendation) or 0.0,
        midpoint * max(1.0, pace_min_mile),
    )
    moderate_minutes, hard_minutes = prescribed_zone_minutes(
        recommendation,
        moving_minutes,
    )
    return PlanningActivity(
        start_time=recommendation.planned_for,
        distance_miles=midpoint,
        moving_minutes=moving_minutes,
        easy_minutes=max(
            0.0,
            moving_minutes - moderate_minutes - hard_minutes,
        ),
        baseline_eligible=False,
    )


def derive_expected_compliance_target(
    observed_activities: list[PlanningActivity],
    recommendations: list[RecommendationResponse],
    as_of: datetime,
    config: dict,
    *,
    pace_min_mile: float,
) -> tuple[int, tuple[float, float], WeeklyTargetEvidence]:
    """Derive the conditional target after prescriptions completed on plan.

    Expected sessions are strictly causal: a workout contributes only after
    its scheduled timestamp.  Recorded history remains untouched and callers
    must replace an expectation with the uploaded activity rather than retain
    both.
    """

    expected = [
        activity
        for recommendation in recommendations
        if recommendation.planned_for is not None
        and recommendation.planned_for < as_of
        if (activity := expected_compliance_activity(
            recommendation,
            pace_min_mile=pace_min_mile,
        ))
        is not None
    ]
    return derive_weekly_target(
        [*observed_activities, *expected],
        as_of,
        config,
    )


@dataclass(frozen=True, slots=True)
class _CandidateSession:
    offset: int
    state: FitnessState
    recommendation: RecommendationResponse


_CandidatePrefixCache = dict[tuple[int, ...], tuple[_CandidateSession, ...]]
_RecommendationLoadCache = dict[
    tuple[int, float], tuple[RecommendationResponse, float]
]
_ProjectedStateCache = dict[
    tuple[int, tuple[int, ...]],
    tuple[FitnessState, tuple[RecommendationResponse, ...], FitnessState],
]
ExpectedTargetProjector = Callable[
    [list[RecommendationResponse], int],
    tuple[tuple[float, float], ...],
]


def expected_compliance_target_trajectory(
    observed_activities: list[PlanningActivity],
    recommendations: list[RecommendationResponse],
    start_date: date,
    horizon_days: int,
    config: dict,
    *,
    pace_min_mile: float,
    observed_trajectory: tuple[
        tuple[
            datetime,
            tuple[float, float],
            WeeklyTargetEvidence,
        ],
        ...,
    ]
    | None = None,
    expected_activities: list[PlanningActivity] | None = None,
) -> tuple[tuple[float, float], ...]:
    """Return the target expected at each future local-day boundary."""

    if horizon_days <= 0:
        return ()
    zone = ZoneInfo(str(config.get("timezone_default", "UTC")))
    if observed_trajectory is None:
        return tuple(
            derive_expected_compliance_target(
                observed_activities,
                recommendations,
                datetime.combine(
                    start_date + timedelta(days=offset),
                    datetime.min.time(),
                    tzinfo=zone,
                ),
                config,
                pace_min_mile=pace_min_mile,
            )[1]
            for offset in range(horizon_days)
        )
    expected = (
        list(expected_activities)
        if expected_activities is not None
        else [
            activity
            for recommendation in recommendations
            if (activity := expected_compliance_activity(
                recommendation,
                pace_min_mile=pace_min_mile,
            ))
            is not None
        ]
    )
    capacity_daily: dict[date, float] = {}
    for item in [*observed_activities, *expected]:
        day = item.start_time.astimezone(zone).date()
        capacity_daily[day] = (
            capacity_daily.get(day, 0.0) + _target_activity_miles(item)
        )
    capacity_first = min(capacity_daily, default=start_date)
    capacity_last = max(
        capacity_daily,
        default=start_date + timedelta(days=horizon_days - 1),
    )
    # ``projected_capacity`` never considers a window endpoint more than 364
    # days before its boundary, and each endpoint's 28-day window reaches at
    # most another 27 days back. Older activity cannot affect this trajectory.
    # Bounding the prefix here avoids rebuilding years of unreachable daily
    # history for every candidate evaluated by the distance allocator.
    prefix_start = max(capacity_first, start_date - timedelta(days=391))
    prefix_end = max(
        capacity_last,
        start_date + timedelta(days=horizon_days - 1),
    )
    prefix: list[float] = [0.0]
    cursor = prefix_start
    while cursor <= prefix_end:
        prefix.append(prefix[-1] + capacity_daily.get(cursor, 0.0))
        cursor += timedelta(days=1)

    daily_count = len(prefix) - 1
    completed_28d_weekly_by_end = tuple(
        max(
            0.0,
            prefix[end_index + 1] - prefix[max(0, end_index - 27)],
        )
        / 4.0
        for end_index in range(daily_count)
    )
    completed_7d_by_end = tuple(
        max(
            0.0,
            prefix[end_index + 1] - prefix[max(0, end_index - 6)],
        )
        for end_index in range(daily_count)
    )

    retention_half_life = float(
        config.get("coaching", {}).get(
            "capacity_retention_half_life_days",
            84,
        )
    )
    retention_grace = int(
        config.get("coaching", {}).get(
            "capacity_retention_grace_days",
            28,
        )
    )
    capacity_decay_by_age = tuple(
        0.5
        ** (
            max(0, age - retention_grace)
            / max(1.0, retention_half_life)
        )
        for age in range(365)
    )

    def projected_capacity(boundary: datetime) -> float:
        """Mirror distance_capacity with integer-indexed prefix windows."""

        end = boundary.astimezone(zone).date()
        end_index = (end - prefix_start).days
        unavailable_today = sum(
            _target_activity_miles(item)
            for item in expected
            if item.start_time.astimezone(zone).date() == end
            and item.start_time >= boundary
        )

        prior_left = max(0, end_index - 35)
        prior_right = min(len(prefix) - 1, end_index - 7)
        prior_weekly = max(
            0.0,
            prefix[prior_right] - prefix[prior_left]
            if prior_right > prior_left
            else 0.0,
        ) / 4.0
        history_start_index = (
            max(capacity_first, end - timedelta(days=364)) - prefix_start
        ).days
        completed_window_indices = range(
            history_start_index,
            end_index - 6,
        )
        best_value = max(
            (
                completed_28d_weekly_by_end[candidate_index]
                for candidate_index in completed_window_indices
            ),
            default=0.0,
        )
        retained = max(
            (
                completed_28d_weekly_by_end[window_end_index]
                * capacity_decay_by_age[end_index - window_end_index]
                for window_end_index in completed_window_indices
            ),
            default=0.0,
        )
        recent_confirmation = max(
            (
                min(
                    max(
                        0.0,
                        completed_7d_by_end[candidate_index]
                        - (
                            unavailable_today
                            if candidate_index == end_index
                            else 0.0
                        ),
                    ),
                    best_value,
                )
                * capacity_decay_by_age[end_index - candidate_index]
                for candidate_index in range(
                    history_start_index,
                    end_index + 1,
                )
            ),
            default=0.0,
        )
        return max(prior_weekly, retained, recent_confirmation)

    general_progression = min(
        0.10,
        max(
            0.0,
            float(
                config.get("coaching", {}).get(
                    "general_fitness_progression_fraction",
                    0.08,
                )
            ),
        ),
    )
    result: list[tuple[float, float]] = []
    opening_capacity = (
        observed_trajectory[0][2].capacity_reference_miles
        if observed_trajectory
        else 0.0
    )
    for boundary, observed_range, evidence in observed_trajectory[:horizon_days]:
        # Goal trajectories and baseline acquisition have additional target
        # rules. Keep their observed calculation until the same conditional
        # transformation is explicitly modeled for those modes.
        if (
            evidence.planning_mode != WeeklyPlanningMode.ESTABLISHED
            or configured_race_goal(config, on_date=boundary.date()) is not None
        ):
            result.append(observed_range)
            continue
        eligible = [
            item
            for item in expected
            if item.start_time < boundary
        ]
        end = boundary.astimezone(zone).date()
        added_7 = sum(
            _target_activity_miles(item)
            for item in eligible
            if end - timedelta(days=6)
            <= item.start_time.astimezone(zone).date()
            <= end
        )
        added_14_weekly = sum(
            _target_activity_miles(item)
            for item in eligible
            if end - timedelta(days=13)
            <= item.start_time.astimezone(zone).date()
            <= end
        ) / 2.0
        added_21_weekly = sum(
            _target_activity_miles(item)
            for item in eligible
            if end - timedelta(days=20)
            <= item.start_time.astimezone(zone).date()
            <= end
        ) / 3.0
        # A compliant continuation refreshes retained capacity rather than
        # allowing it to decay merely because the observed-only counterfactual
        # eventually runs out of activities. Until projected work has itself
        # completed a full capacity window, carry the stronger observed/opening
        # reference forward; do not invent capacity growth from a prescription.
        capacity_reference = max(
            opening_capacity,
            evidence.capacity_reference_miles,
            projected_capacity(boundary),
        )
        continuity = _progression_continuity(
            capacity_reference,
            evidence.recent_7d_miles + added_7,
            evidence.recent_14d_weekly_miles + added_14_weekly,
            evidence.recent_21d_weekly_miles + added_21_weekly,
        )
        midpoint = capacity_reference * (
            1.0 + general_progression * continuity
        )
        low = _tenth_mile(midpoint * 0.97)
        high = max(low, _tenth_mile(midpoint * 1.03))
        result.append((low, high))
    if len(result) != horizon_days:
        raise ValueError("Observed target trajectory must match the horizon")
    return tuple(result)


def make_expected_target_projector(
    observed_activities: list[PlanningActivity],
    start_date: date,
    config: dict,
    *,
    pace_min_mile: float,
    maximum_horizon_days: int,
    opening_target_range: tuple[float, float] | None = None,
    opening_evidence: WeeklyTargetEvidence | None = None,
    ordinary_easy_midpoint_miles: float | None = None,
    prior_schedule: WeeklyScheduleResponse | None = None,
) -> ExpectedTargetProjector:
    """Build one candidate-independent expected-compliance trajectory.

    A candidate must not lower the target used to score itself by omitting a
    session or moving it beyond the hard horizon.  Construct one neutral,
    athlete-relative continuation from observed history and opening evidence,
    then give every candidate the same immutable trajectory. The earliest
    commitment from the prior plan is included so exact compliance is
    anticipated across its upload. Once that session moves behind the
    decision boundary it is already observed; a later prescription is not
    silently promoted into target evidence. This prevents the plan from
    raising its own target one future workout at a time.
    """

    zone = ZoneInfo(str(config.get("timezone_default", "UTC")))
    boundary = datetime.combine(
        start_date,
        datetime.min.time(),
        tzinfo=zone,
    )
    prior_recommendations = [
        day.recommendation
        for day in (
            (prior_schedule.planning_days or prior_schedule.days)
            if prior_schedule is not None
            else []
        )
        if day.recommendation is not None
        and day.recommendation.planned_for is not None
    ]
    earliest_prior = min(
        prior_recommendations,
        key=lambda item: item.planned_for,
        default=None,
    )
    committed_recommendations = [
        earliest_prior
    ] if (
        earliest_prior is not None
        and boundary <= earliest_prior.planned_for
        < boundary + timedelta(days=4)
    ) else []
    committed_activities = [
        activity
        for recommendation in committed_recommendations
        if (
            activity := expected_compliance_activity(
                recommendation,
                pace_min_mile=pace_min_mile,
            )
        )
        is not None
    ]
    reference_observed_activities = [
        *observed_activities,
        *committed_activities,
    ]
    prepared_horizon_days = maximum_horizon_days + int(
        SOFT_CONTINUATION_DAYS
    )
    observed_trajectory = []
    for offset in range(prepared_horizon_days):
        trajectory_boundary = datetime.combine(
            start_date + timedelta(days=offset),
            datetime.min.time(),
            tzinfo=zone,
        )
        if (
            offset == 0
            and opening_target_range is not None
            and opening_evidence is not None
        ):
            target_range = opening_target_range
            evidence = opening_evidence
        else:
            _, target_range, evidence = derive_weekly_target(
                reference_observed_activities,
                trajectory_boundary,
                config,
            )
        observed_trajectory.append(
            (trajectory_boundary, target_range, evidence)
        )
    fixed_observed_trajectory = tuple(observed_trajectory)
    if opening_target_range is None or opening_evidence is None:
        _, derived_target, derived_evidence = derive_weekly_target(
            observed_activities,
            datetime.combine(
                start_date,
                datetime.min.time(),
                tzinfo=zone,
            ),
            config,
        )
        opening_target_range = opening_target_range or derived_target
        opening_evidence = opening_evidence or derived_evidence
    recent_eligible = [
        item.distance_miles
        for item in observed_activities
        if item.baseline_eligible and item.distance_miles > 0
        and item.start_time.astimezone(zone).date()
        >= start_date - timedelta(days=28)
    ]
    neutral_session_miles = max(
        0.5,
        ordinary_easy_midpoint_miles or 0.0,
        median(recent_eligible) if recent_eligible else 0.0,
        opening_evidence.baseline_session_miles or 0.0,
    )
    cadence_target_range = opening_target_range
    if committed_activities:
        committed_boundary = datetime.combine(
            max(
                item.start_time.astimezone(zone).date()
                for item in committed_activities
            )
            + timedelta(days=1),
            datetime.min.time(),
            tzinfo=zone,
        )
        _, cadence_target_range, _ = derive_weekly_target(
            reference_observed_activities,
            committed_boundary,
            config,
        )
    target_midpoint = sum(cadence_target_range) / 2.0
    sessions_per_week = min(
        7.0,
        max(1.0, target_midpoint / neutral_session_miles),
    )
    gap_days = VISIBLE_HORIZON_DAYS / sessions_per_week
    prior_times = [
        item.start_time.astimezone(zone)
        for item in observed_activities
        if item.start_time.astimezone(zone) < boundary
    ]
    prior_times.extend(
        item.start_time.astimezone(zone)
        for item in committed_activities
    )
    cursor = (
        max(prior_times) + timedelta(days=gap_days)
        if prior_times
        else boundary + timedelta(days=gap_days)
    )
    while cursor < boundary:
        cursor += timedelta(days=gap_days)
    continuation_end = boundary + timedelta(days=prepared_horizon_days)
    neutral_activities: list[PlanningActivity] = []
    while cursor < continuation_end:
        moving_minutes = neutral_session_miles * max(1.0, pace_min_mile)
        neutral_activities.append(
            PlanningActivity(
                start_time=cursor,
                distance_miles=neutral_session_miles,
                moving_minutes=moving_minutes,
                easy_minutes=moving_minutes,
                baseline_eligible=False,
                target_distance_miles=neutral_session_miles,
            )
        )
        cursor += timedelta(days=gap_days)
    reference_activities = sorted(
        neutral_activities,
        key=lambda item: item.start_time,
    )
    prior_target_values = (
        list(prior_schedule.expected_target_trajectory)
        if prior_schedule is not None
        and prior_schedule.expected_target_start_date is not None
        else []
    )
    prior_target_shift = (
        (start_date - prior_schedule.expected_target_start_date).days
        if prior_schedule is not None
        and prior_schedule.expected_target_start_date is not None
        else -1
    )
    expected_opening_from_prior = (
        prior_target_values[prior_target_shift]
        if 0 <= prior_target_shift < len(prior_target_values)
        else None
    )
    opening_matches_prior = bool(
        expected_opening_from_prior is not None
        and opening_target_range is not None
        and all(
            abs(actual - expected) <= 0.11
            for actual, expected in zip(
                opening_target_range,
                expected_opening_from_prior,
            )
        )
    )
    completed_earliest_commitment = False
    if earliest_prior is not None and earliest_prior.planned_for < boundary:
        prescribed_range = earliest_prior.distance_range_miles
        completed_earliest_commitment = any(
            item.start_time.astimezone(zone).date()
            == earliest_prior.planned_for.astimezone(zone).date()
            and (
                prescribed_range is None
                or prescribed_range[0] - 1e-9
                <= item.distance_miles
                <= prescribed_range[1] + 1e-9
            )
            for item in observed_activities
        )
    reuse_prior_target = bool(
        prior_target_values
        and prior_target_shift >= 0
        and (opening_matches_prior or completed_earliest_commitment)
    )
    cache: dict[int, tuple[tuple[float, float], ...]] = {}

    def projector(
        recommendations: list[RecommendationResponse],
        horizon_days: int,
    ) -> tuple[tuple[float, float], ...]:
        del recommendations
        if horizon_days > prepared_horizon_days:
            raise ValueError(
                "Expected target horizon exceeds the prepared trajectory"
            )
        if horizon_days not in cache:
            projected = list(expected_compliance_target_trajectory(
                reference_observed_activities,
                [],
                start_date,
                horizon_days,
                config,
                pace_min_mile=pace_min_mile,
                observed_trajectory=fixed_observed_trajectory,
                expected_activities=reference_activities,
            ))
            if reuse_prior_target:
                overlap = min(
                    horizon_days,
                    len(prior_target_values) - prior_target_shift,
                )
                projected[:overlap] = prior_target_values[
                    prior_target_shift:prior_target_shift + overlap
                ]
            # The opening target is authoritative for this decision boundary.
            # A compliant completion may preserve the prior curve even when
            # its shifted first value differs slightly from newly derived
            # evidence. Persisting that stale first value makes an otherwise
            # identical clock refresh reject the curve and regenerate every
            # downstream target. Anchor the splice so its own saved output is
            # a fixed point on the next no-evidence refresh.
            if projected and opening_target_range is not None:
                projected[0] = opening_target_range
            cache[horizon_days] = tuple(projected)
        return cache[horizon_days]

    setattr(projector, "start_date", start_date)
    setattr(projector, "prepared_horizon_days", prepared_horizon_days)

    return projector


def _integrated_target_range(
    target_ranges: tuple[tuple[float, float], ...],
) -> tuple[float, float]:
    """Integrate daily miles/week targets into horizon mileage."""

    return (
        sum(value[0] for value in target_ranges) / VISIBLE_HORIZON_DAYS,
        sum(value[1] for value in target_ranges) / VISIBLE_HORIZON_DAYS,
    )


def _average_target_range(
    target_ranges: tuple[tuple[float, float], ...],
) -> tuple[float, float]:
    if not target_ranges:
        return (0.0, 0.0)
    count = len(target_ranges)
    return (
        sum(value[0] for value in target_ranges) / count,
        sum(value[1] for value in target_ranges) / count,
    )


def _half_mile(value: float) -> float:
    return round(value * 2) / 2


def _tenth_mile(value: float) -> float:
    """Preserve planning precision while keeping API values readable."""

    return round(value, 1)


def _continuous_mileage_path_cost(
    daily_miles: list[float],
    target_weekly_range: tuple[float, float],
    *,
    session_tolerance_miles: float,
    opening_weekly_rate: float | None = None,
    half_life_days: float = 7.0,
    target_weekly_ranges: tuple[tuple[float, float], ...] | None = None,
    impulse_hours: list[float | None] | None = None,
    opening_hour: float = 0.0,
    underload_only: bool = False,
    discount_half_life_days: float | None = None,
) -> float:
    """Score cumulative load against a boundary-free daily target path.

    Weekly mileage remains the familiar display unit, but the optimizer sees
    a continuous per-day rate across every supplied planning day. One ordinary
    session of cumulative timing variation is free; larger deficits or
    surpluses accrue smoothly instead of changing at day 7 or day 14. The
    interior is deliberately free so this term does not reward daily mileage
    smoothing over a recoverable, human training rhythm.
    """

    tolerance = max(0.1, session_tolerance_miles)
    cost = 0.0
    target_curve = (
        target_weekly_ranges
        if target_weekly_ranges is not None
        and len(target_weekly_ranges) == len(daily_miles)
        else tuple(target_weekly_range for _ in daily_miles)
    )

    def day_weight(elapsed_days: int) -> float:
        if discount_half_life_days is None:
            return 1.0
        return 0.5 ** (
            max(0, elapsed_days - 1)
            / max(0.1, discount_half_life_days)
        )

    if opening_weekly_rate is not None:
        rate = max(0.0, opening_weekly_rate)
        half_life = max(0.1, half_life_days)
        daily_decay = 0.5 ** (1.0 / half_life)
        session_normalization = log(2.0) * 7.0 / half_life
        opening_low = min(rate, target_curve[0][0])
        opening_high = max(rate, target_curve[0][1])
        for elapsed_days, (miles, current_target) in enumerate(
            zip(daily_miles, target_curve),
            start=1,
        ):
            impulse_hour = (
                impulse_hours[elapsed_days - 1]
                if impulse_hours is not None
                and elapsed_days - 1 < len(impulse_hours)
                else None
            )
            rate = _advance_continuous_mileage_rate(
                rate,
                miles,
                daily_decay=daily_decay,
                session_normalization=session_normalization,
                impulse_hour=impulse_hour,
                opening_hour=(opening_hour if elapsed_days == 1 else 0.0),
            )
            elapsed_from_opening = (
                (24.0 - min(24.0, max(0.0, opening_hour))) / 24.0
                + elapsed_days
                - 1
            )
            path_decay = daily_decay**elapsed_from_opening
            lower_path = current_target[0] + (
                opening_low - target_curve[0][0]
            ) * path_decay
            upper_path = current_target[1] + (
                opening_high - target_curve[0][1]
            ) * path_decay
            underload = max(0.0, lower_path - tolerance - rate)
            overload = max(0.0, rate - upper_path - tolerance)
            excess = (
                underload
                if underload_only
                else max(underload, overload)
            )
            cost += day_weight(elapsed_days) * (excess / tolerance) ** 2
        return cost

    cumulative = 0.0
    cumulative_low = 0.0
    cumulative_high = 0.0
    for elapsed_days, (miles, current_target) in enumerate(
        zip(daily_miles, target_curve),
        start=1,
    ):
        cumulative += max(0.0, miles)
        cumulative_low += current_target[0] / VISIBLE_HORIZON_DAYS
        cumulative_high += current_target[1] / VISIBLE_HORIZON_DAYS
        underload = max(0.0, cumulative_low - tolerance - cumulative)
        overload = max(0.0, cumulative - cumulative_high - tolerance)
        excess = underload if underload_only else max(underload, overload)
        cost += day_weight(elapsed_days) * (excess / tolerance) ** 2
    return cost


def _continuous_mileage_path_violation(
    daily_miles: list[float],
    target_weekly_range: tuple[float, float],
    *,
    session_tolerance_miles: float,
    opening_weekly_rate: float | None = None,
    half_life_days: float = 7.0,
    target_weekly_ranges: tuple[tuple[float, float], ...] | None = None,
    impulse_hours: list[float | None] | None = None,
    opening_hour: float = 0.0,
    overload_only: bool = False,
) -> float:
    """Return the worst cumulative breach outside the session corridor."""

    tolerance = max(0.1, session_tolerance_miles)
    worst = 0.0
    target_curve = (
        target_weekly_ranges
        if target_weekly_ranges is not None
        and len(target_weekly_ranges) == len(daily_miles)
        else tuple(target_weekly_range for _ in daily_miles)
    )
    if opening_weekly_rate is not None:
        rate = max(0.0, opening_weekly_rate)
        half_life = max(0.1, half_life_days)
        daily_decay = 0.5 ** (1.0 / half_life)
        session_normalization = log(2.0) * 7.0 / half_life
        opening_low = min(rate, target_curve[0][0])
        opening_high = max(rate, target_curve[0][1])
        for elapsed_days, (miles, current_target) in enumerate(
            zip(daily_miles, target_curve),
            start=1,
        ):
            impulse_hour = (
                impulse_hours[elapsed_days - 1]
                if impulse_hours is not None
                and elapsed_days - 1 < len(impulse_hours)
                else None
            )
            rate = _advance_continuous_mileage_rate(
                rate,
                miles,
                daily_decay=daily_decay,
                session_normalization=session_normalization,
                impulse_hour=impulse_hour,
                opening_hour=(opening_hour if elapsed_days == 1 else 0.0),
            )
            elapsed_from_opening = (
                (24.0 - min(24.0, max(0.0, opening_hour))) / 24.0
                + elapsed_days
                - 1
            )
            path_decay = daily_decay**elapsed_from_opening
            lower_path = current_target[0] + (
                opening_low - target_curve[0][0]
            ) * path_decay
            upper_path = current_target[1] + (
                opening_high - target_curve[0][1]
            ) * path_decay
            worst = max(worst, rate - upper_path - tolerance)
            if not overload_only:
                worst = max(worst, lower_path - tolerance - rate)
        return max(0.0, worst)

    cumulative = 0.0
    cumulative_low = 0.0
    cumulative_high = 0.0
    for miles, current_target in zip(daily_miles, target_curve):
        cumulative += max(0.0, miles)
        cumulative_low += current_target[0] / VISIBLE_HORIZON_DAYS
        cumulative_high += current_target[1] / VISIBLE_HORIZON_DAYS
        lower = cumulative_low - tolerance
        upper = cumulative_high + tolerance
        worst = max(worst, cumulative - upper)
        if not overload_only:
            worst = max(worst, lower - cumulative)
    return max(0.0, worst)


def _advance_continuous_mileage_rate(
    opening_rate: float,
    miles: float,
    *,
    daily_decay: float,
    session_normalization: float,
    impulse_hour: float | None,
    opening_hour: float = 0.0,
) -> float:
    """Advance one local day and retain the session's actual clock time.

    The old daily-bin update applied a full day of decay to prior load and no
    decay to that day's run, effectively placing every prescription at the
    following midnight.  After an evening upload, the next plan instead began
    with the real run already partially decayed.  Those two representations of
    compliant work must meet at the same next-midnight state.

    ``None`` preserves the legacy end-of-day convention for callers that have
    only daily totals and no session timestamp.
    """

    start_hour = min(24.0, max(0.0, opening_hour))
    hour = 24.0 if impulse_hour is None else min(24.0, max(start_hour, impulse_hour))
    opening_decay = daily_decay ** ((24.0 - start_hour) / 24.0)
    after_impulse_decay = daily_decay ** ((24.0 - hour) / 24.0)
    return (
        max(0.0, opening_rate) * opening_decay
        + max(0.0, miles) * session_normalization * after_impulse_decay
    )


def _planned_impulse_hours(days: list[WeeklyScheduleDay]) -> list[float | None]:
    """Return local clock hours aligned with one day of planned mileage."""

    values: list[float | None] = []
    for day in days:
        result = day.recommendation
        planned_for = result.planned_for if result is not None else None
        if planned_for is None or planned_for.date() != day.date:
            values.append(None)
            continue
        values.append(
            planned_for.hour
            + planned_for.minute / 60.0
            + planned_for.second / 3600.0
        )
    return values


def _clock_hour(value: datetime) -> float:
    """Return a fractional local hour for continuous within-day decay."""

    return value.hour + value.minute / 60.0 + value.second / 3600.0


def _hard_horizon_program_rate(
    days: list[WeeklyScheduleDay],
) -> float | None:
    """Return prescribed weekly-equivalent mileage across every hard day.

    Empty opening and terminal days are part of the 21-day decision, not
    evidence that an unseen recurring session exists outside it. Recovery's
    soft continuation prices terminal residual load separately; it never
    supplies mileage credit to the hard horizon.
    """

    if not days:
        return None
    prescribed_miles = sum(
        _midpoint(day.recommendation)
        for day in days
        if day.recommendation is not None
        and day.recommendation.workout_type != WorkoutType.REST
    )
    return prescribed_miles * VISIBLE_HORIZON_DAYS / len(days)


def _phase_normalized_program_rate(
    days: list[WeeklyScheduleDay],
) -> float | None:
    """Estimate the recurring rate without treating either edge as a deadline.

    One representative gap closes the observed first-to-last session span.
    This makes the estimate insensitive to whether the next recurrence lands
    just inside or outside day 21. It describes sustained intent only; the
    separate hard-horizon floor prevents this estimate from excusing a plan
    that omits more work than boundary phase can reasonably explain.
    """

    sessions = [
        (index, _midpoint(day.recommendation))
        for index, day in enumerate(days)
        if day.recommendation is not None
        and day.recommendation.workout_type != WorkoutType.REST
    ]
    if not sessions:
        return None
    if len(sessions) == 1:
        return _hard_horizon_program_rate(days)
    gaps = [
        current[0] - previous[0]
        for previous, current in zip(sessions, sessions[1:])
    ]
    recurring_span_days = (
        sessions[-1][0] - sessions[0][0] + float(median(gaps))
    )
    if recurring_span_days <= 0:
        return _hard_horizon_program_rate(days)
    return (
        sum(distance for _, distance in sessions)
        * VISIBLE_HORIZON_DAYS
        / recurring_span_days
    )


def _soft_projected_program_rate(
    days: list[WeeklyScheduleDay],
    target_range: tuple[float, float],
    ordinary_session_miles: float,
    *,
    continuation_days: float = SOFT_CONTINUATION_DAYS,
    boundary_session: tuple[float, float] | None = None,
) -> float | None:
    """Estimate recurring rate with a cheap candidate-independent tail.

    The hard horizon may end just before the next ordinary recurrence.  Extend
    every candidate by the same target-derived ordinary cadence so that this
    phase difference does not make an otherwise identical rolling program
    appear over- or under-funded.  The tail is used only for the soft recurring
    rate; it supplies no credit to the separate hard-horizon funding floor.
    """

    sessions = [
        (float(index), _midpoint(day.recommendation))
        for index, day in enumerate(days)
        if day.recommendation is not None
        and day.recommendation.workout_type != WorkoutType.REST
    ]
    if boundary_session is not None:
        boundary_offset, boundary_miles = boundary_session
        if boundary_offset < 0 and boundary_miles > 0:
            sessions.append((boundary_offset, boundary_miles))
            sessions.sort()
    if not sessions:
        return None
    target_midpoint = sum(target_range) / 2.0
    ordinary = max(0.1, ordinary_session_miles)
    if target_midpoint <= 0 or continuation_days <= 0:
        return _phase_normalized_program_rate(days)
    sessions_per_week = min(7.0, max(1.0, target_midpoint / ordinary))
    reference_gap_days = VISIBLE_HORIZON_DAYS / sessions_per_week
    cursor = float(len(days)) + reference_gap_days
    continuation_end = float(len(days)) + continuation_days
    while cursor <= continuation_end + 1e-9:
        sessions.append((cursor, ordinary))
        cursor += reference_gap_days
    recurring_span_days = (
        sessions[-1][0] - sessions[0][0] + reference_gap_days
    )
    if recurring_span_days <= 0:
        return _phase_normalized_program_rate(days)
    return (
        sum(distance for _, distance in sessions)
        * VISIBLE_HORIZON_DAYS
        / recurring_span_days
    )


def _recent_boundary_session(
    state: FitnessState,
) -> tuple[float, float] | None:
    """Return a recent completed run crossing the decision's left edge.

    Replanning begins on the day after a completed run. A recurring-rate
    estimate that sees only future prescriptions otherwise forgets that run
    exactly when it moves outside the decision array. Keep one observed
    boundary session as phase context; older work remains represented by the
    continuous-load state but is no longer part of this cadence window.
    """

    days_since = state.days_since_last_run
    distance = state.last_run_distance_miles
    if distance is None and state.last_run_prescribed_distance_range_miles:
        distance = sum(state.last_run_prescribed_distance_range_miles) / 2
    if (
        days_since is None
        or distance is None
        or distance <= 0
        or days_since > VISIBLE_HORIZON_DAYS
    ):
        return None
    return (-max(1e-6, days_since), distance)


def _hard_horizon_funding_violation(
    days: list[WeeklyScheduleDay],
    target_miles: tuple[float, float],
    *,
    boundary_allowance_miles: float,
) -> float:
    """Return only funding drift too large to be normal horizon sawtooth.

    A receding boundary can contain N or N+1 occurrences of the same recurring
    policy. Allowing one ordinary session keeps those phases equivalent; any
    larger miss is real underfunding rather than an edge effect.
    """

    actual_miles = sum(
        _midpoint(day.recommendation)
        for day in days
        if day.recommendation is not None
        and day.recommendation.workout_type != WorkoutType.REST
    )
    allowance = max(0.0, boundary_allowance_miles)
    return max(
        0.0,
        target_miles[0] - allowance - actual_miles,
        actual_miles - target_miles[1] - allowance,
    )


def _gate_aligned_hard_funding_violation(
    violation_miles: float,
    ordinary_easy_midpoint: float,
) -> float:
    """Keep a plan that would fail the funding gate noncompetitive."""

    violation = max(0.0, violation_miles)
    if violation <= HARD_FUNDING_ROUNDING_TOLERANCE_MILES + 1e-9:
        return violation
    # The continuous target and one-session boundary allowance already make
    # ordinary horizon sawtooth free. Crossing the remaining release-gate
    # tolerance is a feasibility failure, not a cheap soft underload option.
    return violation + max(0.1, ordinary_easy_midpoint)


def _aerobic_shortfall_fraction(
    prescribed_midpoint: float,
    normal_easy_low: float,
) -> float:
    """Return the additive share missing from an ordinary aerobic session."""

    if normal_easy_low <= 0:
        return 0.0
    return max(0.0, normal_easy_low - prescribed_midpoint) / normal_easy_low


def _rolling_aerobic_support_violation(
    days: list[WeeklyScheduleDay],
    daily_states: list[FitnessState],
) -> float:
    """Return missing ordinary-easy support across the rolling horizon.

    Long and quality mileage can numerically fund a target while producing a
    poor program with no ordinary aerobic support. Require one athlete-relative
    easy-session equivalent per seven days of lookahead, with one whole-session
    allowance for the arbitrary receding-horizon edge. This is a continuous
    horizon rate, not a calendar-week quota. A medium-long run supplies at most
    one support equivalent; extra distance does not replace another easy day.
    """

    required_equivalents = max(
        0.0,
        len(days) / VISIBLE_HORIZON_DAYS - 1.0,
    )
    supplied_equivalents = 0.0
    for index, day in enumerate(days):
        result = day.recommendation
        if result is None or result.workout_type != WorkoutType.EASY:
            continue
        ordinary_low = typical_easy_distance(daily_states[index])[0]
        if ordinary_low <= 0:
            continue
        supplied_equivalents += min(
            1.0,
            _midpoint(result) / ordinary_low,
        )
    return max(0.0, required_equivalents - supplied_equivalents)


def _peak_projected_continuous_mileage_rate(
    opening_weekly_rate: float | None,
    opening_at: datetime,
    days: list[WeeklyScheduleDay],
    *,
    half_life_days: float,
) -> float | None:
    """Project the visible plan on the same boundary-free mileage scale.

    The optimizer evaluates several load signals, but the interface previously
    compared a literal seven-day card total with retained weekly capacity. This
    projection keeps those concepts separate: completed work is already in the
    opening rate, planned midpoint mileage is added at its actual scheduled
    time, and the existing rate decays continuously between sessions.
    """

    if opening_weekly_rate is None:
        return None
    half_life = max(0.1, float(half_life_days))
    session_normalization = log(2.0) * 7.0 / half_life
    rate = max(0.0, opening_weekly_rate)
    peak = rate
    previous_at = opening_at
    planned = sorted(
        (
            day.planned_at,
            sum(day.recommendation.distance_range_miles) / 2,
        )
        for day in days
        if day.planned_at is not None
        and day.recommendation is not None
        and day.recommendation.workout_type != WorkoutType.REST
        and day.recommendation.distance_range_miles is not None
    )
    for planned_at, midpoint_miles in planned:
        elapsed_days = max(
            0.0,
            (planned_at - previous_at).total_seconds() / 86400.0,
        )
        rate *= 0.5 ** (elapsed_days / half_life)
        rate += session_normalization * midpoint_miles
        peak = max(peak, rate)
        previous_at = planned_at
    return peak


def _weekly_rate_alignment_cost(
    projected_weekly_rate: float,
    target_weekly_range: tuple[float, float],
) -> float:
    """Prefer the target midpoint without turning it into a hard quota."""

    target_low, target_high = target_weekly_range
    target_midpoint = sum(target_weekly_range) / 2
    return (
        (projected_weekly_rate - target_midpoint) ** 2 * 8.0
        + max(0.0, target_low - projected_weekly_rate) ** 2 * 30.0
        + max(0.0, projected_weekly_rate - target_high) ** 2 * 40.0
    )


def _progression_continuity(
    capacity_reference: float,
    recent_7d_miles: float,
    recent_14d_weekly_miles: float,
    recent_21d_weekly_miles: float,
) -> float:
    """Measure how continuously retained capacity has been exercised.

    Retained capacity remains the return-to-training anchor. The overlapping
    7-, 14-, and 21-day rates only decide how much *additional* productive
    progression has been earned. A 3:2:1 blend emphasizes current exposure
    without allowing one strong rolling week to erase a preceding off-week.
    """

    if capacity_reference <= 0:
        return 0.0
    ratios = (
        min(1.0, recent_7d_miles / capacity_reference),
        min(1.0, recent_14d_weekly_miles / capacity_reference),
        min(1.0, recent_21d_weekly_miles / capacity_reference),
    )
    weights = (3.0, 2.0, 1.0)
    return sum(ratio * weight for ratio, weight in zip(ratios, weights)) / sum(
        weights
    )


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
            daily[day] = daily.get(day, 0.0) + _target_activity_miles(activity)
    if not daily:
        evidence = WeeklyTargetEvidence(
            recent_7d_miles=0,
            recent_14d_weekly_miles=0,
            recent_21d_weekly_miles=0,
            chronic_42d_weekly_miles=0,
            best_sustained_28d_weekly_miles=0,
            peak_7d_miles=0,
            demonstrated_run_days_per_week=0,
            capacity_reference_miles=0,
            planning_mode=WeeklyPlanningMode.BASELINE_REQUIRED,
            history_run_count=0,
            baseline_session_minutes=None,
            rationale=(
                "No running history is available, so the app will not invent a mileage target. "
                "Import recent runs or record one conversational-effort baseline outing; after "
                "that, baseline sessions repeat observed exposure until weekly capacity is measurable."
            ),
        )
        return 1, (0.0, 0.0), evidence

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
    recent_fourteen_weekly = sum(
        miles
        for activity_date, miles in daily.items()
        if end - timedelta(days=13) <= activity_date <= end
    ) / 2.0
    recent_twenty_one_weekly = sum(
        miles
        for activity_date, miles in daily.items()
        if end - timedelta(days=20) <= activity_date <= end
    ) / 3.0
    half_life = float(config.get("coaching", {}).get("capacity_retention_half_life_days", 84))
    retention_grace = int(config.get("coaching", {}).get("capacity_retention_grace_days", 28))
    shared_capacity = distance_capacity(
        [
            TrainingSession(
                activity_id=index,
                start_time=activity.start_time,
                distance_miles=_target_activity_miles(activity),
                moving_minutes=float(activity.moving_minutes or 0.0),
                zone_load=None,
                hard_minutes=0.0,
            )
            for index, activity in enumerate(activities)
            if _target_activity_miles(activity) > 0
        ],
        as_of,
        retention_grace_days=retention_grace,
        retention_half_life_days=half_life,
    )

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
    # Capacity has one definition across dashboard and planner. A strong acute
    # week is training-load evidence, not instant proof that an older peak has
    # been fully reabsorbed; only sustained 28-day evidence and its configured
    # retention can set this denominator.
    capacity_reference = shared_capacity.reference_miles
    recent_baseline_activities = [
        activity
        for activity in activities
        if activity.distance_miles > 0
        and activity.baseline_eligible
        and end - timedelta(days=27)
        <= activity.start_time.astimezone(zone).date()
        <= end
    ]
    if not recent_baseline_activities:
        recent_baseline_activities = [
            activity
            for activity in sorted(
                (
                    activity
                    for activity in activities
                    if activity.distance_miles > 0
                    and activity.baseline_eligible
                ),
                key=lambda item: item.start_time,
                reverse=True,
            )[:4]
        ]
    recent_activity_distances = [
        activity.distance_miles for activity in recent_baseline_activities
    ]
    recent_activity_minutes = [
        float(activity.moving_minutes)
        for activity in recent_baseline_activities
        if activity.moving_minutes is not None and activity.moving_minutes > 0
    ]
    baseline_session_miles = (
        float(median(recent_activity_distances))
        if recent_activity_distances
        else None
    )
    baseline_session_minutes = (
        float(median(recent_activity_minutes))
        if recent_activity_minutes
        else None
    )
    if baseline_session_miles is None:
        planning_mode = WeeklyPlanningMode.BASELINE_REQUIRED
    elif capacity_reference + 1e-9 < baseline_session_miles:
        planning_mode = WeeklyPlanningMode.BASELINE_BUILDING
    else:
        planning_mode = WeeklyPlanningMode.ESTABLISHED
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
    progression_continuity = (
        _progression_continuity(
            capacity_reference,
            recent_seven,
            recent_fourteen_weekly,
            recent_twenty_one_weekly,
        )
        if planning_mode == WeeklyPlanningMode.ESTABLISHED
        else 0.0
    )
    effective_progression = general_progression * progression_continuity
    goal = configured_race_goal(config, on_date=end)
    if planning_mode == WeeklyPlanningMode.BASELINE_REQUIRED:
        target_low = target_high = 0.0
        goal_trajectory_detail = (
            " Recorded history contains no ordinary easy run suitable for a baseline, "
            "so long, hard, or health-affected mileage is not being repeated as a starter session."
        )
    elif planning_mode == WeeklyPlanningMode.BASELINE_BUILDING:
        assert baseline_session_miles is not None
        target_low = target_high = 0.0
        goal_trajectory_detail = (
            " The weekly-capacity model is not ready to progress mileage yet, so the "
            "next baseline session repeats the median duration actually observed. "
            "Each completed session updates this reference without extrapolating an "
            "unsupported weekly routine."
        )
    elif goal:
        # Goal mode starts from demonstrated training and is subsequently
        # raised only as much as the backward-planned race trajectory needs.
        # Its floor is the same continuity-adjusted productive trajectory as
        # general fitness, so selecting a roomy goal cannot reduce training.
        productive_midpoint = capacity_reference * (1.0 + effective_progression)
        productive_low = _tenth_mile(productive_midpoint * 0.97)
        productive_high = max(
            productive_low,
            _tenth_mile(productive_midpoint * 1.03),
        )
        target_low = max(
            productive_low,
            _tenth_mile(capacity_reference * 0.95),
        )
        target_high = max(
            target_low,
            _tenth_mile(min(peak_seven, capacity_reference * 1.10)),
            productive_high,
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
        productive_midpoint = capacity_reference * (1.0 + effective_progression)
        # The optimizer needs to see the continuity adjustment. Half-mile
        # rounding here can collapse meaningfully different targets into the
        # same band, so retain tenths internally and round individual workout
        # prescriptions separately.
        target_low = _tenth_mile(productive_midpoint * 0.97)
        target_high = max(
            target_low,
            _tenth_mile(productive_midpoint * 1.03),
        )
        goal_trajectory_detail = (
            " In general-fitness mode, successful training earns a modest "
            "percentage-based build above demonstrated capacity. Recent 7-, 14-, "
            f"and 21-day continuity is {progression_continuity:.0%}, earning "
            f"{effective_progression:.1%} of the configured {general_progression:.1%} "
            "growth without reducing retained capacity after one off-week. Recovery "
            "and actual completion determine whether the target becomes the next baseline."
        )
    if goal and planning_mode == WeeklyPlanningMode.ESTABLISHED:
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
                effective_progression,
                min(0.08, required_weekly_rate),
            )
            goal_midpoint = min(
                goal_profile.peak_weekly_miles,
                capacity_reference * (1.0 + planned_weekly_rate),
            )
            goal_trajectory_detail = (
                f" The backward-planned trajectory toward the {goal_profile.label.lower()} "
                "currently fits inside the continuity-adjusted productive target."
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
    if planning_mode == WeeklyPlanningMode.BASELINE_REQUIRED:
        target_runs = 1
    elif planning_mode == WeeklyPlanningMode.BASELINE_BUILDING:
        target_runs = 1
    else:
        target_runs = max(1, min(7, floor(prescribed_frequency + 0.5)))
    evidence = WeeklyTargetEvidence(
        recent_7d_miles=recent_seven,
        recent_14d_weekly_miles=recent_fourteen_weekly,
        recent_21d_weekly_miles=recent_twenty_one_weekly,
        chronic_42d_weekly_miles=chronic_weekly,
        best_sustained_28d_weekly_miles=best_sustained[2],
        peak_7d_miles=peak_seven,
        current_run_days_per_week=recent_frequency,
        demonstrated_run_days_per_week=demonstrated,
        capacity_reference_miles=capacity_reference,
        progression_continuity=progression_continuity,
        earned_progression_fraction=effective_progression,
        planning_mode=planning_mode,
        history_run_count=len(activities),
        baseline_session_miles=baseline_session_miles,
        baseline_session_minutes=baseline_session_minutes,
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
        4: [1, 3, 4, 6],
        5: [1, 2, 3, 5, 6],
        6: [1, 2, 3, 4, 5, 6],
        7: [1, 2, 3, 4, 5, 6],
    }
    typical_rest_days = int((config or {}).get("coaching", {}).get("typical_rest_days_between_runs", 1))
    recovery_too_short_for_today = False
    if typical_rest_days >= 1 and state.days_since_last_run is not None:
        inferred_last_date = (state.as_of - timedelta(days=state.days_since_last_run)).date()
        ran_today = inferred_last_date >= state.as_of.date()
        recovery = estimate_recovery(state)
        ran_yesterday_with_material_recovery_load = (
            inferred_last_date == state.as_of.date() - timedelta(days=1)
            and recovery is not None
            and recovery.hours_until_easy > 0
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
    acute_capacity_ratio = effective_load_ratio(state.recent_load)
    recovery = estimate_recovery(state)
    accumulated_load_manageable = (
        recovery is None or recovery.hours_until_easy <= 0
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
        "accumulated_load_manageable": accumulated_load_manageable,
    }


def _midpoint(result: RecommendationResponse) -> float:
    if result.distance_range_miles:
        return sum(result.distance_range_miles) / 2
    if result.duration_range_minutes:
        pace = 11.0
        return sum(result.duration_range_minutes) / 2 / pace
    return 0.0


def _established_easy_midpoint_floor(
    state: FitnessState,
    result: RecommendationResponse | None = None,
) -> float:
    """Smallest established easy midpoint supported by real history.

    The lower edge of ``typical_easy_distance`` is the athlete-relative
    ordinary-run reference. Add the standard quarter-mile execution margin to
    recover its prescription midpoint. If recovery has already produced a
    shorter standalone prescription, preserve that smaller midpoint instead.
    """

    easy_low, easy_high = typical_easy_distance(state)
    historical_midpoint = (
        easy_low + min(0.25, max(0.0, easy_high - easy_low) / 2)
        if easy_high > 0
        else 0.0
    )
    if result and result.distance_range_miles:
        prescribed_midpoint = _midpoint(result)
        return (
            min(prescribed_midpoint, historical_midpoint)
            if historical_midpoint > 0
            else prescribed_midpoint
        )
    return historical_midpoint


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


def _observed_easy_safety_cap(
    state: FitnessState,
    result: RecommendationResponse,
) -> bool:
    """Identify a shortened easy range justified before candidate planning."""

    if "includes_planned_sessions" in state.recent_load.flags:
        return False
    recovery_pressure = _easy_volume_recovery_pressure(result)
    # Easy readiness means the athlete may run; it does not erase a smaller
    # observed-session volume adjustment that is still present in the exact
    # timed recommendation.  Conversely, a ready result with no such pressure
    # keeps the rollover behavior: an obsolete offset-zero cap is not carried
    # into a later same-day slot.
    if result.readiness.value == "ready" and recovery_pressure <= 0:
        return False
    recovery = estimate_recovery(state)
    weather = assess_training_weather(
        state.planned_weather,
        state.weather_exposure_baseline,
    )
    load_ratio = effective_load_ratio(state.recent_load)
    return bool(
        recovery_pressure > 0
        or state.current_health_status != CurrentHealthStatus.NORMAL
        or (recovery is not None and recovery.hours_until_easy > 0)
        or (load_ratio is not None and load_ratio > 1.0)
        or state.recent_performance_response
        not in {"unknown", "within_recent_range", "stronger_than_recent"}
        or weather.caution
    )


def _easy_volume_recovery_pressure(
    result: RecommendationResponse,
) -> float:
    """Return the continuous easy-volume pressure behind a prescription.

    Recommendation ranges are rounded to half miles for execution. The
    allocator must not infer a physiological hard cap from that display
    rounding: a few minutes of additional recovery can otherwise change an
    upper bound from 3.0 to 3.5 and expose several miles of expansion at once.
    """

    recovery_trace = next(
        (
            item
            for item in result.rule_trace
            if item.rule_id == "recent_recovery_load"
        ),
        None,
    )
    pressure = (
        recovery_trace.facts.get("easy_volume_recovery_pressure")
        if recovery_trace is not None
        else None
    )
    return (
        min(1.0, max(0.0, float(pressure)))
        if isinstance(pressure, (int, float))
        else 0.0
    )


def _session_load_units(
    distance_miles: float,
    easy_reference_miles: float,
    *,
    estimated_duration_ratio: float | None = None,
    prescribed_intensity_factor: float = 1.0,
) -> float:
    """Express projected work through volume, duration, and intensity.

    Distance and duration receive the same normalized 60/40 weighting used by
    completed runs when HR load is unavailable. For an ordinary mileage
    prescription, duration scales with distance at the athlete's own recent
    pace; uploads replace that estimate with what actually happened.
    """

    if easy_reference_miles <= 0:
        return 0.0
    distance_ratio = max(0.0, distance_miles) / easy_reference_miles
    duration_ratio = (
        max(0.0, estimated_duration_ratio)
        if estimated_duration_ratio is not None
        else distance_ratio
    )
    volume_ratio = distance_ratio * 0.60 + duration_ratio * 0.40
    return volume_ratio * max(1.0, prescribed_intensity_factor)


def _prescribed_zone_minutes(
    result: RecommendationResponse,
    total_minutes: float,
) -> tuple[float, float]:
    """Estimate moderate/hard minutes from the actual workout structure."""
    return prescribed_zone_minutes(result, total_minutes)


def _prescribed_intensity_factor(
    result: RecommendationResponse,
    total_minutes: float,
) -> float:
    """Price only the prescribed non-aerobic share, never the workout name."""

    return prescribed_intensity_factor(result, total_minutes)


def _structured_duration_minutes(result: RecommendationResponse) -> float | None:
    """Return the clock duration explicitly represented by workout steps."""
    return structured_duration_minutes(result)


def _recommendation_load_units(
    result: RecommendationResponse,
    easy_reference_miles: float,
) -> float:
    # Calendar placement must remain valid for every dose the prescription
    # calls compliant. Price recovery at the upper edge of a distance range;
    # mileage targets and progression still use the midpoint independently.
    planned_miles = (
        result.distance_range_miles[1]
        if result.distance_range_miles is not None
        else _midpoint(result)
    )
    duration_ratio = (
        planned_miles / easy_reference_miles
        if easy_reference_miles > 0
        else 0.0
    )
    # At planning time, recent pace makes duration proportional to distance.
    # The prescribed work fraction adds the intensity component; recorded HR
    # and actual duration replace both estimates after upload.
    estimated_minutes = max(
        _structured_duration_minutes(result) or 0.0,
        planned_miles * 11.0,
    )
    return _session_load_units(
        planned_miles,
        easy_reference_miles,
        estimated_duration_ratio=duration_ratio,
        prescribed_intensity_factor=_prescribed_intensity_factor(
            result,
            estimated_minutes,
        ),
    )


def _cached_recommendation_load_units(
    result: RecommendationResponse,
    easy_reference_miles: float,
    cache: _RecommendationLoadCache | None,
) -> float:
    """Reuse a pure load calculation inside one schedule search only."""

    if cache is None:
        return _recommendation_load_units(result, easy_reference_miles)
    key = (id(result), easy_reference_miles)
    cached = cache.get(key)
    if cached is not None and cached[0] is result:
        return cached[1]
    value = _recommendation_load_units(result, easy_reference_miles)
    # Holding the model alongside the value prevents object-id reuse while
    # this invocation-local cache is alive.
    cache[key] = (result, value)
    return value


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
            projected_recovery_reference_miles(state),
        )
        if result is not None
        else 1.0
    )
    taxing_next = bool(
        result
        and result.workout_type in {WorkoutType.LONG, *QUALITY_WORKOUT_TYPES}
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
    *,
    planned_load_units: list[float] | None = None,
    recommendation_load_cache: _RecommendationLoadCache | None = None,
) -> float:
    """Project transient session load, allowing it to decay during rest."""
    if planned_load_units is not None and len(planned_load_units) != len(planned):
        raise ValueError("Planned load units must align with planned sessions")
    half_life_hours = RECOVERY_HALF_LIFE_HOURS
    residual = 0.0
    if raw_state.recovery_residual_load is not None:
        residual = decay_recovery_load(
            raw_state.recovery_residual_load,
            max(
                0.0,
                (as_of - raw_state.as_of).total_seconds() / 3600.0,
            ),
            half_life_hours=half_life_hours,
        )
    elif raw_state.last_run and raw_state.days_since_last_run is not None:
        completed_units, _ = athlete_relative_session_load(
            raw_state.last_run,
            prior_typical_load(raw_state),
            performance_response=raw_state.recent_performance_response,
            drift_percent=raw_state.last_run_drift_percent,
            prescribed_intensity_factor=(
                raw_state.last_run_prescribed_intensity_factor
            ),
            prescribed_distance_range_miles=(
                raw_state.last_run_prescribed_distance_range_miles
            ),
        )
        residual += decay_recovery_load(
            completed_units,
            raw_state.days_since_last_run * 24,
            half_life_hours=half_life_hours,
        )
    for position, item in enumerate(planned):
        if (
            item.workout_type == WorkoutType.REST
            or item.planned_for is None
            or item.planned_for >= as_of
        ):
            continue
        elapsed_hours = (as_of - item.planned_for).total_seconds() / 3600
        item_load = (
            planned_load_units[position]
            if planned_load_units is not None
            else _cached_recommendation_load_units(
                item,
                easy_reference_miles,
                recommendation_load_cache,
            )
        )
        residual += decay_recovery_load(
            item_load,
            elapsed_hours,
            half_life_hours=half_life_hours,
        )
    return residual


def _recovery_interaction_cost(
    residual_load: float,
    proposed_load: float,
) -> float:
    """Price only recovery overlap added by the unresolved prior load.

    Load units are normalized so an ordinary easy run is approximately one.
    The two-unit corridor therefore represents the proposed ordinary session
    plus one session's recoverable envelope.  Subtracting the proposed
    session's own overflow gives the interaction a real zero point: a long
    run is not expensive merely because it is long, but becomes progressively
    less attractive when it is placed on top of unresolved work.
    """

    proposed = max(0.0, proposed_load)
    overflow_without_residual = max(0.0, proposed - 2.0)
    overflow_with_residual = max(
        0.0,
        max(0.0, residual_load) + proposed - 2.0,
    )
    return (
        max(0.0, overflow_with_residual - overflow_without_residual)
        * PROGRAM_FIT_UNIT
    )


def _elapsed_cadence_idle_cost(
    planned_times: list[datetime],
    preferred_gap_hours: float,
) -> float:
    """Price only elapsed time beyond a soft, athlete-selected cadence."""

    reference = max(1.0, preferred_gap_hours)
    return (PROGRAM_FIT_UNIT / 2.0) * sum(
        max(
            0.0,
            ((current - previous).total_seconds() / 3600.0 - reference)
            / reference,
        )
        for previous, current in zip(planned_times, planned_times[1:])
    )


def _cadence_underfill_pressure(
    candidate_miles: float,
    target_distance_range: tuple[float, float],
    horizon_days: int,
    ordinary_easy_miles: float,
) -> float:
    """Return cadence urgency only for genuinely unfunded training load."""

    horizon_target_low = (
        target_distance_range[0]
        * max(0, horizon_days)
        / VISIBLE_HORIZON_DAYS
    )
    return min(
        1.0,
        max(0.0, horizon_target_low - max(0.0, candidate_miles))
        / max(0.1, ordinary_easy_miles),
    )


def _key_session_cadence_overdue_cost(
    age_days: float,
    reference_days: float,
) -> float:
    """Apply continuous pressure as a key-session lane moves past target."""

    overdue_days = max(0.0, age_days - reference_days)
    if overdue_days <= 0:
        return 0.0
    return PROGRAM_FIT_UNIT * overdue_days * overdue_days


def _target_derived_bridge_reference(
    state: FitnessState,
    target_distance_range: tuple[float, float],
    easy_reference_miles: float,
) -> tuple[float, float]:
    """Return one candidate-independent density reference for this replan.

    The target supplies the weekly amount of work.  Demonstrated session size
    supplies a plausible size for each exposure, without turning the resulting
    cadence into a run-count requirement.  The protected ordinary-easy
    baseline is a lower bound so incidental short sessions cannot teach the
    planner that increasingly tiny runs are normal.

    Both values are fixed from the opening state.  A candidate therefore
    cannot make a dense calendar look safer merely by adding run days and then
    using its own frequency as the recovery zero point.
    """

    target_miles_per_week = sum(target_distance_range) / 2.0
    recent = state.recent_load.trailing_28d
    demonstrated_session_miles = (
        recent.distance_miles / recent.activity_count
        if recent.activity_count > 0 and recent.distance_miles > 0
        else 0.0
    )
    reference_session_miles = max(
        0.1,
        easy_reference_miles,
        demonstrated_session_miles,
    )
    reference_sessions_per_week = (
        target_miles_per_week / reference_session_miles
        if target_miles_per_week > 0
        else 0.0
    )
    reference_gap_hours = (
        VISIBLE_HORIZON_DAYS * 24.0 / reference_sessions_per_week
        if reference_sessions_per_week > 0
        else 0.0
    )
    reference_load = reference_session_miles / max(
        0.1, easy_reference_miles
    )
    return reference_gap_hours, reference_load


def _three_session_compression_cost(
    prior_times: list[datetime],
    prior_loads: list[float],
    proposed_at: datetime,
    proposed_load: float,
    preferred_gap_hours: float,
) -> float:
    """Price a third compressed session without prohibiting useful doubles.

    A single short gap can be an intentional back-to-back. When two successive
    gaps are both shorter than the athlete's configured cadence preference,
    the third run adds a smooth athlete-relative cost. The lightest of the
    three sessions sets the scale so short recovery work remains cheaper than
    stacking three ordinary or taxing sessions.
    """

    if len(prior_times) < 2 or len(prior_loads) < 2:
        return 0.0
    preferred = max(1.0, preferred_gap_hours)
    prior_gap = max(
        0.0,
        (prior_times[-1] - prior_times[-2]).total_seconds() / 3600.0,
    )
    proposed_gap = max(
        0.0,
        (proposed_at - prior_times[-1]).total_seconds() / 3600.0,
    )
    prior_pressure = max(0.0, 1.0 - prior_gap / preferred)
    proposed_pressure = max(0.0, 1.0 - proposed_gap / preferred)
    load_scale = max(
        0.0,
        min(prior_loads[-2], prior_loads[-1], proposed_load),
    )
    return (
        prior_pressure
        * proposed_pressure
        * load_scale
        * PROGRAM_FIT_UNIT
        * 2.5
    )


def _opening_session_compression_cost(
    elapsed_hours: float,
    completed_load: float,
    proposed_load: float,
    preferred_gap_hours: float,
) -> float:
    """Softly price an early first session across the history/plan seam."""

    preferred = max(1.0, preferred_gap_hours)
    pressure = max(0.0, 1.0 - max(0.0, elapsed_hours) / preferred)
    load_scale = max(0.0, (completed_load + proposed_load) / 2.0)
    return pressure * load_scale * PROGRAM_FIT_UNIT * 0.5


def _soft_continuation_residual_cost(
    opening_residual: float,
    opening_at: datetime,
    planned_times: list[datetime],
    planned_loads: list[float],
    horizon_end: datetime,
    *,
    reference_gap_hours: float,
    reference_load: float,
    half_life_hours: float,
    continuation_days: float = SOFT_CONTINUATION_DAYS,
) -> float:
    """Price load carried beyond the hard horizon with a cheap steady tail.

    The continuation is not another optimized plan. It advances the terminal
    residual through the athlete's target-derived ordinary cadence and charges
    only excess above that cadence's steady state. The most recent planned
    session is allowed at its actual load, so a legitimate long or quality run
    near the boundary is not mistaken for clustered backloading.
    """

    if (
        reference_gap_hours <= 0
        or reference_load <= 0
        or half_life_hours <= 0
        or continuation_days <= 0
        or len(planned_times) != len(planned_loads)
    ):
        return 0.0
    residual = decay_recovery_load(
        max(0.0, opening_residual),
        max(0.0, (horizon_end - opening_at).total_seconds() / 3600.0),
        half_life_hours=half_life_hours,
    )
    for planned_at, load_units in zip(planned_times, planned_loads):
        if planned_at >= horizon_end:
            continue
        residual += decay_recovery_load(
            max(0.0, load_units),
            max(0.0, (horizon_end - planned_at).total_seconds() / 3600.0),
            half_life_hours=half_life_hours,
        )

    gap_decay = decay_recovery_load(
        1.0,
        reference_gap_hours,
        half_life_hours=half_life_hours,
    )
    steady_pre_session = reference_load * gap_decay / max(
        1e-9, 1.0 - gap_decay
    )
    latest_pair = max(
        (
            (planned_at, load_units)
            for planned_at, load_units in zip(planned_times, planned_loads)
            if planned_at < horizon_end
        ),
        key=lambda item: item[0],
        default=None,
    )
    latest_planned = latest_pair[0] if latest_pair is not None else None
    elapsed_since_latest = (
        max(
            0.0,
            (horizon_end - latest_planned).total_seconds() / 3600.0,
        )
        if latest_planned is not None
        else reference_gap_hours
    )
    phase_hours = min(reference_gap_hours, elapsed_since_latest)
    steady_at_phase = reference_load * decay_recovery_load(
        1.0,
        phase_hours,
        half_life_hours=half_life_hours,
    ) / max(1e-9, 1.0 - gap_decay)
    latest_load_allowance = (
        decay_recovery_load(
            max(0.0, latest_pair[1] - reference_load),
            elapsed_since_latest,
            half_life_hours=half_life_hours,
        )
        if latest_pair is not None
        else 0.0
    )
    # Compare with the steady target cadence at the same phase. The latest
    # prescribed dose is accepted as-is; excess residue from earlier clustered
    # work is still visible and carries into every continuation probe.
    terminal_excess = max(
        0.0,
        residual - steady_at_phase - latest_load_allowance,
    )
    cost = terminal_excess * terminal_excess

    probe_count = max(
        1,
        ceil(continuation_days * 24.0 / reference_gap_hours),
    )
    first_gap_hours = max(0.0, reference_gap_hours - phase_hours)
    for probe in range(1, probe_count + 1):
        elapsed_hours = (
            first_gap_hours if probe == 1 else reference_gap_hours
        )
        residual *= decay_recovery_load(
            1.0,
            elapsed_hours,
            half_life_hours=half_life_hours,
        )
        latest_load_allowance *= decay_recovery_load(
            1.0,
            elapsed_hours,
            half_life_hours=half_life_hours,
        )
        excess = max(
            0.0,
            residual - steady_pre_session - latest_load_allowance,
        )
        discount = 0.5 ** (
            (first_gap_hours + (probe - 1) * reference_gap_hours)
            / 24.0
            / max(0.1, continuation_days)
        )
        cost += discount * excess * excess
        residual += reference_load
    return cost


def _finalized_program_recovery_cost(
    days: list[WeeklyScheduleDay],
    daily_states: list[FitnessState],
    config: dict,
    target_distance_range: tuple[float, float] | None = None,
    *,
    session_states: list[FitnessState] | None = None,
    diagnostic_components: dict[str, float] | None = None,
) -> float:
    """Score recovery and committed density at finalized workout distances.

    Transient recovery answers whether the next run is tolerable.  The two
    continuous mileage rates answer a different question: whether repeatedly
    spending that tolerable session creates more accumulated work than the
    current target supports.  Both are evaluated after each proposed run, so a
    fresh daily replan cannot reuse headroom that its prior recommendation had
    already consumed.
    """

    if not daily_states:
        return 0.0
    session_states = session_states or daily_states
    if len(session_states) != len(daily_states):
        raise ValueError("Session states must align with daily states")
    easy_reference_miles = projected_recovery_reference_miles(daily_states[0])
    planned: list[RecommendationResponse] = []
    planned_loads: list[float] = []
    cost = 0.0
    components = {
        "immediate": 0.0,
        "opening_compression": 0.0,
        "three_session_compression": 0.0,
        "short_term_bridge": 0.0,
        "continuous_slow": 0.0,
        "continuous_short": 0.0,
        "soft_continuation": 0.0,
    }

    def finish() -> float:
        if diagnostic_components is not None:
            diagnostic_components.update(components)
        return cost
    slow_half_life = max(
        0.1,
        float(
            config.get("coaching", {}).get(
                "continuous_fatigue_half_life_days", 7
            )
        ),
    )
    short_half_life = short_term_density_half_life_days(
        slow_half_life,
        recovery_half_life_hours=RECOVERY_HALF_LIFE_HOURS,
    )
    scheduled = [
        (index, day.recommendation)
        for index, day in enumerate(days)
        if day.recommendation is not None
        and day.recommendation.workout_type != WorkoutType.REST
        and day.recommendation.planned_for is not None
    ]
    scheduled_loads = [
        _recommendation_load_units(result, easy_reference_miles)
        for _, result in scheduled
    ]
    # Every candidate must be compared with the same recovery-density zero
    # point.  When a funded mileage target is available, derive that reference
    # from target load and demonstrated session size.  The fallback preserves
    # direct recovery-only callers that have no program target to compare.
    if target_distance_range is not None:
        reference_gap_hours, reference_load = (
            _target_derived_bridge_reference(
                daily_states[0],
                target_distance_range,
                easy_reference_miles,
            )
        )
    else:
        reference_gap_hours = (
            len(days) * 24.0 / len(scheduled) if scheduled else 0.0
        )
        reference_load = (
            sum(scheduled_loads) / len(scheduled_loads)
            if scheduled_loads
            else 0.0
        )
    reference_short_decay = (
        decay_recovery_load(
            1.0,
            reference_gap_hours,
            half_life_hours=short_half_life * 24.0,
        )
        if reference_gap_hours > 0
        else 0.0
    )
    reference_immediate_decay = (
        decay_recovery_load(
            1.0,
            reference_gap_hours,
            half_life_hours=RECOVERY_HALF_LIFE_HOURS,
        )
        if reference_gap_hours > 0
        else 0.0
    )
    steady_reference_bridge = reference_load * max(
        0.0,
        reference_short_decay / max(1e-9, 1.0 - reference_short_decay)
        - reference_immediate_decay
        / max(1e-9, 1.0 - reference_immediate_decay),
    )
    preferred_gap_hours = max(
        1.0,
        (
            int(
                config.get("coaching", {}).get(
                    "typical_rest_days_between_runs", 1
                )
            )
            + 1
        )
        * 24.0,
    )
    opening_state = daily_states[0]
    compression_times: list[datetime] = []
    compression_loads: list[float] = []
    opening_completed_load: float | None = None
    if (
        opening_state.last_run is not None
        and opening_state.days_since_last_run is not None
    ):
        opening_completed_load, _ = athlete_relative_session_load(
            opening_state.last_run,
            prior_typical_load(opening_state),
            performance_response=opening_state.recent_performance_response,
            drift_percent=opening_state.last_run_drift_percent,
            prescribed_intensity_factor=(
                opening_state.last_run_prescribed_intensity_factor
            ),
            prescribed_distance_range_miles=(
                opening_state.last_run_prescribed_distance_range_miles
            ),
        )
        compression_times.append(
            opening_state.as_of
            - timedelta(days=opening_state.days_since_last_run)
        )
        compression_loads.append(opening_completed_load)
    for position, ((index, result), proposed_load) in enumerate(
        zip(scheduled, scheduled_loads)
    ):
        residual_load = _decayed_recovery_load(
            session_states[index],
            planned,
            result.planned_for,
            easy_reference_miles,
        )
        immediate_recovery_cost = _recovery_interaction_cost(
            residual_load, proposed_load
        )
        cost += immediate_recovery_cost
        components["immediate"] += immediate_recovery_cost
        if (
            position == 0
            and opening_completed_load is not None
            and opening_state.days_since_last_run is not None
        ):
            elapsed_hours = (
                opening_state.days_since_last_run * 24.0
                + max(
                    0.0,
                    (
                        result.planned_for - opening_state.as_of
                    ).total_seconds()
                    / 3600.0,
                )
            )
            opening_compression_cost = _opening_session_compression_cost(
                elapsed_hours,
                opening_completed_load,
                proposed_load,
                preferred_gap_hours,
            )
            cost += opening_compression_cost
            components["opening_compression"] += opening_compression_cost
        three_session_cost = _three_session_compression_cost(
            compression_times,
            compression_loads,
            result.planned_for,
            proposed_load,
            preferred_gap_hours,
        )
        cost += three_session_cost
        components["three_session_compression"] += three_session_cost
        # Immediate recovery answers whether the next individual run fits.
        # A slower bridge signal separately represents mechanical density that
        # can accumulate across several otherwise-tolerable sessions. Squaring
        # only that accumulated residue leaves an ordinary two-day pair legal,
        # while a third or fourth close session becomes progressively more
        # expensive without a calendar-based streak rule.
        bridge_only_residual = sum(
            load_units
            * max(
                0.0,
                decay_recovery_load(
                    1.0,
                    (result.planned_for - prior.planned_for).total_seconds()
                    / 3600.0,
                    half_life_hours=short_half_life * 24.0,
                )
                - decay_recovery_load(
                    1.0,
                    (result.planned_for - prior.planned_for).total_seconds()
                    / 3600.0,
                    half_life_hours=RECOVERY_HALF_LIFE_HOURS,
                ),
            )
            for prior, load_units in zip(planned, planned_loads)
            if prior.planned_for is not None
            and prior.planned_for < result.planned_for
        )
        short_distance_rate = session_states[
            index
        ].recent_load.continuous_short_term_distance_miles
        if short_distance_rate is not None:
            short_rate_normalization = (
                log(2.0) * 7.0 / short_half_life
            )
            completed_short_residual = (
                short_distance_rate
                / max(1e-9, short_rate_normalization)
                / max(0.1, easy_reference_miles)
            )
            completed_immediate_residual = _decayed_recovery_load(
                session_states[index],
                [],
                result.planned_for,
                easy_reference_miles,
            )
            completed_bridge_residual = max(
                0.0,
                completed_short_residual - completed_immediate_residual,
            )
            reference_bridge_residual = steady_reference_bridge
        else:
            # Sparse/test states without a persisted short-term signal start
            # from an empty history, so their equally-spaced reference grows
            # only as many prior opportunities as actually exist.
            completed_bridge_residual = 0.0
            reference_bridge_residual = reference_load * sum(
                max(
                    0.0,
                    decay_recovery_load(
                        1.0,
                        lag * reference_gap_hours,
                        half_life_hours=short_half_life * 24.0,
                    )
                    - decay_recovery_load(
                        1.0,
                        lag * reference_gap_hours,
                        half_life_hours=RECOVERY_HALF_LIFE_HOURS,
                    ),
                )
                for lag in range(1, position + 1)
            )
        bridge_excess = max(
            0.0,
            completed_bridge_residual
            + bridge_only_residual
            - reference_bridge_residual,
        )
        bridge_cost = (
            bridge_excess**2
            * max(0.0, proposed_load)
            * PROGRAM_FIT_UNIT
        )
        cost += bridge_cost
        components["short_term_bridge"] += bridge_cost
        planned.append(result)
        planned_loads.append(proposed_load)
        compression_times.append(result.planned_for)
        compression_loads.append(proposed_load)

    if target_distance_range is None:
        return finish()

    slow_rate = opening_state.recent_load.continuous_distance_miles
    short_rate = (
        opening_state.recent_load.continuous_short_term_distance_miles
    )
    if slow_rate is None or short_rate is None:
        return finish()

    target_high = target_distance_range[1]
    opening_slow_rate = slow_rate
    opening_short_rate = short_rate
    previous_at = opening_state.as_of

    def incremental_excess_cost(
        before: float,
        after: float,
        upper_path: float,
        *,
        impulse_tolerance: float,
        timescale_weight: float,
    ) -> float:
        reference = max(0.1, easy_reference_miles)
        tolerated_upper = upper_path + max(0.0, impulse_tolerance)
        before_units = max(0.0, before - tolerated_upper) / reference
        after_units = max(0.0, after - tolerated_upper) / reference
        return (
            max(0.0, after_units * after_units - before_units * before_units)
            * PROGRAM_FIT_UNIT
            * timescale_weight
        )

    for result in sorted(
        planned,
        key=lambda item: item.planned_for or opening_state.as_of,
    ):
        planned_for = result.planned_for or previous_at
        elapsed_days = max(
            0.0, (planned_for - previous_at).total_seconds() / 86400.0
        )
        slow_rate *= 0.5 ** (elapsed_days / slow_half_life)
        short_rate *= 0.5 ** (elapsed_days / short_half_life)
        total_elapsed_days = max(
            0.0,
            (planned_for - opening_state.as_of).total_seconds() / 86400.0,
        )
        slow_upper_path = target_high + max(
            0.0, opening_slow_rate - target_high
        ) * 0.5 ** (total_elapsed_days / slow_half_life)
        short_upper_path = target_high + max(
            0.0, opening_short_rate - target_high
        ) * 0.5 ** (total_elapsed_days / short_half_life)
        miles = _midpoint(result)
        # A planned long run is intentionally larger than an ordinary easy
        # session. Its actual load is already priced by recovery interaction,
        # bridge residue, progression limits, and the next session's spacing.
        # Charging the portion above easy-reference mileage again as an
        # instantaneous density excess made skipping every third long run
        # cheaper than maintaining the intended 7–8 day progression cadence.
        tolerated_pulse_miles = (
            miles
            if result.workout_type == WorkoutType.LONG
            else min(miles, easy_reference_miles)
        )
        slow_impulse_tolerance = (
            tolerated_pulse_miles * log(2.0) * 7.0 / slow_half_life
        )
        short_impulse_tolerance = (
            tolerated_pulse_miles * log(2.0) * 7.0 / short_half_life
        )
        next_slow_rate = slow_rate + (
            miles * log(2.0) * 7.0 / slow_half_life
        )
        next_short_rate = short_rate + (
            miles * log(2.0) * 7.0 / short_half_life
        )
        # A discretely scheduled run necessarily creates an instantaneous
        # pulse above a smooth mileage-rate target. Permit one athlete-typical
        # session—or the full intentionally longer dose for a long run—at each
        # timescale; accumulated residue still stacks above this corridor.
        slow_cost = incremental_excess_cost(
            slow_rate,
            next_slow_rate,
            slow_upper_path,
            impulse_tolerance=slow_impulse_tolerance,
            timescale_weight=0.5
            ** (total_elapsed_days / short_half_life),
        )
        cost += slow_cost
        components["continuous_slow"] += slow_cost
        # This target-aware component is deliberately smaller than the direct
        # pairwise bridge interaction above. Its weight is derived from both
        # existing timescale ratios rather than independently tuned.
        short_cost = incremental_excess_cost(
            short_rate,
            next_short_rate,
            short_upper_path,
            impulse_tolerance=short_impulse_tolerance,
            # Its shorter half-life already limits how long concentration
            # matters. Discounting the magnitude as well double-attenuates
            # the signal and lets several individually tolerable runs form an
            # excessive block. Short-term density and accumulated load answer
            # different questions, so each uses the shared unit once.
            timescale_weight=1.0,
        )
        cost += short_cost
        components["continuous_short"] += short_cost
        slow_rate = next_slow_rate
        short_rate = next_short_rate
        previous_at = planned_for

    # A hard 21-day edge must not make late mileage look free merely because
    # its recovery consequences occur just outside the optimized calendar.
    # Continue both immediate and short-term residuals through a target-paced
    # virtual tail. This is O(number of ordinary tail sessions), not another
    # combinatorial schedule search.
    horizon_end = daily_states[0].as_of + timedelta(days=len(days))
    planned_times = [
        result.planned_for
        for result in planned
        if result.planned_for is not None
    ]
    aligned_loads = [
        load_units
        for result, load_units in zip(planned, planned_loads)
        if result.planned_for is not None
    ]
    immediate_opening_residual = _decayed_recovery_load(
        opening_state,
        [],
        opening_state.as_of,
        easy_reference_miles,
    )
    continuation_cost = _soft_continuation_residual_cost(
        immediate_opening_residual,
        opening_state.as_of,
        planned_times,
        aligned_loads,
        horizon_end,
        reference_gap_hours=reference_gap_hours,
        reference_load=reference_load,
        half_life_hours=RECOVERY_HALF_LIFE_HOURS,
    )
    if opening_short_rate is not None:
        short_normalization = log(2.0) * 7.0 / short_half_life
        opening_short_residual = (
            opening_short_rate
            / max(1e-9, short_normalization)
            / max(0.1, easy_reference_miles)
        )
        continuation_cost += _soft_continuation_residual_cost(
            opening_short_residual,
            opening_state.as_of,
            planned_times,
            aligned_loads,
            horizon_end,
            reference_gap_hours=reference_gap_hours,
            reference_load=reference_load,
            half_life_hours=short_half_life * 24.0,
        )
    weighted_continuation_cost = continuation_cost * PROGRAM_FIT_UNIT
    cost += weighted_continuation_cost
    components["soft_continuation"] += weighted_continuation_cost
    return finish()


def _project_window(window: LoadWindow, additions: list[RecommendationResponse], as_of: datetime) -> LoadWindow:
    recent = [
        result for result in additions
        if result.planned_for and 0 <= (as_of - result.planned_for).total_seconds() <= window.days * 86400
    ]
    miles = sum(_midpoint(result) for result in recent)
    recent_pace = (
        window.moving_minutes / window.distance_miles
        if window.distance_miles > 0 and window.moving_minutes > 0
        else 11.0
    )
    moving = miles * recent_pace
    added_hard_minutes = sum(
        _prescribed_zone_minutes(
            result,
            _midpoint(result) * recent_pace,
        )[1]
        for result in recent
    )
    base_load_per_mile = (window.zone_load or 0) / window.distance_miles if window.distance_miles else 18.0
    added_load = sum(
        _midpoint(result)
        * base_load_per_mile
        * _prescribed_intensity_factor(
            result,
            max(
                _structured_duration_minutes(result) or 0.0,
                _midpoint(result) * recent_pace,
            ),
        )
        for result in recent
    )
    return window.model_copy(
        update={
            "distance_miles": window.distance_miles + miles,
            "moving_minutes": window.moving_minutes + moving,
            "zone_load": (window.zone_load + added_load) if window.zone_load is not None else None,
            "hard_minutes": window.hard_minutes + added_hard_minutes,
            "activity_count": window.activity_count + len(recent),
            "zone_load_activity_count": (
                window.zone_load_activity_count + len(recent)
            ),
        }
    )


def _project_state(
    state: FitnessState,
    planned: list[RecommendationResponse],
    config: dict | None = None,
    *,
    recommendation_load_cache: _RecommendationLoadCache | None = None,
) -> FitnessState:
    if not planned:
        return state
    prior_runs = [item for item in planned if item.planned_for and item.planned_for < state.as_of and item.workout_type != WorkoutType.REST]
    if not prior_runs:
        return state
    last = prior_runs[-1]
    # Session-size evidence is observed history, not a quantity that a
    # hypothetical plan is allowed to rewrite. When an older state does not
    # yet carry the dedicated recency-weighted easy baseline, freeze its
    # current observed per-run average before adding projected sessions.
    # Otherwise several short support runs lower the denominator inside the
    # same candidate and make still shorter filler runs appear normal.
    stable_typical_easy_miles = state.typical_easy_run_miles
    if stable_typical_easy_miles is None:
        opening_window = state.recent_load.trailing_28d
        if opening_window.activity_count and opening_window.distance_miles > 0:
            stable_typical_easy_miles = (
                opening_window.distance_miles / opening_window.activity_count
            )
    quality = [
        item for item in prior_runs if item.workout_type in QUALITY_WORKOUT_TYPES
    ]
    prior_runs_14d = [
        item
        for item in prior_runs
        if 0.0
        <= (state.as_of - item.planned_for).total_seconds()
        <= 14 * 86400
    ]
    quality_14d = [
        item
        for item in prior_runs_14d
        if item.workout_type in QUALITY_WORKOUT_TYPES
    ]
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
    recent_pace = (
        state.recent_load.trailing_28d.moving_minutes
        / state.recent_load.trailing_28d.distance_miles
        if state.recent_load.trailing_28d.distance_miles > 0
        and state.recent_load.trailing_28d.moving_minutes > 0
        else 11.0
    )
    moving = distance * recent_pace
    known_minutes_14d = state.recent_load.trailing_14d.moving_minutes
    projected_minutes_14d = sum(
        _midpoint(item) * recent_pace for item in prior_runs_14d
    )
    projected_zone_minutes_14d = [
        _prescribed_zone_minutes(
            item,
            _midpoint(item) * recent_pace,
        )
        for item in prior_runs_14d
    ]
    projected_moderate_minutes = sum(
        moderate for moderate, _ in projected_zone_minutes_14d
    )
    projected_hard_minutes = sum(
        hard for _, hard in projected_zone_minutes_14d
    )
    combined_known_minutes = known_minutes_14d + projected_minutes_14d
    observed_moderate_minutes = (
        (state.moderate_fraction_14d or 0.0) * known_minutes_14d
    )
    observed_hard_minutes = (
        (state.hard_fraction_14d or 0.0) * known_minutes_14d
    )
    projected_moderate_fraction = (
        (observed_moderate_minutes + projected_moderate_minutes)
        / combined_known_minutes
        if combined_known_minutes > 0
        else state.moderate_fraction_14d
    )
    projected_hard_fraction = (
        (observed_hard_minutes + projected_hard_minutes)
        / combined_known_minutes
        if combined_known_minutes > 0
        else state.hard_fraction_14d
    )
    projected_easy_fraction = (
        max(
            0.0,
            1.0
            - (projected_moderate_fraction or 0.0)
            - (projected_hard_fraction or 0.0),
        )
        if combined_known_minutes > 0
        else state.easy_fraction_14d
    )
    projected_recovery_residual = _decayed_recovery_load(
        state,
        prior_runs,
        state.as_of,
        projected_recovery_reference_miles(state),
        recommendation_load_cache=recommendation_load_cache,
    )
    fatigue_half_life_days = float(
        (config or {}).get("coaching", {}).get(
            "continuous_fatigue_half_life_days", 7
        )
    )
    projected_fatigue_miles = state.recent_load.continuous_fatigue_miles
    if projected_fatigue_miles is not None:
        easy_reference = projected_recovery_reference_miles(state)
        projected_fatigue_miles += sum(
            log(2.0)
            * _cached_recommendation_load_units(
                item,
                easy_reference,
                recommendation_load_cache,
            )
            * easy_reference
            * 0.5
            ** (
                max(
                    0.0,
                    (state.as_of - item.planned_for).total_seconds()
                    / 86400.0,
                )
                / max(0.1, fatigue_half_life_days)
            )
            for item in prior_runs
            if item.planned_for
        )
    projected_fatigue_ratio = (
        projected_fatigue_miles / capacity_reference
        if projected_fatigue_miles is not None and capacity_reference
        else None
    )
    projected_distance_miles = state.recent_load.continuous_distance_miles
    if projected_distance_miles is not None:
        distance_normalization = (
            log(2.0) * 7.0 / max(0.1, fatigue_half_life_days)
        )
        projected_distance_miles += sum(
            distance_normalization
            * _midpoint(item)
            * 0.5
            ** (
                max(
                    0.0,
                    (state.as_of - item.planned_for).total_seconds()
                    / 86400.0,
                )
                / max(0.1, fatigue_half_life_days)
            )
            for item in prior_runs
            if item.planned_for
        )
    short_term_half_life_days = short_term_density_half_life_days(
        fatigue_half_life_days
    )
    projected_short_term_distance_miles = (
        state.recent_load.continuous_short_term_distance_miles
    )
    if projected_short_term_distance_miles is not None:
        short_term_normalization = (
            log(2.0) * 7.0 / max(0.1, short_term_half_life_days)
        )
        projected_short_term_distance_miles += sum(
            short_term_normalization
            * _midpoint(item)
            * 0.5
            ** (
                max(
                    0.0,
                    (state.as_of - item.planned_for).total_seconds()
                    / 86400.0,
                )
                / max(0.1, short_term_half_life_days)
            )
            for item in prior_runs
            if item.planned_for
        )
    last_quality = last.workout_type in QUALITY_WORKOUT_TYPES
    last_long = last.workout_type == WorkoutType.LONG
    moderate_minutes, hard_minutes = _prescribed_zone_minutes(last, moving)
    easy_minutes = max(0.0, moving - moderate_minutes - hard_minutes)
    known_minutes = max(1.0, easy_minutes + moderate_minutes + hard_minutes)
    difficulty = SessionDifficulty(
        distance_miles=distance,
        moving_minutes=moving,
        elapsed_minutes=moving,
        stopped_minutes=0,
        zone_load=None,
        zone_breakdown=ZoneBreakdown(
            zone_seconds={
                "z2": easy_minutes * 60,
                "z3": moderate_minutes * 60,
                "z4": hard_minutes * 60,
            },
            zone_fractions={
                "z2": easy_minutes / known_minutes,
                "z3": moderate_minutes / known_minutes,
                "z4": hard_minutes / known_minutes,
            },
            easy_minutes=easy_minutes,
            moderate_minutes=moderate_minutes,
            hard_minutes=hard_minutes,
        ),
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
                continuous_fatigue_miles=projected_fatigue_miles,
                continuous_fatigue_to_capacity_ratio=projected_fatigue_ratio,
                continuous_distance_miles=projected_distance_miles,
                continuous_short_term_distance_miles=(
                    projected_short_term_distance_miles
                ),
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
            "last_run_distance_miles": distance,
            "last_run_activity_id": None,
            "last_run_workout_type": last.workout_type,
            "last_run_prescribed_workout_type": last.workout_type,
            "last_run_prescribed_distance_range_miles": (
                last.distance_range_miles
            ),
            "last_run_completed_prescribed_workout": True,
            "typical_easy_run_miles": stable_typical_easy_miles,
            "recovery_residual_load": projected_recovery_residual,
            # These measurements belong to the latest completed run.  Once a
            # planned session intervenes, its response is unknown; repeating
            # the old completed-run caution across every later workout makes
            # the whole prospective week identical.  Later sessions are
            # therefore conditional on the intervening workout going as
            # prescribed, rather than assumed to inherit stale evidence.
            "last_run_drift_percent": None,
            "last_run_prescribed_intensity_factor": None,
            "recent_performance_response": "unknown",
            "recent_performance_response_activity_id": None,
            "recent_performance_response_at": None,
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
            "quality_sessions_14d": (
                state.quality_sessions_14d + len(quality_14d)
            ),
            "completed_quality_session_count": state.completed_quality_session_count + len(quality),
            "last_completed_quality_session_type": (
                quality[-1].quality_session_type
                if quality and quality[-1].quality_session_type is not None
                else state.last_completed_quality_session_type
            ),
            "running_days_28d": state.running_days_28d + len(prior_runs),
            "moderate_fraction_14d": projected_moderate_fraction,
            "easy_fraction_14d": projected_easy_fraction,
            "hard_fraction_14d": projected_hard_fraction,
            "moderate_evidence_runs_14d": (
                state.moderate_evidence_runs_14d
                + sum(
                    moderate > 0
                    for moderate, _ in projected_zone_minutes_14d
                )
            ),
            "normal_runs_since_health_event": state.normal_runs_since_health_event + len(prior_runs),
        }
    )


def _cached_project_state(
    state: FitnessState,
    planned: list[RecommendationResponse],
    config: dict | None,
    projected_state_cache: _ProjectedStateCache | None,
    recommendation_load_cache: _RecommendationLoadCache | None,
) -> FitnessState:
    """Project an exact state/prefix pair once per planner invocation."""

    if projected_state_cache is None:
        return _project_state(
            state,
            planned,
            config,
            recommendation_load_cache=recommendation_load_cache,
        )
    planned_ids = tuple(id(item) for item in planned)
    key = (id(state), planned_ids)
    cached = projected_state_cache.get(key)
    if (
        cached is not None
        and cached[0] is state
        and len(cached[1]) == len(planned)
        and all(left is right for left, right in zip(cached[1], planned))
    ):
        return cached[2]
    projected = _project_state(
        state,
        planned,
        config,
        recommendation_load_cache=recommendation_load_cache,
    )
    projected_state_cache[key] = (state, tuple(planned), projected)
    return projected


def _elapsed_workout_role(
    state_options: list[FitnessState],
    planned: list[RecommendationResponse],
    config: dict,
    *,
    next_state_options: list[FitnessState] | None = None,
    recommendation_load_cache: _RecommendationLoadCache | None = None,
    projected_state_cache: _ProjectedStateCache | None = None,
) -> str:
    """Choose the key-session lane nearest its rolling target timestamp.

    Compare the best remaining time on this run date with the next proposed
    run opportunity and use whichever is closer to the continuous cadence
    target. Exact recovery checks still decide whether the taxing role is safe.
    """

    if not state_options:
        return "easy"
    projected_options = [
        _cached_project_state(
            state,
            planned,
            config,
            projected_state_cache,
            recommendation_load_cache,
        )
        for state in state_options
    ]
    projected_next_options = [
        _cached_project_state(
            state,
            planned,
            config,
            projected_state_cache,
            recommendation_load_cache,
        )
        for state in (next_state_options or [])
    ]
    settings = config.get("coaching", {})
    tapering = _is_taper_date(config, projected_options[0].as_of.date())
    if tapering:
        return "easy"
    cadence_specs = (
        (
            "long",
            "days_since_long_run",
            float(settings.get("long_run_recency_reference_days", 7)),
        ),
        (
            "quality",
            "days_since_quality_run",
            float(settings.get("quality_recency_reference_days", 7)),
        ),
    )
    candidates: list[tuple[float, float, int, str]] = []
    for role, attribute, raw_reference in cadence_specs:
        reference = max(1.0, raw_reference)

        def cadence_error(age: float) -> float:
            return (
                (reference - age) * 2.0
                if age < reference
                else age - reference
            )

        current_ages = [getattr(state, attribute) for state in projected_options]
        known_current_ages = [
            float(age) for age in current_ages if age is not None
        ]
        current_error = (
            min(cadence_error(age) for age in known_current_ages)
            if known_current_ages
            else 0.0
        )
        current_urgency = (
            max(known_current_ages) / reference
            if known_current_ages
            else float("inf")
        )
        known_next_ages = [
            float(current_age)
            + max(
                0.0,
                (next_state.as_of - current_state.as_of).total_seconds()
                / 86400.0,
            )
            for current_state, current_age in zip(
                projected_options, current_ages
            )
            if current_age is not None
            for next_state in projected_next_options
        ]
        if projected_next_options:
            next_error = (
                min(cadence_error(age) for age in known_next_ages)
                if known_next_ages
                else 0.0
            )
            if known_current_ages and current_error >= next_error - 1e-9:
                continue
            opportunity_advantage = next_error - current_error
        else:
            if known_current_ages and max(known_current_ages) < reference:
                continue
            opportunity_advantage = max(0.0, current_urgency - 1.0)
        candidates.append(
            (
                opportunity_advantage,
                current_urgency,
                1 if role == "long" else 0,
                role,
            )
        )
    if not candidates:
        return "easy"
    return max(candidates)[-1]


def _select_timed_recommendation(
    state_options: list[FitnessState],
    planned: list[RecommendationResponse],
    request: RecommendationRequest,
    config: dict,
    *,
    weekly_role: str | None,
    allowed_candidates: set[str] | None = None,
    recommendation_load_cache: _RecommendationLoadCache | None = None,
    projected_state_cache: _ProjectedStateCache | None = None,
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
        state = _cached_project_state(
            raw_state,
            planned,
            config,
            projected_state_cache,
            recommendation_load_cache,
        )
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
    preserve_selected_role: bool = False,
    recommendation_load_cache: _RecommendationLoadCache | None = None,
    projected_state_cache: _ProjectedStateCache | None = None,
) -> tuple[FitnessState, RecommendationResponse]:
    """Choose the strongest role that fits exact timing and recovery."""
    # Elapsed recency selects the purpose of this candidate slot. Once the
    # coordinator has done that, re-running the generic single-workout score
    # across all three roles lets projected load repeatedly turn due quality
    # work into easy mileage. Start with the selected purpose and fall back to
    # easy only when the exact cumulative recovery check below shows that the
    # taxing role does not fit. The quality allocator can then shorten the
    # dose instead of silently deleting it.
    observed_load_ratio = (
        effective_load_ratio(state_options[0].recent_load)
        if state_options
        and "includes_planned_sessions"
        not in state_options[0].recent_load.flags
        else None
    )
    allowed = (
        {weekly_role}
        if preserve_selected_role
        and weekly_role in {"easy", "long", "quality"}
        else
        {weekly_role, "easy"}
        if weekly_role in {"long", "quality"}
        and observed_load_ratio is not None
        and observed_load_ratio > 1.0
        else {weekly_role}
        if weekly_role in {"easy", "long", "quality"}
        else {"easy", "long", "quality"}
    )
    taxing_spacing_substitution = False
    while True:
        state, result = _select_timed_recommendation(
            state_options,
            planned,
            request,
            config,
            weekly_role=weekly_role,
            allowed_candidates=allowed,
            recommendation_load_cache=recommendation_load_cache,
            projected_state_cache=projected_state_cache,
        )
        candidate = (
            "long"
            if result.workout_type == WorkoutType.LONG
            else "quality"
            if result.workout_type
            in QUALITY_WORKOUT_TYPES
            else "easy"
        )
        latest_taxing = next(
            (
                item
                for item in reversed(planned)
                if item.planned_for
                and item.workout_type
                in {WorkoutType.LONG, *QUALITY_WORKOUT_TYPES}
            ),
            None,
        )
        taxing_residual = 0.0
        if result.planned_for and latest_taxing and latest_taxing.planned_for:
            easy_reference = projected_recovery_reference_miles(state)
            latest_taxing_load = _recommendation_load_units(
                latest_taxing, easy_reference
            )
            elapsed_taxing_hours = (
                result.planned_for - latest_taxing.planned_for
            ).total_seconds() / 3600
            taxing_residual = decay_recovery_load(
                latest_taxing_load,
                elapsed_taxing_hours,
            )
            if latest_taxing.workout_type == WorkoutType.RACE:
                # A race combines sustained intensity with a potentially large
                # distance. Preserve easy-running availability on the immediate
                # recovery curve, but use the existing athlete-relative bridge
                # timescale before another long or quality session. This avoids
                # assigning the same fixed recovery period to a 5K and marathon.
                race_half_life_hours = (
                    short_term_density_half_life_days(
                        float(
                            config.get("coaching", {}).get(
                                "continuous_fatigue_half_life_days", 7
                            )
                        ),
                        recovery_half_life_hours=RECOVERY_HALF_LIFE_HOURS,
                    )
                    * 24.0
                )
                taxing_residual = max(
                    taxing_residual,
                    decay_recovery_load(
                        latest_taxing_load,
                        elapsed_taxing_hours,
                        half_life_hours=race_half_life_hours,
                    ),
                )
        # The projected state contains residue from every completed and
        # planned run. Looking only at the latest taxing session when one was
        # present ignored an intervening easy run and could approve
        # long/easy/quality on three consecutive days despite cumulative load
        # still exceeding the taxing-session readiness limit.
        if (recovery := estimate_recovery(state)) is not None:
            taxing_residual = max(
                taxing_residual,
                recovery.residual_load,
            )
        taxing_recovery_incomplete = bool(
            candidate in {"long", "quality"}
            and result.workout_type != WorkoutType.RACE
            and taxing_residual > TAXING_RUN_RESIDUAL_LIMIT
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
        # Role selection owns physiological purpose and recovery spacing. It
        # must not remove long/quality work by assuming later established easy
        # sessions can collapse to the beginner calibration minimum. The joint
        # date/frequency/distance allocator owns budget feasibility and can
        # choose fewer opportunities or different dates when the roles do not
        # fit.
        if candidate in {"easy", "long", "quality"}:
            if taxing_spacing_substitution:
                spacing_reason = "A second taxing workout was replaced with aerobic running because athlete-relative recovery load was still above the taxing-session reference."
                result = result.model_copy(
                    update={
                        "reasons": [
                            *result.reasons,
                            spacing_reason,
                        ]
                    }
                )
            return state, result


def _materialize_candidate_sessions(
    offsets: tuple[int, ...],
    daily_states: list[FitnessState],
    daily_state_options: list[list[FitnessState]],
    request: RecommendationRequest,
    config: dict,
    *,
    prefix_cache: _CandidatePrefixCache | None = None,
    recommendation_load_cache: _RecommendationLoadCache | None = None,
    projected_state_cache: _ProjectedStateCache | None = None,
    projected_distance_ranges: dict[
        int, tuple[float, float]
    ] | None = None,
    role_overrides_by_offset: dict[int, str] | None = None,
) -> list[_CandidateSession]:
    """Generate the real workout roles used to judge one calendar.

    Frequency selection and final schedule construction must evaluate the
    same recommendation logic. Keeping this materialization in one helper lets
    the outer frequency search pass the resulting sessions through the actual
    distance allocator instead of estimating a hypothetical average run.
    """

    sessions: list[_CandidateSession] = []
    start_position = 0

    def prefix_key(prefix_length: int) -> tuple[int, ...]:
        next_offset = offsets[prefix_length] if prefix_length < len(offsets) else -2
        return (*offsets[:prefix_length], -1, next_offset)

    use_prefix_cache = (
        prefix_cache is not None
        and projected_distance_ranges is None
        and not role_overrides_by_offset
    )
    if use_prefix_cache:
        # Candidate calendars at one frequency share many exact prefixes.
        # Their recommendation sequence is deterministic and the remaining
        # run count depends only on the total frequency and prefix position,
        # so a prefix can be reused without changing planner semantics. The
        # cache is created by ``adaptive_run_day_offsets`` and discarded at
        # the end of that one planning call; no weather, history, or simulated
        # state can leak into a later replan.
        for prefix_length in range(len(offsets), 0, -1):
            cached = prefix_cache.get(prefix_key(prefix_length))
            if cached is None:
                continue
            sessions = list(cached)
            start_position = prefix_length
            break
    planned = [
        session.recommendation
        for session in sessions
        if session.recommendation.workout_type != WorkoutType.REST
    ]
    for position in range(start_position, len(offsets)):
        offset = offsets[position]
        next_state_options = (
            daily_state_options[offsets[position + 1]]
            if position + 1 < len(offsets)
            else None
        )
        role = (role_overrides_by_offset or {}).get(offset)
        retained_role = role is not None
        if role is None:
            role = _elapsed_workout_role(
                daily_state_options[offset],
                planned,
                config,
                next_state_options=next_state_options,
                recommendation_load_cache=recommendation_load_cache,
                projected_state_cache=projected_state_cache,
            )
        state, result = _select_budgeted_timed_recommendation(
            daily_state_options[offset],
            planned,
            request,
            config,
            weekly_role=role,
            preserve_selected_role=retained_role,
            recommendation_load_cache=recommendation_load_cache,
            projected_state_cache=projected_state_cache,
        )
        projected_result = result
        projected_range = (projected_distance_ranges or {}).get(offset)
        if (
            projected_range is not None
            and result.distance_range_miles is not None
        ):
            projected_result = result.model_copy(
                update={"distance_range_miles": projected_range}
            )
        # Persist the same finalized recommendation that is projected into
        # later states.  Keeping ``result`` here while appending
        # ``projected_result`` only to ``planned`` made every reconciliation
        # candidate forget its allocated dose when it was converted back to
        # schedule days, so the next pass restarted from the smaller draft.
        sessions.append(_CandidateSession(offset, state, projected_result))
        if projected_result.workout_type == WorkoutType.REST:
            if use_prefix_cache:
                prefix_cache[prefix_key(position + 1)] = tuple(sessions)
            continue
        planned.append(projected_result)
        if use_prefix_cache:
            prefix_cache[prefix_key(position + 1)] = tuple(sessions)
    return sessions


def _adaptive_candidate_cost(
    offsets: tuple[int, ...],
    daily_states: list[FitnessState],
    daily_state_options: list[list[FitnessState]],
    request: RecommendationRequest,
    config: dict,
    target_distance_range: tuple[float, float],
    *,
    completed_miles_by_offset: dict[int, float] | None = None,
    role_loop_penalty: bool = True,
    include_mileage_path: bool = True,
    include_recovery_interactions: bool = True,
    include_cadence_pressure: bool = True,
    prefix_cache: _CandidatePrefixCache | None = None,
    recommendation_load_cache: _RecommendationLoadCache | None = None,
    projected_state_cache: _ProjectedStateCache | None = None,
    role_overrides_by_offset: dict[int, str] | None = None,
) -> float:
    """Score one continuous run-date combination across the entire horizon.

    Day seven is not a reset point: frequency, workout composition, recovery,
    and mileage are evaluated across the supplied horizon before the UI takes
    a slice.
    """
    completed_miles_by_offset = completed_miles_by_offset or {}
    sessions = _materialize_candidate_sessions(
        offsets,
        daily_states,
        daily_state_options,
        request,
        config,
        prefix_cache=prefix_cache,
        recommendation_load_cache=recommendation_load_cache,
        projected_state_cache=projected_state_cache,
        role_overrides_by_offset=role_overrides_by_offset,
    )
    planned: list[RecommendationResponse] = []
    # Completed mileage is already represented in the opening trailing load.
    # This map contains only proposed work so the reload boundary is not
    # double-counted.
    minimum_density_miles_by_offset: dict[int, float] = {}
    impulse_hours_by_offset: dict[int, float] = {}
    cost = 0.0
    for session in sessions:
        offset = session.offset
        state = session.state
        result = session.recommendation
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
        # Date/frequency selection must price the workout it actually chose.
        # Treating every extra easy or quality run as a hypothetical ten-minute
        # minimum made dense calendars look cheap, even though the allocator
        # subsequently preserved their ordinary athlete-relative distances.
        # The allocator may still shorten a selected run when the full joint
        # mileage problem requires it; this stage must not assume that every
        # workout will collapse to its absolute evidence floor.
        minimum_density_miles = result_midpoint
        minimum_density_miles_by_offset[offset] = (
            minimum_density_miles_by_offset.get(offset, 0.0)
            + minimum_density_miles
        )
        if result.planned_for is not None:
            impulse_hours_by_offset[offset] = (
                result.planned_for.hour
                + result.planned_for.minute / 60.0
                + result.planned_for.second / 3600.0
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
        easy_reference_miles = projected_recovery_reference_miles(daily_states[0])
        residual_load = _decayed_recovery_load(
            selected_raw_state,
            planned[:-1],
            state.as_of,
            easy_reference_miles,
            recommendation_load_cache=recommendation_load_cache,
        )
        proposed_load = _recommendation_load_units(
            result, easy_reference_miles
        )
        # Two load units are the recoverable envelope of roughly two ordinary
        # easy sessions. Rest replenishes that envelope exponentially, so a
        # larger workout can fit after more recovery while the same workout
        # becomes unattractive when prior session load is still present.
        if include_recovery_interactions:
            cost += _recovery_interaction_cost(residual_load, proposed_load)
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
        if candidate_load_ratio is not None:
            cost += max(0.0, candidate_load_ratio - 1.0) * 12.0

        # Date scoring above already uses residual-load × proposed-load. The
        # shared spacing function is reserved for choosing among same-day time
        # options so the same recovery evidence is not charged twice here.

    # Keep the cheap date prefilter aligned with the final whole-program
    # comparison: minimum candidate mileage follows one continuous cumulative
    # path, without separate seven- or fourteen-day budgets.
    if include_mileage_path:
        opening_distance_rate = (
            daily_states[0].recent_load.continuous_distance_miles
        )
        path_violation = _continuous_mileage_path_violation(
            [
                (
                    0.0
                    if opening_distance_rate is not None
                    else completed_miles_by_offset.get(offset, 0.0)
                )
                + minimum_density_miles_by_offset.get(offset, 0.0)
                for offset in range(len(daily_states))
            ],
            target_distance_range,
            session_tolerance_miles=sum(
                typical_easy_distance(daily_states[0])
            )
            / 2,
            opening_weekly_rate=opening_distance_rate,
            half_life_days=float(
                config.get("coaching", {}).get(
                    "continuous_fatigue_half_life_days", 7
                )
            ),
            opening_hour=_clock_hour(daily_states[0].as_of),
            impulse_hours=[
                impulse_hours_by_offset.get(offset)
                for offset in range(len(daily_states))
            ],
        )
        cost += path_violation**2 * 3.0

    typical_rest_days = int(
        config.get("coaching", {}).get("typical_rest_days_between_runs", 1)
    )
    preferred_gap_hours = max(1.0, (typical_rest_days + 1) * 24.0)
    if request.health_status != CurrentHealthStatus.NORMAL:
        preferred_gap_hours += 24.0
    planned_times = [
        item.planned_for
        for item in planned
        if item.planned_for is not None
    ]
    # Cadence pressure exists only while the candidate is genuinely short of
    # the full-horizon mileage need. Once two calendars both fund the target,
    # a shorter gap receives no independent reward: recovery and accumulated
    # density decide which distribution is better. This keeps sparse plans
    # from winning through procrastination without turning 48 hours into a
    # frequency target that manufactures consecutive days.
    candidate_miles = sum(minimum_density_miles_by_offset.values())
    ordinary_easy_miles = max(
        0.1,
        sum(typical_easy_distance(daily_states[0])) / 2,
    )
    cadence_need = _cadence_underfill_pressure(
        candidate_miles,
        target_distance_range,
        len(daily_states),
        ordinary_easy_miles,
    )
    if include_cadence_pressure and cadence_need > 0:
        cost += cadence_need * _elapsed_cadence_idle_cost(
            planned_times,
            preferred_gap_hours,
        )

    # There is deliberately no calendar cost for a consecutive date or a
    # three-day block. Each proposed session has already been charged against
    # the exact decayed load of every preceding session. Mileage-path fit and
    # finalized recovery therefore decide whether density is useful without a
    # hidden preference for an every-other-day calendar.
    if role_loop_penalty:
        role_sequence = [
            "long"
            if item.workout_type == WorkoutType.LONG
            else "quality"
            if item.workout_type
            in QUALITY_WORKOUT_TYPES
            else "easy"
            for item in planned
        ]
        for start in range(max(0, len(role_sequence) - 7)):
            if (
                role_sequence[start : start + 4]
                == role_sequence[start + 4 : start + 8]
            ):
                # Exact four-session role loops are a fragile optimizer
                # attractor: closed-loop adherence can repeat them
                # indefinitely. Apply this only while comparing calendars at
                # the same frequency, never as a reason to add/delete a run.
                cost += 1.0
    # A completed run immediately before the lookahead is already included by
    # ``_decayed_recovery_load``; the history/plan seam receives no separate
    # calendar penalty.
    # Long and quality recurrence are elapsed-time objectives, not positions
    # in a weekly sequence. Allow recovery to move them, but make an
    # unexplained drift beyond the ordinary 7–8 day window progressively
    # expensive across the continuous lookahead.
    cadence_specs = [
        (
            {WorkoutType.LONG},
            daily_states[0].days_since_long_run,
            float(
                config.get("coaching", {}).get(
                    "long_run_recency_reference_days", 7
                )
            ),
        ),
        (
            set(QUALITY_WORKOUT_TYPES),
            daily_states[0].days_since_quality_run,
            float(
                config.get("coaching", {}).get(
                    "quality_recency_reference_days", 7
                )
            ),
        ),
    ]
    for accepted_types, days_since, reference in cadence_specs:
        if days_since is None:
            continue
        occurrence_dates = {
            item.planned_for.date()
            for item in planned
            if item.workout_type in accepted_types and item.planned_for
        }
        age = days_since
        previous_date = daily_states[0].as_of.date()
        for state in daily_states:
            elapsed_days = max(0, (state.as_of.date() - previous_date).days)
            # Taper days suspend key-session recurrence rather than turning the
            # entire lookahead into a taper. The race itself resets quality
            # recency; after the race, ordinary elapsed-time scheduling resumes.
            if not _is_taper_or_race_date(config, state.as_of.date()):
                age += elapsed_days
                # Charge lateness at the instant an overdue key session occurs,
                # then reset its age. Resetting first made a nine-day recurrence
                # look free even though the preceding eight-day state was only at
                # the edge of the ordinary grace window.
                # The projection gate treats an unexplained cadence breach as
                # invalid. Give the optimizer the same semantics rather than
                # letting a small recovery-spacing advantage buy a ninth
                # calendar day. If recovery/readiness prevents the key session,
                # every earlier candidate receives the same pressure and the
                # later guarded recommendation can still win with its explicit
                # exception reason.
                cost += _key_session_cadence_overdue_cost(age, reference)
            if state.as_of.date() in occurrence_dates:
                age = 0.0
            previous_date = state.as_of.date()

    return cost


def _select_joint_finalists(
    scored_candidates: list[tuple[float, tuple[int, ...]]],
    limit: int = JOINT_DATE_FINALISTS,
    *,
    cadence_anchors: list[tuple[int, ...]] | None = None,
) -> list[tuple[float, tuple[int, ...]]]:
    """Preserve distinct origins and clustering before expensive scoring.

    The preliminary score cannot see finalized workout distances and roles.
    A tiny shortlist therefore keeps the numerical leader, a different plan
    origin, the least-clustered calendar, and opening-clustered counterparts
    for retained alternate-origin boundary variants. This lets the full
    model—not beam-search ordering or a double deferred to the far edge—decide
    whether an early consecutive block is worthwhile.
    """

    if not scored_candidates or limit <= 0:
        return []
    ranked = sorted(scored_candidates, key=lambda item: (item[0], item[1]))
    finalists = ranked[:1]
    first_origin = finalists[0][1][0]
    alternate_origin = next(
        (item for item in ranked[1:] if item[1][0] != first_origin),
        None,
    )
    if alternate_origin is not None:
        finalists.append(alternate_origin)
    least_clustered = min(
        ranked,
        key=lambda item: (
            sum(
                current - previous == 1
                for previous, current in zip(item[1], item[1][1:])
            ),
            item[0],
            item[1],
        ),
    )
    if least_clustered not in finalists:
        finalists.append(least_clustered)
    # A boundary variant can have the same useful near-term origin but a
    # different final lookahead date. Pair each retained alternate-origin
    # variant with the closest calendar that moves a consecutive block into
    # the first four sessions while preserving that far-edge endpoint.
    for reference_item in list(finalists):
        reference_offsets = reference_item[1]
        if reference_offsets[0] == first_origin:
            continue
        opening_clustered_candidates = [
            item
            for item in ranked[1:]
            if item[1][0] == reference_offsets[0]
            and item[1][-1] == reference_offsets[-1]
            and any(
                current - previous == 1
                for previous, current in zip(item[1][:4], item[1][1:4])
            )
        ]
        if not opening_clustered_candidates:
            continue
        counterpart = min(
            opening_clustered_candidates,
            key=lambda item: (
                sum(
                    abs(current - reference)
                    for current, reference in zip(
                        item[1], reference_offsets
                    )
                ),
                item[0],
                item[1],
            ),
        )
        if counterpart not in finalists:
            finalists.append(counterpart)
    finalists.extend(item for item in ranked[1:] if item not in finalists)
    finalists = finalists[:limit]

    # Fully score every canonical recovery-spaced anchor. The cheap
    # preliminary objective cannot see finalized roles or distances, and a
    # wider beam can change which origins happen to rank inside the ordinary
    # shortlist. Conditioning anchor admission on those origins therefore made
    # the search non-monotonic: widening the beam could remove a better plan.
    # There are at most four anchors, so admitting the anchors themselves
    # keeps the expensive work constant-sized and gives them no score
    # preference. Do not force clustered and terminal variants for every
    # anchor: that rejected expansion nearly doubled full distance-allocation
    # work at every frequency. The ordinary diverse shortlist and targeted
    # incumbent-edge neighborhoods retain those shapes where evidence shows
    # they matter.
    anchor_set = set(cadence_anchors or ())
    if anchor_set:
        for item in ranked:
            if item[1] in anchor_set:
                if item not in finalists:
                    finalists.append(item)
    return finalists


def _cadence_anchor_candidates(
    allowed_offsets: list[int],
    total_runs: int,
    preferred_gap_days: int,
    required_offsets: set[int],
) -> list[tuple[int, ...]]:
    """Keep a few recovery-spaced calendars alive through beam pruning.

    The cheap beam rank cannot see the role-aware distance allocation. It can
    therefore discard a calendar that begins after a sensible recovery block
    and then follows the athlete's ordinary cadence, even when full scoring
    strongly prefers it. These bounded anchors are candidates, not bonuses:
    they still have to earn a place in the ordinary full-score finalist list.
    """

    if total_runs < 2 or len(allowed_offsets) < total_runs:
        return []
    gap = max(1, preferred_gap_days)
    anchors: list[tuple[int, ...]] = []
    # Four openings cover today through a normal three-day recovery block
    # without making the work grow with the 21-day combinatorial space. Build
    # the cadence from that opening rather than pinning its final occurrence
    # to the hard-horizon edge. Edge-pinning makes the same recurring policy a
    # different search anchor after every daily rollover.
    for first in allowed_offsets[:4]:
        chosen = [first]
        while len(chosen) < total_runs:
            remaining_body = total_runs - len(chosen)
            available = [
                value
                for value in allowed_offsets
                if chosen[-1] < value
                and sum(
                    1
                    for later in allowed_offsets
                    if value < later
                )
                >= remaining_body - 1
            ]
            if not available:
                chosen = []
                break
            target = chosen[-1] + gap
            chosen.append(
                min(
                    available,
                    key=lambda value: (abs(value - target), value),
                )
            )
        if not chosen:
            continue
        candidate = tuple(chosen)
        if (
            len(candidate) == total_runs
            and len(set(candidate)) == total_runs
            and required_offsets.issubset(candidate)
            and candidate not in anchors
        ):
            anchors.append(candidate)
    return anchors


def _single_session_candidate_neighbors(
    offsets: tuple[int, ...],
    allowed_offsets: set[int],
    *,
    maximum_positions: int = 4,
) -> list[tuple[int, ...]]:
    """Return bounded one-day moves near the decision origin.

    Finalized workout roles can make one early session prefer a different day
    even when the rest of a recovery-spaced calendar is sound. Keeping these
    local variants prevents that choice from depending on a prior-plan warm
    start while avoiding an unbounded neighborhood search.
    """

    neighbors: list[tuple[int, ...]] = []
    positions = {
        *range(min(maximum_positions, len(offsets))),
        *range(max(0, len(offsets) - 2), len(offsets)),
    }
    for index in sorted(positions):
        for shift in (-1, 1):
            translated = tuple(
                value + shift if position == index else value
                for position, value in enumerate(offsets)
            )
            if (
                len(set(translated)) != len(translated)
                or tuple(sorted(translated)) != translated
                or not set(translated).issubset(allowed_offsets)
                or translated in neighbors
            ):
                continue
            neighbors.append(translated)
    return neighbors


def _near_term_refinement_neighbors(
    seeds: list[tuple[int, ...]],
    allowed_offsets: set[int],
    *,
    protected_limit: int = NEAR_TERM_STABILITY_DECISION_OFFSETS,
) -> list[tuple[int, ...]]:
    """Return deterministic one-day alternatives for every finalist seed."""

    return list(
        dict.fromkeys(
            neighbor
            for seed_offsets in seeds
            for neighbor in _single_session_candidate_neighbors(
                seed_offsets,
                allowed_offsets,
                maximum_positions=3,
            )
            if any(
                value < protected_limit
                for value in set(seed_offsets).symmetric_difference(neighbor)
            )
        )
    )


def _back_to_back_count_refinement_neighbors(
    seeds: list[tuple[int, ...]],
    allowed_offsets: set[int],
    *,
    protected_limit: int = NEAR_TERM_STABILITY_DECISION_OFFSETS,
) -> list[tuple[int, ...]]:
    """Expose a bounded lower-frequency alternative across beam searches.

    Each frequency is searched independently. A higher-frequency finalist can
    therefore reveal an excellent lower-frequency calendar that the adjacent
    beam omitted: keep the same long-horizon cadence, but remove the second
    day of an early back-to-back pair. Score that exact reduction and its
    ordinary one-day refinements with the full program model. This is candidate
    coverage only; it gives the reduced calendar no scoring preference.
    """

    reductions: list[tuple[int, ...]] = []
    for seed_offsets in seeds:
        for index in range(1, len(seed_offsets)):
            if seed_offsets[index] >= protected_limit:
                break
            if seed_offsets[index] - seed_offsets[index - 1] != 1:
                continue
            reduced = (
                *seed_offsets[:index],
                *seed_offsets[index + 1 :],
            )
            if reduced and reduced not in reductions:
                reductions.append(reduced)

    return list(
        dict.fromkeys(
            (
                *reductions,
                *(
                    neighbor
                    for reduction in reductions
                    for neighbor in _single_session_candidate_neighbors(
                        reduction,
                        allowed_offsets,
                        maximum_positions=3,
                    )
                    if neighbor[0] == reduction[0]
                    if any(
                        value < protected_limit
                        for value in set(reduction).symmetric_difference(
                            neighbor
                        )
                    )
                ),
            )
        )
    )


def _adaptive_run_day_offsets_for_frequency(
    daily_states: list[FitnessState],
    request: RecommendationRequest,
    config: dict,
    target_run_count: int,
    target_distance_range: tuple[float, float],
    *,
    daily_state_options: list[list[FitnessState]] | None = None,
    forced_rest_offsets: set[int] | None = None,
    completed_run_offsets: set[int] | None = None,
    completed_miles_by_offset: dict[int, float] | None = None,
    horizon_run_count: int | None = None,
    joint_program_scoring: bool = False,
    joint_cost_cache: dict[
        tuple[int, ...], tuple[float, float, float, float]
    ] | None = None,
    prefix_cache: _CandidatePrefixCache | None = None,
    recommendation_load_cache: _RecommendationLoadCache | None = None,
    projected_state_cache: _ProjectedStateCache | None = None,
    prior_run_offsets: set[int] | None = None,
    prior_distance_midpoints_by_date: dict[date, float] | None = None,
    expected_target_projector: ExpectedTargetProjector | None = None,
    planner_diagnostics: _PlannerSearchDiagnostics | None = None,
) -> list[int]:
    """Choose dates for one candidate frequency across a continuous horizon.

    The frequency is an enumeration detail used by the outer optimizer, not
    an athlete target. No seven-day sub-window receives its own run quota.
    """
    if request.health_status == CurrentHealthStatus.PAIN_OR_INJURY_CONCERN:
        return []
    planner_diagnostics = (
        planner_diagnostics
        if planner_diagnostics is not None
        else _PlannerSearchDiagnostics()
    )
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
    goal = configured_race_goal(config)
    required_offsets = {
        offset
        for offset in allowed_offsets
        if goal is not None and daily_states[offset].as_of.date() == goal[1]
    }
    if len(required_offsets) > total_runs:
        return []
    optional_offsets = [
        offset for offset in allowed_offsets if offset not in required_offsets
    ]
    optional_run_count = total_runs - len(required_offsets)
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
    weather_costs: dict[int, float] = {}
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
            max(0.0, load_ratio - 1.0) * 12.0
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
        # The prefilter deliberately has no gap cost. Its job is only to keep
        # the combinatorial search tractable; calendar density is evaluated by
        # the full mileage and recovery model below. ``retain_diverse`` keeps
        # both dense and spaced alternatives alive until then.
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

    def diversity_signature(offsets: tuple[int, ...]) -> tuple:
        """Keep materially different calendars alive until full scoring.

        The cheap prefilter cannot see workout roles or projected recovery.
        Retaining only its global top-N can therefore erase every useful
        double, front-loaded plan, or wider recovery pattern before the real
        coaching objective evaluates it.
        """

        gaps = tuple(
            current - previous
            for previous, current in zip(offsets, offsets[1:])
        )
        return (
            offsets[0] if offsets else None,
            offsets[-1] if offsets else None,
            sum(gap == 1 for gap in gaps),
            max(gaps, default=0),
            gaps[-2:],
        )

    def retain_diverse(
        values: list[tuple[float, tuple[int, ...], int]],
    ) -> list[tuple[float, tuple[int, ...], int]]:
        ranked = sorted(values, key=lambda item: (item[0], item[1]))
        if len(ranked) <= MAX_ADAPTIVE_CANDIDATES:
            return ranked
        direct_count = max(1, MAX_ADAPTIVE_CANDIDATES // 2)
        selected = ranked[:direct_count]
        selected_offsets = {item[1] for item in selected}
        signatures = {diversity_signature(item[1]) for item in selected}
        for item in ranked[direct_count:]:
            signature = diversity_signature(item[1])
            if signature in signatures:
                continue
            selected.append(item)
            selected_offsets.add(item[1])
            signatures.add(signature)
            if len(selected) == MAX_ADAPTIVE_CANDIDATES:
                break
        if len(selected) < MAX_ADAPTIVE_CANDIDATES:
            selected.extend(
                item
                for item in ranked
                if item[1] not in selected_offsets
            )
        return sorted(
            selected[:MAX_ADAPTIVE_CANDIDATES],
            key=lambda item: (item[0], item[1]),
        )

    # Fourteen days was still small enough to enumerate before applying the
    # full coaching model. At 21 days, common frequencies create hundreds of
    # thousands or millions of combinations. Build the same kind of
    # whole-horizon candidates incrementally and retain the strongest partial
    # calendars at each depth. The display boundary never participates.
    combination_count = comb(len(optional_offsets), optional_run_count)
    if optional_run_count == 0:
        candidate_offsets = [tuple(sorted(required_offsets))]
    elif combination_count <= 5_000:
        candidate_offsets = [
            tuple(sorted((*values, *required_offsets)))
            for values in combinations(optional_offsets, optional_run_count)
        ]
    else:
        beam: list[tuple[float, tuple[int, ...], int]] = [(0.0, (), -1)]
        for position in range(optional_run_count):
            remaining = optional_run_count - position - 1
            expanded: list[tuple[float, tuple[int, ...], int]] = []
            for score, prefix, previous_index in beam:
                first_index = previous_index + 1
                final_index = len(optional_offsets) - remaining
                for allowed_index in range(first_index, final_index):
                    offset = optional_offsets[allowed_index]
                    expanded.append(
                        (
                            score + incremental_prefilter_cost(prefix, offset),
                            (*prefix, offset),
                            allowed_index,
                        )
                    )
            beam = retain_diverse(expanded)
            if not beam:
                break
        candidate_offsets = [
            tuple(sorted((*prefix, *required_offsets)))
            for _, prefix, _ in beam
        ]

    if len(candidate_offsets) > MAX_ADAPTIVE_CANDIDATES:
        ranked_candidates = [
            (*prefilter_cost(offsets)[:1], offsets, -1)
            for offsets in candidate_offsets
        ]
        candidate_offsets = [
            offsets for _, offsets, _ in retain_diverse(ranked_candidates)
        ]
    cadence_anchors = _cadence_anchor_candidates(
        allowed_offsets,
        total_runs,
        int(
            config.get("coaching", {}).get(
                "typical_rest_days_between_runs", 1
            )
        )
        + 1,
        required_offsets,
    )
    for anchor in cadence_anchors:
        if anchor not in candidate_offsets:
            candidate_offsets.append(anchor)
        for neighbor in _single_session_candidate_neighbors(
            anchor,
            set(allowed_offsets),
        ):
            if neighbor not in candidate_offsets:
                candidate_offsets.append(neighbor)
    # Everything admitted through this point is derived solely from the
    # current evidence, horizon, and deterministic search anchors. Candidates
    # added below are useful incumbent warm starts, but must not redefine the
    # nominally unconstrained optimizer result merely because a different
    # prior plan happened to expose them.
    prior_independent_offsets = set(candidate_offsets)
    prior_candidate = tuple(
        sorted(set(prior_run_offsets or set()) & set(allowed_offsets))
    )
    # A beam is an approximation and can omit a previously optimal calendar
    # merely because the horizon origin moved. Supply the translated prior
    # solution as a warm start, but give it no preference in scoring: it wins
    # only when the ordinary recovery/load/program objective still says it is
    # best. Nearby count variants let the same solution compete when the new
    # far-edge day changes the useful horizon frequency.
    warm_start_candidates: list[tuple[int, ...]] = []
    incumbent_neighborhood: list[tuple[int, ...]] = []
    incumbent_edge_replacements: list[tuple[int, ...]] = []
    incumbent_edge_replacement_neighbors: list[tuple[int, ...]] = []
    incumbent_edge_extensions: list[tuple[int, ...]] = []
    incumbent_edge_neighbors: list[tuple[int, ...]] = []
    incumbent_stable_extensions: list[tuple[int, ...]] = []
    incumbent_stable_reductions: list[tuple[int, ...]] = []
    if prior_candidate:
        difference = total_runs - len(prior_candidate)
        if difference == 0:
            warm_start_candidates = [prior_candidate]
            allowed_set = set(allowed_offsets)
            # A receding horizon can expose the next key-session recurrence
            # without changing the useful run count. Replacing one or two
            # obsolete terminal slots with newly visible far-edge capacity is
            # the faithful same-frequency continuation; one-day shifts alone
            # cannot reach it. These remain unpreferred candidates.
            for replacement_count in range(
                1,
                min(2, len(prior_candidate)) + 1,
            ):
                inherited = prior_candidate[:-replacement_count]
                available = sorted(allowed_set - set(inherited))
                if len(available) < replacement_count:
                    continue
                replacement = tuple(
                    sorted((*inherited, *available[-replacement_count:]))
                )
                if replacement != prior_candidate:
                    incumbent_edge_replacements.append(replacement)
            incumbent_edge_replacement_neighbors = [
                neighbor
                for replacement in incumbent_edge_replacements
                for neighbor in _single_session_candidate_neighbors(
                    replacement,
                    allowed_set,
                )
            ]
            # Receding the origin often changes the full score of the first
            # few incumbent dates together. Guarantee a bounded local
            # neighborhood so the final model can move that prefix earlier or
            # later without relying on the approximate beam ranking.
            for stop in range(1, min(4, len(prior_candidate))):
                for shift in (-1, 1):
                    translated = tuple(
                        value + shift if index < stop else value
                        for index, value in enumerate(prior_candidate)
                    )
                    if (
                        len(set(translated)) != len(translated)
                        or tuple(sorted(translated)) != translated
                        or not set(translated).issubset(allowed_set)
                    ):
                        continue
                    incumbent_neighborhood.append(translated)
            # A later near-term workout can move independently while the
            # earlier dates remain optimal.  Prefix-only translations cannot
            # expose, for example, Tue/Thu/Fri from a Tue/Thu/Sat incumbent;
            # pruning an unused time option could then change which calendar
            # happens to reach final scoring. Guarantee these bounded local
            # alternatives without assigning the incumbent any score bonus.
            for index in range(min(4, len(prior_candidate))):
                for shift in (-1, 1):
                    translated = tuple(
                        value + shift if position == index else value
                        for position, value in enumerate(prior_candidate)
                    )
                    if (
                        len(set(translated)) != len(translated)
                        or tuple(sorted(translated)) != translated
                        or not set(translated).issubset(allowed_set)
                    ):
                        continue
                    incumbent_neighborhood.append(translated)
        elif 0 < difference <= 3:
            available = sorted(set(allowed_offsets) - set(prior_candidate))
            warm_start_candidates = [
                tuple(sorted((*prior_candidate, *extras)))
                for extras in combinations(available, difference)
            ]
            # When the decision origin advances, the faithful continuation of
            # a previously optimal receding plan fills newly exposed capacity
            # at the far edge.  The cheap seed-distance prefilter strongly
            # favors adding dates near the origin and used to discard this
            # candidate before role-aware mileage/recovery scoring.  Preserve
            # the edge extension as an *unpreferred* candidate: it still wins
            # only if the same full objective scores it best.
            if len(available) >= difference:
                incumbent_edge_extensions = [
                    tuple(
                        sorted(
                            (*prior_candidate, *available[-difference:])
                        )
                    )
                ]
                incumbent_edge_neighbors = [
                    neighbor
                    for extension in incumbent_edge_extensions
                    for neighbor in _single_session_candidate_neighbors(
                        extension,
                        set(allowed_offsets),
                    )
                ]
            # Consuming a compliant session removes it from tomorrow's
            # horizon. If the shifted target really needs a replacement, the
            # faithful alternatives are not limited to appending it on the
            # final day (which may sit directly after a long run) or inserting
            # it immediately near the origin. Preserve a bounded set of the
            # strongest extensions wholly outside the protected date window,
            # then let the full recovery/load objective place the new work.
            stable_available = [
                offset
                for offset in available
                if offset >= NEAR_TERM_STABILITY_DECISION_OFFSETS
            ]
            incumbent_stable_extensions = sorted(
                (
                    tuple(sorted((*prior_candidate, *extras)))
                    for extras in combinations(stable_available, difference)
                ),
                key=prefilter_cost,
            )[:4]
        elif -3 <= difference < 0:
            warm_start_candidates = [
                tuple(
                    offset
                    for offset in prior_candidate
                    if offset not in removals
                )
                for removals in combinations(prior_candidate, -difference)
            ]
            removable_after_protected_window = [
                offset
                for offset in prior_candidate
                if offset >= NEAR_TERM_STABILITY_DECISION_OFFSETS
            ]
            incumbent_stable_reductions = sorted(
                (
                    tuple(
                        offset
                        for offset in prior_candidate
                        if offset not in removals
                    )
                    for removals in combinations(
                        removable_after_protected_window,
                        -difference,
                    )
                ),
                key=prefilter_cost,
            )[:4]
        warm_start_candidates = sorted(
            warm_start_candidates,
            key=prefilter_cost,
        )[:8]
        for extension in incumbent_edge_extensions:
            if extension not in warm_start_candidates:
                warm_start_candidates.append(extension)
        for extension in incumbent_stable_extensions:
            if extension not in warm_start_candidates:
                warm_start_candidates.append(extension)
        for reduction in incumbent_stable_reductions:
            if reduction not in warm_start_candidates:
                warm_start_candidates.append(reduction)
    for warm_start in warm_start_candidates:
        if warm_start not in candidate_offsets:
            candidate_offsets.append(warm_start)
    for neighbor in incumbent_neighborhood:
        if neighbor not in candidate_offsets:
            candidate_offsets.append(neighbor)
    for replacement in incumbent_edge_replacements:
        if replacement not in candidate_offsets:
            candidate_offsets.append(replacement)
    for neighbor in incumbent_edge_replacement_neighbors:
        if neighbor not in candidate_offsets:
            candidate_offsets.append(neighbor)
    for neighbor in incumbent_edge_neighbors:
        if neighbor not in candidate_offsets:
            candidate_offsets.append(neighbor)

    scored_candidates: list[tuple[float, tuple[int, ...]]] = []
    for offsets in candidate_offsets:
        cost = _adaptive_candidate_cost(
            offsets,
            daily_states,
            daily_state_options,
            request,
            config,
            target_distance_range,
            completed_miles_by_offset=completed_miles_by_offset,
            prefix_cache=prefix_cache,
            recommendation_load_cache=recommendation_load_cache,
            projected_state_cache=projected_state_cache,
        )
        scored_candidates.append((cost, offsets))
        if best_cost is None or (cost, offsets) < (best_cost, best_offsets):
            best_cost = cost
            best_offsets = offsets
    if joint_program_scoring and scored_candidates:
        # The beam deliberately uses a cheap prefilter, so refine its best
        # complete calendars around the planning-origin seam before invoking
        # the expensive distance allocator. Translating a near-term prefix by
        # one day exposes plans that preserve the same long-horizon density
        # while allowing one more (or less) recovery day after the latest
        # completed run. Without this local refinement, the beam can retain
        # an early lexicographic calendar and miss a lower-cost schedule that
        # the full recovery/role model would prefer.
        allowed_set = set(allowed_offsets)
        known_candidates = {offsets for _, offsets in scored_candidates}
        refinement_seeds = sorted(
            scored_candidates, key=lambda item: (item[0], item[1])
        )[:JOINT_DATE_FINALISTS]
        refinements: list[tuple[int, ...]] = []
        for _, offsets in refinement_seeds:
            for stop in range(1, len(offsets)):
                for shift in (-1, 1):
                    translated = tuple(
                        value + shift if index < stop else value
                        for index, value in enumerate(offsets)
                    )
                    if (
                        translated in known_candidates
                        or len(set(translated)) != len(translated)
                        or tuple(sorted(translated)) != translated
                        or not set(translated).issubset(allowed_set)
                    ):
                        continue
                    known_candidates.add(translated)
                    refinements.append(translated)
        for offsets in refinements:
            cost = _adaptive_candidate_cost(
                offsets,
                daily_states,
                daily_state_options,
                request,
                config,
                target_distance_range,
                completed_miles_by_offset=completed_miles_by_offset,
                prefix_cache=prefix_cache,
                recommendation_load_cache=recommendation_load_cache,
                projected_state_cache=projected_state_cache,
            )
            scored_candidates.append((cost, offsets))

        if planner_diagnostics is not None:
            planner_diagnostics.candidate_count += len(scored_candidates)

        # Preliminary recommendations are intentionally cheap enough to score
        # dozens of calendars. Before choosing dates, run the strongest small
        # shortlist through the actual role-aware distance allocator. Without
        # this pass the outer frequency search can compare final programs, but
        # only after each frequency has already committed to dates that may
        # require clustered medium-long runs to fund the mileage target.
        prior_independent_scored = [
            item
            for item in scored_candidates
            if item[1] in prior_independent_offsets
        ]
        finalists = _select_joint_finalists(
            prior_independent_scored,
            cadence_anchors=cadence_anchors,
        )
        # The translated prior solution is a search warm start, not a scoring
        # preference.  Still, it must reach the same role-aware final scoring
        # as the beam finalists; otherwise an approximate preliminary ranking
        # can discard a calendar that the full objective considers better.
        # Keep this bounded to the exact-count neighborhood and one natural
        # far-edge extension for a changed count.
        required_incumbents = {
            *incumbent_edge_replacements,
            *incumbent_edge_replacement_neighbors,
            *incumbent_edge_extensions,
            *incumbent_edge_neighbors,
            *incumbent_stable_extensions,
            *incumbent_stable_reductions,
        }
        if len(prior_candidate) == total_runs:
            required_incumbents.update(
                {
                    prior_candidate,
                    *incumbent_neighborhood,
                }
            )
        if required_incumbents:
            for required in required_incumbents:
                incumbent = next(
                    (
                        item
                        for item in scored_candidates
                        if item[1] == required
                    ),
                    None,
                )
                if incumbent is not None and incumbent not in finalists:
                    finalists.append(incumbent)
        if planner_diagnostics is not None:
            planner_diagnostics.full_score_count += len(finalists)
        joint_choices: list[tuple[float, tuple[int, ...]]] = []
        joint_breakdowns: dict[tuple[int, ...], PlannerScoreBreakdown] = {}
        ordinary_easy_midpoint = sum(
            typical_easy_distance(daily_states[0])
        ) / 2
        for _, offsets in finalists:
            diagnostic_details: dict[str, object] = {}
            joint_cost = _joint_candidate_program_cost(
                offsets,
                daily_states,
                daily_state_options,
                request,
                config,
                target_distance_range,
                completed_miles_by_offset=completed_miles_by_offset,
                prefix_cache=prefix_cache,
                recommendation_load_cache=recommendation_load_cache,
                projected_state_cache=projected_state_cache,
                expected_target_projector=expected_target_projector,
                prior_distance_midpoints_by_date=(
                    prior_distance_midpoints_by_date
                ),
                diagnostic_details=diagnostic_details,
            )
            if joint_cost_cache is not None:
                joint_cost_cache[offsets] = joint_cost
            recovery_calendar_cost = _adaptive_candidate_cost(
                offsets,
                daily_states,
                daily_state_options,
                request,
                config,
                target_distance_range,
                completed_miles_by_offset=completed_miles_by_offset,
                include_mileage_path=False,
                include_recovery_interactions=False,
                include_cadence_pressure=False,
                prefix_cache=prefix_cache,
                recommendation_load_cache=recommendation_load_cache,
                projected_state_cache=projected_state_cache,
            )
            joint_choices.append(
                (
                    _program_selection_cost(
                        joint_cost,
                        recovery_calendar_cost,
                        ordinary_easy_midpoint,
                    ),
                    offsets,
                )
            )
            joint_breakdowns[offsets] = _score_breakdown(
                offsets,
                joint_cost,
                recovery_calendar_cost,
                ordinary_easy_midpoint,
                diagnostic_details,
            )
        # Report the faithful continuation itself, not the cheapest local
        # neighbor admitted alongside it. Conflating those two made a
        # one-day rewrite appear to be the prior calendar in transition
        # audits, defeating the purpose of the diagnostic.
        continuation_offsets = set(incumbent_edge_extensions)
        continuation_offsets.update(incumbent_stable_extensions)
        continuation_offsets.update(incumbent_stable_reductions)
        if len(prior_candidate) == total_runs:
            continuation_offsets.add(prior_candidate)
        if planner_diagnostics is not None and continuation_offsets:
            continuation = min(
                (
                    joint_breakdowns[offsets]
                    for offsets in continuation_offsets
                    if offsets in joint_breakdowns
                ),
                key=lambda item: (item.total_cost, item.offsets),
                default=None,
            )
            if continuation is not None and (
                planner_diagnostics.prior_continuation_candidate is None
                or continuation.total_cost
                < planner_diagnostics.prior_continuation_candidate.total_cost
            ):
                planner_diagnostics.prior_continuation_candidate = continuation
        if planner_diagnostics is not None and prior_candidate:
            stability_limit = min(
                NEAR_TERM_STABILITY_DECISION_OFFSETS,
                len(daily_states),
            )
            prior_near_term = {
                offset
                for offset in prior_candidate
                if offset < stability_limit
            }
            stable_items = [
                (offsets, breakdown)
                for offsets, breakdown in joint_breakdowns.items()
                if {
                    offset
                    for offset in offsets
                    if offset < stability_limit
                }
                == prior_near_term
            ]
            for offsets, _ in stable_items:
                if offsets not in planner_diagnostics.near_term_stable_offsets:
                    planner_diagnostics.near_term_stable_offsets.append(offsets)
            stable = min(
                (breakdown for _, breakdown in stable_items),
                key=lambda item: (item.total_cost, item.offsets),
                default=None,
            )
            if stable is not None and (
                planner_diagnostics.near_term_stable_candidate is None
                or stable.total_cost
                < planner_diagnostics.near_term_stable_candidate.total_cost
            ):
                planner_diagnostics.near_term_stable_candidate = stable
        if planner_diagnostics is not None:
            independent_choices = [
                item
                for item in joint_choices
                if item[1] in prior_independent_offsets
            ]
            if independent_choices:
                independent_winner = min(
                    independent_choices,
                    key=lambda item: (item[0], item[1]),
                )[1]
                planner_diagnostics.prior_independent_frequency_candidates.append(
                    joint_breakdowns[independent_winner]
                )
        return list(min(joint_choices)[1])
    return list(best_offsets or ())


def _joint_candidate_program_cost(
    offsets: tuple[int, ...],
    daily_states: list[FitnessState],
    daily_state_options: list[list[FitnessState]],
    request: RecommendationRequest,
    config: dict,
    target_distance_range: tuple[float, float],
    *,
    completed_miles_by_offset: dict[int, float] | None = None,
    prefix_cache: _CandidatePrefixCache | None = None,
    recommendation_load_cache: _RecommendationLoadCache | None = None,
    projected_state_cache: _ProjectedStateCache | None = None,
    expected_target_projector: ExpectedTargetProjector | None = None,
    prior_distance_midpoints_by_date: dict[date, float] | None = None,
    role_overrides_by_offset: dict[int, str] | None = None,
    diagnostic_details: dict[str, object] | None = None,
    reconcile_allocated_prefix: bool = False,
    _allocated_override: list[WeeklyScheduleDay] | None = None,
    _sessions_override: list[_CandidateSession] | None = None,
) -> tuple[float, float, float, float]:
    """Return target, support, shape, and finalized recovery costs.

    Elapsed key-session cadence is already evaluated by the calendar scorer; it
    is not recomputed here. Among those calendars, one that can fund the target
    beats one that under- or over-fills it. Ordinary aerobic support is then
    brought toward the athlete's normal easy range before shape permits a
    purposeful medium-long exposure. This prevents residual two-mile fillers
    from coexisting with an oversized aerobic run merely because the total is
    numerically correct.
    """

    completed_miles_by_offset = completed_miles_by_offset or {}
    sessions = _sessions_override or _materialize_candidate_sessions(
        offsets,
        daily_states,
        daily_state_options,
        request,
        config,
        prefix_cache=prefix_cache,
        recommendation_load_cache=recommendation_load_cache,
        projected_state_cache=projected_state_cache,
        role_overrides_by_offset=role_overrides_by_offset,
    )
    def candidate_days_for(
        materialized: list[_CandidateSession],
    ) -> tuple[list[WeeklyScheduleDay], dict[int, _CandidateSession]]:
        materialized_by_offset = {
            session.offset: session for session in materialized
        }
        candidate_days: list[WeeklyScheduleDay] = []
        for offset, state in enumerate(daily_states):
            session = materialized_by_offset.get(offset)
            result = session.recommendation if session else None
            role = (
                "long_run"
                if result and result.workout_type == WorkoutType.LONG
                else "quality_run"
                if result
                and result.workout_type in QUALITY_WORKOUT_TYPES
                else "easy_run"
                if result and result.workout_type != WorkoutType.REST
                else "guardrail_rest_day"
                if result
                else "rest_day"
            )
            candidate_days.append(
                WeeklyScheduleDay(
                    date=state.as_of.date(),
                    planned_at=(session.state.as_of if session else None),
                    recommendation=result,
                    day_role=role,
                    rationale=(
                        "Candidate program used for joint frequency and "
                        "distance scoring."
                    ),
                )
            )
        return candidate_days, materialized_by_offset

    candidate_days, by_offset = candidate_days_for(sessions)
    allocation_states = _allocation_states_at_planned_times(
        candidate_days,
        daily_states,
        daily_state_options,
    )

    candidate_recommendations = [
        session.recommendation for session in sessions
    ]
    if expected_target_projector is not None:
        projected_target_ranges = expected_target_projector(
            candidate_recommendations,
            len(daily_states) + int(SOFT_CONTINUATION_DAYS),
        )
        hard_target_ranges = projected_target_ranges[: len(daily_states)]
        soft_target_ranges = projected_target_ranges
    else:
        hard_target_ranges = tuple(target_distance_range for _ in daily_states)
        soft_target_ranges = hard_target_ranges
    if len(hard_target_ranges) != len(daily_states):
        raise ValueError("Expected target trajectory must match the planning horizon")
    integrated_target = _integrated_target_range(hard_target_ranges)
    average_target = _average_target_range(hard_target_ranges)
    soft_average_target = _average_target_range(soft_target_ranges)
    completed_miles = sum(completed_miles_by_offset.values())
    remaining_target = (
        max(0.0, integrated_target[0] - completed_miles),
        max(0.0, integrated_target[1] - completed_miles),
    )
    ordinary_easy_midpoint = sum(typical_easy_distance(daily_states[0])) / 2

    def allocation_selection_cost(
        candidate: list[WeeklyScheduleDay],
    ) -> float:
        # Calendar dates and workout roles are fixed at this point. Compare
        # the allocator's viable distance shapes with the exact same joint
        # program objective used to compare calendars, instead of committing
        # to the allocator's cheaper balance surrogate first.
        joint_cost = _joint_candidate_program_cost(
            offsets,
            daily_states,
            daily_state_options,
            request,
            config,
            target_distance_range,
            completed_miles_by_offset=completed_miles_by_offset,
            expected_target_projector=expected_target_projector,
            prior_distance_midpoints_by_date=(
                prior_distance_midpoints_by_date
            ),
            role_overrides_by_offset=role_overrides_by_offset,
            reconcile_allocated_prefix=False,
            _allocated_override=candidate,
            _sessions_override=sessions,
        )
        return _program_selection_cost(
            joint_cost,
            0.0,
            ordinary_easy_midpoint,
        )

    use_outer_allocation_selection = (
        reconcile_allocated_prefix and _allocated_override is None
    )
    allocated = (
        _allocated_override
        if _allocated_override is not None
        else _allocate_visible_distance_ranges(
            candidate_days,
            allocation_states,
            remaining_target,
            config,
            weekly_target_range=average_target,
            assignments_per_total=(
                3 if use_outer_allocation_selection else 2
            ),
            allocation_selector=(
                allocation_selection_cost
                if use_outer_allocation_selection
                else None
            ),
            allocation_candidate_limit=(
                4 if use_outer_allocation_selection else 1
            ),
            prior_distance_midpoints_by_date=(
                prior_distance_midpoints_by_date
            ),
        )
    )
    if reconcile_allocated_prefix and _allocated_override is None:
        reconciliation_converged = False
        for _ in range(2):
            projected_ranges = {
                index: day.recommendation.distance_range_miles
                for index, day in enumerate(allocated)
                if day.recommendation is not None
                and day.recommendation.distance_range_miles is not None
            }
            reconciled_sessions = _materialize_candidate_sessions(
                offsets,
                daily_states,
                daily_state_options,
                request,
                config,
                prefix_cache=None,
                recommendation_load_cache=recommendation_load_cache,
                projected_state_cache=projected_state_cache,
                projected_distance_ranges=projected_ranges,
                role_overrides_by_offset=role_overrides_by_offset,
            )
            reconciled_days, reconciled_by_offset = candidate_days_for(
                reconciled_sessions
            )
            reconciled_allocation_states = _allocation_states_at_planned_times(
                reconciled_days,
                daily_states,
                daily_state_options,
            )
            # ``allocation_selection_cost`` closes over ``sessions``. Update
            # that fixed-point state before scoring the next allocation;
            # doing it afterward evaluates every candidate against the prior
            # pass's smaller provisional doses and leaves downstream recovery
            # exactly one iteration stale.
            sessions = reconciled_sessions
            by_offset = reconciled_by_offset
            reconciled = _allocate_visible_distance_ranges(
                reconciled_days,
                reconciled_allocation_states,
                remaining_target,
                config,
                weekly_target_range=average_target,
                assignments_per_total=3,
                preserve_projected_recovery_caps=True,
                allocation_selector=allocation_selection_cost,
                allocation_candidate_limit=4,
                prior_distance_midpoints_by_date=(
                    prior_distance_midpoints_by_date
                ),
            )
            previous_signature = tuple(
                (
                    day.recommendation.workout_type,
                    day.recommendation.planned_for,
                    day.recommendation.distance_range_miles,
                )
                if day.recommendation is not None
                else None
                for day in allocated
            )
            reconciled_signature = tuple(
                (
                    day.recommendation.workout_type,
                    day.recommendation.planned_for,
                    day.recommendation.distance_range_miles,
                )
                if day.recommendation is not None
                else None
                for day in reconciled
            )
            allocated = reconciled
            if previous_signature == reconciled_signature:
                reconciliation_converged = True
                break
        # The bounded reconciliation loop may end because it exhausted its
        # pass budget rather than because the allocation reached a fixed
        # point.  In that case ``allocated`` contains the dose shown to the
        # athlete while ``sessions`` still contains the preceding pass's
        # dose.  That made downstream state one allocator quantum stale (for
        # example, projecting 4.25 miles after prescribing 4.50), so exact
        # compliance appeared to create new load and could rewrite the next
        # few dates. Materialize the final selected doses once more only in
        # that unconverged case; this is state reconciliation and does not
        # reopen allocation.
        if not reconciliation_converged:
            final_projected_ranges = {
                index: day.recommendation.distance_range_miles
                for index, day in enumerate(allocated)
                if day.recommendation is not None
                and day.recommendation.distance_range_miles is not None
            }
            sessions = _materialize_candidate_sessions(
                offsets,
                daily_states,
                daily_state_options,
                request,
                config,
                prefix_cache=None,
                recommendation_load_cache=recommendation_load_cache,
                projected_state_cache=projected_state_cache,
                projected_distance_ranges=final_projected_ranges,
                role_overrides_by_offset=role_overrides_by_offset,
            )
            _, by_offset = candidate_days_for(sessions)
    # Compare the whole receding plan to one continuous cumulative target
    # path. The target is expressed in familiar miles/week, but day 7 and day
    # 14 have no special status and the day-21 edge cannot hide unfunded work.
    opening_distance_rate = daily_states[0].recent_load.continuous_distance_miles
    daily_path_miles = [
            (
                0.0
                if opening_distance_rate is not None
                else completed_miles_by_offset.get(index, 0.0)
            )
            + (
                _midpoint(day.recommendation)
                if day.recommendation
                and day.recommendation.workout_type != WorkoutType.REST
                else 0.0
            )
            for index, day in enumerate(allocated)
        ]
    fatigue_half_life_days = float(
        config.get("coaching", {}).get(
            "continuous_fatigue_half_life_days", 7
        )
    )
    path_tolerance = (
        ordinary_easy_midpoint
        * log(2.0)
        * 7.0
        / max(0.1, fatigue_half_life_days)
        if opening_distance_rate is not None
        else ordinary_easy_midpoint
    )
    # The continuous path answers a near-term transition question: can the
    # candidate move from today's real load toward the target without a sharp
    # deficit or overload? This prevents an otherwise adequate recurring
    # policy from indefinitely postponing its first occurrence. Extending the
    # transient objective all the way to the arbitrary hard edge would again
    # make the score depend on whether the next recurring session falls just
    # inside or outside day 21.
    transition_days = min(
        len(daily_path_miles),
        max(1, ceil(fatigue_half_life_days)),
    )
    path_violation = _continuous_mileage_path_violation(
        daily_path_miles[:transition_days],
        average_target,
        session_tolerance_miles=path_tolerance,
        opening_weekly_rate=opening_distance_rate,
        half_life_days=fatigue_half_life_days,
        target_weekly_ranges=hard_target_ranges[:transition_days],
        impulse_hours=_planned_impulse_hours(allocated)[:transition_days],
        opening_hour=_clock_hour(daily_states[0].as_of),
    )
    # Judge sustained intent without making day 21 a deadline, then reject
    # actual hard-horizon funding misses larger than the one-session sawtooth
    # caused by a receding boundary. The soft recovery continuation below
    # deliberately supplies no mileage credit to either calculation.
    policy_rate = _soft_projected_program_rate(
        allocated,
        soft_average_target,
        ordinary_easy_midpoint,
        boundary_session=_recent_boundary_session(daily_states[0]),
    ) or 0.0
    policy_rate_violation = max(
        0.0,
        soft_average_target[0] - policy_rate,
        policy_rate - soft_average_target[1],
    )
    hard_funding_violation = _gate_aligned_hard_funding_violation(
        _hard_horizon_funding_violation(
            allocated,
            remaining_target,
            boundary_allowance_miles=max(
                ordinary_easy_midpoint,
                (
                    _recent_boundary_session(daily_states[0])
                    or (0.0, 0.0)
                )[1],
            ),
        ),
        ordinary_easy_midpoint,
    )
    target_violation = (
        path_violation
        + policy_rate_violation
        + hard_funding_violation
    )

    medium_long_offsets = [
        index
        for index, day in enumerate(allocated)
        if day.day_role == "medium_long_run"
    ]
    support_violation = 0.0
    fragmentation_violation = 0.0
    for index, day in enumerate(allocated):
        result = day.recommendation
        if (
            not result
            or result.workout_type != WorkoutType.EASY
            or day.day_role == "medium_long_run"
        ):
            continue
        normal_easy_low = typical_easy_distance(daily_states[index])[0]
        if normal_easy_low <= 0:
            continue
        # Missing aerobic support is additive. Squaring each fractional
        # shortfall made a calendar of many 25%-short runs look cheaper than
        # a smaller number of normal athlete-relative sessions, which is the
        # fragmentation failure this term exists to prevent. Keep it soft and
        # continuous, but do not let splitting one deficit into many pieces
        # make that deficit disappear numerically.
        shortfall_fraction = _aerobic_shortfall_fraction(
            _midpoint(result), normal_easy_low
        )
        support_violation += shortfall_fraction
        if (
            request.health_status == CurrentHealthStatus.NORMAL
            and _midpoint(result) < normal_easy_low - 1e-9
        ):
            # Fragmentation is additive in the *missing share* of an ordinary
            # session. A binary whole-session charge made a 2.95-mile midpoint
            # versus a 3.00-mile threshold cost the same as a genuinely tiny
            # run, creating a large replan cliff from distance rounding. The
            # proportional sum still prevents splitting one deficit across
            # several short dates from making that deficit disappear.
            fragmentation_violation += shortfall_fraction
    support_violation += _rolling_aerobic_support_violation(
        allocated,
        daily_states,
    )
    long_offsets = [
        index
        for index, day in enumerate(allocated)
        if day.recommendation
        and day.recommendation.workout_type == WorkoutType.LONG
    ]
    shape_violation = fragmentation_violation
    long_shape_violation = 0.0
    # Retained long-run durability is soft but meaningful. A candidate may
    # shorten the long run to preserve recovery or whole-program load, but it
    # may not make a higher-frequency calendar look artificially cheap by
    # collapsing that run toward ordinary-easy distance. Normalize the loss
    # across the athlete's own easy-to-long distinction: losing the entire
    # distinction is one role-shape violation, while a small step-back remains
    # inexpensive and fully available to the optimizer.
    for index in long_offsets:
        original = by_offset.get(index)
        allocated_result = allocated[index].recommendation
        if original is None or allocated_result is None:
            continue
        preferred_long = max(
            _midpoint(original.recommendation),
            single_session_progression_reference_miles(
                daily_states[index]
            ),
        )
        easy_midpoint = sum(typical_easy_distance(daily_states[index])) / 2
        meaningful_long = easy_midpoint + max(
            0.1,
            round(easy_midpoint * 0.15, 1),
        )
        preservation_span = max(
            0.25,
            preferred_long - meaningful_long,
        )
        lost_distinction = max(
            0.0,
            preferred_long - _midpoint(allocated_result),
        ) / preservation_span
        long_shape_violation += lost_distinction * lost_distinction
    shape_violation += long_shape_violation
    # Price the *amount* of secondary-endurance expansion rather than charging
    # a flat fee for the label. A small, useful extension should not make an
    # extra run day cheaper, while an easy run approaching long-run territory
    # still has to justify its added recovery cost. This matches the allocator
    # below and avoids turning workout taxonomy into a frequency rule.
    medium_long_shape_violation = 0.0
    for index in medium_long_offsets:
        result = allocated[index].recommendation
        if result is None:
            continue
        expansion_reference = _ordinary_easy_expansion_reference(
            daily_states[index]
        )
        expansion_band = max(
            0.5,
            expansion_reference
            - sum(typical_easy_distance(daily_states[index])) / 2,
        )
        expansion = max(
            0.0,
            (_midpoint(result) - expansion_reference) / expansion_band,
        )
        medium_long_shape_violation += (
            expansion * expansion * SECONDARY_ENDURANCE_SHAPE_FRACTION
        )
    shape_violation += medium_long_shape_violation
    if not long_offsets:
        # A medium-long run is a secondary durability session, not a way to
        # avoid ever programming the primary long-run lane.
        shape_violation += len(medium_long_offsets)
    recovery_cost_components: dict[str, float] = {}
    finalized_session_states = _allocation_states_at_planned_times(
        allocated,
        daily_states,
        daily_state_options,
    )
    finalized_recovery_cost = _finalized_program_recovery_cost(
        allocated,
        daily_states,
        config,
        target_distance_range,
        session_states=finalized_session_states,
        diagnostic_components=recovery_cost_components,
    )
    if diagnostic_details is not None:
        diagnostic_details.update(
            {
                "fragmentation_violation": fragmentation_violation,
                "long_shape_violation": long_shape_violation,
                "medium_long_shape_violation": medium_long_shape_violation,
                "recovery_cost_components": recovery_cost_components,
                "allocation_trace": [
                    (
                        f"{index}:{day.day_role}:"
                        f"{day.recommendation.workout_type.value}:"
                        f"{_midpoint(day.recommendation):.3f}"
                    )
                    for index, day in enumerate(allocated)
                    if day.recommendation
                    and day.recommendation.workout_type != WorkoutType.REST
                ],
                "projected_state_trace": [
                    (
                        f"{session.offset}:"
                        f"at={session.state.as_of.isoformat()}:"
                        f"dose={session.recommendation.distance_range_miles}:"
                        f"readiness={session.recommendation.readiness.value}:"
                        f"recovery={session.state.recovery_residual_load}:"
                        f"continuous={session.state.recent_load.continuous_distance_miles}:"
                        f"short={session.state.recent_load.continuous_short_term_distance_miles}:"
                        f"easy={session.state.typical_easy_run_miles}"
                    )
                    for session in sessions
                ],
            }
        )
    return (
        target_violation,
        support_violation,
        shape_violation,
        finalized_recovery_cost,
    )


def _program_selection_cost(
    joint_cost: tuple[float, float, float, float],
    coaching_cost: float,
    ordinary_easy_midpoint: float,
) -> float:
    """Combine program fit with recovery without hard priority cliffs.

    ``coaching_cost`` prices elapsed key-session cadence, weather, and overload.
    The joint allocator contributes the mileage breach,
    finalized-distance recovery interaction, aerobic support, and role shape.
    A complete one-session mileage miss is deliberately stronger than a soft
    role-shape preference, while recovery can still reject genuinely unsafe
    density.
    """

    (
        target_violation,
        support_violation,
        shape_violation,
        finalized_recovery_cost,
    ) = joint_cost
    target_session_violation = target_violation / max(
        0.1, ordinary_easy_midpoint
    )
    return (
        coaching_cost
        + TARGET_FIT_UNIT
        * target_session_violation
        * target_session_violation
        + PROGRAM_FIT_UNIT
        * (
            support_violation * AEROBIC_SUPPORT_FIT_FRACTION
            + shape_violation
        )
        + finalized_recovery_cost
    )


def adaptive_run_day_offsets(
    daily_states: list[FitnessState],
    request: RecommendationRequest,
    config: dict,
    target_run_count: int,
    target_distance_range: tuple[float, float],
    *,
    daily_state_options: list[list[FitnessState]] | None = None,
    forced_rest_offsets: set[int] | None = None,
    completed_run_offsets: set[int] | None = None,
    completed_miles_by_offset: dict[int, float] | None = None,
    prior_run_offsets: set[int] | None = None,
    prior_distance_midpoints_by_date: dict[date, float] | None = None,
    prior_roles_by_offset: dict[int, str] | None = None,
    expected_target_projector: ExpectedTargetProjector | None = None,
    _reuse_candidate_work: bool = True,
    _planner_diagnostics: _PlannerSearchDiagnostics | None = None,
) -> list[int]:
    """Select both frequency and dates from one continuous-horizon model.

    ``target_run_count`` is retained for API compatibility but deliberately
    ignored. Mileage coverage rewards enough useful sessions; recovery/load
    interactions, minimum useful distance, and spacing reject excess ones.
    """
    if request.health_status == CurrentHealthStatus.PAIN_OR_INJURY_CONCERN:
        return []
    del target_run_count
    # The stability selector is part of normal planning, not merely response
    # diagnostics. Keep an invocation-local collector when the caller does
    # not request that diagnostic payload.
    planner_diagnostics = (
        _planner_diagnostics
        if _planner_diagnostics is not None
        else _PlannerSearchDiagnostics()
    )
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
    # Frequency is an optimizer dimension, not a coaching target. Search a
    # broad bounded band and let recurring key-session cadence, squared load
    # fit, and finalized recovery justify every added day. Long-run allocation
    # is independently budgeted below, so a low-count plan cannot inflate one
    # long run until it numerically replaces the rest of the program.
    maximum_horizon_count = min(
        maximum_horizon_count,
        central_horizon_count + 3,
    )
    minimum_horizon_count = max(
        1,
        maximum_horizon_count - MAX_HORIZON_COUNT_OPTIONS + 1,
    )
    frequency_options = range(
        minimum_horizon_count,
        maximum_horizon_count + 1,
    )
    joint_cost_cache: dict[
        tuple[int, ...], tuple[float, float, float, float]
    ] = {}
    # This switch exists for equivalence tests and profiling. Production uses
    # invocation-local reuse; disabling it executes the same candidate search
    # without either new memoization layer.
    prefix_cache: _CandidatePrefixCache | None = (
        {} if _reuse_candidate_work else None
    )
    recommendation_load_cache: _RecommendationLoadCache | None = (
        {} if _reuse_candidate_work else None
    )
    projected_state_cache: _ProjectedStateCache | None = (
        {} if _reuse_candidate_work else None
    )
    choices: list[tuple[float, int, list[int]]] = []
    choice_breakdowns: dict[tuple[int, ...], PlannerScoreBreakdown] = {}
    for horizon_run_count in frequency_options:
        independent_count_before = len(
            planner_diagnostics.prior_independent_frequency_candidates
        )
        offsets = _adaptive_run_day_offsets_for_frequency(
            daily_states,
            request,
            config,
            max(1, round(horizon_run_count / horizon_scale)),
            target_distance_range,
            daily_state_options=daily_state_options,
            forced_rest_offsets=forced_rest_offsets,
            completed_run_offsets=completed_run_offsets,
            completed_miles_by_offset=completed_miles_by_offset,
            horizon_run_count=horizon_run_count,
            joint_program_scoring=True,
            joint_cost_cache=joint_cost_cache,
            prefix_cache=prefix_cache,
            recommendation_load_cache=recommendation_load_cache,
            projected_state_cache=projected_state_cache,
            prior_run_offsets=prior_run_offsets,
            prior_distance_midpoints_by_date=(
                prior_distance_midpoints_by_date
            ),
            expected_target_projector=expected_target_projector,
            planner_diagnostics=planner_diagnostics,
        )
        if not offsets:
            continue
        independent_frequency_candidates = (
            planner_diagnostics.prior_independent_frequency_candidates
        )
        if len(independent_frequency_candidates) > independent_count_before:
            horizon_offsets = tuple(
                independent_frequency_candidates[-1].offsets
            )
            offsets = list(horizon_offsets)
        else:
            horizon_offsets = tuple(offsets)
        coaching_cost = _adaptive_candidate_cost(
            horizon_offsets,
            daily_states,
            daily_state_options,
            request,
            config,
            target_distance_range,
            completed_miles_by_offset=completed_miles_by_offset,
            role_loop_penalty=False,
            include_mileage_path=False,
            include_recovery_interactions=False,
            include_cadence_pressure=False,
            prefix_cache=prefix_cache,
            recommendation_load_cache=recommendation_load_cache,
            projected_state_cache=projected_state_cache,
        )
        joint_cost = joint_cost_cache.get(horizon_offsets)
        if joint_cost is None:
            joint_cost = _joint_candidate_program_cost(
                horizon_offsets,
                daily_states,
                daily_state_options,
                request,
                config,
                target_distance_range,
                completed_miles_by_offset=completed_miles_by_offset,
                prefix_cache=prefix_cache,
                recommendation_load_cache=recommendation_load_cache,
                projected_state_cache=projected_state_cache,
                expected_target_projector=expected_target_projector,
                prior_distance_midpoints_by_date=(
                    prior_distance_midpoints_by_date
                ),
            )
        selection_cost = _program_selection_cost(
            joint_cost,
            coaching_cost,
            ordinary_easy_midpoint,
        )
        choice_breakdowns[horizon_offsets] = _score_breakdown(
            horizon_offsets,
            joint_cost,
            coaching_cost,
            ordinary_easy_midpoint,
        )
        choices.append(
            (
                selection_cost,
                horizon_run_count,
                offsets,
            )
        )
    if not choices:
        return []
    # Frequency is not a coaching target and receives no lower-count tie band.
    # After finding the unconstrained optimum, a prior calendar may break an
    # objective tie only inside the capped model-resolution band and only for
    # its next-four-day dates. Distance allocation remains unconstrained.
    unconstrained_offsets = tuple(
        min(
            choices,
            key=lambda item: (item[0], item[1], item[2]),
        )[2]
    )
    unconstrained_breakdown = choice_breakdowns[unconstrained_offsets]
    stable_breakdowns: list[PlannerScoreBreakdown] = []
    for stable_offsets in planner_diagnostics.near_term_stable_offsets:
        stable_joint_cost = joint_cost_cache.get(stable_offsets)
        if stable_joint_cost is None:
            continue
        stable_coaching_cost = _adaptive_candidate_cost(
            stable_offsets,
            daily_states,
            daily_state_options,
            request,
            config,
            target_distance_range,
            completed_miles_by_offset=completed_miles_by_offset,
            role_loop_penalty=False,
            include_mileage_path=False,
            include_recovery_interactions=False,
            include_cadence_pressure=False,
            prefix_cache=prefix_cache,
            recommendation_load_cache=recommendation_load_cache,
            projected_state_cache=projected_state_cache,
        )
        stable_breakdowns.append(
            _score_breakdown(
                stable_offsets,
                stable_joint_cost,
                stable_coaching_cost,
                ordinary_easy_midpoint,
            )
        )
    stable_breakdown = min(
        stable_breakdowns,
        key=lambda item: (item.total_cost, item.offsets),
        default=None,
    )
    # Preliminary materialization chooses workout purpose before the distance
    # allocator knows whether an earlier easy day will become medium-long.
    # Re-evaluate the bounded frequency finalists with those allocated doses
    # fed into every downstream projected state. Otherwise exact compliance
    # can reveal load the original plan itself failed to project and rewrite
    # the next few dates immediately after upload.
    unconstrained_refinement_pool = sorted(
        (tuple(choice[2]) for choice in choices),
        key=lambda offsets: (
            choice_breakdowns[offsets].total_cost,
            len(offsets),
            offsets,
        ),
    )[:3]
    near_term_neighbors = _near_term_refinement_neighbors(
        unconstrained_refinement_pool,
        set(range(len(daily_states))) - set(forced_rest_offsets),
    )
    unconstrained_refinement_pool.extend(near_term_neighbors)
    count_refinement_neighbors = _back_to_back_count_refinement_neighbors(
        unconstrained_refinement_pool[:3],
        set(range(len(daily_states))) - set(forced_rest_offsets),
    )
    unconstrained_refinement_pool.extend(count_refinement_neighbors)
    unconstrained_refinement_pool = list(
        dict.fromkeys(unconstrained_refinement_pool)
    )
    stable_refinement_pool = sorted(
        stable_breakdowns,
        key=lambda item: (item.total_cost, item.offsets),
    )[:2]
    prior_continuation = planner_diagnostics.prior_continuation_candidate
    protected_limit = min(
        NEAR_TERM_STABILITY_DECISION_OFFSETS,
        len(daily_states),
    )
    translated_prior_near_term = {
        offset
        for offset in prior_run_offsets or set()
        if 0 <= offset < protected_limit
        and offset not in forced_rest_offsets
        and offset not in completed_run_offsets
    }
    continuation_matches_protected_dates = (
        prior_continuation is not None
        and {
            offset
            for offset in prior_continuation.offsets
            if offset < protected_limit
        }
        == translated_prior_near_term
    )
    if (
        prior_continuation is not None
        and continuation_matches_protected_dates
        and all(
            tuple(item.offsets) != tuple(prior_continuation.offsets)
            for item in stable_refinement_pool
        )
    ):
        stable_refinement_pool.append(prior_continuation)
    refinement_offsets = {
        *unconstrained_refinement_pool,
        *(tuple(item.offsets) for item in stable_refinement_pool),
    }
    reconciled_breakdowns: dict[tuple[int, ...], PlannerScoreBreakdown] = {}
    for candidate_offsets in refinement_offsets:
        diagnostic_details: dict[str, object] = {}
        reconciled_joint_cost = _joint_candidate_program_cost(
            candidate_offsets,
            daily_states,
            daily_state_options,
            request,
            config,
            target_distance_range,
            completed_miles_by_offset=completed_miles_by_offset,
            prefix_cache=None,
            recommendation_load_cache=recommendation_load_cache,
            projected_state_cache=projected_state_cache,
            expected_target_projector=expected_target_projector,
            prior_distance_midpoints_by_date=(
                prior_distance_midpoints_by_date
            ),
            diagnostic_details=diagnostic_details,
            reconcile_allocated_prefix=True,
        )
        existing = choice_breakdowns.get(candidate_offsets)
        if existing is None:
            existing = next(
                (
                    item
                    for item in stable_refinement_pool
                    if tuple(item.offsets) == candidate_offsets
                ),
                None,
            )
        coaching_cost = (
            existing.coaching_cost
            if existing is not None
            else _adaptive_candidate_cost(
                candidate_offsets,
                daily_states,
                daily_state_options,
                request,
                config,
                target_distance_range,
                completed_miles_by_offset=completed_miles_by_offset,
                role_loop_penalty=False,
                include_mileage_path=False,
                include_recovery_interactions=False,
                include_cadence_pressure=False,
                prefix_cache=prefix_cache,
                recommendation_load_cache=recommendation_load_cache,
                projected_state_cache=projected_state_cache,
            )
        )
        reconciled_breakdowns[candidate_offsets] = _score_breakdown(
            candidate_offsets,
            reconciled_joint_cost,
            coaching_cost,
            ordinary_easy_midpoint,
            diagnostic_details,
        )
    choice_breakdowns.update(reconciled_breakdowns)
    unconstrained_offsets = min(
        unconstrained_refinement_pool,
        key=lambda offsets: (
            choice_breakdowns[offsets].total_cost,
            len(offsets),
            offsets,
        ),
    )
    unconstrained_breakdown = choice_breakdowns[unconstrained_offsets]
    # Date-only warm starts are insufficient for stability: rematerializing
    # the prior dates can silently turn tomorrow's quality session into easy
    # running and move quality to a later date. Score the actual retained
    # near-term purposes as a separate candidate. This does not constrain the
    # unconstrained search, and normal recovery logic may still substitute an
    # unsafe taxing role with easy running.
    protected_role_overrides = {
        offset: role
        for offset, role in (prior_roles_by_offset or {}).items()
        if 0 <= offset < protected_limit
        and offset in translated_prior_near_term
    }
    role_stable_breakdowns: list[PlannerScoreBreakdown] = []
    if protected_role_overrides:
        for item in stable_refinement_pool:
            candidate_offsets = tuple(item.offsets)
            diagnostic_details: dict[str, object] = {}
            role_stable_joint_cost = _joint_candidate_program_cost(
                candidate_offsets,
                daily_states,
                daily_state_options,
                request,
                config,
                target_distance_range,
                completed_miles_by_offset=completed_miles_by_offset,
                prefix_cache=None,
                recommendation_load_cache=recommendation_load_cache,
                projected_state_cache=projected_state_cache,
                expected_target_projector=expected_target_projector,
                prior_distance_midpoints_by_date=(
                    prior_distance_midpoints_by_date
                ),
                role_overrides_by_offset=protected_role_overrides,
                diagnostic_details=diagnostic_details,
                reconcile_allocated_prefix=True,
            )
            role_stable_coaching_cost = _adaptive_candidate_cost(
                candidate_offsets,
                daily_states,
                daily_state_options,
                request,
                config,
                target_distance_range,
                completed_miles_by_offset=completed_miles_by_offset,
                role_loop_penalty=False,
                include_mileage_path=False,
                include_recovery_interactions=False,
                include_cadence_pressure=False,
                prefix_cache=None,
                recommendation_load_cache=recommendation_load_cache,
                projected_state_cache=projected_state_cache,
                role_overrides_by_offset=protected_role_overrides,
            )
            role_stable_breakdowns.append(
                _score_breakdown(
                    candidate_offsets,
                    role_stable_joint_cost,
                    role_stable_coaching_cost,
                    ordinary_easy_midpoint,
                    diagnostic_details,
                )
            )
    stable_breakdown = min(
        role_stable_breakdowns
        or [
            reconciled_breakdowns[tuple(item.offsets)]
            for item in stable_refinement_pool
        ],
        key=lambda item: (item.total_cost, item.offsets),
        default=None,
    )
    stability_tolerance = _near_term_stability_tolerance(
        unconstrained_breakdown.total_cost
    )
    winning_offsets = unconstrained_offsets
    winning_breakdown = unconstrained_breakdown
    if (
        stable_breakdown is not None
        and stable_breakdown.total_cost
        <= unconstrained_breakdown.total_cost + stability_tolerance
    ):
        winning_offsets = tuple(stable_breakdown.offsets)
        winning_breakdown = stable_breakdown
    planner_diagnostics.unconstrained_winning_candidate = (
        unconstrained_breakdown
    )
    planner_diagnostics.near_term_stable_candidate = stable_breakdown
    planner_diagnostics.stability_tolerance = stability_tolerance
    planner_diagnostics.stability_tie_break_applied = (
        winning_offsets != unconstrained_offsets
    )
    planner_diagnostics.winning_role_overrides = (
        dict(protected_role_overrides)
        if role_stable_breakdowns
        and stable_breakdown is not None
        and tuple(stable_breakdown.offsets) == winning_offsets
        and winning_offsets != unconstrained_offsets
        else {}
    )
    if _planner_diagnostics is not None:
        _planner_diagnostics.winning_candidate = winning_breakdown
    if _planner_diagnostics is not None:
        translated_prior = tuple(
            sorted(
                offset
                for offset in prior_run_offsets or set()
                if 0 <= offset < len(daily_states)
                and offset not in forced_rest_offsets
                and offset not in completed_run_offsets
            )
        )
        if translated_prior:
            prior_coaching_cost = _adaptive_candidate_cost(
                translated_prior,
                daily_states,
                daily_state_options,
                request,
                config,
                target_distance_range,
                completed_miles_by_offset=completed_miles_by_offset,
                role_loop_penalty=False,
                include_mileage_path=False,
                include_recovery_interactions=False,
                include_cadence_pressure=False,
                prefix_cache=prefix_cache,
                recommendation_load_cache=recommendation_load_cache,
                projected_state_cache=projected_state_cache,
            )
            prior_joint_cost = joint_cost_cache.get(translated_prior)
            if prior_joint_cost is None:
                prior_joint_cost = _joint_candidate_program_cost(
                    translated_prior,
                    daily_states,
                    daily_state_options,
                    request,
                    config,
                    target_distance_range,
                    completed_miles_by_offset=completed_miles_by_offset,
                    prefix_cache=prefix_cache,
                    recommendation_load_cache=recommendation_load_cache,
                    projected_state_cache=projected_state_cache,
                    expected_target_projector=expected_target_projector,
                    prior_distance_midpoints_by_date=(
                        prior_distance_midpoints_by_date
                    ),
                )
            _planner_diagnostics.translated_prior_candidate = (
                _score_breakdown(
                    translated_prior,
                    prior_joint_cost,
                    prior_coaching_cost,
                    ordinary_easy_midpoint,
                )
            )
    return list(winning_offsets)


def _allocation_states_at_planned_times(
    days: list[WeeklyScheduleDay],
    daily_states: list[FitnessState],
    daily_state_options: list[list[FitnessState]],
) -> list[FitnessState]:
    """Use each workout's proposed clock time for per-session allocation.

    ``daily_states[0]`` intentionally represents the opening decision instant
    for cumulative target accounting.  On the opening calendar day that can
    be midnight while the retained workout is at 07:00 or 19:00.  Distance
    safety and recovery limits belong to the workout instant, not the refresh
    instant, so select the matching timed option without changing the opening
    state used elsewhere by the rolling-horizon objective.
    """

    result = list(daily_states)
    for index, day in enumerate(days):
        if index >= len(result) or index >= len(daily_state_options):
            break
        planned_at = day.planned_at
        if planned_at is None and day.recommendation is not None:
            planned_at = day.recommendation.planned_for
        if planned_at is None:
            continue
        timed_state = next(
            (
                option
                for option in daily_state_options[index]
                if option.as_of == planned_at
            ),
            None,
        )
        if timed_state is not None:
            result[index] = timed_state
    return result


def _allocate_visible_distance_ranges(
    days: list[WeeklyScheduleDay],
    daily_states: list[FitnessState],
    target_range: tuple[float, float],
    config: dict,
    *,
    weekly_target_range: tuple[float, float] | None = None,
    assignments_per_total: int = ALLOCATION_ASSIGNMENTS_PER_TOTAL,
    _prune_dominated_allocations: bool = True,
    preserve_projected_recovery_caps: bool = False,
    allocation_selector: Callable[[list[WeeklyScheduleDay]], float] | None = None,
    allocation_candidate_limit: int = 1,
    prior_distance_midpoints_by_date: dict[date, float] | None = None,
) -> list[WeeklyScheduleDay]:
    """Jointly allocate session distance and retain only meaningful roles.

    The solver compares a semantically distinct long-run plan with an aerobic
    endurance alternative. It optimizes the supplied horizon on a quarter-mile
    midpoint grid, balancing target coverage, local load density, per-session load,
    progression limits, and role meaning. Recovery/caution sessions remain
    fixed and are never enlarged to make a mileage number work.
    """
    updated = list(days)
    prior_distance_midpoints_by_date = (
        prior_distance_midpoints_by_date or {}
    )
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
            (
                result.readiness.value != "ready"
                and result.workout_type != WorkoutType.EASY
            )
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
            if result.workout_type in QUALITY_WORKOUT_TYPES
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
            easy_midpoint = sum(easy_reference) / 2
            meaningful_margin = max(
                0.1,
                round(easy_midpoint * 0.15, 1),
            )
            minimum = max(
                easy_reference[0],
                ceil((easy_midpoint + meaningful_margin) * 4 - 1e-9)
                / 4,
            )
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
            maintained_reference = (
                single_session_progression_reference_miles(
                    daily_states[index]
                )
            )
            configured_long_progression = float(
                config.get("coaching", {}).get(
                    "long_run_target_progression_fraction",
                    0.05,
                )
            )
            if weekly_target_range is not None:
                program_load_reference = max(
                    daily_states[index].recent_load.capacity_reference_miles
                    or 0.0,
                    daily_states[index].recent_load.continuous_distance_miles
                    or 0.0,
                )
                program_headroom = (
                    max(
                        0.0,
                        sum(weekly_target_range)
                        / 2
                        / program_load_reference
                        - 1.0,
                    )
                    if program_load_reference > 0
                    else configured_long_progression
                )
                planned_long_progression = min(
                    configured_long_progression,
                    program_headroom,
                )
            else:
                planned_long_progression = configured_long_progression
            planned_progression_target = (
                floor(
                    maintained_reference
                    * (1.0 + planned_long_progression)
                    * 4
                    + 0.5
                )
                / 4
            )
            # Keep a maintained long-run distance available when the weekly
            # and progression ceilings support it. Also expose the coaching-
            # derived preferred progression: provisional recovery may have
            # priced a larger preceding easy run that the joint allocator can
            # trim. Sequential recovery is reconciled below before the key
            # durability session is sacrificed.
            maximum = max(
                maximum,
                min(
                    progression_ceiling,
                    max(
                        maintained_reference,
                        planned_progression_target,
                    ),
                ),
            )
            # Demonstrated durability is a preferred target, not a hard floor.
            # Making every completed progression the next long run's minimum
            # creates a ratchet: long distance compounds faster than total
            # program load and eventually deletes useful aerobic days. The
            # joint allocator may step below the newest maximum when recovery
            # or whole-program balance warrants it; retained capacity remains
            # available as the preferred target and future ceiling.
            preferred = min(
                maximum,
                max(
                    minimum,
                    (lower + upper) / 2,
                    planned_progression_target,
                ),
            )
            weight = 1.20
        elif role == "quality":
            # Preserve a useful quality stimulus before deleting the role.
            # Its minimum is the athlete-relative ten-minute aerobic exposure;
            # the scaling pass below converts very short sessions to controlled
            # pickups instead of forcing a universal mileage template.
            minimum = _established_easy_midpoint_floor(
                daily_states[index], result
            )
            # Quality distance is set by the workout's warm-up, work dose,
            # recoveries, and cool-down—not by whatever weekly mileage remains.
            # It may scale down below the original range when load requires,
            # but ordinary budget allocation never enlarges the session.
            original_midpoint = (lower + upper) / 2
            quality_center = round(original_midpoint * 4) / 4
            preferred = quality_center
            maximum = preferred
            weight = 1.05
        else:
            # A selected established-athlete run day must support an ordinary
            # aerobic session. Otherwise splitting the same mileage across
            # more dates lowers modeled recovery and lets a dense calendar of
            # tiny runs defeat fewer useful sessions. This is athlete-relative,
            # not a global mileage floor. A first session already shortened by
            # recorded recovery remains eligible below the baseline; future
            # candidate-created recovery pressure must instead choose a later
            # date rather than manufacture another support slot.
            original_midpoint = (lower + upper) / 2
            established_midpoint = easy_reference[0] + min(
                0.25,
                max(0.0, easy_reference[1] - easy_reference[0]) / 2,
            )
            minimum = established_midpoint
            maximum = max(upper, aerobic_maximum)
            preferred = min(
                maximum,
                ceil((sum(easy_reference) / 2) * 2 - 1e-9) / 2,
            )
            # Mileage after more recovery is cheaper than the same mileage in
            # a compressed slot. This permits a longer aerobic run followed
            # by a shorter pre-long easy run when that better preserves load.
            weight = 1.0 + min(4, gap) * 0.25
            shortened_below_ordinary = upper < easy_reference[1] - 1e-9
            observed_safety_cap = _observed_easy_safety_cap(
                daily_states[index], result
            )
            if (
                index == 0
                and shortened_below_ordinary
                and observed_safety_cap
            ):
                # Options are prescription midpoints, whereas ``upper`` is
                # the top of the executable range. Cap the midpoint at the
                # midpoint already approved by observed recovery or other
                # material caution. Only the opening state is observed;
                # future states retain the continuous projected-recovery rule
                # below so small decay or display-rounding changes cannot
                # create an expansion cliff. Merely becoming offset zero
                # after a midnight refresh is also not safety evidence: a
                # later same-day slot may already clear the guardrail.
                maximum = (lower + upper) / 2
                minimum = min(minimum, maximum)
                preferred = min(preferred, maximum)
            elif preserve_projected_recovery_caps:
                recovery_pressure = _easy_volume_recovery_pressure(result)
                if recovery_pressure > 0:
                    # Keep projected recovery meaningful without converting a
                    # half-mile display-rounding boundary into an all-or-none
                    # expansion gate. First expose an athlete-relative
                    # secondary-endurance ceiling continuously. Only the last
                    # sliver of recovery pressure controls headroom beyond
                    # that into the full aerobic maximum.
                    original_midpoint = (lower + upper) / 2
                    secondary_ceiling = max(
                        original_midpoint,
                        _ordinary_easy_expansion_reference(
                            daily_states[index]
                        )
                        + 0.5,
                    )
                    ordinary_headroom = 1.0 - recovery_pressure
                    full_expansion_headroom = max(
                        0.0,
                        1.0 - recovery_pressure / 0.05,
                    )
                    maximum = (
                        original_midpoint
                        + (secondary_ceiling - original_midpoint)
                        * ordinary_headroom
                        + max(0.0, aerobic_maximum - secondary_ceiling)
                        * full_expansion_headroom
                    )
                    minimum = min(minimum, maximum)
                    preferred = min(preferred, maximum)
                elif shortened_below_ordinary:
                    # A shortened fixture with no continuous recovery evidence
                    # may represent another projected constraint. Preserve it.
                    maximum = (lower + upper) / 2
                    minimum = min(minimum, maximum)
                    preferred = min(preferred, maximum)
        # Every option represents the prescription midpoint. Quarter-mile
        # centers support ordinary half-mile-wide route ranges without moving
        # the value the optimizer actually budgeted.
        maximum = max(minimum, maximum)
        options = _distance_options(minimum, maximum)
        records.append(
            {
                "index": index,
                "role": role,
                "weight": weight,
                "minimum": minimum,
                "maximum": maximum,
                "preferred": preferred,
                "options": options,
                "easy_midpoint": sum(easy_reference) / 2,
                "expansion_reference": (
                    _ordinary_easy_expansion_reference(daily_states[index])
                ),
                "aerobic_options": _distance_options(
                    easy_reference[1],
                    max(easy_reference[1], aerobic_maximum),
                ),
                "prior_midpoint": prior_distance_midpoints_by_date.get(
                    day.date
                ),
            }
        )
        previous_run_index = index

    if not records:
        return updated

    ordinary_easy_midpoint = sum(typical_easy_distance(daily_states[0])) / 2
    long_margin = max(
        0.1,
        round(ordinary_easy_midpoint * 0.15, 1),
    )

    if weekly_target_range is not None:
        # A long run is funded first, but it is not allowed to consume the
        # whole recurring program. Reserve the actual prescribed quality dose
        # (when present) and one established aerobic-support session before
        # exposing long-run growth. This is a dynamic budget from the current
        # athlete and target—not a fixed long-run percentage—and prevents
        # successful weekly long runs from compounding faster than the program
        # that must support them.
        quality_preferences = [
            record["preferred"]
            for record in records
            if record["role"] == "quality"
        ]
        quality_budget = (
            float(median(quality_preferences))
            if quality_preferences
            else 0.0
        )
        aerobic_support_budget = _established_easy_midpoint_floor(
            daily_states[0]
        )
        weekly_program_midpoint = sum(weekly_target_range) / 2
        long_program_budget = floor(
            max(
                ordinary_easy_midpoint + long_margin,
                weekly_program_midpoint
                - quality_budget
                - aerobic_support_budget,
            )
            * 4
            + 1e-9
        ) / 4
        for record in records:
            if record["role"] != "long":
                continue
            record["maximum"] = min(
                record["maximum"], long_program_budget
            )
            record["minimum"] = min(
                record["minimum"], record["maximum"]
            )
            record["preferred"] = min(
                record["preferred"], record["maximum"]
            )
            record["options"] = [
                option
                for option in record["options"]
                if option <= record["maximum"] + 1e-9
            ]
            if not record["options"]:
                record["options"] = [record["maximum"]]

    long_records = [
        record for record in records if record["role"] == "long"
    ]
    primary_long = long_records[0] if long_records else None

    # This allocator now receives the entire rolling lookahead, which can
    # legitimately contain more than one long run.  The original seven-day
    # implementation protected only the first one; later long runs were then
    # treated like generic mileage and could regress even when the horizon had
    # ample room for every selected long-run prescription.  Preserve all of
    # their coaching-derived preferred distances when those preferences fit
    # alongside the minimum useful dose of the other selected sessions.
    preferred_longs_fit = bool(
        long_records
        and completed_miles
        + fixed_midpoint
        + sum(
            record["preferred"]
            if record["role"] == "long"
            else record["minimum"]
            for record in records
        )
        <= target_range[1] + 1e-9
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
                lower = max(
                    0.1,
                    ordinary_easy_midpoint + long_margin,
                    center - 0.25,
                )
            else:
                lower = max(0.1, center - 0.25)
            # Preserve the chosen midpoint even when an athlete-relative
            # lower bound narrows the execution range.
            upper = max(lower, center * 2 - lower)
            ranges[record["index"]] = (lower, upper)
        return ranges

    stable_easy_reference = projected_recovery_reference_miles(daily_states[0])
    records_by_index = {record["index"]: record for record in records}
    base_load_units_by_index = {
        record["index"]: _recommendation_load_units(
            updated[record["index"]].recommendation,
            stable_easy_reference,
        )
        for record in records
        if updated[record["index"]].recommendation is not None
    }
    allocated_load_units_cache: dict[
        tuple[int, tuple[float, float], WorkoutType], float
    ] = {}
    planned_at_by_index = {
        record["index"]: (
            updated[record["index"]].recommendation.planned_for
            or daily_states[record["index"]].as_of
        )
        for record in records
        if updated[record["index"]].recommendation is not None
    }
    planned_impulse_hours = _planned_impulse_hours(updated)
    recorded_recovery_load_by_index = {
        index: _decayed_recovery_load(
            daily_states[index],
            [],
            planned_at,
            stable_easy_reference,
        )
        for index, planned_at in planned_at_by_index.items()
    }
    planned_decay_by_pair = {
        (index, prior_index): decay_recovery_load(
            1.0,
            (planned_at - prior_at).total_seconds() / 3600,
        )
        for index, planned_at in planned_at_by_index.items()
        for prior_index, prior_at in planned_at_by_index.items()
        if prior_at < planned_at
    }

    def allocated_load_units(
        index: int,
        distance_range: tuple[float, float],
        workout_type: WorkoutType,
    ) -> float:
        key = (index, distance_range, workout_type)
        if key not in allocated_load_units_cache:
            result = updated[index].recommendation
            assert result is not None
            if (
                result.distance_range_miles != distance_range
                or result.workout_type != workout_type
            ):
                result = result.model_copy(
                    update={
                        "distance_range_miles": distance_range,
                        "workout_type": workout_type,
                    }
                )
            allocated_load_units_cache[key] = _recommendation_load_units(
                result,
                stable_easy_reference,
            )
        return allocated_load_units_cache[key]

    def apply_recovery_caps(
        candidate_ranges: dict[int, tuple[float, float]],
        retain_long: bool,
    ) -> tuple[dict[int, tuple[float, float]], set[int]]:
        """Apply sequential recovery limits before an allocation is scored."""

        capped_ranges = dict(candidate_ranges)
        capped_indices: set[int] = set()
        projected_indices: list[int] = []
        projected_types: list[WorkoutType] = []
        projected_times: list[datetime | None] = []
        projected_load_units: list[float] = []
        for record in sorted(records, key=lambda item: item["index"]):
            index = record["index"]
            result = updated[index].recommendation
            assert result is not None and result.distance_range_miles is not None
            allocated = capped_ranges[index]
            demoted_long = record is primary_long and not retain_long
            provisional_type = (
                WorkoutType.EASY if demoted_long else result.workout_type
            )
            if projected_indices:
                base_units = base_load_units_by_index[index]
                allocated_units = allocated_load_units(
                    index,
                    allocated,
                    provisional_type,
                )
                recovery_contributions = [
                    load_units
                    * planned_decay_by_pair.get(
                        (index, prior_index),
                        0.0,
                    )
                    for prior_index, load_units in zip(
                        projected_indices,
                        projected_load_units,
                    )
                ]

                def recovery_allowance() -> tuple[float, float]:
                    residual_load = (
                        recorded_recovery_load_by_index[index]
                        + sum(recovery_contributions)
                    )
                    base_overflow = max(
                        0.0, residual_load + base_units - 2.0
                    )
                    return residual_load, max(
                        0.0, 2.0 + base_overflow - residual_load
                    )

                residual, allowed_units = recovery_allowance()
                if (
                    provisional_type == WorkoutType.LONG
                    and allocated_units > allowed_units + 0.05
                ):
                    # Optional easy mileage immediately before a maintained
                    # long run must not silently make that long run regress.
                    # Trim those aerobic ranges in reverse order and recheck
                    # the same recovery equation before capping the key run.
                    for prior_position in range(
                        len(projected_indices) - 1,
                        -1,
                        -1,
                    ):
                        if projected_types[prior_position] != WorkoutType.EASY:
                            continue
                        prior_index = projected_indices[prior_position]
                        prior_center = records_by_index[prior_index]["minimum"]
                        prior_minimum = max(0.1, prior_center - 0.25)
                        prior_maximum = prior_center + 0.25
                        prior_range = capped_ranges[prior_index]
                        if prior_range[1] <= prior_maximum + 1e-9:
                            continue
                        shortened = (prior_minimum, prior_maximum)
                        capped_ranges[prior_index] = shortened
                        projected_load_units[prior_position] = (
                            allocated_load_units(
                                prior_index,
                                shortened,
                                WorkoutType.EASY,
                            )
                        )
                        recovery_contributions[prior_position] = (
                            projected_load_units[prior_position]
                            * planned_decay_by_pair.get(
                                (index, prior_index),
                                0.0,
                            )
                        )
                        capped_indices.add(prior_index)
                        residual, allowed_units = recovery_allowance()
                        if allocated_units <= allowed_units + 0.05:
                            break
                if allocated_units > allowed_units + 0.05:
                    estimated_minutes = max(
                        _structured_duration_minutes(result) or 0.0,
                        _midpoint(result) * 11.0,
                    )
                    intensity_factor = _prescribed_intensity_factor(
                        result,
                        estimated_minutes,
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
                    capped_upper = max(
                        record["minimum"] + 0.25,
                        capped_upper,
                    )
                    capped_lower = max(
                        max(0.1, record["minimum"] - 0.25),
                        min(allocated[0], capped_upper - 0.5),
                    )
                    allocated = (capped_lower, capped_upper)
                    capped_ranges[index] = allocated
                    allocated_units = allocated_load_units(
                        index,
                        allocated,
                        provisional_type,
                    )
                    capped_indices.add(index)
            projected_indices.append(index)
            projected_types.append(provisional_type)
            projected_times.append(result.planned_for)
            projected_load_units.append(
                allocated_load_units(
                    index,
                    allocated,
                    provisional_type,
                )
            )
        return capped_ranges, capped_indices

    allocation_candidate_limit = max(1, allocation_candidate_limit)

    def solve(
        retain_long: bool,
    ) -> list[
        tuple[
            float,
            dict[int, float],
            dict[int, tuple[float, float]],
            set[int],
        ]
    ]:
        long_options: list[float | None]
        if retain_long:
            if primary_long is None:
                return []
            minimum_meaningful = ordinary_easy_midpoint + long_margin
            long_options = [
                option
                for option in primary_long["options"]
                if option + 1e-9
                >= minimum_meaningful
            ]
            if preferred_longs_fit:
                long_options = [
                    option
                    for option in long_options
                    if option + 1e-9 >= primary_long["preferred"]
                ]
            if not long_options:
                return []
        else:
            long_options = [None]

        best: list[
            tuple[
                float,
                dict[int, float],
                dict[int, tuple[float, float]],
                set[int],
            ]
        ] = []
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
            if primary_long is not None and retain_long:
                # Workout selection has already derived a proportional,
                # recovery-aware long-run progression.  Equal-share mileage
                # arithmetic used to place the long's ideal below its current
                # maintained distance, so every successful week grew easy
                # runs while the primary durability lane stayed flat. Treat
                # the selected long recommendation as the soft allocation
                # ideal. Recovery caps and total-load penalties may still
                # scale it down; balance alone no longer erases progression.
                for long_record in long_records:
                    ideals[long_record["index"]] = max(
                        ideals[long_record["index"]],
                        long_record["preferred"],
                    )
            initial_assignment_items = (
                ((primary_long["index"], long_option),)
                if long_option is not None and primary_long is not None
                else ()
            )
            initial_units = _distance_dp_units(long_option or 0)
            initial_cost = (
                (long_option - ideals[primary_long["index"]]) ** 2
                if long_option is not None and primary_long is not None
                else 0.0
            )
            initial_sort_key = (
                initial_cost,
                initial_assignment_items,
            )
            dp: dict[
                int,
                list[
                    tuple[
                        tuple[float, tuple[tuple[int, float], ...]],
                        float,
                        tuple[tuple[int, float], ...],
                    ]
                ],
            ] = {
                initial_units: [
                    (
                        initial_sort_key,
                        initial_cost,
                        initial_assignment_items,
                    )
                ]
            }
            if primary_long is not None and not retain_long:
                scenario_records = records
            # Weekly arithmetic does not turn a selected quality session into
            # generic filler. Recovery can still scale its structure below;
            # otherwise frequency/date selection must make room for the real
            # workout dose.
            preserve_quality_dose = True
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
                    meaningful_easy_ceiling = min(
                        long_option - long_margin,
                        long_option / 1.15,
                    )
                    options = [
                        option
                        for option in options
                        if option <= meaningful_easy_ceiling + 1e-9
                    ]
                if preserve_quality_dose and record["role"] == "quality":
                    options = [
                        option
                        for option in options
                        if option + 1e-9 >= record["preferred"]
                    ]
                if (
                    retain_long
                    and preferred_longs_fit
                    and record["role"] == "long"
                ):
                    options = [
                        option
                        for option in options
                        if option + 1e-9 >= record["preferred"]
                    ]
                if not options:
                    dp = {}
                    break
                next_dp: dict[
                    int,
                    list[
                        tuple[
                            tuple[float, tuple[tuple[int, float], ...]],
                            float,
                            tuple[tuple[int, float], ...],
                        ]
                    ],
                ] = {}
                for units, candidates in dp.items():
                    for _, cost, assignment_items in candidates:
                        for option in options:
                            next_units = units + _distance_dp_units(option)
                            option_cost = (
                                option - ideals[record["index"]]
                            ) ** 2
                            prior_midpoint = record["prior_midpoint"]
                            if isinstance(prior_midpoint, (int, float)):
                                movement = abs(option - prior_midpoint)
                                excess = max(
                                    0.0,
                                    movement
                                    - NEAR_TERM_DISTANCE_STABILITY_TOLERANCE_MILES,
                                )
                                if excess > 0:
                                    proximity = max(
                                        0.0,
                                        (
                                            NEAR_TERM_STABILITY_DECISION_OFFSETS
                                            - record["index"]
                                        )
                                        / NEAR_TERM_STABILITY_DECISION_OFFSETS,
                                    )
                                    normalized = min(
                                        1.0,
                                        excess
                                        / NEAR_TERM_DISTANCE_STABILITY_TOLERANCE_MILES,
                                    )
                                    option_cost += (
                                        proximity
                                        * normalized
                                        * normalized
                                        * NEAR_TERM_STABILITY_ABSOLUTE_CAP
                                    )
                            next_cost = cost + option_cost
                            candidate_assignment_items = (
                                *assignment_items,
                                (record["index"], option),
                            )
                            candidate_sort_key = (
                                next_cost,
                                candidate_assignment_items,
                            )
                            candidate = (
                                candidate_sort_key,
                                next_cost,
                                candidate_assignment_items,
                            )
                            bucket = next_dp.setdefault(next_units, [])
                            # The previous implementation accumulated every
                            # assignment for this total, sorted the complete
                            # list, and then retained this same prefix. Keep
                            # that exact ordered prefix incrementally instead;
                            # later DP stages cannot observe discarded items.
                            if (
                                bucket
                                and len(bucket) >= assignments_per_total
                                and candidate_sort_key >= bucket[-1][0]
                            ):
                                continue
                            insertion_index = len(bucket)
                            for index, existing in enumerate(bucket):
                                if candidate_sort_key < existing[0]:
                                    insertion_index = index
                                    break
                            bucket.insert(insertion_index, candidate)
                            if len(bucket) > assignments_per_total:
                                bucket.pop()
                # Equal total mileage can be distributed in materially
                # different ways across a rolling horizon. Keep several
                # shapes alive so later density scoring can prefer a longer
                # well-recovered easy run and a shorter pre-taxing run instead
                # of being forced into the single most even assignment.
                dp = next_dp
            # Evaluate the strongest exact lower bounds first. Every term
            # added after ``balance_cost`` is non-negative, so once a good
            # incumbent exists the existing dominated-allocation check can
            # safely discard most retained DP shapes before running the
            # expensive sequential recovery projection. Iterating dictionary
            # total buckets previously delayed that incumbent and performed
            # hundreds of equivalent cap/path evaluations per calendar.
            final_candidates = sorted(
                (
                    candidate
                    for candidates in dp.values()
                    for candidate in candidates
                ),
                key=lambda candidate: candidate[0],
            )
            for _, balance_cost, assignment_items in final_candidates:
                    assignment = dict(assignment_items)
                    optimistic_score = balance_cost - (
                        0.25 if retain_long else 0.0
                    )
                    if (
                        _prune_dominated_allocations
                        and len(best) >= allocation_candidate_limit
                        and optimistic_score >= best[-1][0]
                    ):
                        # Every remaining objective term is non-negative. If
                        # balance alone cannot beat the incumbent, sequential
                        # recovery projection and path scoring cannot rescue
                        # this assignment. Equality is also safe to skip: the
                        # solver only replaces ``best`` on a strict decrease,
                        # preserving the existing deterministic tie winner.
                        continue
                    allocated_ranges = ranges_for(assignment, retain_long)
                    allocated_ranges, capped_indices = apply_recovery_caps(
                        allocated_ranges, retain_long
                    )
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
                    role_shape_score = 0.0
                    if (
                        retain_long
                        and primary_long is not None
                        and long_option is not None
                    ):
                        # ``preferred`` is the ordinary proportional build;
                        # ``maximum`` is extra recovery-supported headroom.
                        # Treat spending that headroom as a graded program
                        # choice rather than free mileage. Normalizing by the
                        # actual preferred-to-maximum band keeps the tradeoff
                        # athlete-relative and makes the full supported maximum
                        # comparable to one other whole program-fit objective.
                        # The quarter mile fallback is the allocator's own
                        # resolution, not a separate coaching threshold.
                        progression_band = max(
                            0.25,
                            primary_long["maximum"]
                            - primary_long["preferred"],
                        )
                        surplus_fraction = max(
                            0.0,
                            (long_option - primary_long["preferred"])
                            / progression_band,
                        )
                        role_shape_score += (
                            surplus_fraction
                            * surplus_fraction
                            * PROGRAM_FIT_UNIT
                        )
                    # Expansion above the athlete's ordinary easy band stays
                    # available, but it must beat the recovery cost of adding
                    # a useful aerobic day. Giving the largest expansion in
                    # every long-run segment zero cost made three-session
                    # medium-long/long/quality loops an optimizer attractor.
                    easy_expansions: dict[int, float] = {}
                    for record in records:
                        if record["role"] != "easy":
                            continue
                        index = record["index"]
                        easy_center = sum(allocated_ranges[index]) / 2
                        expansion_reference = (
                            record["expansion_reference"]
                        )
                        expansion_band = max(
                            0.5,
                            expansion_reference
                            - record["easy_midpoint"],
                        )
                        easy_expansions[index] = max(
                            0.0,
                            (easy_center - expansion_reference)
                            / expansion_band,
                        )
                    role_shape_score += sum(
                        expansion * expansion
                        for expansion in easy_expansions.values()
                    ) * PROGRAM_FIT_UNIT * SECONDARY_ENDURANCE_SHAPE_FRACTION
                    if weekly_target_range is not None:
                        # Use the same boundary-free cumulative path as date
                        # selection. Day 7 and day 14 are display coordinates,
                        # while one ordinary session of timing variation stays
                        # available anywhere in the receding horizon.
                        opening_distance_rate = (
                            daily_states[0].recent_load.continuous_distance_miles
                        )
                        daily_path_miles: list[float] = []
                        for index, day in enumerate(updated):
                            result = day.recommendation
                            planned_midpoint = 0.0
                            if (
                                result
                                and result.workout_type != WorkoutType.REST
                                and result.distance_range_miles
                            ):
                                lower, upper = allocated_ranges.get(
                                    index, result.distance_range_miles
                                )
                                planned_midpoint = (lower + upper) / 2
                            daily_path_miles.append(
                                (
                                    0.0
                                    if opening_distance_rate is not None
                                    else sum(
                                        activity.distance_miles
                                        for activity in day.completed_activities
                                    )
                                )
                                + planned_midpoint
                            )
                        fatigue_half_life_days = float(
                            config.get("coaching", {}).get(
                                "continuous_fatigue_half_life_days", 7
                            )
                        )
                        path_tolerance = (
                            ordinary_easy_midpoint
                            * log(2.0)
                            * 7.0
                            / max(0.1, fatigue_half_life_days)
                            if opening_distance_rate is not None
                            else ordinary_easy_midpoint
                        )
                        # This allocator feeds the joint candidate objective,
                        # whose continuous path is only the opening-load
                        # transition. Scoring all 21 days here made distance
                        # allocation optimize a different, hard-edge-sensitive
                        # objective than the one used to compare calendars.
                        # In particular, it could manufacture a second
                        # medium-long expansion which the caller immediately
                        # penalized as poor role shape. Sustained funding is
                        # handled by the hard floor and soft recurring-rate
                        # projection after allocation.
                        transition_days = min(
                            len(daily_path_miles),
                            max(1, ceil(fatigue_half_life_days)),
                        )
                        transition_miles = daily_path_miles[:transition_days]
                        transition_impulses = planned_impulse_hours[
                            :transition_days
                        ]
                        path_violation = _continuous_mileage_path_violation(
                            transition_miles,
                            weekly_target_range,
                            session_tolerance_miles=path_tolerance,
                            opening_weekly_rate=opening_distance_rate,
                            half_life_days=fatigue_half_life_days,
                            impulse_hours=transition_impulses,
                            opening_hour=_clock_hour(daily_states[0].as_of),
                        )
                        density_score += path_violation**2 * 20.0
                        if opening_distance_rate is not None:
                            density_score += _continuous_mileage_path_cost(
                                transition_miles,
                                weekly_target_range,
                                session_tolerance_miles=path_tolerance,
                                opening_weekly_rate=opening_distance_rate,
                                half_life_days=fatigue_half_life_days,
                                impulse_hours=transition_impulses,
                                opening_hour=_clock_hour(daily_states[0].as_of),
                            ) * 8.0
                    score = balance_cost + density_score + role_shape_score - (
                        0.25 if retain_long else 0.0
                    )
                    if weekly_target_range is None:
                        # A literal seven-day plan has a real end and can be
                        # aligned directly to its total range. A longer
                        # lookahead is receding: day 21 is replaced tomorrow,
                        # so forcing all scaled mileage to be consumed before
                        # that boundary creates terminal-pressure artifacts.
                        score += (
                            (projected_mid - target_midpoint) ** 2
                            + total_shortfall**2 * 30.0
                            + total_excess**2 * 40.0
                        )
                    else:
                        # The horizon edge is not a deadline, but its average
                        # prescribed rate still has to fund the rolling target.
                        # This term changes only distance allocation on already
                        # selected dates; it cannot push a catch-up run onto day
                        # 21. The cumulative path above controls where that
                        # mileage is supportable without calendar boundaries.
                        horizon_scale = len(updated) / VISIBLE_HORIZON_DAYS
                        projected_weekly_rate = (
                            projected_mid / max(1.0, horizon_scale)
                        )
                        # The range describes execution flexibility around one
                        # intended load; it is not a flat objective where its
                        # center and upper edge are equally good. Aim distance
                        # allocation at the midpoint, while leaving run-day
                        # selection governed by recovery and useful session
                        # size rather than manufacturing an extra workout.
                        if opening_distance_rate is None:
                            score += _weekly_rate_alignment_cost(
                                projected_weekly_rate,
                                weekly_target_range,
                            )
                    candidate_solution = (
                        score,
                        assignment,
                        allocated_ranges,
                        capped_indices,
                    )
                    candidate_key = (
                        score,
                        tuple(sorted(assignment.items())),
                    )
                    insertion_index = len(best)
                    for solution_index, existing in enumerate(best):
                        existing_key = (
                            existing[0],
                            tuple(sorted(existing[1].items())),
                        )
                        if candidate_key < existing_key:
                            insertion_index = solution_index
                            break
                    best.insert(insertion_index, candidate_solution)
                    if len(best) > allocation_candidate_limit:
                        best.pop()
        return best

    long_solutions = solve(True)
    # Workout selection has already evaluated recovery, load, recency, and
    # whether the proposed long is meaningfully distinct. If that safe long
    # can fit, reserve it before allocating quality and easy mileage. The
    # distance allocator distributes the week; it must not silently overrule
    # the coaching decision merely because equal-sized aerobic runs produce a
    # tidier numerical balance. Only solve the aerobic fallback when the long
    # solution is unavailable; computing both and discarding the aerobic
    # result was pure duplicate search work.
    retain_long = bool(long_solutions)
    solutions = long_solutions if retain_long else solve(False)
    if not solutions:
        return updated

    def materialize_solution(
        selected: tuple[
            float,
            dict[int, float],
            dict[int, tuple[float, float]],
            set[int],
        ],
    ) -> list[WeeklyScheduleDay]:
        materialized = list(updated)
        allocated_ranges = selected[2]
        recovery_capped_indices = selected[3]

        for record in records:
            index = record["index"]
            day = materialized[index]
            result = day.recommendation
            assert result is not None and result.distance_range_miles is not None
            raw_lower, raw_upper = allocated_ranges[index]
            allocated = (round(raw_lower, 1), round(raw_upper, 1))
            if (
                raw_upper - raw_lower >= 0.5 - 1e-9
                and allocated[1] - allocated[0] < 0.5 - 1e-9
            ):
                # Decimal presentation must not turn a genuine half-mile
                # range such as 6.75-7.25 into the narrower-looking 6.8-7.2.
                allocated = (
                    round(
                        max(
                            0.1,
                            allocated[1] - 0.5,
                        ),
                        1,
                    ),
                    allocated[1],
                )
            demoted_long = record is primary_long and not retain_long
            medium_long = bool(
                record["role"] == "easy"
                and sum(allocated) / 2
                > _ordinary_easy_expansion_reference(daily_states[index])
                + 1e-9
            )
            support_easy = bool(
                record["role"] == "easy"
                and not medium_long
                and sum(allocated) / 2 < record["preferred"] - 1e-9
            )
            revised = result.model_copy(
                update={
                    "planning_role": (
                        "medium_long"
                        if medium_long
                        else "support_easy"
                        if support_easy
                        else "ordinary_easy"
                        if record["role"] == "easy" or demoted_long
                        else "long"
                        if record["role"] == "long"
                        else "quality"
                        if record["role"] == "quality"
                        else None
                    ),
                    "workout_type": (
                        WorkoutType.EASY
                        if demoted_long
                        else result.workout_type
                    ),
                    "title": (
                        "Easy aerobic run"
                        if demoted_long
                        else result.title
                        if not medium_long
                        else "Medium-long aerobic run"
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
                            "This is an intentional secondary endurance session: longer than an ordinary easy run, but subordinate to the primary long run.",
                            "Distance was jointly allocated across the full week under recovery, progression, and workout-role constraints.",
                        ]
                        if medium_long
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
            recent_distance = (
                daily_states[index].recent_load.trailing_28d.distance_miles
            )
            recent_minutes = (
                daily_states[index].recent_load.trailing_28d.moving_minutes
            )
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
            materialized[index] = day.model_copy(
                update={
                    "recommendation": revised,
                    "day_role": (
                        "easy_run"
                        if demoted_long
                        else day.day_role
                        if not medium_long
                        else "medium_long_run"
                    ),
                    "rationale": (
                        f"{day.rationale} The long-run label was removed because a distinct long run would worsen weekly load balance."
                        if demoted_long
                        else f"{day.rationale} Mileage was jointly allocated across the full week."
                    ),
                }
            )
        return materialized

    materialized_solutions = [
        materialize_solution(solution) for solution in solutions
    ]
    if allocation_selector is None:
        return materialized_solutions[0]
    return min(
        materialized_solutions,
        key=lambda candidate: (
            allocation_selector(candidate),
            tuple(
                (
                    index,
                    _midpoint(day.recommendation),
                )
                for index, day in enumerate(candidate)
                if day.recommendation is not None
                and day.recommendation.workout_type != WorkoutType.REST
            ),
        ),
    )


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
        return ""
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
        f"Today's run window remains feasible. The full 21-day recovery and "
        f"projected-load comparison instead selects {day_name}; this is an "
        "optimizer tradeoff, not a hard recovery prohibition."
    )


def _build_baseline_schedule(
    daily_states: list[FitnessState],
    daily_state_options: list[list[FitnessState]],
    request: RecommendationRequest,
    config: dict,
    target_evidence: WeeklyTargetEvidence,
    completed_activities_by_offset: dict[int, list[TrailingDayActivity]],
    forced_rest_offsets: set[int],
) -> WeeklyScheduleResponse:
    """Coach baseline acquisition without inventing weekly mileage.

    With no history, the app prescribes one time-based calibration run rather
    than publishing a starter mileage number. Once one or more outings exist
    but weekly capacity still cannot support even the observed median session,
    it schedules only one easy repeat. Uploading each run closes the loop and
    updates the evidence before another session is prescribed.
    """

    mode = target_evidence.planning_mode
    baseline_miles = target_evidence.baseline_session_miles
    baseline_minutes = target_evidence.baseline_session_minutes
    selected: tuple[int, FitnessState, RecommendationResponse] | None = None
    if (
        (
            mode == WeeklyPlanningMode.BASELINE_REQUIRED
            or (
                mode == WeeklyPlanningMode.BASELINE_BUILDING
                and baseline_miles is not None
            )
        )
        and request.health_status
        not in {
            CurrentHealthStatus.SICK_OR_RECOVERING,
            CurrentHealthStatus.PAIN_OR_INJURY_CONCERN,
        }
    ):
        for offset in range(VISIBLE_HORIZON_DAYS):
            if (
                offset in forced_rest_offsets
                or completed_activities_by_offset.get(offset)
            ):
                continue
            state, result = _select_timed_recommendation(
                daily_state_options[offset],
                [],
                request,
                config,
                weekly_role="easy",
                allowed_candidates={"easy"},
            )
            if result.workout_type != WorkoutType.EASY:
                continue
            if mode == WeeklyPlanningMode.BASELINE_REQUIRED:
                result = result.model_copy(
                    update={
                        "title": "Conversational baseline run",
                        "distance_range_miles": None,
                        "duration_range_minutes": (BASELINE_MINIMUM_AEROBIC_MINUTES, 30.0),
                        "target_zones": ["Z2"],
                        "structure": [
                            WorkoutStep(
                                instruction=(
                                    "Run or run/walk at conversational effort in Zone 2 "
                                    "for at least 10 minutes."
                                ),
                                target_zones=["Z2"],
                            ),
                            WorkoutStep(
                                instruction=(
                                    "Stop for pain, fatigue, and/or elevated heart rate "
                                    "and end the run at a maximum of 30 minutes."
                                ),
                                target_zones=["Z2"],
                            ),
                        ],
                        "reasons": [
                            *result.reasons,
                            (
                                "Ten minutes provides a meaningful first aerobic sample. "
                                "Continuing while effort, heart rate, and your legs remain "
                                "comfortable provides more evidence without inventing a "
                                "distance target."
                            ),
                        ],
                        "modification_rules": [
                            *result.modification_rules,
                            (
                                "Stop for pain, fatigue, and/or elevated heart rate and "
                                "end the run at a maximum of 30 minutes."
                            ),
                        ],
                    }
                )
            else:
                assert baseline_miles is not None
                exact_minutes = (
                    max(10.0, round(baseline_minutes))
                    if baseline_minutes is not None
                    else None
                )
                exact_miles = max(0.1, _tenth_mile(baseline_miles))
                result = result.model_copy(
                    update={
                        "title": "Easy baseline run",
                        "distance_range_miles": (
                            None
                            if exact_minutes is not None
                            else (exact_miles, exact_miles)
                        ),
                        "duration_range_minutes": (
                            (exact_minutes, exact_minutes)
                            if exact_minutes is not None
                            else None
                        ),
                        "structure": [
                            WorkoutStep(
                                instruction="Settle into conversational Z1/Z2 effort without chasing pace.",
                                target_zones=["Z1", "Z2"],
                            ),
                            WorkoutStep(
                                instruction=(
                                    f"Finish at {exact_minutes:.0f} minutes, or stop earlier for pain, fatigue, or elevated heart rate."
                                    if exact_minutes is not None
                                    else f"Finish at {exact_miles:.1f} miles, or stop earlier for pain, fatigue, or elevated heart rate."
                                ),
                                target_zones=["Z1", "Z2"],
                            ),
                        ],
                        "reasons": [
                            *result.reasons,
                            (
                                f"The {exact_minutes:.0f}-minute duration repeats the median of your observed aerobic sessions."
                                if exact_minutes is not None
                                else f"The {exact_miles:.1f}-mile distance repeats the median of legacy runs without usable duration."
                            ),
                        ],
                        "modification_rules": [
                            *result.modification_rules,
                            "Keep the effort conversational and stop for pain, fatigue, or elevated heart rate.",
                        ],
                    }
                )
            selected = (offset, state, result)
            break

    selected_offset = selected[0] if selected else None
    days: list[WeeklyScheduleDay] = []
    for offset in range(VISIBLE_HORIZON_DAYS):
        raw_state = daily_states[offset]
        completed = completed_activities_by_offset.get(offset, [])
        if completed:
            days.append(
                WeeklyScheduleDay(
                    date=raw_state.as_of.date(),
                    planned_at=None,
                    recommendation=None,
                    day_role="completed_run",
                    rationale="Your uploaded run is now part of the baseline evidence.",
                    completed_activities=completed,
                )
            )
        elif offset in forced_rest_offsets:
            days.append(
                WeeklyScheduleDay(
                    date=raw_state.as_of.date(),
                    planned_at=None,
                    recommendation=None,
                    day_role="forced_rest_day",
                    rationale="You marked this date as unavailable for running.",
                    forced_rest=True,
                )
            )
        elif selected_offset == offset and selected is not None:
            _, state, result = selected
            days.append(
                WeeklyScheduleDay(
                    date=raw_state.as_of.date(),
                    planned_at=state.as_of,
                    recommendation=result,
                    day_role="baseline_run",
                    rationale=(
                        "This is the first time-based calibration session; it measures "
                        "comfortable aerobic exposure without guessing a distance."
                        if mode == WeeklyPlanningMode.BASELINE_REQUIRED
                        else "This is the earliest recovery-compatible opportunity to repeat "
                        "an observed aerobic duration and collect another baseline session."
                    ),
                )
            )
        else:
            days.append(
                WeeklyScheduleDay(
                    date=raw_state.as_of.date(),
                    planned_at=None,
                    recommendation=None,
                    day_role="baseline_collection_day",
                    rationale=(
                        "Only one calibration session is prescribed before the plan reads its result."
                        if mode == WeeklyPlanningMode.BASELINE_REQUIRED
                        else "Only one baseline session is prescribed before the plan reads the new result."
                    ),
                )
            )

    completed_miles = sum(
        activity.distance_miles
        for day in days
        for activity in day.completed_activities
    )
    # The first time range is deliberately not converted into estimated miles:
    # pace is exactly the missing measurement this session is meant to collect.
    planned_miles = 0.0
    total = round(completed_miles + planned_miles, 1)
    generated_at = datetime.now(daily_states[0].as_of.tzinfo)
    summary = (
        "Complete this 10–30-minute conversational run or run/walk, then upload it. The app will use the observed duration, distance, effort, and heart rate instead of assuming a beginner mileage target."
        if mode == WeeklyPlanningMode.BASELINE_REQUIRED and selected is not None
        else "Complete this one easy baseline run and upload it. The next plan will update from the observed distance, effort, and recovery instead of assuming a weekly routine."
        if selected is not None
        else "Baseline progression is paused by current health, recovery, or availability constraints."
    )
    completed_run_count = sum(bool(day.completed_activities) for day in days)
    return WeeklyScheduleResponse(
        planner_version=WEEKLY_PLANNER_VERSION,
        generated_at=generated_at,
        emergency_alerts_checked_at=generated_at,
        start_date=daily_states[0].as_of.date(),
        end_date=daily_states[VISIBLE_HORIZON_DAYS - 1].as_of.date(),
        target_run_count=completed_run_count + (1 if selected else 0),
        target_distance_range_miles=(
            (planned_miles, planned_miles)
            if selected is not None
            else (0.0, 0.0)
        ),
        target_evidence=target_evidence,
        completed_run_count=completed_run_count,
        run_count=1 if selected else 0,
        projected_distance_range_miles=(total, total),
        summary=summary,
        days=days,
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
    prior_schedule: WeeklyScheduleResponse | None = None,
    expected_target_projector: ExpectedTargetProjector | None = None,
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
    prior_plan_days = (
        prior_schedule.planning_days
        if prior_schedule is not None and prior_schedule.planning_days
        else prior_schedule.days
        if prior_schedule is not None
        else []
    )
    prior_run_offsets = {
        (day.date - daily_states[0].as_of.date()).days
        for day in prior_plan_days
        if day.recommendation is not None
        and day.recommendation.workout_type != WorkoutType.REST
        and 0
        <= (day.date - daily_states[0].as_of.date()).days
        < planning_horizon_days
    }
    prior_run_offsets -= forced_rest_offsets
    prior_run_offsets -= completed_run_offsets
    prior_runs_by_date = {
        day.date: day.recommendation
        for day in prior_plan_days
        if day.recommendation is not None
        and day.recommendation.workout_type != WorkoutType.REST
        and day.recommendation.distance_range_miles is not None
    }
    prior_distance_midpoints_by_date = {
        plan_date: _midpoint(result)
        for plan_date, result in prior_runs_by_date.items()
    }
    prior_roles_by_offset = {
        offset: (
            "long"
            if result.workout_type == WorkoutType.LONG
            else "quality"
            if result.workout_type in QUALITY_WORKOUT_TYPES
            else "easy"
        )
        for plan_date, result in prior_runs_by_date.items()
        if (
            offset := (plan_date - daily_states[0].as_of.date()).days
        ) in prior_run_offsets
    }
    if (
        target_evidence is not None
        and target_evidence.planning_mode
        in {
            WeeklyPlanningMode.BASELINE_REQUIRED,
            WeeklyPlanningMode.BASELINE_BUILDING,
        }
    ):
        return _build_baseline_schedule(
            daily_states,
            daily_state_options,
            request,
            config,
            target_evidence,
            completed_activities_by_offset,
            forced_rest_offsets,
        )
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
    if goal:
        _, race_date, _ = goal
        race_offset = (race_date - daily_states[0].as_of.date()).days
    planner_search_diagnostics = _PlannerSearchDiagnostics()
    desired_offsets = (
        adaptive_run_day_offsets(
            daily_states,
            request,
            config,
            int(target_run_count or 0),
            target_distance_range,
            daily_state_options=daily_state_options,
            forced_rest_offsets=forced_rest_offsets,
            completed_run_offsets=completed_run_offsets,
            completed_miles_by_offset=completed_miles_by_offset,
            prior_run_offsets=prior_run_offsets,
            prior_distance_midpoints_by_date=(
                prior_distance_midpoints_by_date
            ),
            prior_roles_by_offset=prior_roles_by_offset,
            expected_target_projector=expected_target_projector,
            _planner_diagnostics=planner_search_diagnostics,
        )
        if planning_horizon_days > VISIBLE_HORIZON_DAYS
        else (
            []
            if target_run_count is not None
            and target_run_count - completed_run_count <= 0
            else automatic_run_day_offsets(
                offset_state,
                request.health_status,
                config,
                (
                    max(0, target_run_count - completed_run_count)
                    if target_run_count is not None
                    else None
                ),
                horizon_days=planning_horizon_days,
            )
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
    next_run_offset = {
        offset: offsets[index + 1]
        for index, offset in enumerate(offsets[:-1])
    }
    planned: list[RecommendationResponse] = []
    days: list[WeeklyScheduleDay] = []
    low_cost_consecutive, consecutive_facts = consecutive_day_evidence(
        daily_states[0], config
    )
    for offset, raw_state in enumerate(daily_states):
        completed = completed_activities_by_offset.get(offset, [])
        if completed:
            workout_types = {item.workout_type for item in completed}
            role = (
                "completed_quality_run"
                if workout_types & QUALITY_WORKOUT_TYPES
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
        retained_role = planner_search_diagnostics.winning_role_overrides.get(
            offset
        )
        state, result = _select_budgeted_timed_recommendation(
            daily_state_options[offset],
            planned,
            request,
            config,
            weekly_role=(
                retained_role
                or _elapsed_workout_role(
                    daily_state_options[offset],
                    planned,
                    config,
                    next_state_options=(
                        daily_state_options[next_run_offset[offset]]
                        if offset in next_run_offset
                        else None
                    ),
                )
            ),
            preserve_selected_role=retained_role is not None,
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
                    "Selected from the continuous rolling 21-day plan using projected recovery and accumulated load. "
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
    def final_allocation_selector(
        source_days: list[WeeklyScheduleDay],
    ) -> Callable[[list[WeeklyScheduleDay]], float]:
        source_sessions: list[_CandidateSession] = []
        for index, day in enumerate(source_days):
            result = day.recommendation
            if result is None or result.workout_type == WorkoutType.REST:
                continue
            state = daily_states[index]
            if day.planned_at is not None and state.as_of != day.planned_at:
                state = state.model_copy(update={"as_of": day.planned_at})
            source_sessions.append(
                _CandidateSession(
                    offset=index,
                    state=state,
                    recommendation=result,
                )
            )

        ordinary_easy_midpoint = sum(
            typical_easy_distance(daily_states[0])
        ) / 2

        def distance_stability_cost(
            candidate: list[WeeklyScheduleDay],
        ) -> float:
            """Price material near-term dose rewrites without locking them."""

            cost = 0.0
            for index, day in enumerate(
                candidate[:NEAR_TERM_STABILITY_DECISION_OFFSETS]
            ):
                result = day.recommendation
                prior = prior_runs_by_date.get(day.date)
                if (
                    result is None
                    or prior is None
                    or result.workout_type != prior.workout_type
                    or result.distance_range_miles is None
                    or prior.distance_range_miles is None
                ):
                    continue
                movement = abs(_midpoint(result) - _midpoint(prior))
                excess = max(
                    0.0,
                    movement
                    - NEAR_TERM_DISTANCE_STABILITY_TOLERANCE_MILES,
                )
                if excess <= 0:
                    continue
                proximity = (
                    NEAR_TERM_STABILITY_DECISION_OFFSETS - index
                ) / NEAR_TERM_STABILITY_DECISION_OFFSETS
                normalized = min(
                    1.0,
                    excess
                    / NEAR_TERM_DISTANCE_STABILITY_TOLERANCE_MILES,
                )
                cost += (
                    proximity
                    * normalized
                    * normalized
                    * NEAR_TERM_STABILITY_ABSOLUTE_CAP
                )
            return cost

        def select(candidate: list[WeeklyScheduleDay]) -> float:
            joint_cost = _joint_candidate_program_cost(
                tuple(offsets),
                daily_states,
                daily_state_options,
                request,
                config,
                target_distance_range,
                completed_miles_by_offset=completed_miles_by_offset,
                expected_target_projector=expected_target_projector,
                reconcile_allocated_prefix=False,
                _allocated_override=candidate,
                _sessions_override=source_sessions,
            )
            return _program_selection_cost(
                joint_cost,
                0.0,
                ordinary_easy_midpoint,
            ) + distance_stability_cost(candidate)

        return select

    if planning_horizon_days > VISIBLE_HORIZON_DAYS:
        target_ranges = (
            expected_target_projector(planned, planning_horizon_days)
            if expected_target_projector is not None
            else tuple(target_distance_range for _ in daily_states)
        )
        if len(target_ranges) != planning_horizon_days:
            raise ValueError(
                "Expected target trajectory must match the planning horizon"
            )
        horizon_target_range = _integrated_target_range(target_ranges)
        allocation_weekly_target = _average_target_range(target_ranges)
        allocation_states = _allocation_states_at_planned_times(
            days,
            daily_states,
            daily_state_options,
        )
        allocated_horizon = _allocate_visible_distance_ranges(
            days,
            allocation_states,
            horizon_target_range,
            config,
            weekly_target_range=allocation_weekly_target,
            assignments_per_total=3,
            allocation_selector=final_allocation_selector(days),
            allocation_candidate_limit=4,
            prior_distance_midpoints_by_date=(
                prior_distance_midpoints_by_date
            ),
        )
        # Feed finalized doses back through downstream recommendation state.
        # The first materialization cannot know that allocation will expand an
        # ordinary easy session into a medium-long one. Without this bounded
        # fixed-point pass, tomorrow is evaluated against the smaller draft
        # dose and exact compliance appears to be surprising new load.
        for _ in range(2):
            projected_ranges = {
                index: day.recommendation.distance_range_miles
                for index, day in enumerate(allocated_horizon)
                if day.recommendation is not None
                and day.recommendation.distance_range_miles is not None
            }
            reconciled_sessions = _materialize_candidate_sessions(
                tuple(offsets),
                daily_states,
                daily_state_options,
                request,
                config,
                prefix_cache=None,
                projected_distance_ranges=projected_ranges,
                role_overrides_by_offset=(
                    planner_search_diagnostics.winning_role_overrides
                ),
            )
            reconciled_days = list(days)
            for session in reconciled_sessions:
                result = session.recommendation
                role = (
                    "long_run"
                    if result.workout_type == WorkoutType.LONG
                    else "quality_run"
                    if result.workout_type in QUALITY_WORKOUT_TYPES
                    else "easy_run"
                    if result.workout_type != WorkoutType.REST
                    else "guardrail_rest_day"
                )
                reconciled_days[session.offset] = reconciled_days[
                    session.offset
                ].model_copy(
                    update={
                        "planned_at": session.state.as_of,
                        "recommendation": result,
                        "day_role": role,
                    }
                )
            reconciled = _allocate_visible_distance_ranges(
                reconciled_days,
                _allocation_states_at_planned_times(
                    reconciled_days,
                    daily_states,
                    daily_state_options,
                ),
                horizon_target_range,
                config,
                weekly_target_range=allocation_weekly_target,
                assignments_per_total=3,
                preserve_projected_recovery_caps=True,
                allocation_selector=final_allocation_selector(
                    reconciled_days
                ),
                allocation_candidate_limit=4,
                prior_distance_midpoints_by_date=(
                    prior_distance_midpoints_by_date
                ),
            )
            previous_signature = tuple(
                (
                    day.recommendation.workout_type,
                    day.recommendation.planned_for,
                    day.recommendation.distance_range_miles,
                )
                if day.recommendation is not None
                else None
                for day in allocated_horizon
            )
            reconciled_signature = tuple(
                (
                    day.recommendation.workout_type,
                    day.recommendation.planned_for,
                    day.recommendation.distance_range_miles,
                )
                if day.recommendation is not None
                else None
                for day in reconciled
            )
            allocated_horizon = reconciled
            if previous_signature == reconciled_signature:
                break
        visible_days = allocated_horizon[:VISIBLE_HORIZON_DAYS]
        planning_days = allocated_horizon
    else:
        visible_source_days = days[:VISIBLE_HORIZON_DAYS]
        visible_days = _allocate_visible_distance_ranges(
            visible_source_days,
            _allocation_states_at_planned_times(
                visible_source_days,
                daily_states[:VISIBLE_HORIZON_DAYS],
                daily_state_options[:VISIBLE_HORIZON_DAYS],
            ),
            target_distance_range,
            config,
            assignments_per_total=3,
            # The direct seven-day path has no later reconciliation pass.
            # Apply the same continuous recovery-pressure ceiling used by
            # the 21-day path so target funding cannot restore mileage that
            # the recommendation deliberately removed after a recent run.
            preserve_projected_recovery_caps=True,
            allocation_selector=final_allocation_selector(
                visible_source_days
            ),
            allocation_candidate_limit=4,
            prior_distance_midpoints_by_date=(
                prior_distance_midpoints_by_date
            ),
        )
        planning_days = visible_days
    planning_days = add_strength_suggestions(
        planning_days,
        config,
        days_since_quality_run=daily_states[0].days_since_quality_run,
        days_since_long_run=daily_states[0].days_since_long_run,
    )
    visible_days = planning_days[:VISIBLE_HORIZON_DAYS]
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
    visible_7d_scheduled_miles = sum(
        sum(day.recommendation.distance_range_miles) / 2
        for day in visible_days
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
    planned_14d_weekly_rate = (
        context_miles * 7.0 / len(context_days)
        if context_days
        else None
    )
    fatigue_half_life_days = float(
        config.get("coaching", {}).get(
            "continuous_fatigue_half_life_days",
            7,
        )
    )
    peak_projected_continuous_mileage_rate = (
        _peak_projected_continuous_mileage_rate(
            daily_states[0].recent_load.continuous_distance_miles,
            daily_states[0].as_of,
            visible_days,
            half_life_days=fatigue_half_life_days,
        )
    )
    if guardrail_rest_count:
        horizon_explanation = f"{guardrail_rest_count} planned day{'s were' if guardrail_rest_count != 1 else ' was'} changed to rest based on recovery, health, load, or weather."
    elif forced_rest_count:
        horizon_explanation = (
            "The continuous 21-day plan was recalculated around your selected "
            "rest-day constraints."
        )
    elif (
        planned_14d_weekly_rate is not None
        and sum(projected_range) / 2 > target_distance_range[1]
        and target_distance_range[0]
        <= planned_14d_weekly_rate
        <= target_distance_range[1]
        and (
            peak_projected_continuous_mileage_rate is None
            or peak_projected_continuous_mileage_rate
            <= target_distance_range[1] + 1e-9
        )
    ):
        horizon_explanation = (
            "The first seven days are busier, but your 14-day average and "
            "rolling mileage load stay on target."
        )
    elif (
        planned_14d_weekly_rate is not None
        and sum(projected_range) / 2 < target_distance_range[0]
        and target_distance_range[0]
        <= planned_14d_weekly_rate
        <= target_distance_range[1]
    ):
        horizon_explanation = (
            "The first seven days are lighter, but your 14-day average stays "
            "on target."
        )
    else:
        horizon_explanation = summarize_distance_alignment(
            projected_range,
            target_distance_range,
            target_evidence.capacity_reference_miles,
        )
    persisted_target_start_date = getattr(
        expected_target_projector,
        "start_date",
        None,
    )
    persisted_target_horizon = int(
        getattr(
            expected_target_projector,
            "prepared_horizon_days",
            0,
        )
    )
    persisted_target_trajectory = (
        list(expected_target_projector([], persisted_target_horizon))
        if expected_target_projector is not None
        and persisted_target_start_date is not None
        and persisted_target_horizon > 0
        else []
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
        visible_7d_scheduled_miles=round(visible_7d_scheduled_miles, 2),
        planned_14d_weekly_rate=(
            round(planned_14d_weekly_rate, 2)
            if planned_14d_weekly_rate is not None
            else None
        ),
        peak_projected_continuous_mileage_rate=(
            round(peak_projected_continuous_mileage_rate, 2)
            if peak_projected_continuous_mileage_rate is not None
            else None
        ),
        summary=horizon_explanation,
        days=visible_days,
        planning_days=planning_days,
        expected_target_start_date=persisted_target_start_date,
        expected_target_trajectory=persisted_target_trajectory,
        planner_diagnostics=PlannerDiagnostics(
            candidate_count=planner_search_diagnostics.candidate_count,
            full_score_count=planner_search_diagnostics.full_score_count,
            winning_candidate=(
                planner_search_diagnostics.winning_candidate
            ),
            translated_prior_candidate=(
                planner_search_diagnostics.translated_prior_candidate
            ),
            prior_continuation_candidate=(
                planner_search_diagnostics.prior_continuation_candidate
            ),
            near_term_stable_candidate=(
                planner_search_diagnostics.near_term_stable_candidate
            ),
            near_term_stable_offsets=[
                list(offsets)
                for offsets in planner_search_diagnostics.near_term_stable_offsets
            ],
            prior_independent_frequency_candidates=(
                planner_search_diagnostics.prior_independent_frequency_candidates
            ),
            unconstrained_winning_candidate=(
                planner_search_diagnostics.unconstrained_winning_candidate
            ),
            stability_tie_break_applied=(
                planner_search_diagnostics.stability_tie_break_applied
            ),
            stability_tolerance=(
                planner_search_diagnostics.stability_tolerance
            ),
        ),
    )
