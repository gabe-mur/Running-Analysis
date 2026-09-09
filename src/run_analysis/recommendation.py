"""Deterministic next-run recommendation engine with inspectable rules.

No database or frontend code belongs here. The engine accepts a `FitnessState`
and configuration, evaluates named rules in a fixed order, and returns both a
prescription and the complete rule trace.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import ceil, floor, sqrt
from typing import Any

from .environmental_stress import assess_training_weather
from .race_goals import (
    RACE_GOALS,
    build_weeks_before_taper,
    configured_race_goal,
    format_pace,
    required_compound_progression,
)
from .recovery import (
    EASY_RUN_RESIDUAL_LIMIT,
    EASY_VOLUME_RESIDUAL_FLOOR,
    TAXING_RUN_RESIDUAL_LIMIT,
    estimate_recovery,
)
from .web.schemas import (
    ConfidenceLevel,
    CurrentHealthStatus,
    FitnessState,
    LoadContext,
    QualitySessionType,
    ReadinessFlag,
    RecommendationRequest,
    RecommendationResponse,
    RuleTrace,
    WorkoutStep,
    WorkoutType,
)


@dataclass(frozen=True, slots=True)
class RuleDefinition:
    rule_id: str
    description: str


RULE_CATALOG: tuple[RuleDefinition, ...] = (
    RuleDefinition("planned_timing", "Recovery spacing and rolling load are projected to the planned run time."),
    RuleDefinition("planned_weather", "Forecast stress is graded against recent training exposure, with absolute guardrails for extreme conditions; ordinary rain is neutral."),
    RuleDefinition("health_pain", "Pain or injury concern blocks a running prescription."),
    RuleDefinition("health_sick", "Sick/recovering limits training to a short, low-load return-to-running check. Judging symptom severity is the athlete's call, not the app's."),
    RuleDefinition("recent_recovery_load", "Athlete-relative session load decays continuously; easy and taxing workouts use different readiness levels."),
    RuleDefinition("high_recent_load", "High recent load combines retained mileage capacity with confidence-weighted HR-load evidence."),
    RuleDefinition("moderate_leakage", "Recent Z3 time contributes graded evidence based on its excess and sample size."),
    RuleDefinition("recent_costly_response", "Execution drift or high perceived effort scales the next prescription; pace-at-HR response is already counted in recovery."),
    RuleDefinition("recent_drift_caution", "Only drift above the reference contributes, and contradictory strong performance discounts it."),
    RuleDefinition("recent_high_rpe", "A high reported effort adds recovery caution without replacing recorded HR load."),
    RuleDefinition("mechanical_load", "Available elevation data can identify hilly or downhill-heavy mechanical stress."),
    RuleDefinition("returning_consistency", "Sparse recent running favors rebuilding routine before quality."),
    RuleDefinition("post_illness_quality_check", "Sick/recovering remains blocked until the current self-report returns to normal; no hidden clearance stage is added afterward."),
    RuleDefinition("quality_variant", "Enabled quality stimuli vary deterministically; a long gap favors adaptable effort-based work rather than forcing track intervals."),
    RuleDefinition("race_goal", "A validated race goal changes workout composition and taper priority without overriding health or load guardrails."),
    RuleDefinition("weekly_sequence_priority", "The rolling planner applies only a small workout-composition preference; recovery and training evidence remain primary."),
    RuleDefinition("workout_scoring", "Easy, long, and quality candidates receive additive evidence scores."),
    RuleDefinition("long_run_eligible", "Long-run recency adds priority only when demonstrated progression supports a meaningful distance beyond ordinary easy running."),
    RuleDefinition("quality_eligible", "Quality readiness depends on several signals; longest-run distance is not a gate."),
    RuleDefinition("default_aerobic", "When no higher-priority rule fires, prescribe ordinary aerobic running."),
)

RULES = {rule.rule_id: rule for rule in RULE_CATALOG}


def _settings(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "long_run_progression_factor": 1.10,
        "long_run_target_progression_fraction": 0.05,
        "moderate_intensity_leakage_fraction": 0.17,
        "quality_recency_reference_days": 7,
        "typical_rest_days_between_runs": 1,
        "capacity_retention_half_life_days": 84,
        "capacity_retention_grace_days": 28,
        "minimum_running_days_28d_for_quality": 8,
        "long_run_recency_reference_days": 7,
        "reduced_volume_factor": 0.70,
        **{
            key: value
            for key, value in config.get("coaching", {}).items()
            if key != "quality_sessions"
        },
        "quality_sessions": {
            "fartlek": True,
            "short_intervals": True,
            "long_intervals": True,
            "threshold": True,
            "progression": True,
            "hill_repeats": False,
            **(
                config.get("coaching", {}).get("quality_sessions") or {}
            ),
        },
    }


def _moderate_leakage_strength(
    state: FitnessState,
    settings: dict[str, Any],
) -> tuple[float, float]:
    """Return graded excess-Z3 evidence and its uncertainty transition band."""
    evidence_runs = state.moderate_evidence_runs_14d
    fraction = state.moderate_fraction_14d
    # Correlated minute-by-minute HR samples cannot be treated as independent.
    # Use run count as the effective sample size and retain a practical floor
    # so a tiny numerical crossing never becomes a full coaching reversal.
    uncertainty_band = max(0.015, 0.06 / sqrt(max(1, evidence_runs)))
    if fraction is None or evidence_runs < 2:
        return 0.0, uncertainty_band
    reference = float(settings["moderate_intensity_leakage_fraction"])
    strength = min(
        1.0,
        max(0.0, (fraction - reference) / uncertainty_band),
    )
    return strength, uncertainty_band


def effective_load_ratio(load: LoadContext) -> float | None:
    """Combine retained mileage capacity with confidence-weighted HR load.

    Distance is the stable primary signal because its retained capacity has a
    meaningful long-term denominator.  HR-derived load still matters when it
    rises faster than mileage, but its influence is limited by confidence so a
    weak or temporarily depressed prior-load norm cannot take over the plan.
    The combined value never reduces a mileage-based caution.
    """
    continuous_ratio = load.continuous_fatigue_to_capacity_ratio
    short_term_distance_ratio = (
        load.continuous_short_term_distance_miles
        / load.capacity_reference_miles
        if load.continuous_short_term_distance_miles is not None
        and load.capacity_reference_miles
        else None
    )
    continuous_ratios = [
        value
        for value in (continuous_ratio, short_term_distance_ratio)
        if value is not None
    ]
    if continuous_ratios:
        return max(continuous_ratios)
    distance_ratio = load.acute_distance_to_capacity_ratio
    hr_ratio = load.acute_to_prior_ratio
    # Future sessions have distance and workout structure, but no observed HR
    # response. Their estimated zone load must not create an HR premium and
    # then penalize the same planned work a second time; projected recovery
    # load already accounts for workout intensity and spacing.
    if "includes_planned_sessions" in load.flags and distance_ratio is not None:
        return distance_ratio
    if distance_ratio is None:
        return hr_ratio
    if hr_ratio is None or hr_ratio <= distance_ratio:
        return distance_ratio
    confidence_weight = {
        ConfidenceLevel.HIGH: 0.35,
        ConfidenceLevel.MODERATE: 0.20,
        ConfidenceLevel.LOW: 0.10,
    }[load.confidence]
    # A temporarily depressed prior-HR-load denominator can make the raw
    # ratio arbitrarily large. HR disagreement may add a meaningful caution,
    # but it cannot contribute more than 0.25 ratio points above the retained
    # distance-capacity signal by itself.
    hr_premium = min(
        0.25,
        (hr_ratio - distance_ratio) * confidence_weight,
    )
    return distance_ratio + hr_premium


def _zone_copy(config: dict[str, Any], zone: str) -> str:
    """Render a configured zone as athlete-facing copy.

    Instruction text must never hard-code a bpm range; the athlete's zones are
    configurable and the prescription has to agree with them.
    """

    bounds = (config.get("zones") or {}).get(zone)
    if not bounds or len(bounds) < 2:
        return f"{zone.upper()}"
    return f"{int(bounds[0])}–{int(bounds[1])} bpm"


def long_run_reference_miles(state: FitnessState) -> float:
    """Best retained evidence of general single-run durability."""

    return max(
        state.longest_run_30d_miles,
        state.retained_long_run_capacity_miles,
    )


def single_session_progression_reference_miles(state: FitnessState) -> float:
    """Current progression guardrail, with a conservative re-entry fallback."""

    if state.longest_run_30d_miles > 0:
        return state.longest_run_30d_miles
    return state.retained_long_run_capacity_miles * 0.80


def _long_run_distance(
    cap_miles: float,
    weekly_norm_miles: float,
    recent_single_run_miles: float,
    demonstrated_runs_per_week: float,
    target_progression_fraction: float,
    returning_to_retained_capacity: bool,
) -> tuple[float, float, bool]:
    """Resolve the next step in a maintained long-run progression lane.

    Returns ``(lower, upper, capped_by_progression)``. The target is a
    proportional increase from maintained recent distance, separate from the
    larger maximum guardrail in ``cap_miles``. Weekly program balance is
    reconciled by the joint schedule allocator. The standalone coaching target
    therefore expresses maintained single-run progression instead of lowering
    it merely because recent run frequency rose. Otherwise adding support days
    reduces the permitted long share, which adds more support days—a circular
    suppression of the primary durability lane.
    """

    del weekly_norm_miles, demonstrated_runs_per_week
    target = recent_single_run_miles * (1.0 + target_progression_fraction)
    practical = min(cap_miles, target)
    # Center the range on a practical quarter mile. Quarter-mile endpoints
    # keep a proportional target from turning into either a fixed half-mile
    # jump or zero midpoint progression after display rounding.
    center = max(
        1.0,
        (
            ceil(practical * 4 - 1e-9) / 4
            if returning_to_retained_capacity
            else floor(practical * 4 + 0.5) / 4
        ),
    )
    # The progression ceiling governs the prescribed midpoint. The displayed
    # route/GPS range is execution flexibility around that target; clipping
    # only its upper endpoint creates misleading ranges such as 7.0-7.04.
    upper = center + 0.25
    lower = max(0.1, center - 0.25)
    return (
        round(lower, 2),
        round(max(upper, lower), 2),
        practical >= cap_miles - 1e-9,
    )


def typical_easy_distance(state: FitnessState) -> tuple[float, float]:
    load = state.recent_load.trailing_28d
    if state.typical_easy_run_miles is not None:
        average = state.typical_easy_run_miles
    elif load.activity_count > 0 and load.distance_miles > 0:
        average = load.distance_miles / load.activity_count
    else:
        return (0.0, 0.0)
    lower = max(0.1, round(average * 0.85 * 2) / 2)
    upper = max(lower, round(average * 2) / 2)
    return (lower, upper)


def _trace(rule: RuleDefinition, fired: bool, **facts) -> RuleTrace:
    return RuleTrace(
        rule_id=rule.rule_id,
        description=rule.description,
        fired=fired,
        facts={key: value for key, value in facts.items()},
    )


def _result(
    state: FitnessState,
    *,
    workout_type: WorkoutType,
    quality_session_type: QualitySessionType | None = None,
    title: str,
    distance: tuple[float, float] | None = None,
    duration: tuple[float, float] | None = None,
    zones: list[str] | None = None,
    structure: list[WorkoutStep] | None = None,
    reasons: list[str] | None = None,
    warnings: list[str] | None = None,
    modifications: list[str] | None = None,
    confidence: ConfidenceLevel = ConfidenceLevel.MODERATE,
    readiness: ReadinessFlag = ReadinessFlag.READY,
    readiness_reason: str | None = None,
    trace: list[RuleTrace],
) -> RecommendationResponse:
    resolved_reasons = reasons or []
    if readiness_reason is None:
        if readiness == ReadinessFlag.READY:
            readiness_reason = "Current health, recovery spacing, load, and forecast checks allow this workout."
        elif resolved_reasons:
            readiness_reason = resolved_reasons[0]
        else:
            readiness_reason = "Recent training or recovery calls for an adjustment."
    return RecommendationResponse(
        generated_at=datetime.now(timezone.utc),
        fitness_state_as_of=state.as_of,
        planned_for=state.as_of,
        planned_weather=state.planned_weather,
        workout_type=workout_type,
        quality_session_type=quality_session_type,
        title=title,
        distance_range_miles=distance,
        duration_range_minutes=duration,
        target_zones=zones or [],
        structure=structure or [],
        reasons=resolved_reasons,
        warnings=warnings or [],
        modification_rules=modifications or [],
        confidence=confidence,
        readiness=readiness,
        readiness_reason=readiness_reason,
        rule_trace=trace,
    )


def _weather_scaled_distance(
    distance: tuple[float, float], weather_stress: float
) -> tuple[float, float]:
    if weather_stress <= 0:
        return distance
    factor = max(0.70, 1.0 - 0.12 * weather_stress)
    low = round(max(0.1, distance[0] * factor), 1)
    high = round(max(low, distance[1] * factor), 1)
    return low, high


def _select_quality_variant(
    state: FitnessState, settings: dict[str, Any]
) -> QualitySessionType:
    configured = settings.get("quality_sessions") or {}
    ordinary_order = [
        QualitySessionType.FARTLEK,
        QualitySessionType.PROGRESSION,
        QualitySessionType.THRESHOLD,
        QualitySessionType.SHORT_INTERVALS,
        QualitySessionType.LONG_INTERVALS,
        QualitySessionType.HILL_REPEATS,
    ]
    goal_name = str(settings.get("training_goal", "general_fitness"))
    profile = RACE_GOALS.get(goal_name)
    preferred = [QualitySessionType(value) for value in profile.preferred_quality] if profile else []
    order = [*preferred, *(item for item in ordinary_order if item not in preferred)]
    enabled = [item for item in order if bool(configured.get(item.value, False))]
    if not enabled:
        # Settings validation prevents this in the application; retain a safe
        # deterministic fallback for direct library callers.
        return QualitySessionType.FARTLEK
    if profile:
        # Goal-specific order describes the useful stimulus, not a mandatory
        # workout. Rotate within it so even race preparation does not repeat
        # the same structure indefinitely.
        preferred_enabled = [item for item in preferred if item in enabled]
        if preferred_enabled:
            return preferred_enabled[
                state.completed_quality_session_count % len(preferred_enabled)
            ]
    if state.days_since_quality_run is None or state.days_since_quality_run >= 21:
        adaptable = [
            item
            for item in (
                QualitySessionType.FARTLEK,
                QualitySessionType.PROGRESSION,
                QualitySessionType.THRESHOLD,
            )
            if item in enabled
        ]
        if adaptable:
            return adaptable[
                state.completed_quality_session_count % len(adaptable)
            ]
    return enabled[state.completed_quality_session_count % len(enabled)]


def _quality_structure_distance(
    state: FitnessState,
    kind: QualitySessionType,
    structure: list[WorkoutStep],
    fallback: tuple[float, float],
) -> tuple[float, float]:
    """Size fixed-time quality sessions from the work actually prescribed.

    Warm-up, work, recoveries, and cool-down all count toward mileage.  What
    does not count is an invented easy-running block added only to make a
    weekly arithmetic target.  Distance remains an execution range because
    the same timed effort covers different ground at different paces.
    """

    if kind not in {
        QualitySessionType.SHORT_INTERVALS,
        QualitySessionType.LONG_INTERVALS,
        QualitySessionType.THRESHOLD,
    }:
        return fallback
    total_minutes = 0.0
    for step in structure:
        if step.duration_minutes is not None:
            total_minutes += step.duration_minutes
        elif step.repetitions and step.work_duration_minutes:
            total_minutes += step.repetitions * step.work_duration_minutes
            if step.recovery_duration_minutes:
                total_minutes += (
                    max(0, step.repetitions - 1)
                    * step.recovery_duration_minutes
                )
        else:
            return fallback
    pace = (
        state.standardized_pace_at_target_hr.minutes_per_mile
        if state.standardized_pace_at_target_hr is not None
        else None
    )
    recent = state.recent_load.trailing_28d
    if pace is None and recent.distance_miles > 0 and recent.moving_minutes > 0:
        pace = recent.moving_minutes / recent.distance_miles
    if pace is None or pace <= 0:
        return fallback
    pace = min(20.0, max(6.0, pace))
    center = max(0.5, round((total_minutes / pace) * 4) / 4)
    return (round(max(0.1, center - 0.25), 2), round(center + 0.25, 2))


def _quality_prescription(kind: QualitySessionType) -> dict[str, Any]:
    prescriptions: dict[QualitySessionType, dict[str, Any]] = {
        QualitySessionType.FARTLEK: {
            "workout_type": WorkoutType.INTERVALS,
            "title": "Fartlek by feel",
            "distance": (4.0, 5.0),
            "zones": ["Z1", "Z2", "controlled strong effort"],
            "structure": [
                WorkoutStep(instruction="Warm up easily until stride and breathing feel natural; 10–15 minutes is usually enough.", target_zones=["Z1", "Z2"]),
                WorkoutStep(instruction="For 20 minutes, add 8 controlled surges using natural landmarks on the route. Let each surge last roughly 45–90 seconds, then run easily until breathing is composed before starting the next one. Do not force identical repetitions.", duration_minutes=20, phase="work", repetitions=8, work_duration_range_minutes=(0.75, 1.5), target_zones=["Z3", "Z4 effort"]),
                WorkoutStep(instruction="Cool down easily. Finish while another controlled surge would still have been possible.", target_zones=["Z1", "Z2"]),
            ],
        },
        QualitySessionType.SHORT_INTERVALS: {
            "workout_type": WorkoutType.INTERVALS,
            "title": "Short controlled pickups",
            "distance": (4.0, 4.5),
            "zones": ["Z1", "Z2", "Z4 effort"],
            "structure": [
                WorkoutStep(instruction="Easy warm-up.", duration_minutes=12, target_zones=["Z1", "Z2"]),
                WorkoutStep(instruction="Run 8 × 1 minute at controlled fast effort with 90 seconds of easy jogging after each. Keep the final pickup as smooth as the first; do not sprint or chase HR lag.", phase="work", repetitions=8, work_duration_minutes=1, recovery_duration_minutes=1.5, target_zones=["Z4 effort"]),
                WorkoutStep(instruction="Easy cool-down.", duration_minutes=10, target_zones=["Z1", "Z2"]),
            ],
        },
        QualitySessionType.LONG_INTERVALS: {
            "workout_type": WorkoutType.INTERVALS,
            "title": "Long controlled efforts",
            "distance": (4.5, 5.0),
            "zones": ["Z1", "Z2", "Z4 effort"],
            "structure": [
                WorkoutStep(instruction="Easy warm-up.", duration_minutes=12, target_zones=["Z1", "Z2"]),
                WorkoutStep(instruction="Run 3 × 5 minutes at controlled hard effort with 2 minutes of easy jogging between efforts. Finish the third segment at the same effort as the first.", phase="work", repetitions=3, work_duration_minutes=5, recovery_duration_minutes=2, target_zones=["upper Z3", "low Z4"]),
                WorkoutStep(instruction="Easy cool-down.", duration_minutes=10, target_zones=["Z1", "Z2"]),
            ],
        },
        QualitySessionType.THRESHOLD: {
            "workout_type": WorkoutType.TEMPO_THRESHOLD,
            "title": "Continuous threshold run",
            "distance": (4.5, 5.0),
            "zones": ["Z1", "Z2", "upper Z3 / low Z4"],
            "structure": [
                WorkoutStep(instruction="Easy warm-up.", duration_minutes=12, target_zones=["Z1", "Z2"]),
                WorkoutStep(instruction="Run 18 minutes continuously at controlled threshold effort. Start conservatively and finish feeling that another 2–3 minutes would have been possible.", duration_minutes=18, phase="work", target_zones=["upper Z3", "low Z4"]),
                WorkoutStep(instruction="Easy cool-down.", duration_minutes=10, target_zones=["Z1", "Z2"]),
            ],
        },
        QualitySessionType.PROGRESSION: {
            "workout_type": WorkoutType.TEMPO_THRESHOLD,
            "title": "Controlled progression run",
            "distance": (4.0, 5.0),
            "zones": ["Z1", "Z2", "Z3 finish"],
            "structure": [
                WorkoutStep(instruction="Run the first half relaxed in Z1/Z2.", target_zones=["Z1", "Z2"]),
                WorkoutStep(instruction="Gradually increase through the second half, finishing controlled in Z3 without sprinting.", phase="work", target_zones=["Z2", "Z3"]),
            ],
        },
        QualitySessionType.HILL_REPEATS: {
            "workout_type": WorkoutType.INTERVALS,
            "title": "Hills by terrain",
            "distance": (4.0, 4.5),
            "zones": ["Z1", "Z2", "strong controlled effort"],
            "structure": [
                WorkoutStep(instruction="Easy warm-up on flat terrain.", duration_minutes=12, target_zones=["Z1", "Z2"]),
                WorkoutStep(instruction="Run 8 × 45 seconds uphill at strong controlled effort. Jog gently downhill, regain control, and stop early if form changes.", phase="work", repetitions=8, work_duration_minutes=0.75, target_zones=["strong controlled effort"]),
                WorkoutStep(instruction="Easy cool-down.", duration_minutes=10, target_zones=["Z1", "Z2"]),
            ],
        },
    }
    return prescriptions[kind]


def scale_quality_session(
    result: RecommendationResponse,
    original_distance: tuple[float, float],
    allocated_distance: tuple[float, float],
) -> RecommendationResponse:
    """Scale the work dose when weekly allocation shortens a quality run."""

    kind = result.quality_session_type
    if kind is None or len(result.structure) < 2:
        return result
    original_midpoint = sum(original_distance) / 2
    allocated_midpoint = sum(allocated_distance) / 2
    # Small range normalization (for example 4.0–5.0 becoming 4.0–4.5)
    # trims discretionary easy mileage, not the coached work segment. Scale
    # the actual quality dose only after a material reduction.
    if original_midpoint <= 0 or allocated_midpoint >= original_midpoint * 0.90:
        return result
    raw_scale = min(1.0, allocated_midpoint / original_midpoint)
    # Preserve a real threshold/interval session while its allocated distance
    # can still carry most of the coached work. Only a severe contraction turns
    # it into pickups; otherwise a normal, merely shorter quality workout is
    # more faithful to the planner's intent.
    if raw_scale < 0.50:
        repetitions = max(3, round(6 * raw_scale))
        return result.model_copy(
            update={
                "workout_type": WorkoutType.INTERVALS,
                "quality_session_type": QualitySessionType.SHORT_INTERVALS,
                "title": "Easy run with controlled pickups",
                "target_zones": ["Z1", "Z2", "Z4 effort"],
                "structure": [
                    WorkoutStep(
                        instruction="Run easily until stride and breathing feel settled.",
                        target_zones=["Z1", "Z2"],
                    ),
                    WorkoutStep(
                        instruction=(
                            f"Run {repetitions} × 30 seconds at controlled fast effort "
                            "with 90 seconds of easy running after each. Stay smooth; do not sprint."
                        ),
                        phase="work",
                        repetitions=repetitions,
                        work_duration_minutes=0.5,
                        recovery_duration_minutes=1.5,
                        target_zones=["Z4 effort"],
                    ),
                    WorkoutStep(
                        instruction="Finish the remaining distance conversationally.",
                        target_zones=["Z1", "Z2"],
                    ),
                ],
                "reasons": [
                    *result.reasons,
                    "The available load supports a short quality stimulus, so controlled pickups replace a full interval or threshold session.",
                ],
            }
        )
    scale = raw_scale
    if kind == QualitySessionType.FARTLEK:
        minutes = max(12, round(20 * scale))
        repetitions = max(5, round(8 * scale))
        work = WorkoutStep(
            instruction=(
                f"For {minutes} minutes, add {repetitions} controlled surges using natural landmarks. "
                "Let each last roughly 45–90 seconds, then run easily until breathing is composed."
            ),
            duration_minutes=float(minutes), phase="work", repetitions=repetitions,
            work_duration_range_minutes=(0.75, 1.5), target_zones=["Z3", "Z4 effort"],
        )
    elif kind == QualitySessionType.SHORT_INTERVALS:
        repetitions = max(5, round(8 * scale))
        work = WorkoutStep(
            instruction=(
                f"Run {repetitions} × 1 minute at controlled fast effort with 90 seconds easy after each. "
                "Keep the final pickup as smooth as the first."
            ),
            phase="work", repetitions=repetitions, work_duration_minutes=1,
            recovery_duration_minutes=1.5, target_zones=["Z4 effort"],
        )
    elif kind == QualitySessionType.LONG_INTERVALS:
        repetitions = max(2, round(3 * scale))
        work = WorkoutStep(
            instruction=(
                f"Run {repetitions} × 5 minutes at controlled hard effort with 2 minutes easy between efforts. "
                "Finish the final segment at the same effort as the first."
            ),
            phase="work", repetitions=repetitions, work_duration_minutes=5,
            recovery_duration_minutes=2, target_zones=["upper Z3", "low Z4"],
        )
    elif kind == QualitySessionType.THRESHOLD:
        minutes = max(12, round(18 * scale))
        work = WorkoutStep(
            instruction=(
                f"Run {minutes} minutes continuously at controlled threshold effort. "
                "Start conservatively and finish with another 2–3 minutes available."
            ),
            duration_minutes=float(minutes),
            phase="work",
            target_zones=["upper Z3", "low Z4"],
        )
    elif kind == QualitySessionType.HILL_REPEATS:
        repetitions = max(5, round(8 * scale))
        work = WorkoutStep(
            instruction=(
                f"Run {repetitions} × 45 seconds uphill at strong controlled effort. "
                "Jog gently downhill and stop early if form changes."
            ),
            phase="work", repetitions=repetitions, work_duration_minutes=0.75,
            target_zones=["strong controlled effort"],
        )
    else:
        # A progression workout already defines its work by fractions of the
        # total run, so shortening the allocated distance scales it directly.
        return result.model_copy(
            update={
                "reasons": [
                    *result.reasons,
                    "The progression stimulus was retained while total distance was shortened to fit the weekly load.",
                ]
            }
        )
    return result.model_copy(
        update={
            "structure": [
                result.structure[0],
                work,
                *result.structure[2:],
            ],
            "reasons": [
                *result.reasons,
                "The quality dose was shortened instead of deleting the week's quality stimulus.",
            ],
        }
    )


EXTENDED_QUALITY_DURATION_MINUTES = 120.0


def structure_extended_quality_session(
    result: RecommendationResponse,
    estimated_duration_minutes: float,
) -> RecommendationResponse:
    """Turn a 2–3 hour quality day into one bounded multi-part workout.

    Extending the ordinary repetitions would turn an endurance session into
    hours of hard work. The quality dose therefore stays finite while easy
    aerobic running supplies the additional duration.
    """
    if (
        result.quality_session_type is None
        or result.workout_type
        not in {WorkoutType.INTERVALS, WorkoutType.TEMPO_THRESHOLD}
    ):
        return result
    kind = result.quality_session_type
    if estimated_duration_minutes < EXTENDED_QUALITY_DURATION_MINUTES:
        return result
    work: dict[QualitySessionType, tuple[str, WorkoutStep]] = {
        QualitySessionType.FARTLEK: (
            "Extended fartlek aerobic session",
            WorkoutStep(
                instruction="Complete 3 sets of 4 × 1-minute controlled surges with 90 seconds easy after each surge and 8 minutes easy between sets.",
                phase="work", repetitions=12, work_duration_minutes=1,
                recovery_duration_minutes=1.5,
                target_zones=["Z3", "Z4 effort"],
            ),
        ),
        QualitySessionType.SHORT_INTERVALS: (
            "Extended aerobic session with short pickups",
            WorkoutStep(
                instruction="Complete 2 sets of 6 × 1 minute controlled fast with 90 seconds easy after each pickup and 8 minutes easy between sets.",
                phase="work", repetitions=12, work_duration_minutes=1,
                recovery_duration_minutes=1.5,
                target_zones=["Z4 effort"],
            ),
        ),
        QualitySessionType.LONG_INTERVALS: (
            "Extended aerobic session with long efforts",
            WorkoutStep(
                instruction="Run 3 × 8 minutes at controlled hard effort with 3 minutes easy between efforts.",
                phase="work", repetitions=3, work_duration_minutes=8,
                recovery_duration_minutes=3,
                target_zones=["upper Z3", "low Z4"],
            ),
        ),
        QualitySessionType.THRESHOLD: (
            "Extended aerobic session with threshold blocks",
            WorkoutStep(
                instruction="Run 3 × 12 minutes at controlled threshold effort with 5 minutes easy between blocks.",
                phase="work", repetitions=3, work_duration_minutes=12,
                recovery_duration_minutes=5,
                target_zones=["upper Z3", "low Z4"],
            ),
        ),
        QualitySessionType.PROGRESSION: (
            "Extended aerobic progression session",
            WorkoutStep(
                instruction="After the opening aerobic running, progress for 30 minutes from steady Z2 to controlled Z3, then return to easy effort.",
                duration_minutes=30,
                phase="work",
                target_zones=["Z2", "Z3"],
            ),
        ),
        QualitySessionType.HILL_REPEATS: (
            "Extended aerobic session with hills",
            WorkoutStep(
                instruction="Complete 2 sets of 4 × 45 seconds uphill at strong controlled effort, jogging gently downhill and running 10 minutes easy between sets.",
                phase="work", repetitions=8, work_duration_minutes=0.75,
                target_zones=["strong controlled effort"],
            ),
        ),
    }
    title, quality_step = work[kind]
    rounded_duration = int(round(estimated_duration_minutes / 5) * 5)
    return result.model_copy(
        update={
            "title": title,
            "structure": [
                WorkoutStep(
                    instruction="Run the first 20 minutes easily before beginning any quality work.",
                    duration_minutes=20,
                    target_zones=["Z1", "Z2"],
                ),
                quality_step,
                WorkoutStep(
                    instruction=f"Run all remaining time easily to finish about {rounded_duration} minutes total. Do not add more quality work to fill the duration.",
                    target_zones=["Z1", "Z2"],
                ),
            ],
            "reasons": [
                *result.reasons,
                "Because the estimated session is at least two hours, the quality dose is capped and embedded inside aerobic endurance running.",
            ],
            "warnings": [
                *result.warnings,
                "Only the defined work blocks are quality effort; the rest of this extended session stays easy.",
            ],
            "modification_rules": [
                *result.modification_rules,
                "If normal fueling, hydration, or long-duration preparation is unavailable, shorten the total easy duration rather than compressing or extending the quality blocks.",
            ],
        }
    )


def _goal_quality_context(
    kind: QualitySessionType,
    goal_label: str,
    goal_pace: float,
) -> WorkoutStep:
    pace_text = format_pace(goal_pace)
    if kind in {QualitySessionType.SHORT_INTERVALS, QualitySessionType.LONG_INTERVALS}:
        instruction = (
            f"Goal context: {goal_label} pace is {pace_text}. Repetitions may approach that pace "
            "when conditions and form are normal, but controlled effort and even reps take priority."
        )
    elif kind == QualitySessionType.THRESHOLD:
        instruction = (
            f"Goal context: {goal_label} pace is {pace_text}. Use it as a reference—not a split mandate—and "
            "keep every work segment controlled enough to complete the session evenly."
        )
    else:
        instruction = (
            f"Goal context: {goal_label} pace is {pace_text}. Approach it only late in the controlled portion "
            "when effort, weather, and form remain normal."
        )
    return WorkoutStep(instruction=instruction)


def recommend_next_run(
    state: FitnessState,
    request: RecommendationRequest,
    config: dict[str, Any],
    *,
    weekly_role: str | None = None,
    allowed_candidates: set[str] | None = None,
) -> RecommendationResponse:
    """Evaluate guardrails, then long/quality eligibility, then easy default."""

    settings = _settings(config)
    trace: list[RuleTrace] = []
    health = request.health_status
    easy_distance = typical_easy_distance(state)
    load_ratio = effective_load_ratio(state.recent_load)
    # Completed/projected load contributes continuously once it exceeds the
    # athlete's capacity reference. A doubling of the reference represents a
    # full-strength adjustment; there is no separate 130% planning cliff.
    load_stress = (
        min(1.0, max(0.0, load_ratio - 1.0))
        if load_ratio is not None
        else 0.0
    )
    maximum_load_reduction = 1.0 - float(settings["reduced_volume_factor"])
    high_load = (
        sum(easy_distance) / 2 * load_stress * maximum_load_reduction
        >= 0.25
    )
    moderate_leakage_strength, moderate_uncertainty_band = (
        _moderate_leakage_strength(state, settings)
    )
    z3_leakage = moderate_leakage_strength >= 0.5
    recovery = estimate_recovery(state)

    trace.append(
        _trace(
            RULES["planned_timing"],
            True,
            planned_for=state.as_of.isoformat(),
            hours_since_last_run=(
                round(state.days_since_last_run * 24, 1)
                if state.days_since_last_run is not None
                else None
            ),
        )
    )
    weather = state.planned_weather
    weather_stress = assess_training_weather(
        weather, state.weather_exposure_baseline
    )
    environmental_adaptation = weather_stress.needs_adaptation
    environmental_caution = weather_stress.caution
    trace.append(
        _trace(
            RULES["planned_weather"],
            environmental_adaptation,
            forecast_available=weather is not None,
            temperature_f=weather.temperature_f if weather else None,
            apparent_temperature_f=weather.apparent_temperature_f if weather else None,
            dewpoint_f=weather.dewpoint_f if weather else None,
            wind_speed_mph=weather.wind_speed_mph if weather else None,
            stress_score=round(weather_stress.score, 3),
            stress_band=weather_stress.band,
            heat_component=round(weather_stress.heat, 3),
            humidity_component=round(weather_stress.humidity, 3),
            cold_component=round(weather_stress.cold, 3),
            wind_component=round(weather_stress.wind, 3),
            precipitation_component=round(weather_stress.precipitation, 3),
            relative_spike=round(weather_stress.relative_spike, 3),
            recent_weather_samples=(
                state.weather_exposure_baseline.sample_count
                if state.weather_exposure_baseline
                else 0
            ),
            extreme_weather=weather_stress.extreme,
            extreme_reasons="; ".join(weather_stress.extreme_reasons),
            emergency_alerts_checked=(
                weather.emergency_alerts_checked if weather else False
            ),
            emergency_alerts=(
                "; ".join(alert.event for alert in weather.emergency_alerts)
                if weather
                else ""
            ),
        )
    )

    rule = RULES["health_pain"]
    fired = health == CurrentHealthStatus.PAIN_OR_INJURY_CONCERN
    trace.append(_trace(rule, fired, health_status=health.value))
    if fired:
        return _result(
            state,
            workout_type=WorkoutType.REST,
            title="No running prescription",
            reasons=["You reported pain or an injury concern."],
            warnings=["This is a training guardrail, not a diagnosis. Seek appropriate clinical guidance for concerning or persistent symptoms."],
            modifications=["Do not convert this to a run because the app cannot assess the cause or severity of pain."],
            confidence=ConfidenceLevel.HIGH,
            readiness=ReadinessFlag.NOT_READY,
            trace=trace,
        )

    rule = RULES["health_sick"]
    recovering_mode = health == CurrentHealthStatus.SICK_OR_RECOVERING
    trace.append(_trace(rule, recovering_mode, health_status=health.value, recent_illness=state.recent_illness_or_recovery))
    if recovering_mode:
        recovery_distance = (
            max(0.1, round(easy_distance[0] * 0.5 * 2) / 2),
            max(0.1, round(easy_distance[1] * 0.6 * 2) / 2),
        )
        return _result(
            state,
            workout_type=WorkoutType.RECOVERY,
            title="Recovery-mode aerobic run",
            distance=recovery_distance,
            zones=["Z1", "low Z2"],
            structure=[
                WorkoutStep(
                    instruction="Keep the entire run conversational and stop if respiratory or systemic symptoms increase.",
                    target_zones=["Z1", "low Z2"],
                )
            ],
            reasons=["Sick/recovering status permits only a short, low-load return-to-running check."],
            warnings=["This is not medical clearance, and the app cannot assess your symptoms. Fever, chest pain, unusual shortness of breath, or worsening symptoms mean do not run today — ignore this prescription entirely."],
            modifications=["Change the current status to Normal when you are ready for ordinary load and workout scoring."],
            confidence=ConfidenceLevel.MODERATE,
            readiness=ReadinessFlag.CAUTION,
            trace=trace,
        )

    if weather_stress.extreme:
        return _result(
            state,
            workout_type=WorkoutType.REST,
            title="Extreme-weather no-run window",
            reasons=[
                "The forecast indicates "
                + ", ".join(weather_stress.extreme_reasons)
                + "."
            ],
            warnings=[
                "Forecast data can miss local warnings and rapidly changing conditions; official emergency guidance always takes precedence."
            ],
            modifications=[
                "Move the run to a non-extreme time rather than trying to preserve the planned outdoor session."
            ],
            confidence=ConfidenceLevel.HIGH,
            readiness=ReadinessFlag.NOT_READY,
            readiness_reason="The forecast crosses an absolute extreme-weather guardrail.",
            trace=trace,
        )

    if (
        state.last_run is None
        and state.recent_load.trailing_28d.activity_count == 0
    ):
        return _result(
            state,
            workout_type=WorkoutType.EASY,
            title="Conversational baseline run",
            duration=(10.0, 30.0),
            zones=["Z2"],
            structure=[
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
            reasons=[
                "A time-based aerobic sample is needed before distance can be prescribed responsibly."
            ],
            modifications=[
                "Stop for pain, fatigue, and/or elevated heart rate and end the run at a maximum of 30 minutes."
            ],
            confidence=ConfidenceLevel.MODERATE,
            trace=trace,
        )

    goal = configured_race_goal(config, on_date=state.as_of.date())

    rule = RULES["recent_recovery_load"]
    easy_recovery_pressure = (
        min(
            1.0,
            max(
                0.0,
                (recovery.residual_load - EASY_RUN_RESIDUAL_LIMIT) / 0.50,
            ),
        )
        if recovery is not None
        else 0.0
    )
    # Readiness and volume answer different questions. Falling below the
    # easy-run guardrail means an easy run is permissible; it does not mean
    # the latest session has become physiologically free. Keep a smaller,
    # continuous volume adjustment inside the ready range so next-day easy
    # mileage grows back gradually as residual load decays. The volume floor
    # ignores sub-display residue, while a genuinely cheap prior run produces
    # little or no visible change after half-mile rounding.
    easy_volume_recovery_pressure = (
        min(
            1.0,
            max(
                0.0,
                (recovery.residual_load - EASY_VOLUME_RESIDUAL_FLOOR)
                / (
                    EASY_RUN_RESIDUAL_LIMIT
                    - EASY_VOLUME_RESIDUAL_FLOOR
                ),
            ),
        )
        if recovery is not None
        else 0.0
    )
    taxing_recovery_pressure = (
        min(
            1.0,
            max(
                0.0,
                (recovery.residual_load - TAXING_RUN_RESIDUAL_LIMIT) / 0.30,
            ),
        )
        if recovery is not None
        else 0.0
    )
    # Ignore sub-rounding numerical residue in athlete-facing caution copy.
    # The underlying score remains continuous at every value.
    recovery_caution = easy_recovery_pressure >= 0.05
    trace.append(
        _trace(
            rule,
            taxing_recovery_pressure > 0,
            elapsed_hours=(round(recovery.elapsed_hours, 1) if recovery else None),
            initial_relative_load=(round(recovery.initial_load, 3) if recovery else None),
            residual_load=(round(recovery.residual_load, 3) if recovery else None),
            easy_run_limit=EASY_RUN_RESIDUAL_LIMIT,
            hours_until_easy=(round(recovery.hours_until_easy, 1) if recovery else 0.0),
            hours_until_taxing=(round(recovery.hours_until_taxing, 1) if recovery else 0.0),
            easy_recovery_pressure=round(easy_recovery_pressure, 3),
            easy_volume_recovery_pressure=round(
                easy_volume_recovery_pressure, 3
            ),
            taxing_recovery_pressure=round(taxing_recovery_pressure, 3),
            distance_ratio=(round(recovery.distance_ratio, 3) if recovery and recovery.distance_ratio is not None else None),
            duration_ratio=(round(recovery.duration_ratio, 3) if recovery and recovery.duration_ratio is not None else None),
            zone_load_ratio=(round(recovery.zone_load_ratio, 3) if recovery and recovery.zone_load_ratio is not None else None),
        )
    )
    if recovery_caution:
        if health == CurrentHealthStatus.LITTLE_TIRED:
            return _result(
                state,
                workout_type=WorkoutType.REST,
                title="Recovery day",
                reasons=[f"About {recovery.hours_until_easy:.0f} more recovery hours are projected before even an easy run fits normally."],
                warnings=["Same-day recovery data such as sleep and soreness are unavailable."],
                modifications=["Easy walking is optional if it is comfortable; no workout needs to be made up."],
                confidence=ConfidenceLevel.MODERATE,
                readiness=ReadinessFlag.CAUTION,
                trace=trace,
            )
    if recovery is not None and recovery.residual_load > 1.25:
        reduction = max(0.55, 1.0 - 0.30 * easy_recovery_pressure)
        reduced = (
            max(0.1, easy_distance[0] * reduction),
            max(0.1, easy_distance[1] * reduction),
        )
        return _result(
            state,
            workout_type=WorkoutType.RECOVERY,
            title="Short recovery run",
            distance=tuple(max(0.1, round(value * 2) / 2) for value in reduced),
            zones=["Z1", "low Z2"],
            structure=[WorkoutStep(instruction="Keep the entire run conversational; no fast finish.", target_zones=["Z1", "low Z2"])],
            reasons=["The latest session still carries elevated athlete-relative recovery load, so only a short easy run fits."],
            modifications=["Stop early if effort or HR is unusually high for the pace."],
            confidence=ConfidenceLevel.MODERATE,
            readiness=ReadinessFlag.CAUTION,
            trace=trace,
        )

    rule = RULES["high_recent_load"]
    trace.append(
        _trace(
            rule,
            high_load,
            acute_distance_to_capacity_ratio=state.recent_load.acute_distance_to_capacity_ratio,
            raw_hr_load_to_prior_ratio=state.recent_load.acute_to_prior_ratio,
            effective_load_ratio=(round(load_ratio, 3) if load_ratio is not None else None),
            capacity_reference_miles=state.recent_load.capacity_reference_miles,
            capacity_reference_ratio=1.0,
            surplus_strength=round(load_stress, 3),
            full_penalty_ratio=2.0,
        )
    )
    rule = RULES["moderate_leakage"]
    trace.append(
        _trace(
            rule,
            z3_leakage,
            z3_fraction_14d=state.moderate_fraction_14d,
            eligible_easy_runs=state.moderate_evidence_runs_14d,
            minimum_evidence_runs=2,
            threshold=settings["moderate_intensity_leakage_fraction"],
            evidence_strength=round(moderate_leakage_strength, 3),
            uncertainty_band=round(moderate_uncertainty_band, 3),
        )
    )
    high_rpe = bool(
        state.last_run
        and state.last_run.perceived_exertion is not None
        and (state.days_since_last_run is None or state.days_since_last_run <= 3)
        and (
            state.last_run.perceived_exertion >= 9
            or (
                state.last_run_workout_type in {WorkoutType.EASY, WorkoutType.RECOVERY}
                and state.last_run.perceived_exertion >= 7
            )
        )
    )
    trace.append(
        _trace(
            RULES["recent_high_rpe"],
            high_rpe,
            perceived_exertion=(state.last_run.perceived_exertion if state.last_run else None),
            last_workout_type=(state.last_run_workout_type.value if state.last_run_workout_type else None),
        )
    )
    mechanical_flags = set(state.last_run.difficulty_flags if state.last_run else [])
    mechanical_load = bool(
        (state.days_since_last_run is None or state.days_since_last_run <= 2)
        and mechanical_flags & {"hilly_session", "substantial_downhill_load"}
    )
    trace.append(
        _trace(
            RULES["mechanical_load"],
            mechanical_load,
            hilly_session="hilly_session" in mechanical_flags,
            substantial_downhill_load="substantial_downhill_load" in mechanical_flags,
        )
    )
    drift_percent = state.last_run_drift_percent
    raw_drift_strength = (
        min(1.0, max(0.0, (drift_percent - 5.0) / 7.0))
        if drift_percent is not None
        else 0.0
    )
    # Strong whole-run pace-at-HR evidence does not erase genuine drift, but
    # it sharply discounts the claim that drift alone proves a costly run.
    drift_strength = raw_drift_strength * (
        0.25
        if state.recent_performance_response == "stronger_than_recent"
        else 1.0
    )
    response_applies_to_last = (
        state.recent_performance_response_activity_id is None
        or state.last_run_activity_id is None
        or state.recent_performance_response_activity_id
        == state.last_run_activity_id
    )
    performance_response_strength = (
        0.75
        if state.recent_performance_response == "higher_cost_than_recent"
        and response_applies_to_last
        else 0.0
    )
    rpe_strength = 1.0 if high_rpe else 0.0
    # Pace-at-HR response already changes the transient recovery residue. It is
    # intentionally absent here: applying it again to distance would make one
    # observation delay the next run and independently shrink it. Drift and RPE
    # remain direct execution evidence and can shape the next prescription.
    response_stress = 1.0 - (
        (1.0 - drift_strength)
        * (1.0 - rpe_strength)
    )
    costly = response_stress >= 0.5
    drift_caution = 0.0 < response_stress < 0.5
    rule = RULES["recent_costly_response"]
    trace.append(
        _trace(
            rule,
            costly,
            performance_response=state.recent_performance_response,
            performance_response_activity_id=(
                state.recent_performance_response_activity_id
            ),
            last_run_activity_id=state.last_run_activity_id,
            response_applies_to_last=response_applies_to_last,
            performance_response_accounted_in_recovery=(
                performance_response_strength > 0
            ),
            drift_percent=drift_percent,
            drift_excess_strength=round(drift_strength, 3),
            performance_strength=performance_response_strength,
            rpe_strength=rpe_strength,
            response_stress=round(response_stress, 3),
            maximum_volume_reduction=(
                1.0 - float(settings["reduced_volume_factor"])
            ),
        )
    )
    trace.append(
        _trace(
            RULES["recent_drift_caution"],
            drift_caution,
            drift_percent=drift_percent,
            performance_response=state.recent_performance_response,
            response_stress=round(response_stress, 3),
            quality_penalty_points=round(3.0 * response_stress, 2),
            volume_reduction_fraction=round(
                response_stress
                * (1.0 - float(settings["reduced_volume_factor"])),
                3,
            ),
        )
    )
    consistency_reference = max(
        1.0,
        float(settings["minimum_running_days_28d_for_quality"]),
    )
    consistency_strength = min(
        1.0,
        max(0.0, state.running_days_28d / consistency_reference),
    )
    consistency_gap = 1.0 - consistency_strength
    sparse = consistency_gap > 1e-9
    rule = RULES["returning_consistency"]
    trace.append(
        _trace(
            rule,
            sparse,
            running_days_28d=state.running_days_28d,
            reference_run_days_28d=consistency_reference,
            established_consistency_strength=round(consistency_strength, 3),
        )
    )
    post_illness_check = False
    trace.append(
        _trace(
            RULES["post_illness_quality_check"],
            post_illness_check,
            current_health_status=health.value,
            historical_health_disruption=state.recent_illness_or_recovery,
            hidden_normal_run_requirement=0,
        )
    )

    scores = {"easy": 2.0, "long": 1.0, "quality": 0.0}
    evidence: dict[str, list[str]] = {
        "easy": ["ordinary aerobic default +2"],
        "long": ["durability-development option +1"],
        "quality": [],
    }

    def add(candidate: str, value: float, reason: str) -> None:
        scores[candidate] += value
        evidence[candidate].append(f"{reason} {value:+g}")

    if taxing_recovery_pressure > 0:
        add("easy", 2.0 * taxing_recovery_pressure, "remaining recovery load")
        add("long", -3.0 * taxing_recovery_pressure, "remaining recovery load")
        add("quality", -4.0 * taxing_recovery_pressure, "remaining recovery load")

    days_to_goal = None
    tapering = False
    if goal:
        goal_profile, goal_date, goal_pace = goal
        days_to_goal = (goal_date - state.as_of.date()).days
        tapering = 0 < days_to_goal <= goal_profile.taper_days
        if tapering:
            add("easy", 2.5, f"{goal_profile.label} taper")
            add("long", -4, f"{goal_profile.label} taper")
            add("quality", -1.5, f"{goal_profile.label} taper")
        else:
            add("long", goal_profile.long_run_bias, f"{goal_profile.label} goal")
            add("quality", goal_profile.quality_bias, f"{goal_profile.label} goal")
        trace.append(
            _trace(
                RULES["race_goal"],
                True,
                goal=goal_profile.label,
                goal_date=goal_date.isoformat(),
                goal_pace_min_mile=goal_pace,
                days_remaining=days_to_goal,
                tapering=tapering,
                long_run_bias=goal_profile.long_run_bias,
                quality_bias=goal_profile.quality_bias,
            )
        )
    else:
        trace.append(_trace(RULES["race_goal"], False, goal="general_fitness"))

    if goal and days_to_goal == 0:
        race_caution = high_load or costly or health == CurrentHealthStatus.LITTLE_TIRED or environmental_caution
        finish_minutes = goal_profile.distance_miles * goal_pace
        return _result(
            state,
            workout_type=WorkoutType.RACE,
            title=f"{goal_profile.label} goal race",
            distance=(goal_profile.distance_miles, goal_profile.distance_miles),
            duration=(finish_minutes, finish_minutes),
            zones=["race effort"],
            structure=[
                WorkoutStep(instruction="Warm up gradually and reassess current health and effort before starting."),
                WorkoutStep(instruction=f"Target approximately {format_pace(goal_pace)} only while effort remains controlled.", distance_miles=goal_profile.distance_miles, target_zones=["race effort"]),
            ],
            reasons=[f"Today matches the validated {goal_profile.label} goal date."],
            warnings=["Goal pace is a target, not a requirement; conditions and current symptoms take precedence."],
            modifications=["Do not race through pain, concerning respiratory symptoms, or clearly abnormal warm-up effort."],
            confidence=ConfidenceLevel.MODERATE,
            readiness=ReadinessFlag.CAUTION if race_caution else ReadinessFlag.READY,
            readiness_reason=(
                "A load, recent-response, tiredness, or weather caution is present on race day; reassess rather than forcing goal pace."
                if race_caution
                else "The validated goal date is today and current health guardrails do not block racing."
            ),
            trace=trace,
        )

    if load_stress > 0:
        add("easy", 3 * load_stress, "graded acute-load surplus")
        add("long", -3 * load_stress, "graded acute-load surplus")
        add("quality", -4 * load_stress, "graded acute-load surplus")
    if moderate_leakage_strength > 0:
        add("easy", 2 * moderate_leakage_strength, "graded excess Z3")
        add("quality", -2 * moderate_leakage_strength, "graded excess Z3")
    if response_stress > 0:
        add("easy", 2 * response_stress, "graded recent-response cost")
        add("long", -2 * response_stress, "graded recent-response cost")
        add("quality", -3 * response_stress, "graded recent-response cost")
    if mechanical_load:
        add("easy", 1, "recent mechanical load")
        add("long", -1, "recent mechanical load")
        add("quality", -2, "recent mechanical load")
    if consistency_gap > 0:
        add("easy", 1.5 * consistency_gap, "limited recent consistency")
    add(
        "long",
        consistency_strength - 1.5 * consistency_gap,
        "continuous 28-day consistency",
    )
    add(
        "quality",
        2.0 * consistency_strength - 3.0 * consistency_gap,
        "continuous 28-day consistency",
    )
    if post_illness_check:
        add("easy", 3, "post-illness aerobic check")
        add("long", -3, "readiness check should be ordinary distance")
        add("quality", -5, "awaiting one more normal aerobic response")
    if health == CurrentHealthStatus.LITTLE_TIRED:
        add("easy", 2, "reported tiredness")
        add("long", -2, "reported tiredness")
        add("quality", -3, "reported tiredness")
    if weather_stress.score > 0:
        # Mild stress adapts the prescription without erasing the planned
        # workout. Only genuinely high stress strongly redirects long or
        # quality work, and every transition is continuous.
        severe_excess = max(0.0, (weather_stress.score - 0.75) / 0.75)
        add(
            "easy",
            0.50 * weather_stress.score + 3.0 * severe_excess,
            f"graded {weather_stress.band} weather stress",
        )
        add(
            "long",
            -0.25 * weather_stress.score - 3.0 * severe_excess,
            f"graded {weather_stress.band} weather stress",
        )
        add(
            "quality",
            -0.75 * weather_stress.score - 4.0 * severe_excess,
            f"graded {weather_stress.band} weather stress",
        )

    recency_reference = float(settings["long_run_recency_reference_days"])
    if state.days_since_long_run is None or state.days_since_long_run >= recency_reference:
        add("long", 2.0, "rolling long-run recency pressure")
        add("quality", -1, "long-run recency opportunity cost")
    elif state.days_since_long_run >= max(0.0, recency_reference - 1.0):
        add("long", 0.5, "approaching long-run target spacing")
    else:
        add("long", -1, "recent long run")
    progression_reference = single_session_progression_reference_miles(state)
    if progression_reference <= 0:
        add("long", -4, "no recent long-run baseline")
    easy_midpoint = sum(easy_distance) / 2
    meaningful_long_threshold = max(
        easy_distance[1],
        round(easy_midpoint * 1.15, 1),
    )
    progression_ceiling = (
        round(
            progression_reference
            * float(settings["long_run_progression_factor"])
            * 2
        )
        / 2
    )
    ordinary_progression_fraction = float(
        settings["long_run_target_progression_fraction"]
    )
    maximum_progression_fraction = max(
        0.0,
        float(settings["long_run_progression_factor"]) - 1.0,
    )
    retained_return_gap_fraction = (
        max(
            0.0,
            state.retained_long_run_capacity_miles / progression_reference - 1.0,
        )
        if progression_reference > 0
        else 0.0
    )
    target_progression_fraction = min(
        maximum_progression_fraction,
        max(
            ordinary_progression_fraction,
            retained_return_gap_fraction * 0.5,
        ),
    )
    goal_required_long_progression = 0.0
    if goal and not tapering:
        goal_required_long_progression = required_compound_progression(
            progression_reference,
            goal_profile.peak_long_run_miles,
            build_weeks_before_taper(
                goal_profile,
                goal_date,
                state.as_of.date(),
            ),
        )
        target_progression_fraction = min(
            maximum_progression_fraction,
            max(target_progression_fraction, goal_required_long_progression),
        )
    returning_to_retained_capacity = (
        retained_return_gap_fraction >= ordinary_progression_fraction
    )
    if (
        returning_to_retained_capacity
        and state.days_since_long_run is not None
        and state.days_since_long_run >= 7.0
    ):
        add("long", 1.0, "return toward retained long-run capacity")
        add("quality", -0.5, "return-to-capacity opportunity cost")
    if progression_ceiling < meaningful_long_threshold:
        add(
            "long",
            -5,
            "progression ceiling does not support a run meaningfully longer than ordinary easy distance",
        )

    quality_recency_reference = float(settings["quality_recency_reference_days"])
    quality_recency_strength = (
        1.0
        if state.days_since_quality_run is None
        else min(
            1.0,
            max(
                0.0,
                state.days_since_quality_run / quality_recency_reference,
            ),
        )
    )
    expected_quality_sessions_14d = 14.0 / quality_recency_reference
    # A recorded recency inside this same window is itself evidence of one
    # quality session. Keeping the two signals internally consistent prevents
    # a partially populated state from reading as both "yesterday" and
    # "zero recent quality," which would manufacture maximum urgency.
    effective_quality_sessions_14d = max(
        state.quality_sessions_14d,
        1
        if state.days_since_quality_run is not None
        and state.days_since_quality_run <= 14.0
        else 0,
    )
    recent_quality_saturation = min(
        1.0,
        effective_quality_sessions_14d / expected_quality_sessions_14d,
    )
    # A recent dose satisfies quality need, but that satisfaction fades as the
    # most recent session ages. A raw 14-day count is otherwise a boxcar: a
    # workout 12 days ago suppresses priority exactly as much as yesterday's.
    remaining_quality_satisfaction = recent_quality_saturation * (
        1.0 - quality_recency_strength
    )
    quality_need = quality_recency_strength * (
        1.0 - remaining_quality_satisfaction
    )
    add(
        "quality",
        -3.0 + 4.0 * quality_need,
        "continuous quality recency and recent-dose need",
    )
    if state.recent_performance_response == "within_recent_range":
        add("long", 0.5, "normal recent response")
        add("quality", 1, "normal recent response")
    if state.fitness_trend.value == "improving" and state.trend_confidence in {
        ConfidenceLevel.HIGH,
        ConfidenceLevel.MODERATE,
    }:
        add("long", 0.5, "improving efficiency signal")
        add("quality", 1.5, "improving efficiency signal")
    sequence_preference_points = 1.0
    if weekly_role in scores:
        add(weekly_role, sequence_preference_points, "coordinated weekly sequence")
    trace.append(
        _trace(
            RULES["weekly_sequence_priority"],
            weekly_role in scores,
            preferred_role=weekly_role,
            preference_points=(sequence_preference_points if weekly_role in scores else 0),
        )
    )

    tie_priority = {"easy": 2, "long": 1, "quality": 0}
    available = [
        candidate
        for candidate in scores
        if allowed_candidates is None or candidate in allowed_candidates
    ]
    if not available:
        available = ["easy"]
    selected = max(
        available,
        key=lambda candidate: (scores[candidate], tie_priority[candidate]),
    )
    trace.append(
        _trace(
            RULES["workout_scoring"],
            True,
            selected=selected,
            easy_score=round(scores["easy"], 2),
            long_score=round(scores["long"], 2),
            quality_score=round(scores["quality"], 2),
            allowed_candidates=(
                ", ".join(sorted(allowed_candidates))
                if allowed_candidates is not None
                else "easy, long, quality"
            ),
            easy_evidence="; ".join(evidence["easy"]),
            long_evidence="; ".join(evidence["long"]),
            quality_evidence="; ".join(evidence["quality"]),
        )
    )
    trace.append(
        _trace(
            RULES["long_run_eligible"],
            selected == "long",
            score=round(scores["long"], 2),
            days_since_long_run=state.days_since_long_run,
            recency_reference_days=recency_reference,
            meaningful_long_threshold_miles=meaningful_long_threshold,
            progression_ceiling_miles=progression_ceiling,
            retained_long_run_capacity_miles=round(
                state.retained_long_run_capacity_miles, 2
            ),
            target_progression_fraction=round(
                target_progression_fraction, 3
            ),
            goal_required_long_progression_fraction=(
                round(goal_required_long_progression, 3)
                if goal and not tapering
                else 0.0
            ),
        )
    )
    trace.append(
        _trace(
            RULES["quality_eligible"],
            selected == "quality",
            score=round(scores["quality"], 2),
            days_since_quality=state.days_since_quality_run,
            quality_recency_reference_days=quality_recency_reference,
            quality_recency_strength=round(quality_recency_strength, 3),
            recent_quality_saturation=round(recent_quality_saturation, 3),
            remaining_quality_satisfaction=round(
                remaining_quality_satisfaction, 3
            ),
            quality_need=round(quality_need, 3),
            quality_sessions_14d=state.quality_sessions_14d,
            effective_quality_sessions_14d=effective_quality_sessions_14d,
            running_days_28d=state.running_days_28d,
            longest_run_is_gate=False,
        )
    )
    trace.append(
        _trace(
            RULES["default_aerobic"],
            selected == "easy",
            score=round(scores["easy"], 2),
            typical_distance_low=easy_distance[0],
            typical_distance_high=easy_distance[1],
        )
    )

    if selected == "long":
        cap = progression_reference * float(settings["long_run_progression_factor"])
        weekly_norm = max(
            state.recent_load.trailing_28d.distance_miles / 4,
            state.recent_load.capacity_reference_miles or 0,
        )
        lower, upper, capped_by_progression = _long_run_distance(
            cap,
            weekly_norm,
            progression_reference,
            state.running_days_28d / 4.0,
            target_progression_fraction,
            returning_to_retained_capacity,
        )
        long_distance = _weather_scaled_distance(
            (lower, upper), weather_stress.score
        )
        warnings = ["The target is proportional to maintained recent distance and rounded to a practical quarter mile; the separate progression limit is a warning, not a hard safety line."]
        if capped_by_progression:
            warnings.append(
                f"Your current single-run progression reference is {progression_reference:.1f} miles, "
                f"so the progression limit is about {cap:.1f} miles and this long run follows that "
                "rather than a conventional distance. Half-mile rounding may put the upper end "
                "level with the limit."
            )
        if environmental_adaptation:
            warnings.append(
                f"The forecast adds {weather_stress.band} environmental stress relative to recent training exposure; ordinary rain does not count against the workout."
            )
        modifications = [
            "Cut the run to ordinary easy distance if fatigue, pain, or illness symptoms appear."
        ]
        if environmental_adaptation:
            modifications.append(
                "Run by easy effort rather than normal-weather pace and use the lower end if conditions feel more costly than forecast."
            )
        return _result(
            state,
            workout_type=WorkoutType.LONG,
            title="Easy long run",
            distance=long_distance,
            zones=["Z1", "Z2"],
            structure=[
                WorkoutStep(instruction="First 10 minutes relaxed in Z1 or low Z2.", duration_minutes=10, target_zones=["Z1", "low Z2"]),
                WorkoutStep(instruction="Remain primarily Z2; no planned fast finish.", target_zones=["Z2"]),
            ],
            reasons=["A long run fits your recent mileage, recovery, and time since the last one.", "The distance stays close to your longest run in the past 30 days."],
            warnings=warnings,
            modifications=modifications,
            confidence=ConfidenceLevel.MODERATE,
            readiness=(
                ReadinessFlag.CAUTION
                if environmental_caution
                else ReadinessFlag.READY
            ),
            readiness_reason=(
                "The workout still fits, but the graded forecast stress calls for effort-based pacing and flexible distance."
                if environmental_caution
                else None
            ),
            trace=trace,
        )

    if selected == "quality":
        quality_kind = _select_quality_variant(state, settings)
        prescription = _quality_prescription(quality_kind)
        structure = list(prescription["structure"])
        quality_distance = _quality_structure_distance(
            state,
            quality_kind,
            structure,
            prescription["distance"],
        )
        if goal:
            structure.append(_goal_quality_context(quality_kind, goal_profile.label, goal_pace))
        enabled_quality = [
            name
            for name, enabled in (settings.get("quality_sessions") or {}).items()
            if enabled
        ]
        trace.append(
            _trace(
                RULES["quality_variant"],
                True,
                selected_variant=quality_kind.value,
                enabled_variants=", ".join(enabled_quality),
                completed_quality_session_count=state.completed_quality_session_count,
                days_since_quality_run=state.days_since_quality_run,
                adaptable_return_from_quality_gap=(
                    state.days_since_quality_run is None
                    or state.days_since_quality_run >= 21
                ),
            )
        )
        quality_warnings = [
            "Historical workout pace is context only; today’s prescription is controlled effort, not a fixed split mandate."
        ]
        quality_modifications = [
            "Follow the single prescribed structure, but use effort rather than pace to adapt it to the route and conditions.",
            "Convert to an ordinary easy run if warm-up HR/effort is abnormal or any pain appears.",
        ]
        if environmental_adaptation:
            quality_warnings.append(
                f"The planned forecast adds {weather_stress.band} graded environmental stress to this workout."
            )
            quality_modifications.append(
                "Use effort rather than normal-weather pace; if conditions feel oppressive during the warm-up, replace the quality work with ordinary easy running."
            )
        return _result(
            state,
            workout_type=prescription["workout_type"],
            quality_session_type=quality_kind,
            title=prescription["title"],
            distance=quality_distance,
            zones=prescription["zones"],
            structure=structure,
            reasons=[
                "Recent training, recovery, and workout spacing support a harder session.",
                f"{quality_kind.value.replace('_', ' ').title()} provides variety while matching the current quality stimulus and enabled preferences.",
            ],
            warnings=quality_warnings,
            modifications=quality_modifications,
            confidence=ConfidenceLevel.MODERATE,
            readiness=(ReadinessFlag.CAUTION if environmental_caution else ReadinessFlag.READY),
            trace=trace,
        )

    response_factor = 1.0 - response_stress * (
        1.0 - float(settings["reduced_volume_factor"])
    )
    load_factor = 1.0 - load_stress * (
        1.0 - float(settings["reduced_volume_factor"])
    )
    factor = min(
        load_factor,
        response_factor,
        1.0 - 0.25 * easy_volume_recovery_pressure,
    )
    if tapering:
        factor = min(factor, 0.80)
    factor *= max(0.70, 1.0 - 0.12 * weather_stress.score)
    prescribed_distance = tuple(
        max(0.1, round(value * factor * 2) / 2) for value in easy_distance
    )
    cautions = high_load or z3_leakage or response_stress > 0 or recovery_caution or sparse or health == CurrentHealthStatus.LITTLE_TIRED or environmental_caution
    reasons = ["An easy run best fits your recent training and recovery."]
    caution_explanations: list[str] = []
    if high_load: reasons.append(f"Your projected recent load is about {load_ratio * 100:.0f}% of your usual weekly capacity.")
    if high_load:
        load_reduction_percent = (1.0 - load_factor) * 100
        caution_explanations.append(
            f"Load is above the reference, so ordinary distance is reduced by about {load_reduction_percent:.0f}% before rounding; the reduction grows with the surplus."
        )
    if z3_leakage:
        z3_percent = state.moderate_fraction_14d * 100
        z3_threshold_percent = float(settings["moderate_intensity_leakage_fraction"]) * 100
        reasons.append(f"{z3_percent:.0f}% of recorded HR time in the last 14 days was moderate intensity.")
        caution_explanations.append(
            f"Moderate effort was {z3_percent:.0f}% of recent HR time, above the {z3_threshold_percent:.0f}% easy-running reference. Keep this run truly easy."
        )
    if response_stress > 0:
        reduction_percent = (1.0 - response_factor) * 100
        reasons.append(
            f"The latest response adds {response_stress * 100:.0f}% of the maximum recent-response penalty."
        )
        caution_explanations.append(
            f"Recent-response evidence reduces ordinary distance by about {reduction_percent:.0f}% before rounding; the adjustment scales with the excess rather than switching on all at once."
        )
    if easy_volume_recovery_pressure >= 0.05 and recovery is not None:
        recovery_reduction_percent = easy_volume_recovery_pressure * 25
        reasons.append(
            f"The latest session's remaining recovery load reduces ordinary distance by about {recovery_reduction_percent:.0f}% before rounding."
        )
        if recovery_caution:
            caution_explanations.append(
                f"Recovery load is still above the easy-run reference and is projected to decay into the normal range in about {recovery.hours_until_easy:.0f} hours."
            )
    if sparse:
        reasons.append("Recent running has been less consistent than usual.")
        caution_explanations.append("Recent consistency is below the quality-work reference, so this remains an easy session.")
    if health == CurrentHealthStatus.LITTLE_TIRED:
        caution_explanations.append("Current status is a little tired, so ordinary easy running is favored over added stress.")
    if environmental_caution:
        caution_explanations.append("The graded forecast stress is meaningful, so pace and distance should remain flexible.")
    elif environmental_adaptation:
        reasons.append(
            f"The forecast adds {weather_stress.band} environmental stress, so the distance and pacing guidance are adjusted gradually."
        )
    if post_illness_check: reasons.append("This is the final easy aerobic check before reintroducing quality, provided the response is normal.")
    easy_modifications = ["Keep the run easy or shorten it if today’s HR/effort is unusually high."]
    if post_illness_check:
        easy_modifications.append(
            "If respiratory symptoms meaningfully return during or after this run, replace the planned quality session with an easy run."
        )
    if environmental_adaptation:
        easy_modifications.append(
            "Use easy effort rather than normal-weather pace and shorten only if conditions feel more costly than forecast."
        )
    return _result(
        state,
        workout_type=WorkoutType.EASY,
        title="Deliberately easy aerobic run" if cautions else "Easy aerobic run",
        distance=prescribed_distance,
        zones=["Z1", "Z2"],
        structure=[
            WorkoutStep(instruction="First 10 minutes in Z1 or low Z2.", duration_minutes=10, target_zones=["Z1", "low Z2"]),
            WorkoutStep(instruction=f"Then stay primarily {_zone_copy(config, 'z2')}; no fast finish.", target_zones=["Z2"]),
        ],
        reasons=reasons,
        warnings=["Sleep, soreness, stress, hydration, and unrecorded activity are unavailable."],
        modifications=easy_modifications,
        confidence=(ConfidenceLevel.LOW if "latest_run_pace_quality_low" in state.data_quality_flags else ConfidenceLevel.MODERATE),
        readiness=ReadinessFlag.CAUTION if cautions else ReadinessFlag.READY,
        readiness_reason=(
            " ".join(caution_explanations)
            if caution_explanations
            else "Current health, recovery spacing, load, and forecast checks allow this workout."
        ),
        trace=trace,
    )
