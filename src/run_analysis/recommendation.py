"""Deterministic next-run recommendation engine with inspectable rules.

No database or frontend code belongs here. The engine accepts a `FitnessState`
and configuration, evaluates named rules in a fixed order, and returns both a
prescription and the complete rule trace.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .race_goals import RACE_GOALS, configured_race_goal, format_pace
from .web.schemas import (
    ConfidenceLevel,
    CurrentHealthStatus,
    FitnessState,
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
    RuleDefinition("planned_weather", "Forecast heat, humidity, wind, and precipitation can make a future workout more costly."),
    RuleDefinition("health_pain", "Pain or injury concern blocks a running prescription."),
    RuleDefinition("health_sick", "Sick/recovering limits training to reduced recovery work; explicit active or concerning symptoms still block running."),
    RuleDefinition("long_or_hard_yesterday", "A long/hard run within 36 hours favors rest or short recovery."),
    RuleDefinition("high_recent_load", "A high 7-day mileage load relative to retained demonstrated capacity blocks added volume or quality."),
    RuleDefinition("moderate_leakage", "Excess recent Z3 time redirects the next run to deliberate Z1/Z2."),
    RuleDefinition("recent_costly_response", "Unusually costly performance or high drift blocks quality."),
    RuleDefinition("recent_high_rpe", "A high reported effort adds recovery caution without replacing recorded HR load."),
    RuleDefinition("mechanical_load", "Available elevation data can identify hilly or downhill-heavy mechanical stress."),
    RuleDefinition("returning_consistency", "Sparse recent running favors rebuilding routine before quality."),
    RuleDefinition("post_illness_quality_check", "Sick/recovering remains blocked until the current self-report returns to normal; no hidden clearance stage is added afterward."),
    RuleDefinition("quality_variant", "Enabled quality-session types rotate deterministically, with short intervals used after a long quality gap."),
    RuleDefinition("race_goal", "A validated race goal changes workout composition and taper priority without overriding health or load guardrails."),
    RuleDefinition("weekly_sequence_priority", "The seven-day planner can coordinate earlier sessions, but the final session receives no workout-type bonus merely for ending the horizon."),
    RuleDefinition("workout_scoring", "Easy, long, and quality candidates receive additive evidence scores."),
    RuleDefinition("long_run_eligible", "Long-run recency raises priority but does not independently prescribe one."),
    RuleDefinition("quality_eligible", "Quality readiness depends on several signals; longest-run distance is not a gate."),
    RuleDefinition("default_aerobic", "When no higher-priority rule fires, prescribe ordinary aerobic running."),
)

RULES = {rule.rule_id: rule for rule in RULE_CATALOG}


def _settings(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "long_run_progression_factor": 1.10,
        "high_load_ratio": 1.30,
        "moderate_intensity_leakage_fraction": 0.17,
        "minimum_days_between_quality_sessions": 4,
        "quality_recency_reference_days": 7,
        "typical_rest_days_between_runs": 1,
        "capacity_retention_half_life_days": 42,
        "capacity_retention_grace_days": 28,
        "minimum_running_days_28d_for_quality": 8,
        "long_run_recency_reference_days": 10,
        "reduced_volume_factor": 0.70,
        "quality_sessions": {
            "short_intervals": True,
            "long_intervals": True,
            "threshold": True,
            "progression": True,
            "hill_repeats": False,
        },
        **config.get("coaching", {}),
    }


def typical_easy_distance(state: FitnessState) -> tuple[float, float]:
    load = state.recent_load.trailing_28d
    if load.activity_count <= 0 or load.distance_miles <= 0:
        return (3.0, 4.0)
    average = load.distance_miles / load.activity_count
    lower = max(2.0, round((average * 0.85) * 2) / 2)
    upper = lower + 0.5
    durability_cap = max(3.0, state.longest_run_30d_miles * 0.8)
    return (min(lower, durability_cap), min(upper, durability_cap))


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


def _select_quality_variant(
    state: FitnessState, settings: dict[str, Any]
) -> QualitySessionType:
    configured = settings.get("quality_sessions") or {}
    ordinary_order = [
        QualitySessionType.SHORT_INTERVALS,
        QualitySessionType.THRESHOLD,
        QualitySessionType.LONG_INTERVALS,
        QualitySessionType.PROGRESSION,
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
        return QualitySessionType.SHORT_INTERVALS
    if (
        QualitySessionType.SHORT_INTERVALS in enabled
        and (state.days_since_quality_run is None or state.days_since_quality_run >= 21)
    ):
        return QualitySessionType.SHORT_INTERVALS
    return enabled[state.completed_quality_session_count % len(enabled)]


def _quality_prescription(kind: QualitySessionType) -> dict[str, Any]:
    prescriptions: dict[QualitySessionType, dict[str, Any]] = {
        QualitySessionType.SHORT_INTERVALS: {
            "workout_type": WorkoutType.INTERVALS,
            "title": "Controlled 400 m intervals",
            "distance": (4.0, 4.5),
            "zones": ["Z1", "Z2", "Z4 effort"],
            "structure": [
                WorkoutStep(instruction="Easy warm-up.", duration_minutes=12, target_zones=["Z1", "Z2"]),
                WorkoutStep(instruction="6 × 400 m controlled fast, each followed by 200 m slow jog. Do not sprint or chase HR lag.", distance_miles=2.25, target_zones=["Z4 effort"]),
                WorkoutStep(instruction="Easy cool-down.", duration_minutes=10, target_zones=["Z1", "Z2"]),
            ],
        },
        QualitySessionType.LONG_INTERVALS: {
            "workout_type": WorkoutType.INTERVALS,
            "title": "Controlled 800 m intervals",
            "distance": (4.5, 5.0),
            "zones": ["Z1", "Z2", "Z4 effort"],
            "structure": [
                WorkoutStep(instruction="Easy warm-up.", duration_minutes=12, target_zones=["Z1", "Z2"]),
                WorkoutStep(instruction="4 × 800 m at controlled hard effort, each followed by 400 m easy jog. Keep the final rep consistent.", distance_miles=3.0, target_zones=["Z4 effort"]),
                WorkoutStep(instruction="Easy cool-down.", duration_minutes=10, target_zones=["Z1", "Z2"]),
            ],
        },
        QualitySessionType.THRESHOLD: {
            "workout_type": WorkoutType.TEMPO_THRESHOLD,
            "title": "Cruise threshold intervals",
            "distance": (4.5, 5.0),
            "zones": ["Z1", "Z2", "upper Z3 / low Z4"],
            "structure": [
                WorkoutStep(instruction="Easy warm-up.", duration_minutes=12, target_zones=["Z1", "Z2"]),
                WorkoutStep(instruction="3 × 6 minutes at controlled threshold effort with 2 minutes easy jog between. Finish able to repeat the final interval.", duration_minutes=22, target_zones=["upper Z3", "low Z4"]),
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
                WorkoutStep(instruction="Gradually increase through the second half, finishing controlled in Z3 without sprinting.", target_zones=["Z2", "Z3"]),
            ],
        },
        QualitySessionType.HILL_REPEATS: {
            "workout_type": WorkoutType.INTERVALS,
            "title": "Short hill repeats",
            "distance": (4.0, 4.5),
            "zones": ["Z1", "Z2", "strong controlled effort"],
            "structure": [
                WorkoutStep(instruction="Easy warm-up on flat terrain.", duration_minutes=12, target_zones=["Z1", "Z2"]),
                WorkoutStep(instruction="8 × 45 seconds uphill at strong controlled effort; jog gently downhill and fully regain control.", duration_minutes=16, target_zones=["strong controlled effort"]),
                WorkoutStep(instruction="Easy cool-down.", duration_minutes=10, target_zones=["Z1", "Z2"]),
            ],
        },
    }
    return prescriptions[kind]


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
) -> RecommendationResponse:
    """Evaluate guardrails, then long/quality eligibility, then easy default."""

    settings = _settings(config)
    trace: list[RuleTrace] = []
    health = request.health_status
    easy_distance = typical_easy_distance(state)
    load_ratio = (
        state.recent_load.acute_distance_to_capacity_ratio
        if state.recent_load.acute_distance_to_capacity_ratio is not None
        else state.recent_load.acute_to_prior_ratio
    )
    high_load = load_ratio is not None and load_ratio >= float(settings["high_load_ratio"])
    z3_leakage = (
        state.moderate_fraction_14d is not None
        and state.moderate_evidence_runs_14d >= 2
        and state.moderate_fraction_14d >= float(settings["moderate_intensity_leakage_fraction"])
    )
    notes = request.notes.casefold()
    pain_note = any(phrase in notes for phrase in ("sharp pain", "chest pain", "injury concern"))
    sickness_note = any(phrase in notes for phrase in ("shortness of breath", "fever", "actively sick"))

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
    hot_or_humid = bool(
        weather
        and (
            (weather.apparent_temperature_f is not None and weather.apparent_temperature_f >= 85)
            or (weather.temperature_f is not None and weather.temperature_f >= 85)
            or (weather.dewpoint_f is not None and weather.dewpoint_f >= 70)
        )
    )
    strong_wind = bool(
        weather
        and (
            (weather.wind_speed_mph is not None and weather.wind_speed_mph >= 20)
            or (weather.wind_gust_mph is not None and weather.wind_gust_mph >= 30)
        )
    )
    environmental_caution = hot_or_humid or strong_wind
    trace.append(
        _trace(
            RULES["planned_weather"],
            environmental_caution,
            forecast_available=weather is not None,
            temperature_f=weather.temperature_f if weather else None,
            apparent_temperature_f=weather.apparent_temperature_f if weather else None,
            dewpoint_f=weather.dewpoint_f if weather else None,
            wind_speed_mph=weather.wind_speed_mph if weather else None,
        )
    )

    rule = RULES["health_pain"]
    fired = health == CurrentHealthStatus.PAIN_OR_INJURY_CONCERN or pain_note
    trace.append(_trace(rule, fired, health_status=health.value, concerning_note=pain_note))
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
    trace.append(_trace(rule, recovering_mode or sickness_note, health_status=health.value, concerning_note=sickness_note, recent_illness=state.recent_illness_or_recovery))
    if sickness_note:
        return _result(
            state,
            workout_type=WorkoutType.REST,
            title="Recovery day",
            reasons=["Your notes indicate active or concerning illness symptoms, which override training-load signals."],
            warnings=["Resume with reduced easy volume only when symptoms and ordinary daily activity are clearly tolerated."],
            modifications=["If you feel normal later, request a new recommendation rather than using this stale one."],
            confidence=ConfidenceLevel.HIGH,
            readiness=ReadinessFlag.NOT_READY,
            trace=trace,
        )
    if recovering_mode:
        recovery_distance = (
            round(max(1.5, easy_distance[0] * 0.5) * 2) / 2,
            round(max(2.0, easy_distance[1] * 0.6) * 2) / 2,
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
            warnings=["This is not medical clearance. Fever, chest pain, unusual shortness of breath, or worsening symptoms mean do not run."],
            modifications=["Change the current status to Normal when you are ready for ordinary load and workout scoring."],
            confidence=ConfidenceLevel.MODERATE,
            readiness=ReadinessFlag.CAUTION,
            trace=trace,
        )

    goal = configured_race_goal(config, on_date=state.as_of.date())

    rule = RULES["long_or_hard_yesterday"]
    recent = state.days_since_last_run is not None and state.days_since_last_run < 1.5
    taxing_last = bool(state.last_run and (state.last_run.is_long_run or state.last_run.is_quality_session))
    fired = recent and taxing_last
    trace.append(_trace(rule, fired, days_since_last_run=state.days_since_last_run, last_run_taxing=taxing_last))
    if fired:
        if health == CurrentHealthStatus.LITTLE_TIRED or (state.days_since_last_run or 0) < 0.75:
            return _result(
                state,
                workout_type=WorkoutType.REST,
                title="Recovery day",
                reasons=["The last run was long or hard and occurred within the last 36 hours."],
                warnings=["Same-day recovery data such as sleep and soreness are unavailable."],
                modifications=["Easy walking is optional if it is comfortable; no workout needs to be made up."],
                confidence=ConfidenceLevel.MODERATE,
                readiness=ReadinessFlag.CAUTION,
                trace=trace,
            )
        reduced = (max(2.0, easy_distance[0] * 0.6), max(2.5, easy_distance[1] * 0.7))
        return _result(
            state,
            workout_type=WorkoutType.RECOVERY,
            title="Short recovery run",
            distance=tuple(round(value * 2) / 2 for value in reduced),
            zones=["Z1", "low Z2"],
            structure=[WorkoutStep(instruction="Keep the entire run conversational; no fast finish.", target_zones=["Z1", "low Z2"])],
            reasons=["The last run was long or hard and occurred within the last 36 hours."],
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
            capacity_reference_miles=state.recent_load.capacity_reference_miles,
            threshold=settings["high_load_ratio"],
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
    costly = high_rpe or state.recent_performance_anomaly == "unusually_costly" or (
        state.last_run_drift_percent is not None and state.last_run_drift_percent > 5
    )
    rule = RULES["recent_costly_response"]
    trace.append(_trace(rule, costly, performance_anomaly=state.recent_performance_anomaly, drift_percent=state.last_run_drift_percent))
    sparse = state.running_days_28d < int(settings["minimum_running_days_28d_for_quality"])
    rule = RULES["returning_consistency"]
    trace.append(_trace(rule, sparse, running_days_28d=state.running_days_28d, minimum=settings["minimum_running_days_28d_for_quality"]))
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

    if high_load:
        add("easy", 3, "high acute load")
        add("long", -3, "high acute load")
        add("quality", -4, "high acute load")
    if z3_leakage:
        add("easy", 2, "excess Z3")
        add("quality", -2, "excess Z3")
    if costly:
        add("easy", 2, "costly recent response")
        add("long", -2, "costly recent response")
        add("quality", -3, "costly recent response")
    if mechanical_load:
        add("easy", 1, "recent mechanical load")
        add("long", -1, "recent mechanical load")
        add("quality", -2, "recent mechanical load")
    if sparse:
        severity = 3 if state.running_days_28d < int(settings["minimum_running_days_28d_for_quality"]) / 2 else 1
        add("easy", 0.5 * severity, "limited recent consistency")
        add("long", -0.5 * severity, "limited recent consistency")
        add("quality", -1.0 * severity, "limited recent consistency")
    else:
        add("long", 1, "established 28-day consistency")
        add("quality", 2, "established 28-day consistency")
    if post_illness_check:
        add("easy", 3, "post-illness aerobic check")
        add("long", -3, "readiness check should be ordinary distance")
        add("quality", -5, "awaiting one more normal aerobic response")
    if health == CurrentHealthStatus.LITTLE_TIRED:
        add("easy", 2, "reported tiredness")
        add("long", -2, "reported tiredness")
        add("quality", -3, "reported tiredness")
    if environmental_caution:
        add("easy", 2, "weather caution")
        add("long", -2, "weather caution")
        add("quality", -2, "weather caution")

    recency_reference = float(settings["long_run_recency_reference_days"])
    if state.days_since_long_run is None or state.days_since_long_run >= recency_reference:
        add("long", 2, "long-run recency")
        add("quality", -1, "long-run recency opportunity cost")
    elif state.days_since_long_run >= recency_reference * 0.6:
        add("long", 1, "long-run recency")
    else:
        add("long", -1, "recent long run")
    if state.longest_run_30d_miles <= 0:
        add("long", -4, "no recent long-run baseline")

    minimum_quality_spacing = float(settings["minimum_days_between_quality_sessions"])
    quality_recency_reference = float(settings["quality_recency_reference_days"])
    quality_spacing = state.days_since_quality_run is None or state.days_since_quality_run >= minimum_quality_spacing
    quality_due = state.days_since_quality_run is None or state.days_since_quality_run >= quality_recency_reference
    if not quality_spacing:
        add("quality", -5, "insufficient quality spacing")
    elif quality_due:
        add("quality", 1, "weekly quality opportunity")
    else:
        add("quality", -2, "quality already completed this week")
    if state.quality_sessions_14d == 0:
        add("quality", 1, "no quality session in 14 days")
    elif state.quality_sessions_14d >= 2:
        add("quality", -3, "two quality sessions already in 14 days")
    if state.recent_performance_anomaly == "within_recent_range":
        add("long", 0.5, "normal recent response")
        add("quality", 1, "normal recent response")
    if state.fitness_trend.value == "improving" and state.trend_confidence in {
        ConfidenceLevel.HIGH,
        ConfidenceLevel.MODERATE,
    }:
        add("long", 0.5, "improving efficiency signal")
        add("quality", 1.5, "improving efficiency signal")
    if weekly_role in scores:
        add(weekly_role, 3, "coordinated weekly sequence")
    trace.append(
        _trace(
            RULES["weekly_sequence_priority"],
            weekly_role in scores,
            preferred_role=weekly_role,
            preference_points=3 if weekly_role in scores else 0,
        )
    )

    tie_priority = {"easy": 2, "long": 1, "quality": 0}
    selected = max(scores, key=lambda candidate: (scores[candidate], tie_priority[candidate]))
    trace.append(
        _trace(
            RULES["workout_scoring"],
            True,
            selected=selected,
            easy_score=round(scores["easy"], 2),
            long_score=round(scores["long"], 2),
            quality_score=round(scores["quality"], 2),
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
        )
    )
    trace.append(
        _trace(
            RULES["quality_eligible"],
            selected == "quality",
            score=round(scores["quality"], 2),
            days_since_quality=state.days_since_quality_run,
            quality_recency_reference_days=quality_recency_reference,
            quality_sessions_14d=state.quality_sessions_14d,
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
        cap = state.longest_run_30d_miles * float(settings["long_run_progression_factor"])
        weekly_norm = max(
            state.recent_load.trailing_28d.distance_miles / 4,
            state.recent_load.capacity_reference_miles or 0,
        )
        practical = max(5.0, min(cap, weekly_norm * 0.35 if weekly_norm else cap))
        upper = max(5.0, round(practical * 2) / 2)
        lower = max(4.5, upper - 0.5)
        return _result(
            state,
            workout_type=WorkoutType.LONG,
            title="Easy long run",
            distance=(lower, upper),
            zones=["Z1", "Z2"],
            structure=[
                WorkoutStep(instruction="First 10 minutes relaxed in Z1 or low Z2.", duration_minutes=10, target_zones=["Z1", "low Z2"]),
                WorkoutStep(instruction="Remain primarily Z2; no planned fast finish.", target_zones=["Z2"]),
            ],
            reasons=["A long run fits your recent mileage, recovery, and time since the last one.", "The distance stays close to your longest run in the past 30 days."],
            warnings=["The distance is rounded to a practical half mile; the progression limit is a warning, not a hard safety line."],
            modifications=["Cut the run to ordinary easy distance if fatigue, pain, or illness symptoms appear."],
            confidence=ConfidenceLevel.MODERATE,
            readiness=ReadinessFlag.READY,
            trace=trace,
        )

    if selected == "quality":
        quality_kind = _select_quality_variant(state, settings)
        prescription = _quality_prescription(quality_kind)
        structure = list(prescription["structure"])
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
            )
        )
        quality_warnings = [
            "Historical workout pace is context only; today’s prescription is controlled effort, not a fixed split mandate."
        ]
        quality_modifications = [
            "Convert to an ordinary easy run if warm-up HR/effort is abnormal or any pain appears."
        ]
        if environmental_caution:
            quality_warnings.append(
                "The planned forecast adds heat, humidity, or wind stress to this workout."
            )
            quality_modifications.append(
                "Prefer a cooler time; if conditions remain oppressive, replace the intervals with ordinary easy running."
            )
        return _result(
            state,
            workout_type=prescription["workout_type"],
            quality_session_type=quality_kind,
            title=prescription["title"],
            distance=prescription["distance"],
            zones=prescription["zones"],
            structure=structure,
            reasons=[
                "Recent training, recovery, and workout spacing support a harder session.",
                f"{quality_kind.value.replace('_', ' ').title()} is the next enabled workout in your rotation.",
            ],
            warnings=quality_warnings,
            modifications=quality_modifications,
            confidence=ConfidenceLevel.MODERATE,
            readiness=(ReadinessFlag.CAUTION if environmental_caution else ReadinessFlag.READY),
            trace=trace,
        )

    factor = float(settings["reduced_volume_factor"]) if (high_load or costly) else 1.0
    if tapering:
        factor = min(factor, 0.80)
    if environmental_caution:
        factor = min(factor, 0.85)
    prescribed_distance = tuple(round(max(2.0, value * factor) * 2) / 2 for value in easy_distance)
    cautions = high_load or z3_leakage or costly or sparse or health == CurrentHealthStatus.LITTLE_TIRED or environmental_caution
    reasons = ["An easy run best fits your recent training and recovery."]
    caution_explanations: list[str] = []
    if high_load: reasons.append(f"Your last 7 days are about {load_ratio * 100:.0f}% of your usual weekly capacity.")
    if high_load:
        caution_explanations.append("You have already run more than usual this week, so added mileage is limited.")
    if z3_leakage:
        z3_percent = state.moderate_fraction_14d * 100
        z3_threshold_percent = float(settings["moderate_intensity_leakage_fraction"]) * 100
        reasons.append(f"{z3_percent:.0f}% of recorded HR time in the last 14 days was moderate intensity.")
        caution_explanations.append(
            f"Moderate effort was {z3_percent:.0f}% of recent HR time, above the {z3_threshold_percent:.0f}% easy-running reference. Keep this run truly easy."
        )
    if costly:
        reasons.append("Your latest run cost more effort than usual or showed meaningful second-half drift.")
        caution_explanations.append("The latest comparable response was unusually costly, so the run is kept easy or reduced.")
    if sparse:
        reasons.append("Recent running has been less consistent than usual.")
        caution_explanations.append("Recent consistency is below the quality-work reference, so this remains an easy session.")
    if health == CurrentHealthStatus.LITTLE_TIRED:
        caution_explanations.append("Current status is a little tired, so ordinary easy running is favored over added stress.")
    if environmental_caution:
        caution_explanations.append("The forecast adds meaningful heat, humidity, or wind stress, so pace and distance should remain flexible.")
    if post_illness_check: reasons.append("This is the final easy aerobic check before reintroducing quality, provided the response is normal.")
    easy_modifications = ["Keep the run easy or shorten it if today’s HR/effort is unusually high."]
    if post_illness_check:
        easy_modifications.append(
            "If respiratory symptoms meaningfully return during or after this run, replace the planned quality session with an easy run."
        )
    return _result(
        state,
        workout_type=WorkoutType.EASY,
        title="Deliberately easy aerobic run" if cautions else "Easy aerobic run",
        distance=prescribed_distance,
        zones=["Z1", "Z2"],
        structure=[
            WorkoutStep(instruction="First 10 minutes in Z1 or low Z2.", duration_minutes=10, target_zones=["Z1", "low Z2"]),
            WorkoutStep(instruction="Then stay primarily 141–153 bpm; no fast finish.", target_zones=["Z2"]),
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
