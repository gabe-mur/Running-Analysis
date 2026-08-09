"""A single headline state for what training is currently doing.

This is deliberately not another fitness score. Garmin-style "Productive /
Unproductive" verdicts collapse several independent signals into one number
and then cannot explain themselves. Here the headline is a *classification*
over evidence the application already computes separately, every rule is
named, and the individual signals stay visible underneath it. The athlete can
always see which rule fired and on what facts.

The states answer "what is my training doing right now", not "how fit am I":

``building``
    Load sits near demonstrated capacity with appropriate quality exposure and
    no recovery flags.
``maintaining``
    Normal load and stable performance, with no strong signal either way.
``rebuilding``
    Below retained capacity after a gap, but training is climbing back
    appropriately.
``recovering``
    Current health status or recent responses indicate recovery.
``strained``
    Acute load is high, or the most recent comparable effort was unusually
    costly.
``underloaded``
    Sustained running materially below retained capacity for long enough to
    matter.
``insufficient_data``
    Not enough recent evidence to say.

Rules are evaluated in a fixed precedence order, because these states are not
independent: a recovering athlete whose acute load is also high is recovering
first. Health outranks load, load outranks progression.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .web.schemas import (
    ConfidenceLevel,
    CurrentHealthStatus,
    FitnessState,
    FitnessTrend,
    RuleTrace,
    TrainingStatus,
    TrainingStatusSummary,
)


@dataclass(frozen=True, slots=True)
class StatusRule:
    rule_id: str
    description: str


STATUS_RULES: tuple[StatusRule, ...] = (
    StatusRule("status_insufficient_evidence", "Too few recent runs or no demonstrated capacity to classify training."),
    StatusRule("status_recovering", "Current health status, or a health-tagged run not yet followed by normal running, indicates recovery."),
    StatusRule("status_strained", "Acute load is high relative to demonstrated capacity, or the latest comparable effort was unusually costly."),
    StatusRule("status_rebuilding", "Running is below retained capacity after a gap, but the most recent week is climbing back toward it."),
    StatusRule("status_underloaded", "Sustained running has been materially below retained capacity long enough to matter."),
    StatusRule("status_building", "Load sits near demonstrated capacity with quality exposure and no recovery or strain flags."),
    StatusRule("status_maintaining", "Load and performance are steady, with no strong signal in either direction."),
)

RULES = {rule.rule_id: rule for rule in STATUS_RULES}

#: A week this far above demonstrated capacity is a strain signal. Shared with
#: the recommendation engine's ``high_load_ratio`` when configured.
DEFAULT_HIGH_LOAD_RATIO = 1.30

#: Sustained running below this fraction of capacity is underloading.
UNDERLOAD_RATIO = 0.70

#: At or above this fraction of capacity counts as training near capacity.
NEAR_CAPACITY_RATIO = 0.85

#: The acute week must exceed the 28-day average by this much to read as a
#: deliberate climb back rather than ordinary week-to-week noise.
REBUILD_MARGIN = 0.15

#: Below this many activities in the trailing 28 days there is not enough
#: recent evidence. Deliberately read from the 28-day load window rather than
#: ``state.running_days_28d``, which despite its name spans whatever window the
#: caller asked for -- 90 days on the dashboard.
MINIMUM_RECENT_ACTIVITIES = 4

#: A second-half heart-rate decoupling above this is a costly-effort signal.
HIGH_DRIFT_PERCENT = 8.0

#: Check-in states that mean recovery. "A little tired" is an ordinary
#: training day and must not headline the dashboard as recovery.
RECOVERY_HEALTH_STATUSES = frozenset(
    {CurrentHealthStatus.SICK_OR_RECOVERING, CurrentHealthStatus.PAIN_OR_INJURY_CONCERN}
)


def _trace(rule_id: str, fired: bool, **facts: Any) -> RuleTrace:
    rule = RULES[rule_id]
    return RuleTrace(
        rule_id=rule.rule_id, description=rule.description, fired=fired, facts=facts
    )


def _ratios(state: FitnessState) -> tuple[float | None, float | None, float | None]:
    """Acute, sustained, and capacity figures, all weekly miles."""
    capacity = state.recent_load.capacity_reference_miles
    if not capacity:
        return None, None, None
    acute = state.recent_load.acute_distance_to_capacity_ratio
    sustained_weekly = state.recent_load.trailing_28d.distance_miles / 4.0
    return acute, sustained_weekly / capacity, capacity


def build_training_status(state: FitnessState, config: dict | None = None) -> TrainingStatusSummary:
    """Classify current training, with the full rule trace attached."""

    coaching = (config or {}).get("coaching", {})
    high_load_ratio = float(coaching.get("high_load_ratio", DEFAULT_HIGH_LOAD_RATIO))
    acute_ratio, sustained_ratio, capacity = _ratios(state)
    trace: list[RuleTrace] = []

    costly = state.recent_performance_anomaly == "unusually_costly"
    high_drift = (
        state.last_run_drift_percent is not None
        and state.last_run_drift_percent > HIGH_DRIFT_PERCENT
    )
    unresolved_health = state.recent_illness_or_recovery and state.normal_runs_since_health_event < 3
    health_flag = state.current_health_status in RECOVERY_HEALTH_STATUSES

    # 1. Evidence gate. Everything below divides by capacity or reasons about
    #    recent consistency, so an empty history must stop here rather than
    #    produce a confident label from nothing.
    recent_activities = state.recent_load.trailing_28d.activity_count
    insufficient = (
        capacity is None
        or sustained_ratio is None
        or recent_activities < MINIMUM_RECENT_ACTIVITIES
    )
    trace.append(
        _trace(
            "status_insufficient_evidence",
            insufficient,
            activities_28d=recent_activities,
            minimum_activities_28d=MINIMUM_RECENT_ACTIVITIES,
            capacity_reference_miles=round(capacity, 1) if capacity else None,
        )
    )
    if insufficient:
        return _summary(
            TrainingStatus.INSUFFICIENT_DATA,
            "There is not enough recent running to describe what your training is doing.",
            ConfidenceLevel.UNAVAILABLE,
            trace,
            state,
        )

    # 2. Health outranks load. An athlete who is unwell is recovering even if
    #    their mileage happens to look ideal.
    recovering = health_flag or unresolved_health
    trace.append(
        _trace(
            "status_recovering",
            recovering,
            current_health_status=state.current_health_status.value,
            recent_illness_or_recovery=state.recent_illness_or_recovery,
            normal_runs_since_health_event=state.normal_runs_since_health_event,
        )
    )
    if recovering:
        detail = (
            "Your current check-in is not normal, so training is treated as recovery."
            if health_flag
            else (
                f"A health-tagged run is still recent and only {state.normal_runs_since_health_event} "
                "normal run(s) have followed it."
            )
        )
        return _summary(TrainingStatus.RECOVERING, detail, ConfidenceLevel.HIGH, trace, state)

    # 3. Strain outranks progression: a costly response matters more than
    #    whether the mileage chart is pointing the right way.
    strained = (acute_ratio is not None and acute_ratio >= high_load_ratio) or costly or high_drift
    trace.append(
        _trace(
            "status_strained",
            strained,
            acute_to_capacity_ratio=round(acute_ratio, 2) if acute_ratio is not None else None,
            high_load_ratio=high_load_ratio,
            recent_performance_anomaly=state.recent_performance_anomaly,
            last_run_drift_percent=state.last_run_drift_percent,
        )
    )
    if strained:
        reasons = []
        if acute_ratio is not None and acute_ratio >= high_load_ratio:
            reasons.append(
                f"the last 7 days are {acute_ratio * 100:.0f}% of the {capacity:.1f} mi/week you have demonstrated"
            )
        if costly:
            reasons.append("your latest comparable run cost more effort than usual")
        if high_drift:
            reasons.append(f"the latest run drifted {state.last_run_drift_percent:.1f}% in its second half")
        return _summary(
            TrainingStatus.STRAINED,
            "Training is running ahead of what you have absorbed: " + ", and ".join(reasons) + ".",
            ConfidenceLevel.MODERATE,
            trace,
            state,
        )

    below_capacity = sustained_ratio < UNDERLOAD_RATIO
    climbing = acute_ratio is not None and acute_ratio >= sustained_ratio + REBUILD_MARGIN

    # 4. Below capacity but climbing back is rebuilding, which is a normal and
    #    correct thing to be doing; it must not read as a deficiency.
    rebuilding = below_capacity and climbing
    trace.append(
        _trace(
            "status_rebuilding",
            rebuilding,
            sustained_to_capacity_ratio=round(sustained_ratio, 2),
            acute_to_capacity_ratio=round(acute_ratio, 2) if acute_ratio is not None else None,
            underload_ratio=UNDERLOAD_RATIO,
            rebuild_margin=REBUILD_MARGIN,
        )
    )
    if rebuilding:
        return _summary(
            TrainingStatus.REBUILDING,
            (
                f"Your last four weeks average {sustained_ratio * 100:.0f}% of your demonstrated "
                f"{capacity:.1f} mi/week, and the most recent week is climbing back toward it."
            ),
            ConfidenceLevel.MODERATE,
            trace,
            state,
        )

    # 5. Below capacity and not climbing.
    trace.append(
        _trace(
            "status_underloaded",
            below_capacity,
            sustained_to_capacity_ratio=round(sustained_ratio, 2),
            underload_ratio=UNDERLOAD_RATIO,
        )
    )
    if below_capacity:
        return _summary(
            TrainingStatus.UNDERLOADED,
            (
                f"Your last four weeks average {sustained_ratio * 100:.0f}% of the "
                f"{capacity:.1f} mi/week you have demonstrated, and the most recent week is not "
                "climbing back toward it."
            ),
            ConfidenceLevel.MODERATE,
            trace,
            state,
        )

    # 6. Near capacity with quality exposure and nothing flagged.
    near_capacity = sustained_ratio >= NEAR_CAPACITY_RATIO
    has_quality = state.quality_sessions_14d > 0
    improving = (
        state.fitness_trend == FitnessTrend.IMPROVING
        or (
            state.recent_load.previous_capacity_reference_miles is not None
            and capacity > state.recent_load.previous_capacity_reference_miles
        )
    )
    building = near_capacity and has_quality and improving
    trace.append(
        _trace(
            "status_building",
            building,
            sustained_to_capacity_ratio=round(sustained_ratio, 2),
            near_capacity_ratio=NEAR_CAPACITY_RATIO,
            quality_sessions_14d=state.quality_sessions_14d,
            fitness_trend=state.fitness_trend.value,
        )
    )
    if building:
        return _summary(
            TrainingStatus.BUILDING,
            (
                f"You are training at {sustained_ratio * 100:.0f}% of your demonstrated "
                f"{capacity:.1f} mi/week with recent quality work and no recovery flags."
            ),
            ConfidenceLevel.MODERATE,
            trace,
            state,
        )

    # 7. Everything else is steady training.
    trace.append(_trace("status_maintaining", True, sustained_to_capacity_ratio=round(sustained_ratio, 2)))
    missing = []
    if not near_capacity:
        missing.append("volume is a little under your demonstrated capacity")
    if not has_quality:
        missing.append("there has been no quality session in 14 days")
    if not improving:
        missing.append("no signal is pointing clearly upward")
    detail = (
        f"Load and performance are steady at {sustained_ratio * 100:.0f}% of your demonstrated "
        f"{capacity:.1f} mi/week"
    )
    detail += f"; {', and '.join(missing)}." if missing else "."
    return _summary(TrainingStatus.MAINTAINING, detail, ConfidenceLevel.MODERATE, trace, state)


def _summary(
    status: TrainingStatus,
    detail: str,
    confidence: ConfidenceLevel,
    trace: list[RuleTrace],
    state: FitnessState,
) -> TrainingStatusSummary:
    return TrainingStatusSummary(
        status=status,
        label=STATUS_LABELS[status],
        detail=detail,
        confidence=confidence,
        as_of=state.as_of,
        rule_trace=trace,
    )


STATUS_LABELS: dict[TrainingStatus, str] = {
    TrainingStatus.BUILDING: "Building",
    TrainingStatus.MAINTAINING: "Maintaining",
    TrainingStatus.REBUILDING: "Rebuilding",
    TrainingStatus.RECOVERING: "Recovering",
    TrainingStatus.STRAINED: "Strained",
    TrainingStatus.UNDERLOADED: "Underloaded",
    TrainingStatus.INSUFFICIENT_DATA: "Not enough data",
}
