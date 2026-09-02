"""Comparability weights for the steady aerobic fitness trend.

Quality sessions remain first-class training-load and workout-performance
evidence, but their deliberately changing intensity does not measure the same
thing as a steady submaximal run.  Until a separate holistic model can make
that comparison honestly, they are context on this chart rather than inputs to
its trend line.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .web.schemas import WorkoutType


def trend_evidence_weight(
    health_tag: str,
    workout: WorkoutType,
    result: Mapping[str, Any] | None = None,
) -> float:
    """Return the run's contextual contribution to the fitness trend.

    This is multiplied by the analytics layer's uncertainty-derived weight.
    Consequently a noisy estimate receives less influence without encoding the
    same penalty twice here.
    """

    health_weight = {
        "normal": 1.0,
        "illness_recovery": 0.65,
        "illness": 0.25,
        "injury_affected": 0.25,
    }.get(health_tag, 0.5)

    if workout in {
        WorkoutType.HIKE,
        WorkoutType.BIKE,
        WorkoutType.INTERVALS,
        WorkoutType.TEMPO_THRESHOLD,
        WorkoutType.RACE,
    }:
        return 0.0
    if workout == WorkoutType.RUN_WALK:
        return health_weight * 0.5
    return health_weight


def trend_evidence_reason(
    health_tag: str,
    workout: WorkoutType,
    weight: float,
    result: Mapping[str, Any] | None = None,
) -> str:
    """Explain the comparability decision in athlete-facing language."""

    percent = round(weight * 100)
    if weight <= 0:
        if workout in {
            WorkoutType.INTERVALS,
            WorkoutType.TEMPO_THRESHOLD,
            WorkoutType.RACE,
        }:
            return (
                "Counts fully toward training load and quality-workout progress, "
                "but has 0% influence on the steady aerobic trend."
            )
        return f"Shown here, but {workout.value.replace('_', ' ')} is not running trend evidence."
    reasons: list[str] = []
    if result and result.get("fitness_evidence_phase") == "pre_quality_only":
        excluded = int(result.get("quality_excluded_window_count") or 0)
        reasons.append(
            "only pre-work aerobic windows were used"
            + (f"; {excluded} work/cooldown windows were excluded" if excluded else "")
        )
    elif workout in {WorkoutType.INTERVALS, WorkoutType.TEMPO_THRESHOLD, WorkoutType.RACE}:
        reasons.append(
            "reliable comparable windows were found within a variable-intensity workout"
        )
    elif workout == WorkoutType.RUN_WALK:
        reasons.append("run/walk structure is less comparable with continuous running")
    if health_tag != "normal":
        reasons.append(f"the run is tagged {health_tag.replace('_', ' ')}")
    if not reasons and weight >= 0.999:
        return "Used normally in the fitness trend."
    detail = " and ".join(reasons) or "the evidence is less comparable than a normal aerobic run"
    return f"Used with {percent}% influence because {detail}."
