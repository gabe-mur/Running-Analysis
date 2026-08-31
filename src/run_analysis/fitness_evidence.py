"""Comparability weights for per-run fitness trend observations.

Measurement uncertainty is already handled by inverse-variance weighting in
``analytics.py``.  The weight here answers a different question: how closely
does the run resemble the steady, submaximal evidence the trend is intended to
compare?  Structured quality sessions can answer that question imperfectly;
they should not disappear merely because of their workout label.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

from .web.schemas import WorkoutType


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _quality_session_comparability(result: Mapping[str, Any] | None) -> float:
    """Return 0..1 evidence quality inside a variable-intensity session.

    A model result only exists after the window loader has enforced continuity,
    heart-rate, distance, and submaximal-range checks.  We then ask whether the
    score had several effective windows, support near the reference time, and a
    stable fixed-time benchmark.  Missing fields retain conservative behavior
    for model results written by older versions.
    """

    if not result:
        return 0.55

    effective_windows = result.get("effective_window_count")
    if effective_windows is None:
        window_factor = 0.75
    else:
        window_factor = _clamp(
            math.sqrt(max(0.0, float(effective_windows)) / 4.0), 0.35, 1.0
        )

    reference_support = str(result.get("reference_time_support") or "legacy")
    reference_factor = {
        "interpolation": 1.0,
        "limited_extrapolation": 0.80,
        "legacy": 0.85,
    }.get(reference_support, 0.50)

    benchmark = result.get("steady_aerobic_benchmark")
    if isinstance(benchmark, Mapping):
        if benchmark.get("selection_quality") == "strict_observed":
            benchmark_factor = 1.0
        else:
            window_evidence = _clamp(
                float(benchmark.get("window_evidence_weight") or 0.0), 0.0, 1.0
            )
            benchmark_factor = 0.55 + 0.40 * window_evidence
    else:
        benchmark_factor = 0.50

    return _clamp(window_factor * reference_factor * benchmark_factor, 0.10, 1.0)


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

    if workout in {WorkoutType.HIKE, WorkoutType.BIKE}:
        return 0.0
    if workout == WorkoutType.RUN_WALK:
        return health_weight * 0.5
    if workout in {WorkoutType.TEMPO_THRESHOLD, WorkoutType.RACE}:
        return health_weight * 0.65 * _quality_session_comparability(result)
    if workout == WorkoutType.INTERVALS:
        return health_weight * 0.45 * _quality_session_comparability(result)
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
