"""Shared conversion from a structured prescription to expected intensity."""

from __future__ import annotations

from .web.schemas import RecommendationResponse


def prescribed_zone_minutes(
    result: RecommendationResponse, total_minutes: float
) -> tuple[float, float]:
    moderate = 0.0
    hard = 0.0
    for step in result.structure:
        zones = " ".join(step.target_zones).casefold()
        if step.repetitions and step.work_duration_minutes:
            work = step.repetitions * step.work_duration_minutes
        elif step.repetitions and step.work_duration_range_minutes:
            low, high = step.work_duration_range_minutes
            work = step.repetitions * ((low + high) / 2)
        elif step.phase == "work" and step.duration_minutes:
            work = step.duration_minutes
        elif step.phase == "work":
            work = total_minutes * 0.25
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
    quality_minutes = moderate + hard
    if quality_minutes > total_minutes > 0:
        scale = total_minutes / quality_minutes
        moderate *= scale
        hard *= scale
    return moderate, hard


def prescribed_intensity_factor(
    result: RecommendationResponse, total_minutes: float
) -> float:
    if total_minutes <= 0:
        return 1.0
    moderate, hard = prescribed_zone_minutes(result, total_minutes)
    return 1.0 + 0.15 * (moderate / total_minutes) + 0.50 * (
        hard / total_minutes
    )


def structured_duration_minutes(result: RecommendationResponse) -> float | None:
    total = 0.0
    known = False
    for step in result.structure:
        if step.duration_minutes is not None:
            total += step.duration_minutes
            known = True
            continue
        if step.repetitions and step.work_duration_minutes:
            total += step.repetitions * step.work_duration_minutes
            if step.recovery_duration_minutes:
                total += max(0, step.repetitions - 1) * step.recovery_duration_minutes
            known = True
            continue
        if step.repetitions and step.work_duration_range_minutes:
            low, high = step.work_duration_range_minutes
            total += step.repetitions * ((low + high) / 2)
            if step.recovery_duration_minutes:
                total += max(0, step.repetitions - 1) * step.recovery_duration_minutes
            known = True
    return total if known and total > 0 else None
