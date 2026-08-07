"""Published VO2-max cross-checks; deliberately separate from Garmin values."""

from __future__ import annotations

from datetime import date

from .web.schemas import ConfidenceLevel, FitnessTrend, LocalVo2Estimate


def _age_on(birth_date: date, on_date: date) -> int:
    return on_date.year - birth_date.year - (
        (on_date.month, on_date.day) < (birth_date.month, birth_date.day)
    )


def _george_vo2(
    pace_min_mile: float, target_hr: float, weight_kg: float, male: bool
) -> float:
    speed_mph = 60.0 / pace_min_mile
    return 54.07 + 7.062 * int(male) - 0.1938 * weight_kg + 4.47 * speed_mph - 0.1453 * target_hr


def estimate_local_vo2(
    *, current_pace, as_of, recent_load, fitness_trend: FitnessTrend, config: dict
) -> LocalVo2Estimate:
    """Apply published equations only when their required inputs are present.

    George et al. was a controlled steady-state treadmill jogging protocol. We
    use reference-condition pace at 145 bpm as an approximation, so the result
    is explicitly low-confidence even though the original protocol SEE was
    about 3.1 mL/kg/min. Jackson is shown only as a broad demographic baseline.
    """

    profile = config.get("profile")
    if not profile or current_pace is None:
        return LocalVo2Estimate(
            method="George steady-state jogging equation (not calculated)",
            confidence=ConfidenceLevel.UNAVAILABLE,
            trend=FitnessTrend.INSUFFICIENT_DATA,
            interpretation="Profile inputs or a current standardized pace are unavailable.",
        )
    birth_date = date.fromisoformat(str(profile["birth_date"]))
    age = _age_on(birth_date, as_of.date())
    male = str(profile["sex"]).casefold() == "male"
    weight_kg = float(profile["weight_lb"]) * 0.45359237
    height_m = float(profile["height_in"]) * 0.0254
    bmi = weight_kg / height_m**2
    value = _george_vo2(
        current_pace.minutes_per_mile,
        float(config["target_hr"]),
        weight_kg,
        male,
    )
    # The NASA/JSC activity rating is objectively 7 only when documented
    # running exceeds 10 mi/week; below that, do not invent a questionnaire answer.
    weekly_miles = recent_load.trailing_28d.distance_miles / 4.0
    activity_rating = 7 if weekly_miles > 10 else None
    demographic = None
    if activity_rating is not None:
        demographic = (
            56.363
            + 1.921 * activity_rating
            - 0.381 * age
            - 0.754 * bmi
            + 10.987 * int(male)
        )
    return LocalVo2Estimate(
        value_ml_kg_min=round(value, 1),
        # 1.96 × 3.1 protocol SEE, rounded upward; outdoor adaptation adds
        # unquantified error and is disclosed rather than hidden in fake precision.
        uncertainty_95_ml_kg_min=6.1,
        method="George steady-state jogging equation adapted to standardized pace @ target HR",
        confidence=ConfidenceLevel.LOW,
        trend=fitness_trend,
        demographic_baseline_ml_kg_min=round(demographic, 1) if demographic else None,
        demographic_uncertainty_95_ml_kg_min=11.2 if demographic else None,
        interpretation=(
            "This cross-check estimates aerobic capacity, but its change follows the same "
            "pace/HR evidence and is not an independent vote on fitness."
        ),
        limitations=[
            "The George equation was validated using a controlled, level, steady-state treadmill jog; outdoor historical runs are only an approximation.",
            "The demographic Jackson baseline uses a broad population model and does not measure training response.",
            "A metabolic-cart maximal test remains the direct measurement standard.",
        ],
    )
