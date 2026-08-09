"""VO2 max estimated as a submaximal exercise test, with propagated error.

The application's central measurement — reference-condition speed at a fixed
heart rate — is exactly the observation a submaximal VO2-max test needs: a
steady-state workload paired with the heart rate it cost, under controlled
conditions. Grade, temperature, dew point, wind, and within-run position have
already been removed from it, which is the part a treadmill protocol achieves
by holding the laboratory constant.

That makes a principled two-step estimate available:

1. **Metabolic demand of the standardized speed.** The ACSM running equation
   converts speed to oxygen cost:

       VO2 (mL/kg/min) = 0.2 x S + 0.9 x S x G + 3.5,  S in m/min, G fractional

   It is validated for running speeds at or above 134 m/min (5 mph); below
   that, runners may be walking or jogging outside the equation's range, so no
   estimate is produced rather than a quiet extrapolation.

2. **Extrapolation to maximum by heart-rate reserve.** Percentage of oxygen
   uptake reserve tracks percentage of heart-rate reserve closely and more
   faithfully than %VO2max tracks %HRmax (Swain & Leutholtz, 1997):

       (VO2 - VO2rest) / (VO2max - VO2rest) = (HR - HRrest) / (HRmax - HRrest)

   Rearranged for VO2max, with VO2rest taken as 3.5 mL/kg/min (1 MET). This
   uses the athlete's own measured resting and maximum heart rates instead of
   a population regression.

The George et al. (1993) submaximal jogging equation is retained as a second,
differently-derived estimator. When the two agree within their combined
uncertainty they are pooled and confidence rises; when they disagree the
disagreement is reported rather than averaged away.

What makes the number defensible is not the point estimate but the interval.
Uncertainty is propagated numerically from every input that carries error —
the standardized pace's own 95% interval, the maximum-heart-rate figure, and
each equation's published standard error — so a wide interval reflects a real
lack of information rather than an arbitrary disclaimer.

None of this is a metabolic-cart measurement, and none of it is an independent
fitness signal: it is a unit conversion of the same pace-at-heart-rate
evidence, so it moves with that trend by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math

from .web.schemas import ConfidenceLevel, FitnessTrend, LocalVo2Estimate, Vo2TrendPoint

METERS_PER_MINUTE_PER_MPH = 26.8224

#: Resting oxygen uptake, one metabolic equivalent.
RESTING_VO2_ML_KG_MIN = 3.5

#: Lower bound of the ACSM running equation, in m/min (5 mph, 12:00 min/mile).
ACSM_RUNNING_FLOOR_M_MIN = 134.0

#: Standard error of estimate for the ACSM running equation, mL/kg/min.
ACSM_SEE_ML_KG_MIN = 2.4

#: Standard error of estimate for the George submaximal jogging equation.
GEORGE_SEE_ML_KG_MIN = 3.1

#: 1 SD of maximum heart rate depending on how the athlete arrived at it.
#: A measured maximum still varies day to day; an age-predicted one carries
#: the Tanaka et al. (2001) population error, which is far larger.
MAX_HR_SD_BPM = {"measured": 3.0, "estimated": 7.0}

#: Heart-rate reserve must be wide enough that the ratio is stable. Below this
#: the extrapolation multiplies a small denominator and becomes meaningless.
MINIMUM_RESERVE_FRACTION = 0.25

#: The comparison HR must sit below maximum for the extrapolation to be one.
MAXIMUM_RESERVE_FRACTION = 0.90


@dataclass(frozen=True, slots=True)
class _Estimator:
    name: str
    value: float
    sigma: float


def _age_on(birth_date: date, on_date: date) -> int:
    return on_date.year - birth_date.year - (
        (on_date.month, on_date.day) < (birth_date.month, birth_date.day)
    )


def _speed_m_min(pace_min_mile: float) -> float:
    return (60.0 / pace_min_mile) * METERS_PER_MINUTE_PER_MPH


def acsm_running_vo2(speed_m_min: float, grade_fraction: float = 0.0) -> float:
    """Oxygen cost of running at a given speed (ACSM metabolic equation)."""
    return 0.2 * speed_m_min + 0.9 * speed_m_min * grade_fraction + RESTING_VO2_ML_KG_MIN


def vo2_max_from_reserve(
    submaximal_vo2: float, submaximal_hr: float, resting_hr: float, maximum_hr: float
) -> float:
    """Extrapolate to maximum using %VO2 reserve = %heart-rate reserve."""
    reserve_fraction = (submaximal_hr - resting_hr) / (maximum_hr - resting_hr)
    return RESTING_VO2_ML_KG_MIN + (submaximal_vo2 - RESTING_VO2_ML_KG_MIN) / reserve_fraction


def george_vo2_max(
    pace_min_mile: float, submaximal_hr: float, weight_kg: float, male: bool
) -> float:
    """George et al. (1993) submaximal jogging VO2-max equation."""
    speed_mph = 60.0 / pace_min_mile
    return (
        54.07
        + 7.062 * int(male)
        - 0.1938 * weight_kg
        + 4.47 * speed_mph
        - 0.1453 * submaximal_hr
    )


def _unavailable(reason: str, detail: str) -> LocalVo2Estimate:
    return LocalVo2Estimate(
        method="Submaximal heart-rate-reserve estimate (not calculated)",
        confidence=ConfidenceLevel.UNAVAILABLE,
        trend=FitnessTrend.INSUFFICIENT_DATA,
        interpretation=detail,
        limitations=[reason],
    )


def _propagated_sigma(evaluate, inputs: dict[str, float], sigmas: dict[str, float]) -> float:
    """Combine input errors through ``evaluate`` by numeric partial derivatives.

    First-order propagation is used rather than a Monte Carlo draw so the
    result is deterministic, which the rest of the application relies on. The
    partials are evaluated numerically because the reserve extrapolation is a
    ratio and its sensitivity to maximum heart rate is not constant.
    """

    total = 0.0
    for name, sigma in sigmas.items():
        if sigma <= 0:
            continue
        step = max(abs(inputs[name]) * 1e-4, 1e-4)
        high = evaluate({**inputs, name: inputs[name] + step})
        low = evaluate({**inputs, name: inputs[name] - step})
        derivative = (high - low) / (2 * step)
        total += (derivative * sigma) ** 2
    return math.sqrt(total)


def _sentence_list(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def _pool(estimators: list[_Estimator]) -> tuple[float, float, bool]:
    """Inverse-variance pool, reporting whether the estimators actually agree."""
    if len(estimators) == 1:
        return estimators[0].value, estimators[0].sigma, True
    first, second = estimators[0], estimators[1]
    separation = abs(first.value - second.value)
    combined = math.sqrt(first.sigma**2 + second.sigma**2)
    agree = separation <= 1.96 * combined
    if not agree:
        # Pooling disagreeing estimators would manufacture a tight interval
        # around a number neither method supports. Keep the primary and widen
        # it to span the disagreement instead.
        widened = math.sqrt(first.sigma**2 + (separation / 2.0) ** 2)
        return first.value, widened, False
    weights = [1.0 / item.sigma**2 for item in estimators]
    value = sum(w * item.value for w, item in zip(weights, estimators)) / sum(weights)
    return value, math.sqrt(1.0 / sum(weights)), True


def vo2_series(
    trend_points, *, config: dict, as_of
) -> list[Vo2TrendPoint]:
    """Apply the same estimate to each point of the trailing pace curve.

    This is the identical calculation as the headline figure, so the curve is a
    monotone transform of the pace trend rather than independent evidence.
    Points whose pace falls outside the equation's validated range are dropped
    rather than extrapolated, which can leave gaps -- that is honest.
    """

    profile = config.get("profile") or {}
    if not profile:
        return []
    try:
        maximum_hr = float(config["max_hr"])
        resting_hr = float(config["resting_hr"])
        target_hr = float(config["target_hr"])
    except (KeyError, TypeError, ValueError):
        return []
    if maximum_hr <= resting_hr:
        return []
    reserve_fraction = (target_hr - resting_hr) / (maximum_hr - resting_hr)
    if not MINIMUM_RESERVE_FRACTION <= reserve_fraction <= MAXIMUM_RESERVE_FRACTION:
        return []
    try:
        weight_kg = float(profile["weight_lb"]) * 0.45359237
        male = str(profile["sex"]).casefold() == "male"
    except (KeyError, TypeError, ValueError):
        return []
    max_hr_sigma = MAX_HR_SD_BPM.get(
        str(profile.get("max_hr_source") or "estimated").casefold(), MAX_HR_SD_BPM["estimated"]
    )

    output: list[Vo2TrendPoint] = []
    for point in trend_points:
        pace = float(point.pace_min_mile)
        if _speed_m_min(pace) < ACSM_RUNNING_FLOOR_M_MIN:
            continue
        inputs = {"pace": pace, "maximum_hr": maximum_hr, "equation": 0.0}

        def reserve(values: dict[str, float]) -> float:
            submaximal = acsm_running_vo2(_speed_m_min(values["pace"])) + values["equation"]
            return vo2_max_from_reserve(submaximal, target_hr, resting_hr, values["maximum_hr"])

        def george(values: dict[str, float]) -> float:
            return george_vo2_max(values["pace"], target_hr, weight_kg, male) + values["equation"]

        pace_sigma = float(point.uncertainty_95_min_mile or 0.0) / 1.96
        estimators = [
            _Estimator(
                "reserve",
                reserve(inputs),
                _propagated_sigma(
                    reserve, inputs,
                    {"pace": pace_sigma, "maximum_hr": max_hr_sigma, "equation": ACSM_SEE_ML_KG_MIN},
                ),
            ),
            _Estimator(
                "george",
                george(inputs),
                _propagated_sigma(
                    george, inputs, {"pace": pace_sigma, "equation": GEORGE_SEE_ML_KG_MIN}
                ),
            ),
        ]
        value, sigma, _agree = _pool(estimators)
        output.append(
            Vo2TrendPoint(
                as_of=point.as_of,
                value_ml_kg_min=round(value, 2),
                uncertainty_95_ml_kg_min=round(1.96 * sigma, 2),
            )
        )
    return output


def estimate_local_vo2(
    *,
    current_pace,
    current_pace_uncertainty_95=None,
    as_of,
    recent_load,
    fitness_trend,
    config: dict,
    series: list[Vo2TrendPoint] | None = None,
) -> LocalVo2Estimate:
    """Estimate VO2 max from the standardized pace, or explain why it cannot."""

    profile = config.get("profile")
    if not profile:
        return _unavailable(
            "Birth date, sex, weight, and height are required by both equations.",
            "Add the optional profile fields in Settings to calculate this estimate.",
        )
    if current_pace is None:
        return _unavailable(
            "No standardized pace at the comparison heart rate is available yet.",
            "Upload enough comparable runs for a standardized pace to be estimated.",
        )

    try:
        maximum_hr = float(config["max_hr"])
        resting_hr = float(config["resting_hr"])
        target_hr = float(config["target_hr"])
    except (KeyError, TypeError, ValueError):
        return _unavailable(
            "Maximum, resting, and comparison heart rates are all required to extrapolate "
            "from a submaximal observation.",
            "Set your maximum, resting, and comparison heart rates in Settings to calculate "
            "this estimate.",
        )
    if maximum_hr <= resting_hr:
        return _unavailable(
            "Maximum heart rate must exceed resting heart rate.",
            "Check your maximum and resting heart rates in Settings.",
        )
    reserve_fraction = (target_hr - resting_hr) / (maximum_hr - resting_hr)
    if not MINIMUM_RESERVE_FRACTION <= reserve_fraction <= MAXIMUM_RESERVE_FRACTION:
        return _unavailable(
            f"The comparison heart rate sits at {reserve_fraction * 100:.0f}% of heart-rate "
            "reserve, outside the range where extrapolating to maximum is meaningful.",
            "Set a comparison heart rate between roughly 25% and 90% of your heart-rate "
            "reserve to enable this estimate.",
        )

    speed = _speed_m_min(current_pace.minutes_per_mile)
    if speed < ACSM_RUNNING_FLOOR_M_MIN:
        floor_pace = 60.0 / (ACSM_RUNNING_FLOOR_M_MIN / METERS_PER_MINUTE_PER_MPH)
        return _unavailable(
            f"The ACSM running equation is validated at or above {ACSM_RUNNING_FLOOR_M_MIN:.0f} m/min "
            f"({int(floor_pace)}:{round((floor_pace % 1) * 60):02d} per mile); your standardized pace is slower.",
            "Your standardized pace at the comparison heart rate is below the range where "
            "the published running equation applies, so no estimate is shown rather than "
            "an extrapolation beyond its validation.",
        )

    weight_kg = float(profile["weight_lb"]) * 0.45359237
    height_m = float(profile["height_in"]) * 0.0254
    male = str(profile["sex"]).casefold() == "male"
    age = _age_on(date.fromisoformat(str(profile["birth_date"])), as_of.date())
    max_hr_source = str(profile.get("max_hr_source") or "estimated").casefold()
    max_hr_sigma = MAX_HR_SD_BPM.get(max_hr_source, MAX_HR_SD_BPM["estimated"])

    # 95% interval on the standardized pace, converted to a 1 SD pace error.
    pace_sigma = (
        float(current_pace_uncertainty_95) / 1.96
        if current_pace_uncertainty_95
        else 0.0
    )
    inputs = {
        "pace": float(current_pace.minutes_per_mile),
        "maximum_hr": maximum_hr,
        "equation": 0.0,
    }

    def reserve_estimate(values: dict[str, float]) -> float:
        submaximal = acsm_running_vo2(_speed_m_min(values["pace"])) + values["equation"]
        return vo2_max_from_reserve(submaximal, target_hr, resting_hr, values["maximum_hr"])

    def george_estimate(values: dict[str, float]) -> float:
        return george_vo2_max(values["pace"], target_hr, weight_kg, male) + values["equation"]

    reserve_value = reserve_estimate(inputs)
    reserve_sigma = _propagated_sigma(
        reserve_estimate,
        inputs,
        {"pace": pace_sigma, "maximum_hr": max_hr_sigma, "equation": ACSM_SEE_ML_KG_MIN},
    )
    george_value = george_estimate(inputs)
    george_sigma = _propagated_sigma(
        george_estimate,
        inputs,
        {"pace": pace_sigma, "equation": GEORGE_SEE_ML_KG_MIN},
    )

    estimators = [
        _Estimator("ACSM + heart-rate reserve", reserve_value, reserve_sigma),
        _Estimator("George submaximal jogging", george_value, george_sigma),
    ]
    value, sigma, agree = _pool(estimators)
    uncertainty_95 = 1.96 * sigma

    # Confidence follows the width of the interval and whether two
    # differently-derived equations landed in the same place.
    if agree and uncertainty_95 <= 4.0:
        confidence = ConfidenceLevel.MODERATE
    elif agree and uncertainty_95 <= 7.0:
        confidence = ConfidenceLevel.LOW
    else:
        confidence = ConfidenceLevel.LOW

    weekly_miles = recent_load.trailing_28d.distance_miles / 4.0
    # The NASA/JSC activity rating is objectively 7 only when documented
    # running exceeds 10 mi/week; below that, do not invent a questionnaire answer.
    bmi = weight_kg / height_m**2
    demographic = (
        56.363 + 1.921 * 7 - 0.381 * age - 0.754 * bmi + 10.987 * int(male)
        if weekly_miles > 10
        else None
    )

    agreement = (
        f"Two independently derived equations agree: {reserve_value:.1f} from the "
        f"heart-rate-reserve extrapolation and {george_value:.1f} from George."
        if agree
        else f"The two equations disagree ({reserve_value:.1f} versus {george_value:.1f}), "
        "so the range below is widened to span both rather than averaging them."
    )
    contributors = []
    if pace_sigma:
        contributors.append("your standardized pace's own uncertainty")
    contributors.append(
        "a measured maximum heart rate" if max_hr_source == "measured"
        else "an unmeasured maximum heart rate, which is the largest single contributor"
    )
    contributors.append("each equation's published standard error")

    return LocalVo2Estimate(
        value_ml_kg_min=round(value, 1),
        uncertainty_95_ml_kg_min=round(uncertainty_95, 1),
        method=(
            "ACSM running equation at your standardized pace, extrapolated to maximum by "
            "heart-rate reserve (Swain & Leutholtz), cross-checked against George et al."
        ),
        confidence=confidence,
        trend=fitness_trend,
        demographic_baseline_ml_kg_min=round(demographic, 1) if demographic else None,
        demographic_uncertainty_95_ml_kg_min=11.2 if demographic else None,
        series=series or [],
        interpretation=(
            f"{agreement} The range comes from propagating "
            + _sentence_list(contributors)
            + "."
        ),
        limitations=[
            "This is a unit conversion of your pace-at-heart-rate evidence, not a second "
            "opinion about it; it moves with that trend by construction.",
            "The heart-rate-reserve extrapolation assumes a linear oxygen-uptake/heart-rate "
            "relationship that flattens near maximum, which biases a single submaximal "
            "observation.",
            (
                "Maximum heart rate is recorded as measured, so its day-to-day variation is "
                "the assumed error."
                if max_hr_source == "measured"
                else "Maximum heart rate is not recorded as measured. Measuring it would "
                "narrow this range more than any other single change."
            ),
            "The ACSM equation was derived on treadmills; outdoor running adds air "
            "resistance that is not corrected here.",
            "A metabolic-cart maximal test remains the direct measurement standard.",
        ],
    )
