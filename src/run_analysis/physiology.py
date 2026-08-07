"""Published physiological/environmental transforms used by the model."""

from __future__ import annotations

from math import atan, exp, sqrt


def minetti_running_cost_j_per_kg_m(grade_percent: float) -> float:
    """Metabolic cost of running from Minetti et al. (2002).

    The fifth-order polynomial uses grade as a fraction and is only applied
    inside the configured plausible-grade range. At zero grade it returns
    3.6 J/kg/m.
    """
    grade = grade_percent / 100.0
    return (
        155.4 * grade**5
        - 30.4 * grade**4
        - 43.3 * grade**3
        + 46.3 * grade**2
        + 19.5 * grade
        + 3.6
    )


def grade_energy_ratio(grade_percent: float, maximum_absolute_grade: float = 12.0) -> float:
    grade = max(-maximum_absolute_grade, min(maximum_absolute_grade, grade_percent))
    return minetti_running_cost_j_per_kg_m(grade) / minetti_running_cost_j_per_kg_m(0.0)


def relative_humidity_from_dewpoint(temperature_f: float, dewpoint_f: float) -> float:
    """Magnus approximation for relative humidity from dry/dew temperatures."""
    temperature_c = (temperature_f - 32.0) * 5.0 / 9.0
    dewpoint_c = (dewpoint_f - 32.0) * 5.0 / 9.0
    vapor = exp((17.625 * dewpoint_c) / (243.04 + dewpoint_c))
    saturation = exp((17.625 * temperature_c) / (243.04 + temperature_c))
    return max(0.0, min(100.0, 100.0 * vapor / saturation))


def wet_bulb_temperature_f(temperature_f: float, relative_humidity_percent: float) -> float:
    """Stull (2011) empirical wet-bulb approximation at sea-level pressure."""
    temperature_c = (temperature_f - 32.0) * 5.0 / 9.0
    humidity = max(5.0, min(99.0, relative_humidity_percent))
    wet_bulb_c = (
        temperature_c * atan(0.151977 * sqrt(humidity + 8.313659))
        + atan(temperature_c + humidity)
        - atan(humidity - 1.676331)
        + 0.00391838 * humidity**1.5 * atan(0.023101 * humidity)
        - 4.686035
    )
    return wet_bulb_c * 9.0 / 5.0 + 32.0


def estimated_shade_wbgt_f(
    temperature_f: float,
    relative_humidity_percent: float | None = None,
    dewpoint_f: float | None = None,
) -> float:
    """Approximate WBGT without solar load.

    Uses the standard shade/indoor weighting 0.7 wet-bulb + 0.3 globe,
    with air temperature substituted for globe temperature. It must not be
    interpreted as measured outdoor WBGT in direct sun.
    """
    humidity = relative_humidity_percent
    if humidity is None:
        if dewpoint_f is None:
            raise ValueError("Relative humidity or dew point is required")
        humidity = relative_humidity_from_dewpoint(temperature_f, dewpoint_f)
    wet_bulb = wet_bulb_temperature_f(temperature_f, humidity)
    return 0.7 * wet_bulb + 0.3 * temperature_f
