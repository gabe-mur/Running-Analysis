"""Continuous forecast stress for outdoor training decisions.

The forecast does not include solar radiation or WBGT, so this is deliberately
an adaptation signal rather than a claim to measure medical heat risk. Smooth
ramps prevent a one-degree forecast change from replacing an entire workout.
"""

from __future__ import annotations

from dataclasses import dataclass

from .web.schemas import PlannedWeather, WeatherExposureBaseline


@dataclass(frozen=True, slots=True)
class TrainingWeatherStress:
    score: float
    band: str
    heat: float
    humidity: float
    cold: float
    wind: float
    precipitation: float
    relative_spike: float
    extreme_reasons: tuple[str, ...]

    @property
    def needs_adaptation(self) -> bool:
        return self.score >= 0.20

    @property
    def caution(self) -> bool:
        return self.score >= 0.40

    @property
    def severe(self) -> bool:
        return self.score >= 0.85

    @property
    def extreme(self) -> bool:
        return bool(self.extreme_reasons)


def _ramp(value: float | None, low: float, high: float) -> float:
    if value is None or value <= low:
        return 0.0
    if value >= high:
        return 1.0
    return (value - low) / (high - low)


def _extreme_reasons(
    weather: PlannedWeather, apparent_f: float | None
) -> tuple[str, ...]:
    reasons: list[str] = []
    reasons.extend(
        f"official NWS {alert.event}"
        for alert in weather.emergency_alerts
        if alert.blocks_outdoor_run
    )
    code = weather.weather_code
    sustained = weather.wind_speed_mph or 0.0
    gust = weather.wind_gust_mph or 0.0
    visibility = weather.visibility_miles
    snow_codes = {71, 73, 75, 77, 85, 86}
    if code in {95, 96, 99}:
        reasons.append("thunderstorm forecast")
    if code in {56, 57, 66, 67}:
        reasons.append("freezing precipitation forecast")
    if code in snow_codes and max(sustained, gust) >= 35 and visibility is not None and visibility <= 0.25:
        reasons.append("blizzard-like wind, snow, and visibility")
    if visibility is not None and visibility <= 0.25:
        reasons.append("visibility at or below one quarter mile")
    if sustained >= 74:
        reasons.append("hurricane-force sustained wind")
    elif sustained >= 39:
        reasons.append("tropical-storm-force sustained wind")
    elif gust >= 58:
        reasons.append("high-wind-warning-level gusts")
    if apparent_f is not None and apparent_f >= 103:
        reasons.append("dangerous apparent heat")
    if apparent_f is not None and apparent_f <= -10:
        reasons.append("dangerous apparent cold")
    return tuple(dict.fromkeys(reasons))


def assess_training_weather(
    weather: PlannedWeather | None,
    baseline: WeatherExposureBaseline | None = None,
) -> TrainingWeatherStress:
    if weather is None:
        return TrainingWeatherStress(
            0.0, "none", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ()
        )

    apparent = (
        weather.apparent_temperature_f
        if weather.apparent_temperature_f is not None
        else weather.temperature_f
    )
    has_baseline = bool(baseline and baseline.sample_count >= 3)
    # Relative change matters only once conditions enter a range where the
    # change can actually add training cost. A pleasant 75 F day after a cool
    # month is not heat stress merely because it is 20 degrees warmer.
    absolute_heat_gate = _ramp(apparent, 75, 85)
    absolute_humidity_gate = _ramp(weather.dewpoint_f, 55, 70)
    absolute_cold_gate = _ramp(
        -apparent if apparent is not None else None, -55, -40
    )
    heat_spike = (
        0.75
        * _ramp(
            apparent - baseline.warm_apparent_temperature_f
            if apparent is not None
            and baseline
            and baseline.warm_apparent_temperature_f is not None
            else None,
            5,
            20,
        )
        * absolute_heat_gate
        if has_baseline
        else 0.0
    )
    humidity_spike = (
        0.45
        * _ramp(
            weather.dewpoint_f - baseline.humid_dewpoint_f
            if weather.dewpoint_f is not None
            and baseline
            and baseline.humid_dewpoint_f is not None
            else None,
            3,
            12,
        )
        * absolute_humidity_gate
        if has_baseline
        else 0.0
    )
    cold_spike = (
        0.60
        * _ramp(
            baseline.cold_apparent_temperature_f - apparent
            if apparent is not None
            and baseline
            and baseline.cold_apparent_temperature_f is not None
            else None,
            10,
            30,
        )
        * absolute_cold_gate
        if has_baseline
        else 0.0
    )
    heat = max(
        0.70 * _ramp(apparent, 85, 103),
        heat_spike,
        0.25 * _ramp(apparent, 80, 90) if not has_baseline else 0.0,
    )
    humidity = max(
        0.30 * _ramp(weather.dewpoint_f, 75, 85),
        humidity_spike,
        0.15 * _ramp(weather.dewpoint_f, 70, 80) if not has_baseline else 0.0,
    )
    cold = max(
        0.80 * _ramp(-apparent if apparent is not None else None, -20, 10),
        cold_spike,
    )
    wind = 0.80 * max(
        _ramp(weather.wind_speed_mph, 20, 39),
        _ramp(weather.wind_gust_mph, 30, 58),
    )
    # Ordinary rain is execution context, not a reason to delete training.
    precipitation = 0.0
    relative_spike = max(heat_spike, humidity_spike, cold_spike)
    components = sorted(
        (heat, humidity, cold, wind, precipitation), reverse=True
    )
    score = min(1.5, components[0] + 0.25 * components[1] + 0.10 * components[2])
    extreme_reasons = _extreme_reasons(weather, apparent)
    if extreme_reasons:
        score = 1.5
    band = (
        "extreme"
        if extreme_reasons
        else "none"
        if score < 0.20
        else "mild"
        if score < 0.40
        else "moderate"
        if score < 0.85
        else "high"
        if score < 1.20
        else "severe"
    )
    return TrainingWeatherStress(
        score=score,
        band=band,
        heat=heat,
        humidity=humidity,
        cold=cold,
        wind=wind,
        precipitation=precipitation,
        relative_spike=relative_spike,
        extreme_reasons=extreme_reasons,
    )
