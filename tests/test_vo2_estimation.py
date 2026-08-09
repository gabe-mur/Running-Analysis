from __future__ import annotations

from datetime import datetime, timezone

import pytest

from run_analysis.vo2_estimation import (
    acsm_running_vo2,
    estimate_local_vo2,
    george_vo2_max,
    vo2_max_from_reserve,
)
from run_analysis.web.schemas import (
    ConfidenceLevel,
    FitnessTrend,
    LoadContext,
    LoadWindow,
    PaceValue,
)


def _window(days: int, miles: float) -> LoadWindow:
    return LoadWindow(days=days, distance_miles=miles, moving_minutes=miles * 10, hard_minutes=0, activity_count=4)


def _load(miles_28d: float = 48) -> LoadContext:
    return LoadContext(
        trailing_7d=_window(7, miles_28d / 4),
        trailing_14d=_window(14, miles_28d / 2),
        trailing_28d=_window(28, miles_28d),
        confidence=ConfidenceLevel.HIGH,
    )


def _config(**changes) -> dict:
    config = {
        "max_hr": 190,
        "resting_hr": 55,
        "target_hr": 145,
        "profile": {
            "birth_date": "1990-01-01",
            "sex": "male",
            "weight_lb": 150,
            "height_in": 70,
        },
    }
    config.update(changes)
    return config


def _estimate(pace: float = 10.925, uncertainty=None, **changes):
    return estimate_local_vo2(
        current_pace=PaceValue(minutes_per_mile=pace, display="x"),
        current_pace_uncertainty_95=uncertainty,
        as_of=datetime(2026, 8, 7, tzinfo=timezone.utc),
        recent_load=_load(),
        fitness_trend=FitnessTrend.UNCERTAIN,
        config=_config(**changes),
    )


def test_acsm_running_equation_matches_the_published_form() -> None:
    # 10 mph = 268.224 m/min level: 0.2 * 268.224 + 3.5
    assert acsm_running_vo2(268.224) == pytest.approx(57.14, abs=0.01)
    # A 5% grade adds 0.9 * S * 0.05.
    assert acsm_running_vo2(200.0, 0.05) == pytest.approx(0.2 * 200 + 0.9 * 200 * 0.05 + 3.5)


def test_reserve_extrapolation_inverts_the_percent_hrr_identity() -> None:
    # At 50% of heart-rate reserve, VO2 reserve must also be 50%.
    submaximal = vo2_max_from_reserve(
        submaximal_vo2=30.0, submaximal_hr=122.5, resting_hr=55, maximum_hr=190
    )
    assert submaximal == pytest.approx(3.5 + (30.0 - 3.5) * 2, abs=0.01)


def test_estimate_pools_two_agreeing_equations() -> None:
    estimate = _estimate()
    reserve = vo2_max_from_reserve(
        acsm_running_vo2((60.0 / 10.925) * 26.8224), 145, 55, 190
    )
    george = george_vo2_max(10.925, 145, 150 * 0.45359237, True)
    assert estimate.value_ml_kg_min is not None
    # The pooled value must lie between the two estimators it pooled.
    assert min(reserve, george) < estimate.value_ml_kg_min < max(reserve, george)
    assert "agree" in estimate.interpretation
    assert estimate.uncertainty_95_ml_kg_min is not None


def test_a_measured_maximum_heart_rate_narrows_the_interval() -> None:
    """The estimate must reward measuring max HR, and say that it does."""
    estimated = _estimate()
    measured = _estimate(profile={**_config()["profile"], "max_hr_source": "measured"})
    assert measured.uncertainty_95_ml_kg_min < estimated.uncertainty_95_ml_kg_min
    assert any("Measuring it would" in item for item in estimated.limitations)
    assert any("recorded as measured" in item for item in measured.limitations)


def test_pace_uncertainty_widens_the_interval() -> None:
    tight = _estimate(uncertainty=0.05)
    loose = _estimate(uncertainty=0.60)
    assert loose.uncertainty_95_ml_kg_min > tight.uncertainty_95_ml_kg_min


def test_a_pace_below_the_acsm_running_range_produces_no_number() -> None:
    """Better to show nothing than to extrapolate past the validation."""
    estimate = _estimate(pace=13.0)
    assert estimate.value_ml_kg_min is None
    assert estimate.confidence == ConfidenceLevel.UNAVAILABLE
    assert "134 m/min" in estimate.limitations[0]


def test_a_comparison_hr_outside_the_usable_reserve_band_produces_no_number() -> None:
    estimate = _estimate(target_hr=60)  # barely above resting
    assert estimate.value_ml_kg_min is None
    assert "heart-rate reserve" in estimate.limitations[0]


def test_missing_heart_rates_explain_themselves_instead_of_crashing() -> None:
    estimate = estimate_local_vo2(
        current_pace=PaceValue(minutes_per_mile=9.0, display="x"),
        as_of=datetime(2026, 8, 7, tzinfo=timezone.utc),
        recent_load=_load(),
        fitness_trend=FitnessTrend.UNCERTAIN,
        config={"profile": _config()["profile"]},
    )
    assert estimate.confidence == ConfidenceLevel.UNAVAILABLE
    assert "resting" in estimate.limitations[0]


def test_the_estimate_never_claims_to_be_independent_evidence() -> None:
    estimate = _estimate()
    assert any("not a second opinion" in item for item in estimate.limitations)


def test_demographic_baseline_needs_documented_running_volume() -> None:
    quiet = estimate_local_vo2(
        current_pace=PaceValue(minutes_per_mile=10.925, display="x"),
        as_of=datetime(2026, 8, 7, tzinfo=timezone.utc),
        recent_load=_load(miles_28d=20),  # 5 mi/week
        fitness_trend=FitnessTrend.UNCERTAIN,
        config=_config(),
    )
    assert quiet.demographic_baseline_ml_kg_min is None
    assert _estimate().demographic_baseline_ml_kg_min == pytest.approx(50.9, abs=0.1)
