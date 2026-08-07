from __future__ import annotations

from datetime import datetime, timezone

import pytest

from run_analysis.vo2_estimation import estimate_local_vo2
from run_analysis.web.schemas import (
    ConfidenceLevel,
    FitnessTrend,
    LoadContext,
    LoadWindow,
    PaceValue,
)


def _window(days: int, miles: float) -> LoadWindow:
    return LoadWindow(days=days, distance_miles=miles, moving_minutes=miles * 10, hard_minutes=0, activity_count=4)


def test_published_vo2_cross_checks_are_labeled_and_bounded() -> None:
    load = LoadContext(
        trailing_7d=_window(7, 12), trailing_14d=_window(14, 24), trailing_28d=_window(28, 48),
        confidence=ConfidenceLevel.HIGH,
    )
    estimate = estimate_local_vo2(
        current_pace=PaceValue(minutes_per_mile=10.925, display="10:56/mi"),
        as_of=datetime(2026, 8, 7, tzinfo=timezone.utc),
        recent_load=load,
        fitness_trend=FitnessTrend.UNCERTAIN,
        config={
            "target_hr": 145,
            "profile": {"birth_date": "1995-11-28", "sex": "male", "weight_lb": 140, "height_in": 68},
        },
    )
    assert estimate.value_ml_kg_min == pytest.approx(52.3, abs=0.1)
    assert estimate.demographic_baseline_ml_kg_min == pytest.approx(53.3, abs=0.1)
    assert estimate.confidence == ConfidenceLevel.LOW
    assert "not an independent" in estimate.interpretation
