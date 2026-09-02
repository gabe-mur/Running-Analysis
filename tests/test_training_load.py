from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from run_analysis.training_load import (
    TrainingSession,
    acute_to_prior_weekly_ratio,
    calculate_session_load,
    continuous_distance_rate,
    continuous_fatigue_load,
    distance_capacity,
    rolling_load,
    short_term_density_half_life_days,
)


def test_session_load_preserves_distance_independent_intensity_context() -> None:
    load = calculate_session_load(
        {"z1": 600, "z2": 1200, "z3": 300, "z4": 120, "z5": 60, "unknown": 120},
        2400,
    )
    assert load.zone_load == pytest.approx(10 + 40 + 15 + 8 + 5)
    assert load.easy_minutes == 30
    assert load.moderate_minutes == 5
    assert load.hard_minutes == 3
    assert load.hr_coverage == pytest.approx(0.95)


def test_zone_load_is_missing_when_hr_coverage_is_too_low() -> None:
    load = calculate_session_load({"z2": 600, "unknown": 1800}, 2400)
    assert load.zone_load is None
    assert load.unknown_hr_minutes == 30


def test_rolling_load_counts_duration_distance_and_intensity() -> None:
    anchor = datetime(2026, 8, 7, tzinfo=timezone.utc)
    sessions = [
        TrainingSession(1, anchor - timedelta(days=2), 8, 80, 160, 10),
        TrainingSession(2, anchor - timedelta(days=10), 2, 20, 25, 0),
    ]
    seven = rolling_load(sessions, anchor, 7)
    twenty_eight = rolling_load(sessions, anchor, 28)
    assert (seven.distance_miles, seven.activity_count, seven.zone_load) == (8, 1, 160)
    assert (twenty_eight.distance_miles, twenty_eight.activity_count, twenty_eight.zone_load) == (10, 2, 185)


def test_continuous_fatigue_has_no_seven_day_cliff() -> None:
    anchor = datetime(2026, 8, 7, tzinfo=timezone.utc)
    just_inside = TrainingSession(
        1, anchor - timedelta(days=6.99), 4, 44, 88, 0
    )
    just_outside = TrainingSession(
        1, anchor - timedelta(days=7.01), 4, 44, 88, 0
    )
    recent_context = [
        TrainingSession(2, anchor - timedelta(days=2), 4, 44, 88, 0),
        TrainingSession(3, anchor - timedelta(days=4), 4, 44, 88, 0),
    ]

    inside = continuous_fatigue_load([just_inside, *recent_context], anchor)
    outside = continuous_fatigue_load([just_outside, *recent_context], anchor)

    assert inside.equivalent_weekly_miles == pytest.approx(
        outside.equivalent_weekly_miles, rel=0.01
    )


def test_continuous_distance_rate_has_no_reload_or_seven_day_boundary() -> None:
    anchor = datetime(2026, 9, 2, 19, tzinfo=timezone.utc)
    runs = [
        TrainingSession(index, anchor - timedelta(days=age), 4.0, 40.0, None, 0.0)
        for index, age in enumerate((1, 3, 5, 7, 9, 11, 13))
    ]

    now = continuous_distance_rate(runs, anchor)
    tomorrow = continuous_distance_rate(runs, anchor + timedelta(days=1))

    assert tomorrow == pytest.approx(now * 0.5 ** (1 / 7))


def test_short_term_density_timescale_is_derived_from_recovery_and_load() -> None:
    half_life = short_term_density_half_life_days(
        7.0, recovery_half_life_hours=12.0
    )

    assert half_life == pytest.approx((7.0 * 0.5) ** 0.5)
    assert 0.5 < half_life < 7.0


def test_continuous_fatigue_prices_intensity_without_a_workout_bucket() -> None:
    anchor = datetime(2026, 8, 7, tzinfo=timezone.utc)
    history = [
        TrainingSession(index, anchor - timedelta(days=day), 4, 44, 88, 0)
        for index, day in enumerate(range(2, 28, 3), start=1)
    ]
    ordinary = TrainingSession(100, anchor - timedelta(hours=1), 4, 44, 88, 0)
    harder = TrainingSession(100, anchor - timedelta(hours=1), 4, 44, 176, 8)

    ordinary_load = continuous_fatigue_load([*history, ordinary], anchor)
    harder_load = continuous_fatigue_load([*history, harder], anchor)

    assert harder_load.equivalent_weekly_miles > ordinary_load.equivalent_weekly_miles


def test_acute_ratio_uses_prior_four_week_weekly_mean() -> None:
    anchor = datetime(2026, 8, 7, tzinfo=timezone.utc)
    sessions = [TrainingSession(1, anchor - timedelta(days=2), 5, 50, 100, 0)]
    sessions += [
        TrainingSession(index + 2, anchor - timedelta(days=9 + index * 7), 5, 50, 100, 0)
        for index in range(4)
    ]
    assert acute_to_prior_weekly_ratio(sessions, anchor) == pytest.approx(1.0)


def test_retained_capacity_prevents_short_disruption_from_becoming_the_new_normal() -> None:
    anchor = datetime(2026, 8, 7, tzinfo=timezone.utc)
    sessions = [
        TrainingSession(index, anchor - timedelta(days=day), 4, 44, 80, 0)
        for index, day in enumerate((1, 3, 5, 6), start=1)
    ]
    sessions += [
        TrainingSession(index + 10, anchor - timedelta(days=day), 4, 44, 80, 0)
        for index, day in enumerate((10, 24), start=1)
    ]
    sessions += [
        TrainingSession(index + 20, anchor - timedelta(days=day), 4, 44, 80, 0)
        for index, day in enumerate(
            day for day in range(36, 64) if day % 7 in {0, 1, 3, 5}
        )
    ]
    raw = acute_to_prior_weekly_ratio(sessions, anchor)
    capacity = distance_capacity(sessions, anchor)
    assert raw is not None and raw > 2
    assert capacity.sustained_weekly_miles >= 15
    assert capacity.acute_to_capacity_ratio is not None
    assert capacity.acute_to_capacity_ratio < 1.3
