from __future__ import annotations

from datetime import datetime, timezone

from run_analysis.training_status import build_training_status
from run_analysis.web.schemas import (
    ConfidenceLevel,
    CurrentHealthStatus,
    FitnessState,
    FitnessTrend,
    LoadContext,
    LoadWindow,
    TrainingStatus,
)


def _window(days: int, miles: float, activities: int = 12) -> LoadWindow:
    return LoadWindow(
        days=days,
        distance_miles=miles,
        moving_minutes=miles * 10,
        zone_load=miles * 10,
        hard_minutes=5,
        activity_count=activities,
    )


def _load(
    *,
    weekly_28d: float = 20.0,
    capacity: float = 20.0,
    acute_ratio: float | None = 1.0,
    previous_capacity: float | None = 20.0,
    activities: int = 12,
) -> LoadContext:
    return LoadContext(
        trailing_7d=_window(7, weekly_28d, 3),
        trailing_14d=_window(14, weekly_28d * 2, 6),
        trailing_28d=_window(28, weekly_28d * 4, activities),
        acute_distance_to_capacity_ratio=acute_ratio,
        capacity_reference_miles=capacity,
        previous_capacity_reference_miles=previous_capacity,
        confidence=ConfidenceLevel.HIGH,
    )


def _state(**changes) -> FitnessState:
    values = dict(
        as_of=datetime(2026, 8, 8, tzinfo=timezone.utc),
        window_days=28,
        fitness_trend=FitnessTrend.STABLE,
        trend_confidence=ConfidenceLevel.MODERATE,
        recent_load=_load(),
        quality_sessions_14d=1,
        recent_performance_anomaly="within_recent_range",
        recent_illness_or_recovery=False,
        normal_runs_since_health_event=0,
        current_health_status=CurrentHealthStatus.NORMAL,
    )
    values.update(changes)
    return FitnessState(**values)


def _fired(summary) -> set[str]:
    return {item.rule_id for item in summary.rule_trace if item.fired}


def test_too_little_recent_running_is_not_classified() -> None:
    summary = build_training_status(_state(recent_load=_load(activities=2)))
    assert summary.status == TrainingStatus.INSUFFICIENT_DATA
    assert summary.confidence == ConfidenceLevel.UNAVAILABLE
    assert _fired(summary) == {"status_insufficient_evidence"}


def test_no_demonstrated_capacity_is_not_classified() -> None:
    summary = build_training_status(_state(recent_load=_load(capacity=0.0)))
    assert summary.status == TrainingStatus.INSUFFICIENT_DATA


def test_health_outranks_an_otherwise_ideal_load() -> None:
    """A sick athlete with perfect mileage is recovering, not building."""
    summary = build_training_status(
        _state(
            current_health_status=CurrentHealthStatus.SICK_OR_RECOVERING,
            fitness_trend=FitnessTrend.IMPROVING,
        )
    )
    assert summary.status == TrainingStatus.RECOVERING
    assert "status_strained" not in _fired(summary)


def test_being_a_little_tired_is_an_ordinary_training_day_not_recovery() -> None:
    summary = build_training_status(
        _state(current_health_status=CurrentHealthStatus.LITTLE_TIRED)
    )
    assert summary.status != TrainingStatus.RECOVERING


def test_pain_or_injury_concern_is_recovery() -> None:
    summary = build_training_status(
        _state(current_health_status=CurrentHealthStatus.PAIN_OR_INJURY_CONCERN)
    )
    assert summary.status == TrainingStatus.RECOVERING


def test_recent_illness_clears_once_normal_runs_follow() -> None:
    unresolved = build_training_status(
        _state(recent_illness_or_recovery=True, normal_runs_since_health_event=1)
    )
    assert unresolved.status == TrainingStatus.RECOVERING
    resolved = build_training_status(
        _state(recent_illness_or_recovery=True, normal_runs_since_health_event=3)
    )
    assert resolved.status != TrainingStatus.RECOVERING


def test_acute_load_far_above_capacity_is_strained() -> None:
    summary = build_training_status(_state(recent_load=_load(acute_ratio=1.45)))
    assert summary.status == TrainingStatus.STRAINED
    assert "145%" in summary.detail


def test_an_unusually_costly_response_is_strained_even_at_normal_volume() -> None:
    summary = build_training_status(_state(recent_performance_anomaly="unusually_costly"))
    assert summary.status == TrainingStatus.STRAINED
    assert "cost more effort" in summary.detail


def test_high_second_half_drift_is_strained() -> None:
    summary = build_training_status(_state(last_run_drift_percent=12.0))
    assert summary.status == TrainingStatus.STRAINED
    assert "drifted" in summary.detail


def test_below_capacity_but_climbing_is_rebuilding_not_a_deficiency() -> None:
    summary = build_training_status(
        _state(recent_load=_load(weekly_28d=12.0, capacity=20.0, acute_ratio=0.90))
    )
    assert summary.status == TrainingStatus.REBUILDING
    assert "climbing back" in summary.detail


def test_below_capacity_and_flat_is_underloaded() -> None:
    summary = build_training_status(
        _state(recent_load=_load(weekly_28d=12.0, capacity=20.0, acute_ratio=0.60))
    )
    assert summary.status == TrainingStatus.UNDERLOADED
    assert "not climbing back" in summary.detail


def test_near_capacity_with_quality_and_an_upward_signal_is_building() -> None:
    summary = build_training_status(
        _state(
            recent_load=_load(weekly_28d=19.0, capacity=20.0, acute_ratio=1.0),
            quality_sessions_14d=2,
            fitness_trend=FitnessTrend.IMPROVING,
        )
    )
    assert summary.status == TrainingStatus.BUILDING


def test_growing_capacity_alone_can_make_it_building() -> None:
    summary = build_training_status(
        _state(
            recent_load=_load(weekly_28d=19.0, capacity=22.0, previous_capacity=18.0, acute_ratio=1.0),
            quality_sessions_14d=1,
            fitness_trend=FitnessTrend.STABLE,
        )
    )
    assert summary.status == TrainingStatus.BUILDING


def test_steady_training_without_quality_is_maintaining_and_says_why() -> None:
    summary = build_training_status(
        _state(recent_load=_load(weekly_28d=19.0, capacity=20.0), quality_sessions_14d=0)
    )
    assert summary.status == TrainingStatus.MAINTAINING
    assert "no quality session" in summary.detail


def test_every_state_carries_the_rules_that_produced_it() -> None:
    """The point of this feature is that the athlete can see why."""
    for state in (
        _state(),
        _state(current_health_status=CurrentHealthStatus.SICK_OR_RECOVERING),
        _state(recent_load=_load(acute_ratio=1.5)),
        _state(recent_load=_load(weekly_28d=10.0, capacity=20.0, acute_ratio=0.4)),
    ):
        summary = build_training_status(state)
        assert summary.rule_trace
        assert len(_fired(summary)) == 1
        assert all(item.description for item in summary.rule_trace)
