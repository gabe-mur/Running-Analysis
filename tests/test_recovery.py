from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace

import pytest

from run_analysis.fitness_state import _cumulative_recovery_residual
from run_analysis.recovery import (
    EASY_RUN_RESIDUAL_LIMIT,
    TAXING_RUN_RESIDUAL_LIMIT,
    athlete_relative_session_load,
    cumulative_recovery_load,
    estimate_recovery,
    prior_typical_load,
)
from run_analysis.web.schemas import LoadWindow
from run_analysis.web.schemas import WorkoutType
from test_recommendation import _difficulty, _state


def test_recent_session_recovery_residue_is_additive() -> None:
    residual = cumulative_recovery_load(
        [
            (1.0, 12.0),
            (1.0, 24.0),
        ]
    )

    assert residual == pytest.approx(0.75)
    assert residual > cumulative_recovery_load([(1.0, 12.0)])


def test_recovery_uses_persisted_multi_session_residue() -> None:
    state = _state(
        days_since_last_run=1.0,
        last_run=_difficulty(miles=4),
    ).model_copy(update={"recovery_residual_load": 0.9})

    recovery = estimate_recovery(state)

    assert recovery is not None
    assert recovery.residual_load == pytest.approx(0.9)


def test_fitness_state_recovery_residue_keeps_more_than_latest_run() -> None:
    state = _state()
    recent = SimpleNamespace(
        start_time=state.as_of - timedelta(hours=12),
        session_difficulty=_difficulty(miles=4),
        workout_type=WorkoutType.EASY,
    )
    prior = SimpleNamespace(
        start_time=state.as_of - timedelta(hours=24),
        session_difficulty=_difficulty(miles=8, long=True),
        workout_type=WorkoutType.LONG,
    )

    latest_only = _cumulative_recovery_residual(
        [recent], state.as_of, state.recent_load.trailing_28d
    )
    combined = _cumulative_recovery_residual(
        [recent, prior], state.as_of, state.recent_load.trailing_28d
    )

    assert combined > latest_only


def test_session_load_scales_with_athlete_relative_work() -> None:
    state = _state()
    ordinary, _ = athlete_relative_session_load(
        _difficulty(miles=4), state.recent_load.trailing_28d
    )
    long, _ = athlete_relative_session_load(
        _difficulty(miles=8), state.recent_load.trailing_28d
    )

    assert long > ordinary
    assert long / ordinary > 1.5


def test_workout_role_does_not_change_completed_session_load() -> None:
    state = _state()
    ordinary, ordinary_evidence = athlete_relative_session_load(
        _difficulty(miles=5), state.recent_load.trailing_28d
    )
    quality, quality_evidence = athlete_relative_session_load(
        _difficulty(miles=5, quality=True), state.recent_load.trailing_28d
    )
    assert quality == ordinary
    assert ordinary_evidence == quality_evidence


def test_long_label_does_not_change_completed_session_load() -> None:
    state = _state()
    ordinary, _ = athlete_relative_session_load(
        _difficulty(miles=5), state.recent_load.trailing_28d
    )
    long, _ = athlete_relative_session_load(
        _difficulty(miles=5, long=True), state.recent_load.trailing_28d
    )

    assert long == ordinary


def test_observed_hr_load_changes_recovery_even_when_run_is_labeled_easy() -> None:
    state = _state()
    ordinary_session = _difficulty(miles=5).model_copy(update={"zone_load": 100})
    blown_up_easy = ordinary_session.model_copy(update={"zone_load": 160})

    ordinary, _ = athlete_relative_session_load(
        ordinary_session, state.recent_load.trailing_28d
    )
    costly, _ = athlete_relative_session_load(
        blown_up_easy, state.recent_load.trailing_28d
    )

    assert costly > ordinary


def test_actual_duration_changes_recovery_at_equal_distance_and_hr_load() -> None:
    state = _state()
    ordinary_session = _difficulty(miles=5)
    prolonged_session = ordinary_session.model_copy(
        update={"moving_minutes": 75, "elapsed_minutes": 75}
    )

    ordinary, _ = athlete_relative_session_load(
        ordinary_session, state.recent_load.trailing_28d
    )
    prolonged, _ = athlete_relative_session_load(
        prolonged_session, state.recent_load.trailing_28d
    )

    assert prolonged > ordinary


def test_recovery_baseline_excludes_the_session_being_evaluated() -> None:
    state = _state(
        days_since_last_run=0,
        last_run=_difficulty(miles=8),
    )

    prior = prior_typical_load(state)

    assert prior.activity_count == state.recent_load.trailing_28d.activity_count - 1
    assert prior.distance_miles == state.recent_load.trailing_28d.distance_miles - 8
    assert prior.moving_minutes == state.recent_load.trailing_28d.moving_minutes - 88


def test_zone_load_ratio_uses_only_runs_with_known_hr_load() -> None:
    typical = LoadWindow(
        days=28,
        distance_miles=40,
        moving_minutes=440,
        zone_load=200,
        hard_minutes=10,
        activity_count=10,
        zone_load_activity_count=2,
    )

    _, evidence = athlete_relative_session_load(
        _difficulty(miles=4), typical
    )

    assert evidence["zone_load_ratio"] == 1.0


def test_recovery_decays_smoothly_across_old_hour_boundary() -> None:
    session = _difficulty(long=True, miles=8)
    before = estimate_recovery(
        _state(days_since_last_run=35.9 / 24, last_run=session)
    )
    after = estimate_recovery(
        _state(days_since_last_run=36.1 / 24, last_run=session)
    )

    assert before is not None and after is not None
    assert before.residual_load > after.residual_load
    assert before.residual_load - after.residual_load < 0.01


def test_easy_running_becomes_ready_before_another_taxing_session() -> None:
    recovery = estimate_recovery(
        _state(days_since_last_run=0, last_run=_difficulty(long=True, miles=8))
    )

    assert recovery is not None
    assert EASY_RUN_RESIDUAL_LIMIT > TAXING_RUN_RESIDUAL_LIMIT
    assert recovery.hours_until_easy < recovery.hours_until_taxing


def test_strong_whole_run_response_discounts_drift_recovery_cost() -> None:
    ordinary = estimate_recovery(
        _state(
            days_since_last_run=0,
            last_run=_difficulty(miles=5),
            last_run_drift_percent=10,
        )
    )
    strong = estimate_recovery(
        _state(
            days_since_last_run=0,
            last_run=_difficulty(miles=5),
            last_run_drift_percent=10,
            recent_performance_anomaly="unusually_strong",
        )
    )

    assert ordinary is not None and strong is not None
    assert strong.initial_load < ordinary.initial_load
