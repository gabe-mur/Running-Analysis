from __future__ import annotations

from run_analysis.recovery import (
    EASY_RUN_RESIDUAL_LIMIT,
    TAXING_RUN_RESIDUAL_LIMIT,
    athlete_relative_session_load,
    estimate_recovery,
)
from test_recommendation import _difficulty, _state


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
