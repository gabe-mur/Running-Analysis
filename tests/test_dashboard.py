from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from run_analysis.db import connect, initialize
from run_analysis.web.app import create_app
from test_web_phase1 import _write_config


def test_empty_dashboard_still_provides_conservative_next_step(tmp_path: Path) -> None:
    _write_config(tmp_path)
    database = tmp_path / "data" / "test.sqlite"
    with connect(database) as connection:
        initialize(connection)
    response = TestClient(create_app(tmp_path)).get("/api/dashboard")
    assert response.status_code == 200
    payload = response.json()
    assert payload["last_run"] is None
    assert payload["progress"]["window_days"] == 90
    assert payload["fitness_interpretation"]["short_term"]["window_days"] == 90
    assert payload["fitness_interpretation"]["long_term"]["window_days"] == 90
    assert payload["progress"]["fitness_trend"] == "insufficient_data"
    assert payload["recommendation"]["workout_type"] == "easy"
    assert payload["recommendation"]["rule_trace"]
    assert "starter plan" in payload["weekly_schedule"]["summary"].casefold()
    assert [item["label"] for item in payload["fitness_interpretation"]["signals"]] == [
        "Aerobic efficiency",
        "Durability",
        "Training capacity",
        "Training volume",
        "Recent form",
        "High-intensity fitness",
    ]


def _state(**changes):
    """Minimal fitness state for the recent-form rule."""
    from types import SimpleNamespace
    from run_analysis.web.schemas import ConfidenceLevel

    values = dict(
        recent_illness_or_recovery=False,
        normal_runs_since_health_event=0,
        recent_performance_anomaly="within_recent_range",
        trend_confidence=ConfidenceLevel.MODERATE,
    )
    values.update(changes)
    return SimpleNamespace(**values)


def test_recent_form_stays_recovering_immediately_after_a_health_tagged_run() -> None:
    from run_analysis.dashboard import _recent_form

    _trend, status, detail = _recent_form(
        _state(recent_illness_or_recovery=True, normal_runs_since_health_event=0)
    )
    assert status == "Recovering"
    assert "3 more" not in detail  # nothing to count yet


def test_recent_form_yields_to_measured_response_after_normal_runs() -> None:
    """Three normal runs later, the calendar is weaker evidence than the runs."""
    from run_analysis.dashboard import _recent_form

    _trend, status, detail = _recent_form(
        _state(recent_illness_or_recovery=True, normal_runs_since_health_event=3)
    )
    assert status == "Within recent range"
    # The illness remains visible as context rather than disappearing.
    assert "health-tagged run remains" in detail


def test_recent_form_counts_down_while_still_recovering() -> None:
    from run_analysis.dashboard import _recent_form

    _trend, status, detail = _recent_form(
        _state(recent_illness_or_recovery=True, normal_runs_since_health_event=1)
    )
    assert status == "Recovering"
    assert "2 more" in detail


def _load(current, previous, confidence=None):
    from types import SimpleNamespace
    from run_analysis.web.schemas import ConfidenceLevel

    return SimpleNamespace(
        capacity_reference_miles=current,
        previous_capacity_reference_miles=previous,
        confidence=confidence or ConfidenceLevel.HIGH,
    )


def test_training_capacity_uses_retained_capacity_not_period_mileage() -> None:
    from run_analysis.dashboard import _capacity_signal
    from run_analysis.web.schemas import FitnessTrend

    signal, direction = _capacity_signal(_load(28.0, 22.0), 28)
    assert direction == FitnessTrend.IMPROVING
    assert signal.status == "Improved"
    assert "28.0 mi/week" in signal.detail and "22.0 mi/week" in signal.detail

    signal, direction = _capacity_signal(_load(22.0, 22.5), 28)
    assert direction == FitnessTrend.STABLE
    assert signal.status == "Holding"

    signal, direction = _capacity_signal(_load(0.0, None), 28)
    assert direction == FitnessTrend.INSUFFICIENT_DATA
