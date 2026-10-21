from __future__ import annotations

from pathlib import Path
import pytest

from fastapi.testclient import TestClient

from run_analysis.db import connect, initialize
from run_analysis.web.app import create_app
from test_web_phase1 import _write_config


def test_recorded_long_run_accomplishment_is_separate_from_supported_capacity(tmp_path,monkeypatch):
    from datetime import datetime,timedelta,timezone
    from run_analysis.config import load_config
    from run_analysis.fitness_state import build_fitness_state
    from test_progress import _insert_run
    _write_config(tmp_path);cfg=load_config(tmp_path/'config.yaml')
    now=datetime.now(timezone.utc)
    with connect(tmp_path/'data/test.sqlite') as db:
        initialize(db)
        _insert_run(db,1,now-timedelta(days=15),9.536,95,100)
        _insert_run(db,2,now-timedelta(days=40),12,120,100)
        hike=_insert_run(db,3,now-timedelta(days=1),15,200,100)
        db.execute("INSERT INTO run_overrides(activity_id,workout_type) VALUES ('external-3','hike')")
        db.commit()
        monkeypatch.setattr('run_analysis.fitness_state.supported_long_run_capacity',lambda *args,**kwargs:8.9)
        state=build_fitness_state(db,cfg,as_of=now)
    assert state.longest_run_30d_miles==8.9
    assert state.longest_completed_run_30d_miles==pytest.approx(9.536)


@pytest.mark.parametrize("window", [14, 28, 42, 56, 90, 180, 365])
def test_dashboard_uses_requested_progress_window_without_changing_training_context(tmp_path, window):
    _write_config(tmp_path)
    with connect(tmp_path / "data/test.sqlite") as connection:
        initialize(connection)
    client = TestClient(create_app(tmp_path))
    response = client.get(f"/api/dashboard?fitness_window_days={window}")
    assert response.status_code == 200
    payload = response.json()
    assert payload["fitness_interpretation"]["short_term"]["window_days"] == window
    assert payload["progress"]["window_days"] == 90
    assert payload["fitness_interpretation"]["long_term"]["window_days"] == 90
    aerobic = payload["fitness_interpretation"]["signals"][0]
    assert f"Last {window} days" in aerobic["detail"]
    assert client.get("/api/dashboard?fitness_window_days=30").status_code == 422


def test_dashboard_does_not_replace_uncertain_selected_window_with_improving_long_window(tmp_path):
    from run_analysis.config import load_config
    from run_analysis.progress import build_progress
    from run_analysis.fitness_state import build_fitness_state
    from run_analysis.dashboard import _interpret_fitness
    from run_analysis.web.schemas import FitnessSignal, FitnessTrend
    _write_config(tmp_path)
    cfg = load_config(tmp_path / "config.yaml")
    with connect(tmp_path / "data/test.sqlite") as connection:
        initialize(connection)
        short = build_progress(connection, 28, config=cfg).model_copy(update={
            "period_change": _aerobic_change("improving", "clear", -43),
            "within_window_trend": _aerobic_change("declining", "clear", 46),
        })
        long = build_progress(connection, 90, config=cfg).model_copy(update={
            "period_change": _aerobic_change("improving", "likely", -16),
            "fitness_trend": FitnessTrend.IMPROVING,
        })
        state = build_fitness_state(connection, cfg)
        result = _interpret_fitness(short, long, long, state, FitnessSignal(
            label="High-intensity fitness", trend="insufficient_data", status="Not enough data", confidence="unavailable", detail="Test",
        ))
    assert result.short_term.trend == FitnessTrend.UNCERTAIN
    assert result.long_term.trend == FitnessTrend.IMPROVING
    assert result.signals[0].trend == FitnessTrend.UNCERTAIN
    assert result.signals[0].status == "No clear change"
    assert result.headline == "Maintaining aerobic efficiency."
    assert "likely improving" not in result.headline


def _aerobic_change(direction: str, evidence: str, change: float):
    from run_analysis.web.schemas import AerobicChangeEvidence

    return AerobicChangeEvidence(
        basis="test",
        direction=direction,
        evidence=evidence,
        confidence="moderate",
        pace_change_seconds_per_mile=change,
        uncertainty_95_seconds_per_mile=20,
        probability_faster=0.85 if change < 0 else 0.15,
        run_count=6,
        coverage_fraction=0.75,
    )


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
    assert payload["fitness_interpretation"]["short_term"]["window_days"] == 28
    assert payload["fitness_interpretation"]["long_term"]["window_days"] == 90
    assert payload["progress"]["fitness_trend"] == "insufficient_data"
    assert payload["fitness_interpretation"]["headline"] == "More comparable runs are needed."
    assert payload["recommendation"]["workout_type"] == "easy"
    assert payload["recommendation"]["rule_trace"]
    assert payload['weekly_schedule'] is None
    assert payload['weekly_plan_needs_refresh']
    # Calibration is still provided when Weekly Plan is explicitly opened.
    weekly = TestClient(create_app(tmp_path)).get('/api/weekly-schedule/latest').json()
    assert weekly['run_count'] == 1
    assert weekly['target_evidence']['planning_mode'] == 'baseline_required'
    baseline = next(
        day["recommendation"]
        for day in weekly["days"]
        if day["recommendation"] is not None
    )
    assert baseline["duration_range_minutes"] == [10.0, 30.0]
    assert "10–30-minute" in weekly["summary"]
    assert [item["label"] for item in payload["fitness_interpretation"]["signals"]] == [
        "Aerobic efficiency",
        "Durability",
        "Training capacity",
        "Training volume",
        "Recent form",
        "High-intensity fitness",
    ]


def test_dashboard_horizon_can_show_gradual_within_period_change() -> None:
    from types import SimpleNamespace

    from run_analysis.dashboard import _horizon
    from run_analysis.web.schemas import ConfidenceLevel, FitnessTrend

    horizon = _horizon(
        "Recent",
        SimpleNamespace(
            window_days=28,
            period_change=_aerobic_change("uncertain", "inconclusive", -2),
            within_window_trend=_aerobic_change("improving", "likely", -14),
            fitness_trend=FitnessTrend.UNCERTAIN,
            fitness_confidence=ConfidenceLevel.HIGH,
            pace_change_seconds_per_mile=-2,
            current_pace=None,
        ),
    )

    assert horizon.trend == FitnessTrend.IMPROVING
    assert horizon.confidence == ConfidenceLevel.MODERATE
    assert horizon.pace_change_seconds_per_mile == -14


def test_dashboard_horizon_does_not_hide_conflicting_methods() -> None:
    from types import SimpleNamespace

    from run_analysis.dashboard import _horizon
    from run_analysis.web.schemas import ConfidenceLevel, FitnessTrend

    horizon = _horizon(
        "Recent",
        SimpleNamespace(
            window_days=28,
            period_change=_aerobic_change("improving", "likely", -12),
            within_window_trend=_aerobic_change("declining", "likely", 10),
            fitness_trend=FitnessTrend.UNCERTAIN,
            fitness_confidence=ConfidenceLevel.HIGH,
            pace_change_seconds_per_mile=-12,
            current_pace=None,
        ),
    )

    assert horizon.trend == FitnessTrend.UNCERTAIN


def test_likely_slope_does_not_override_opposite_period_point_estimate() -> None:
    from types import SimpleNamespace

    from run_analysis.dashboard import _horizon
    from run_analysis.web.schemas import ConfidenceLevel, FitnessTrend

    horizon = _horizon(
        "Recent",
        SimpleNamespace(
            window_days=28,
            # Inconclusive, but its point estimate leans faster.
            period_change=_aerobic_change("uncertain", "inconclusive", -2),
            within_window_trend=_aerobic_change("declining", "likely", 53),
            fitness_trend=FitnessTrend.UNCERTAIN,
            fitness_confidence=ConfidenceLevel.HIGH,
            pace_change_seconds_per_mile=-2,
            current_pace=None,
        ),
    )

    assert horizon.trend == FitnessTrend.UNCERTAIN
    assert horizon.pace_change_seconds_per_mile == -2


def _state(**changes):
    """Minimal fitness state for the recent-form rule."""
    from types import SimpleNamespace
    from run_analysis.web.schemas import ConfidenceLevel

    values = dict(
        recent_illness_or_recovery=False,
        normal_runs_since_health_event=0,
        recent_performance_response="within_recent_range",
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
