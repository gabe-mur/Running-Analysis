from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from run_analysis.db import connect, initialize
from run_analysis.external_fitness import save_snapshot, summarize_external_fitness
from run_analysis.web.app import create_app
from run_analysis.web.schemas import ExternalFitnessSnapshotInput, FitnessTrend
from test_web_phase1 import _write_config


def test_garmin_snapshots_are_a_separate_improving_signal(tmp_path: Path) -> None:
    database = tmp_path / "fitness.sqlite"
    with connect(database) as connection:
        initialize(connection)
        save_snapshot(connection, ExternalFitnessSnapshotInput(
            measured_at="2026-01-10", vo2_max=44, predicted_5k_seconds=1680
        ))
        save_snapshot(connection, ExternalFitnessSnapshotInput(
            measured_at="2026-07-10", vo2_max=48, predicted_5k_seconds=1500
        ))
        summary = summarize_external_fitness(connection, datetime(2026, 8, 1, tzinfo=timezone.utc))
    assert summary.vo2_max_trend == FitnessTrend.IMPROVING
    assert summary.race_prediction_trend == FitnessTrend.IMPROVING
    assert "both improved" in summary.interpretation


def test_external_fitness_api_persists_a_snapshot(tmp_path: Path) -> None:
    _write_config(tmp_path)
    database = tmp_path / "data" / "test.sqlite"
    with connect(database) as connection:
        initialize(connection)
    client = TestClient(create_app(tmp_path))
    response = client.post("/api/external-fitness", json={
        "measured_at": "2026-08-01", "vo2_max": 47.0,
        "predicted_5k_seconds": 1560, "source": "Garmin",
    })
    assert response.status_code == 200
    summary = client.get("/api/external-fitness").json()
    assert len(summary["snapshots"]) == 1
