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
        "Recent form",
        "High-intensity fitness",
    ]
