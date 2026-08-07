from __future__ import annotations

from pathlib import Path
from datetime import date, datetime, timedelta, timezone
import json
from zoneinfo import ZoneInfo

import yaml

from fastapi.testclient import TestClient

from run_analysis.db import connect
from run_analysis.web.app import create_app
from run_analysis.web.upload_service import UploadPayload, run_upload_pipeline
from test_tcx import TCX_TEMPLATE
from test_web_phase1 import _write_config


def _tcx_bytes(
    gps: bool = False,
    *,
    start: str = "2024-07-01T12:00:00Z",
    end: str = "2024-07-01T12:00:20Z",
) -> bytes:
    position = (
        "<Position><LatitudeDegrees>40.7</LatitudeDegrees>"
        "<LongitudeDegrees>-73.95</LongitudeDegrees></Position>"
        if gps
        else ""
    )
    return TCX_TEMPLATE.format(
        activity_id=start,
        start=start,
        end=end,
        notes="upload test",
        position=position,
        hr="<HeartRateBpm><Value>145</Value></HeartRateBpm>",
    ).encode()


def test_upload_rejects_non_tcx_before_writing(tmp_path: Path) -> None:
    _write_config(tmp_path)
    client = TestClient(create_app(tmp_path))
    response = client.post("/api/uploads", files=[("files", ("run.txt", b"nope", "text/plain"))])
    assert response.status_code == 422
    assert not (tmp_path / "uploads").exists()


def test_upload_pipeline_imports_and_reports_independent_stage_failures(tmp_path: Path) -> None:
    _write_config(tmp_path)
    result = run_upload_pipeline(
        tmp_path,
        "config.yaml",
        [UploadPayload("../My Run.tcx", _tcx_bytes(gps=False))],
    )
    assert result.files[0].status == "imported"
    assert result.files[0].activity_ids
    assert result.primary_activity_id == result.files[0].activity_ids[0]
    assert [stage.name for stage in result.stages] == ["save", "import", "process", "weather", "model", "schedule"]
    assert result.stages[2].status == "complete"
    assert result.stages[3].status == "complete"
    assert '"historical_weather_enabled": false' in result.stages[3].detail
    assert (tmp_path / "uploads").exists()
    assert all(path.parent == tmp_path / "uploads" for path in (tmp_path / "uploads").iterdir())


def test_upload_endpoint_accepts_multiple_tcx_files(tmp_path: Path) -> None:
    _write_config(tmp_path)
    client = TestClient(create_app(tmp_path))
    response = client.post(
        "/api/uploads",
        files=[
            ("files", ("first.tcx", _tcx_bytes(), "application/xml")),
            ("files", ("copy.tcx", _tcx_bytes(), "application/xml")),
        ],
    )
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["files"]) == 2
    assert {item["status"] for item in payload["files"]} <= {"imported", "duplicate", "unchanged"}
    assert payload["primary_activity_id"] is None


def test_uploading_todays_run_refreshes_tomorrow_forward_schedule(tmp_path: Path) -> None:
    config = yaml.safe_load((Path(__file__).parents[1] / "config.example.yaml").read_text())
    config["paths"].update(
        {
            "database": "data/test.sqlite",
            "report": "output/report.html",
            "weather_cache": "data/weather_cache",
            "overrides": "run_overrides.csv",
        }
    )
    config["weather"]["estimated_location_sources"] = {}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    start = datetime.now(timezone.utc).replace(microsecond=0)
    end = start + timedelta(seconds=20)
    result = run_upload_pipeline(
        tmp_path,
        "config.yaml",
        [
            UploadPayload(
                "today.tcx",
                _tcx_bytes(
                    gps=False,
                    start=start.isoformat().replace("+00:00", "Z"),
                    end=end.isoformat().replace("+00:00", "Z"),
                ),
            )
        ],
    )
    assert next(stage for stage in result.stages if stage.name == "model").status == "deferred"
    assert next(stage for stage in result.stages if stage.name == "schedule").status == "complete"
    with connect(tmp_path / "data" / "test.sqlite") as connection:
        saved = connection.execute(
            "SELECT value_json FROM app_state WHERE key='weekly_schedule'"
        ).fetchone()
    schedule = json.loads(saved[0])
    local_today = datetime.now(timezone.utc).astimezone(
        ZoneInfo(config["timezone_default"])
    ).date()
    assert date.fromisoformat(schedule["start_date"]) == local_today + timedelta(days=1)
    assert schedule["days"][0]["date"] == schedule["start_date"]
    assert schedule["trailing_days"][-1]["date"] == local_today.isoformat()
    assert schedule["trailing_days"][-1]["activities"]
