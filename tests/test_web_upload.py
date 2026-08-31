from __future__ import annotations

from pathlib import Path
from datetime import date, datetime, timedelta, timezone
import json
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import yaml

from fastapi.testclient import TestClient

from run_analysis.db import connect
from run_analysis.recommendation_service import (
    _today_plan_time_is_stale,
    _weekly_emergency_alerts_are_stale,
    _weekly_plan_shape_is_stale,
)
from run_analysis.weekly_schedule import WEEKLY_PLANNER_VERSION
from run_analysis.web.schemas import WorkoutType
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


def test_uploading_todays_run_refreshes_today_forward_schedule(tmp_path: Path) -> None:
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
    assert date.fromisoformat(schedule["start_date"]) == local_today
    assert schedule["days"][0]["date"] == schedule["start_date"]
    assert schedule["days"][0]["recommendation"] is None
    assert schedule["days"][0]["completed_activities"]
    assert schedule["completed_run_count"] == 1
    assert schedule["trailing_days"][0]["date"] == (
        local_today - timedelta(days=7)
    ).isoformat()
    assert schedule["trailing_days"][-1]["date"] == (
        local_today - timedelta(days=1)
    ).isoformat()
    assert all(day["date"] != local_today.isoformat() for day in schedule["trailing_days"])


def test_passed_early_slot_refreshes_but_evening_slot_lasts_until_midnight() -> None:
    now = datetime(2026, 8, 25, 22, tzinfo=timezone.utc)
    config = {"weather": {"automatic_run_time_hours_local": [7, 12, 19]}}

    def schedule_at(hour: int):
        return SimpleNamespace(
            days=[
                SimpleNamespace(
                    date=now.date(),
                    recommendation=object(),
                    planned_at=now.replace(hour=hour),
                )
            ]
        )

    assert _today_plan_time_is_stale(schedule_at(7), now, config) is True
    assert _today_plan_time_is_stale(schedule_at(19), now, config) is False


def test_ordinary_today_rest_refreshes_when_an_early_option_closes() -> None:
    zone = ZoneInfo("America/New_York")
    now = datetime(2026, 8, 28, 12, 5, tzinfo=zone)
    config = {"weather": {"automatic_run_time_hours_local": [7, 12, 19]}}

    def rest_schedule(*, generated_at: datetime, forced: bool = False, completed=False):
        return SimpleNamespace(
            generated_at=generated_at,
            days=[
                SimpleNamespace(
                    date=now.date(),
                    recommendation=None,
                    forced_rest=forced,
                    completed_activities=[object()] if completed else [],
                )
            ],
        )

    # Noon stopped being a future option at 11:50, so the remaining week must
    # be optimized again instead of preserving the ordinary rest day all day.
    assert _today_plan_time_is_stale(
        rest_schedule(generated_at=now.replace(hour=11, minute=29)), now, config
    ) is True
    assert _today_plan_time_is_stale(
        rest_schedule(generated_at=now.replace(hour=11, minute=55)), now, config
    ) is False
    assert _today_plan_time_is_stale(
        rest_schedule(generated_at=now.replace(hour=11, minute=29), forced=True),
        now,
        config,
    ) is False
    assert _today_plan_time_is_stale(
        rest_schedule(generated_at=now.replace(hour=11, minute=29), completed=True),
        now,
        config,
    ) is False


def test_actionable_today_plan_refreshes_official_alerts_without_staling_rest() -> None:
    now = datetime(2026, 8, 28, 18, tzinfo=timezone.utc)
    config = {"weather": {"emergency_alerts_enabled": True}}

    def schedule(*, age_seconds: int, workout=WorkoutType.EASY):
        return SimpleNamespace(
            generated_at=now - timedelta(seconds=age_seconds),
            days=[
                SimpleNamespace(
                    date=now.date(),
                    recommendation=(
                        None
                        if workout is None
                        else SimpleNamespace(workout_type=workout)
                    ),
                )
            ],
        )

    assert _weekly_emergency_alerts_are_stale(
        schedule(age_seconds=61), now, config
    ) is True
    assert _weekly_emergency_alerts_are_stale(
        schedule(age_seconds=30), now, config
    ) is False
    assert _weekly_emergency_alerts_are_stale(
        schedule(age_seconds=61, workout=None), now, config
    ) is False


def test_saved_plan_above_visible_target_is_not_stale_by_shape_alone() -> None:
    today = date(2026, 8, 28)
    schedule = SimpleNamespace(
        planner_version=WEEKLY_PLANNER_VERSION,
        projected_distance_range_miles=(11.5, 13.0),
        target_distance_range_miles=(9.5, 11.0),
        start_date=today,
        trailing_days=[
            SimpleNamespace(date=today - timedelta(days=offset))
            for offset in range(7, 0, -1)
        ],
    )

    assert _weekly_plan_shape_is_stale(schedule) is False


def test_saved_plan_from_previous_planner_version_is_stale() -> None:
    schedule = SimpleNamespace(
        planner_version=1,
        projected_distance_range_miles=(9.5, 11.0),
        target_distance_range_miles=(9.5, 11.0),
    )

    assert _weekly_plan_shape_is_stale(schedule) is True


def test_saved_plan_that_includes_today_in_recent_history_is_stale() -> None:
    today = date(2026, 8, 28)
    schedule = SimpleNamespace(
        planner_version=WEEKLY_PLANNER_VERSION,
        projected_distance_range_miles=(9.5, 11.0),
        target_distance_range_miles=(9.5, 11.0),
        start_date=today,
        trailing_days=[
            SimpleNamespace(date=today - timedelta(days=offset))
            for offset in range(6, -1, -1)
        ],
    )

    assert _weekly_plan_shape_is_stale(schedule) is True


def test_rest_day_constraint_persists_replans_and_can_be_removed(tmp_path: Path) -> None:
    _write_config(tmp_path)
    client = TestClient(create_app(tmp_path))
    original = client.get("/api/weekly-schedule/latest")
    assert original.status_code == 200
    original_plan = original.json()
    selected = next(day for day in original_plan["days"] if day["recommendation"])

    forced = client.post(
        "/api/weekly-schedule/rest-day",
        json={"date": selected["date"], "is_rest_day": True},
    )

    assert forced.status_code == 200
    forced_plan = forced.json()
    forced_day = next(
        day for day in forced_plan["days"] if day["date"] == selected["date"]
    )
    assert forced_day["forced_rest"] is True
    assert forced_day["day_role"] == "forced_rest_day"
    assert forced_day["recommendation"] is None
    assert forced_plan["run_count"] == forced_plan["target_run_count"]
    assert {
        day["date"]
        for day in forced_plan["days"]
        if day["recommendation"]
    } != {
        day["date"]
        for day in original_plan["days"]
        if day["recommendation"]
    }
    persisted = client.get("/api/weekly-schedule/latest").json()
    assert next(
        day for day in persisted["days"] if day["date"] == selected["date"]
    )["forced_rest"] is True

    removed = client.post(
        "/api/weekly-schedule/rest-day",
        json={"date": selected["date"], "is_rest_day": False},
    )

    assert removed.status_code == 200
    assert next(
        day for day in removed.json()["days"] if day["date"] == selected["date"]
    )["forced_rest"] is False
