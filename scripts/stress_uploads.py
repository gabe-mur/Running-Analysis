"""Isolated stress runner using representative files from the local corpus.

The script never opens the live database for writing. It shifts timestamps in
memory, uploads into temporary project roots, and reports API/scheduler
invariants for empty, underload, overload, mixed-quality, duplicate, malformed,
and randomized histories.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from random import Random
import re
from tempfile import TemporaryDirectory
from time import perf_counter
from unittest.mock import patch

from fastapi.testclient import TestClient
import yaml

from run_analysis.config import load_config, resolve_project_path
from run_analysis.db import connect
from run_analysis.web.app import create_app
from run_analysis.web.upload_service import UploadPayload, run_upload_pipeline


ROOT = Path(__file__).resolve().parents[1]
TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z")


def _discover_source_files() -> dict[str, Path]:
    """Select useful templates without embedding private activity identifiers."""
    config = load_config(ROOT / "config.yaml")
    database = resolve_project_path(ROOT, config["paths"]["database"])
    selections = {
        "short": "a.total_distance_m BETWEEN 400 AND 3218.68 ORDER BY a.total_distance_m ASC",
        "ordinary": "a.total_distance_m BETWEEN 4828.02 AND 8046.72 ORDER BY a.start_time_utc_epoch DESC",
        "intervals": "a.lap_count >= 5 ORDER BY a.lap_count DESC, a.start_time_utc_epoch DESC",
        "no_gps": "a.gps_quality = 'gps_missing' ORDER BY a.start_time_utc_epoch DESC",
        "long": "a.total_distance_m IS NOT NULL ORDER BY a.total_distance_m DESC",
    }
    selected: dict[str, Path] = {}
    with connect(database) as connection:
        for kind, condition in selections.items():
            row = connection.execute(
                f"""
                SELECT sf.path
                FROM activities a
                JOIN activity_sources activity_source
                  ON activity_source.activity_id = a.id
                 AND activity_source.is_primary = 1
                JOIN source_files sf ON sf.id = activity_source.source_file_id
                WHERE a.sport = 'Running' AND {condition}
                LIMIT 1
                """
            ).fetchone()
            if row is None:
                raise RuntimeError(f"No local {kind} activity is available for stress testing")
            path = Path(row["path"])
            selected[kind] = path if path.is_absolute() else ROOT / path
    return selected


def _first_timestamp(content: str) -> datetime:
    match = TIMESTAMP.search(content)
    if match is None:
        raise ValueError("TCX template has no UTC timestamp")
    return datetime.fromisoformat(match.group(0).replace("Z", "+00:00"))


def shifted_payload(
    source_files: dict[str, Path], kind: str, start: datetime, index: int
) -> UploadPayload:
    source = source_files[kind]
    content = source.read_text(encoding="utf-8-sig")
    delta = start.astimezone(timezone.utc) - _first_timestamp(content)

    def replace(match: re.Match[str]) -> str:
        original = datetime.fromisoformat(match.group(0).replace("Z", "+00:00"))
        shifted = original + delta
        if "." in match.group(0):
            return shifted.isoformat(timespec="milliseconds").replace("+00:00", "Z")
        return shifted.isoformat(timespec="seconds").replace("+00:00", "Z")

    return UploadPayload(
        filename=f"stress-{kind}-{index:03d}.tcx",
        content=TIMESTAMP.sub(replace, content).encode("utf-8"),
    )


def _write_isolated_config(root: Path) -> None:
    config = load_config(ROOT / "config.yaml")
    config["paths"] = {
        "database": "data/stress.sqlite",
        "report": "output/report.html",
        "weather_cache": "data/weather_cache",
        "overrides": "run_overrides.csv",
    }
    config["weather"]["forecast_enabled"] = False
    config["weather"]["estimated_location_sources"] = {}
    (root / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")


def _api_snapshot(client: TestClient) -> dict:
    endpoints = {
        "root": "/",
        "dashboard": "/api/dashboard",
        "progress": "/api/progress?window_days=90",
        "runs": "/api/runs?limit=100&sort_by=date&sort_order=desc",
        "schedule": "/api/weekly-schedule/latest",
        "settings": "/api/settings",
    }
    responses = {name: client.get(path) for name, path in endpoints.items()}
    dashboard = responses["dashboard"].json() if responses["dashboard"].status_code == 200 else {}
    schedule = responses["schedule"].json() if responses["schedule"].status_code == 200 else {}
    return {
        "status_codes": {name: response.status_code for name, response in responses.items()},
        "dashboard_trend": dashboard.get("progress", {}).get("fitness_trend"),
        "dashboard_next": dashboard.get("recommendation", {}).get("workout_type"),
        "schedule_runs": schedule.get("run_count"),
        "schedule_miles": schedule.get("projected_distance_range_miles"),
        "schedule_types": [
            day.get("recommendation", {}).get("workout_type")
            for day in schedule.get("days", [])
            if day.get("recommendation")
        ],
        "schedule_readiness": [
            day.get("recommendation", {}).get("readiness")
            for day in schedule.get("days", [])
            if day.get("recommendation")
        ],
        "summary": schedule.get("summary"),
    }


def _scenario_payloads(
    source_files: dict[str, Path], name: str, now: datetime
) -> list[UploadPayload]:
    if name == "underload":
        return [
            shifted_payload(source_files, "ordinary", now - timedelta(days=50), 0),
            shifted_payload(source_files, "short", now - timedelta(days=4), 1),
        ]
    if name == "overload":
        kinds = ["ordinary", "intervals", "long", "ordinary", "short", "ordinary"]
        return [
            shifted_payload(
                source_files,
                kinds[index % len(kinds)],
                now - timedelta(days=index % 7, hours=index // 7 * 3),
                index,
            )
            for index in range(18)
        ]
    if name == "random":
        rng = Random(145)
        kinds = list(source_files)
        used: set[tuple[int, int]] = set()
        payloads: list[UploadPayload] = []
        while len(payloads) < 80:
            day = rng.randint(0, 364)
            minute = rng.randint(0, 1439)
            if (day, minute) in used:
                continue
            used.add((day, minute))
            payloads.append(
                shifted_payload(
                    source_files,
                    rng.choice(kinds),
                    now - timedelta(days=day, minutes=minute),
                    len(payloads),
                )
            )
        duplicate = payloads[0]
        payloads.extend(
            [
                UploadPayload("duplicate-a.tcx", duplicate.content),
                UploadPayload("duplicate-b.tcx", duplicate.content),
                UploadPayload("malformed.tcx", b"<TrainingCenterDatabase><broken>"),
            ]
        )
        return payloads
    raise ValueError(name)


def run_scenario(source_files: dict[str, Path], name: str, now: datetime) -> dict:
    with TemporaryDirectory(prefix=f"running-analysis-{name}-") as temp:
        root = Path(temp)
        _write_isolated_config(root)
        client = TestClient(create_app(root))
        before = _api_snapshot(client)
        payloads = _scenario_payloads(source_files, name, now)
        started = perf_counter()
        # Weather/model failure is deliberate: upload resilience and coaching
        # must remain usable when optional enrichment is unavailable.
        with (
            patch(
                "run_analysis.web.upload_service.update_weather",
                side_effect=RuntimeError("offline stress weather"),
            ),
            patch(
                "run_analysis.web.upload_service.fit_models",
                side_effect=RuntimeError("offline stress model"),
            ),
        ):
            upload = run_upload_pipeline(root, "config.yaml", payloads)
        elapsed = perf_counter() - started
        after = _api_snapshot(client)
        with connect(root / "data" / "stress.sqlite") as connection:
            counts = {
                "activities": connection.execute("SELECT COUNT(*) FROM activities").fetchone()[0],
                "trackpoints": connection.execute("SELECT COUNT(*) FROM trackpoints").fetchone()[0],
                "metrics": connection.execute("SELECT COUNT(*) FROM activity_metrics").fetchone()[0],
            }
        return {
            "scenario": name,
            "payload_count": len(payloads),
            "elapsed_seconds": round(elapsed, 3),
            "before": before,
            "after": after,
            "database": counts,
            "file_statuses": {
                status: sum(item.status == status for item in upload.files)
                for status in sorted({item.status for item in upload.files})
            },
            "stages": [stage.model_dump() for stage in upload.stages],
        }


def main() -> None:
    source_files = _discover_source_files()
    missing = [str(path) for path in source_files.values() if not path.exists()]
    if missing:
        raise SystemExit(f"Missing real TCX templates: {missing}")
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    results = [
        run_scenario(source_files, name, now)
        for name in ("underload", "overload", "random")
    ]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
