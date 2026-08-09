"""Upload orchestration for incremental local analysis.

The service owns pipeline decisions; the route only validates multipart input
and serializes the result.  Each stage is reported independently because a run
can be imported and analyzed locally even if historical weather is temporarily
unavailable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
import hashlib
import json
import re

from ..config import load_config, resolve_project_path
from ..db import connect, initialize
from ..importer import SUPPORTED_SUFFIXES, import_files
from ..modeling import InsufficientModelDataError, fit_models
from ..overrides import sync_overrides
from ..processing import process_activities
from ..privacy import private_directory, private_file
from ..recommendation_service import generate_weekly_schedule, load_current_status
from ..weather import update_weather
from .schemas import UploadResponse, UploadedFileResult, UploadStage, WeeklyScheduleRequest


MAX_UPLOAD_BYTES = 50 * 1024 * 1024
_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True, slots=True)
class UploadPayload:
    filename: str
    content: bytes


def _safe_filename(filename: str) -> str:
    name = Path(filename).name
    cleaned = _SAFE_FILENAME.sub("-", name).strip("-.")
    return cleaned or "activity.dat"


def validate_upload(payload: UploadPayload) -> None:
    if not payload.filename.casefold().endswith(SUPPORTED_SUFFIXES):
        raise ValueError(
            f"Only {', '.join(SUPPORTED_SUFFIXES)} files are supported: {payload.filename}"
        )
    if not payload.content:
        raise ValueError(f"The uploaded file is empty: {payload.filename}")
    if len(payload.content) > MAX_UPLOAD_BYTES:
        raise ValueError(f"The uploaded file exceeds 50 MB: {payload.filename}")


def _persist_upload(root: Path, payload: UploadPayload) -> Path:
    digest = hashlib.sha256(payload.content).hexdigest()
    upload_dir = private_directory(root / "uploads")
    destination = upload_dir / f"{digest[:16]}-{_safe_filename(payload.filename)}"
    if not destination.exists():
        destination.write_bytes(payload.content)
    private_file(destination)
    return destination


def _activity_ids_for_source(connection, path: Path) -> list[int]:
    return [
        int(row[0])
        for row in connection.execute(
            """
            SELECT activity_sources.activity_id
            FROM activity_sources
            JOIN source_files ON source_files.id = activity_sources.source_file_id
            WHERE source_files.path = ?
            ORDER BY activity_sources.source_activity_index
            """,
            (str(path.resolve()),),
        )
    ]


def _detail(summary: object) -> str:
    return json.dumps(asdict(summary), sort_keys=True)


def run_upload_pipeline(
    project_root: str | Path,
    config_path: str | Path,
    payloads: Iterable[UploadPayload],
) -> UploadResponse:
    """Persist uploads and run import → process → weather → model once."""

    root = Path(project_root).resolve()
    selected_config = Path(config_path)
    if not selected_config.is_absolute():
        selected_config = root / selected_config
    config = load_config(selected_config)
    database = resolve_project_path(root, config["paths"]["database"])
    files = list(payloads)
    if not files:
        raise ValueError("At least one TCX file is required")
    for payload in files:
        validate_upload(payload)

    saved = [(payload, _persist_upload(root, payload)) for payload in files]
    stages = [
        UploadStage(
            name="save",
            status="complete",
            detail=f"Saved {len(saved)} content-addressed upload(s) locally.",
        )
    ]
    results: list[UploadedFileResult] = []
    imported_any = False

    with connect(database) as connection:
        initialize(connection)
        for payload, path in saved:
            try:
                summary = import_files(
                    connection,
                    root,
                    default_timezone=config["timezone_default"],
                    paths=[path],
                )
                activity_ids = _activity_ids_for_source(connection, path)
                if summary.failed_files:
                    source = connection.execute(
                        "SELECT parse_error FROM source_files WHERE path = ?", (str(path.resolve()),)
                    ).fetchone()
                    results.append(
                        UploadedFileResult(
                            filename=payload.filename,
                            status="failed",
                            error=str(source[0]) if source and source[0] else "TCX parsing failed",
                        )
                    )
                else:
                    status = "unchanged" if summary.unchanged_files else (
                        "duplicate" if summary.duplicate_activities and not summary.activities_added else "imported"
                    )
                    results.append(
                        UploadedFileResult(
                            filename=payload.filename,
                            status=status,
                            activity_ids=activity_ids,
                            warnings=["Parser reported warnings"] if summary.warning_files else [],
                        )
                    )
                    imported_any = imported_any or bool(activity_ids)
            except Exception as exc:  # isolate malformed files in a multi-file upload
                results.append(
                    UploadedFileResult(filename=payload.filename, status="failed", error=str(exc))
                )
        stages.append(
            UploadStage(
                name="import",
                status="complete" if imported_any else "failed",
                detail=f"{sum(item.status != 'failed' for item in results)} of {len(results)} file(s) accepted.",
            )
        )

        if imported_any:
            overrides_path = resolve_project_path(root, config["paths"]["overrides"])
            try:
                sync_overrides(connection, overrides_path)
                process_summary = process_activities(connection, config)
                stages.append(UploadStage(name="process", status="complete", detail=_detail(process_summary)))
            except Exception as exc:
                stages.append(UploadStage(name="process", status="failed", detail=str(exc)))

            try:
                weather_summary = update_weather(connection, config, root)
                weather_status = "partial" if weather_summary.failures else "complete"
                stages.append(UploadStage(name="weather", status=weather_status, detail=_detail(weather_summary)))
            except Exception as exc:
                stages.append(
                    UploadStage(
                        name="weather",
                        status="failed",
                        detail=f"Run remains available without weather: {exc}",
                    )
                )

            try:
                output = root / "output" / "model_results.json"
                output.parent.mkdir(parents=True, exist_ok=True)
                model_summary = fit_models(connection, config, output)
                stages.append(UploadStage(name="model", status="complete", detail=_detail(model_summary)))
            except InsufficientModelDataError as exc:
                stages.append(
                    UploadStage(
                        name="model",
                        status="deferred",
                        detail=f"More usable history is needed before fitness modeling: {exc}",
                    )
                )
            except Exception as exc:
                stages.append(
                    UploadStage(
                        name="model",
                        status="failed",
                        detail=f"Run remains available without a fitness score: {exc}",
                    )
                )
            try:
                current = load_current_status(connection)
                schedule = generate_weekly_schedule(
                    connection,
                    config,
                    WeeklyScheduleRequest(health_status=current.health_status),
                    root,
                )
                stages.append(
                    UploadStage(
                        name="schedule",
                        status="complete",
                        detail=(
                            f"Refreshed today-through-{schedule.end_date.isoformat()} plan "
                            f"from the updated activity history."
                        ),
                    )
                )
            except Exception as exc:
                stages.append(
                    UploadStage(
                        name="schedule",
                        status="failed",
                        detail=f"Run was analyzed but the saved weekly plan could not refresh: {exc}",
                    )
                )
        else:
            for name in ("process", "weather", "model", "schedule"):
                stages.append(UploadStage(name=name, status="skipped", detail="No activity was imported."))

    successful_activity_ids = [activity_id for item in results for activity_id in item.activity_ids]
    return UploadResponse(
        files=results,
        stages=stages,
        primary_activity_id=successful_activity_ids[0] if len(successful_activity_ids) == 1 else None,
    )
