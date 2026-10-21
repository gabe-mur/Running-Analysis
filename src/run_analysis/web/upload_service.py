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
from contextlib import contextmanager
import hashlib
import json
import re
import threading

from ..config import load_config, resolve_project_path
from ..db import connect, initialize
from ..importer import SUPPORTED_SUFFIXES, import_files, parser_for
from ..modeling import InsufficientModelDataError, fit_models
from ..models import SENSOR_EXTRACTION_VERSION
from ..overrides import sync_overrides
from ..processing import process_activities
from ..prescription_matching import match_activities_to_prescriptions
from ..privacy import private_directory, private_file
from ..recommendation_service import generate_weekly_schedule, load_current_status
from ..weather import update_weather
from .schemas import UploadResponse, UploadedFileResult, UploadStage, WeeklyScheduleRequest
from .upload_archive import fit_members, MAX_EXPANDED_BYTES


MAX_UPLOAD_BYTES = 50 * 1024 * 1024
UPLOAD_SUFFIXES = (*SUPPORTED_SUFFIXES, '.zip')
_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")
_UPLOAD_LOCKS: dict[Path, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


class UploadBusyError(Exception):
    pass


@contextmanager
def upload_operation(root: Path):
    """Serialize manual uploads and sync through download and analysis."""
    with _LOCKS_GUARD:
        lock = _UPLOAD_LOCKS.setdefault(root.resolve(), threading.Lock())
    if not lock.acquire(blocking=False):
        raise UploadBusyError("A run upload or sync is already in progress. Please wait.")
    try:
        yield
    finally:
        lock.release()


def run_locked_upload_pipeline(root, config_path, payloads):
    with upload_operation(Path(root)):
        return run_upload_pipeline(root, config_path, payloads, refresh_plan=False)


@dataclass(frozen=True, slots=True)
class UploadPayload:
    filename: str
    content: bytes


def _safe_filename(filename: str) -> str:
    name = Path(filename).name
    cleaned = _SAFE_FILENAME.sub("-", name).strip("-.")
    return cleaned or "activity.dat"


def validate_upload(payload: UploadPayload) -> None:
    if not payload.filename.casefold().endswith(UPLOAD_SUFFIXES):
        raise ValueError(
            f"Only {', '.join(UPLOAD_SUFFIXES)} files are supported: {payload.filename}"
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


def _known_complete_upload(connection, payload: UploadPayload) -> Path | None:
    """Reuse successful content, regardless of filename, but allow FIT backfill.

    The ordinary importer retains separate source provenance. Uploads can avoid
    creating another source for exactly the same bytes once extraction is done.
    Failed imports and incomplete source/activity linkage are never cache hits.
    """
    digest = hashlib.sha256(payload.content).hexdigest()
    fit = payload.filename.casefold().endswith((".fit", ".fit.gz"))
    return complete_source_for_digest(connection, digest, fit=fit,
                                      parser=parser_for(Path(payload.filename)))


def complete_source_for_digest(connection, digest: str, *, fit: bool, parser=None) -> Path | None:
    """Find ingested content with complete linkage and (for FIT) sensors."""
    for source in connection.execute(
        "SELECT id,path,activity_count FROM source_files WHERE sha256=? "
        "AND parse_status IN ('ok','warning') ORDER BY id DESC", (digest,),
    ):
        # The same bytes uploaded with an incompatible extension must still parse.
        known_fit = source['path'].casefold().endswith((".fit", ".fit.gz"))
        if known_fit != fit or (parser is not None and parser_for(Path(source['path'])) is not parser):
            continue
        links = connection.execute("""SELECT s.source_activity_index,r.source_file_id,r.extraction_version
            FROM activity_sources s LEFT JOIN activity_sensor_streams r
            ON r.source_file_id=s.source_file_id
            AND r.source_activity_index=s.source_activity_index
            WHERE s.source_file_id=?""", (source['id'],)).fetchall()
        if (links and len(links) == source['activity_count']
                and (not fit or all(row['source_file_id'] is not None and row['extraction_version'] >= SENSOR_EXTRACTION_VERSION for row in links))):
            return Path(source['path'])
    return None


def run_upload_pipeline(
    project_root: str | Path,
    config_path: str | Path,
    payloads: Iterable[UploadPayload],
    *,
    refresh_plan: bool = True,
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
        raise ValueError("At least one activity file is required")
    if len(files) > 25 or sum(len(p.content) for p in files) > MAX_EXPANDED_BYTES:
        raise ValueError('Upload batch exceeds 25 files or 100 MB')
    expanded = []
    expanded_size = 0
    for payload in files:
        validate_upload(payload)
        members = (fit_members(payload.content) if payload.filename.casefold().endswith('.zip')
                   else [(payload.filename, payload.content)])
        for name, content in members:
            expanded_size += len(content)
            if expanded_size > MAX_EXPANDED_BYTES or len(expanded) >= 100:
                raise ValueError('Expanded upload batch exceeds 100 files or 100 MB')
            child = UploadPayload(name, content)
            validate_upload(child)
            label = f'{payload.filename} / {name}' if payload.filename.casefold().endswith('.zip') else name
            expanded.append((label, child))

    stages = [
        UploadStage(
            name="save",
            status="complete",
            detail=f"Accepted {len(expanded)} upload(s); existing identical content is reused locally.",
        )
    ]
    results: list[UploadedFileResult] = []
    changed_activity_ids: set[int] = set()

    with connect(database) as connection:
        initialize(connection)
        for label, payload in expanded:
            try:
                known = _known_complete_upload(connection, payload)
                if known is not None:
                    results.append(UploadedFileResult(
                        filename=label, status="unchanged",
                        activity_ids=_activity_ids_for_source(connection, known),
                    ))
                    continue
                path = _persist_upload(root, payload)
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
                            filename=label,
                            status="failed",
                            error=str(source[0]) if source and source[0] else "Activity parsing failed",
                        )
                    )
                else:
                    status = "unchanged" if summary.unchanged_files else (
                        "duplicate" if summary.duplicate_activities and not summary.activities_added else "imported"
                    )
                    warnings = ["Parser reported warnings"] if summary.warning_files else []
                    if path.suffix.casefold() == '.gpx':
                        source_warnings = connection.execute(
                            'SELECT parse_warnings_json FROM source_files WHERE path=?',
                            (str(path.resolve()),),
                        ).fetchone()
                        codes = json.loads(source_warnings[0]) if source_warnings else []
                        warnings = []
                        if 'gpx_heart_rate_missing' in codes:
                            warnings.append('GPX has no heart-rate samples. The activity was kept, but cannot measure aerobic fitness.')
                        if any(code in codes for code in ('gpx_sport_unspecified', 'gpx_sport_unrecognized')):
                            warnings.append('GPX does not identify this activity as a run. Check its activity type before using it for training.')
                    results.append(
                        UploadedFileResult(
                            filename=label,
                            status=status,
                            activity_ids=activity_ids,
                            warnings=warnings,
                        )
                    )
                    changed_activity_ids.update(summary.evidence_changed_activity_ids)
            except Exception as exc:  # isolate malformed files in a multi-file upload
                results.append(
                    UploadedFileResult(filename=label, status="failed", error=str(exc))
                )
        stages.append(
            UploadStage(
                name="import",
                status="complete" if any(item.status != 'failed' for item in results) else "failed",
                detail=f"{sum(item.status != 'failed' for item in results)} of {len(results)} file(s) accepted.",
            )
        )

        # Recheck watch settings before classification/modeling. A duplicate
        # FIT extraction upgrade can contain new metadata without new running.
        zones_changed = False
        try:
            from ..settings_service import refresh_watch_zones
            config, zones_changed = refresh_watch_zones(selected_config, connection)
            if zones_changed:
                stages.append(UploadStage(name='watch_zones', status='complete',
                    detail='Updated maximum HR, resting HR and zones from the newest recorded watch settings.'))
        except ValueError as exc:
            stages.append(UploadStage(name='watch_zones', status='failed', detail=str(exc)))

        if changed_activity_ids or zones_changed:
            from ..plan_refresh import mark_plan_refresh
            mark_plan_refresh(connection)
            overrides_path = resolve_project_path(root, config["paths"]["overrides"])
            processing_complete = False
            try:
                sync_overrides(connection, overrides_path)
                process_summary = process_activities(connection, config, force=True) if zones_changed else process_activities(connection, config)
                matched_activity_ids = match_activities_to_prescriptions(
                    connection,
                    sorted(changed_activity_ids),
                )
                if matched_activity_ids:
                    # Matching changes the effective workout type. A second
                    # incremental pass recomputes only activities whose
                    # fingerprint changed, before modeling and replanning use
                    # the classification.
                    process_summary = process_activities(connection, config)
                stages.append(UploadStage(name="process", status="complete", detail=_detail(process_summary)))
                processing_complete = True
                if matched_activity_ids:
                    stages.append(
                        UploadStage(
                            name="prescription_match",
                            status="complete",
                            detail=(
                                f"Matched {len(matched_activity_ids)} uploaded run(s) "
                                "to the prescription saved before the workout."
                            ),
                        )
                    )
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

            if processing_complete:
                # The comparison HR is an analysis coordinate, not planner
                # normalization. Fit all history at the updated coordinate;
                # never stamp a new label over old standardized pace results.
                try:
                    from ..settings_service import refresh_automatic_comparison
                    config, coordinate_changed = refresh_automatic_comparison(selected_config, connection)
                    if coordinate_changed:
                        stages.append(UploadStage(name='comparison_hr', status='complete',
                            detail='Updated the automatic pace-comparison heart rate from recent easy-running evidence.'))
                except Exception as exc:
                    stages.append(UploadStage(name='comparison_hr', status='failed', detail=str(exc)))
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
                if not refresh_plan:
                    stages.append(UploadStage(name='schedule', status='deferred',
                        detail='Run data is ready. The plan updates when you open or refresh Weekly Plan.'))
                else:
                    _refresh_uploaded_plan(connection, config, root, stages)
            else:
                detail = "Skipped because activity processing did not complete."
                stages.extend(
                    [
                        UploadStage(name="model", status="skipped", detail=detail),
                        UploadStage(name="schedule", status="skipped", detail=detail),
                    ]
                )
            if not refresh_plan:
                # A plan opened concurrently may have used partially processed
                # input. The completed analysis is a new revision, even if that
                # earlier search happened to finish while this upload ran.
                mark_plan_refresh(connection)
        else:
            for name in ("process", "weather", "model", "schedule"):
                stages.append(UploadStage(name=name, status="skipped", detail="No new activity evidence was imported."))

    successful_activity_ids = [activity_id for item in results for activity_id in item.activity_ids]
    return UploadResponse(
        files=results,
        stages=stages,
        primary_activity_id=successful_activity_ids[0] if len(successful_activity_ids) == 1 else None,
    )


def _refresh_uploaded_plan(connection, config, root, stages):
    try:
        current = load_current_status(connection)
        schedule = generate_weekly_schedule(
            connection, config, WeeklyScheduleRequest(health_status=current.health_status), root)
        stages.append(UploadStage(name='schedule', status='complete',
            detail=f'Refreshed today-through-{schedule.end_date.isoformat()} plan from the updated activity history.'))
    except Exception as exc:
        stages.append(UploadStage(name='schedule', status='failed',
            detail=f'Run was analyzed but the saved weekly plan could not refresh: {exc}'))
