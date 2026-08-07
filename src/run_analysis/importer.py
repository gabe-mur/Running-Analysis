"""Incremental TCX-to-SQLite import with activity-level deduplication."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
import hashlib
import json
import sqlite3

from .db import initialize, transaction
from .models import Activity
from .tcx import parse_tcx


@dataclass(slots=True)
class ImportSummary:
    discovered_files: int = 0
    unchanged_files: int = 0
    imported_files: int = 0
    warning_files: int = 0
    failed_files: int = 0
    activities_added: int = 0
    duplicate_activities: int = 0
    trackpoints_added: int = 0


def discover_tcx_files(root: str | Path) -> list[Path]:
    return sorted(path for path in Path(root).rglob("*") if path.is_file() and path.suffix.lower() == ".tcx")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _activity_uid(activity: Activity) -> str:
    start = _iso(activity.start_time_utc) or "unknown-start"
    signature = "|".join(
        (
            activity.sport.casefold(),
            start,
            f"{activity.total_distance_m:.1f}" if activity.total_distance_m is not None else "unknown-distance",
            f"{activity.lap_recorded_time_s:.1f}" if activity.lap_recorded_time_s is not None else "unknown-time",
        )
    )
    return hashlib.sha256(signature.encode("utf-8")).hexdigest()


def _within(left: float | None, right: float | None, absolute: float, relative: float) -> bool:
    if left is None or right is None:
        return True
    return abs(left - right) <= max(absolute, relative * max(abs(left), abs(right)))


def _find_duplicate(connection: sqlite3.Connection, activity: Activity) -> tuple[int, str] | None:
    if activity.activity_id:
        row = connection.execute(
            "SELECT id FROM activities WHERE activity_id = ? AND sport = ? ORDER BY id LIMIT 1",
            (activity.activity_id, activity.sport),
        ).fetchone()
        if row:
            return int(row["id"]), "same_activity_id"
    if activity.start_time_utc is None:
        return None
    epoch = activity.start_time_utc.timestamp()
    rows = connection.execute(
        """
        SELECT id, total_distance_m, lap_recorded_time_s
        FROM activities
        WHERE sport = ? AND start_time_utc_epoch BETWEEN ? AND ?
        """,
        (activity.sport, epoch - 2.0, epoch + 2.0),
    ).fetchall()
    for row in rows:
        if _within(activity.total_distance_m, row["total_distance_m"], 10.0, 0.005) and _within(
            activity.lap_recorded_time_s, row["lap_recorded_time_s"], 10.0, 0.01
        ):
            return int(row["id"]), "matching_start_distance_time"
    return None


def _delete_previous_source(connection: sqlite3.Connection, source_file_id: int) -> None:
    old_activity_ids = [
        int(row["activity_id"])
        for row in connection.execute(
            "SELECT activity_id FROM activity_sources WHERE source_file_id = ?", (source_file_id,)
        )
    ]
    connection.execute("DELETE FROM source_files WHERE id = ?", (source_file_id,))
    for activity_id in old_activity_ids:
        still_referenced = connection.execute(
            "SELECT 1 FROM activity_sources WHERE activity_id = ? LIMIT 1", (activity_id,)
        ).fetchone()
        if not still_referenced:
            connection.execute("DELETE FROM activities WHERE id = ?", (activity_id,))


def _insert_source(
    connection: sqlite3.Connection,
    path: Path,
    display_path: str,
    digest: str,
    status: str,
    encoding: str | None,
    warnings: list[str],
    error: str | None,
    activity_count: int,
) -> int:
    stat = path.stat()
    cursor = connection.execute(
        """
        INSERT INTO source_files(
            path, display_path, size_bytes, mtime_ns, sha256, parse_status,
            parse_encoding, parse_warnings_json, parse_error, activity_count, imported_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(path.resolve()),
            display_path,
            stat.st_size,
            stat.st_mtime_ns,
            digest,
            status,
            encoding,
            json.dumps(warnings),
            error,
            activity_count,
            _utc_now(),
        ),
    )
    return int(cursor.lastrowid)


def _insert_activity(connection: sqlite3.Connection, activity: Activity) -> int:
    now = _utc_now()
    data_quality = {
        "gps": activity.gps_quality,
        "heart_rate": activity.hr_quality,
        "elevation": activity.elevation_quality,
        "cadence": activity.cadence_quality,
        "distance_source": activity.distance_source,
        "parse_warnings": activity.parse_warnings,
    }
    uid = _activity_uid(activity)
    if connection.execute("SELECT 1 FROM activities WHERE activity_uid = ?", (uid,)).fetchone():
        uid = hashlib.sha256(f"{uid}|{now}".encode()).hexdigest()
    cursor = connection.execute(
        """
        INSERT INTO activities(
            activity_uid, activity_id, sport, start_time_utc, start_time_utc_epoch,
            start_time_local, timezone_name, timezone_source, total_elapsed_time_s,
            lap_recorded_time_s, total_distance_m, calories, average_hr_bpm,
            maximum_hr_bpm, notes, creator, lap_count, trackpoint_count,
            gps_quality, hr_quality, elevation_quality, cadence_quality,
            distance_source, namespaces_json, data_quality_json, created_at_utc, updated_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            uid,
            activity.activity_id,
            activity.sport,
            _iso(activity.start_time_utc),
            activity.start_time_utc.timestamp() if activity.start_time_utc else None,
            _iso(activity.start_time_local),
            activity.timezone_name,
            activity.timezone_source,
            activity.total_elapsed_time_s,
            activity.lap_recorded_time_s,
            activity.total_distance_m,
            activity.calories,
            activity.average_hr_bpm,
            activity.maximum_hr_bpm,
            activity.notes,
            activity.creator,
            len(activity.laps),
            len(activity.trackpoints),
            activity.gps_quality,
            activity.hr_quality,
            activity.elevation_quality,
            activity.cadence_quality,
            activity.distance_source,
            json.dumps(activity.namespaces),
            json.dumps(data_quality),
            now,
            now,
        ),
    )
    activity_row_id = int(cursor.lastrowid)
    connection.executemany(
        """
        INSERT INTO laps(
            activity_id, lap_index, start_time_utc, total_time_s, distance_m,
            calories, average_hr_bpm, maximum_hr_bpm, maximum_speed_mps,
            intensity, trigger_method
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                activity_row_id,
                lap.lap_index,
                _iso(lap.start_time_utc),
                lap.total_time_s,
                lap.distance_m,
                lap.calories,
                lap.average_hr_bpm,
                lap.maximum_hr_bpm,
                lap.maximum_speed_mps,
                lap.intensity,
                lap.trigger_method,
            )
            for lap in activity.laps
        ],
    )
    connection.executemany(
        """
        INSERT INTO trackpoints(
            activity_id, lap_index, track_index, point_index, timestamp_utc,
            latitude, longitude, gps_valid, altitude_m, distance_m,
            heart_rate_bpm, cadence, run_cadence, cadence_source, speed_mps,
            parse_flags_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                activity_row_id,
                point.lap_index,
                point.track_index,
                point.point_index,
                _iso(point.timestamp_utc),
                point.latitude,
                point.longitude,
                int(point.gps_valid),
                point.altitude_m,
                point.distance_m,
                point.heart_rate_bpm,
                point.cadence,
                point.run_cadence,
                point.cadence_source,
                point.speed_mps,
                json.dumps(point.parse_flags),
            )
            for point in activity.trackpoints
        ],
    )
    return activity_row_id


def import_files(
    connection: sqlite3.Connection,
    project_root: str | Path,
    default_timezone: str,
    paths: Iterable[Path] | None = None,
    force: bool = False,
) -> ImportSummary:
    initialize(connection)
    root = Path(project_root).resolve()
    files = list(paths) if paths is not None else discover_tcx_files(root)
    summary = ImportSummary(discovered_files=len(files))
    for source_path in files:
        path = source_path.resolve()
        digest = sha256_file(path)
        existing = connection.execute(
            "SELECT id, sha256 FROM source_files WHERE path = ?", (str(path),)
        ).fetchone()
        if existing and existing["sha256"] == digest and not force:
            summary.unchanged_files += 1
            continue
        try:
            parsed = parse_tcx(path, default_timezone=default_timezone)
        except Exception as error:
            with transaction(connection):
                if existing:
                    _delete_previous_source(connection, int(existing["id"]))
                _insert_source(
                    connection,
                    path,
                    str(path.relative_to(root)) if path.is_relative_to(root) else path.name,
                    digest,
                    "failed",
                    None,
                    [],
                    str(error),
                    0,
                )
            summary.failed_files += 1
            continue

        status = "warning" if parsed.warnings else "ok"
        with transaction(connection):
            if existing:
                _delete_previous_source(connection, int(existing["id"]))
            source_id = _insert_source(
                connection,
                path,
                str(path.relative_to(root)) if path.is_relative_to(root) else path.name,
                digest,
                status,
                parsed.encoding,
                parsed.warnings,
                None,
                len(parsed.activities),
            )
            for activity_index, activity in enumerate(parsed.activities):
                duplicate = _find_duplicate(connection, activity)
                if duplicate:
                    activity_row_id, reason = duplicate
                    summary.duplicate_activities += 1
                    is_primary = 0
                else:
                    activity_row_id = _insert_activity(connection, activity)
                    reason = None
                    is_primary = 1
                    summary.activities_added += 1
                    summary.trackpoints_added += len(activity.trackpoints)
                connection.execute(
                    """
                    INSERT INTO activity_sources(
                        activity_id, source_file_id, source_activity_index, is_primary, duplicate_reason
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (activity_row_id, source_id, activity_index, is_primary, reason),
                )
        summary.imported_files += 1
        if parsed.warnings:
            summary.warning_files += 1
    return summary

