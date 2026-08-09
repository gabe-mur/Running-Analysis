"""SQLite persistence and schema migrations."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import json
import sqlite3

from .privacy import private_directory, private_file


SCHEMA_VERSION = 13


def connect(path: str | Path) -> sqlite3.Connection:
    database_path = Path(path)
    private_directory(database_path.parent)
    connection = sqlite3.connect(database_path)
    private_file(database_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    private_file(f"{database_path}-wal")
    private_file(f"{database_path}-shm")
    connection.execute("PRAGMA synchronous = NORMAL")
    return connection


@contextmanager
def transaction(connection: sqlite3.Connection):
    try:
        connection.execute("BEGIN")
        yield
    except Exception:
        connection.rollback()
        raise
    else:
        connection.commit()


def initialize(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS schema_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS source_files (
            id INTEGER PRIMARY KEY,
            path TEXT NOT NULL UNIQUE,
            display_path TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            mtime_ns INTEGER NOT NULL,
            sha256 TEXT NOT NULL,
            parse_status TEXT NOT NULL CHECK(parse_status IN ('ok', 'warning', 'failed')),
            parse_encoding TEXT,
            parse_warnings_json TEXT NOT NULL DEFAULT '[]',
            parse_error TEXT,
            activity_count INTEGER NOT NULL DEFAULT 0,
            imported_at_utc TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_source_sha256 ON source_files(sha256);

        CREATE TABLE IF NOT EXISTS activities (
            id INTEGER PRIMARY KEY,
            activity_uid TEXT NOT NULL UNIQUE,
            activity_id TEXT,
            sport TEXT NOT NULL,
            start_time_utc TEXT,
            start_time_utc_epoch REAL,
            start_time_local TEXT,
            timezone_name TEXT,
            timezone_source TEXT,
            total_elapsed_time_s REAL,
            lap_recorded_time_s REAL,
            total_distance_m REAL,
            calories INTEGER,
            average_hr_bpm REAL,
            maximum_hr_bpm INTEGER,
            notes TEXT,
            creator TEXT,
            lap_count INTEGER NOT NULL,
            trackpoint_count INTEGER NOT NULL,
            gps_quality TEXT NOT NULL,
            hr_quality TEXT NOT NULL,
            elevation_quality TEXT NOT NULL,
            cadence_quality TEXT NOT NULL,
            distance_source TEXT NOT NULL,
            namespaces_json TEXT NOT NULL,
            data_quality_json TEXT NOT NULL,
            created_at_utc TEXT NOT NULL,
            updated_at_utc TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_activities_start ON activities(start_time_utc_epoch);
        CREATE INDEX IF NOT EXISTS idx_activities_external_id ON activities(activity_id);

        CREATE TABLE IF NOT EXISTS activity_sources (
            activity_id INTEGER NOT NULL REFERENCES activities(id) ON DELETE CASCADE,
            source_file_id INTEGER NOT NULL REFERENCES source_files(id) ON DELETE CASCADE,
            source_activity_index INTEGER NOT NULL,
            is_primary INTEGER NOT NULL DEFAULT 0 CHECK(is_primary IN (0, 1)),
            duplicate_reason TEXT,
            PRIMARY KEY(source_file_id, source_activity_index)
        );

        CREATE TABLE IF NOT EXISTS laps (
            id INTEGER PRIMARY KEY,
            activity_id INTEGER NOT NULL REFERENCES activities(id) ON DELETE CASCADE,
            lap_index INTEGER NOT NULL,
            start_time_utc TEXT,
            total_time_s REAL,
            distance_m REAL,
            calories INTEGER,
            average_hr_bpm INTEGER,
            maximum_hr_bpm INTEGER,
            maximum_speed_mps REAL,
            intensity TEXT,
            trigger_method TEXT,
            UNIQUE(activity_id, lap_index)
        );

        CREATE TABLE IF NOT EXISTS trackpoints (
            id INTEGER PRIMARY KEY,
            activity_id INTEGER NOT NULL REFERENCES activities(id) ON DELETE CASCADE,
            lap_index INTEGER NOT NULL,
            track_index INTEGER NOT NULL,
            point_index INTEGER NOT NULL,
            timestamp_utc TEXT,
            latitude REAL,
            longitude REAL,
            gps_valid INTEGER NOT NULL CHECK(gps_valid IN (0, 1)),
            altitude_m REAL,
            distance_m REAL,
            heart_rate_bpm INTEGER,
            cadence INTEGER,
            run_cadence INTEGER,
            cadence_source TEXT,
            pause_after_s REAL,
            speed_mps REAL,
            parse_flags_json TEXT NOT NULL DEFAULT '[]',
            UNIQUE(activity_id, lap_index, track_index, point_index)
        );

        CREATE INDEX IF NOT EXISTS idx_trackpoints_activity_time
            ON trackpoints(activity_id, timestamp_utc);

        CREATE TABLE IF NOT EXISTS segments (
            id INTEGER PRIMARY KEY,
            activity_id INTEGER NOT NULL REFERENCES activities(id) ON DELETE CASCADE,
            segment_index INTEGER NOT NULL,
            metrics_json TEXT NOT NULL,
            UNIQUE(activity_id, segment_index)
        );

        CREATE TABLE IF NOT EXISTS weather_cache (
            id INTEGER PRIMARY KEY,
            provider TEXT NOT NULL,
            latitude_key REAL NOT NULL,
            longitude_key REAL NOT NULL,
            date_local TEXT NOT NULL,
            response_json TEXT NOT NULL,
            fetched_at_utc TEXT NOT NULL,
            UNIQUE(provider, latitude_key, longitude_key, date_local)
        );

        CREATE TABLE IF NOT EXISTS activity_weather (
            activity_id INTEGER PRIMARY KEY REFERENCES activities(id) ON DELETE CASCADE,
            weather_cache_id INTEGER REFERENCES weather_cache(id),
            derived_weather_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS activity_metrics (
            activity_id INTEGER PRIMARY KEY REFERENCES activities(id) ON DELETE CASCADE,
            metrics_json TEXT NOT NULL,
            calculated_at_utc TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS model_runs (
            id INTEGER PRIMARY KEY,
            activity_id INTEGER NOT NULL REFERENCES activities(id) ON DELETE CASCADE,
            model_name TEXT NOT NULL,
            model_version TEXT NOT NULL,
            result_json TEXT NOT NULL,
            UNIQUE(activity_id, model_name, model_version)
        );

        CREATE TABLE IF NOT EXISTS run_overrides (
            activity_id TEXT PRIMARY KEY,
            include_in_model INTEGER,
            workout_type TEXT,
            illness INTEGER,
            notes TEXT
        );

        CREATE TABLE IF NOT EXISTS model_metadata (
            id INTEGER PRIMARY KEY,
            model_name TEXT NOT NULL,
            model_version TEXT NOT NULL,
            fitted_at_utc TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            UNIQUE(model_name, model_version)
        );
        """
    )
    _reject_newer_database(connection)
    connection.execute(
        "INSERT INTO schema_metadata(key, value) VALUES('schema_version', ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (str(SCHEMA_VERSION),),
    )
    # Renames run before the additive migrations so that an ``ADD COLUMN``
    # further down does not create an empty twin of a column being renamed.
    _rename_legacy_columns(connection)
    _migrate_v2(connection)
    _migrate_v3(connection)
    _migrate_v4(connection)
    _migrate_v5(connection)
    _migrate_v6(connection)
    _migrate_v7(connection)
    _migrate_v8(connection)
    _migrate_v9(connection)
    _migrate_v10(connection)
    _migrate_v11(connection)
    _migrate_v12(connection)
    _migrate_v13(connection)
    connection.commit()


class DatabaseTooNewError(RuntimeError):
    """The database was migrated by a newer version of this application."""


def _reject_newer_database(connection: sqlite3.Connection) -> None:
    """Refuse to run against a database a newer build already migrated.

    Migrations only move forward, so older code cannot understand a newer
    database — but it also cannot tell. A renamed column or model name simply
    matches nothing, and every query returns an empty, confident-looking
    answer. Worse, the version marker below would then be rewritten *downward*,
    erasing the only evidence of the mismatch.

    This is the exact failure a long-running server hits when the code on disk
    is upgraded underneath it: the process keeps serving stale code until it is
    restarted. Fail loudly and say so.
    """

    row = connection.execute(
        "SELECT value FROM schema_metadata WHERE key='schema_version'"
    ).fetchone()
    if row is None:
        return
    try:
        stored = int(row[0])
    except (TypeError, ValueError):
        return
    if stored > SCHEMA_VERSION:
        raise DatabaseTooNewError(
            f"This database is at schema version {stored}, but this build of the "
            f"application understands version {SCHEMA_VERSION}. A newer version "
            "has already migrated it. If a server is running, restart it so it "
            "picks up the current code."
        )


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {str(row["name"]) for row in connection.execute(f"PRAGMA table_info({table})")}


def _add_columns(connection: sqlite3.Connection, table: str, definitions: list[str]) -> None:
    existing = _columns(connection, table)
    for definition in definitions:
        name = definition.split()[0]
        if name not in existing:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {definition}")


#: ``(table, old_name, new_name)`` renames applied before additive migrations.
_COLUMN_RENAMES = (
    (
        "activity_metrics",
        "standardized_pace_145_min_mile",
        "standardized_pace_at_target_hr_min_mile",
    ),
)


def _rename_legacy_columns(connection: sqlite3.Connection) -> None:
    for table, old_name, new_name in _COLUMN_RENAMES:
        columns = _columns(connection, table)
        if old_name in columns and new_name not in columns:
            connection.execute(f"ALTER TABLE {table} RENAME COLUMN {old_name} TO {new_name}")
        elif old_name in columns:
            # Both names present: an additive migration created the new column
            # before the rename existed, so the rename was skipped and the old
            # one was orphaned. Drop it only when it holds nothing, so a real
            # column is never destroyed by a cleanup step.
            populated = connection.execute(
                f"SELECT COUNT({old_name}) FROM {table}"
            ).fetchone()[0]
            if not populated:
                connection.execute(f"ALTER TABLE {table} DROP COLUMN {old_name}")


def _migrate_v2(connection: sqlite3.Connection) -> None:
    _add_columns(
        connection,
        "segments",
        [
            "start_time_utc TEXT",
            "end_time_utc TEXT",
            "distance_m REAL",
            "moving_time_s REAL",
            "elapsed_time_s REAL",
            "stopped_time_s REAL",
            "moving_pace_min_mile REAL",
            "average_hr_bpm REAL",
            "maximum_hr_bpm INTEGER",
            "average_cadence REAL",
            "elevation_gain_m REAL",
            "elevation_loss_m REAL",
            "net_elevation_change_m REAL",
            "average_grade_percent REAL",
            "distance_into_run_m REAL",
            "elapsed_minutes_into_run REAL",
            "moving_minutes_into_run REAL",
            "gps_complete_fraction REAL",
            "route_bearing_degrees REAL",
            "is_pathological INTEGER NOT NULL DEFAULT 0",
            "flags_json TEXT NOT NULL DEFAULT '[]'",
        ],
    )
    _add_columns(
        connection,
        "activity_metrics",
        [
            "processing_fingerprint TEXT",
            "elapsed_time_s REAL",
            "device_timer_time_s REAL",
            "calculated_moving_time_s REAL",
            "stopped_time_s REAL",
            "very_slow_time_s REAL",
            "elapsed_pace_min_mile REAL",
            "device_timer_pace_min_mile REAL",
            "moving_pace_min_mile REAL",
            "moving_average_hr_bpm REAL",
            "moving_maximum_hr_bpm INTEGER",
            "stop_fraction REAL",
            "analysis_distance_m REAL",
            "distance_coverage_fraction REAL",
            "segment_count INTEGER",
            "pathological_segment_count INTEGER",
            "model_eligible INTEGER",
            "exclusion_reason TEXT",
            "hr_zone_seconds_json TEXT NOT NULL DEFAULT '{}'",
            "diagnostics_json TEXT NOT NULL DEFAULT '{}'",
        ],
    )


def _migrate_v3(connection: sqlite3.Connection) -> None:
    weather_columns = [
        "temperature_f REAL",
        "dewpoint_f REAL",
        "relative_humidity_percent REAL",
        "apparent_temperature_f REAL",
        "wind_speed_mph REAL",
        "wind_gust_mph REAL",
        "wind_direction_degrees REAL",
        "precipitation_in REAL",
        "surface_pressure_hpa REAL",
        "headwind_mph REAL",
        "tailwind_mph REAL",
        "crosswind_mph REAL",
        "weather_quality TEXT",
    ]
    _add_columns(connection, "segments", weather_columns)
    _add_columns(connection, "activity_weather", weather_columns)
    _add_columns(
        connection,
        "weather_cache",
        [
            "cache_file TEXT",
            "request_url TEXT",
            "hourly_units_json TEXT NOT NULL DEFAULT '{}'",
        ],
    )


def _migrate_v4(connection: sqlite3.Connection) -> None:
    _add_columns(
        connection,
        "activity_metrics",
        [
            "previous_7d_miles REAL",
            "previous_7d_minutes REAL",
            "previous_28d_miles REAL",
            "previous_28d_minutes REAL",
            "days_since_previous_run REAL",
            "days_since_previous_hard_run REAL",
            "workload_json TEXT NOT NULL DEFAULT '{}'",
        ],
    )


def _migrate_v5(connection: sqlite3.Connection) -> None:
    _add_columns(
        connection,
        "activity_metrics",
        [
            "standardized_pace_at_target_hr_min_mile REAL",
            "standardized_pace_uncertainty_min_mile REAL",
            "raw_aerobic_efficiency_min_mile REAL",
            "environmental_adjustment_min_mile REAL",
            "selected_model_name TEXT",
            "selected_model_version TEXT",
        ],
    )


def _migrate_v6(connection: sqlite3.Connection) -> None:
    _add_columns(
        connection,
        "activity_metrics",
        [
            "session_zone_load REAL",
            "easy_minutes REAL",
            "moderate_minutes REAL",
            "hard_minutes REAL",
            "hr_load_coverage REAL",
        ],
    )
    _add_columns(connection, "run_overrides", ["health_tag TEXT"])
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS app_state (
            key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL,
            updated_at_utc TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS recommendation_history (
            id INTEGER PRIMARY KEY,
            generated_at_utc TEXT NOT NULL,
            fitness_state_json TEXT NOT NULL,
            request_json TEXT NOT NULL,
            result_json TEXT NOT NULL
        );
        """
    )


def _migrate_v7(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS external_fitness_snapshots (
            id INTEGER PRIMARY KEY,
            measured_at TEXT NOT NULL,
            vo2_max REAL,
            predicted_5k_seconds INTEGER,
            predicted_10k_seconds INTEGER,
            predicted_half_marathon_seconds INTEGER,
            predicted_marathon_seconds INTEGER,
            source TEXT NOT NULL DEFAULT 'Garmin',
            created_at_utc TEXT NOT NULL,
            UNIQUE(measured_at, source)
        );
        CREATE INDEX IF NOT EXISTS idx_external_fitness_date
            ON external_fitness_snapshots(measured_at);
        """
    )


def _migrate_v8(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS activity_location_overrides (
            activity_id INTEGER PRIMARY KEY REFERENCES activities(id) ON DELETE CASCADE,
            postal_code TEXT NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            locality TEXT,
            region TEXT,
            country_code TEXT NOT NULL DEFAULT 'US',
            source TEXT NOT NULL DEFAULT 'open_meteo_geocoding',
            updated_at_utc TEXT NOT NULL
        );
        """
    )


def _migrate_v9(connection: sqlite3.Connection) -> None:
    _add_columns(connection, "run_overrides", ["perceived_exertion INTEGER"])


def _migrate_v10(connection: sqlite3.Connection) -> None:
    """Store segment cadence in total steps per minute.

    ``segments.average_cadence`` held whatever the device recorded, which for
    Garmin's RunCadence extension is one leg only. The replacement column has
    a unit in its name so a half-cadence value can never be silently compared
    against a steps-per-minute threshold. Existing rows are left for the
    processor to rewrite; the processor version bump forces that reprocessing.
    """

    _add_columns(connection, "segments", ["average_cadence_spm REAL"])


def _migrate_v11(connection: sqlite3.Connection) -> None:
    """Drop the hard-coded comparison heart rate from stored model names.

    The comparison heart rate is configurable, so a column, model name, or
    result key that says "145" becomes wrong the moment the athlete changes
    it. Stored values are not recomputed here: they were produced at whatever
    ``target_hr`` was configured at the time, and each model run now records
    that heart rate alongside its result so the two can never drift apart
    unnoticed. The matching column rename happens in
    :func:`_rename_legacy_columns`.
    """

    for table in ("model_runs", "model_metadata"):
        connection.execute(
            f"UPDATE {table} SET model_name='standardized_pace_at_target_hr' "
            "WHERE model_name='standardized_pace_145'"
        )
    # Stored result payloads carry the same keys and must be rewritten too,
    # including inside the nested steady-aerobic benchmark. Left alone, every
    # reader would silently see a missing estimate.
    for old_key, new_key in (
        ("standardized_pace_145_min_mile", "standardized_pace_at_target_hr_min_mile"),
        ("raw_pace_145_min_mile", "raw_pace_at_target_hr_min_mile"),
    ):
        connection.execute(
            "UPDATE model_runs SET result_json=REPLACE(result_json, ?, ?) "
            "WHERE result_json LIKE ?",
            (f'"{old_key}"', f'"{new_key}"', f'%"{old_key}"%'),
        )


def _migrate_v12(connection: sqlite3.Connection) -> None:
    """Drop the free-text note from the stored current-status payload.

    The note was written to a single ``app_state`` slot that every later
    check-in overwrote, so it was never a record of anything, and nothing read
    it. Removing the field from the request model makes strict validation
    reject any payload that still carries it, so the stored JSON is rewritten
    rather than left to fail on the next read. Per-run notes are a different
    field on ``run_overrides`` and are untouched.
    """

    row = connection.execute(
        "SELECT value_json FROM app_state WHERE key='current_health'"
    ).fetchone()
    if not row:
        return
    try:
        payload = json.loads(row[0])
    except (TypeError, ValueError):
        return
    if not isinstance(payload, dict) or "notes" not in payload:
        return
    payload.pop("notes")
    connection.execute(
        "UPDATE app_state SET value_json=? WHERE key='current_health'",
        (json.dumps(payload),),
    )


def _migrate_v13(connection: sqlite3.Connection) -> None:
    """Record device-stated pause duration alongside each trackpoint.

    FIT files carry explicit timer start/stop events, which is better evidence
    than inferring a stop from speed and distance. TCX has nothing equivalent,
    so this stays NULL for those runs and the inference path is unchanged.
    """

    _add_columns(connection, "trackpoints", ["pause_after_s REAL"])
