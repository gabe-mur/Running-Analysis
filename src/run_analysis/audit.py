"""Phase 4 dataset sanity audit."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import sqlite3


def build_audit(connection: sqlite3.Connection) -> dict:
    scalar = lambda sql, params=(): connection.execute(sql, params).fetchone()[0]
    date_row = connection.execute("SELECT MIN(start_time_utc), MAX(start_time_utc) FROM activities").fetchone()
    quality = {
        kind: {
            row[0]: row[1]
            for row in connection.execute(f"SELECT {kind}, COUNT(*) FROM activities GROUP BY {kind}")
        }
        for kind in ("gps_quality", "hr_quality", "elevation_quality", "cadence_quality")
    }
    weather_count = scalar("SELECT COUNT(*) FROM activity_weather")
    warnings: list[str] = []
    if weather_count == 0:
        warnings.append(
            "Historical weather has not been fetched; temperature/dew-point coverage and heat-run counts are unavailable."
        )
    hr_row = connection.execute(
        "SELECT MIN(heart_rate_bpm), MAX(heart_rate_bpm) FROM trackpoints WHERE heart_rate_bpm IS NOT NULL"
    ).fetchone()
    exclusions = {
        (row[0] or "eligible"): row[1]
        for row in connection.execute(
            "SELECT exclusion_reason, COUNT(*) FROM activity_metrics GROUP BY exclusion_reason ORDER BY COUNT(*) DESC"
        )
    }
    weather_ranges = connection.execute(
        "SELECT MIN(temperature_f),MAX(temperature_f),MIN(dewpoint_f),MAX(dewpoint_f) FROM activity_weather"
    ).fetchone()
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_tcx_files": scalar("SELECT COUNT(*) FROM source_files"),
        "unique_running_activities": scalar("SELECT COUNT(*) FROM activities WHERE sport = 'Running'"),
        "date_range_utc": {"start": date_row[0], "end": date_row[1]},
        "quality_counts": quality,
        "model_eligible_activities_pre_weather": scalar(
            "SELECT COUNT(*) FROM activity_metrics WHERE model_eligible = 1"
        ),
        "total_segments": scalar("SELECT COUNT(*) FROM segments"),
        "pathological_segments": scalar("SELECT COUNT(*) FROM segments WHERE is_pathological = 1"),
        "heart_rate_range_bpm": {"minimum": hr_row[0], "maximum": hr_row[1]},
        "weather": {
            "activities_with_weather": weather_count,
            "temperature_range_f": [weather_ranges[0], weather_ranges[1]] if weather_count else None,
            "dewpoint_range_f": [weather_ranges[2], weather_ranges[3]] if weather_count else None,
            "runs_ge_75_f": scalar("SELECT COUNT(*) FROM activity_weather WHERE temperature_f >= 75") if weather_count else None,
            "runs_ge_80_f": scalar("SELECT COUNT(*) FROM activity_weather WHERE temperature_f >= 80") if weather_count else None,
            "runs_ge_85_f": scalar("SELECT COUNT(*) FROM activity_weather WHERE temperature_f >= 85") if weather_count else None,
            "runs_le_60_f": scalar("SELECT COUNT(*) FROM activity_weather WHERE temperature_f <= 60") if weather_count else None,
        },
        "eligibility_breakdown": exclusions,
        "warnings": warnings,
    }


def write_audit(audit: dict, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return output
