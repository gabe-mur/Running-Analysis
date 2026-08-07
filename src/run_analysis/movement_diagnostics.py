"""Auditable moving-time diagnostics for representative activities."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sqlite3
from typing import Any

from .movement import MovementInterval, classify_movement
from .processing import _load_points


def _event(intervals: list[MovementInterval]) -> dict[str, Any]:
    first, last = intervals[0], intervals[-1]
    elapsed = sum(item.elapsed_s for item in intervals)
    distance = sum(item.distance_m for item in intervals)
    removed = sum(item.stopped_time_s for item in intervals)
    return {
        "start_time_utc": first.start.timestamp_utc.isoformat() if first.start.timestamp_utc else None,
        "end_time_utc": last.end.timestamp_utc.isoformat() if last.end.timestamp_utc else None,
        "classification": first.classification,
        "elapsed_seconds": elapsed,
        "removed_seconds": removed,
        "distance_meters": distance,
        "average_computed_speed_mps": distance / elapsed if elapsed else None,
        "interval_count": len(intervals),
        "flags": sorted({flag for item in intervals for flag in item.flags}),
    }


def _group_events(intervals: list[MovementInterval]) -> list[dict[str, Any]]:
    selected = [
        interval
        for interval in intervals
        if interval.classification in {"stopped", "mixed_gap"}
        and interval.stopped_time_s > 0
    ]
    groups: list[list[MovementInterval]] = []
    for interval in selected:
        if (
            groups
            and interval.index == groups[-1][-1].index + 1
            and interval.classification == groups[-1][-1].classification
        ):
            groups[-1].append(interval)
        else:
            groups.append([interval])
    return [_event(group) for group in groups]


def _candidate_rows(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    return connection.execute(
        """
        SELECT a.id,a.activity_id,a.start_time_utc,a.total_distance_m,a.gps_quality,
               m.model_eligible,m.calculated_moving_time_s,m.device_timer_time_s,
               COALESCE(o.workout_type,'unknown') AS workout_type,
               COALESCE(o.health_tag,'normal') AS health_tag
        FROM activities a JOIN activity_metrics m ON m.activity_id=a.id
        LEFT JOIN run_overrides o ON o.activity_id=a.activity_id
        WHERE a.start_time_utc >= '2026-04-01' AND a.gps_quality != 'missing'
          AND a.total_distance_m >= 2414.016 AND m.model_eligible=1
          AND COALESCE(o.workout_type,'unknown') NOT IN ('hike','bike')
        ORDER BY a.start_time_utc
        """
    ).fetchall()


def build_movement_diagnostic(
    connection: sqlite3.Connection, config: dict, *, representative_count: int = 8
) -> dict[str, Any]:
    """Sample the stop-fraction distribution and report every removed interval."""

    analyzed: list[tuple[sqlite3.Row, Any]] = []
    for row in _candidate_rows(connection):
        result = classify_movement(_load_points(connection, int(row["id"])), config["moving_time"])
        elapsed = sum(interval.elapsed_s for interval in result.intervals)
        stopped = sum(interval.stopped_time_s for interval in result.intervals)
        analyzed.append((row, result, stopped / elapsed if elapsed else 0.0))
    analyzed.sort(key=lambda item: (item[2], item[0]["start_time_utc"]))
    if not analyzed:
        return {"generated_at_utc": datetime.utcnow().isoformat(), "activities": []}
    count = min(representative_count, len(analyzed))
    positions = sorted(
        {
            round(index * (len(analyzed) - 1) / max(1, count - 1))
            for index in range(count)
        }
    )
    activities = []
    for position in positions:
        row, result, stop_fraction = analyzed[position]
        elapsed = sum(interval.elapsed_s for interval in result.intervals)
        moving = sum(interval.moving_time_s for interval in result.intervals)
        stopped = sum(interval.stopped_time_s for interval in result.intervals)
        slow_kept = sum(
            interval.very_slow_time_s
            for interval in result.intervals
            if interval.classification != "stopped"
        )
        activities.append(
            {
                "database_activity_id": int(row["id"]),
                "garmin_activity_id": row["activity_id"],
                "start_time_utc": row["start_time_utc"],
                "distance_miles": float(row["total_distance_m"] or 0) / 1609.344,
                "workout_type": row["workout_type"],
                "health_tag": row["health_tag"],
                "elapsed_trackpoint_seconds": elapsed,
                "classified_moving_seconds": moving,
                "removed_seconds": stopped,
                "removed_fraction": stop_fraction,
                "very_slow_but_kept_seconds": slow_kept,
                "device_timer_seconds": row["device_timer_time_s"],
                "removed_events": _group_events(result.intervals),
                "classifier_diagnostics": result.diagnostics,
            }
        )
    return {
        "generated_at_utc": datetime.utcnow().isoformat(),
        "selection": "eight quantiles of removed-time fraction among GPS activities since 2026-04-01",
        "settings": config["moving_time"],
        "interpretation": (
            "Removed events require sustained low distance plus low recorded/GPS speed; "
            "isolated slow intervals remain moving. Review event timestamps against route context."
        ),
        "activities": activities,
    }


def write_movement_diagnostic(
    connection: sqlite3.Connection, config: dict, output_path: str | Path
) -> dict[str, Any]:
    report = build_movement_diagnostic(connection, config)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
