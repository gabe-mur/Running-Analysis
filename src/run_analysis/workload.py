"""Leakage-safe recent running workload features."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import sqlite3

from .segmentation import METERS_PER_MILE


@dataclass(slots=True)
class WorkloadInput:
    activity_row_id: int
    start_time: datetime
    distance_miles: float
    duration_minutes: float
    is_hard: bool


@dataclass(slots=True)
class WorkloadFeatures:
    activity_row_id: int
    previous_7d_miles: float
    previous_7d_minutes: float
    previous_28d_miles: float
    previous_28d_minutes: float
    days_since_previous_run: float | None
    days_since_previous_hard_run: float | None


def compute_workloads(records: list[WorkloadInput]) -> list[WorkloadFeatures]:
    ordered = sorted(records, key=lambda record: (record.start_time, record.activity_row_id))
    features: list[WorkloadFeatures] = []
    prior: list[WorkloadInput] = []
    previous: WorkloadInput | None = None
    previous_hard: WorkloadInput | None = None
    for current in ordered:
        start_7d = current.start_time - timedelta(days=7)
        start_28d = current.start_time - timedelta(days=28)
        previous_7d = [record for record in prior if record.start_time >= start_7d]
        previous_28d = [record for record in prior if record.start_time >= start_28d]
        features.append(
            WorkloadFeatures(
                activity_row_id=current.activity_row_id,
                previous_7d_miles=sum(record.distance_miles for record in previous_7d),
                previous_7d_minutes=sum(record.duration_minutes for record in previous_7d),
                previous_28d_miles=sum(record.distance_miles for record in previous_28d),
                previous_28d_minutes=sum(record.duration_minutes for record in previous_28d),
                days_since_previous_run=(current.start_time - previous.start_time).total_seconds() / 86400
                if previous
                else None,
                days_since_previous_hard_run=(
                    (current.start_time - previous_hard.start_time).total_seconds() / 86400
                    if previous_hard
                    else None
                ),
            )
        )
        prior.append(current)
        previous = current
        if current.is_hard:
            previous_hard = current
    return features


def update_workloads(connection: sqlite3.Connection, config: dict) -> int:
    z3_lower = float(config["zones"]["z3"][0])
    rows = connection.execute(
        """
        SELECT a.id,a.start_time_utc,a.total_distance_m,a.activity_id,
               m.calculated_moving_time_s,m.device_timer_time_s,m.moving_average_hr_bpm,
               m.hr_zone_seconds_json,m.exclusion_reason,o.workout_type
        FROM activities a JOIN activity_metrics m ON m.activity_id=a.id
        LEFT JOIN run_overrides o ON o.activity_id=a.activity_id
        WHERE a.start_time_utc IS NOT NULL ORDER BY a.start_time_utc_epoch,a.id
        """
    ).fetchall()
    records: list[WorkloadInput] = []
    for row in rows:
        workout_type = str(row["workout_type"] or "").casefold()
        exclusion = str(row["exclusion_reason"] or "")
        if workout_type in {"hike", "hiking", "bike", "cycling"} or any(
            marker in exclusion
            for marker in ("probable_walk_or_hike_sensor_signature", "probable_bike_sensor_signature")
        ):
            # These remain visible activities but do not redefine running load.
            continue
        duration_s = row["calculated_moving_time_s"]
        if not duration_s:
            duration_s = row["device_timer_time_s"] or 0.0
        zone_seconds = json.loads(row["hr_zone_seconds_json"] or "{}")
        hard_zone_seconds = sum(zone_seconds.get(name, 0.0) for name in ("z3", "z4", "z5", "above_z5"))
        is_hard = (
            (row["moving_average_hr_bpm"] is not None and row["moving_average_hr_bpm"] >= z3_lower)
            or (duration_s > 0 and hard_zone_seconds / duration_s >= 0.25)
            or workout_type in {"interval", "intervals", "tempo", "race"}
        )
        records.append(
            WorkloadInput(
                activity_row_id=int(row["id"]),
                start_time=datetime.fromisoformat(row["start_time_utc"]),
                distance_miles=float(row["total_distance_m"] or 0) / METERS_PER_MILE,
                duration_minutes=float(duration_s) / 60.0,
                is_hard=is_hard,
            )
        )
    computed = compute_workloads(records)
    for feature in computed:
        payload = {
            "previous_7d_miles": feature.previous_7d_miles,
            "previous_7d_minutes": feature.previous_7d_minutes,
            "previous_28d_miles": feature.previous_28d_miles,
            "previous_28d_minutes": feature.previous_28d_minutes,
            "days_since_previous_run": feature.days_since_previous_run,
            "days_since_previous_hard_run": feature.days_since_previous_hard_run,
            "future_activity_used": False,
        }
        connection.execute(
            """
            UPDATE activity_metrics SET previous_7d_miles=?,previous_7d_minutes=?,
                previous_28d_miles=?,previous_28d_minutes=?,days_since_previous_run=?,
                days_since_previous_hard_run=?,workload_json=? WHERE activity_id=?
            """,
            (
                feature.previous_7d_miles,
                feature.previous_7d_minutes,
                feature.previous_28d_miles,
                feature.previous_28d_minutes,
                feature.days_since_previous_run,
                feature.days_since_previous_hard_run,
                json.dumps(payload),
                feature.activity_row_id,
            ),
        )
    connection.commit()
    return len(computed)
