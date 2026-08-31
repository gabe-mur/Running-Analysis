"""Persist planned workouts and reconcile delayed activity uploads to them."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import sqlite3

from .segmentation import METERS_PER_MILE
from .web.schemas import RecommendationResponse, WeeklyScheduleResponse


def archive_weekly_prescriptions(
    connection: sqlite3.Connection,
    schedule: WeeklyScheduleResponse,
) -> None:
    """Keep immutable workout snapshots beyond the latest visible plan."""

    for day in schedule.days:
        result = day.recommendation
        planned_for = result.planned_for if result else None
        if result is None or planned_for is None or result.workout_type.value == "rest":
            continue
        distance = result.distance_range_miles
        connection.execute(
            """
            INSERT OR IGNORE INTO planned_workout_history(
                schedule_generated_at,plan_date,planned_for,workout_type,
                quality_session_type,title,distance_low_miles,
                distance_high_miles,recommendation_json
            ) VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (
                schedule.generated_at.isoformat(),
                day.date.isoformat(),
                planned_for.isoformat(),
                result.workout_type.value,
                (
                    result.quality_session_type.value
                    if result.quality_session_type is not None
                    else None
                ),
                result.title,
                distance[0] if distance else None,
                distance[1] if distance else None,
                result.model_dump_json(),
            ),
        )


def _candidate_score(
    activity_start: datetime,
    activity_miles: float,
    row: sqlite3.Row,
) -> tuple[float, float, float] | None:
    generated_at = datetime.fromisoformat(row["schedule_generated_at"])
    planned_for = datetime.fromisoformat(row["planned_for"])
    # A plan generated after the activity began is a retrospective calendar,
    # not evidence of what the athlete was asked to do.
    if generated_at > activity_start:
        return None
    timing_delta = abs(
        (activity_start - planned_for).total_seconds()
    ) / 3600
    if timing_delta > 18.0:
        return None
    low = row["distance_low_miles"]
    high = row["distance_high_miles"]
    distance_delta = 0.0
    if low is not None and high is not None:
        if activity_miles < float(low):
            distance_delta = float(low) - activity_miles
        elif activity_miles > float(high):
            distance_delta = activity_miles - float(high)
        midpoint = (float(low) + float(high)) / 2
        if distance_delta > max(1.0, midpoint * 0.25):
            return None
    return timing_delta / 6.0 + distance_delta * 2.0, timing_delta, distance_delta


def match_activities_to_prescriptions(
    connection: sqlite3.Connection,
    activity_ids: list[int],
) -> list[int]:
    """Match new runs to the prescription that existed before they started.

    Timing and distance identify the intended session. Execution quality is
    evaluated later from laps, pace, and HR; a poorly executed prescribed
    workout is still an attempted instance of that workout rather than an
    unrelated easy run.
    """

    matched: list[int] = []
    candidates = connection.execute(
        "SELECT * FROM planned_workout_history"
    ).fetchall()
    for activity_id in activity_ids:
        activity = connection.execute(
            """
            SELECT id,activity_id,start_time_utc,total_distance_m
            FROM activities WHERE id=?
            """,
            (activity_id,),
        ).fetchone()
        if activity is None or not activity["start_time_utc"]:
            continue
        start = datetime.fromisoformat(activity["start_time_utc"])
        miles = float(activity["total_distance_m"] or 0) / METERS_PER_MILE
        ranked: list[tuple[float, float, float, sqlite3.Row]] = []
        for candidate in candidates:
            score = _candidate_score(start, miles, candidate)
            if score is not None:
                ranked.append((*score, candidate))
        if not ranked:
            continue
        _, timing_delta, distance_delta, candidate = min(
            ranked,
            key=lambda item: (item[0], item[3]["planned_for"]),
        )
        confidence = (
            "high"
            if timing_delta <= 6.0 and distance_delta <= 0.25
            else "moderate"
        )
        connection.execute(
            """
            INSERT INTO activity_plan_matches(
                activity_id,planned_workout_id,timing_delta_hours,
                distance_delta_miles,match_confidence,matched_at_utc
            ) VALUES (?,?,?,?,?,?)
            ON CONFLICT(activity_id) DO UPDATE SET
                planned_workout_id=excluded.planned_workout_id,
                timing_delta_hours=excluded.timing_delta_hours,
                distance_delta_miles=excluded.distance_delta_miles,
                match_confidence=excluded.match_confidence,
                matched_at_utc=excluded.matched_at_utc
            """,
            (
                activity_id,
                int(candidate["id"]),
                timing_delta,
                distance_delta,
                confidence,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        # Existing user metadata wins. When no explicit workout type exists,
        # project the matched prescription through the established metadata
        # path so load, fitness, and every report agree on classification.
        connection.execute(
            """
            INSERT INTO run_overrides(activity_id,workout_type)
            VALUES (?,?)
            ON CONFLICT(activity_id) DO UPDATE SET
                workout_type=COALESCE(run_overrides.workout_type,excluded.workout_type)
            """,
            (activity["activity_id"], candidate["workout_type"]),
        )
        matched.append(activity_id)
    connection.commit()
    return matched


def prescribed_recommendation_from_row(
    row: sqlite3.Row,
) -> RecommendationResponse | None:
    payload = (
        row["prescribed_recommendation_json"]
        if "prescribed_recommendation_json" in row.keys()
        else None
    )
    if not payload:
        return None
    try:
        return RecommendationResponse.model_validate(json.loads(payload))
    except (TypeError, ValueError):
        return None
