"""Persist planned workouts and reconcile delayed activity uploads to them."""

from __future__ import annotations

from datetime import datetime, time, timezone
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
        if planned_for is None:
            planned_for = datetime.combine(
                day.date,
                time(12),
                tzinfo=schedule.generated_at.tzinfo or timezone.utc,
            )
        distance = result.distance_range_miles if result else None
        duration = result.duration_range_minutes if result else None
        connection.execute(
            """
            INSERT OR IGNORE INTO planned_workout_history(
                schedule_generated_at,plan_date,planned_for,workout_type,
                quality_session_type,title,distance_low_miles,
                distance_high_miles,duration_low_minutes,
                duration_high_minutes,recommendation_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                schedule.generated_at.isoformat(),
                day.date.isoformat(),
                planned_for.isoformat(),
                result.workout_type.value if result else "rest",
                (
                    result.quality_session_type.value
                    if result and result.quality_session_type is not None
                    else None
                ),
                result.title if result else "No run planned",
                distance[0] if distance else None,
                distance[1] if distance else None,
                duration[0] if duration else None,
                duration[1] if duration else None,
                result.model_dump_json() if result else "null",
            ),
        )


def _candidate_score(
    activity_start: datetime,
    activity_miles: float,
    activity_minutes: float,
    row: sqlite3.Row,
) -> tuple[float, float, float, float] | None:
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
    duration_low = row["duration_low_minutes"]
    duration_high = row["duration_high_minutes"]
    duration_delta = 0.0
    if duration_low is not None and duration_high is not None:
        if activity_minutes < float(duration_low):
            duration_delta = float(duration_low) - activity_minutes
        elif activity_minutes > float(duration_high):
            duration_delta = activity_minutes - float(duration_high)
    score = timing_delta / 6.0 + distance_delta * 2.0
    if duration_low is not None and duration_high is not None:
        reference_minutes = max(
            1.0,
            (float(duration_low) + float(duration_high)) / 2.0,
        )
        score += duration_delta / reference_minutes
    return score, timing_delta, distance_delta, duration_delta


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
            SELECT a.id,a.activity_id,a.start_time_utc,a.total_distance_m,
                   m.calculated_moving_time_s,m.device_timer_time_s
            FROM activities a
            LEFT JOIN activity_metrics m ON m.activity_id=a.id
            WHERE a.id=?
            """,
            (activity_id,),
        ).fetchone()
        if activity is None or not activity["start_time_utc"]:
            continue
        start = datetime.fromisoformat(activity["start_time_utc"])
        miles = float(activity["total_distance_m"] or 0) / METERS_PER_MILE
        minutes = float(
            activity["calculated_moving_time_s"]
            or activity["device_timer_time_s"]
            or 0
        ) / 60.0
        pre_run_candidates = [
            candidate
            for candidate in candidates
            if datetime.fromisoformat(candidate["schedule_generated_at"]) <= start
        ]
        if not pre_run_candidates:
            continue
        # A schedule snapshot is a single planning decision. Only the newest
        # snapshot that existed when the athlete started can establish what
        # they were asked to do; distance is evidence of adherence, not a way
        # to fish through older plans for a more convenient label.
        latest_generation = max(
            datetime.fromisoformat(candidate["schedule_generated_at"])
            for candidate in pre_run_candidates
        )
        active_candidates = [
            candidate
            for candidate in pre_run_candidates
            if datetime.fromisoformat(candidate["schedule_generated_at"])
            == latest_generation
        ]
        ranked: list[tuple[float, float, float, float, sqlite3.Row]] = []
        for candidate in active_candidates:
            if candidate["workout_type"] == "rest":
                continue
            score = _candidate_score(start, miles, minutes, candidate)
            if score is not None:
                ranked.append((*score, candidate))
        if not ranked:
            continue
        _, timing_delta, distance_delta, duration_delta, candidate = min(
            ranked,
            key=lambda item: (item[0], item[4]["planned_for"]),
        )
        confidence = (
            "high"
            if (
                timing_delta <= 6.0
                and distance_delta <= 0.25
                and duration_delta <= 1e-9
            )
            else "moderate"
        )
        connection.execute(
            """
            INSERT INTO activity_plan_matches(
                activity_id,planned_workout_id,timing_delta_hours,
                distance_delta_miles,duration_delta_minutes,
                match_confidence,matched_at_utc
            ) VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(activity_id) DO UPDATE SET
                planned_workout_id=excluded.planned_workout_id,
                timing_delta_hours=excluded.timing_delta_hours,
                distance_delta_miles=excluded.distance_delta_miles,
                duration_delta_minutes=excluded.duration_delta_minutes,
                match_confidence=excluded.match_confidence,
                matched_at_utc=excluded.matched_at_utc
            """,
            (
                activity_id,
                int(candidate["id"]),
                timing_delta,
                distance_delta,
                duration_delta,
                confidence,
                datetime.now(timezone.utc).isoformat(),
            ),
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
