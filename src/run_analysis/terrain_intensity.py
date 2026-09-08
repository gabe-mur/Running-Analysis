"""Separate terrain context from the physiological cost of elevated HR.

Heart-rate load remains the source of truth for recovery.  This module answers
the narrower coaching question: how much moderate-zone time is plausibly
attributable to the extra energetic cost of climbing rather than to an
unexplained increase in easy-day effort?
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import sqlite3

from .physiology import grade_energy_ratio


@dataclass(frozen=True, slots=True)
class TerrainModerateContext:
    raw_moderate_minutes: float
    effective_moderate_minutes: float
    grade_attributed_minutes: float
    evidence_minutes: float
    available: bool


def terrain_moderate_context(
    connection: sqlite3.Connection,
    activity_id: int,
    *,
    z3_low_bpm: float,
    z3_high_bpm: float,
    raw_moderate_minutes: float,
) -> TerrainModerateContext:
    """Return a continuous, energy-based terrain attribution for Z3 time.

    Segment HR is only used to locate moderate-zone portions of the run.  For
    each such segment, ``1 - 1 / grade_energy_ratio`` is the share of its
    equivalent level-running cost attributable to climbing.  This avoids a
    binary "hilly route" cutoff and never erases the recorded HR from training
    load or recovery calculations.
    """

    raw = max(0.0, float(raw_moderate_minutes))
    if raw <= 0:
        return TerrainModerateContext(raw, raw, 0.0, 0.0, True)
    try:
        rows = connection.execute(
            """
            SELECT moving_time_s,average_hr_bpm,average_grade_percent,metrics_json
            FROM segments
            WHERE activity_id=? AND is_pathological=0
              AND moving_time_s>0 AND average_hr_bpm IS NOT NULL
            """,
            (activity_id,),
        ).fetchall()
    except sqlite3.OperationalError:
        # Small unit-test and legacy databases may not have processed segments.
        rows = []
    represented_seconds = 0.0
    attributed_seconds = 0.0
    for row in rows:
        heart_rate = float(row["average_hr_bpm"])
        if not z3_low_bpm <= heart_rate <= z3_high_bpm:
            continue
        seconds = float(row["moving_time_s"] or 0.0)
        if seconds <= 0:
            continue
        represented_seconds += seconds
        try:
            metrics = json.loads(row["metrics_json"] or "{}")
        except (json.JSONDecodeError, TypeError):
            metrics = {}
        ratio = metrics.get("grade_energy_ratio")
        if ratio is None and row["average_grade_percent"] is not None:
            ratio = grade_energy_ratio(float(row["average_grade_percent"]))
        ratio = float(ratio or 1.0)
        if ratio > 1.0:
            attributed_seconds += seconds * (1.0 - 1.0 / ratio)

    if represented_seconds <= 0:
        return TerrainModerateContext(raw, raw, 0.0, 0.0, False)
    attributed_fraction = min(1.0, attributed_seconds / represented_seconds)
    attributed_minutes = raw * attributed_fraction
    return TerrainModerateContext(
        raw_moderate_minutes=raw,
        effective_moderate_minutes=max(0.0, raw - attributed_minutes),
        grade_attributed_minutes=attributed_minutes,
        evidence_minutes=represented_seconds / 60.0,
        available=True,
    )
