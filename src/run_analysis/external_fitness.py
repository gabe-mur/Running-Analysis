"""Inspectable Garmin/external fitness snapshots kept separate from run scoring."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import sqlite3

from .web.schemas import (
    ConfidenceLevel,
    ExternalFitnessSnapshot,
    ExternalFitnessSnapshotInput,
    ExternalFitnessSummary,
    FitnessTrend,
)


def save_snapshot(
    connection: sqlite3.Connection, snapshot: ExternalFitnessSnapshotInput
) -> ExternalFitnessSnapshot:
    connection.execute(
        """
        INSERT INTO external_fitness_snapshots(
            measured_at,vo2_max,predicted_5k_seconds,predicted_10k_seconds,
            predicted_half_marathon_seconds,predicted_marathon_seconds,source,created_at_utc
        ) VALUES (?,?,?,?,?,?,?,?)
        ON CONFLICT(measured_at,source) DO UPDATE SET
            vo2_max=excluded.vo2_max,
            predicted_5k_seconds=excluded.predicted_5k_seconds,
            predicted_10k_seconds=excluded.predicted_10k_seconds,
            predicted_half_marathon_seconds=excluded.predicted_half_marathon_seconds,
            predicted_marathon_seconds=excluded.predicted_marathon_seconds
        """,
        (
            snapshot.measured_at.isoformat(),
            snapshot.vo2_max,
            snapshot.predicted_5k_seconds,
            snapshot.predicted_10k_seconds,
            snapshot.predicted_half_marathon_seconds,
            snapshot.predicted_marathon_seconds,
            snapshot.source,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    connection.commit()
    row = connection.execute(
        "SELECT * FROM external_fitness_snapshots WHERE measured_at=? AND source=?",
        (snapshot.measured_at.isoformat(), snapshot.source),
    ).fetchone()
    assert row is not None
    return _snapshot(row)


def _snapshot(row: sqlite3.Row) -> ExternalFitnessSnapshot:
    return ExternalFitnessSnapshot(
        id=int(row["id"]),
        measured_at=row["measured_at"],
        vo2_max=row["vo2_max"],
        predicted_5k_seconds=row["predicted_5k_seconds"],
        predicted_10k_seconds=row["predicted_10k_seconds"],
        predicted_half_marathon_seconds=row["predicted_half_marathon_seconds"],
        predicted_marathon_seconds=row["predicted_marathon_seconds"],
        source=row["source"],
    )


def list_snapshots(
    connection: sqlite3.Connection, as_of: datetime, days: int = 365
) -> list[ExternalFitnessSnapshot]:
    start = (as_of.date() - timedelta(days=days)).isoformat()
    rows = connection.execute(
        """SELECT * FROM external_fitness_snapshots
           WHERE measured_at>=? AND measured_at<=? ORDER BY measured_at,id""",
        (start, as_of.date().isoformat()),
    ).fetchall()
    return [_snapshot(row) for row in rows]


def _direction(first: float | int | None, last: float | int | None, *, lower_is_better: bool) -> FitnessTrend:
    if first is None or last is None:
        return FitnessTrend.INSUFFICIENT_DATA
    delta = float(last) - float(first)
    threshold = max(abs(float(first)) * 0.01, 0.5 if not lower_is_better else 15.0)
    if abs(delta) < threshold:
        return FitnessTrend.STABLE
    improved = delta < 0 if lower_is_better else delta > 0
    return FitnessTrend.IMPROVING if improved else FitnessTrend.DECLINING


def summarize_external_fitness(
    connection: sqlite3.Connection, as_of: datetime
) -> ExternalFitnessSummary:
    snapshots = list_snapshots(connection, as_of)
    vo2 = [item for item in snapshots if item.vo2_max is not None]
    race = [item for item in snapshots if item.predicted_5k_seconds is not None]
    vo2_trend = _direction(
        vo2[0].vo2_max if vo2 else None,
        vo2[-1].vo2_max if vo2 else None,
        lower_is_better=False,
    )
    race_trend = _direction(
        race[0].predicted_5k_seconds if race else None,
        race[-1].predicted_5k_seconds if race else None,
        lower_is_better=True,
    )
    series_count = max(len(vo2), len(race))
    confidence = (
        ConfidenceLevel.MODERATE if series_count >= 3
        else ConfidenceLevel.LOW if series_count >= 2
        else ConfidenceLevel.UNAVAILABLE
    )
    if vo2_trend == FitnessTrend.IMPROVING and race_trend == FitnessTrend.IMPROVING:
        interpretation = "Garmin VO₂ max and predicted 5K have both improved over the last year."
    elif not snapshots:
        interpretation = "No Garmin snapshots have been added yet."
    else:
        interpretation = "Garmin's estimates are mixed, or there are too few snapshots to show a trend."
    return ExternalFitnessSummary(
        snapshots=snapshots,
        vo2_max_trend=vo2_trend,
        race_prediction_trend=race_trend,
        confidence=confidence,
        interpretation=interpretation,
    )
