"""Synchronization of the user-editable run override CSV."""

from __future__ import annotations

from pathlib import Path
import csv
import sqlite3


def _boolean(value: str) -> int | None:
    normalized = value.strip().casefold()
    if not normalized:
        return None
    if normalized in {"1", "true", "yes", "y"}:
        return 1
    if normalized in {"0", "false", "no", "n"}:
        return 0
    raise ValueError(f"Invalid boolean value in run overrides: {value!r}")


def sync_overrides(connection: sqlite3.Connection, path: str | Path) -> int:
    override_path = Path(path)
    if not override_path.exists():
        return 0
    count = 0
    with override_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"activity_id", "include_in_model", "workout_type", "illness", "notes"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"run_overrides.csv must contain: {', '.join(sorted(required))}")
        for row in reader:
            activity_id = row["activity_id"].strip()
            if not activity_id:
                continue
            connection.execute(
                """
                INSERT INTO run_overrides(activity_id, include_in_model, workout_type, illness, notes, health_tag, perceived_exertion)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(activity_id) DO UPDATE SET
                    include_in_model=excluded.include_in_model,
                    workout_type=excluded.workout_type,
                    illness=excluded.illness,
                    notes=excluded.notes,
                    health_tag=excluded.health_tag,
                    perceived_exertion=excluded.perceived_exertion
                """,
                (
                    activity_id,
                    _boolean(row["include_in_model"]),
                    row["workout_type"].strip() or None,
                    _boolean(row["illness"]),
                    row["notes"].strip() or None,
                    row.get("health_tag", "").strip() or ("illness" if _boolean(row["illness"]) else "normal"),
                    int(row["perceived_exertion"]) if row.get("perceived_exertion", "").strip() else None,
                ),
            )
            count += 1
    connection.commit()
    return count
