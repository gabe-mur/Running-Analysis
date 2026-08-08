"""Persist manual activity metadata to SQLite and the editable override CSV."""

from __future__ import annotations

from pathlib import Path
import csv
import sqlite3

from .web.schemas import ActivityHealthTag, RunMetadataPatch
from .privacy import private_file


FIELDS = ["activity_id", "include_in_model", "workout_type", "illness", "health_tag", "perceived_exertion", "notes"]


def _text_bool(value: int | bool | None) -> str:
    if value is None:
        return ""
    return "1" if bool(value) else "0"


def update_run_metadata(
    connection: sqlite3.Connection,
    overrides_path: str | Path,
    activity_row_id: int,
    patch: RunMetadataPatch,
) -> None:
    activity = connection.execute(
        "SELECT activity_id,activity_uid FROM activities WHERE id=?", (activity_row_id,)
    ).fetchone()
    if not activity:
        raise LookupError("Run not found")
    key = str(activity["activity_id"] or activity["activity_uid"])
    existing = connection.execute("SELECT * FROM run_overrides WHERE activity_id=?", (key,)).fetchone()
    values = {
        "include_in_model": existing["include_in_model"] if existing else None,
        "workout_type": existing["workout_type"] if existing else None,
        "illness": existing["illness"] if existing else 0,
        "health_tag": existing["health_tag"] if existing and "health_tag" in existing.keys() else "normal",
        "perceived_exertion": existing["perceived_exertion"] if existing and "perceived_exertion" in existing.keys() else None,
        "notes": existing["notes"] if existing else None,
    }
    fields = patch.model_fields_set
    if "include_in_model" in fields:
        values["include_in_model"] = int(patch.include_in_model) if patch.include_in_model is not None else None
    if "workout_type" in fields:
        values["workout_type"] = patch.workout_type.value if patch.workout_type else None
    if "health_tag" in fields:
        values["health_tag"] = patch.health_tag.value if patch.health_tag else ActivityHealthTag.NORMAL.value
        values["illness"] = int(patch.health_tag in {ActivityHealthTag.ILLNESS, ActivityHealthTag.ILLNESS_RECOVERY}) if patch.health_tag else 0
    if "notes" in fields:
        values["notes"] = patch.notes or None
    if "perceived_exertion" in fields:
        values["perceived_exertion"] = patch.perceived_exertion
    connection.execute(
        """
        INSERT INTO run_overrides(activity_id,include_in_model,workout_type,illness,health_tag,perceived_exertion,notes)
        VALUES (?,?,?,?,?,?,?)
        ON CONFLICT(activity_id) DO UPDATE SET
          include_in_model=excluded.include_in_model,workout_type=excluded.workout_type,
          illness=excluded.illness,health_tag=excluded.health_tag,
          perceived_exertion=excluded.perceived_exertion,notes=excluded.notes
        """,
        (key, values["include_in_model"], values["workout_type"], values["illness"], values["health_tag"], values["perceived_exertion"], values["notes"]),
    )
    connection.commit()

    path = Path(overrides_path)
    rows: dict[str, dict[str, str]] = {}
    if path.exists():
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("activity_id"):
                    rows[row["activity_id"]] = {field: row.get(field, "") for field in FIELDS}
    rows[key] = {
        "activity_id": key,
        "include_in_model": _text_bool(values["include_in_model"]),
        "workout_type": str(values["workout_type"] or ""),
        "illness": _text_bool(values["illness"]),
        "health_tag": str(values["health_tag"] or "normal"),
        "perceived_exertion": str(values["perceived_exertion"] or ""),
        "notes": str(values["notes"] or ""),
    }
    # The default overrides file lives in the repository root. Protect the
    # file itself without changing permissions on the whole repository.
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows[key] for key in sorted(rows))
    private_file(temporary)
    temporary.replace(path)
    private_file(path)
