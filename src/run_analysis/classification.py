"""Conservative activity-type review candidates; no automatic relabeling."""

from __future__ import annotations

from pathlib import Path
import csv
import sqlite3


def write_activity_type_review(connection: sqlite3.Connection, config: dict, path: str | Path) -> int:
    settings = config["activity_classification"]
    rows = connection.execute(
        """
        SELECT a.activity_id,a.start_time_local,a.total_distance_m/1609.344 AS distance_miles,
               m.moving_pace_min_mile,m.moving_average_hr_bpm,m.stop_fraction,
               AVG(s.average_cadence_spm) AS average_cadence_spm,m.model_eligible,m.exclusion_reason,
               a.notes,o.activity_id AS override_activity_id
        FROM activities a JOIN activity_metrics m ON m.activity_id=a.id
        LEFT JOIN segments s ON s.activity_id=a.id AND s.is_pathological=0
        LEFT JOIN run_overrides o ON o.activity_id=a.activity_id
        GROUP BY a.id
        """
    ).fetchall()
    candidates = []
    for row in rows:
        # A manual override is a resolved classification, even when its sensor
        # signature would otherwise place it in the review queue.
        if row["override_activity_id"] is not None:
            continue
        pace = row["moving_pace_min_mile"]
        cadence_spm = row["average_cadence_spm"]
        if pace is None or cadence_spm is None:
            continue
        score = 0
        reasons = []
        suggestion = "review"
        confidence = "low"
        if pace >= float(settings["high_confidence_walk_pace_min_mile"]):
            score += 5
            reasons.append("walking-range moving pace")
        elif pace >= 15:
            score += 3
            reasons.append("unusually slow relative to corpus")
        elif pace >= float(settings["review_slow_pace_min_mile"]):
            score += 1
            reasons.append("slow-tail pace")
        # Thresholds are total steps per minute, matching the canonical
        # conversion applied when segments were written.
        if cadence_spm is not None:
            if cadence_spm <= float(settings["high_confidence_walk_cadence_max_spm"]):
                score += 5
                reasons.append("walking-range cadence")
            elif cadence_spm < float(settings["very_low_cadence_max_spm"]):
                score += 3
                reasons.append("very low cadence")
            elif cadence_spm <= float(settings["review_low_cadence_max_spm"]):
                score += 1
                reasons.append("low cadence")
        if pace <= float(settings["high_confidence_bike_pace_min_mile"]) and row["distance_miles"] >= 5:
            score += 8
            reasons.append("bike-range sustained pace")
            suggestion = "bike"
        elif (
            pace >= float(settings["high_confidence_walk_pace_min_mile"])
            and cadence_spm is not None
            and cadence_spm <= float(settings["high_confidence_walk_cadence_max_spm"])
        ):
            suggestion = "hike_or_walk"
            confidence = "high"
        elif score >= 5:
            suggestion = "walk_run_or_hike"
            confidence = "medium"
        if score >= int(settings["minimum_review_score"]):
            candidates.append((score, dict(row), suggestion, confidence, reasons))
    candidates.sort(key=lambda item: (-item[0], item[1]["start_time_local"]))
    candidates = candidates[: int(settings["review_limit"])]
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "activity_id",
        "start_time_local",
        "distance_miles",
        "moving_pace_min_mile",
        "moving_average_hr_bpm",
        "average_cadence_spm",
        "stop_fraction",
        "suggested_type",
        "confidence",
        "evidence",
        "currently_model_eligible",
        "current_exclusion_reason",
        "notes",
        "user_decision",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for _score, row, suggestion, confidence, reasons in candidates:
            writer.writerow(
                {
                    "activity_id": row["activity_id"],
                    "start_time_local": row["start_time_local"],
                    "distance_miles": round(row["distance_miles"], 3),
                    "moving_pace_min_mile": round(row["moving_pace_min_mile"], 3),
                    "moving_average_hr_bpm": round(row["moving_average_hr_bpm"], 1)
                    if row["moving_average_hr_bpm"] is not None
                    else "",
                    "average_cadence_spm": round(row["average_cadence_spm"], 1)
                    if row["average_cadence_spm"] is not None
                    else "",
                    "stop_fraction": round(row["stop_fraction"], 3),
                    "suggested_type": suggestion,
                    "confidence": confidence,
                    "evidence": "; ".join(reasons),
                    "currently_model_eligible": row["model_eligible"],
                    "current_exclusion_reason": row["exclusion_reason"] or "",
                    "notes": row["notes"] or "",
                    "user_decision": "",
                }
            )
    return len(candidates)
