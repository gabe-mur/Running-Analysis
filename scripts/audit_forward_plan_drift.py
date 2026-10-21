"""Compare still-actionable dates with plans issued at least two days earlier.

Read-only report analysis; does not replan, change gates, or assume that changes
with recorded material evidence are unjustified. Windows stay anchored to the
original visible plan, not to each later refresh's moving horizon.
"""

from argparse import ArgumentParser
from datetime import datetime, timedelta
import json
from pathlib import Path


def anchored_comparisons(replans, *, horizon_days=7, minimum_age_days=2):
    ordered = sorted(replans, key=lambda x: x["generated_at"])
    comparisons = []

    def window(item):
        tomorrow = datetime.fromisoformat(item["generated_at"]).date() + timedelta(days=1)
        start = datetime.fromisoformat(item["decision_start_date"]).date() if item.get("decision_start_date") else tomorrow
        return start, max(start, tomorrow) + timedelta(days=horizon_days)

    def sessions(item, start, end):
        return {datetime.fromisoformat(x["planned_for"]).date(): x for x in item["planned_sessions"]
                if start <= datetime.fromisoformat(x["planned_for"]).date() < end}

    for i, anchor in enumerate(ordered):
        anchor_at = datetime.fromisoformat(anchor["generated_at"])
        anchor_start, anchor_end = window(anchor)
        for j in range(i + 1, len(ordered)):
            current = ordered[j]
            age = (datetime.fromisoformat(current["generated_at"]) - anchor_at).total_seconds() / 86400
            if age < minimum_age_days:
                continue
            start = max(anchor_start, window(current)[0])
            if start >= anchor_end:
                continue
            before, after = sessions(anchor, start, anchor_end), sessions(current, start, anchor_end)
            shared = set(before) & set(after)
            types = {day for day in shared if before[day]["workout_type"] != after[day]["workout_type"]}
            distances = [{"date": day.isoformat(), "before_miles": before[day]["midpoint_miles"],
                          "after_miles": after[day]["midpoint_miles"],
                          "delta_miles": after[day]["midpoint_miles"] - before[day]["midpoint_miles"]}
                         for day in sorted(shared - types)
                         if abs(after[day]["midpoint_miles"] - before[day]["midpoint_miles"]) > 1e-9]
            comparisons.append({
                "anchor_at": anchor["generated_at"], "current_at": current["generated_at"],
                "age_days": age, "remaining_window_start": str(start), "original_window_end_exclusive": str(anchor_end),
                "date_slot_changes": [str(x) for x in sorted(set(before) ^ set(after))],
                "workout_type_changes": [str(x) for x in sorted(types)], "distance_changes": distances,
                "material_evidence_since_anchor": sorted({reason for x in ordered[i + 1:j + 1]
                                                          for reason in x.get("material_evidence_reasons", [])}),
            })
    return comparisons


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        parser.error("Choose a new report path")
    report = json.loads(args.report.read_text())
    result = {"scope": __doc__, "source": str(args.report),
              "windows": {str(n): anchored_comparisons(report["replans"], horizon_days=n) for n in (4, 7)}}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("x") as out:
        json.dump(result, out, indent=2)
        out.write("\n")
    print(json.dumps({n: {"comparisons": len(rows),
                         "date_or_type_changed_pairs": sum(bool(x["date_slot_changes"] or x["workout_type_changes"]) for x in rows),
                         "distance_changed_pairs": sum(bool(x["distance_changes"]) for x in rows)}
                      for n, rows in result["windows"].items()}))


if __name__ == "__main__":
    main()
