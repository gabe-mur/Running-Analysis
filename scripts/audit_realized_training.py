"""Read-only moving-horizon delivery audit, not a new scheduler objective.

Calendar-day windows are half-open, in the simulation's timestamp timezone.
History fills opening rolling windows; wholly simulated windows are identified.
Forecast cohorts overlap and are not independent outcomes. Date/type matching
does not establish identity when workouts move. An absent forecast run is not
automatically unjustified churn, especially in imperfect-adherence scenarios.
"""

from argparse import ArgumentParser
from datetime import datetime, timedelta
import json
from pathlib import Path

from run_analysis.adherence_projection import QUALITY_TYPES


def at(value):
    return datetime.fromisoformat(value)


def _day(value):
    return value.replace(hour=0, minute=0, second=0, microsecond=0)


def _bounds(snapshot):
    generated = at(snapshot["generated_at"])
    start = _day(generated)
    if snapshot.get("decision_start_date"):
        start = at(snapshot["decision_start_date"]).replace(tzinfo=generated.tzinfo)
    return max(start, generated)


def audit(report):
    ledger = report["realized_execution"]
    start, until = at(ledger["start_at"]), at(ledger["observed_until"])
    actual = sorted(ledger["sessions"], key=lambda x: x["occurred_at"])
    history = ledger["historical_sessions"]
    all_runs = sorted(history + actual, key=lambda x: x["occurred_at"])
    snapshots = sorted(report["replans"], key=lambda x: x["generated_at"])
    rolling = []
    end = _day(start) + timedelta(days=1)
    while end <= until:
        row = {"day": (end - timedelta(days=1)).date().isoformat(), "windows": {}}
        for days in (7, 14, 28):
            lower = end - timedelta(days=days)
            runs = [x for x in all_runs if lower <= at(x["occurred_at"]) < end]
            miles = sum(x["distance_miles"] for x in runs)
            row["windows"][str(days)] = {
                "miles": miles, "runs": len(runs), "weekly_equivalent_miles": miles * 7 / days,
                "wholly_simulated": lower >= start,
            }
        rolling.append(row)
        end += timedelta(days=1)

    # Distinct date commitments from first entry into each near-term window,
    # including rest dates so surprise additions cannot disappear from counts.
    commitments = {}
    cohorts = {}
    for horizon in (4, 7):
        by_date, forecasts = {}, []
        for snap in snapshots:
            decision = _bounds(snap)
            lower = _day(decision)
            upper = lower + timedelta(days=horizon)
            planned = [x for x in snap["planned_sessions"]
                       if decision <= at(x["planned_for"]) < upper]
            mapping = {at(x["planned_for"]).date().isoformat(): x for x in planned}
            for offset in range(horizon):
                day = lower + timedelta(days=offset)
                key = day.date().isoformat()
                if key not in by_date:
                    by_date[key] = {"date": key, "first_seen_at": snap["generated_at"],
                                    "initial": mapping.get(key), "revisions": []}
                item = by_date[key]
                # A post-upload plan dropping consumed work is not a revision.
                performed = [x for x in actual if at(x["occurred_at"]).date().isoformat() == key]
                if any(at(x["occurred_at"]) <= at(snap["generated_at"]) for x in performed):
                    continue
                current = mapping.get(key)
                initial = item["initial"]
                def signature(session):
                    return None if session is None else (
                        session["workout_type"], session.get("planning_role"), session.get("distance_low_miles"),
                        session.get("distance_high_miles"), session["midpoint_miles"])
                prior = item["revisions"][-1]["plan"] if item["revisions"] else initial
                if signature(prior) != signature(current):
                    item["revisions"].append({"at": snap["generated_at"], "plan": current,
                                              "material_evidence": snap.get("material_evidence_reasons", [])})
            if upper <= until:
                delivered = [x for x in actual if decision <= at(x["occurred_at"]) < upper]
                forecasts.append({
                    "issued_at": snap["generated_at"], "from": decision.isoformat(),
                    "until_exclusive": upper.isoformat(),
                    "forecast_runs": len(planned), "delivered_runs": len(delivered),
                    "forecast_low_miles": sum(x.get("distance_low_miles") if x.get("distance_low_miles") is not None else x["midpoint_miles"] for x in planned),
                    "forecast_high_miles": sum(x.get("distance_high_miles") if x.get("distance_high_miles") is not None else x["midpoint_miles"] for x in planned),
                    "delivered_miles": sum(x["distance_miles"] for x in delivered),
                })
        matured, pending = [], []
        for key, item in sorted(by_date.items()):
            end_day = at(key).replace(tzinfo=start.tzinfo) + timedelta(days=1)
            if end_day > until:
                pending.append(item)
                continue
            runs = [x for x in actual if at(x["occurred_at"]).date().isoformat() == key
                    and at(x["occurred_at"]) >= at(item["first_seen_at"])]
            initial = item["initial"]
            if initial is None:
                outcome = "added_run" if runs else "rest_preserved"
            elif not runs:
                outcome = "forecast_run_not_completed_on_date"
            elif len(runs) != 1 or runs[0]["workout_type"] != initial["workout_type"]:
                outcome = "type_or_count_changed"
            else:
                low = initial.get("distance_low_miles")
                high = initial.get("distance_high_miles")
                if low is None or high is None:
                    outcome = "same_date_type_range_unknown"
                else:
                    outcome = "within_initial_range" if low - 1e-9 <= runs[0]["distance_miles"] <= high + 1e-9 else "outside_initial_range"
            matured.append({**item, "actual": runs, "outcome": outcome})
        counts = {}
        for item in matured:
            counts[item["outcome"]] = counts.get(item["outcome"], 0) + 1
        commitments[str(horizon)] = {"outcomes": counts, "matured": matured, "pending": pending,
                                     "matured_dates_with_revisions": sum(bool(x["revisions"]) for x in matured)}
        cohorts[str(horizon)] = forecasts

    gaps = {}
    for name, types in (("all", None), ("long", {"long"}),
                        ("quality", {x.value for x in QUALITY_TYPES})):
        relevant = [x for x in all_runs if types is None or x["workout_type"] in types]
        intervals = [{"from": a["occurred_at"], "to": b["occurred_at"],
                      "days": (at(b["occurred_at"]) - at(a["occurred_at"])).total_seconds() / 86400}
                     for a, b in zip(relevant, relevant[1:]) if at(b["occurred_at"]) >= start]
        prior = [x for x in relevant if at(x["occurred_at"]) < until]
        gaps[name] = {"completed_intervals": intervals,
                      "open_gap_at_end_days": (until - at(prior[-1]["occurred_at"])).total_seconds() / 86400 if prior else None}

    return {"scope": __doc__, "scenario": report["mode"], "complete": ledger["complete"],
            "start_at": ledger["start_at"], "observed_until": ledger["observed_until"],
            "total_completed_miles": sum(x["distance_miles"] for x in actual),
            "completed_runs": len(actual), "completed_sessions": actual,
            "rolling_daily": rolling, "commitments": commitments, "forecast_delivery_cohorts": cohorts,
            "session_gaps": gaps, "capacity_audit": report.get("capacity_audit"),
            "failures": report.get("failures", []), "advisories": report.get("advisories", [])}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    result = audit(json.loads(args.report.read_text()))
    with args.output_json.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("complete", "total_completed_miles", "completed_runs")}))
    print(json.dumps({h: rows["outcomes"] for h, rows in result["commitments"].items()}))


if __name__ == "__main__":
    main()
