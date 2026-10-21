"""Rescore saved funding alternatives on one frozen input; no calendar search."""

from argparse import ArgumentParser
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

from scripts.audit_capped_calendar import SEARCH_SIGNATURE, thaw_inputs, write_new
from scripts.audit_first_calendar_refresh import score_calendar
from scripts.replay_easy_dose_program import replay_policy
from run_analysis import weekly_schedule as planner


def finalized_coaching_probe(inputs, candidate, scored):
    """Change only coaching's session input; retain all objective weights."""
    origin = inputs["daily_states"][0].as_of.date()
    ranges = {(datetime.fromisoformat(str(s["date"])).date() - origin).days: tuple(s["range"])
              for s in scored["sessions"]}
    common = {key: inputs[key] for key in ("daily_states", "daily_state_options", "request", "config")}
    roles = {int(k): v for k, v in candidate["roles"].items()} or None
    materialized = planner._materialize_candidate_sessions(tuple(candidate["offsets"]), **common,
        projected_distance_ranges=ranges, role_overrides_by_offset=roles)
    expected = [(str(x["date"]), str(x["type"]), tuple(x["range"]), x["at"].isoformat()) for x in scored["sessions"]]
    actual = [(s.recommendation.planned_for.date().isoformat(), s.recommendation.workout_type.value,
               s.recommendation.distance_range_miles, s.recommendation.planned_for.isoformat()) for s in materialized]
    # Regeneration is diagnostic only: do not pretend a changed dose/time/type
    # is the same finalized program if reconciliation does not reproduce it.
    if actual != expected:
        return {"sessions_match": False, "expected": expected, "actual": actual}
    original = planner._materialize_candidate_sessions
    try:
        planner._materialize_candidate_sessions = lambda *a, **kw: materialized
        cost = planner._adaptive_candidate_cost(tuple(candidate["offsets"]), **common,
            target_distance_range=inputs["target_distance_range"],
            completed_miles_by_offset=inputs["completed_miles_by_offset"],
            include_mileage_path=False, include_recovery_interactions=False,
            include_cadence_pressure=False, role_overrides_by_offset=roles)
    finally:
        planner._materialize_candidate_sessions = original
    return {"sessions_match": True, "original_coaching": scored["coaching"],
            "finalized_coaching": cost, "delta": cost - scored["coaching"],
            "counterfactual_total": scored["total"] - scored["coaching"] + cost,
            "load_terms": [{"date": s.recommendation.planned_for.date(),
                "ratio": planner.effective_load_ratio(s.state.recent_load)} for s in materialized]}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        parser.error("Choose a new output path")
    audit = json.loads(args.audit_json.read_text())
    if not audit["first_four_rounds_identical"]:
        parser.error("Require the matched control audit")
    frozen = json.loads(Path(audit["frozen_input"]).read_text())
    inputs = thaw_inputs(frozen)
    trace = audit["probe_trace"]
    accepted = [x for x in trace["evaluated"] if not x["acceptance"]["rejected"]]
    winner = trace["winner"]
    selected = [("selected", next(x for x in accepted if x["offsets"] == winner["offsets"] and not x["roles"]))]
    for label, minimum in (("more_funded", 48), ("nominally_funded", winner["integrated_target_low_miles"] - .5)):
        selected.append((label, min((x for x in accepted if x["planned_midpoint_miles"] >= minimum),
                                  key=lambda x: x["total_cost"])))
    rows = []
    original = planner.adaptive_run_day_offsets

    def inspect(*positional, **kwargs):
        bound = SEARCH_SIGNATURE.bind(*positional, **kwargs)
        bound.apply_defaults()
        for label, candidate in selected:
            stamp = perf_counter()
            row = score_calendar(bound.arguments, tuple(candidate["offsets"]),
                                 roles={int(k): v for k, v in candidate["roles"].items()} or None)
            assert abs(row["total"] - candidate["total_cost"]) < 1e-9
            probe = finalized_coaching_probe(bound.arguments, candidate, row)
            rows.append({"label": label, "seconds": perf_counter() - stamp, "score": row,
                         "finalized_coaching_probe": probe})
        return list(winner["offsets"])

    try:
        planner.adaptive_run_day_offsets = inspect
        with replay_policy("history_trajectory"):
            planner.build_weekly_schedule(**inputs)
    finally:
        planner.adaptive_run_day_offsets = original
    write_new(args.output_json, {"scope": __doc__, "source": str(args.audit_json), "rows": rows})
    print(f"Rescored {len(rows)} candidates in {sum(x['seconds'] for x in rows):.3f}s; {args.output_json}")


if __name__ == "__main__":
    main()
