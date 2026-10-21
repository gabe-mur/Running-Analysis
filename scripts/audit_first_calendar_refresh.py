"""Trace the fresh synthetic calendar rollover without changing planner policy.

Rescore selected and original calendars inside the same active dose adapter,
using actual search inputs and the full objective. No DB, deployment, weights,
search-width changes, or long simulation. Hook outputs must match the earlier
uninstrumented control before drawing conclusions.
"""

from argparse import ArgumentParser
from datetime import datetime, timezone
import json
from pathlib import Path

import yaml

from scripts.audit_first_dose_refresh import capture_openings, sessions
from scripts.audit_easy_dose_search import trace_search, search_stage
from scripts.audit_easy_dose_program import capture_allocations
from run_analysis import weekly_schedule as planner


def score_calendar(inputs, offsets, *, roles=None):
    common = {key: inputs[key] for key in (
        "daily_states", "daily_state_options", "request", "config",
        "target_distance_range", "completed_miles_by_offset")}
    details = {}
    with capture_allocations() as allocations:
        joint = planner._joint_candidate_program_cost(
            offsets, **common, reconcile_allocated_prefix=True,
            expected_target_projector=inputs["expected_target_projector"],
            prior_distance_midpoints_by_date=inputs["prior_distance_midpoints_by_date"],
            role_overrides_by_offset=roles, diagnostic_details=details)
    materialize = planner._materialize_candidate_sessions
    coaching_sessions = []

    def capture(*args, **kwargs):
        result = materialize(*args, **kwargs)
        coaching_sessions.extend(result)
        return result

    try:
        planner._materialize_candidate_sessions = capture
        coaching = planner._adaptive_candidate_cost(
            offsets, **common, include_mileage_path=False,
            include_recovery_interactions=False, include_cadence_pressure=False,
            role_overrides_by_offset=roles)
    finally:
        planner._materialize_candidate_sessions = materialize
    load_costs = [{"date": s.state.as_of.date(), "range": s.recommendation.distance_range_miles,
                   "readiness": s.recommendation.readiness,
                   "load_ratio": planner.effective_load_ratio(s.state.recent_load),
                   "load_cost": max(0, (planner.effective_load_ratio(s.state.recent_load) or 0) - 1) * 12}
                  for s in coaching_sessions]
    ordinary = sum(planner.typical_easy_distance(inputs["daily_states"][0])) / 2
    return {"offsets": offsets, "roles": roles, "joint": joint,
            "coaching": coaching, "total": planner._program_selection_cost(joint, coaching, ordinary),
            "coaching_load_terms": load_costs,
            "coaching_other_cost": coaching - sum(x["load_cost"] for x in load_costs),
            "details": details,
            "sessions": [{"date": d.date, "type": d.recommendation.workout_type,
                          "range": d.recommendation.distance_range_miles,
                          "at": d.recommendation.planned_for}
                         for d in allocations["days"] if d.recommendation
                         and d.recommendation.workout_type != planner.WorkoutType.REST]}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--control-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        parser.error("Choose a fresh output path")
    control = json.loads(args.control_json.read_text())
    config = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.example.yaml").read_text())
    decisions = []

    def inspect(trace):
        inputs = trace["inputs"]
        # Run here, before build exits its history-led allocator context.
        probes = []
        if not decisions:
            origin = inputs["daily_states"][0].as_of.date()
            for label, calendar in (("opening_original", control["probes"][0]["sessions"]),
                                    ("opening_next_winner", control["probes"][1]["sessions"])):
                offsets = tuple((datetime.fromisoformat(s["date"]).date() - origin).days for s in calendar)
                row = score_calendar(inputs, offsets)
                row.update(name=label, search_stage=search_stage({"offsets": offsets}, trace))
                probes.append(row)
                print(label, row["total"], row["joint"], row["coaching"], flush=True)
        else:
            selected = tuple(trace["offsets"])
            prior = tuple(sorted(inputs["prior_run_offsets"]))
            for label, offsets, roles in (
                ("selected", selected, trace["diagnostics"]["winning_role_overrides"] or None),
                ("original", prior, None),
                ("original_protected_roles", prior,
                 {i: role for i, role in inputs["prior_roles_by_offset"].items()
                  if i < planner.NEAR_TERM_STABILITY_DECISION_OFFSETS}),
            ):
                row = score_calendar(inputs, offsets, roles=roles)
                row.update(name=label, search_stage=search_stage({"offsets": offsets}, trace))
                probes.append(row)
                print(label, row["total"], row["joint"], row["coaching"], flush=True)
        decisions.append({"as_of": inputs["daily_states"][0].as_of,
                          "offsets": trace["offsets"], "diagnostics": trace["diagnostics"],
                          "joint_calls": list(trace["joint_calls"]),
                          "shortlists": list(trace["shortlists"]), "probes": probes})
        trace["joint_calls"].clear()
        trace["shortlists"].clear()

    with trace_search(after_search=inspect):
        captures = capture_openings(config, datetime(2026, 9, 22, tzinfo=timezone.utc))
    for index, capture in enumerate(captures):
        expected = control["probes"][index]["sessions"]
        actual = json.loads(json.dumps(sessions(capture["plan"])))
        assert actual == expected, "Instrumentation changed the normal output"
        decisions[index]["published_sessions"] = actual
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("x") as out:
        json.dump({"scope": __doc__, "control": str(args.control_json),
                   "controls_identical": True, "decisions": decisions}, out, indent=2,
                  default=lambda x: x.model_dump(mode="json") if hasattr(x, "model_dump") else x.isoformat())
        out.write("\n")
    print(f"Wrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
