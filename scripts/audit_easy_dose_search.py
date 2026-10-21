"""Bounded, read-only snapshot audit of search/dose/continuity interactions.

Calls the actual build path, then scores a common calendar pool with its exact
saved-distance context and frozen target projector. Does not deploy, mutate a
database, change weights, or claim a daily-replanning/global-optimality result.
The supplied historical prior is intentional; this bypasses the application's
version-migration filter, not any planner recovery check.
"""

from argparse import ArgumentParser
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from inspect import signature
import json
from pathlib import Path
from time import perf_counter

from scripts import audit_easy_dose_program as audit
from run_analysis.web.schemas import RecommendationRequest, TrailingDayActivity

planner = audit.planner


@contextmanager
def trace_search(*, after_search=None):
    names = ("adaptive_run_day_offsets", "_joint_candidate_program_cost",
             "_select_joint_finalists")
    originals = {name: getattr(planner, name) for name in names}
    records = {"joint_calls": [], "shortlists": []}
    searching = False

    def adaptive(*args, **kwargs):
        nonlocal searching
        bound = signature(originals["adaptive_run_day_offsets"]).bind(*args, **kwargs)
        bound.apply_defaults()
        records["inputs"] = bound.arguments
        searching = True
        try:
            result = originals["adaptive_run_day_offsets"](*args, **kwargs)
        finally:
            searching = False
        records["offsets"] = result
        records["diagnostics"] = asdict(bound.arguments["_planner_diagnostics"])
        if after_search is not None:
            after_search(records)
        return result

    def joint(*args, **kwargs):
        result = originals["_joint_candidate_program_cost"](*args, **kwargs)
        if searching and kwargs.get("_allocated_override") is None:
            records["joint_calls"].append({
                "offsets": list(args[0]), "joint_cost": result,
                "reconciled": kwargs.get("reconcile_allocated_prefix", False),
                "roles": kwargs.get("role_overrides_by_offset"),
                "details": dict(kwargs.get("diagnostic_details") or {}),
            })
        return result

    def shortlist(scored, *args, **kwargs):
        result = originals["_select_joint_finalists"](scored, *args, **kwargs)
        records["shortlists"].append({"ranked": sorted(scored), "selected": result})
        return result

    try:
        planner.adaptive_run_day_offsets = adaptive
        planner._joint_candidate_program_cost = joint
        planner._select_joint_finalists = shortlist
        yield records
    finally:
        for name, original in originals.items():
            setattr(planner, name, original)


def build_inputs(case):
    payload = case["payload"]
    return dict(
        daily_states=case["states"], daily_state_options=case["options"],
        request=RecommendationRequest.model_validate(payload["request"]),
        config=payload["config"], target_run_count=payload.get("target_run_count"),
        target_distance_range=tuple(payload["target_distance_range"]),
        target_evidence=case["evidence"], prior_schedule=case["prior"],
        completed_activities_by_offset={int(k): [TrailingDayActivity.model_validate(x) for x in v]
                                        for k, v in payload.get("completed_activities_by_offset", {}).items()},
        forced_rest_offsets=set(payload.get("forced_rest_offsets", [])),
        expected_target_projector=lambda _candidate, n: case["targets"][:n],
    )


def plan_summary(plan):
    days = plan.planning_days or plan.days
    sessions = [{"date": day.date, "workout_type": day.recommendation.workout_type,
                 "range": day.recommendation.distance_range_miles,
                 "planned_for": day.recommendation.planned_for}
                for day in days if day.recommendation and
                day.recommendation.workout_type != audit.WorkoutType.REST]
    return {"sessions": sessions, "miles": sum(sum(x["range"]) / 2 for x in sessions if x["range"])}


def search_stage(row, trace):
    offsets = tuple(row["offsets"])
    ranks = [next((i for i, (_, dates) in enumerate(group["ranked"], 1)
                   if tuple(dates) == offsets), None) for group in trace["shortlists"]]
    calls = [call for call in trace["joint_calls"] if tuple(call["offsets"]) == offsets]
    return {"preliminary_ranks": [x for x in ranks if x is not None],
            "joint_calls": calls,
            "reached_reconciled_scoring": any(x["reconciled"] for x in calls)}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--variants", default="current,history_trajectory")
    args = parser.parse_args()
    if args.output_json.exists():
        parser.error("Choose a fresh output path")
    variants = args.variants.split(",")
    if any(x not in {"current", "history_trajectory"} for x in variants):
        parser.error("Unknown variant")
    start = perf_counter()
    case = audit.load_case(args.database)
    inputs = build_inputs(case)
    trajectory = audit.dose_trajectory(
        case["profile"], [x.as_of.date() for x in case["states"]], case["targets"][:21],
        case["evidence"].capacity_reference_miles, mode="history_trajectory",
        fixed_fraction=case["fixed_fraction"],
    )
    priors = audit.eligible_prior_prescriptions(case["prior"], trajectory.dates,
                                               excluded_offsets=set(range(21)) - case["allowed"])
    pool = audit.opportunity_pool(case["seed"], case["allowed"], case["required"], max_additions=3)
    report = {"scope": __doc__, "snapshot_as_of": case["states"][0].as_of,
              "snapshot_version": case["payload"].get("planner_version"),
              "prior_version": case["prior"].planner_version if case["prior"] else None,
              "planner_version": planner.WEEKLY_PLANNER_VERSION,
              "targets": case["targets"], "variants": []}
    for mode in variants:
        stamp = perf_counter()
        print(f"Normal search: {mode}", flush=True)
        policy = nullcontext() if mode == "current" else audit.history_allocator(
            case["profile"], trajectory, planner._established_easy_midpoint_floor(case["states"][0]),
            prior_prescriptions=priors)
        with policy:
            with trace_search() as trace:
                plan = planner.build_weekly_schedule(**inputs)
            print(f"  selected {trace['offsets']}, built {plan_summary(plan)['miles']:.2f} miles", flush=True)
            search_inputs = trace.pop("inputs")
            midpoints = search_inputs["prior_distance_midpoints_by_date"]
            candidates = sorted(set(pool) | {tuple(trace["offsets"])})
            print(f"Matched-context pool: {len(candidates)} calendars", flush=True)
            rows = audit.evaluate_pool(case, candidates, prior_distance_midpoints_by_date=midpoints)
            selected = next(row for row in rows if tuple(row["offsets"]) == tuple(trace["offsets"]))
            # Preserve the actual role-stable selected candidate as a separate
            # alternative; do not equate same dates with same workout purposes.
            overrides = trace["diagnostics"]["winning_role_overrides"]
            role_selected = audit.evaluate_pool(case, [tuple(trace["offsets"])],
                prior_distance_midpoints_by_date=midpoints, role_overrides_by_offset=overrides)[0] if overrides else selected
            probe_dates = sorted({tuple(row["offsets"]) for row in rows[:3]} |
                                 {tuple(trace["offsets"]), tuple(case["seed"])})
            no_hints = audit.evaluate_pool(case, probe_dates)
            # This ablation removes only prior distance hints, not saved dose
            # options, target projection, or the date-continuity mechanism.
            report["variants"].append({
                "variant": mode, "seconds": perf_counter() - stamp,
                "normal_plan": plan_summary(plan), "normal_trace": trace,
                "normal_calendar_rescored": selected, "normal_roles_rescored": role_selected,
                "pool_winner": rows[0], "pool_winner_search_stage": search_stage(rows[0], trace),
                "distance_hint_ablation": no_hints, "candidates": rows,
                "prior_midpoints": {str(k): v for k, v in midpoints.items()},
            })
        print(f"  pool best {rows[0]['offsets']}: {rows[0]['planned_midpoint_miles']:.2f} miles, score {rows[0]['score']:.3f}; selected rescore {role_selected['score']:.3f}", flush=True)
    report["seconds"] = perf_counter() - start
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("x") as stream:
        json.dump(report, stream, indent=2, default=lambda x: x.model_dump(mode="json") if hasattr(x, "model_dump") else x.isoformat())
        stream.write("\n")
    print(f"Wrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
