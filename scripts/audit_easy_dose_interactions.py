"""Small crossed comparison of calendars, dose policy, and saved-dose options.

Consumes a prior matched-context audit; does not rerun or widen normal search.
Targets, saved-distance hints and all objective weights remain identical.
"""

from argparse import ArgumentParser
from contextlib import contextmanager, nullcontext
import json
from pathlib import Path

from scripts import audit_easy_dose_program as audit


@contextmanager
def allocation_budget(*, assignments, finalists):
    """Diagnostic computational budgets only; no objective/constraint changes."""
    original = audit.planner._allocate_visible_distance_ranges

    def allocate(*args, **kwargs):
        kwargs["assignments_per_total"] = assignments
        kwargs["allocation_candidate_limit"] = finalists
        return original(*args, **kwargs)

    try:
        audit.planner._allocate_visible_distance_ranges = allocate
        yield
    finally:
        audit.planner._allocate_visible_distance_ranges = original


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--search-report", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--probe-allocation-budgets", action="store_true")
    args = parser.parse_args()
    if args.output_json.exists():
        parser.error("Choose a fresh output path")
    source = json.loads(args.search_report.read_text())
    case = audit.load_case(args.database)
    assert case["states"][0].as_of.isoformat() == source["snapshot_as_of"]
    assert [list(x) for x in case["targets"]] == source["targets"]
    dates = [state.as_of.date() for state in case["states"]]
    trajectory = audit.dose_trajectory(case["profile"], dates, case["targets"][:21],
        case["evidence"].capacity_reference_miles, mode="history_trajectory",
        fixed_fraction=case["fixed_fraction"])
    priors = audit.eligible_prior_prescriptions(case["prior"], dates,
                                               excluded_offsets=set(range(21)) - case["allowed"])
    midpoints = {day.date: audit.planner._midpoint(day.recommendation)
                 for day in (case["prior"].planning_days or case["prior"].days)
                 if day.recommendation and day.recommendation.distance_range_miles
                 and day.recommendation.workout_type != audit.WorkoutType.REST}
    calendars = {tuple(case["seed"]), (1, 3, 5, 6, 10, 11, 12, 14, 17, 19)}
    for variant in source["variants"]:
        diagnostics = variant["normal_trace"]["diagnostics"]
        for key in ("winning_candidate", "unconstrained_winning_candidate", "near_term_stable_candidate"):
            if diagnostics.get(key):
                calendars.add(tuple(diagnostics[key]["offsets"]))
    report = {"scope": __doc__, "search_report": str(args.search_report), "variants": []}
    for mode in ("current", "history_with_saved_options", "history_without_saved_options"):
        policy = nullcontext() if mode == "current" else audit.history_allocator(
            case["profile"], trajectory, audit.planner._established_easy_midpoint_floor(case["states"][0]),
            prior_prescriptions=priors if mode == "history_with_saved_options" else {})
        with policy:
            rows = audit.evaluate_pool(case, sorted(calendars), prior_distance_midpoints_by_date=midpoints)
        report["variants"].append({"variant": mode, "candidates": rows})
        print(mode, [(x["offsets"], round(x["planned_midpoint_miles"], 2), round(x["score"], 3)) for x in rows], flush=True)
    if args.probe_allocation_budgets:
        # Two distinct calendars, not just the calendar revealing the issue.
        probes = sorted({tuple(v["normal_trace"]["offsets"]) for v in source["variants"]})
        report["allocation_budget_probes"] = []
        for saved in (False, True):
            for assignments, finalists in ((3, 16), (8, 4), (8, 16)):
                with audit.history_allocator(case["profile"], trajectory,
                        audit.planner._established_easy_midpoint_floor(case["states"][0]),
                        prior_prescriptions=priors if saved else {}):
                    with allocation_budget(assignments=assignments, finalists=finalists):
                        rows = audit.evaluate_pool(case, probes, prior_distance_midpoints_by_date=midpoints)
                report["allocation_budget_probes"].append({"saved_options": saved,
                    "assignments_per_total": assignments, "finalist_limit": finalists,
                    "candidates": rows})
                print("budget", saved, assignments, finalists,
                      [(x["offsets"], round(x["score"], 3)) for x in rows], flush=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("x") as stream:
        json.dump(report, stream, indent=2, default=lambda x: x.isoformat())
        stream.write("\n")


if __name__ == "__main__":
    main()
