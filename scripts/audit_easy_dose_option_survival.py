"""Paired allocations at IDENTICAL reconciliation states, with optional doses.

Both choices traverse the real allocator and its recovery constraints. The
alternate is scored but never fed back into the enclosing reconciliation loop.
This separates one-pass pruning effects from changed downstream projected load.
"""

from argparse import ArgumentParser
import json
from pathlib import Path

from scripts import audit_easy_dose_program as audit


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        parser.error("Choose a fresh output path")
    case = audit.load_case(args.database)
    planner = audit.planner
    dates = [x.as_of.date() for x in case["states"]]
    trajectory = audit.dose_trajectory(case["profile"], dates, case["targets"][:21],
        case["evidence"].capacity_reference_miles, mode="history_trajectory", fixed_fraction=case["fixed_fraction"])
    priors = audit.eligible_prior_prescriptions(case["prior"], dates,
                                               excluded_offsets=set(range(21)) - case["allowed"])
    midpoints = {d.date: planner._midpoint(d.recommendation)
                 for d in case["prior"].planning_days if d.recommendation and d.recommendation.distance_range_miles}
    original = planner._allocate_visible_distance_ranges
    no_saved_options = planner._easy_allocation_options
    comparisons = []

    def summary(days, selector):
        return {"score": selector(days), "sessions": [
            {"date": d.date, "role": d.day_role, "range": d.recommendation.distance_range_miles}
            for d in days if d.recommendation and d.recommendation.distance_range_miles]}

    def paired(*args, **kwargs):
        selected = original(*args, **kwargs)
        selector = kwargs.get("allocation_selector")
        if selector is not None:
            saved_options = planner._easy_allocation_options
            try:
                planner._easy_allocation_options = no_saved_options
                alternative = original(*args, **kwargs)
            finally:
                planner._easy_allocation_options = saved_options
            wider = original(*args, **{**kwargs, "allocation_candidate_limit": 16})
            comparisons.append({"preserve_projected_recovery_caps": kwargs.get("preserve_projected_recovery_caps", False),
                "saved_options": summary(selected, selector),
                "without_saved_options": summary(alternative, selector),
                "saved_options_16_finalists": summary(wider, selector)})
        return selected

    try:
        with audit.history_allocator(case["profile"], trajectory, planner._established_easy_midpoint_floor(case["states"][0]),
                                     prior_prescriptions=priors):
            planner._allocate_visible_distance_ranges = paired
            rows = audit.evaluate_pool(case, [(1, 2, 4, 6, 8, 10, 12, 13, 17, 18, 20)],
                                       prior_distance_midpoints_by_date=midpoints)
    finally:
        planner._allocate_visible_distance_ranges = original
    report = {"scope": __doc__, "paired_allocations": comparisons, "unchanged_enclosing_result": rows}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("x") as stream:
        json.dump(report, stream, indent=2, default=lambda x: x.isoformat())
        stream.write("\n")
    for i, row in enumerate(comparisons):
        print(i, {k: v["score"] for k, v in row.items() if isinstance(v, dict)})


if __name__ == "__main__":
    main()
