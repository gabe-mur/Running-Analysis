"""Fixed-program parity and matched-input off/on final-scoring trial.

Both builds retain four neighbor passes and shared candidate acceptance.
The second receives the same frozen prior, not the first build's output.
No settings, weights, target accounting or live plan changes.
"""

from argparse import ArgumentParser
import json
from pathlib import Path
from time import perf_counter

from scripts.audit_capped_calendar import SEARCH_SIGNATURE, thaw_inputs, write_new
from scripts.audit_easy_dose_program import capture_allocations
from scripts.calendar_neighbor_prototype import calendar_neighbor_policy
from scripts.finalized_scoring_trial import finalized_scoring_policy
from scripts.replay_easy_dose_program import replay_policy
from run_analysis import weekly_schedule as planner


def fixed_program_parity(inputs, baseline):
    original = planner.adaptive_run_day_offsets
    rows = []

    def inspect(*positional, **kwargs):
        bound = SEARCH_SIGNATURE.bind(*positional, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        common = {key: values[key] for key in ("daily_states", "daily_state_options", "request", "config", "target_distance_range")}
        for reference in baseline["rows"]:
            scored = reference["score"]
            with capture_allocations() as allocated:
                planner._joint_candidate_program_cost(tuple(scored["offsets"]), **common,
                    completed_miles_by_offset=values["completed_miles_by_offset"],
                    prior_distance_midpoints_by_date=values["prior_distance_midpoints_by_date"],
                    expected_target_projector=values["expected_target_projector"],
                    role_overrides_by_offset={int(k): v for k, v in (scored["roles"] or {}).items()},
                    reconcile_allocated_prefix=True)
            cost, sessions = planner._finalized_program_coaching_cost(allocated["days"], **common)
            expected = reference["finalized_coaching_probe"]["finalized_coaching"]
            assert abs(cost - expected) < 1e-9, (reference["label"], cost, expected)
            actual = [(s.recommendation.planned_for.date().isoformat(), s.recommendation.workout_type.value,
                       s.recommendation.distance_range_miles) for s in sessions
                      if s.recommendation.workout_type != planner.WorkoutType.REST]
            assert actual == [(s["date"], s["type"], tuple(s["range"])) for s in scored["sessions"]]
            rows.append(dict(label=reference["label"], cost=cost, sessions_match=True))
        return list(baseline["rows"][0]["score"]["offsets"])

    try:
        planner.adaptive_run_day_offsets = inspect
        with finalized_scoring_policy(False), replay_policy("history_trajectory"):
            planner.build_weekly_schedule(**inputs)
    finally:
        planner.adaptive_run_day_offsets = original
    return rows


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-input", type=Path, required=True)
    parser.add_argument("--fixed-program-baseline", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        parser.error("Choose a new output path")
    frozen = json.loads(args.frozen_input.read_text())
    baseline = json.loads(args.fixed_program_baseline.read_text())
    fixed = fixed_program_parity(thaw_inputs(frozen), baseline)
    print("Three fixed programs: finalized coaching and prescriptions match prior independent probe", flush=True)
    rows = []
    for enabled in (False, True):
        inputs = thaw_inputs(frozen)
        stamp = perf_counter()
        with finalized_scoring_policy(enabled) as measurements, calendar_neighbor_policy(
                max_passes=4, enforce_acceptance=True) as traces, replay_policy("history_trajectory"):
            plan = planner.build_weekly_schedule(**inputs)
        seconds = perf_counter() - stamp
        trace = traces[0]
        if not enabled:
            assert abs(trace["winner"]["total_cost"] - baseline["rows"][0]["score"]["total"]) < 1e-9
            assert trace["winner"]["offsets"] == baseline["rows"][0]["score"]["offsets"]
        scored = trace["winner"]["allocation_trace"]
        published = [f"{i}:{day.day_role}:{day.recommendation.workout_type.value}:{planner._midpoint(day.recommendation):.3f}"
                     for i, day in enumerate(plan.planning_days)
                     if day.recommendation and day.recommendation.workout_type != planner.WorkoutType.REST]
        # Day-role labels may differ between build presentation and scoring;
        # compare date, type and dose, retaining the complete raw traces.
        key = lambda line: tuple(x for i, x in enumerate(line.split(":")) if i != 1)
        matches = [key(x) for x in scored] == [key(x) for x in published]
        rows.append(dict(enabled=enabled, seconds=seconds, measurements=measurements, trace=trace,
                         scored_publication_match=matches,
                         plan=plan.model_dump(mode="json"),
                         planning_days=[x.model_dump(mode="json") for x in plan.planning_days]))
        print(f"enabled={enabled} {seconds:.2f}s score={trace['winner']['total_cost']:.6f} "
              f"miles={trace['winner']['planned_midpoint_miles']:.2f} publication_match={matches}", flush=True)
    write_new(args.output_json, dict(scope=__doc__, frozen_input=str(args.frozen_input),
        fixed_program_parity=fixed, variants=rows,
        wall_delta_seconds=rows[1]["seconds"] - rows[0]["seconds"]))
    if not all(x["scored_publication_match"] for x in rows):
        raise ValueError("Scored versus published plan differs; evidence retained, do not continue")


if __name__ == "__main__":
    main()
