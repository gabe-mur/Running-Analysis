"""Freeze first two synthetic decisions and isolate calendar/dose allocation.

No personal data, weight changes or deployment. Fixed-calendar builds retain
the production allocator/reconciliation and actual prototype adapter. Forcing
an old dose is a diagnostic restriction, not a suggested scheduling policy.
"""

from argparse import ArgumentParser
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from inspect import signature
import json
from pathlib import Path

import yaml

from scripts.replay_easy_dose_program import replay_policy
from scripts.replay_synthetic_easy_dose import synthetic_history
from run_analysis import adherence_projection as projection
from run_analysis import weekly_schedule as planner


def sessions(plan):
    return [{"date": day.date.isoformat(), "at": day.recommendation.planned_for.isoformat(),
             "type": day.recommendation.workout_type.value, "range": day.recommendation.distance_range_miles}
            for day in plan.planning_days if day.recommendation
            and day.recommendation.workout_type != projection.WorkoutType.REST]


class CapturedEnough(Exception):
    pass


def capture_openings(config, start):
    bind = signature(planner.build_weekly_schedule)
    captures = []
    state, history = synthetic_history(start)
    with replay_policy("history_trajectory"):
        original = projection.build_weekly_schedule

        def build(*args, **kwargs):
            inputs = bind.bind(*args, **kwargs)
            inputs.apply_defaults()
            values = inputs.arguments
            targets = tuple(values["expected_target_projector"]([], 49))
            values["expected_target_projector"] = lambda _candidate, n: targets[:n]
            result = original(**values)
            captures.append({"inputs": values, "targets": targets, "plan": result})
            print(f"Captured {len(captures)}: {values['daily_states'][0].as_of}", flush=True)
            if len(captures) == 2:
                raise CapturedEnough
            return result

        try:
            projection.build_weekly_schedule = build
            projection.simulate_adherence(state, history, config, start, simulation_days=2)
        except CapturedEnough:
            pass
        finally:
            projection.build_weekly_schedule = original
    assert len(captures) == 2
    return captures


@contextmanager
def fixed_calendar(offsets, *, role_overrides, probe_date, old_range, force_old_dose=False):
    original_dates = planner.adaptive_run_day_offsets
    original_allocate = planner._allocate_visible_distance_ranges
    original_options = planner._easy_allocation_options
    original_joint = planner._joint_candidate_program_cost
    passes = []
    latest_joint = {}

    def joint(*args, **kwargs):
        details = kwargs.get("diagnostic_details")
        if details is None:
            details = {}
        result = original_joint(*args, **{**kwargs, "diagnostic_details": details})
        latest_joint.clear()
        latest_joint.update({"costs": result, "details": details})
        return result

    def dates(*args, **kwargs):
        kwargs["_planner_diagnostics"].winning_role_overrides = role_overrides
        return list(offsets)

    def options(*args, **kwargs):
        values, ranges = original_options(*args, **kwargs)
        if force_old_dose and kwargs["plan_date"] == probe_date:
            midpoint = sum(old_range) / 2
            if midpoint not in values:
                raise ValueError("Old midpoint is not an allowed option at this state")
            return [midpoint], {midpoint: old_range}
        return values, ranges

    def allocate(*args, **kwargs):
        selector = kwargs.get("allocation_selector")
        row = {"reconciled": kwargs.get("preserve_projected_recovery_caps", False), "finalists": []}

        def select(candidate):
            cost = selector(candidate)
            target = next(d for d in candidate if d.date == probe_date)
            row["finalists"].append({"score": cost, "range": target.recommendation.distance_range_miles,
                "miles": sum(planner._midpoint(d.recommendation) for d in candidate if d.recommendation)})
            return cost

        result = original_allocate(*args, **{**kwargs, "allocation_selector": select if selector else None})
        if selector:
            row["selected_score"] = selector(result)
            row["selected_joint"] = dict(latest_joint)
            target_index = next(i for i, d in enumerate(result) if d.date == probe_date)
            day = result[target_index]
            row["selected_range"] = day.recommendation.distance_range_miles
            # Same-state objective probe, not a reoptimized whole program.
            reduced = list(result)
            reduced[target_index] = day.model_copy(update={"recommendation": day.recommendation.model_copy(
                update={"distance_range_miles": old_range})})
            row["old_range_only_score"] = selector(reduced)
            row["old_range_only_joint"] = dict(latest_joint)
            passes.append(row)
        return result

    try:
        planner.adaptive_run_day_offsets = dates
        planner._allocate_visible_distance_ranges = allocate
        planner._easy_allocation_options = options
        planner._joint_candidate_program_cost = joint
        yield passes
    finally:
        planner.adaptive_run_day_offsets = original_dates
        planner._allocate_visible_distance_ranges = original_allocate
        planner._easy_allocation_options = original_options
        planner._joint_candidate_program_cost = original_joint


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        parser.error("Choose a fresh output path")
    config = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.example.yaml").read_text())
    start = datetime(2026, 9, 22, tzinfo=timezone.utc)
    captures = capture_openings(config, start)
    old, new = captures
    probe_date = (start + timedelta(days=2)).date()
    old_range = next(d.recommendation.distance_range_miles for d in old["plan"].planning_days if d.date == probe_date)
    rows = []
    for label, context, calendar, force in (
        ("opening_control", old, old["plan"], False),
        ("refresh_control", new, new["plan"], False),
        ("refresh_old_calendar", new, old["plan"], False),
        ("refresh_old_dose", new, new["plan"], True),
    ):
        origin = context["inputs"]["daily_states"][0].as_of.date()
        offsets = [(d.date - origin).days for d in calendar.planning_days if d.recommendation
                   and d.recommendation.workout_type != projection.WorkoutType.REST and origin <= d.date < origin + timedelta(days=21)]
        # Retain the actual selected near-term workout purposes for each control.
        roles = {(d.date - origin).days: "long" if d.recommendation.workout_type == projection.WorkoutType.LONG else
                 "quality" if d.recommendation.workout_type in planner.QUALITY_WORKOUT_TYPES else "easy"
                 for d in calendar.planning_days if d.recommendation and (d.date - origin).days in offsets
                 and (d.date - origin).days < planner.NEAR_TERM_STABILITY_DECISION_OFFSETS}
        with replay_policy("history_trajectory"):
            # Bind the option probe INSIDE the adapter's lifetime by wrapping
            # its allocator call; replay_policy activates that adapter in build.
            original_build = planner.build_weekly_schedule
            with fixed_calendar(offsets, role_overrides=roles, probe_date=probe_date,
                                old_range=old_range, force_old_dose=force) as passes:
                result = original_build(**context["inputs"])
        if label.endswith("control"):
            assert sessions(result) == sessions(context["plan"]), "Fixed-calendar control changed materialization"
        if force:
            assert next(x["range"] for x in sessions(result) if x["date"] == str(probe_date)) == old_range
        rows.append({"name": label, "sessions": sessions(result), "allocation_passes": passes})
        print(label, next(x for x in sessions(result) if x["date"] == str(probe_date)), flush=True)
    report = {"scope": __doc__, "cases": [
        {"as_of": c["inputs"]["daily_states"][0].as_of.isoformat(), "targets": c["targets"],
         "states": [s.model_dump(mode="json") for s in c["inputs"]["daily_states"]],
         "options": [[s.model_dump(mode="json") for s in group] for group in c["inputs"]["daily_state_options"]],
         "plan": c["plan"].model_dump(mode="json")} for c in captures], "probes": rows}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("x") as stream:
        json.dump(report, stream, indent=2, default=lambda x: x.isoformat())
        stream.write("\n")


if __name__ == "__main__":
    main()
