"""Reproduce the seven-day control, freeze its final inputs, probe one extra pass.

Offline diagnosis only. The extra-pass build is never fed into the simulation.
Frozen typed inputs permit later single-decision probes without replaying a week.
"""

from argparse import ArgumentParser
from contextlib import contextmanager
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from inspect import signature
import json
from pathlib import Path
from time import perf_counter
from typing import get_type_hints

from pydantic import TypeAdapter

from scripts.audit_first_calendar_refresh import score_calendar
from scripts.calendar_neighbor_prototype import calendar_neighbor_policy
from scripts.issued_plan_execution import OpeningPlanExecution
from scripts.replay_easy_dose_program import replay_policy
from scripts.replay_synthetic_easy_dose import synthetic_history
from run_analysis import adherence_projection as projection
from run_analysis import weekly_schedule as planner
from run_analysis.projection_gate import ProjectionGate, ProjectionGateConfig, _json_ready


BUILD_SIGNATURE = signature(planner.build_weekly_schedule)
BUILD_TYPES = get_type_hints(planner.build_weekly_schedule)
SEARCH_SIGNATURE = signature(planner.adaptive_run_day_offsets)
COMPARISON_FIELDS = (
    "generated_at", "trigger", "planned_sessions", "target_trajectory",
    "capacity_reference_miles", "material_evidence_reasons", "recovery_surprise_units",
    "opening_recovery_residual_load", "opening_continuous_distance_miles",
    "ordinary_easy_midpoint_miles", "boundary_session_miles", "decision_start_date",
)


def freeze_inputs(inputs):
    count = len(inputs["daily_states"]) + int(planner.SOFT_CONTINUATION_DAYS)
    targets = inputs["expected_target_projector"]([], count)
    frozen = {
        "targets": [list(x) for x in targets],
        "inputs": {key: TypeAdapter(BUILD_TYPES[key]).dump_python(value, mode="json")
                   for key, value in inputs.items() if key != "expected_target_projector"},
    }
    prior = inputs.get("prior_schedule")
    if prior is not None:
        # API serialization deliberately excludes internal optimizer fields.
        frozen["inputs"]["prior_schedule"].update(
            planning_days=[x.model_dump(mode="json") for x in prior.planning_days],
            planner_diagnostics=prior.planner_diagnostics.model_dump(mode="json")
            if prior.planner_diagnostics else None)
    return frozen


def thaw_inputs(frozen):
    prior = frozen["inputs"].get("prior_schedule")
    if prior is not None and "planning_days" not in prior:
        raise ValueError("Snapshot lost internal prior planning_days; do not use as an identical-input replay")
    values = {key: TypeAdapter(BUILD_TYPES[key]).validate_python(value)
              for key, value in frozen["inputs"].items()}
    targets = tuple(tuple(x) for x in frozen["targets"])

    def projector(_candidate, count):
        if count > len(targets):
            raise ValueError("Frozen target horizon is too short")
        return targets[:count]

    values["expected_target_projector"] = projector
    return values


def restore_prior_from_control(frozen, control):
    """Repair only the legacy diagnostic snapshot, with explicit provenance.

    Search and the dose adapter read prior dates, types and distance ranges.
    These are all present in the preceding decision snapshot. Preserve complete
    visible recommendations; outside them restore only those consumed fields.
    This is not recovery of original prose, workout structures or diagnostics.
    """
    fixed = deepcopy(frozen)
    prior = fixed["inputs"]["prior_schedule"]
    if "planning_days" in prior:
        raise ValueError("Prior planning days already present; no restoration needed")
    previous = control["replans"][-2]
    start = date.fromisoformat(previous["decision_start_date"])
    visible = {x["date"]: x for x in prior["days"]}
    runs = {x["planned_for"][:10]: x for x in previous["planned_sessions"]}
    days = []
    for offset in range(len(previous["target_trajectory"])):
        key = (start + timedelta(days=offset)).isoformat()
        if key in visible:
            days.append(visible[key])
            continue
        run = runs.get(key)
        rec = planner.RecommendationResponse(
            generated_at=previous["generated_at"], fitness_state_as_of=previous["generated_at"],
            planned_for=run["planned_for"], workout_type=run["workout_type"],
            title="Diagnostic prior reconstructed from saved decision",
            distance_range_miles=(run["distance_low_miles"], run["distance_high_miles"]),
            confidence="moderate", readiness=run["readiness"],
        ) if run else None
        days.append(planner.WeeklyScheduleDay(date=key, recommendation=rec,
            day_role="easy_run" if run else "rest_day", rationale="Diagnostic prior fields only").model_dump(mode="json"))
    restored_runs = {x["date"]: (x["recommendation"]["workout_type"], tuple(x["recommendation"]["distance_range_miles"]))
                     for x in days if x.get("recommendation") and x["recommendation"]["workout_type"] != "rest"}
    expected = {key: (x["workout_type"], (x["distance_low_miles"], x["distance_high_miles"])) for key, x in runs.items()}
    assert restored_runs == expected
    prior["planning_days"] = days
    fixed["restored_prior_provenance"] = {
        "decision": previous["generated_at"],
        "fields": "Full search-consumed prior dates, types, exact ranges; visible records preserved",
        "limitation": "Not original far-horizon prose or workout structures; first-four-round parity still required",
    }
    return fixed


@contextmanager
def inspect_winner():
    """Capture exact scoring components while the history-dose adapter is active."""
    original = planner.adaptive_run_day_offsets
    rows = []

    def inspect(*args, **kwargs):
        bound = SEARCH_SIGNATURE.bind(*args, **kwargs)
        bound.apply_defaults()
        inputs = bound.arguments
        offsets = original(*args, **kwargs)
        diagnostics = inputs["_planner_diagnostics"]
        measured = {}
        names = ("_continuous_mileage_path_violation", "_soft_projected_program_rate",
                 "_hard_horizon_funding_violation", "_gate_aligned_hard_funding_violation")
        originals = {name: getattr(planner, name) for name in names}

        def instrument(name, function):
            def wrapped(*positional, **named):
                value = function(*positional, **named)
                measured[name] = value
                if name == "_hard_horizon_funding_violation":
                    measured["boundary_allowance_miles"] = named["boundary_allowance_miles"]
                    measured["remaining_target"] = positional[1]
                if name == "_soft_projected_program_rate":
                    measured["soft_average_target"] = positional[1]
                    measured["boundary_session"] = named.get("boundary_session")
                return value
            return wrapped

        try:
            for name, function in originals.items():
                setattr(planner, name, instrument(name, function))
            result = score_calendar(inputs, tuple(offsets),
                                    roles=diagnostics.winning_role_overrides or None)
        finally:
            for name, function in originals.items():
                setattr(planner, name, function)
        assert abs(result["total"] - diagnostics.winning_candidate.total_cost) < 1e-9
        rows.append({"components": measured, "score": result})
        return offsets

    try:
        planner.adaptive_run_day_offsets = inspect
        yield rows
    finally:
        planner.adaptive_run_day_offsets = original


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, default=lambda x: x.model_dump(mode="json")
                  if hasattr(x, "model_dump") else x.isoformat())
        stream.write("\n")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--control-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--frozen-input", type=Path)
    parser.add_argument("--restore-prior-from-control", action="store_true")
    parser.add_argument("--probe-passes", type=int, default=5)
    args = parser.parse_args()
    frozen_path = args.output_json.with_suffix(".inputs.json")
    if args.output_json.exists() or (not args.frozen_input and frozen_path.exists()):
        parser.error("Choose fresh output paths")
    if args.probe_passes < 4:
        parser.error("Probe must retain at least the four-pass control budget")
    control = json.loads(args.control_json.read_text())
    assert control["status"] == "passed" and control["daily_replan_count"] == 7
    assert control["synthetic_sandbox"]["execution_plan"] == "opening"
    reproduced = []
    frozen = json.loads(args.frozen_input.read_text()) if args.frozen_input else None
    if args.restore_prior_from_control:
        if frozen is None or frozen_path.exists():
            parser.error("Restoration requires legacy input and a new snapshot path")
        frozen = restore_prior_from_control(frozen, control)
        write_new(frozen_path, frozen)
    scoring = []
    if frozen is None:
        start = datetime(2026, 9, 22, tzinfo=timezone.utc)
        state, history = synthetic_history(start)
        contract = OpeningPlanExecution()
        gate = ProjectionGate(ProjectionGateConfig(**control["gate_policy"]), fail_fast=True)
        build_count = 0

        def observe(snapshot):
            gate.observe(snapshot)
            expected = control["replans"][len(reproduced)]
            actual = _json_ready(snapshot)
            for key in COMPARISON_FIELDS:
                assert actual[key] == expected[key], (snapshot.generated_at, key)
            reproduced.append(actual)
            print(snapshot.generated_at.isoformat(), snapshot.trigger,
                  f"{snapshot.planning_seconds:.2f}s identical control", flush=True)

        with replay_policy("history_trajectory"):
            original = projection.build_weekly_schedule

            def build(*positional, **kwargs):
                nonlocal build_count, frozen
                build_count += 1
                bound = BUILD_SIGNATURE.bind(*positional, **kwargs)
                bound.apply_defaults()
                last = build_count == len(control["replans"])
                if last:
                    frozen = freeze_inputs(bound.arguments)
                    write_new(frozen_path, frozen)
                with calendar_neighbor_policy(max_passes=4, enforce_acceptance=True):
                    if last:
                        with inspect_winner() as measured:
                            result = original(*positional, **kwargs)
                        scoring.extend(measured)
                    else:
                        result = original(*positional, **kwargs)
                return result

            try:
                projection.build_weekly_schedule = build
                weeks = projection.simulate_adherence(
                    state, history, control["synthetic_sandbox"]["config"], start,
                    simulation_days=7, weeks=1, replan_interval_days=1,
                    emit_post_upload_replans=True, replan_observer=observe,
                    execution_schedule_selector=contract.select, execution_observer=contract.observe)
                gate.finalize(weeks)
            finally:
                projection.build_weekly_schedule = original
                contract.close()
        assert len(reproduced) == len(control["replans"])
    inputs = thaw_inputs(frozen)
    with replay_policy("history_trajectory"), calendar_neighbor_policy(
            max_passes=args.probe_passes, enforce_acceptance=True) as traces, inspect_winner() as measured:
        stamp = perf_counter()
        plan = projection.build_weekly_schedule(**inputs)
        seconds = perf_counter() - stamp
    expected = control["synthetic_sandbox"]["calendar_neighbor_prototype"][-1]
    # Extra search begins from exactly the same normal-search result, and its
    # first four rounds must reproduce both lanes, not merely a similar calendar.
    actual = traces[0]
    comparison_errors = []
    if actual["original_selected"] != expected["original_selected"]:
        comparison_errors.append({"scope": "original_selected", "fields": [
            key for key in actual["original_selected"]
            if actual["original_selected"][key] != expected["original_selected"].get(key)]})
    for a, b in zip(actual["rounds"][:4], expected["rounds"]):
        for key in ("independent_offsets", "independent_cost", "stable_offsets", "stable_cost"):
            if a[key] != b[key]:
                comparison_errors.append({"scope": f"round_{a['pass']}", "field": key,
                                          "expected": b[key], "actual": a[key]})
    report = {"scope": __doc__, "control": str(args.control_json),
              "reproduced_decisions": len(reproduced), "first_four_rounds_identical": not comparison_errors,
              "comparison_errors": comparison_errors,
              "frozen_input": str(frozen_path if args.restore_prior_from_control else args.frozen_input or frozen_path),
              "control_scoring": scoring, "probe_scoring": measured,
              "probe_passes": args.probe_passes, "probe_seconds_including_inspection": seconds,
              "probe_trace": actual, "probe_plan": plan.model_dump(mode="json"),
              "probe_planning_days": [day.model_dump(mode="json") for day in plan.planning_days]}
    write_new(args.output_json, report)
    if comparison_errors:
        raise ValueError(f"Frozen replay differs from control; evidence saved to {args.output_json}: {comparison_errors}")
    print(f"Probe {seconds:.2f}s, {actual['stop_reason']}, score {actual['winner']['total_cost']:.6f}, "
          f"miles {actual['winner']['planned_midpoint_miles']:.2f}; wrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
