"""Read-only tracing of one known calendar and allocation coverage gap.

No beam/DP budget, objective, recovery rule or chosen production program changes.
Same-state shorter allocations are scored but never fed into reconciliation.
"""

from argparse import ArgumentParser
from contextlib import contextmanager
from datetime import datetime
from inspect import getsourcelines, signature
import json
from pathlib import Path
import sys
from time import perf_counter
from types import CodeType

from scripts.audit_easy_dose_search import trace_search, search_stage, plan_summary
from scripts.audit_capped_calendar import SEARCH_SIGNATURE
from scripts.audit_training_opportunities import available_history, restrict_opening_dose, run_comparison
from scripts.calendar_neighbor_prototype import calendar_neighbor_policy
from scripts.finalized_scoring_trial import finalized_scoring_policy
from scripts.replay_easy_dose_program import replay_policy
from scripts.replay_recorded_hr import inputs_for, write_report
from run_analysis import weekly_schedule as planner


TARGET = (0, 2, 4, 6, 9, 12, 13, 17, 18, 20)


class PrefixCaptured(Exception):
    pass


def source_line(function, marker):
    lines, start = getsourcelines(function)
    found = [start + i for i, line in enumerate(lines) if line.strip() == marker]
    if len(found) != 1:
        raise ValueError(f"Trace marker changed or ambiguous: {marker}")
    return found[0]


def nested_code(function, name):
    return next(x for x in function.__code__.co_consts if isinstance(x, CodeType) and x.co_name == name)


@contextmanager
def trace_calendar_prefix(target, *, stop_after_beam=False):
    original = planner._adaptive_run_day_offsets_for_frequency
    binding = signature(original)
    retain = nested_code(original, "retain_diverse")
    anchors_line = source_line(original, "cadence_anchors = _cadence_anchor_candidates(")
    scoring_line = source_line(original, "scored_candidates: list[tuple[float, tuple[int, ...]]] = []")
    rows = []

    def wrapper(*args, **kwargs):
        values = binding.bind(*args, **kwargs)
        values.apply_defaults()
        if values.arguments["horizon_run_count"] != len(target):
            return original(*args, **kwargs)
        previous = sys.gettrace()
        record = dict(prefixes=[])
        rows.append(record)

        def local(frame, event, arg):
            if frame.f_code is retain and event == "return":
                pool = frame.f_locals["values"]
                length = len(pool[0][1]) if pool else 0
                prefix = target[:length]
                ranked = sorted(pool, key=lambda x: (x[0], x[1]))
                item = next((x for x in ranked if x[1] == prefix), None)
                record["prefixes"].append(dict(length=length, prefix=prefix,
                    generated=item is not None, retained=any(x[1] == prefix for x in arg),
                    rank=ranked.index(item) + 1 if item else None,
                    cost=item[0] if item else None, expanded=len(pool), kept=len(arg)))
            elif frame.f_code is original.__code__ and event == "line":
                if frame.f_lineno == anchors_line:
                    record["after_beam"] = target in frame.f_locals["candidate_offsets"]
                    record["weather_and_raw_load_costs"] = frame.f_locals["weather_costs"]
                    record["seed_offsets"] = frame.f_locals["seed_offsets"]
                    if stop_after_beam:
                        raise PrefixCaptured()
                elif frame.f_lineno == scoring_line:
                    record["after_anchors"] = target in frame.f_locals["candidate_offsets"]
                    record["candidate_count_before_full_scoring"] = len(frame.f_locals["candidate_offsets"])
                    # Do not trace the expensive scorer; method wrappers below
                    # capture shortlisting/final scoring with negligible overhead.
                    sys.settrace(previous)
            return local

        def dispatch(frame, event, arg):
            return local if frame.f_code in (original.__code__, retain) else None

        try:
            sys.settrace(dispatch)
            return original(*args, **kwargs)
        finally:
            sys.settrace(previous)

    try:
        planner._adaptive_run_day_offsets_for_frequency = wrapper
        yield rows
    finally:
        planner._adaptive_run_day_offsets_for_frequency = original


def program_signature(days):
    return [(i, d.recommendation.workout_type.value, d.recommendation.planned_for,
             d.recommendation.distance_range_miles) for i, d in enumerate(days)
            if d.recommendation and d.recommendation.workout_type != planner.WorkoutType.REST]


@contextmanager
def allocation_trace(original, target=None):
    """Capture exact materialized assignments and targeted DP pruning only."""
    solve = nested_code(original, "solve")
    materialize = nested_code(original, "materialize_solution")
    dp_line = source_line(original, "dp = next_dp")
    project_line = source_line(original, "allocated_ranges = ranges_for(assignment, retain_long)")
    score_line = source_line(original, "candidate_solution = (")
    result = dict(materialized=[], dp_steps=[], projection_reached=False, proxy_evaluated=False,
                  retained_by_proxy=False)
    previous = sys.gettrace()

    def local(frame, event, arg):
        values = frame.f_locals
        if frame.f_code is materialize and event == "return":
            result["materialized"].append(dict(signature=program_signature(arg),
                proxy_score=values["selected"][0], assignment=values["selected"][1]))
        if frame.f_code is solve and target is not None:
            if event == "line" and frame.f_lineno == dp_line:
                primary = values["primary_long"]
                if primary is not None and values["long_option"] == target.get(primary["index"]):
                    current = values["record"]["index"]
                    indices = [r["index"] for r in values["scenario_records"]]
                    keys = [primary["index"], *indices[:indices.index(current) + 1]]
                    prefix = {key: target[key] for key in keys}
                    units = sum(planner._distance_dp_units(value) for value in prefix.values())
                    bucket = values["next_dp"].get(units, [])
                    prefix_cost = sum((target[key] - values["ideals"][key]) ** 2 for key in keys)
                    result["dp_steps"].append(dict(index=current, prefix=prefix,
                        option_permitted=target[current] in values["options"],
                        retained=any(dict(x[2]) == prefix for x in bucket),
                        ideals={key: values["ideals"][key] for key in keys},
                        weights={r["index"]: r["weight"] for r in values["records"] if r["index"] in keys},
                        target_balance_cost=prefix_cost,
                        kept=[dict(cost=x[1], assignment=x[2]) for x in bucket]))
            elif event == "line" and frame.f_lineno in (project_line, score_line):
                if values["assignment"] == target:
                    result["projection_reached" if frame.f_lineno == project_line else "proxy_evaluated"] = True
                    if frame.f_lineno == score_line:
                        result["target_proxy_score"] = values["score"]
            elif event == "return" and arg:
                result["retained_by_proxy"] |= any(x[1] == target for x in arg)
        return local

    def dispatch(frame, event, arg):
        return local if frame.f_code is materialize or (target is not None and frame.f_code is solve) else None

    try:
        sys.settrace(dispatch)
        yield result
    finally:
        sys.settrace(previous)


@contextmanager
def paired_allocations():
    original = planner._allocate_visible_distance_ranges
    joint = planner._joint_candidate_program_cost
    active = False
    rows = []

    def mark(*args, **kwargs):
        nonlocal active
        old = active
        if kwargs.get("_allocated_override") is None:
            active = tuple(args[0]) == TARGET and kwargs.get("role_overrides_by_offset") is None
        try:
            return joint(*args, **kwargs)
        finally:
            active = old

    def allocate(*args, **kwargs):
        selected = original(*args, **kwargs)
        selector = kwargs.get("allocation_selector")
        if not active or selector is None:
            return selected
        state = args[1][0]
        with restrict_opening_dose(state.as_of.date(), state.typical_easy_run_miles, "shorter"), allocation_trace(original) as alternate_trace:
            shorter = original(*args, **kwargs)
        exact = next(x for x in alternate_trace["materialized"] if x["signature"] == program_signature(shorter))
        with allocation_trace(original, exact["assignment"]) as control_trace:
            repeated = original(*args, **kwargs)
        assert program_signature(repeated) == program_signature(selected), "Tracing changed allocation"
        rows.append(dict(reconciled=kwargs.get("preserve_projected_recovery_caps", False),
            assignments_per_total=kwargs.get("assignments_per_total"), finalists=kwargs.get("allocation_candidate_limit"),
            selected=dict(score=selector(selected), signature=program_signature(selected)),
            shorter=dict(score=selector(shorter), **exact), normal_trace=control_trace))
        return selected  # never let the diagnostic alternative change the rollout

    try:
        planner._joint_candidate_program_cost = mark
        planner._allocate_visible_distance_ranges = allocate
        yield rows
    finally:
        planner._joint_candidate_program_cost = joint
        planner._allocate_visible_distance_ranges = original


def calendar_case(report):
    decision = report["decisions"][3]
    at = datetime.fromisoformat(decision["at"])
    history = sorted(available_history(report, at), key=lambda r: r.start_time)
    inputs = inputs_for(history, at, report["config"], prior=None)
    targets = tuple(inputs["expected_target_projector"]([], 21 + int(planner.SOFT_CONTINUATION_DAYS)))
    inputs["expected_target_projector"] = lambda _candidate, count: targets[:count]
    stamp = perf_counter()
    with finalized_scoring_policy(), trace_search() as trace:
        # The legacy tracing wrapper has *args/**kwargs; preserve the real
        # signature before the neighbor wrapper binds its inputs.
        planner.adaptive_run_day_offsets.__signature__ = SEARCH_SIGNATURE
        with calendar_neighbor_policy(max_passes=4, enforce_acceptance=True) as neighbors, trace_calendar_prefix(TARGET) as prefixes, replay_policy("history_trajectory"):
            plan = planner.build_weekly_schedule(**inputs)
    stage = search_stage(dict(offsets=TARGET), trace)
    return dict(seconds=perf_counter() - stamp, prefixes=prefixes, search_stage=stage,
        neighbor_target=[x for x in neighbors[0]["evaluated"] if tuple(x["offsets"]) == TARGET],
        normal_winner=neighbors[0]["winner"], rounds=neighbors[0]["rounds"], plan=plan_summary(plan))


def timing_probe(report):
    at = datetime.fromisoformat(report["decisions"][3]["at"])
    inputs = inputs_for(sorted(available_history(report, at), key=lambda r: r.start_time), at, report["config"], prior=None)
    rows = []
    for aligned in (False, True):
        states = list(inputs["daily_states"])
        if aligned:
            states[0] = inputs["daily_state_options"][0][0]
        with trace_calendar_prefix(TARGET, stop_after_beam=True) as trace:
            try:
                planner._adaptive_run_day_offsets_for_frequency(
                    states, inputs["request"], inputs["config"], inputs["target_run_count"],
                    inputs["target_distance_range"], daily_state_options=inputs["daily_state_options"],
                    horizon_run_count=len(TARGET), joint_program_scoring=True)
            except PrefixCaptured:
                pass
            else:
                raise ValueError("Expected to stop before full scoring")
        rows.append(dict(aligned_to_first_available_time=aligned, raw_time=states[0].as_of,
            raw_load_ratio=planner.effective_load_ratio(states[0].recent_load), trace=trace))
    return dict(scope="Cheap prefilter only: same history/target/beam size; replace opening midnight summary with existing first available timed state. No full-plan or release inference.", probes=rows)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--recorded-replay", type=Path, required=True)
    parser.add_argument("--mode", choices=("calendar", "dose", "timing"), required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--control", type=Path, help="Previous matched-normal report; verify instrumented winner parity")
    args = parser.parse_args()
    if args.output_json.exists():
        parser.error("Choose a fresh output")
    report = json.loads(args.recorded_replay.read_text())
    if args.mode == "calendar":
        result = calendar_case(report)
        if args.control:
            expected = json.loads(args.control.read_text())["normal_search"]["trace"]["winner"]
            for key in ("offsets", "total_cost", "allocation_trace"):
                assert result["normal_winner"][key] == expected[key], ("Instrumentation changed winner", key)
            result["control_parity"] = True
    elif args.mode == "timing":
        result = timing_probe(report)
    else:
        with paired_allocations() as pairs:
            result = run_comparison(report, 3, unrestricted_opening=True)
        result["paired_allocations"] = pairs
    result.setdefault("scope", __doc__)
    result.update(source=str(args.recorded_replay), mode=args.mode)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    write_report(args.output_json, result)
    print(f"Wrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
