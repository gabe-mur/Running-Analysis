"""Bounded funding and key-session conflict comparisons, not a policy change.

Uses frozen recorded HR, repaired search-era dose policy and one target per panel.
Upper-easy probes restrict existing dose options; they do not enlarge bounds.
An absent funded candidate is not a proof of infeasibility. No live writes.
"""

from argparse import ArgumentParser
from contextlib import contextmanager
from datetime import datetime, timedelta
from itertools import combinations
import json
from pathlib import Path
from time import perf_counter

from scripts.audit_capped_calendar import SEARCH_SIGNATURE
from scripts.audit_easy_dose_program import capture_allocations
from scripts.audit_training_opportunities import available_history, restrict_opening_dose, UnavailableDose
from scripts.calendar_neighbor_prototype import calendar_neighbors
from scripts.finalized_scoring_trial import finalized_scoring_policy
from scripts.opportunity_search_repair import opportunity_search_policy
from scripts.replay_easy_dose_program import replay_policy
from scripts.replay_recorded_hr import inputs_for, write_report
from run_analysis import weekly_schedule as planner
from run_analysis.aerobic_acceptance import assess_aerobic_exposure


@contextmanager
def upper_easy_options(enabled):
    original = planner._easy_allocation_options
    def options(*args, **kwargs):
        values, ranges = original(*args, **kwargs)
        if not enabled or not values:
            return values, ranges
        upper = max(values)
        return [upper], {upper: ranges[upper]} if upper in ranges else {}
    try:
        planner._easy_allocation_options = options
        yield
    finally:
        planner._easy_allocation_options = original


def funding_pool(seed, allowed):
    seed = tuple(sorted(set(seed)))
    allowed = set(allowed)
    if not seed or not set(seed) <= allowed:
        raise ValueError("Require a nonempty allowed reference calendar")
    pool = {seed, *calendar_neighbors(seed, allowed)}
    if len(seed) > 2:
        pool.update(tuple(d for d in seed if d not in removed) for removed in combinations(seed, 2))
    # Explicit sparse/dense controls, not favored templates or calendar weeks.
    for count in range(6, 13):
        dates = tuple(round(i * (max(allowed) - min(allowed)) / (count - 1)) + min(allowed)
                      for i in range(count))
        if len(set(dates)) == count and set(dates) <= allowed:
            pool.add(dates)
    if len(pool) > 128:
        raise ValueError("Funding diagnostic exceeds its 128-calendar budget")
    return sorted(pool, key=lambda dates: (len(dates), dates))


def purpose(kind):
    return "quality" if kind in {"intervals", "tempo_threshold"} else "long" if kind == "long" else "easy" if kind == "easy" else kind


def evaluate(inputs, history, offsets, *, roles=None):
    if tuple(sorted(set(offsets))) != tuple(offsets):
        raise ValueError("Dates must be distinct and ordered")
    blocked = set(inputs.get("forced_rest_offsets") or ()) | set(inputs.get("completed_run_offsets") or ())
    if set(offsets) & blocked or not set(offsets) <= set(range(len(inputs["daily_states"]))):
        raise ValueError("Candidate contains unavailable dates")
    common = {k: inputs[k] for k in ("daily_states", "daily_state_options", "request", "config",
                                    "target_distance_range", "completed_miles_by_offset")}
    started = perf_counter()
    details = {}
    with capture_allocations() as captured:
        joint = planner._joint_candidate_program_cost(tuple(offsets), **common,
            expected_target_projector=inputs["expected_target_projector"],
            role_overrides_by_offset=roles, reconcile_allocated_prefix=True,
            diagnostic_details=details, _collect_acceptance_facts=True)
    checked, sessions = planner._finalized_program_coaching_cost(captured["days"],
        **{k: v for k, v in common.items() if k != "completed_miles_by_offset"})
    if abs(checked - details["finalized_coaching_cost"]) > 1e-9:
        raise ValueError("Scored and materialized prescriptions differ")
    runs = [s for s in sessions if s.recommendation.workout_type != planner.WorkoutType.REST]
    facts = details["aerobic_acceptance_facts"]
    acceptance = assess_aerobic_exposure(**facts, planning_mode=inputs["_acceptance_planning_mode"])
    types = {s.offset: purpose(s.recommendation.workout_type.value) for s in runs}
    missing = sorted(set(offsets) - set(types))
    mismatched = {offset: dict(requested=role, actual=types.get(offset))
                  for offset, role in (roles or {}).items() if types.get(offset) != role}
    not_ready = [s.offset for s in sessions if s.recommendation.readiness.value == "not_ready"]
    low, high = (sum(pair[i] for pair in facts["target_trajectory"]) / 7 for i in (0, 1))
    miles = details["planned_midpoint_miles"]
    times = sorted([r.start_time for r in history][-1:] + [s.recommendation.planned_for for s in runs])
    planned = [(s.recommendation.planned_for, planner._midpoint(s.recommendation)) for s in runs]
    origin = inputs["daily_states"][0].as_of.replace(hour=0, minute=0, second=0, microsecond=0)
    combined = [(r.start_time, r.distance_miles) for r in history] + planned
    windows = []
    for offset in range(1, 22):
        end = origin + timedelta(days=offset)
        in_window = [(t, m) for t, m in combined if end - timedelta(days=7) <= t < end]
        windows.append(dict(end=end, miles=sum(m for _, m in in_window), runs=len(in_window)))
    return dict(offsets=offsets, roles=roles or {}, seconds=perf_counter() - started,
        eligible=not missing and not mismatched and not not_ready and not acceptance.rejected
            and acceptance.reason != "missing_context",
        missing_dates=missing, mismatched_roles=mismatched, not_ready=not_ready,
        acceptance=dict(rejected=acceptance.rejected, reason=acceptance.reason, evidence=acceptance.evidence),
        run_count=len(runs), planned_miles=miles, integrated_target=(low, high),
        target_shortfall=max(0., low-miles), target_excess=max(0., miles-high),
        meets_target_floor=miles >= low - 1e-9,
        strictly_funded=low - 1e-9 <= miles <= high + 1e-9,
        funded_with_existing_tolerance=low - .5 - 1e-9 <= miles <= high + .5 + 1e-9,
        total_cost=planner._program_selection_cost(joint, checked,
            sum(planner.typical_easy_distance(inputs["daily_states"][0])) / 2),
        joint_cost=joint, coaching_cost=checked, details=details, rolling7=windows,
        seam_and_gaps_hours=[(b-a).total_seconds()/3600 for a,b in zip(times, times[1:])],
        longest_fraction=max((m for _,m in planned), default=0) / max(miles, 1e-9),
        sessions=[dict(offset=s.offset, at=s.recommendation.planned_for,
            type=s.recommendation.workout_type.value, distance_range=s.recommendation.distance_range_miles,
            residual=s.state.recovery_residual_load,
            load_ratio=planner.effective_load_ratio(s.state.recent_load),
            readiness=s.recommendation.readiness.value) for s in sessions])


def compact(row):
    return {k: row[k] for k in ("label", "offsets", "run_count", "planned_miles", "total_cost",
        "strictly_funded", "funded_with_existing_tolerance", "longest_fraction", "seam_and_gaps_hours")}


def funding_summary(rows):
    eligible = [r for r in rows if r["eligible"]]
    result = {}
    for count in sorted({r["run_count"] for r in rows}):
        pool = [r for r in eligible if r["run_count"] == count]
        best = lambda candidates: compact(min(candidates, key=lambda r: (r["total_cost"], r["offsets"]))) if candidates else None
        result[count] = dict(best=best(pool), funded=best([r for r in pool if r["strictly_funded"]]),
            near_funded=best([r for r in pool if r["funded_with_existing_tolerance"]]),
            maximum_returned_miles=max((r["planned_miles"] for r in pool), default=None))
    return result


def panels(inputs, history, seed):
    allowed = set(range(len(inputs["daily_states"]))) - set(inputs.get("forced_rest_offsets") or ()) - set(inputs.get("completed_run_offsets") or ())
    funding = []
    for offsets in funding_pool(seed, allowed):
        for upper in (False, True):
            with upper_easy_options(upper):
                row = evaluate(inputs, history, offsets)
            funding.append(dict(label="upper_easy" if upper else "free", **row))
    # Preserve existing purposes of the first selected quality/long sessions;
    # assess a nearby extra easy opportunity without silently displacing them.
    reference = next(r for r in funding if tuple(r["offsets"]) == seed and r["label"] == "free")
    funded = [r for r in funding if r["eligible"] and r["strictly_funded"]]
    matched = []
    matched_base = min(funded, key=lambda r: (r["run_count"], r["total_cost"], r["offsets"])) if funded else None
    if matched_base is not None:
        # Hold key-session dates/purposes while offering one extra easy day.
        # The regular grid controls need not have the same workout composition.
        base_offsets = tuple(matched_base["offsets"])
        key_roles = {s["offset"]: purpose(s["type"]) for s in matched_base["sessions"]
                     if purpose(s["type"]) in {"long", "quality"}}
        for extra in sorted(allowed - set(base_offsets)):
            for upper in (False, True):
                with upper_easy_options(upper):
                    row = evaluate(inputs, history, tuple(sorted((*base_offsets, extra))),
                        roles={**key_roles, extra: "easy"})
                matched.append(dict(label="upper_easy" if upper else "free", added_offset=extra, **row))
    conflicts = []
    for role in ("quality", "long"):
        key = next((s["offset"] for s in reference["sessions"] if purpose(s["type"]) == role), None)
        if key is None or not {key - 1, key + 1} <= allowed or {key - 1, key + 1} & set(seed):
            conflicts.append(dict(key_role=role, unavailable="No distinct adjacent slots around first key session"))
            continue
        # Keep the first quality and long roles in every branch of this panel.
        roles = {next(s["offset"] for s in reference["sessions"] if purpose(s["type"]) == r): r
                 for r in ("quality", "long") if any(purpose(s["type"]) == r for s in reference["sessions"])}
        rows = []
        for label, extra, mode in (("no_extra", None, "free"), ("before_ordinary", key-1, "ordinary"),
                                   ("before_shorter", key-1, "shorter"), ("after_ordinary", key+1, "ordinary")):
            offsets = seed if extra is None else tuple(sorted((*seed, extra)))
            overrides = roles if extra is None else {**roles, extra: "easy"}
            date = inputs["daily_states"][extra if extra is not None else 0].as_of.date()
            try:
                with restrict_opening_dose(date, inputs["daily_states"][0].typical_easy_run_miles, mode):
                    row = evaluate(inputs, history, offsets, roles=overrides)
                rows.append(dict(label=label, **row))
            except UnavailableDose as error:
                rows.append(dict(label=label, eligible=False, unavailable=str(error)))
        # Fixed-clock attribution: same permitted late option for both doses.
        # Do not mistake a changed downstream start time for a dose-only effect.
        timed_inputs = dict(inputs)
        timed_inputs["daily_state_options"] = list(inputs["daily_state_options"])
        latest = max(inputs["daily_state_options"][key], key=lambda state: state.as_of)
        timed_inputs["daily_state_options"][key] = [latest]
        controls = []
        for label, mode in (("before_ordinary", "ordinary"), ("before_shorter", "shorter")):
            extra = key - 1
            date = inputs["daily_states"][extra].as_of.date()
            try:
                with restrict_opening_dose(date, inputs["daily_states"][0].typical_easy_run_miles, mode):
                    row = evaluate(timed_inputs, history, tuple(sorted((*seed, extra))), roles={**roles, extra: "easy"})
                controls.append(dict(label=label, **row))
            except UnavailableDose as error:
                controls.append(dict(label=label, eligible=False, unavailable=str(error)))
        conflicts.append(dict(key_role=role, key_offset=key, rows=rows,
            common_key_time=latest.as_of, timing_controls=controls))
    return dict(funding=funding, funding_summary=funding_summary(funding), conflicts=conflicts,
                matched_funded_base=compact(matched_base) if matched_base else None,
                matched_extensions=matched, matched_extension_summary=funding_summary(matched),
                evaluated=len(funding) + len(matched) + sum(len(c.get("rows", [])) + len(c.get("timing_controls", [])) for c in conflicts))


class PanelsComplete(Exception):
    pass


def run(report, seed, *, decision_index=3, target_scale=1.):
    if target_scale not in (.85, 1.):
        raise ValueError("Predeclared demand panels are 0.85 and 1.0; not observed-fatigue changes")
    at = datetime.fromisoformat(report["decisions"][decision_index]["at"])
    history = sorted(available_history(report, at), key=lambda r: r.start_time)
    inputs = inputs_for(history, at, report["config"], prior=None)
    targets = tuple(tuple(x * target_scale for x in pair) for pair in
        inputs["expected_target_projector"]([], 21 + int(planner.SOFT_CONTINUATION_DAYS)))
    inputs["target_distance_range"] = tuple(x * target_scale for x in inputs["target_distance_range"])
    inputs["expected_target_projector"] = lambda _candidate, count: targets[:count]
    original = planner.adaptive_run_day_offsets
    output = {}
    def inspect(*args, **kwargs):
        bound = SEARCH_SIGNATURE.bind(*args, **kwargs)
        bound.apply_defaults()
        output.update(panels(bound.arguments, history, tuple(seed)))
        raise PanelsComplete()
    started = perf_counter()
    try:
        planner.adaptive_run_day_offsets = inspect
        with opportunity_search_policy(), finalized_scoring_policy(), replay_policy("history_trajectory"):
            try:
                planner.build_weekly_schedule(**inputs)
            except PanelsComplete:
                pass
    finally:
        planner.adaptive_run_day_offsets = original
    return dict(scope=__doc__, at=at, known_runs=len(history),
        known_hr_runs=sum(r.difficulty.zone_load is not None for r in history),
        target_scale=target_scale, targets=targets, prior_schedule=None,
        seconds=perf_counter()-started, **output)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--recorded-replay", type=Path, required=True)
    parser.add_argument("--reference-search", type=Path, required=True)
    parser.add_argument("--target-scale", type=float, default=1.)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        parser.error("Choose a new output")
    report = json.loads(args.recorded_replay.read_text())
    reference = json.loads(args.reference_search.read_text())
    seed = reference["normal_search"]["trace"]["winner"]["offsets"]
    result = run(report, seed, target_scale=args.target_scale)
    if args.target_scale == 1.:
        control = next(r for r in result["funding"] if tuple(r["offsets"]) == tuple(seed) and r["label"] == "free")
        expected = reference["normal_search"]["trace"]["winner"]
        if (abs(control["total_cost"] - expected["total_cost"]) > 1e-9
                or control["details"]["allocation_trace"] != expected["allocation_trace"]):
            raise ValueError("Frozen reference does not match rescored control")
        result["reference_parity"] = True
    result.update(source=str(args.recorded_replay), reference=str(args.reference_search))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    write_report(args.output_json, result)
    print("Within-band counts:", [count for count, row in result["funding_summary"].items() if row["funded"]])
    print(f"Scored {result['evaluated']} programs in {result['seconds']:.2f}s; {args.output_json}")


if __name__ == "__main__":
    main()
