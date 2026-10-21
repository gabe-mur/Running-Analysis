"""Conditional today/shorter/tomorrow audit; no production search/weight changes.

Reads a recorded-HR replay, uses only history available at the selected decision,
and reoptimizes doses/roles on a fixed or bounded common 21-day calendar pool.
Optional target sensitivity is explicit counterfactual demand, not changed
observed capacity. Not a global optimum, daily adherence replay, physiological
validation or live plan.
"""

from argparse import ArgumentParser
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timedelta
from itertools import combinations, product
from math import isfinite
import json
from pathlib import Path
from time import perf_counter

from scripts.audit_capped_calendar import SEARCH_SIGNATURE
from scripts.audit_easy_dose_program import capture_allocations
from scripts.calendar_neighbor_prototype import calendar_neighbors, calendar_neighbor_policy
from scripts.finalized_scoring_trial import finalized_scoring_policy
from scripts.replay_easy_dose_program import replay_policy
from scripts.replay_recorded_hr import inputs_for, write_report
from run_analysis import adherence_projection as projection
from run_analysis import weekly_schedule as planner
from run_analysis.aerobic_acceptance import assess_aerobic_exposure
from run_analysis.web.schemas import SessionDifficulty, WorkoutType


class UnavailableDose(ValueError):
    pass


def available_history(report, at):
    return [projection.ProjectionRun(
        start_time=datetime.fromisoformat(row["start_time"]),
        distance_miles=row["distance_miles"], moving_minutes=row["moving_minutes"],
        workout_type=WorkoutType(row["workout_type"]),
        difficulty=SessionDifficulty.model_validate(row["difficulty"]),
    ) for row in report["recorded_inputs"]
        if datetime.fromisoformat(row["available_at"]) <= at]


@contextmanager
def restrict_opening_dose(day, ordinary, mode):
    """Restrict existing options only; never inject a larger or smaller dose."""
    original = planner._easy_allocation_options

    def options(*args, **kwargs):
        values, ranges = original(*args, **kwargs)
        if mode == "free" or kwargs["plan_date"] != day:
            return values, ranges
        if not values:
            raise UnavailableDose("No permitted opening easy dose")
        center = min(values, key=lambda value: (abs(value - ordinary), value))
        if mode == "shorter":
            shorter = [value for value in values if value < center - 1e-9]
            if not shorter:
                raise UnavailableDose("No distinct permitted shorter dose")
            center = min(shorter)
        return [center], {center: ranges[center]} if center in ranges else {}

    try:
        planner._easy_allocation_options = options
        yield
    finally:
        planner._easy_allocation_options = original


def tail_calendars(tail, allowed, *, spread_references=False):
    """Exact bounded pool, not a beam: one edit, two removals, opening pair shifts.

    Opening today/tomorrow is held separately. Same tail options for all three
    branches; no branch gets extra candidate coverage or a changed target.
    """
    tail = tuple(sorted(tail))
    allowed = set(allowed)
    if not tail or len(set(tail)) != len(tail) or not set(tail) <= allowed:
        raise ValueError("Require a distinct, nonempty allowed tail")
    pool = {tail, *calendar_neighbors(tail, allowed)}
    if len(tail) > 2:
        pool.update(tuple(x for x in tail if x not in removed)
                    for removed in combinations(tail, 2))
    if len(tail) >= 2:
        for shifts in product((-1, 0, 1), repeat=2):
            candidate = (tail[0] + shifts[0], tail[1] + shifts[1], *tail[2:])
            if len(set(candidate)) == len(candidate) and set(candidate) <= allowed:
                pool.add(tuple(sorted(candidate)))
    if spread_references:
        # Counterfactual controls, not preferred calendar templates. Sample
        # sparse and dense distributions beyond the seed's local neighbors.
        for count, first, last in product(range(max(2, len(tail) - 2), len(tail) + 2),
                                           (2, 3), (max(allowed) - 1, max(allowed))):
            candidate = tuple(round(first + i * (last - first) / (count - 1)) for i in range(count))
            if len(set(candidate)) == count and set(candidate) <= allowed:
                pool.add(candidate)
    if len(pool) > 100:
        raise ValueError("Tail pool exceeds bounded 100-calendar budget")
    return sorted(pool, key=lambda x: (len(x), x))


def summarize(rows):
    """Only rank realized, ready, acceptance-passing programs; retain all rows."""
    valid = [row for row in rows if row.get("eligible", False)]
    def compact(row):
        return {key: row[key] for key in ("label", "offsets", "total_cost", "run_count",
            "planned_miles", "integrated_band_funded", "date_gaps_days", "seconds")}
    def best(candidates):
        return compact(min(candidates, key=lambda r: (r["total_cost"], r["offsets"]))) if candidates else None
    labels = sorted({row["label"] for row in rows})
    counts = sorted({row["run_count"] for row in valid})
    return dict(evaluated=len(rows), eligible=len(valid),
        candidate_seconds=sum(row["seconds"] for row in rows),
        best_by_branch={label: best([row for row in valid if row["label"] == label]) for label in labels},
        best_by_count={str(count): best([row for row in valid if row["run_count"] == count]) for count in counts},
        funded_by_count={str(count): best([row for row in valid if row["run_count"] == count
                                           and row["integrated_band_funded"]]) for count in counts})


def compare(inputs, tail, history, *, expand_tail=False, spread_references=False, unrestricted_opening=False):
    common = {key: inputs[key] for key in (
        "daily_states", "daily_state_options", "request", "config",
        "target_distance_range", "completed_miles_by_offset")}
    state = inputs["daily_states"][0]
    origin = state.as_of.date()
    ordinary = state.typical_easy_run_miles
    blocked = set(inputs.get("forced_rest_offsets") or ()) | set(inputs.get("completed_run_offsets") or ())
    tails = tail_calendars(tail, set(range(2, len(inputs["daily_states"]))) - blocked,
                          spread_references=spread_references) if expand_tail else [tuple(tail)]
    targets = inputs["expected_target_projector"]([], len(inputs["daily_states"]))
    integrated_low, integrated_high = (sum(pair[i] for pair in targets) / 7 for i in (0, 1))
    rows = []
    branches = (("ordinary_today", 0, "ordinary"), ("shorter_today", 0, "shorter"), ("tomorrow", 1, "ordinary"))
    if unrestricted_opening:
        branches += (("unrestricted_today", 0, "free"),)
    for (label, offset, mode), candidate_tail in product(branches, tails):
        offsets = tuple(sorted((offset, *candidate_tail)))
        details = {}
        stamp = perf_counter()
        if set(offsets) & blocked:
            rows.append(dict(label=label, offsets=offsets, unavailable="Blocked date", seconds=0.))
            continue
        try:
            with restrict_opening_dose(origin + timedelta(days=offset), ordinary, mode), capture_allocations() as captured:
                joint = planner._joint_candidate_program_cost(offsets, **common,
                    expected_target_projector=inputs["expected_target_projector"],
                    role_overrides_by_offset=None if mode == "free" else {offset: "easy"},
                    reconcile_allocated_prefix=True, diagnostic_details=details,
                    _collect_acceptance_facts=True)
        except UnavailableDose as error:
            rows.append(dict(label=label, offsets=offsets, unavailable=str(error), seconds=perf_counter() - stamp))
            continue
        days = captured["days"]
        coaching = details["finalized_coaching_cost"]
        checked_coaching, sessions = planner._finalized_program_coaching_cost(days, **{
            key: common[key] for key in common if key != "completed_miles_by_offset"})
        if abs(checked_coaching - coaching) > 1e-9:
            raise ValueError("Captured prescriptions do not match finalized coaching score")
        runs = [s for s in sessions if s.recommendation.workout_type != WorkoutType.REST]
        opening = next((s for s in runs if s.offset == offset), None)
        facts = details["aerobic_acceptance_facts"]
        acceptance = assess_aerobic_exposure(**facts, planning_mode=inputs["_acceptance_planning_mode"])
        branch_realized = opening is not None and (mode == "free" or opening.recommendation.workout_type == WorkoutType.EASY)
        not_ready = [s.offset for s in sessions if s.recommendation.readiness.value == "not_ready"]
        missing_dates = sorted(set(offsets) - {s.offset for s in runs})
        planned = [(s.recommendation.planned_for, planner._midpoint(s.recommendation)) for s in runs]
        times = sorted([r.start_time for r in history][-1:] + [t for t, _ in planned])
        rolling = []
        combined = [(r.start_time, r.distance_miles) for r in history] + planned
        for day in range(1, 22):
            end = state.as_of.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=day)
            window = [(t, miles) for t, miles in combined if end - timedelta(days=7) <= t < end]
            rolling.append(dict(end=end, miles=sum(m for _, m in window), runs=len(window)))
        rows.append(dict(label=label, offsets=offsets, seconds=perf_counter() - stamp,
            ordinary_reference_miles=ordinary,
            branch_realized=branch_realized, not_ready=not_ready, missing_dates=missing_dates,
            eligible=branch_realized and not not_ready and not missing_dates
                and not acceptance.rejected and acceptance.reason != "missing_context",
            planned_miles=details["planned_midpoint_miles"], integrated_target=(integrated_low, integrated_high),
            integrated_band_funded=integrated_low - .5 - 1e-9 <= details["planned_midpoint_miles"] <= integrated_high + .5 + 1e-9,
            total_cost=planner._program_selection_cost(joint, coaching, sum(planner.typical_easy_distance(state)) / 2),
            joint_cost=joint, coaching_cost=coaching, details=details,
            acceptance=dict(rejected=acceptance.rejected, reason=acceptance.reason, evidence=acceptance.evidence),
            run_count=len(runs), date_gaps_days=[(b.date() - a.date()).days for a, b in zip(times, times[1:])],
            rolling7=rolling, sessions=[dict(offset=s.offset, at=s.recommendation.planned_for,
                type=s.recommendation.workout_type.value, distance_range=s.recommendation.distance_range_miles,
                readiness=s.recommendation.readiness.value, residual=s.state.recovery_residual_load,
                load_ratio=planner.effective_load_ratio(s.state.recent_load)) for s in sessions]))
        if expand_tail and len(rows) % 30 == 0:
            print(f"Scored {len(rows)}/{len(tails) * len(branches)} conditional programs", flush=True)
    return rows


class ComparisonComplete(Exception):
    pass


def run_comparison(report, decision_index, *, expand_tail=False, spread_references=False, target_scale=1., normal_search=False, unrestricted_opening=False):
    if spread_references and not expand_tail:
        raise ValueError("Spread references require expanded tail comparison")
    if unrestricted_opening and expand_tail:
        raise ValueError("Unrestricted opening probe is a fixed-calendar diagnostic")
    if not isfinite(target_scale) or not .5 <= target_scale <= 1.5:
        raise ValueError("Target sensitivity must be finite and between 0.5 and 1.5")
    decision = report["decisions"][decision_index]
    at = datetime.fromisoformat(decision["at"])
    history = sorted(available_history(report, at), key=lambda r: r.start_time)
    inputs = inputs_for(history, at, report["config"], prior=None)
    origin = inputs["daily_states"][0].as_of.date()
    calendar = sorted((datetime.fromisoformat(r["planned_for"]).date() - origin).days for r in decision["plan"])
    if not calendar or calendar[0] != 1:
        raise ValueError("First probe requires an existing first run tomorrow and a common tail after tomorrow")
    if decision["plan"][0]["workout_type"] != "easy":
        raise ValueError("First probe compares an easy opportunity, not replacing a key session")
    unscaled_targets = tuple(inputs["expected_target_projector"]([], 21 + int(planner.SOFT_CONTINUATION_DAYS)))
    # Explicit diagnostic demand intervention, not a claim of changed athlete
    # capacity or an appropriate training target. Apply to all objective inputs.
    targets = tuple(tuple(value * target_scale for value in pair) for pair in unscaled_targets)
    inputs["target_distance_range"] = tuple(value * target_scale for value in inputs["target_distance_range"])
    inputs["expected_target_projector"] = lambda _candidate, count: targets[:count]
    normal_inputs = deepcopy(inputs) if normal_search else None
    original = planner.adaptive_run_day_offsets
    result = {}

    def inspect(*args, **kwargs):
        bound = SEARCH_SIGNATURE.bind(*args, **kwargs)
        bound.apply_defaults()
        result["comparisons"] = compare(bound.arguments, calendar[1:], history,
            expand_tail=expand_tail, spread_references=spread_references, unrestricted_opening=unrestricted_opening)
        result["summary"] = summarize(result["comparisons"])
        raise ComparisonComplete()

    try:
        planner.adaptive_run_day_offsets = inspect
        with finalized_scoring_policy(), replay_policy("history_trajectory"):
            try:
                planner.build_weekly_schedule(**inputs)
            except ComparisonComplete:
                pass
    finally:
        planner.adaptive_run_day_offsets = original
    if normal_search:
        # Same frozen evidence, target and empty prior; opening branch/dose
        # restrictions are absent. Search budget is the existing prototype's.
        stamp = perf_counter()
        with finalized_scoring_policy(), calendar_neighbor_policy(max_passes=4, enforce_acceptance=True) as traces, replay_policy("history_trajectory"):
            plan = planner.build_weekly_schedule(**normal_inputs)
        winner = traces[0]["winner"]
        published = [f"{i}:{day.recommendation.workout_type.value}:{planner._midpoint(day.recommendation):.3f}"
            for i, day in enumerate(plan.planning_days)
            if day.recommendation and day.recommendation.workout_type != WorkoutType.REST]
        scored = [":".join(part for i, part in enumerate(line.split(":")) if i != 1)
                  for line in winner["allocation_trace"]]
        if published != scored:
            raise ValueError("Normal search scored/published programs differ")
        result["normal_search"] = dict(seconds=perf_counter() - stamp, trace=traces[0],
            scored_publication_match=True, sessions=[dict(date=day.date,
                type=day.recommendation.workout_type, distance_range=day.recommendation.distance_range_miles,
                readiness=day.recommendation.readiness) for day in plan.planning_days
                if day.recommendation and day.recommendation.workout_type != WorkoutType.REST])
    return dict(scope=__doc__, expand_tail=expand_tail, spread_references=spread_references,
        unrestricted_opening=unrestricted_opening, at=at, known_runs=len(history),
        known_hr_runs=sum(r.difficulty.zone_load is not None for r in history),
        prior_schedule=None, target_scale=target_scale, unscaled_targets=unscaled_targets,
        targets=targets, tail_offsets=calendar[1:], **result)


def main():
    from scripts.opportunity_search_repair import opportunity_search_policy
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--recorded-replay", type=Path, required=True)
    parser.add_argument("--decision", type=int, default=3)
    parser.add_argument("--expand-tail", action="store_true", help="Bounded common downstream date pool; no production search change")
    parser.add_argument("--spread-references", action="store_true", help="Add matched-count distributed calendars as counterfactual controls, not scheduling templates")
    parser.add_argument("--target-scale", type=float, default=1., help="Diagnostic target-demand sensitivity only; observed history/capacity unchanged")
    parser.add_argument("--normal-search", action="store_true", help="Also run the current four-pass prototype on identical inputs, without conditional opening restrictions")
    parser.add_argument("--opportunity-repair", action="store_true", help="Bounded timing, opening calendar and dose coverage repair")
    parser.add_argument("--unrestricted-opening", action="store_true", help="Also score today's calendar without opening dose or role restrictions")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        parser.error("Choose a fresh output path")
    report = json.loads(args.recorded_replay.read_text())
    with opportunity_search_policy(args.opportunity_repair):
        result = run_comparison(report, args.decision, expand_tail=args.expand_tail,
            spread_references=args.spread_references, target_scale=args.target_scale,
            normal_search=args.normal_search, unrestricted_opening=args.unrestricted_opening)
    result["opportunity_search_repair"] = args.opportunity_repair
    result["source"] = str(args.recorded_replay)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    write_report(args.output_json, result)
    if args.expand_tail:
        print(json.dumps(result["summary"], indent=2), flush=True)
        return
    for row in result["comparisons"]:
        print(row["label"], row.get("unavailable") or f"branch_realized={row['branch_realized']}, score={row['total_cost']:.4f}, runs={row['run_count']}, miles={row['details']['planned_midpoint_miles']:.2f}", flush=True)


if __name__ == "__main__":
    main()
