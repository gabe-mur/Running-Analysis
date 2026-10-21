"""Offline full-horizon allocation using history-led easy-dose bands.

Only the extracted easy-dose bounds/options are replaced, within this process. Existing
role generation, sequential recovery limits, allocation, target projection and
program scoring remain active. No defaults, settings or saved plans are changed.
"""

from argparse import ArgumentParser
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import date, datetime
from hashlib import sha256
from itertools import combinations
import json
from math import floor, isfinite
from pathlib import Path
import sqlite3
from time import perf_counter

from scripts.audit_dose_assumptions import full_calendar_pool
from scripts.audit_easy_dose_prototype import frozen_history
from scripts.easy_dose_prototype import HistoryProfile, Observation, generate_intents
import run_analysis.weekly_schedule as planner
from run_analysis.adherence_projection import ProjectionPlanSession, ProjectionReplan
from run_analysis.projection_gate import ProjectionGateConfig, evaluate_replan_regressions
from run_analysis.web.schemas import (
    FitnessState, RecommendationRequest, WeeklyScheduleDay,
    WeeklyScheduleResponse, WeeklyTargetEvidence, WorkoutType,
)


@dataclass(frozen=True)
class DoseTrajectory:
    dates: tuple[date, ...]
    upper_miles: tuple[float, ...]
    target_ranges: tuple[tuple[float, float], ...]
    capacity_reference_miles: float
    mode: str


def history_profile_from_state(state, config):
    """Use observed evidence identically in replay and cheap continuation."""
    return HistoryProfile(
        as_of=state.as_of,
        half_life_days=float(config.get('coaching', {}).get('capacity_retention_half_life_days', 84)),
        ordinary_planning_miles=state.typical_easy_run_miles,
        ordinary_actual_miles=None, support_planning_miles=None,
        ordinary_count=0, support_count=0, recovery_response_count=0,
        newest_observation=None, observations=(), ordinary_sources=(), exclusions=(),
    )


def dose_trajectory(profile, dates, target_ranges, capacity, *, mode, fixed_fraction):
    """Transfer candidate-independent program headroom, not shortfall or rest.

    This proportional transfer remains an experimental policy. It introduces no
    compounded growth per calendar week, per planned session or per refresh.
    The observed ordinary reference is never advanced by proposed work.
    """
    ordinary = profile.ordinary_planning_miles
    if ordinary is None or not isfinite(capacity) or capacity <= 0:
        raise ValueError("Established ordinary history and positive capacity required")
    if mode not in {"history_fixed", "history_trajectory"}:
        raise ValueError("Unknown dose trajectory")
    if len(dates) != len(target_ranges) or len(set(dates)) != len(dates):
        raise ValueError("One target range per distinct date required")
    if not isfinite(fixed_fraction) or fixed_fraction < 0:
        raise ValueError("Invalid fixed progression fraction")
    upper = []
    for low, high in target_ranges:
        if not (isfinite(low) and isfinite(high) and 0 <= low <= high):
            raise ValueError("Invalid target range")
        ratio = 1 + fixed_fraction if mode == "history_fixed" else max(1., (low + high) / 2 / capacity)
        upper.append(ordinary * ratio)
    return DoseTrajectory(tuple(dates), tuple(upper), tuple(target_ranges), capacity, mode)


@contextmanager
def history_allocator(profile, trajectory, useful_floor, *, prior_prescriptions=None):
    """Process-local adapter; restore even after exceptions. Not thread-safe."""
    original = planner._easy_allocation_bounds
    original_options = planner._easy_allocation_options
    prior_prescriptions = dict(prior_prescriptions or {})
    by_date = dict(zip(trajectory.dates, trajectory.upper_miles, strict=True))
    ordinary = profile.ordinary_planning_miles
    menus = {
        day: generate_intents(
            profile, useful_floor_miles=useful_floor,
            progression_allowance_miles=max(0, upper - ordinary),
            progression_source=trajectory.mode + ":candidate_independent_program_target",
        ) for day, upper in by_date.items()
    }
    measurements = {"calls": 0, "material_caution_fallbacks": 0, "unknown_date_fallbacks": 0,
                    "prior_candidates_available": 0, "prior_rejections": {}}

    def bounds(state, result, aerobic_maximum, gap, *, preserve_projected_recovery_caps):
        legacy = original(state, result, aerobic_maximum, gap,
                          preserve_projected_recovery_caps=preserve_projected_recovery_caps)
        measurements["calls"] += 1
        if planner._easy_has_material_dose_limit(state, result):
            measurements["material_caution_fallbacks"] += 1
            return legacy
        menu = menus.get(state.as_of.date())
        if menu is None:
            measurements["unknown_date_fallbacks"] += 1
            return legacy
        # The allocator already uses a quarter-mile grid. Round intended dose
        # endpoints to its nearest center, not down: 3.999 intent must not lose
        # the 4.0 option. This is a presentation/search convention, NOT rounding
        # away a material restriction or the independent aerobic maximum.
        centers = sorted({floor(x.midpoint_miles * 4 + .5) / 4 for x in menu.options})
        centers = [x for x in centers if x > 0 and x <= aerobic_maximum + 1e-9]
        if not centers:
            return legacy
        preferred = min(centers, key=lambda x: (abs(x - ordinary), x))
        return centers[0], centers[-1], preferred, legacy[3]

    def options(state, result, minimum, maximum, aerobic_maximum, *, plan_date):
        choices, ranges = original_options(state, result, minimum, maximum, aerobic_maximum,
                                            plan_date=plan_date)
        prior = prior_prescriptions.get(plan_date)
        if prior is None:
            return choices, ranges
        reason = None
        saved = prior.distance_range_miles
        if plan_date not in menus:
            reason = "outside_intent_horizon"
        elif prior.workout_type != WorkoutType.EASY or result.workout_type != WorkoutType.EASY:
            reason = "different_workout_type"
        elif saved is None or not all(isfinite(x) for x in saved) or not 0 < saved[0] <= saved[1]:
            reason = "invalid_saved_range"
        elif planner._easy_has_material_dose_limit(state, result):
            # Retain the existing restriction policy, not a special exemption
            # for having prescribed more mileage in the past.
            reason = "material_caution"
        elif sum(saved) / 2 > aerobic_maximum + 1e-9:
            reason = "independent_aerobic_maximum"
        if reason is not None:
            counts = measurements["prior_rejections"]
            counts[reason] = counts.get(reason, 0) + 1
            return choices, ranges
        midpoint = prior.planned_distance_miles
        measurements["prior_candidates_available"] += 1
        # Keep the exact center and range. Neither becomes evidence for the
        # other dates' ordinary/progression references or expands this grid.
        return sorted(set(choices) | {midpoint}), {**ranges, midpoint: tuple(saved)}

    try:
        planner._easy_allocation_bounds = bounds
        planner._easy_allocation_options = options
        yield measurements
    finally:
        planner._easy_allocation_bounds = original
        planner._easy_allocation_options = original_options


def eligible_prior_prescriptions(prior_schedule, dates, *, excluded_offsets=()):
    """Match absolute unconsumed dates; never carry yesterday's dose forward."""
    if prior_schedule is None:
        return {}
    excluded = set(excluded_offsets)
    allowed = {day for index, day in enumerate(dates) if index not in excluded}
    return {
        day.date: day.recommendation
        for day in (prior_schedule.planning_days or prior_schedule.days)
        if day.date in allowed and day.recommendation is not None
        and day.recommendation.workout_type == WorkoutType.EASY
        and day.recommendation.distance_range_miles is not None
    }


def opportunity_pool(seed, allowed, required, *, max_additions=3):
    """Exhaustive added-date subsets around one seed, plus opening shifts/drop.

    This is not exhaustive date search: seed dates outside the opening moves
    remain fixed. No day-of-week or maximum-consecutive-run rule is introduced.
    """
    if max_additions not in (0, 1, 2, 3):
        raise ValueError("Use 0–3 additional opportunities for this bounded audit")
    seed = tuple(sorted(seed))
    pool = set(full_calendar_pool(seed, allowed, required))
    unused = sorted(set(allowed) - set(seed))
    for count in range(1, max_additions + 1):
        pool.update(tuple(sorted((*seed, *extra))) for extra in combinations(unused, count))
    pool = sorted(x for x in pool if set(x).issubset(allowed) and set(required).issubset(x)
                  and len(x) <= len(seed) + max_additions)
    if len(pool) > 500:
        raise ValueError("Pool exceeds 500 calendars")
    return pool


def load_case(database):
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as db:
        payload = json.loads(db.execute("SELECT value_json FROM app_state WHERE key='weekly_planner_snapshot'").fetchone()[0])
        internal = json.loads(db.execute("SELECT value_json FROM app_state WHERE key='weekly_schedule_internal'").fetchone()[0])
    states = [FitnessState.model_validate(x) for x in payload["daily_states"]]
    options = [[FitnessState.model_validate(x) for x in group] for group in payload["daily_state_options"]]
    if len(states) != 21 or len(options) != 21:
        raise ValueError("This audit requires a complete 21-day snapshot")
    evidence = WeeklyTargetEvidence.model_validate(payload["target_evidence"])
    prior = WeeklyScheduleResponse.model_validate(payload["prior_schedule"]) if payload.get("prior_schedule") else None
    if prior and payload.get("prior_planning_days"):
        prior = prior.model_copy(update={"planning_days": [WeeklyScheduleDay.model_validate(x) for x in payload["prior_planning_days"]]})
    observed = [planner.PlanningActivity(
        start_time=datetime.fromisoformat(x["start_time"]), distance_miles=x["distance_miles"],
        moving_minutes=x.get("moving_minutes"), easy_minutes=x.get("easy_minutes"),
        baseline_eligible=x.get("baseline_eligible", True), target_distance_miles=x.get("target_distance_miles"),
    ) for x in payload["observed_planning_activities"]]
    projector = planner.make_expected_target_projector(
        observed, states[0].as_of.date(), payload["config"],
        pace_min_mile=payload["target_projection_pace_min_mile"], maximum_horizon_days=21,
        opening_target_range=tuple(payload["target_distance_range"]), opening_evidence=evidence,
        ordinary_easy_midpoint_miles=payload.get("ordinary_easy_midpoint_miles"), prior_schedule=prior,
    )
    # Freeze the same trajectory before comparing any candidate or dose policy.
    targets = tuple(projector([], 21 + int(planner.SOFT_CONTINUATION_DAYS)))
    completed = {int(k): sum(x["distance_miles"] for x in group)
                 for k, group in payload.get("completed_activities_by_offset", {}).items()}
    completed.update({int(k): v for k, v in payload.get("completed_miles_by_offset", {}).items()})
    allowed = set(range(21)) - set(payload.get("forced_rest_offsets", [])) - set(payload.get("completed_run_offsets", [])) - set(completed)
    lookup = {state.as_of.date().isoformat(): i for i, state in enumerate(states)}
    seed = tuple(sorted(lookup[x["date"]] for x in internal["days"] if x["date"] in lookup
                        and lookup[x["date"]] in allowed and x.get("recommendation")
                        and x["recommendation"]["workout_type"] != "rest"))
    goal = planner.configured_race_goal(payload["config"], on_date=states[0].as_of.date())
    required = {i for i in allowed if goal and states[i].as_of.date() == goal[1]}
    history = frozen_history(database)
    raw_profile = history["profile"]
    profile = HistoryProfile(**{**raw_profile, "observations": tuple(Observation(**x) for x in raw_profile["observations"])})
    return dict(payload=payload, states=states, options=options, evidence=evidence,
                prior=prior, targets=targets, completed=completed, allowed=allowed,
                required=required, seed=seed, profile=profile,
                fixed_fraction=history["transferred_progression_fraction"])


@contextmanager
def capture_allocations():
    original = planner._allocate_visible_distance_ranges
    captured = {}

    def allocate(*args, **kwargs):
        result = original(*args, **kwargs)
        captured["days"] = result
        return result

    try:
        planner._allocate_visible_distance_ranges = allocate
        yield captured
    finally:
        planner._allocate_visible_distance_ranges = original


def evaluate_pool(case, calendars, *, prior_distance_midpoints_by_date=None,
                  role_overrides_by_offset=None):
    states = case["states"]
    common = dict(
        daily_states=states, daily_state_options=case["options"],
        request=RecommendationRequest.model_validate(case["payload"]["request"]),
        config=case["payload"]["config"], target_distance_range=tuple(case["payload"]["target_distance_range"]),
        completed_miles_by_offset=case["completed"], prefix_cache={},
        recommendation_load_cache={}, projected_state_cache={},
    )
    ordinary = sum(planner.typical_easy_distance(states[0])) / 2
    targets = case["targets"]
    rows = []
    for number, offsets in enumerate(calendars, 1):
        details = {}
        with capture_allocations() as captured:
            joint = planner._joint_candidate_program_cost(
                offsets, **common, reconcile_allocated_prefix=True,
                expected_target_projector=lambda _candidate, n: targets[:n], diagnostic_details=details,
                prior_distance_midpoints_by_date=prior_distance_midpoints_by_date,
                role_overrides_by_offset=role_overrides_by_offset,
            )
        coaching = planner._adaptive_candidate_cost(offsets, **common, include_mileage_path=False,
                                                   include_recovery_interactions=False, include_cadence_pressure=False,
                                                   role_overrides_by_offset=role_overrides_by_offset)
        days = captured["days"]
        plan = tuple(ProjectionPlanSession(
            planned_for=day.recommendation.planned_for or day.planned_at,
            workout_type=day.recommendation.workout_type,
            midpoint_miles=planner._midpoint(day.recommendation),
            distance_low_miles=day.recommendation.distance_range_miles[0],
            distance_high_miles=day.recommendation.distance_range_miles[1],
            planning_role=day.recommendation.planning_role, readiness=day.recommendation.readiness.value,
        ) for day in days if day.recommendation and day.recommendation.workout_type != WorkoutType.REST
                     and day.recommendation.distance_range_miles)
        boundary = planner._recent_boundary_session(states[0])
        check = ProjectionReplan(
            generated_at=states[0].as_of, opening_load_ratio=None,
            target_low_miles=common["target_distance_range"][0], target_high_miles=common["target_distance_range"][1],
            planned_sessions=plan, committed_sessions=(), ordinary_easy_midpoint_miles=ordinary,
            boundary_session_miles=boundary[1] if boundary else None,
            target_trajectory=targets[:21], decision_start_date=states[0].as_of.date(),
        )
        # Partial static checks only: no peak-load field or upload/refresh pair
        # is manufactured here. Full ProjectionGate runs belong to daily replay.
        failures = evaluate_replan_regressions(check, ProjectionGateConfig())
        dates = sorted(x.planned_for.date() for x in plan)
        gaps = [(b - a).days for a, b in zip(dates, dates[1:])]
        rows.append({
            "offsets": offsets, "requested_run_count": len(offsets), "actual_run_count": len(plan),
            "score": planner._program_selection_cost(joint, coaching, ordinary),
            "joint_cost": joint, "coaching_cost": coaching, **details,
            "sessions": [asdict(x) for x in plan], "maximum_date_gap": max(gaps, default=0),
            "partial_static_gate_failures": [asdict(x) for x in failures],
        })
        if number % 25 == 0:
            print(f"  evaluated {number}/{len(calendars)} calendars", flush=True)
    return sorted(rows, key=lambda x: (x["score"], x["offsets"]))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-additions", type=int, default=2)
    parser.add_argument("--variants", default="current,history_fixed,history_trajectory")
    parser.add_argument("--retain-saved-prescriptions", action="store_true",
                        help="Offer eligible saved easy prescriptions as isolated candidates (offline only)")
    args = parser.parse_args()
    if args.output_json.exists():
        parser.error("Choose a fresh output filename")
    variants = args.variants.split(",")
    if not variants or variants[0] != "current" or len(set(variants)) != len(variants) or any(x not in {"current", "history_fixed", "history_trajectory"} for x in variants):
        parser.error("Use distinct known variants with current first")
    started = perf_counter()
    case = load_case(args.database)
    pool = opportunity_pool(case["seed"], case["allowed"], case["required"], max_additions=args.max_additions)
    priors = eligible_prior_prescriptions(
        case["prior"], [x.as_of.date() for x in case["states"]],
        excluded_offsets=set(range(21)) - case["allowed"],
    ) if args.retain_saved_prescriptions else {}
    report = {
        "scope": "Common full-21-day pool with exhaustive additions around a fixed seed, not global date/role/dose optimality or daily replay. Existing scores, guards and allocator retained. No deployment.",
        "planner_version": planner.WEEKLY_PLANNER_VERSION,
        "snapshot_version": case["payload"].get("planner_version"),
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed": case["seed"], "candidate_count": len(pool),
        "max_additions": args.max_additions, "target_trajectory": case["targets"],
        "ordinary_reference_miles": case["profile"].ordinary_planning_miles,
        "capacity_reference_miles": case["evidence"].capacity_reference_miles,
        "saved_prescriptions": {day.isoformat(): result.distance_range_miles for day, result in priors.items()},
        "variants": [],
    }
    for mode in variants:
        print(f"Starting {mode}: {len(pool)} calendars", flush=True)
        stamp = perf_counter()
        if mode == "current":
            rows = evaluate_pool(case, pool)
            trajectory, measurements = None, None
        else:
            trajectory = dose_trajectory(
                case["profile"], [x.as_of.date() for x in case["states"]], case["targets"][:21],
                case["evidence"].capacity_reference_miles, mode=mode, fixed_fraction=case["fixed_fraction"],
            )
            with history_allocator(case["profile"], trajectory, planner._established_easy_midpoint_floor(case["states"][0]),
                                   prior_prescriptions=priors) as measurements:
                rows = evaluate_pool(case, pool)
        report["variants"].append({
            "variant": mode, "seconds": perf_counter() - stamp,
            "dose_trajectory": asdict(trajectory) if trajectory else None,
            "adapter_calls": measurements, "winner": rows[0], "candidates": rows,
        })
        print(f"Finished {mode}: {rows[0]['actual_run_count']} runs, {rows[0]['planned_midpoint_miles']:.2f} miles, score {rows[0]['score']:.3f}", flush=True)
    report["seconds"] = perf_counter() - started
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("x") as out:
        json.dump(report, out, indent=2, default=lambda x: x.isoformat())
        out.write("\n")
    print(f"Wrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
