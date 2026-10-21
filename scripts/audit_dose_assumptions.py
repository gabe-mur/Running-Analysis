"""Offline sensitivity/ablation audit; never changes production defaults.

Synthetic single-session probes separate allowable distance from chosen dose.
An optional read-only snapshot compares a bounded common set of calendars.
Neither experiment validates physiology or replaces daily-replanning tests.
"""

from __future__ import annotations

from argparse import ArgumentParser
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from inspect import getsource
from itertools import combinations
import json
from math import floor
from pathlib import Path
import sqlite3
from time import perf_counter

import run_analysis.weekly_schedule as planner
from run_analysis.recommendation import recommend_next_run, typical_easy_distance
from run_analysis.recovery import RECOVERY_HALF_LIFE_HOURS
from run_analysis.web.schemas import (
    ConfidenceLevel, CurrentHealthStatus, FitnessState, FitnessTrend, LoadContext,
    LoadWindow, PlannedWeather, RecommendationRequest, SessionDifficulty,
    WeeklyScheduleDay, WeeklyScheduleResponse, WeeklyTargetEvidence, WorkoutType, ZoneBreakdown,
)


@dataclass(frozen=True)
class Variant:
    name: str
    multiplier: float = 1.25
    allowance_miles: float = 0.5
    release_pressure: float = 0.05
    mode: str = "two_stage"
    multiplier_scope: str = "envelope_only"


VARIANTS = {
    item.name: item for item in (
        Variant("current"),
        Variant("multiplier_removed", multiplier=1.0),
        Variant("multiplier_low", multiplier=1.10),
        Variant("multiplier_high", multiplier=1.50),
        Variant("multiplier_low_all", multiplier=1.10, multiplier_scope="all_consumers"),
        Variant("multiplier_high_all", multiplier=1.50, multiplier_scope="all_consumers"),
        Variant("allowance_removed", allowance_miles=0.0),
        Variant("allowance_high", allowance_miles=1.0),
        Variant("release_narrow", release_pressure=0.025),
        Variant("release_wide", release_pressure=0.10),
        Variant("joint_low", multiplier=1.10, allowance_miles=0.0, release_pressure=0.025),
        Variant("joint_high", multiplier=1.50, allowance_miles=1.0, release_pressure=0.10),
        Variant("single_stage", mode="single_stage"),
        Variant("envelope_removed", mode="none"),
    )
}
REQUEST = RecommendationRequest(health_status=CurrentHealthStatus.NORMAL)
CONFIG = {"coaching": {"long_run_progression_factor": 1.10, "reduced_volume_factor": 0.70}}
ORIGIN = datetime(2026, 9, 22, 19, tzinfo=timezone.utc)


def expansion_reference(state: FitnessState, variant: Variant) -> float:
    low, high = typical_easy_distance(state)
    return max(high, floor((low + high) / 2 * variant.multiplier * 2 + 1e-9) / 2)


def ceiling(state, result, maximum: float, variant: Variant) -> float:
    pressure = planner._easy_volume_recovery_pressure(result)
    if pressure <= 0 or variant.mode == "none":
        return maximum
    ordinary = planner._midpoint(result)
    if variant.mode == "single_stage":
        return min(maximum, ordinary + max(0.0, maximum - ordinary) * (1 - pressure))
    secondary = max(ordinary, expansion_reference(state, variant) + variant.allowance_miles)
    return min(
        maximum,
        ordinary + (secondary - ordinary) * (1 - pressure)
        + max(0.0, maximum - secondary) * max(0.0, 1 - pressure / variant.release_pressure),
    )


@contextmanager
def apply_variant(variant: Variant):
    """Process-local audit only; restore even on failure. Not thread-safe."""
    original_ceiling = planner._easy_recovery_allocation_ceiling
    original_reference = planner._ordinary_easy_expansion_reference
    measurements = {"ceiling_calls": 0, "restrictive_ceiling_calls": 0}

    def measured_ceiling(state, result, maximum):
        value = (
            original_ceiling(state, result, maximum)
            if variant.name == "current"
            else ceiling(state, result, maximum, variant)
        )
        measurements["ceiling_calls"] += 1
        measurements["restrictive_ceiling_calls"] += int(value < maximum - 1e-9)
        return value

    try:
        planner._easy_recovery_allocation_ceiling = measured_ceiling
        if variant.name != "current":
            if variant.multiplier_scope == "all_consumers":
                planner._ordinary_easy_expansion_reference = (
                    lambda state: expansion_reference(state, variant)
                )
        yield measurements
    finally:
        planner._easy_recovery_allocation_ceiling = original_ceiling
        planner._ordinary_easy_expansion_reference = original_reference


def synthetic_state(ordinary_miles: float, residual: float) -> FitnessState:
    def window(days: int) -> LoadWindow:
        count = 3 * days // 7
        return LoadWindow(
            days=days, distance_miles=count * ordinary_miles,
            moving_minutes=count * ordinary_miles * 11,
            zone_load=count * ordinary_miles * 25, activity_count=count,
        )
    return FitnessState(
        as_of=ORIGIN, window_days=28, fitness_trend=FitnessTrend.STABLE,
        trend_confidence=ConfidenceLevel.MODERATE,
        recent_load=LoadContext(
            trailing_7d=window(7), trailing_14d=window(14), trailing_28d=window(28),
            acute_to_prior_ratio=1.0, confidence=ConfidenceLevel.HIGH,
        ),
        days_since_last_run=3, days_since_quality_run=4, days_since_long_run=3,
        last_run=SessionDifficulty(
            distance_miles=ordinary_miles, moving_minutes=ordinary_miles * 11,
            elapsed_minutes=ordinary_miles * 11, zone_load=ordinary_miles * 25,
            stopped_minutes=0, zone_breakdown=ZoneBreakdown(),
        ),
        last_run_workout_type=WorkoutType.EASY, longest_run_30d_miles=ordinary_miles * 2,
        running_days_28d=12, easy_fraction_14d=0.82, moderate_fraction_14d=0.10,
        moderate_evidence_runs_14d=3, hard_fraction_14d=0.08,
        recent_performance_response="within_recent_range",
        typical_easy_run_miles=ordinary_miles, recovery_residual_load=residual,
    )


def recommendation(state):
    return recommend_next_run(state, REQUEST, CONFIG, weekly_role="easy", allowed_candidates={"easy"})


def aerobic_maximum(state) -> float:
    high = typical_easy_distance(state)[1]
    return max(high, min(high * 1.75, planner.long_run_reference_miles(state) * 1.05))


def allocate_one(state, result, target):
    day = WeeklyScheduleDay(
        date=state.as_of.date(), planned_at=state.as_of, recommendation=result,
        day_role="easy_run", rationale="Offline dose sensitivity fixture",
    )
    allocated = planner._allocate_visible_distance_ranges(
        [day], [state], target, CONFIG, preserve_projected_recovery_caps=True,
    )[0]
    return {
        "range": allocated.recommendation.distance_range_miles,
        "midpoint": planner._midpoint(allocated.recommendation),
        "role": allocated.day_role,
    }


def synthetic_audit(variants, profiles=(2.0, 3.7, 6.0)):
    # Dense probes around the secondary-release region; wider probes retain
    # the rest of the ready range. These are sensitivity inputs, not cohorts.
    curve_residuals = sorted({round(i * .005, 6) for i in range(111)} | {.0999, .1001, .11125, .1225})
    dose_residuals = (0.0, .10, .105, .11, .12, .1225, .13, .177, .236, .30, .50, .55)
    curves, allocations, cautions = [], [], []
    baseline_curve = {}
    for size in profiles:
        for residual in curve_residuals:
            state = synthetic_state(size, residual)
            result = recommendation(state)
            maximum = aerobic_maximum(state)
            current = planner._easy_recovery_allocation_ceiling(state, result, maximum)
            reproduced = ceiling(state, result, maximum, VARIANTS["current"])
            if abs(current - reproduced) > 1e-9:
                raise AssertionError("Audit formula no longer matches production")
            baseline_curve[size, residual] = current
    for variant in variants:
        started = perf_counter()
        with apply_variant(variant):
            for size in profiles:
                for residual in curve_residuals:
                    state = synthetic_state(size, residual)
                    result = recommendation(state)
                    maximum = aerobic_maximum(state)
                    current = planner._easy_recovery_allocation_ceiling(state, result, maximum)
                    later = state.model_copy(update={
                        "as_of": state.as_of + timedelta(hours=1),
                        "days_since_last_run": state.days_since_last_run + 1 / 24,
                        "recovery_residual_load": residual * .5 ** (1 / RECOVERY_HALF_LIFE_HOURS),
                    })
                    later_result = recommendation(later)
                    later_ceiling = planner._easy_recovery_allocation_ceiling(later, later_result, maximum)
                    curves.append({
                        "variant": variant.name, "ordinary_history_miles": size,
                        "residual": residual, "pressure": planner._easy_volume_recovery_pressure(result),
                        "ceiling": current, "one_hour_later_ceiling": later_ceiling,
                        "one_hour_change": later_ceiling - current,
                        "baseline_difference": current - baseline_curve[size, residual],
                    })
                for residual in dose_residuals:
                    state = synthetic_state(size, residual)
                    result = recommendation(state)
                    later = state.model_copy(update={
                        "as_of": state.as_of + timedelta(hours=1),
                        "days_since_last_run": state.days_since_last_run + 1 / 24,
                        "recovery_residual_load": residual * .5 ** (1 / RECOVERY_HALF_LIFE_HOURS),
                    })
                    for demand in ("ordinary", "expanded"):
                        factor = 1.0 if demand == "ordinary" else 1.6
                        target = (size * factor - .25, size * factor + .25)
                        first = allocate_one(state, result, target)
                        second = allocate_one(later, recommendation(later), target)
                        allocations.append({
                            "variant": variant.name, "ordinary_history_miles": size,
                            "residual": residual, "demand": demand, "target": target,
                            "initial": first, "one_hour_later": second,
                            "one_hour_change": second["midpoint"] - first["midpoint"],
                        })
                for evidence in ("recovery", "load", "rpe", "weather"):
                    state = synthetic_state(size, .236)
                    if evidence == "recovery":
                        state = state.model_copy(update={"recovery_residual_load": .8})
                    elif evidence == "load":
                        state = state.model_copy(update={"recent_load": state.recent_load.model_copy(
                            update={"acute_distance_to_capacity_ratio": 1.5}
                        )})
                    elif evidence == "rpe":
                        state = state.model_copy(update={"last_run": state.last_run.model_copy(
                            update={"perceived_exertion": 8}
                        )})
                    else:
                        state = state.model_copy(update={"planned_weather": PlannedWeather(
                            forecast_time=state.as_of, apparent_temperature_f=98,
                            temperature_f=95, dewpoint_f=74,
                        )})
                    result = recommendation(state)
                    dose = allocate_one(state, result, (size * 2, size * 2 + .5))
                    # Preserved restrictions, not an independent safety proof.
                    protected = dose["midpoint"] <= planner._midpoint(result) + .05 + 1e-9
                    cautions.append({
                        "variant": variant.name, "ordinary_history_miles": size,
                        "evidence": evidence, "standalone_range": result.distance_range_miles,
                        "allocated": dose, "restriction_preserved": protected,
                    })
        print(f"Synthetic {variant.name}: {perf_counter() - started:.2f}s", flush=True)
    baseline_doses = {
        (row["ordinary_history_miles"], row["residual"], row["demand"]): row
        for row in allocations if row["variant"] == "current"
    }
    summaries = []
    for variant in variants:
        selected = [x for x in allocations if x["variant"] == variant.name]
        raw = [x for x in curves if x["variant"] == variant.name]
        changes = [
            x["initial"]["midpoint"] - baseline_doses[
                x["ordinary_history_miles"], x["residual"], x["demand"]
            ]["initial"]["midpoint"] for x in selected
        ]
        summaries.append({
            "variant": variant.name, "allocated_case_count": len(selected),
            "changed_dose_cases": sum(abs(x) > 1e-9 for x in changes),
            "largest_dose_difference_from_current": max(map(abs, changes)),
            "largest_one_hour_ceiling_change": max(raw, key=lambda x: abs(x["one_hour_change"])),
            "largest_one_hour_dose_change": max(selected, key=lambda x: abs(x["one_hour_change"])),
            "caution_failures": sum(not x["restriction_preserved"] for x in cautions if x["variant"] == variant.name),
        })
    return {"summaries": summaries, "curves": curves, "allocations": allocations, "cautions": cautions}


def full_calendar_pool(seed, allowed, required):
    """Perturb only the opening sessions; keep the full downstream horizon."""
    seed = tuple(sorted(seed))
    candidates = {seed}
    for offset in seed[:3]:
        for delta in (-1, 1):
            replacement = offset + delta
            if replacement in allowed and replacement not in seed:
                candidates.add(tuple(sorted((set(seed) - {offset}) | {replacement})))
    additions = sorted(set(allowed) - set(seed))
    if additions:
        candidates.add(tuple(sorted((*seed, additions[0]))))
    if len(seed) > 1:
        candidates.add(seed[1:])
    return sorted(c for c in candidates if set(c).issubset(allowed) and required.issubset(c))


def calendar_audit(database: Path, variants, *, horizon=6, counts=(2, 3), mode="full_pool"):
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as db:
        row = db.execute("SELECT value_json FROM app_state WHERE key='weekly_planner_snapshot'").fetchone()
        internal_row = db.execute("SELECT value_json FROM app_state WHERE key='weekly_schedule_internal'").fetchone()
    if row is None:
        raise ValueError("Missing frozen planner snapshot")
    payload = json.loads(row[0])
    if mode == "full_pool":
        horizon = len(payload["daily_states"])
    states = [FitnessState.model_validate(x) for x in payload["daily_states"][:horizon]]
    options = [[FitnessState.model_validate(x) for x in group] for group in payload["daily_state_options"][:horizon]]
    if len(states) != horizon or len(options) != horizon:
        raise ValueError("Snapshot shorter than audit horizon")
    completed_miles = {
        int(k): sum(item["distance_miles"] for item in group)
        for k, group in payload.get("completed_activities_by_offset", {}).items()
    }
    completed_miles.update({int(k): v for k, v in payload.get("completed_miles_by_offset", {}).items()})
    forbidden = set(payload.get("forced_rest_offsets", [])) | set(payload.get("completed_run_offsets", [])) | set(completed_miles)
    allowed = sorted(set(range(horizon)) - forbidden)
    goal = planner.configured_race_goal(payload["config"], on_date=states[0].as_of.date())
    required = {i for i in allowed if goal and states[i].as_of.date() == goal[1]}
    projector = None
    if mode == "full_pool":
        internal = json.loads(internal_row[0]) if internal_row else {}
        seed_days = internal.get("days") or payload.get("prior_planning_days")
        if not seed_days or len(seed_days) < horizon:
            raise ValueError("Full-horizon saved calendar required; refusing to pad a seven-day seed")
        day_offsets = {state.as_of.date().isoformat(): i for i, state in enumerate(states)}
        seed = tuple(sorted(
            day_offsets[day["date"]] for day in seed_days
            if day.get("recommendation") and day["recommendation"]["workout_type"] != "rest"
            and day["date"] in day_offsets and day_offsets[day["date"]] in allowed
        ))
        if not seed:
            raise ValueError("No available sessions in saved full-horizon calendar")
        calendars = full_calendar_pool(seed, allowed, required)
        observed = [planner.PlanningActivity(
            start_time=datetime.fromisoformat(item["start_time"]),
            distance_miles=float(item["distance_miles"]), moving_minutes=item.get("moving_minutes"),
            easy_minutes=item.get("easy_minutes"), baseline_eligible=bool(item.get("baseline_eligible", True)),
            target_distance_miles=item.get("target_distance_miles"),
        ) for item in payload["observed_planning_activities"]]
        prior = WeeklyScheduleResponse.model_validate(payload["prior_schedule"]) if payload.get("prior_schedule") else None
        if prior and payload.get("prior_planning_days"):
            prior = prior.model_copy(update={"planning_days": [WeeklyScheduleDay.model_validate(x) for x in payload["prior_planning_days"]]})
        projector = planner.make_expected_target_projector(
            observed, states[0].as_of.date(), payload["config"],
            pace_min_mile=float(payload["target_projection_pace_min_mile"]),
            maximum_horizon_days=horizon,
            opening_target_range=tuple(payload["target_distance_range"]),
            opening_evidence=WeeklyTargetEvidence.model_validate(payload["target_evidence"]),
            ordinary_easy_midpoint_miles=payload.get("ordinary_easy_midpoint_miles"),
            prior_schedule=prior,
        )
    else:
        seed = None
        calendars = [c for count in counts for c in combinations(allowed, count) if required.issubset(c)]
    if not calendars or len(calendars) > 100:
        raise ValueError("Require 1–100 common calendars per variant")
    rows = []
    for variant in variants:
        started = perf_counter()
        with apply_variant(variant) as measurements:
            common = dict(
                daily_states=states, daily_state_options=options,
                request=RecommendationRequest.model_validate(payload["request"]),
                config=payload["config"], target_distance_range=tuple(payload["target_distance_range"]),
                completed_miles_by_offset={k: v for k, v in completed_miles.items() if k < horizon},
                prefix_cache={}, recommendation_load_cache={}, projected_state_cache={},
            )
            candidates = []
            for offsets in calendars:
                details = {}
                joint = planner._joint_candidate_program_cost(
                    offsets, **common, diagnostic_details=details, reconcile_allocated_prefix=True,
                    expected_target_projector=projector,
                )
                coaching = planner._adaptive_candidate_cost(
                    offsets, **common, include_mileage_path=False,
                    include_recovery_interactions=False, include_cadence_pressure=False,
                )
                candidates.append({
                    "offsets": offsets, "run_count": len(offsets),
                    "score": planner._program_selection_cost(joint, coaching, sum(typical_easy_distance(states[0])) / 2),
                    "joint_cost": joint, "coaching_cost": coaching, **details,
                })
            candidates.sort(key=lambda x: (x["score"], x["offsets"]))
            rows.append({"variant": variant.name, "seconds": perf_counter() - started,
                         "winner": candidates[0], "candidates": candidates,
                         "envelope_evaluations": measurements.copy(),
                         "tied_best_count": sum(abs(x["score"] - candidates[0]["score"]) < 1e-9 for x in candidates)})
        print(f"Calendars {variant.name}: {len(calendars)} candidates, {perf_counter() - started:.2f}s", flush=True)
    return {
        "snapshot_generated_at": payload.get("generated_at"),
        "snapshot_planner_version": payload.get("planner_version"),
        "horizon_days": horizon, "candidate_counts": sorted({len(c) for c in calendars}),
        "mode": mode, "seed_offsets": seed, "expected_target_projector": projector is not None,
        "scope": "Common calendar pool; production role/dose reconciliation; no continuity switching cost or daily replans; no global optimum claim; scores only comparable within a variant. Full-pool mode retains the complete horizon and expected target trajectory. Short-exhaustive mode uses a frozen target and can be dominated by boundary allowances.",
        "variants": rows,
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--database", type=Path, help="Optional read-only frozen snapshot for bounded calendar comparison")
    parser.add_argument("--calendar-mode", choices=("full_pool", "short_exhaustive"), default="full_pool")
    parser.add_argument("--program-variants", default="current,allowance_removed,release_wide,single_stage,envelope_removed")
    args = parser.parse_args()
    if args.output_json.exists():
        parser.error("Choose a fresh output filename")
    names = args.variants.split(",")
    program_names = args.program_variants.split(",")
    if names[0] != "current" or program_names[0] != "current" or any(x not in VARIANTS for x in names + program_names):
        parser.error("Use known variants and place current first")
    if len(names) != len(set(names)) or len(program_names) != len(set(program_names)):
        parser.error("Duplicate variants are not supported")
    started = perf_counter()
    report = {
        "planner_version": planner.WEEKLY_PLANNER_VERSION,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "production_helper_sha256": sha256((getsource(planner._ordinary_easy_expansion_reference) + getsource(planner._easy_recovery_allocation_ceiling)).encode()).hexdigest(),
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "variants": [asdict(VARIANTS[name]) for name in names],
        "program_variants": [asdict(VARIANTS[name]) for name in program_names] if args.database else [],
        "interpretation": "Engineering sensitivity, not parameter estimation or physiological validation. Stress ranges are not confidence intervals. No production defaults changed.",
        "synthetic": synthetic_audit([VARIANTS[name] for name in names]),
    }
    if args.database:
        report["calendars"] = calendar_audit(args.database, [VARIANTS[name] for name in program_names], mode=args.calendar_mode)
    report["seconds"] = perf_counter() - started
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    failures = sum(x["caution_failures"] for x in report["synthetic"]["summaries"])
    print(f"Wrote {args.output_json}; {failures} caution-preservation failures; {report['seconds']:.2f}s total")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
