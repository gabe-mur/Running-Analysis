"""Bounded chronological TCX-metric replay with actual recorded HR.

Read-only source; no saved plans, old compliance associations, fitted pace-model
outputs, or weather forecasts. Current settings and per-run annotations are
frozen, not reconstructed historical settings. Availability is recorded finish,
not the original upload timestamp. This is actual-history adaptation, NOT a
perfect-adherence simulation or a prediction of physiological improvement.
"""

from argparse import ArgumentParser
from dataclasses import dataclass, replace
from datetime import datetime, time, timedelta
import json
from pathlib import Path
import sqlite3
from statistics import median
from time import perf_counter
from zoneinfo import ZoneInfo

from scripts.calendar_neighbor_prototype import calendar_neighbor_policy
from scripts.finalized_scoring_trial import finalized_scoring_policy
from scripts.replay_easy_dose_program import replay_policy
from run_analysis import adherence_projection as projection
from run_analysis import weekly_schedule as planner
from run_analysis.config import load_config, resolve_project_path
from run_analysis.projection_gate import _json_ready
from run_analysis.run_feedback import _difficulty, _infer_workout_type, _long_run_threshold
from run_analysis.web.schemas import FitnessState, LoadContext, LoadWindow, RecommendationRequest, WorkoutType


@dataclass(frozen=True)
class RecordedRun:
    activity_id: int
    available_at: datetime
    run: projection.ProjectionRun
    hr_coverage: float | None


def write_report(path, result):
    # Validate serialization before creating a file, including YAML date-valued
    # settings. Keep the original typed settings for actual planner calls.
    encoded = json.dumps(_json_ready(result), indent=2) + "\n"
    with path.open("x") as stream:
        stream.write(encoded)


def load_recorded(database, zone):
    # Deliberately omit prescription joins, cached fitness estimates and model
    # results. Per-activity duration/zones/elevation are independent of later runs.
    with sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT a.id, a.start_time_utc, a.total_distance_m, a.total_elapsed_time_s,
                   m.calculated_moving_time_s, m.device_timer_time_s, m.elapsed_time_s,
                   m.analysis_distance_m, m.hr_zone_seconds_json, m.hr_load_coverage,
                   COALESCE(o.workout_type,m.detected_workout_type) AS workout_type,
                   o.perceived_exertion,
                   (SELECT SUM(s.elevation_gain_m) FROM segments s
                    WHERE s.activity_id=a.id AND s.is_pathological=0) AS derived_elevation_gain_m,
                   (SELECT SUM(s.elevation_loss_m) FROM segments s
                    WHERE s.activity_id=a.id AND s.is_pathological=0) AS derived_elevation_loss_m
            FROM activities a JOIN activity_metrics m ON m.activity_id=a.id
            LEFT JOIN run_overrides o ON o.activity_id=a.activity_id
            WHERE a.sport='Running' AND a.start_time_utc IS NOT NULL
            ORDER BY a.start_time_utc_epoch,a.id
        """).fetchall()
    recorded, past_rows = [], []
    for row in rows:
        started = datetime.fromisoformat(row["start_time_utc"]).astimezone(zone)
        moving_s = row["calculated_moving_time_s"] or row["device_timer_time_s"]
        distance_m = row["total_distance_m"] or row["analysis_distance_m"]
        if not moving_s or not distance_m:
            raise ValueError(f"Activity {row['id']} lacks duration/distance; do not fabricate a run")
        threshold = _long_run_threshold([r for available, r in past_rows if available <= started][-500:])
        miles = distance_m / 1609.344
        zones = json.loads(row["hr_zone_seconds_json"] or "{}")
        workout = _infer_workout_type(row["workout_type"], None, miles, moving_s / 60, zones, threshold)
        difficulty = _difficulty(row, workout, zones)
        elapsed = max(moving_s, row["elapsed_time_s"] or row["total_elapsed_time_s"] or moving_s)
        available = started + timedelta(seconds=elapsed)
        run = projection.ProjectionRun(started, miles, moving_s / 60, workout, difficulty)
        recorded.append(RecordedRun(row["id"], available, run, row["hr_load_coverage"]))
        past_rows.append((available, row))
    return sorted(recorded, key=lambda x: x.available_at)


def known_runs(records, at):
    return sorted([x.run for x in records if x.available_at <= at], key=lambda x: x.start_time)


def activities(history):
    return [planner.PlanningActivity(
        run.start_time, run.distance_miles, moving_minutes=run.moving_minutes,
        easy_minutes=run.difficulty.zone_breakdown.easy_minutes,
        baseline_eligible=run.workout_type in {WorkoutType.EASY, WorkoutType.RECOVERY, WorkoutType.RUN_WALK, WorkoutType.UNKNOWN}
        and not run.difficulty.is_long_run and not run.difficulty.is_quality_session,
    ) for run in history]


def state_at(history, at, config, capacity=None):
    if capacity is None:
        _, _, evidence = planner.derive_weekly_target(activities(history), at, config)
        capacity = evidence.capacity_reference_miles
    empty = {f"trailing_{n}d": LoadWindow(days=n, distance_miles=0, moving_minutes=0, activity_count=0)
             for n in (7, 14, 28)}
    template = FitnessState(as_of=at, window_days=28, fitness_trend="uncertain",
                            trend_confidence="low", recent_load=LoadContext(**empty, confidence="low"))
    return projection._state_at(template, history, at, capacity,
        easy_baseline_half_life_days=float(config.get("coaching", {}).get("capacity_retention_half_life_days", 84)),
        fatigue_half_life_days=float(config.get("coaching", {}).get("continuous_fatigue_half_life_days", 7)))


def inputs_for(history, at, config, prior=None):
    # Match simulator/live completed-day boundary: after a completion, optimize
    # from tomorrow, while retaining that run in all observed-history states.
    completed_today = any(r.start_time.astimezone(at.tzinfo).date() == at.date() for r in history)
    first_date = at.date() + timedelta(days=int(completed_today))
    origin = datetime.combine(first_date, time.min, tzinfo=at.tzinfo)
    observed = activities(history)
    count, target, evidence = planner.derive_weekly_target(observed, origin, config)
    hours = sorted(set(config.get("weather", {}).get("automatic_run_time_hours_local", [7, 12, 19])))
    options = []
    for offset in range(planner.PLANNING_HORIZON_DAYS):
        day = first_date + timedelta(days=offset)
        times = [datetime.combine(day, time(int(h)), tzinfo=at.tzinfo) for h in hours]
        times = [t for t in times if t > at + timedelta(minutes=10)] or [max(origin, at)]
        options.append([state_at(history, t, config, evidence.capacity_reference_miles) for t in times])
    states = [state_at(history, origin, config, evidence.capacity_reference_miles), *[x[0] for x in options[1:]]]
    window = projection._load_window(history, at, 28)
    pace = min(15., max(7., window.moving_minutes / window.distance_miles if window.distance_miles else 11.))
    projector = planner.make_expected_target_projector(observed, first_date, config,
        pace_min_mile=pace, maximum_horizon_days=len(states), opening_target_range=target,
        opening_evidence=evidence, ordinary_easy_midpoint_miles=states[0].typical_easy_run_miles,
        prior_schedule=prior)
    return dict(daily_states=states, daily_state_options=options,
        request=RecommendationRequest(health_status="normal"), config=config,
        target_run_count=count, target_distance_range=target, target_evidence=evidence,
        completed_activities_by_offset={}, prior_schedule=prior, expected_target_projector=projector)


def recommendations(plan):
    return [d.recommendation for d in (plan.planning_days or plan.days)
            if d.recommendation and d.recommendation.workout_type != WorkoutType.REST]


def hr_reference(history, item, at):
    # Descriptive prior observations, not calibrated prediction/confidence bands.
    def group(kind):
        return "quality" if kind in projection.QUALITY_TYPES else "long" if kind == WorkoutType.LONG else "aerobic"
    samples = [r for r in history if at - timedelta(days=90) <= r.start_time < at
               and group(r.workout_type) == group(item.workout_type)
               and r.difficulty.zone_load is not None and r.moving_minutes > 0]
    if not samples:
        return {"samples": 0}
    rates = [r.difficulty.zone_load / r.moving_minutes for r in samples]
    pace = median(r.moving_minutes / r.distance_miles for r in samples)
    minutes = sum(item.distance_range_miles) / 2 * pace
    return {"samples": len(samples), "latest_sample_at": max(r.start_time for r in samples).isoformat(),
            "zone_load_per_minute_min_median_max": [min(rates), median(rates), max(rates)],
            "expected_minutes_at_prescribed_midpoint": minutes,
            "midpoint_hr_load_using_prior_median": minutes * median(rates)}


def changes(before, after, origin, horizon=7):
    def selected(plan):
        return {x.planned_for.date(): x for x in recommendations(plan)
                if origin <= x.planned_for.date() < origin + timedelta(days=horizon)}
    a, b = selected(before), selected(after)
    return {"date_slots": [d.isoformat() for d in sorted(set(a) ^ set(b))],
            "types": [d.isoformat() for d in sorted(set(a) & set(b)) if a[d].workout_type != b[d].workout_type],
            "ranges": [{"date": d.isoformat(), "before": a[d].distance_range_miles, "after": b[d].distance_range_miles}
                       for d in sorted(set(a) & set(b)) if a[d].distance_range_miles != b[d].distance_range_miles]}


def upload_comparison(record, history, prior, config, prior_hr_references):
    run = record.run
    at = datetime.combine(record.available_at.date() + timedelta(days=1), time.min, tzinfo=record.available_at.tzinfo)
    actual = state_at(history + [run], at, config)
    # Post-hoc attribution ONLY: same actual distance/duration/zones, with the
    # numeric observed HR load hidden. Never feed this counterfactual to planning.
    hidden = replace(run, difficulty=run.difficulty.model_copy(update={"zone_load": None}))
    without_hr = state_at(history + [hidden], at, config)
    item = next((x for x in recommendations(prior) if x.planned_for.date() == run.start_time.date()), None)
    planned = None
    if item:
        raw = state_at(history, at, config)
        planned = planner._project_state(raw, [item], config)
    return {"activity_id": record.activity_id, "run_at": run.start_time.isoformat(),
        "available_at": record.available_at.isoformat(), "actual_miles": run.distance_miles,
        "actual_minutes": run.moving_minutes, "actual_type": run.workout_type.value,
        "hr_load": run.difficulty.zone_load, "hr_coverage": record.hr_coverage,
        "same_date_prescription": item.model_dump(mode="json") if item else None,
        "distance_in_range": item.distance_range_miles[0] <= run.distance_miles <= item.distance_range_miles[1] if item else None,
        "type_matches": run.workout_type == item.workout_type if item else None,
        "timing_delta_hours": (run.start_time - item.planned_for).total_seconds() / 3600 if item else None,
        "prior_only_hr_reference": prior_hr_references.get(run.start_time.date().isoformat()),
        "recovery_evaluated_at": at.isoformat(),
        "production_plan_projection_recovery": planned.recovery_residual_load if planned else None,
        "recorded_recovery": actual.recovery_residual_load,
        "same_actual_run_without_numeric_hr_recovery": without_hr.recovery_residual_load,
        "hr_channel_delta_posthoc": actual.recovery_residual_load - without_hr.recovery_residual_load,
        "interpretation": "Date/range fit is descriptive, not proof of following this counterfactual plan. Recovery forecast differences mix timing, dose, HR and reference changes; not automatically instability."}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2026-09-20")
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    journal = args.output_json.with_suffix(".events.jsonl")
    if args.output_json.exists() or journal.exists() or not 1 <= args.days <= 7:
        parser.error("Choose a new report and a bounded 1–7 day replay")
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "config.yaml")
    zone = ZoneInfo(str(config["timezone_default"]))
    start = datetime.fromisoformat(args.start).replace(tzinfo=zone)
    end = start + timedelta(days=args.days)
    records = load_recorded(resolve_project_path(root, config["paths"]["database"]), zone)
    events = [(start + timedelta(days=i), "daily", None) for i in range(args.days)]
    events += [(r.available_at, "upload", r) for r in records if start <= r.available_at < end]
    events.sort(key=lambda x: (x[0], x[1]))
    result = {"scope": __doc__, "start_at": start.isoformat(), "until": end.isoformat(),
              "config": config, "decisions": [], "uploads": [], "status": "running",
              "recorded_inputs": [{"activity_id": r.activity_id, "available_at": r.available_at.isoformat(),
                                   "start_time": r.run.start_time.isoformat(), "distance_miles": r.run.distance_miles,
                                   "moving_minutes": r.run.moving_minutes, "workout_type": r.run.workout_type.value,
                                   "difficulty": r.run.difficulty.model_dump(mode="json"), "hr_coverage": r.hr_coverage}
                                  for r in records if r.available_at < end],
              "limitations": ["Current per-run classifications/annotations, not historical versions.",
                              "Weather absent; no inferred future HR, sleep, soreness or measured fitness improvement.",
                              "Uses shared load-state projection, not full HTTP/FIT/TCX ingestion or pace-model reprocessing.",
                              "HR references are descriptive prior samples, not calibrated confidence intervals."]}
    prior, prior_hr_references = None, {}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with journal.open("x") as stream:
        stream.write(json.dumps(_json_ready({"inputs": result})) + "\n")
    try:
        with finalized_scoring_policy() as scoring, calendar_neighbor_policy(max_passes=4, enforce_acceptance=True) as search, replay_policy("history_trajectory") as doses:
            for at, trigger, record in events:
                history = known_runs(records, at)
                if record and prior:
                    before = known_runs([r for r in records if r.activity_id != record.activity_id], at)
                    result["uploads"].append(upload_comparison(record, before, prior, config, prior_hr_references))
                inputs = inputs_for(history, at, config, prior)
                started = perf_counter()
                plan = planner.build_weekly_schedule(**inputs)
                seconds = perf_counter() - started
                state = inputs["daily_states"][0]
                not_ready = [x.planned_for.isoformat() for x in recommendations(plan) if x.readiness.value == "not_ready"]
                references = {x.planned_for.date().isoformat(): hr_reference(history, x, at) for x in recommendations(plan)}
                entry = {"at": at.isoformat(), "trigger": trigger, "planning_seconds": seconds,
                    "known_activity_ids": [r.activity_id for r in records if r.available_at <= at],
                    "known_runs": len(history), "known_hr_runs": sum(r.difficulty.zone_load is not None for r in history),
                    "state": state.model_dump(mode="json"),
                    "target_range": inputs["target_distance_range"],
                    "plan": [x.model_dump(mode="json") for x in recommendations(plan)],
                    "prior_only_hr_references": references,
                    "near4_changes": changes(prior, plan, state.as_of.date(), 4) if prior else None,
                    "visible7_changes": changes(prior, plan, state.as_of.date(), 7) if prior else None,
                    "selected_not_ready": not_ready}
                result["decisions"].append(entry)
                with journal.open("a") as stream:
                    stream.write(json.dumps(_json_ready({"decision": entry, "uploads": result["uploads"]})) + "\n")
                print(f"{at.isoformat()} {trigger}: {seconds:.2f}s; known {len(history)} runs; recovery {state.recovery_residual_load:.3f}", flush=True)
                if not_ready:
                    raise ValueError("Selected not-ready session; stop and inspect report")
                prior, prior_hr_references = plan, references
            result.update(status="completed", scoring_measurements=scoring, search=search, dose_builds=doses)
    except Exception as error:
        result.update(status="stopped", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_report(args.output_json, result)
        print(f"Wrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
