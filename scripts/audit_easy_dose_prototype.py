"""Compare offline history-led intentions with existing selected easy doses.

No production hooks, database writes, live replans or physiological validation.
The menu is NOT a replacement allocator; full-program selection is a next step.
"""

from argparse import ArgumentParser
from dataclasses import asdict
from datetime import timedelta
from hashlib import sha256
import json
from pathlib import Path
import sqlite3

from scripts import audit_dose_assumptions as current
from scripts.easy_dose_prototype import Observation, build_profile, generate_intents, restrict_intents
from run_analysis.run_feedback import list_runs


def paired_probes():
    rows = []
    for size in (2., 3.7, 6.):
        history = tuple(Observation(str(i), current.ORIGIN - timedelta(days=i * 3 + 2), size)
                        for i in range(8))
        profile = build_profile(history, current.ORIGIN, half_life_days=84)
        later_profile = build_profile(history, current.ORIGIN + timedelta(hours=1), half_life_days=84)
        for residual in (.10, .105, .11, .12, .1225, .13, .177, .236, .30, .50, .55):
            state = current.synthetic_state(size, residual)
            result = current.recommendation(state)
            later = state.model_copy(update={
                "as_of": state.as_of + timedelta(hours=1),
                "days_since_last_run": state.days_since_last_run + 1 / 24,
                "recovery_residual_load": residual * .5 ** (1 / current.RECOVERY_HALF_LIFE_HOURS),
            })
            # Hold training-intent inputs constant across the recovery comparison.
            floor = current.planner._established_easy_midpoint_floor(state)
            kwargs = dict(useful_floor_miles=floor, progression_allowance_miles=size * .08,
                          progression_source="explicit_8_percent_transfer_hypothesis")
            menu = generate_intents(profile, **kwargs)
            later_menu = generate_intents(later_profile, **kwargs)
            target = (size * 1.6 - .25, size * 1.6 + .25)
            first = current.allocate_one(state, result, target)
            second = current.allocate_one(later, current.recommendation(later), target)
            rows.append({
                "size": size, "residual": residual, "current_initial": first,
                "current_one_hour_later": second,
                "current_change": second["midpoint"] - first["midpoint"],
                "prototype_intents": asdict(menu),
                "prototype_intents_unchanged": menu == later_menu,
                "warning": "Stable intentions are not evidence of stable final schedule selection.",
            })
    return rows


def frozen_history(database):
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        row = db.execute("SELECT value_json FROM app_state WHERE key='weekly_planner_snapshot'").fetchone()
        if row is None:
            raise ValueError("Missing snapshot")
        payload = json.loads(row[0])
        # Read all summaries, preserving the application's classification and
        # prescription matching rather than inventing another SQL classifier.
        count = db.execute("SELECT COUNT(*) FROM activities").fetchone()[0]
        runs = list_runs(db, limit=count)
    state = current.FitnessState.model_validate(payload["daily_states"][0])
    observations = [Observation(
        activity_id=str(run.activity_id), occurred_at=run.start_time,
        actual_miles=run.distance_miles, moving_minutes=run.moving_minutes,
        workout_type=run.workout_type.value, health=run.health_tag.value,
        planning_role=run.prescribed_planning_role,
        prescribed_range=run.prescribed_distance_range_miles,
        perceived_exertion=run.session_difficulty.perceived_exertion if run.session_difficulty else None,
        # Available summaries do not contain explicit post-run recovery feedback.
        recovery_response=None,
    ) for run in runs if run.start_time is not None and run.distance_miles > 0]
    config = payload["config"].get("coaching", {})
    half_life = float(config.get("capacity_retention_half_life_days", 84))
    profile = build_profile(observations, state.as_of, half_life_days=half_life)
    fraction = min(.10, max(0., float(config.get("general_fitness_progression_fraction", .08))))
    menu = generate_intents(
        profile, useful_floor_miles=current.planner._established_easy_midpoint_floor(state),
        progression_allowance_miles=(profile.ordinary_planning_miles or 0) * fraction,
        progression_source="experimental_transfer_of_existing_program_progression_fraction",
    )
    return {
        "snapshot_version": payload.get("planner_version"), "profile": asdict(profile),
        "current_state_reference": state.typical_easy_run_miles,
        "transferred_progression_fraction": fraction, "intent_menu": asdict(menu),
        "warning": "Applying a program growth fraction to one session is an explicit prototype assumption, not calibrated physiology. No frozen calendar or live plan is changed.",
    }


def funding_probe(database, history):
    """An optimistic bound, not allocation or recovery clearance.

    Retain each current candidate's long/quality sessions and dates. Replace
    every easy dose with the largest proposed intent. If that still cannot
    meet the target, these menus need a different calendar or progression
    policy; don't discover that only after a 90-day simulation.
    """
    options = history["intent_menu"]["options"]
    if not options:
        return {"scope": "No ordinary history; no funding comparison", "rows": []}
    largest = max(x["midpoint_miles"] for x in options)
    comparison = current.calendar_audit(database, [current.VARIANTS["current"]])
    rows = []
    for candidate in comparison["variants"][0]["candidates"]:
        trace = [x.split(":") for x in candidate["allocation_trace"]]
        easy_count = sum(x[2] == "easy" for x in trace)
        fixed_miles = sum(float(x[3]) for x in trace if x[2] != "easy")
        maximum = fixed_miles + easy_count * largest
        target = candidate["integrated_target_low_miles"]
        rows.append({
            "offsets": candidate["offsets"], "easy_count": easy_count,
            "current_total": candidate["planned_midpoint_miles"],
            "optimistic_prototype_total": maximum, "target_lower_bound": target,
            "optimistic_shortfall": max(0, target - maximum),
        })
    return {
        "scope": "Fixed calendar/key doses, frozen opening intent profile. Optimistic budget diagnostic only; no recovery screening, target tolerance, future progression or date search. Shortfall is not an automatic gate failure.",
        "rows": rows,
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--database", type=Path)
    args = parser.parse_args()
    if args.output_json.exists():
        parser.error("Choose a fresh output filename")
    rows = paired_probes()
    profile = build_profile([Observation("probe", current.ORIGIN - timedelta(days=2), 4)],
                            current.ORIGIN, half_life_days=84)
    menu = generate_intents(profile, useful_floor_miles=3, progression_allowance_miles=.32,
                            progression_source="explicit_test_allowance")
    available, rejected = restrict_intents(menu, maximum_midpoint_miles=3.5,
                                           reason="controlled_material_caution")
    report = {
        "scope": "Intent generation and diagnostic filtering only; not a new optimizer, daily replay, safety validation or release test.",
        "planner_version": current.planner.WEEKLY_PLANNER_VERSION,
        "prototype_sha256": sha256(Path(__file__).with_name("easy_dose_prototype.py").read_bytes()).hexdigest(),
        "paired_probes": rows,
        "summary": {
            "paired_cases": len(rows),
            "unchanged_intent_menus": sum(x["prototype_intents_unchanged"] for x in rows),
            "largest_current_one_hour_dose_change": max(abs(x["current_change"]) for x in rows),
            "prototype_selected_dose_change": None,
        },
        "restriction_probe": {"available": [asdict(x) for x in available], "rejected": rejected},
    }
    if args.database:
        report["frozen_history"] = frozen_history(args.database)
        report["funding_probe"] = funding_probe(args.database, report["frozen_history"])
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("x") as out:
        json.dump(report, out, indent=2, default=lambda x: x.isoformat())
        out.write("\n")
    print(json.dumps(report["summary"], indent=2))
    if args.database:
        print("Frozen ordinary reference:", report["frozen_history"]["profile"]["ordinary_planning_miles"])
        print("Intent centers:", [x["midpoint_miles"] for x in report["frozen_history"]["intent_menu"]["options"]])


if __name__ == "__main__":
    main()
