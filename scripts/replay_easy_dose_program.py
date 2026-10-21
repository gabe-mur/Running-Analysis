"""Run actual daily/upload replans with an offline easy-dose policy adapter.

Copies a read-only frozen snapshot to a NEW explicit disposable database. All
prototype hooks are process-local. This tests dose policy with the production
observed-history estimator, not the separate medium-long normalization proposal.
"""

from argparse import ArgumentParser
from contextlib import contextmanager
from datetime import datetime
from inspect import signature
import json
from pathlib import Path
import sqlite3
import sys

from scripts import simulate_adherence as runner
from scripts.audit_easy_dose_program import (dose_trajectory, eligible_prior_prescriptions,
    history_allocator, history_profile_from_state)
import run_analysis.adherence_projection as projection
import run_analysis.weekly_schedule as planner


@contextmanager
def replay_policy(mode):
    original_build = planner.build_weekly_schedule
    original_projection_build = projection.build_weekly_schedule
    call_signature = signature(original_build)
    records = []

    def build(*args, **kwargs):
        bound = call_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        inputs = bound.arguments
        states, evidence, projector = inputs["daily_states"], inputs["target_evidence"], inputs["expected_target_projector"]
        state = states[0]
        ordinary = state.typical_easy_run_miles
        if not ordinary or evidence is None or evidence.capacity_reference_miles <= 0 or projector is None:
            records.append({"as_of": state.as_of.isoformat(), "fallback": "missing_established_reference_or_projector"})
            return original_build(*args, **kwargs)
        config = inputs["config"].get("coaching", {})
        # The simulator has already rebuilt this reference from realized history.
        # Do not freeze the opening profile or add planned sessions as evidence.
        profile = history_profile_from_state(state, inputs['config'])
        targets = projector([], len(states))
        trajectory = dose_trajectory(
            profile, [x.as_of.date() for x in states], targets, evidence.capacity_reference_miles,
            mode=mode, fixed_fraction=min(.10, max(0., float(config.get("general_fitness_progression_fraction", .08)))),
        )
        excluded = set(inputs.get("forced_rest_offsets") or ()) | {
            offset for offset, activities in (inputs.get("completed_activities_by_offset") or {}).items()
            if activities
        }
        priors = eligible_prior_prescriptions(
            inputs.get("prior_schedule"), trajectory.dates, excluded_offsets=excluded,
        )
        with history_allocator(profile, trajectory, planner._established_easy_midpoint_floor(state),
                               prior_prescriptions=priors) as measurements:
            result = original_build(*args, **kwargs)
        records.append({
            "as_of": state.as_of.isoformat(), "ordinary_reference": ordinary,
            "capacity_reference": evidence.capacity_reference_miles,
            "upper_intent_miles": trajectory.upper_miles, "adapter_calls": measurements.copy(),
            "eligible_prior_dates": [day.isoformat() for day in sorted(priors)],
        })
        return result

    try:
        if mode != "current":
            planner.build_weekly_schedule = build
            projection.build_weekly_schedule = build
        yield records
    finally:
        planner.build_weekly_schedule = original_build
        projection.build_weekly_schedule = original_projection_build


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--prototype", choices=("current", "history_fixed", "history_trajectory"), default="history_trajectory")
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--scenario", default="perfect")
    parser.add_argument("--start-at", default="2026-09-22T19:00:00-04:00")
    args = parser.parse_args()
    if args.database.exists() or args.output_json.exists():
        parser.error("Disposable database and report must both be new paths")
    if not args.snapshot.is_file() or args.days < 1 or datetime.fromisoformat(args.start_at).utcoffset() is None:
        parser.error("Require existing snapshot, positive days and timezone-aware start")
    args.database.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(args.snapshot.resolve().as_uri() + "?mode=ro", uri=True) as source:
        with sqlite3.connect(args.database) as destination:
            source.backup(destination)
    previous_argv = sys.argv
    try:
        sys.argv = ["simulate_adherence", "--project-root", str(args.project_root),
                    "--database", str(args.database), "--start-at", args.start_at,
                    "--simulation-days", str(args.days), "--scenario", args.scenario,
                    "--replan-days", "1", "--ignore-saved-plan", "--progress-replans",
                    "--fail-on-regression", "--stop-on-first-failure", "--output-json", str(args.output_json)]
        with replay_policy(args.prototype) as records:
            try:
                runner.main()
            finally:
                if args.output_json.exists():
                    report = json.loads(args.output_json.read_text())
                    report["offline_dose_prototype"] = {
                        "mode": args.prototype, "builds": records,
                        "scope": "Daily plus upload replans; current search, weights and observed-history estimator. No alternative medium-long normalization, search widening, settings change or deployment.",
                    }
                    args.output_json.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        sys.argv = previous_argv


if __name__ == "__main__":
    main()
