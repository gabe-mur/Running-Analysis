"""Paired identical-input opening benchmark, acceptance off/on; no live data.

Both builds use the four-pass offline search. The second build gets the same
frozen arguments, not the first output as a prior. One pair is a cost screen,
not a statistical performance guarantee. No timing-dependent selection.
"""

from argparse import ArgumentParser
from datetime import datetime, timezone
from inspect import signature
import json
from pathlib import Path
from time import perf_counter

import yaml

from scripts.calendar_neighbor_prototype import calendar_neighbor_policy
from scripts.replay_easy_dose_program import replay_policy
from scripts.replay_synthetic_easy_dose import synthetic_history
from scripts.audit_first_dose_refresh import sessions
from run_analysis import adherence_projection as projection
from run_analysis import weekly_schedule as planner


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--order", choices=("off,on", "on,off"), default="off,on")
    args = parser.parse_args()
    if args.output_json.exists():
        parser.error("Choose a fresh output path")
    config = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.example.yaml").read_text())
    start = datetime(2026, 9, 22, tzinfo=timezone.utc)
    state, history = synthetic_history(start)
    bind = signature(planner.build_weekly_schedule)
    rows = []
    with replay_policy("history_trajectory"):
        original = projection.build_weekly_schedule

        def paired(*positional, **kwargs):
            values = bind.bind(*positional, **kwargs)
            values.apply_defaults()
            inputs = values.arguments
            targets = tuple(inputs["expected_target_projector"]([], 49))
            inputs["expected_target_projector"] = lambda _candidate, n: targets[:n]
            for enabled in (x == "on" for x in args.order.split(",")):
                stamp = perf_counter()
                with calendar_neighbor_policy(max_passes=4, enforce_acceptance=enabled) as traces:
                    plan = original(**inputs)
                rows.append({"acceptance_enabled": enabled, "seconds": perf_counter() - stamp,
                             "sessions": sessions(plan), "trace": traces[0]})
                print(enabled, rows[-1]["seconds"], flush=True)
            return plan

        try:
            projection.build_weekly_schedule = paired
            projection.simulate_adherence(state, history, config, start, simulation_days=1)
        finally:
            projection.build_weekly_schedule = original
    assert len(rows) == 2
    by_mode = {x["acceptance_enabled"]: x for x in rows}
    report = dict(scope=__doc__, order=args.order, sessions_equal=rows[0]["sessions"] == rows[1]["sessions"],
                  wall_delta_seconds=by_mode[True]["seconds"] - by_mode[False]["seconds"], variants=rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("x") as out:
        json.dump(report, out, indent=2)
        out.write("\n")
    print(f"Wrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
