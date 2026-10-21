"""Compare production search with exhaustive final scoring on a small snapshot.

Use a copied SQLite database. No database writes or application replans occur.
This enumerates date choices at a fixed frequency; role/dose allocation remains
the production allocator, so it is not a proof of global physiological optimality.
"""

from argparse import ArgumentParser
from datetime import datetime
from itertools import combinations
import json
from math import comb
from pathlib import Path
import sqlite3
from time import perf_counter

import run_analysis.weekly_schedule as planner
from run_analysis.web.schemas import FitnessState, RecommendationRequest


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--exact-limit", type=int, default=None)
    parser.add_argument("--beam-width", type=int, default=None)
    parser.add_argument("--finalists", type=int, default=None)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    for name, value in (
        ("MAX_EXACT_PROGRAM_CANDIDATES", args.exact_limit),
        ("MAX_ADAPTIVE_CANDIDATES", args.beam_width),
        ("JOINT_DATE_FINALISTS", args.finalists),
    ):
        if value is not None:
            setattr(planner, name, max(0 if name == "MAX_EXACT_PROGRAM_CANDIDATES" else 1, value))
    if not 1 <= args.count <= args.horizon <= 21:
        parser.error("Require 1 <= count <= horizon <= 21")
    if comb(args.horizon, args.count) > 1000:
        parser.error("Bounded audit supports at most 1000 calendars")
    if args.output_json.exists():
        parser.error("Choose a fresh output filename")
    with sqlite3.connect(f"{args.database.resolve().as_uri()}?mode=ro", uri=True) as db:
        row = db.execute(
            "SELECT value_json FROM app_state WHERE key='weekly_planner_snapshot'"
        ).fetchone()
    if row is None:
        raise ValueError("Snapshot missing")
    payload = json.loads(row[0])
    states = [FitnessState.model_validate(x) for x in payload["daily_states"][:args.horizon]]
    options = [
        [FitnessState.model_validate(x) for x in group]
        for group in payload["daily_state_options"][:args.horizon]
    ]
    if len(states) != args.horizon:
        parser.error("Snapshot shorter than requested horizon")
    request = RecommendationRequest.model_validate(payload["request"])
    config = payload["config"]
    target = tuple(payload["target_distance_range"])
    # Keep the exact saved per-day state/forecast. Clip availability inputs to
    # this bounded horizon just as production does.
    forced = set(payload.get("forced_rest_offsets", []))
    completed = set(payload.get("completed_run_offsets", []))
    completed_miles = {
        int(k): v for k, v in payload.get("completed_miles_by_offset", {}).items()
    }
    allowed = set(range(args.horizon)) - forced - completed
    goal = planner.configured_race_goal(config)
    required = {i for i in allowed if goal and states[i].as_of.date() == goal[1]}
    prefix_cache, load_cache, state_cache, joint_cache = {}, {}, {}, {}
    common = dict(
        daily_states=states, daily_state_options=options, request=request,
        config=config, target_distance_range=target,
        completed_miles_by_offset=completed_miles, prefix_cache=prefix_cache,
        recommendation_load_cache=load_cache, projected_state_cache=state_cache,
    )
    started = perf_counter()
    selected = tuple(planner._adaptive_run_day_offsets_for_frequency(
        **common, target_run_count=args.count, horizon_run_count=args.count,
        forced_rest_offsets=forced, completed_run_offsets=completed,
        joint_program_scoring=True, joint_cost_cache=joint_cache,
    ))
    search_seconds = perf_counter() - started
    ordinary = sum(planner.typical_easy_distance(states[0])) / 2
    rows = []
    remaining_count = args.count - len(completed & set(range(args.horizon)))
    for offsets in combinations(sorted(allowed), remaining_count):
        if not required.issubset(offsets):
            continue
        joint = joint_cache.get(offsets)
        if joint is None:
            joint = planner._joint_candidate_program_cost(offsets, **common)
        base = planner._adaptive_candidate_cost(
            offsets, **common,
            include_mileage_path=False, include_recovery_interactions=False,
            include_cadence_pressure=False,
        )
        rows.append({
            "offsets": offsets,
            "total_cost": planner._program_selection_cost(joint, base, ordinary),
            "joint_cost": joint, "coaching_cost": base,
        })
    rows.sort(key=lambda x: (x["total_cost"], x["offsets"]))
    selected_row = next(x for x in rows if x["offsets"] == selected)
    report = {
        "planner_version": planner.WEEKLY_PLANNER_VERSION,
        "snapshot_generated_at": payload.get("generated_at"),
        "captured_at": datetime.now().astimezone().isoformat(),
        "horizon_days": args.horizon, "run_count": args.count,
        "exact_limit": planner.MAX_EXACT_PROGRAM_CANDIDATES,
        "beam_width": planner.MAX_ADAPTIVE_CANDIDATES,
        "finalists": planner.JOINT_DATE_FINALISTS,
        "candidate_count": len(rows), "selected": selected_row,
        "exhaustive_best": rows[0],
        "regret": selected_row["total_cost"] - rows[0]["total_cost"],
        "search_seconds": search_seconds,
        "total_seconds": perf_counter() - started, "ranked_candidates": rows,
        "scope": "Exact date comparison for fixed count; production role/dose allocator; saved states; no future target projector",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "ranked_candidates"}, indent=2))


if __name__ == "__main__":
    main()
