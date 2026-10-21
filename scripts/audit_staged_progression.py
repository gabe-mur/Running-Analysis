"""Read-only provisional seed and arithmetic sensitivity; no scheduler replay."""
from argparse import ArgumentParser
from datetime import datetime, timedelta
import json
from pathlib import Path
import sqlite3
from time import perf_counter

from scripts.audit_athlete_projection import json_ready
from scripts.audit_workload_contract import load_sessions
from scripts.audit_workload_expectation import app_session_types
from run_analysis.config import load_config
from run_analysis.projection_evidence import evidence_from_activity
from run_analysis.staged_progression import bootstrap_stage, ProgressionPolicy, MILE, tenth


def original_history(database, config, as_of):
    """Read-only original-file evidence shared by bounded offline audits."""
    roles = app_session_types(database)
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True) as db:
        eligible = {str(key): not illness and health in (None, '', 'normal') and included != 0
            for key, illness, health, included in db.execute('''
                SELECT a.id,o.illness,o.health_tag,o.include_in_model FROM activities a
                LEFT JOIN run_overrides o ON o.activity_id=a.activity_id''')}
    history, unresolved = [], []

    def collect(activity, entry, profile):
        end = max(entry.end,
            activity.start_time_utc+timedelta(seconds=activity.lap_recorded_time_s or 0),
            *(p.timestamp_utc for p in activity.trackpoints if p.timestamp_utc is not None))
        history.append(evidence_from_activity(activity, config, key=entry.key,
            available_at=end, workout_type=roles[entry.key], profile=profile,
            capacity_eligible=eligible[entry.key]))

    def missing(activity, key, path):
        unresolved.append(dict(key=key, start=activity.start_time_utc, source=str(path)))

    load_sessions(database, config, as_of-timedelta(days=85), as_of,
        session_types=roles, activity_observer=collect, missing_timing_observer=missing)
    return history, unresolved


def audit(database, config, as_of):
    started = perf_counter()
    history, unresolved = original_history(database, config, as_of)
    if unresolved:
        return dict(status='unresolved_history_requires_review', unresolved=unresolved, planner_influence=False)
    seed = bootstrap_stage(history, as_of, goal_long_miles=17., policy=ProgressionPolicy(14, .05))
    sensitivity = []
    if seed['stage'] is not None:
        s = seed['stage']
        prior = [x for x in history if as_of-timedelta(days=30) < x.end <= as_of and x.available_at <= as_of]
        cap = tenth(max(x.meters for x in prior)/MILE*1.1, down=True)
        for days in (7, 14, 21):
            for fraction in (.025, .05, .075):
                # Sensitivity arithmetic only; no invented post-seed completions,
                # no claim that a future review passed or that this is schedulable.
                long = min(s.goal_long_miles, tenth(s.long_miles*(1+fraction)), cap)
                total = tenth(s.weekly_equivalent_miles*(1+fraction))
                nonlong = (total-s.weekly_equivalent_miles-(long-s.long_miles)*7/s.observed_long_gap_days
                           if s.observed_long_gap_days else None)
                sensitivity.append(dict(review_days=days, growth_fraction=fraction,
                    earliest_review=as_of+timedelta(days=days), hypothetical_long_miles=long,
                    hypothetical_total_miles_per_7_days=total,
                    hypothetical_extra_nonlong_miles_per_7_days=nonlong,
                    long_guard_cap_at_seed_only=cap, admissibility_established=False))
    return dict(seed=seed, sensitivity=sensitivity, original_sessions=len(history),
        elapsed_seconds=perf_counter()-started, planner_influence=False,
        limitations=['Current health/settings/roles held fixed, availability assumed at completion',
            'No historically issued stage or response ledger reconstructed',
            'Sensitivity is arithmetic, not a future training schedule or recovered-state forecast',
            'No preferred growth fraction or review interval selected from this audit'])


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('config.yaml'))
    parser.add_argument('--as-of', type=datetime.fromisoformat, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.database, load_config(args.config), args.as_of), default=json_ready, indent=2))


if __name__ == '__main__':
    main()
