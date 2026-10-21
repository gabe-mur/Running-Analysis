"""Read-only descriptive baseline from original files; not a compliance replay."""
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
from run_analysis.demonstrated_training import demonstrated_baseline, UntimedSession
from run_analysis.projection_evidence import evidence_from_activity


def audit(database, config, as_of):
    started = perf_counter()
    roles = app_session_types(database)
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True) as db:
        eligible = {str(key): not illness and health in (None, '', 'normal') and included != 0
            for key, illness, health, included in db.execute('''
                SELECT a.id,o.illness,o.health_tag,o.include_in_model FROM activities a
                LEFT JOIN run_overrides o ON o.activity_id=a.activity_id''')}
    history, untimed = [], []

    def collect(activity, entry, profile):
        available = max(entry.end,
            activity.start_time_utc+timedelta(seconds=activity.lap_recorded_time_s or 0),
            *(p.timestamp_utc for p in activity.trackpoints if p.timestamp_utc is not None))
        history.append(evidence_from_activity(activity, config, key=entry.key,
            available_at=available, workout_type=roles[entry.key], profile=profile,
            capacity_eligible=eligible[entry.key]))

    def missing(activity, key, path):
        untimed.append(UntimedSession(key, activity.start_time_utc, as_of, str(path)))

    # Include one extra day so a session crossing the earliest completion
    # boundary is not lost by the loader's start-time filter.
    load_sessions(database, config, as_of-timedelta(days=85), as_of,
        session_types=roles, activity_observer=collect, missing_timing_observer=missing)
    result = demonstrated_baseline(history, as_of, timezone=config['timezone_default'], untimed=untimed)
    result.update(elapsed_seconds=perf_counter()-started,
        source='original_files_read_only', source_start_lookback_days=85,
        limitations=['Current roles and health flags held fixed',
            'Availability assumed at completion, not reconstructed upload times',
            'No independently asserted completeness or verified ordinary/support intent',
            'No inference of tolerance, sustainable capacity, recovery or recommended growth',
            'Start-filter padding assumes no run lasts longer than one day; not a full archive inventory'])
    return result


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('config.yaml'))
    parser.add_argument('--as-of', type=datetime.fromisoformat, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.database, load_config(args.config), args.as_of),
                     default=json_ready, indent=2))


if __name__ == '__main__':
    main()
