"""Original-file delivered-dose inventory, not retrospective compliance."""
from argparse import ArgumentParser
from datetime import datetime, timedelta
import json
from pathlib import Path
from time import perf_counter

from scripts.audit_athlete_projection import json_ready
from scripts.audit_workload_contract import load_sessions
from scripts.audit_workload_expectation import app_session_types
from run_analysis.config import load_config
from run_analysis.delivered_training import dose_from_profile, dose_metrics, continuous_phase_from_activity


def audit(database, config, since, as_of):
    started = perf_counter()
    roles = app_session_types(database)
    rows, unresolved = [], []

    def collect(activity, entry, profile):
        end = max(entry.end,
            activity.start_time_utc+timedelta(seconds=activity.lap_recorded_time_s or 0),
            *(p.timestamp_utc for p in activity.trackpoints if p.timestamp_utc is not None))
        if end > as_of:
            unresolved.append(dict(key=entry.key, reason='not_completed_at_cutoff'))
            return
        measured = dose_from_profile(profile, key=entry.key, start=entry.start,
            end=end, available_at=end,
            continuous_phase=(continuous_phase_from_activity(activity, config)
                              if roles[entry.key] == 'tempo_threshold' else None))
        metrics = dose_metrics(measured)
        rows.append(dict(key=entry.key, start=entry.start, role=measured.role,
            metrics=metrics, bouts=measured.bouts, structure_basis=measured.structure_basis,
            issued_contract='not_reconstructed', purpose_evidence='not_reconstructed',
            physiological_equivalence_established=False))

    def missing(activity, key, path):
        unresolved.append(dict(key=key, reason='missing_elapsed_trace',
                               start=activity.start_time_utc))

    load_sessions(database, config, since, as_of, session_types=roles,
                  activity_observer=collect, missing_timing_observer=missing)
    return dict(since=since, as_of=as_of, sessions=rows, unresolved=unresolved,
        elapsed_seconds=perf_counter()-started, planner_influence=False,
        limitations=['Current roles held fixed; original upload availability not reconstructed',
            'Start-time inventory window, not a rolling completion-window training target',
            'No verified issued prescription; no retrospective adherence assertion',
            'Lap/pace contrast is not physiological intensity or verified purpose',
            'Missing structured work is unknown; no new detector or fitted thresholds'])


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('config.yaml'))
    parser.add_argument('--since', type=datetime.fromisoformat, required=True)
    parser.add_argument('--as-of', type=datetime.fromisoformat, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.database, load_config(args.config), args.since, args.as_of),
                     default=json_ready, indent=2))


if __name__ == '__main__':
    main()
