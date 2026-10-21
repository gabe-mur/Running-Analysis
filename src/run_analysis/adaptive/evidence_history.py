"""Read-only original-file evidence loading; no research-driver dependency."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
from run_analysis.db import initialize
from run_analysis.run_feedback import list_runs
from run_analysis.importer import parser_for
from run_analysis.workload_profile import observed_profile
from run_analysis.workload_accounting import observed_workload
from run_analysis.recovery_assessment import SessionEntry
from run_analysis.delivered_training import dose_from_profile, continuous_phase_from_activity
from .program_evidence_adapter import observed_run

def app_session_types(database):
    """Use the production run-list resolution, including explicit overrides."""
    # Old archives may predate optional sensor/feedback tables. Migrate only a
    # disposable in-memory copy, never the athlete's read-only source database.
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True) as source:
        db = sqlite3.connect(':memory:')
        try:
            source.backup(db)
            db.row_factory = sqlite3.Row
            initialize(db)
            count = db.execute('SELECT COUNT(*) FROM activities').fetchone()[0]
            return {str(r.activity_id): r.workout_type.value for r in list_runs(db, limit=count)}
        finally:
            db.close()

def load_sessions(database, config, since, until, *, session_types=None, activity_observer=None,
                  missing_timing_observer=None):
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True) as db:
        rows = db.execute('''SELECT a.id,a.start_time_utc,s.path FROM activities a
            JOIN activity_sources l ON l.activity_id=a.id
            JOIN source_files s ON s.id=l.source_file_id
            WHERE l.is_primary=1 AND a.sport='Running' AND a.start_time_utc >= ?
            AND a.start_time_utc < ? ORDER BY a.start_time_utc,a.id''',
            (since.isoformat(), until.isoformat())).fetchall()
    result = []
    for key, start, source in rows:
        path = Path(source)
        parser = parser_for(path)
        if parser is None:
            raise ValueError(f'Unsupported original format: {path.suffix}')
        parsed = parser(path, default_timezone=config['timezone_default'])
        at = datetime.fromisoformat(start).astimezone(timezone.utc)
        matches = [a for a in parsed.activities if a.start_time_utc == at]
        if len(matches) != 1:
            raise ValueError(f'Original activity mismatch for {key}')
        activity = matches[0]
        if not activity.total_elapsed_time_s or activity.total_elapsed_time_s <= 0:
            if missing_timing_observer is not None:
                missing_timing_observer(activity, str(key), path)
                continue
            raise ValueError('Missing end-time evidence')
        profile = observed_profile(activity, config,
            workout_type=session_types.get(str(key)) if session_types is not None else None)
        entry = SessionEntry(str(key), at, at+timedelta(seconds=activity.total_elapsed_time_s),
                             observed_workload(profile))
        if activity_observer is not None:
            activity_observer(activity, entry, profile)
        result.append((entry, profile))
    if len({x.key for x, _ in result}) != len(result):
        raise ValueError('Multiple primary source records would duplicate work')
    return result

def load_history(database, config, as_of, reference, *, run_transform=None, since=None):
    roles = app_session_types(database)
    rows, unresolved = [], []
    def collect(activity, entry, profile):
        end = max(entry.end, activity.start_time_utc+timedelta(seconds=activity.lap_recorded_time_s or 0),
                  *(p.timestamp_utc for p in activity.trackpoints if p.timestamp_utc is not None))
        if end > as_of:
            unresolved.append(dict(key=entry.key, reason='not_completed'))
            return
        dose = dose_from_profile(profile, key=entry.key, start=entry.start, end=end, available_at=end,
            continuous_phase=continuous_phase_from_activity(activity, config)
            if roles[entry.key] == 'tempo_threshold' else None)
        role = ('quality' if roles[entry.key] in {'intervals', 'tempo_threshold'} else
                'long' if roles[entry.key] == 'long' else
                'easy' if roles[entry.key] in {'easy', 'recovery', 'medium_long'} else 'unknown')
        run = observed_run(profile, dose, role=role, reference=reference, as_of=as_of)
        rows.append(run_transform(run, activity) if run_transform else run)
    load_sessions(database, config, since if since is not None else as_of-timedelta(days=84), as_of, session_types=roles,
        activity_observer=collect, missing_timing_observer=lambda a, key, path:
            unresolved.append(dict(key=key, reason='missing_elapsed_trace')))
    return tuple(rows), unresolved
