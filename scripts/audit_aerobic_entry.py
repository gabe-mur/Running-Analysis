"""Original-run HR-band entry audit; no physiological clearance or planner writes."""
from argparse import ArgumentParser
from datetime import datetime, timedelta
from pathlib import Path
from time import perf_counter
import hashlib
import json
import sqlite3

from scripts.audit_workload_contract import load_sessions
from scripts.audit_workload_expectation import app_session_types
from scripts.audit_recency_demand import json_ready
from run_analysis.aerobic_entry import moving_hr_spans, describe_entry
from run_analysis.config import load_config
from run_analysis.demonstrated_training import distribution
from run_analysis.movement import classify_movement


def audit(database, config, as_of):
    started = perf_counter()
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    roles = app_session_types(database)
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True) as db:
        eligible = {str(k): not illness and health in (None, '', 'normal') and included != 0
            for k, illness, health, included in db.execute('''SELECT a.id,o.illness,o.health_tag,o.include_in_model
                FROM activities a LEFT JOIN run_overrides o ON o.activity_id=a.activity_id''')}
    low, high = config['zones']['z2'][0], config['zones']['z3'][0]
    settings = {'5min_80pct': (300, .8, 0), '5min_90pct': (300, .9, 0),
                '3min_80pct': (180, .8, 0), 'band_minus3': (300, .8, -3), 'band_plus3': (300, .8, 3)}
    rows, unresolved = [], []
    def collect(activity, entry, profile):
        movement = classify_movement(activity.trackpoints, config['moving_time'])
        intervals = movement.intervals
        end = max([entry.end, *[i.end.timestamp_utc for i in intervals]])
        if end > as_of:
            unresolved.append(dict(key=entry.key, reason='not_completed_at_snapshot'))
            return
        elapsed = (end-activity.start_time_utc).total_seconds()
        spans = moving_hr_spans(activity, intervals)
        cases = {key: describe_entry(spans, elapsed_seconds=elapsed,
            low_bpm=low+shift, high_bpm_exclusive=high+shift,
            confirmation_seconds=window, required_fraction=fraction)
            for key, (window, fraction, shift) in settings.items()}
        rows.append(dict(key=entry.key, start=activity.start_time_utc, end=end,
            role=roles[entry.key], eligible=eligible[entry.key],
            miles=profile['running_exposure']['distance_meters']/1609.344,
            moving_minutes=sum(i.moving_time_s for i in intervals)/60,
            elapsed_minutes=elapsed/60, cases=cases))
    load_sessions(database, config, as_of-timedelta(days=84), as_of,
        session_types=roles, activity_observer=collect,
        missing_timing_observer=lambda a, key, path: unresolved.append(dict(key=key, reason='missing_timing')))
    summary = {}
    for days in (28, 84):
        groups = {}
        for name, included in [('easy', {'easy', 'recovery', 'medium_long'}), ('long', {'long'}),
                               ('quality_context_only', {'intervals', 'tempo_threshold', 'race'})]:
            selected = [r for r in rows if r['eligible'] and r['role'] in included and r['end'] > as_of-timedelta(days=days)]
            variants = {}
            for key in settings:
                values = [r['cases'][key] for r in selected]
                uncensored = [v for v in values if v['first_observed_hr_offset_seconds'] is not None
                    and v['first_observed_hr_offset_seconds'] <= 1
                    and not v['already_at_or_above_band_on_first_observation']]
                variants[key] = dict(qualifying_count=sum(v['sustained_entry_seconds'] is not None for v in values),
                    first_entry_minutes=distribution(v['first_entry_seconds']/60 for v in values if v['first_entry_seconds'] is not None),
                    sustained_entry_minutes=distribution(v['sustained_entry_seconds']/60 for v in values if v['sustained_entry_seconds'] is not None),
                    uncensored_sustained_entry_minutes=distribution(v['sustained_entry_seconds']/60 for v in uncensored if v['sustained_entry_seconds'] is not None),
                    zone_minutes=distribution(v['zone_seconds']/60 for v in values),
                    no_qualifying_keys=[r['key'] for r in selected if r['cases'][key]['sustained_entry_seconds'] is None])
            groups[name] = dict(count=len(selected), source_keys=[r['key'] for r in selected], variants=variants)
        summary[str(days)] = groups
    compact = [dict(r, cases={key: {name: v[name] for name in (
        'first_entry_seconds', 'sustained_entry_seconds', 'zone_seconds',
        'known_moving_hr_seconds', 'first_observed_hr_offset_seconds',
        'already_at_or_above_band_on_first_observation')} for key, v in r['cases'].items()}) for r in rows]
    return dict(as_of=as_of, original_sessions=len(rows), zone_2=config['zones']['z2'],
        source_database_bytes_unchanged=before == hashlib.sha256(database.read_bytes()).hexdigest(),
        sessions=compact, unresolved=unresolved, summaries=summary, elapsed_seconds=perf_counter()-started,
        planner_influence=False, limitations=[
            'Current zones/roles/health held fixed; availability assumed at completion',
            'Elapsed since recording start, not time since all pre-recording warm-up began',
            'HR-band entry does not establish tissue readiness or minimum effective training dose',
            'Confirmation settings and shifted bands are diagnostics, not fitted training rules',
            'One-second candidate-start resolution; exact piecewise-linear band-duration accounting',
            'No sustained window is not no exercise benefit; all original load remains accounted',
            'Quality sessions are context only, not donors for easy-run entry allowance'])


if __name__ == '__main__':
    p = ArgumentParser(description=__doc__)
    p.add_argument('--database', type=Path, required=True)
    p.add_argument('--config', type=Path, default=Path('config.yaml'))
    p.add_argument('--as-of', type=datetime.fromisoformat, required=True)
    args = p.parse_args()
    print(json.dumps(audit(args.database, load_config(args.config), args.as_of), default=json_ready, allow_nan=False, indent=2))
