"""Inventory locally measurable run context, without fitting or live changes.

Coverage is counted per physical activity, not per duplicate file or trackpoint.
Same-date wellness is only a retrospective join: the API does not establish
when a value was measured or available. No medication inference is attempted.
"""
from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import sqlite3

from run_analysis.config import load_config, resolve_project_path
from run_analysis.privacy import private_file
from run_analysis.progress import prepare_progress_data
from run_analysis.progress_wip import _canonical_load
from run_analysis.read_cache import invariant_snapshot
from run_analysis.wellness_signals import read_wellness


def known(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def sensor_coverage(rows):
    coverage = defaultdict(set)
    for row in rows:
        identity = row['activity_id']
        payload = json.loads(row['payload_json'])
        if any(known(v) and v > 0 for b in payload.get('rr_batches', [])
               for v in b.get('intervals_seconds', [])):
            coverage['exercise_rr'].add(identity)
        for record in payload.get('performance_records', []):
            for name, field in record.get('fields', {}).items():
                if known(field.get('value')):
                    coverage['record:' + name].add(identity)
        for name, field in payload.get('performance_session_fields', {}).items():
            if known(field.get('value')):
                coverage['summary:' + name].add(identity)
        for name, value in payload.get('session_fields', {}).items():
            if known(value):
                coverage['session:' + name].add(identity)
    return dict(coverage)


def wellness_coverage(series, activities):
    by_date = {r['date']: r for r in series}
    fields = sorted({key for r in series for key in r if key != 'date'})
    return {field: dict(
        reported_days=sum(known(r.get(field)) for r in series),
        same_date_run_ids=sorted(a['id'] for a in activities
            if known(by_date.get(a['start_time_local'][:10], {}).get(field))),
        attribution='Reported calendar date only; not verified pre-run availability or night timing')
        for field in fields}


def scan_fit_sources(connection, root):
    """Inspect field names/coverage only; never export routes or sensor serials."""
    import fitdecode
    from run_analysis.fit import _open
    fields, messages = defaultdict(set), defaultdict(set)
    seen, failures, multiple = set(), [], set()
    links = {r[0]: r[1] for r in connection.execute(
        'SELECT source_file_id,count(DISTINCT activity_id) FROM activity_sources GROUP BY source_file_id')}
    for row in connection.execute('''SELECT f.id,f.path,s.activity_id FROM source_files f
        JOIN activity_sources s ON s.source_file_id=f.id ORDER BY f.id'''):
        path = resolve_project_path(root, row['path'])
        if not path.name.lower().endswith(('.fit', '.fit.gz')):
            continue
        if links[row['id']] != 1:
            multiple.add(row['id'])
            continue
        key = (row['id'], row['activity_id'])
        if key in seen:
            continue
        seen.add(key)
        try:
            with fitdecode.FitReader(_open(path)) as reader:
                for frame in reader:
                    if frame.frame_type != fitdecode.FIT_FRAME_DATA:
                        continue
                    messages[frame.name].add(row['activity_id'])
                    for field in frame.fields:
                        if field.value is not None:
                            fields[frame.name + ':' + field.name].add(row['activity_id'])
        except (OSError, ValueError, fitdecode.exceptions.FitError) as error:
            failures.append(dict(source_id=row['id'], error=type(error).__name__))
    return dict(files_activity_links=len(seen), failures=failures,
        multi_activity_files_not_attributed=sorted(multiple),
        message_activity_ids={key: sorted(value) for key, value in sorted(messages.items())},
        field_activity_ids={key: sorted(value) for key, value in sorted(fields.items())},
        interpretation='Source field presence, not validated measurements; no raw values exported')


def audit(connection, root, *, scan_fit=False):
    activities = [dict(r) for r in connection.execute('''SELECT a.id,a.start_time_local,
        a.start_time_utc,a.sport,m.moving_pace_min_mile,m.moving_average_hr_bpm,
        m.session_zone_load,m.hr_load_coverage,m.stop_fraction,m.analysis_distance_m,
        m.previous_7d_miles,m.previous_28d_miles,m.days_since_previous_run,
        m.days_since_previous_hard_run FROM activities a JOIN activity_metrics m
        ON m.activity_id=a.id ORDER BY a.start_time_utc''')]
    runs = [r for r in activities if r['sport'].lower() in {'run', 'running'}]
    recent = [r for r in runs if r['start_time_local'] >= '2026-03-01']
    ids = {r['id'] for r in recent}
    sensors = sensor_coverage(connection.execute('SELECT activity_id,payload_json FROM activity_sensor_streams'))
    wellness = read_wellness(root) or {}
    weather = [dict(r) for r in connection.execute('SELECT * FROM activity_weather') if r['activity_id'] in ids]
    coverage = {key: dict(physical_runs=len(value), recent_runs=len(value & ids),
                         activity_ids=sorted(value)) for key, value in sorted(sensors.items())}
    point_fields = {}
    for field in ('heart_rate_bpm', 'gps_valid', 'altitude_m', 'cadence', 'speed_mps'):
        condition = 'gps_valid=1' if field == 'gps_valid' else field + ' IS NOT NULL'
        present = {r[0] for r in connection.execute(f'SELECT DISTINCT activity_id FROM trackpoints WHERE {condition}')}
        point_fields[field] = dict(recent_runs=len(present & ids))
    prepared = prepare_progress_data(connection)
    ledger = _canonical_load(connection, root, prepared.sessions, datetime.now(timezone.utc))
    model = ledger.get('fatigue_model')
    ledger_result = {key: ledger.get(key) for key in ('status', 'reason', 'snapshot_as_of')}
    ledger_result.update(verified_activity_ids=sorted(r['activity_id'] for r in ledger['rows']), model=model)
    if model:
        # Conservative coverage screen matching the display: five e-folds of
        # observed history plus no unmatched activity in that lookback interval.
        found = {r['activity_id'] for r in ledger['rows'] if r['accounting_key'] == model['accounting_key']}
        lead = math.ceil(5 * model['scenario']['memory_days'])
        first = datetime.fromisoformat(model['history_start'])
        missing = [datetime.fromisoformat(r['start_time_utc']) for r in runs if r['id'] not in found]
        complete = []
        for row in recent:
            at = datetime.fromisoformat(row['start_time_utc'])
            if first <= at-timedelta(days=lead) and not any(at-timedelta(days=lead) <= t < at for t in missing):
                complete.append(row['id'])
        ledger_result['conservative_complete_recent_run_ids'] = complete
    return dict(version='measurable-context-inventory-v1', source_database_read_only=True,
        live_changed=False, planner_called=False, imported_activities=len(activities),
        recent_running_activities=len(recent), recent_start='2026-03-01',
        trackpoint_coverage=point_fields, sensor_coverage=coverage,
        activity_metric_coverage={key: sum(known(r.get(key)) for r in recent)
            for key in recent[0] if key not in {'id', 'start_time_local', 'start_time_utc', 'sport'}} if recent else {},
        weather_coverage={key: sum(known(r.get(key)) for r in weather)
            for key in ('temperature_f', 'dewpoint_f', 'wind_speed_mph', 'wind_gust_mph',
                        'precipitation_in', 'surface_pressure_hpa')},
        wellness=dict(retrieved_at=wellness.get('retrieved_at'),
            coverage=wellness_coverage(wellness.get('series', []), recent)),
        health_reports=connection.execute('SELECT count(*) FROM user_health_reports').fetchone()[0],
        committed_fatigue_coverage=ledger_result,
        fit_source_fields=scan_fit_sources(connection, root) if scan_fit else None,
        limitations=[
            'Coverage is not accuracy, causality, or independent predictive value.',
            'Exercise HRV and daily HRV are separate measurements; neither is treated as measured fatigue.',
            'Power/cadence/stride metrics and post-run reports may be consequences of pace; do not use them as independent causal corrections.',
            'No per-day medication times, hydration, nutrition, alcohol, shoe or strength-session log is inferred.',
            'Committed fatigue history is not extended backward by assuming missing evidence means zero fatigue.',
            'Current extraction coverage can be lower than source-file coverage; optional source inspection discloses this.',
            'Sleep/resting-HR/HRV date joins are retrospective only, not proof of pre-run availability.'
        ])


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scan-fit', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output path')
    root = Path.cwd()
    config = load_config(root/'config.yaml')
    database = resolve_project_path(root, config['paths']['database'])
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True) as connection:
        connection.row_factory = sqlite3.Row
        with invariant_snapshot(connection):
            result = audit(connection, root, scan_fit=args.scan_fit)
            assert connection.total_changes == 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        json.dump(result, output, indent=2, allow_nan=False)
    private_file(args.output)
    print(json.dumps({key: result[key] for key in ('imported_activities', 'recent_running_activities',
        'trackpoint_coverage', 'weather_coverage', 'wellness', 'health_reports')}, indent=2))
    print('Saved private inventory:', args.output)


if __name__ == '__main__':
    main()
