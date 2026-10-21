"""Read-only HR load history; smoothing is not a recovery/fitness prediction.

All curves use the same continuous HR-reserve integral as observed planner
cardio accounting. Unit-gain daily EWMAs include recorded rest dates; a display
window never truncates their input. No planner state or database is changed.
"""
from collections import OrderedDict
from datetime import datetime, time, timedelta
import json
from math import exp
from statistics import mean, median
from threading import RLock
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from .continuous_training_dose import ContinuousDoseReference
from .movement import classify_movement
from .processing import _load_points
from .training_signals import HeartRateReference, observed_cardio_trace
from .derived_load_cache import read_traces,store_traces,trace_key
from .read_cache import _version


# Pure per-activity results only. Import changes, source changes and HR/movement
# settings invalidate the key. Never cache connection objects or whole responses.
_TRACE_CACHE = OrderedDict()
_CACHE_LIMIT = 512
_CACHE_LOCK = RLock()


def recorded_cardio(connection, config, end):
    reference = ContinuousDoseReference('Progress read-only recorded HR load')
    hr = HeartRateReference(config['resting_hr'], config['max_hr'], 'current configured HR references')
    zone = ZoneInfo(config['timezone_default'])
    database = connection.execute('PRAGMA database_list').fetchone()[2]
    version=_version(connection)
    disk=read_traces(database) if database and version is not None else {}
    pending={}
    rows = []
    for raw in connection.execute('''SELECT a.*,
        (SELECT f.sha256 FROM activity_sources s JOIN source_files f ON f.id=s.source_file_id
         WHERE s.activity_id=a.id AND s.is_primary=1) source_hash
        FROM activities a WHERE a.sport='Running' AND a.start_time_utc_epoch<=?
        ORDER BY a.start_time_utc_epoch,a.id''', (end.timestamp(),)).fetchall():
        start = datetime.fromisoformat(raw['start_time_utc'])
        key = (database, raw['id'], raw['updated_at_utc'], raw['source_hash'],
            raw['start_time_utc'], raw['total_elapsed_time_s'], raw['lap_recorded_time_s'],
            raw['trackpoint_count'], hr.resting_bpm, hr.maximum_bpm, reference.accounting_key,
            json.dumps(config['moving_time'], sort_keys=True))
        with _CACHE_LOCK:
            trace = _TRACE_CACHE.get(key) if database else None
            if trace is not None:
                _TRACE_CACHE.move_to_end(key)
        disk_key=trace_key(key) if database and raw['source_hash'] and version is not None else None
        if trace is None and disk_key is not None:
            trace=disk.get(disk_key)
        if trace is None:
            points = _load_points(connection, raw['id'])
            activity = SimpleNamespace(start_time_utc=start, trackpoints=points,
                total_elapsed_time_s=raw['total_elapsed_time_s'], lap_recorded_time_s=raw['lap_recorded_time_s'])
            intervals = classify_movement(points, config['moving_time']).intervals
            trace = observed_cardio_trace(activity, intervals, hr, reference.policy, compute_density=False)
            if disk_key is not None:
                pending[disk_key]=trace
            if database:
                with _CACHE_LOCK:
                    _TRACE_CACHE[key] = trace
                    _TRACE_CACHE.move_to_end(key)
                    while len(_TRACE_CACHE) > _CACHE_LIMIT:
                        _TRACE_CACHE.popitem(last=False)
        # An activity still in progress must not contribute future recorded HR.
        if start + timedelta(seconds=trace['elapsed_seconds']) > end:
            continue
        bounds = trace['load_bounds_within_reference']
        rows.append(dict(activity_id=raw['id'], start_time=start.isoformat(),
            day=start.astimezone(zone).date(),
            low=bounds['low'] if bounds and trace['elapsed_seconds'] > 0 else None,
            high=bounds['high'] if bounds and trace['elapsed_seconds'] > 0 else None,
            covered_seconds=trace['covered_seconds'], elapsed_seconds=trace['elapsed_seconds']))
    if version is not None and _version(connection)==version:
        store_traces(database,pending)
    return rows


def ewma(rows, day, memory_days):
    """Unit-gain average: equal daily doses converge to that same daily dose.

Missing HR remains bounded unknown, not zero. Unsupported references are not
silently clipped or assigned a finite upper limit. Before-run callers pass the
previous local date, never the run's own completed load.
"""
    history = [r for r in rows if r['day'] <= day]
    if not history:
        return dict(complete=False, value=None, low=None, high=None)
    initialized = min(r['day'] for r in history) <= day-timedelta(days=5*memory_days)
    supported = all(r['low'] is not None and r['high'] is not None for r in history)
    if not initialized or not supported:
        return dict(complete=False, value=None, low=None, high=None)
    gain = 1-exp(-1/memory_days)
    low = sum(gain*exp(-(day-r['day']).days/memory_days)*r['low'] for r in history)
    high = sum(gain*exp(-(day-r['day']).days/memory_days)*r['high'] for r in history)
    return dict(complete=True, value=(low+high)/2, low=low, high=high)


def training_load_history(rows, sessions, points, details, end, window_days, timezone):
    zone = ZoneInfo(timezone)
    today = end.astimezone(zone).date()
    daily = []
    for offset in range(window_days+1):
        day = today-timedelta(days=window_days-offset)
        daily.append(dict(as_of=min(datetime.combine(day, time(12), zone), end).isoformat(),
            short=ewma(rows, day, 2), recent=ewma(rows, day, 7), baseline=ewma(rows, day, 42)))
    current = daily[-1]
    recent, baseline = current['recent'], current['baseline']
    ratio = recent['value']/baseline['value'] if recent['complete'] and baseline['complete'] and baseline['value'] > 0 else None
    ratio_bounds = dict(low=recent['low']/baseline['high'], high=recent['high']/baseline['low']) if ratio is not None and baseline['low'] > 0 else None
    runs = []
    for s in sessions:
        p = points.get(s.activity_id)
        if not end-timedelta(days=42) < s.start_time <= end or not p or not p.included_in_trend:
            continue
        if details[s.activity_id]['health_tag'] != 'normal' or p.standardized_pace_min_mile is None:
            continue
        before = ewma(rows, s.start_time.astimezone(zone).date()-timedelta(days=1), 7)
        if before['complete']:
            runs.append(dict(activity_id=s.activity_id, start_time=s.start_time.isoformat(),
                adjusted_pace=p.standardized_pace_min_mile, uncertainty_95=p.uncertainty_95_min_mile,
                load=before['value'], low=before['low'], high=before['high']))
    comparison = None
    if len(runs) >= 6:
        boundary = median(r['load'] for r in runs)
        lower = [r for r in runs if r['load'] < boundary]
        higher = [r for r in runs if r['load'] > boundary]
        if len(lower) >= 3 and len(higher) >= 3:
            comparison = dict(lower_runs=len(lower), higher_runs=len(higher),
                difference_seconds=60*(mean(r['adjusted_pace'] for r in higher)-mean(r['adjusted_pace'] for r in lower)))
    recent_rows = [r for r in rows if today-timedelta(days=42) < r['day'] <= today]
    elapsed = sum(r['elapsed_seconds'] for r in recent_rows)
    return dict(status='available' if ratio is not None else 'incomplete',
        reason=None if ratio is not None else 'More recorded history or valid HR accounting is needed for the longer-term baseline.',
        daily=daily, current=current, ratio=ratio, ratio_bounds=ratio_bounds,
        runs=runs, comparison=comparison, comparison_days=42,
        hr_coverage=sum(r['covered_seconds'] for r in recent_rows)/elapsed if elapsed else None,
        recorded_runs=len(rows), history_start=min((r['start_time'] for r in rows), default=None),
        unit='HR-reserve equivalent minutes per day', memory_days=dict(short=2, recent=7, baseline=42),
        initialization='Zero at first recorded run; at least five time constants before display',
        source='Recorded runs under current HR references; not Garmin EPOC or measured fatigue')
