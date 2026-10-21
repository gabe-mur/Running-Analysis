"""Recorded running-zone boundaries; no inference from a calculation method.

Garmin FIT time_in_zone (216) can include six upper boundaries: below-Z1,
then Z1 through Z5. Session-scoped reports identify the settings used for this
activity. Lap/split totals are not another settings update. HR normalization
remains owned by the adaptive journal. Current watch settings are one bundle,
not imported zones combined with an unrelated manual HR reference.
"""
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
import json


@lru_cache(maxsize=32)
def _read_original(path, digest):
    """Read old extraction versions without mutating importer/application state."""
    from .fit import parse_fit
    parsed = parse_fit(path)
    if len(parsed.activities) != 1:
        return None
    activity = parsed.activities[0]
    if activity.start_time_utc is None:
        return None
    return activity.start_time_utc, json.dumps(activity.sensor_stream.watch_hr, sort_keys=True)


def extract_watch_hr(reports, *, sport, session_count):
    """Retain raw settings, accept only one unambiguous running configuration."""
    if sport != 'running' or session_count != 1:
        return dict(status='unsupported_activity', reports=reports)
    distinct = {json.dumps(r, sort_keys=True): r for r in reports}
    if not distinct:
        return dict(status='not_recorded', reports=[])
    if len(distinct) != 1:
        return dict(status='conflicting_reports', reports=reports)
    report = next(iter(distinct.values()))
    if report.get('reference_index') != 0:
        return dict(status='unsupported_session_index', reports=reports)
    edges = report.get('hr_zone_high_boundary')
    if (not isinstance(edges, (list, tuple)) or len(edges) != 6
            or any(type(v) is not int or not 25 <= v <= 250 for v in edges)
            or any(b <= a for a, b in zip(edges, edges[1:]))):
        return dict(status='incomplete_boundaries', reports=reports)
    maximum = report.get('max_heart_rate')
    resting = report.get('resting_heart_rate')
    if (type(maximum) is not int or not edges[-1] <= maximum <= 250
            or type(resting) is not int or not 25 <= resting < edges[0]):
        return dict(status='incomplete_reference', reports=reports)
    # Integer bpm: the preceding zone ends at its high boundary. No default
    # 50/60/...% scheme, rounded percentages, or guessed first-zone floor.
    zones = {f'z{i+1}': [edges[i]+1, edges[i+1]] for i in range(5)}
    return dict(status='recorded', zones=zones, maximum_hr=maximum,
        resting_hr=resting, calculation=report.get('hr_calc_type'),
        threshold_hr=report.get('threshold_heart_rate'), reports=reports)


def latest_watch_zones(connection, *, as_of=None):
    """Newest measured running settings, independent of upload/filename order.

    Incomplete newer reports never erase a usable record. Same-time conflicting
    exports are unresolved rather than selected by filename or database ID.
    All duplicate source streams are considered; canonical TCX selection must
    not hide the original FIT's settings.
    """
    if connection is None:
        return None
    as_of = as_of or datetime.now(timezone.utc)
    candidates = []
    latest = None
    for row in connection.execute('''SELECT a.start_time_utc,s.sha256,s.path,r.payload_json,r.extraction_version
        FROM activity_sources l JOIN activities a ON a.id=l.activity_id
        JOIN source_files s ON s.id=l.source_file_id LEFT JOIN activity_sensor_streams r
        ON r.source_file_id=s.id AND r.source_activity_index=l.source_activity_index
        WHERE a.sport='Running' AND s.parse_status IN ('ok','warning')
        AND (LOWER(s.path) LIKE '%.fit' OR LOWER(s.path) LIKE '%.fit.gz')
        ORDER BY a.start_time_utc DESC'''):
        if not row['start_time_utc']:
            continue
        at = datetime.fromisoformat(row['start_time_utc'])
        if at.tzinfo is None or at > as_of:
            continue
        if latest is not None and at < latest:
            break
        if row['extraction_version'] is None or row['extraction_version'] < 2:
            from .importer import sha256_file
            try:
                if sha256_file(Path(row['path'])) != row['sha256']:
                    raise ValueError('Original watch-settings source changed; no settings imported')
                original = _read_original(row['path'], row['sha256'])
            except OSError:
                continue
            except ValueError:
                raise
            except Exception as exc:
                raise ValueError('Could not read recorded watch settings; existing settings were retained') from exc
            if original is None or original[0] != at:
                continue
            watch = json.loads(original[1])
        else:
            watch = json.loads(row['payload_json']).get('watch_hr', {})
        if watch.get('status') != 'recorded':
            continue
        # Revalidate stored evidence at the consumption boundary too.
        validated = extract_watch_hr(watch.get('reports', []), sport='running', session_count=1)
        if validated != watch:
            raise ValueError('Stored watch-zone evidence changed; no settings imported')
        latest = at
        candidates.append(dict(watch, recorded_at=at.isoformat(), source_sha256=row['sha256']))
    if not candidates:
        return None
    identities = {json.dumps({k: r[k] for k in ('zones','maximum_hr','resting_hr','calculation')},sort_keys=True)
                  for r in candidates}
    if len(identities) != 1:
        raise ValueError('Conflicting watch settings for the same run; no settings imported')
    return min(candidates, key=lambda r:r['source_sha256'])


def watch_zone_patch(config, connection, *, as_of=None, required=False):
    """Watch mode owns max/resting HR AND zones, with one dated provenance."""
    if (config.get('setup') or {}).get('zone_method') != 'device':
        return {}
    watch = latest_watch_zones(connection, as_of=as_of)
    if watch is None:
        if required:
            raise ValueError('Import a FIT run with recorded watch HR settings, or choose another heart-rate method.')
        return {}
    previous = config.get('watch_zone_import') or {}
    if previous.get('recorded_at') and datetime.fromisoformat(watch['recorded_at']) < datetime.fromisoformat(previous['recorded_at']):
        if required:
            raise ValueError('The latest imported watch settings are older than the saved record. Import the current FIT before switching to watch HR.')
        return {}
    return dict(max_hr=watch['maximum_hr'], resting_hr=watch['resting_hr'],
        **({'profile': {'max_hr_source': 'watch'}} if config.get('profile') else {}),
        zones=watch['zones'], watch_zone_import={k:watch[k] for k in (
        'recorded_at','source_sha256','maximum_hr','resting_hr','calculation','threshold_hr')})
