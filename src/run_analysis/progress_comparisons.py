"""Descriptive matched-history comparisons, not estimates of training adaptation.

Grouping uses workout structure, never the size/direction of a pace change.
Medians describe sessions (not individual repetitions), with observed ranges;
these ranges are not confidence intervals. Policy tolerances live here and are
documented in progress_wip_prototype.md, not silently fitted to one athlete.
"""
from datetime import datetime, timedelta
from statistics import median
import math

RECENT_DAYS = 42
MIN_SESSIONS = 2
DISTANCE_TOLERANCE = .10
DURATION_TOLERANCE = .20
RECOVERY_TOLERANCE = .25
HR_TOLERANCE = 5
TEMPERATURE_TOLERANCE = 10
DEWPOINT_TOLERANCE = 10
GRADE_TOLERANCE = .5


def _date(value):
    return datetime.fromisoformat(value) if isinstance(value, str) else value


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def _near(a, b, fraction):
    return _finite(a) and _finite(b) and a > 0 and b > 0 and abs(a-b) <= fraction*b


def _context_matches(a, b):
    """Screen recorded conditions; do not pretend this removes all confounding."""
    a, b = a.get('context', {}), b.get('context', {})
    return all(_finite(a.get(k)) and _finite(b.get(k)) and abs(a[k]-b[k]) <= tolerance
        for k, tolerance in (('temperature_f', TEMPERATURE_TOLERANCE),
            ('dewpoint_f', DEWPOINT_TOLERANCE), ('mean_abs_grade_percent', GRADE_TOLERANCE)))


def _summary(rows, field):
    values = [r[field] for r in rows if _finite(r.get(field))]
    return dict(count=len(values), median=median(values) if values else None,
        low=min(values) if values else None, high=max(values) if values else None,
        activity_ids=[r['activity_id'] for r in rows],
        first=min((r['start_time'] for r in rows), default=None),
        last=max((r['start_time'] for r in rows), default=None))


def _compare(rows, field, end, reference=None):
    boundary = end-timedelta(days=RECENT_DAYS)
    recent = [r for r in rows if boundary < _date(r['start_time']) <= end]
    earlier = [r for r in rows if end-timedelta(days=180) < _date(r['start_time']) <= boundary]
    anchor = reference or max(recent, key=lambda r: r['start_time'], default=None)
    raw = dict(recent=_summary(recent, field), earlier=_summary(earlier, field))
    screened = dict(recent=_summary([r for r in recent if anchor and _context_matches(r, anchor)], field),
        earlier=_summary([r for r in earlier if anchor and _context_matches(r, anchor)], field))
    # A single session may be shown as a recording, never called a typical level.
    supported = all(screened[k]['count'] >= MIN_SESSIONS for k in ('recent', 'earlier'))
    observed = bool(raw['recent']['count'] and raw['earlier']['count'])
    selected = screened if supported else raw
    delta = (selected['recent']['median']-selected['earlier']['median']) if observed else None
    return dict(**selected, delta=delta,
        status='matched_observations' if supported else 'different_conditions_or_sparse' if observed else 'insufficient_history',
        screened=screened, raw=raw, recent_days=RECENT_DAYS,
        minimum_sessions_per_period=MIN_SESSIONS)


def _interval_observation(q):
    interval = q.get('interval') or {}
    work = [r for r in interval.get('repetitions', []) if r['kind']=='work'
        and _finite(r.get('distance_miles')) and r['distance_miles'] > 0]
    recovery = [r for r in interval.get('repetitions', []) if r['kind']=='recovery']
    if not interval.get('available') or len(work) < 2:
        return None
    seconds = sum(r['duration_seconds'] for r in work)
    miles = sum(r['distance_miles'] for r in work)
    recoveries = [r['duration_seconds'] for r in recovery if r['duration_seconds'] > 0]
    recovery_paces = [r['pace_min_mile'] for r in recovery if _finite(r.get('pace_min_mile'))]
    return dict(activity_id=q['activity_id'], start_time=q['start_time'], context=q.get('context', {}),
        pace=interval['mean_work_pace_min_mile'] if _finite(interval.get('mean_work_pace_min_mile')) else seconds/60/miles,
        repetitions=len(work), distance=median(r['distance_miles'] for r in work),
        duration=median(r['duration_seconds'] for r in work),
        recovery=median(recoveries) if recoveries else None,
        recovery_pace=median(recovery_paces) if recovery_paces else None)


def _same_repeats(a, b):
    return (_near(a['distance'], b['distance'], DISTANCE_TOLERANCE)
        and abs(a['repetitions']-b['repetitions']) <= 1
        and _near(a['recovery'], b['recovery'], RECOVERY_TOLERANCE)
        and _near(a['recovery_pace'], b['recovery_pace'], RECOVERY_TOLERANCE))


def quality_comparisons(quality, end):
    quality = [q for q in quality if end-timedelta(days=180) < _date(q['start_time']) <= end
        and q.get('health_tag', 'normal')=='normal']
    groups = []
    observations = sorted(filter(None, (_interval_observation(q) for q in quality)), key=lambda r:r['start_time'])
    for row in observations:
        group = next((g for g in groups if _same_repeats(row, g[0])), None)
        if group is None:
            groups.append([row])
        else:
            group.append(row)
    repeats = [dict(label=f"About {round(g[0]['distance']*1609.344/50)*50} m repeats",
        rep_count=g[0]['repetitions'], total_sessions=len(g),
        comparison=_compare(g, 'pace', end)) for g in groups]
    # Most-supported family first; never select the largest apparent improvement.
    repeats.sort(key=lambda g: (-g['total_sessions'], g['label']))
    sustained = []
    for q in quality:
        p = q.get('continuous')
        if p and _finite(p.get('pace_min_mile')) and _finite(p.get('average_hr_bpm')):
            sustained.append(dict(activity_id=q['activity_id'], start_time=q['start_time'],
                context=q.get('context', {}), pace=p['pace_min_mile'], duration=p['duration_seconds'],
                hr=p['average_hr_bpm'], structure=p['structure']))
    anchor = max(sustained, key=lambda r:r['start_time'], default=None)
    comparable = [r for r in sustained if anchor and r['structure']==anchor['structure']
        and _near(r['duration'], anchor['duration'], DURATION_TOLERANCE)
        and abs(r['hr']-anchor['hr']) <= HR_TOLERANCE]
    return dict(repeats=repeats, sustained=dict(total_sessions=len(sustained),
        reference_minutes=anchor['duration']/60 if anchor else None,
        reference_hr=anchor['hr'] if anchor else None,
        comparison=_compare(comparable, 'pace', end, anchor)))


def endurance_comparisons(durability, end):
    durability = [r for r in durability if end-timedelta(days=180) < _date(r['start_time']) <= end]
    profiles = []
    for low, high in ((35,60),(60,90),(90,math.inf)):
        rows = [r for r in durability if r.get('available') and _finite(r.get('decoupling_percent'))
            and low <= r['moving_minutes'] < high and r.get('health_tag', 'normal')=='normal']
        anchor = max(rows, key=lambda r:r['start_time'], default=None)
        if not anchor:
            continue
        # Bands choose representative profiles, not eligibility boundaries:
        # e.g. an 89-minute run can support the 90+-minute profile.
        similar = [r for r in durability if r.get('available')
            and r.get('health_tag', 'normal')=='normal' and _finite(r.get('decoupling_percent'))
            and _near(r['moving_minutes'], anchor['moving_minutes'], DURATION_TOLERANCE)
            and _finite(r.get('first_hr')) and abs(r['first_hr']-anchor['first_hr']) <= HR_TOLERANCE]
        profiles.append(dict(label=f"Around {round(anchor['moving_minutes']/5)*5} minutes",
            reference_minutes=anchor['moving_minutes'], reference_hr=anchor['first_hr'],
            total_sessions=len(similar), comparison=_compare(similar, 'decoupling_percent', end, anchor)))
    # Avoid presenting substantially overlapping duration profiles as separate
    # evidence: retain the longer exposure rather than the nicer-looking result.
    distinct = []
    for profile in sorted(profiles, key=lambda p:-p['reference_minutes']):
        if not any(_near(profile['reference_minutes'], p['reference_minutes'], DURATION_TOLERANCE)
                   for p in distinct):
            distinct.append(profile)
    return list(reversed(distinct))
