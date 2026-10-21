"""Display-only activity context, separate from aerobic scoring and HR load."""
from .read_cache import invariant_read

VERSION = 'progress-chart-context-v1'
QUALITY_ROLES = {'intervals', 'tempo_threshold', 'race'}


def cached_chart_context(connection):
    return invariant_read(connection,(VERSION,),lambda:build_chart_context(connection))


def build_chart_context(connection):
    from .progress import prepare_progress_data
    prepared=prepare_progress_data(connection)
    rows=connection.execute('''SELECT a.id,o.include_in_model,
        COALESCE(o.workout_type,m.detected_workout_type,'unknown') workout_type,
        m.exclusion_reason,m.moving_average_hr_bpm
        FROM activities a LEFT JOIN activity_metrics m ON m.activity_id=a.id
        LEFT JOIN run_overrides o ON o.activity_id=a.activity_id''').fetchall()
    manual={r['id'] for r in rows if r['include_in_model']==0}
    non_running={r['id'] for r in rows if r['workout_type'] in {'hike','bike'} or any(
        flag in (r['exclusion_reason'] or '') for flag in
        ('probable_walk_or_hike_sensor_signature','probable_bike_sensor_signature'))}
    hidden=manual|non_running;by_id={r['id']:r for r in rows};quality=[]
    for s in prepared.sessions:
        role=prepared.details[s.activity_id]['workout'].value
        if s.activity_id in hidden or role not in QUALITY_ROLES:
            continue
        quality.append(dict(activity_id=s.activity_id,as_of=s.start_time.isoformat(),workout_type=role,
            distance_miles=s.distance_miles,observed_pace_min_mile=s.moving_minutes/s.distance_miles if s.distance_miles>0 else None,
            observed_average_hr_bpm=by_id[s.activity_id]['moving_average_hr_bpm'],
            included_in_trend=False,trend_weight=0.))
    return dict(version=VERSION,manually_excluded_activity_ids=sorted(manual),
        hidden_activity_ids=sorted(hidden),quality_sessions=quality)
