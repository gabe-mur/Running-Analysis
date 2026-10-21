"""Resolve source-specific self reports without copying them into user overrides.

FIT session workout_rpe is CR10 multiplied by ten; workout_feel is a separate
0–100 bad-to-good scale (FIT profile fields 193 and 192). Neither is readiness.
"""
from math import isfinite


def _rating(value, scale=1):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not isfinite(value) or not 0 <= value <= 100:
        return None
    return float(value) / scale


def feedback_from_session_fields(fields):
    """Decode a standalone FIT with the same scales used by DB resolution.

    No manual override exists on a standalone file. DB callers must instead
    pass their source-resolved feedback so this cannot replace manual effort.
    """
    effort = _rating(fields.get('workout_rpe'), 10)
    feeling = _rating(fields.get('workout_feel'))
    return dict(perceived_exertion=effort, workout_feel=feeling,
        effort_source='fit_session' if effort is not None else None,
        feeling_source='fit_session' if feeling is not None else None,
        feedback_conflict=False)


def with_session_feedback(connection, rows):
    """Bulk resolve ratings for API and planner consumers; manual effort wins.

    Prefer canonical source, then newest source. Keep provenance and conflicts
    visible. Missing/invalid fields never erase a valid rating from another source.
    No writes, and no decoding of potentially large RR arrays in Python.
    """
    result = [dict(row) for row in rows]
    if not result:
        return result
    ids = sorted({row['id'] for row in result})
    sources = []
    # The run-list endpoint can request thousands of activities; keep below
    # SQLite builds with the older 999-parameter limit.
    for start in range(0, len(ids), 500):
        batch = ids[start:start+500]
        sources.extend(connection.execute(f"""SELECT r.activity_id,r.source_file_id,
        json_extract(r.payload_json,'$.session_fields.workout_rpe') AS rpe,
        json_extract(r.payload_json,'$.session_fields.workout_feel') AS feel
        ,f.imported_at_utc AS feedback_imported_at
        FROM activity_sensor_streams r JOIN activity_sources s
        ON s.source_file_id=r.source_file_id
        AND s.source_activity_index=r.source_activity_index
        JOIN source_files f ON f.id=r.source_file_id
        WHERE r.activity_id IN ({','.join('?' for _ in batch)})
        ORDER BY s.is_primary DESC,r.source_file_id DESC,r.source_activity_index
        """, batch).fetchall())
    by_id = {}
    for source in sources:
        by_id.setdefault(source['activity_id'], []).append(source)
    for row in result:
        row['effort_source'] = 'manual' if row.get('perceived_exertion') is not None else None
        row['workout_feel'] = None
        row['feeling_source'] = None
        row['feedback_conflict'] = False
        # FIT fields belong to immutable imported bytes. They do not become a
        # new self report whenever an application consumer reads them. Manual
        # overrides have no immutable report journal and retain unknown timing.
        row['feedback_source_imported_at'] = []
        row['feedback_time_known'] = row['effort_source'] != 'manual'
        for key, field, scale, provenance in (
            ('perceived_exertion', 'rpe', 10, 'effort_source'),
            ('workout_feel', 'feel', 1, 'feeling_source'),
        ):
            values = [(value, source['source_file_id'], source['feedback_imported_at'])
                      for source in by_id.get(row['id'], [])
                      if (value := _rating(source[field], scale)) is not None]
            conflict = len({value for value, _, _ in values}) > 1
            if conflict:
                row['feedback_conflict'] = True
            if row.get(key) is None and values:
                row[key] = values[0][0]
                row[provenance] = f'fit:{values[0][1]}'
                row['feedback_source_imported_at'].extend(
                    value[2] for value in (values if conflict else values[:1]))
        row['feedback_time_known'] = (row['feedback_time_known'] and
                                     bool(row['feedback_source_imported_at']))
    return result
