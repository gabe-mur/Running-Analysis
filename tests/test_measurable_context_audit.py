import json

from scripts.audit_measurable_context import known, sensor_coverage, wellness_coverage


def test_known_does_not_turn_missing_or_bool_into_measurement():
    assert known(0)
    assert not known(None)
    assert not known(True)
    assert not known(float('nan'))


def test_sensor_coverage_deduplicates_physical_runs_and_requires_values():
    payload = dict(rr_batches=[dict(intervals_seconds=[None, .5])],
        performance_records=[dict(fields={'power': {'value': 180}, 'respiration_rate': {'value': None}})],
        session_fields={'workout_rpe': 0, 'workout_feel': None})
    rows = [dict(activity_id=7, payload_json=json.dumps(payload))] * 2
    result = sensor_coverage(rows)
    assert result['exercise_rr'] == {7}
    assert result['record:power'] == {7}
    assert result['session:workout_rpe'] == {7}
    assert 'record:respiration_rate' not in result
    assert 'session:workout_feel' not in result


def test_invalid_rr_is_not_available_exercise_hrv():
    result = sensor_coverage([dict(activity_id=1,
        payload_json=json.dumps({'rr_batches': [{'intervals_seconds': [None, 0, -1]}]}))])
    assert 'exercise_rr' not in result


def test_wellness_counts_reported_days_not_runs_or_interpolated_nights():
    rows = [dict(date='2026-07-06', sleep_seconds=20000),
            dict(date='2026-07-07', hrv_ms=50)]
    runs = [dict(id=1, start_time_local='2026-07-06T18:00:00-04:00'),
            dict(id=2, start_time_local='2026-07-06T20:00:00-04:00'),
            dict(id=3, start_time_local='2026-07-08T18:00:00-04:00')]
    result = wellness_coverage(rows, runs)
    assert result['sleep_seconds']['reported_days'] == 1
    assert result['sleep_seconds']['same_date_run_ids'] == [1, 2]
    assert result['hrv_ms']['same_date_run_ids'] == []
    assert 'not verified pre-run' in result['sleep_seconds']['attribution']
