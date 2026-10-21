from datetime import date, datetime, timedelta, timezone
import math

import pytest

from scripts.audit_measured_performance_context import (
    check_family, entering_load, matched_routes, route_overlap, sleep_context,
)


def test_sleep_uses_reported_date_and_never_future_or_missing_nights():
    series = {'2026-07-05': {'sleep_seconds': 8*3600},
              '2026-07-06': {'sleep_seconds': 7*3600, 'hrv_ms': 50},
              '2026-07-07': {'sleep_seconds': 15*3600}}
    value = sleep_context(series, date(2026, 7, 6))
    assert value['sleep_hours'] == 7
    assert value['previous_date_sleep_hours'] == 8
    assert value['sleep_three_nights_hours'] is None
    assert value['daily_hrv'] == 50
    assert value['daily_resting_hr'] is None


def test_three_night_sleep_requires_all_three_and_preserves_zero():
    series = {f'2026-07-0{i}': {'sleep_seconds': 3600*i} for i in (4, 5, 6)}
    assert sleep_context(series, date(2026, 7, 6))['sleep_three_nights_hours'] == 5


def history_row(local_date, minutes=30, low=30, high=30):
    return dict(local_date=local_date, end=local_date+'T23:00:00+00:00',
                moving_minutes=minutes, cardio_low=low, cardio_high=high)


def test_entering_load_excludes_same_date_and_future():
    at = datetime(2026, 7, 20, 18, tzinfo=timezone.utc)
    history = [history_row('2026-07-01'), history_row('2026-07-19'),
               history_row('2026-07-20', minutes=999), history_row('2026-07-21', minutes=999)]
    value = entering_load(history, at)
    assert value['entering_running_minutes'] == pytest.approx(30*(math.exp(-19/2)+math.exp(-1/2)))
    assert value['entering_cardio'] == pytest.approx(value['entering_running_minutes'])


def test_unsupported_recent_hr_is_missing_not_zero_load():
    at = datetime(2026, 7, 20, 18, tzinfo=timezone.utc)
    value = entering_load([history_row('2026-07-01'), history_row('2026-07-19', low=None, high=None)], at)
    assert value['entering_running_minutes'] > 0
    assert value['entering_cardio'] is None


def test_broad_missing_hr_bounds_do_not_produce_precise_fatigue():
    at = datetime(2026, 7, 20, 18, tzinfo=timezone.utc)
    value = entering_load([history_row('2026-07-01'), history_row('2026-07-19', low=0, high=100)], at)
    assert value['entering_cardio'] is None


def test_route_overlap_requires_both_route_directions_not_just_subset():
    assert route_overlap(set(), {(1, 1)}) is None
    assert route_overlap({(0, 0)}, {(0, 0), (100, 100)}) == .5
    assert route_overlap({(0, 0)}, {(1, 1)}) == 1


def test_matched_routes_do_not_choose_on_pace_or_sleep_and_never_reuse_runs():
    start = datetime(2026, 7, 1, tzinfo=timezone.utc)
    rows = [dict(activity_id=i, start_time_utc=(start+timedelta(days=i)).isoformat(),
        device='same', workout_type='easy', sampled_hr_bpm=148, heat_c=10,
        duration_minutes=30, headwind_mph=0, pace=10, sleep_hours=8, entering_cardio=20)
        for i in range(4)]
    routes = {i: {(0, 0), (1, 1)} for i in range(4)}
    first = matched_routes(rows, routes)
    for i, r in enumerate(rows):
        r.update(pace=100-i*10, sleep_hours=i)
    second = matched_routes(rows, routes)
    ids = lambda x: [(p['earlier_id'], p['later_id']) for p in x['pairs']]
    assert ids(first) == ids(second)
    assert len({i for pair in ids(second) for i in pair}) == 2*len(second['pairs'])


def test_sparse_family_is_descriptive_not_regression():
    rows = [dict(activity_id=1, start_time_utc='2026-07-01T00:00:00+00:00', sleep_hours=8)]
    result = check_family(rows, ('sleep_hours',))
    assert result['runs'] == 1
    assert result['status'].startswith('descriptive_only')
    assert 'association' not in result


def test_two_calendar_blocks_cannot_claim_narrow_uncertainty():
    start = datetime(2026, 7, 1, tzinfo=timezone.utc)
    rows = [dict(activity_id=i, start_time_utc=(start+timedelta(days=3*i)).isoformat(),
        sleep_hours=6+i%4, pace=10+.01*i) for i in range(12)]
    result = check_family(rows, ('sleep_hours',), baseline=())
    assert result['calendar_blocks'] <= 2
    assert result['association']['conditional_block_range_95'] is None
    assert 'range_unavailable_reason' in result['association']
