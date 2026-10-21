from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from scripts.speed_duration_reference import ObservedSpeedCurve, WINDOWS, fit_frontier, freeze_frontier

AT = datetime(2026, 9, 26, tzinfo=timezone.utc)


def row(key='a', *, slope=3., intercept=120., days=1):
    end = AT-timedelta(days=days)
    return ObservedSpeedCurve(key, end-timedelta(hours=1), end, end,
        tuple((t, slope+intercept/t) for t in WINDOWS))


def test_known_model_recovered_without_claiming_threshold_or_confidence():
    result = freeze_frontier([row()], AT)
    fit = result['fit']
    assert fit['slope_mps'] == pytest.approx(3.)
    assert fit['intercept_meters'] == pytest.approx(120.)
    assert fit['r_squared'] == pytest.approx(1.)
    assert fit['distinct_anchor_sessions'] == 1
    assert not result['selection_ready']
    assert not result['physiological_threshold_established']
    assert result['contributor_deletions']['a']['status'] == 'incomplete_duration_support'


def test_deleting_fastest_session_can_raise_slope_not_increase_capacity():
    fast = row('fast', slope=3., intercept=200.)
    slower = row('slower', slope=3.1, intercept=100.)
    assert all(v > w for (_, v), (_, w) in zip(fast.points, slower.points))
    result = freeze_frontier([fast, slower], AT)
    assert result['fit']['slope_mps'] == pytest.approx(3.)
    assert result['contributor_deletions']['fast']['slope_mps'] == pytest.approx(3.1)
    assert not result['selection_ready']


def test_future_incomplete_late_available_and_old_runs_cannot_leak():
    valid = row()
    future = row('future', slope=10, days=-1)
    late = replace(row('late', slope=10), available_at=AT+timedelta(hours=1))
    old = row('old', slope=10, days=43)
    incomplete = replace(row('incomplete', days=0), end=AT+timedelta(hours=1),
                         available_at=AT+timedelta(hours=1))
    assert freeze_frontier([valid, future, late, old, incomplete], AT) == freeze_frontier([valid], AT)


def test_input_order_and_exact_ties_do_not_change_reference():
    rows = [row('b'), row('a')]
    assert freeze_frontier(rows, AT) == freeze_frontier(reversed(rows), AT)
    assert all(a['source_key'] == 'a' for a in freeze_frontier(rows, AT)['fit']['anchors'])


def test_speed_rescaling_preserves_dimensionless_shape():
    ordinary, fast = fit_frontier([row()]), fit_frontier([row(slope=6, intercept=240)])
    assert fast['slope_mps'] == pytest.approx(2*ordinary['slope_mps'])
    assert fast['intercept_meters'] == pytest.approx(2*ordinary['intercept_meters'])
    assert fast['r_squared'] == pytest.approx(ordinary['r_squared'])


def test_steady_easy_only_perfect_fit_does_not_establish_critical_speed():
    result = freeze_frontier([row(intercept=0)], AT)
    assert result['fit']['r_squared'] == pytest.approx(1.)
    assert result['fit']['status'] == 'incompatible_or_uninformative_shape'
    assert not result['selection_ready']


def test_missing_duration_not_filled_from_other_repetitions():
    partial = replace(row(), points=((180., 4.),))
    assert fit_frontier([partial])['status'] == 'incomplete_duration_support'


@pytest.mark.parametrize('points', [((180., float('nan')),), ((180., -1.),),
    ((180., 3.), (180., 4.)), ((4000., 3.),)])
def test_invalid_curve_rejected(points):
    with pytest.raises(ValueError):
        replace(row(), points=points)


def test_duplicate_sessions_rejected():
    with pytest.raises(ValueError):
        freeze_frontier([row(), row()], AT)
