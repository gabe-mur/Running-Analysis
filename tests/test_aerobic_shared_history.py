from copy import deepcopy
from datetime import datetime, timedelta, timezone
import math

import numpy as np
import pytest

from scripts.audit_aerobic_shared_history import frame, snapshot, crop, comparisons


def fixture():
    end = datetime(2026, 10, 15, tzinfo=timezone.utc)
    rows = [dict(start_time=(end-timedelta(days=d)).isoformat(), pace_min_mile=10.+.002*d+.03*math.sin(d))
        for d in (115, 105, 95, 80, 70, 60, 55, 45, 35, 27, 17, 7)]
    m = np.full(len(rows), .01**2)
    g = np.linspace(.1, 1., len(rows))[:, None]
    cov = np.array([[.005**2]])
    return rows, m, g, cov, end


def test_frame_respects_exact_calendar_boundaries_and_excludes_future():
    end = datetime(2026, 10, 15, tzinfo=timezone.utc)
    rows = [dict(start_time=(end-timedelta(days=d)).isoformat())
        for d in (118, 117.99, 90, 89.99, 56, 55.99, 28, 27.99, -1)]
    indices, observed, groups = frame(rows, end)
    assert indices.tolist() == list(range(1, 8))
    assert observed == [0, 1, 2, 3]
    assert groups.tolist() == [2, 2, 3, 3, 1, 1, 0]


def test_missing_periods_are_compacted_not_invented():
    rows, m, g, cov, end = fixture()
    indices = np.array([0, 1, 2, 9, 10, 11])
    result = snapshot([rows[i] for i in indices], m[indices], g[indices], cov, end)
    assert result['available'] and result['observed_periods'] == [0, 2]
    assert result['counts'] == [3, 3]
    assert len(result['state']['period_log_pace']) == 2


def test_sparse_or_missing_current_history_does_not_fabricate_a_level():
    rows, m, g, cov, end = fixture()
    for indices in (np.array([11]), np.array([0, 1, 2])):
        result = snapshot([rows[i] for i in indices], m[indices], g[indices], cov, end)
        assert not result['available'] and 'state' not in result


def test_snapshot_keeps_requested_anchor_and_ignores_later_runs_without_mutating_inputs():
    rows, m, g, cov, end = fixture()
    before = deepcopy((rows, m, g, cov))
    original = snapshot(rows, m, g, cov, end)
    assert original['anchor'] == end.isoformat() != rows[-1]['start_time']
    extended = rows+[dict(start_time=(end+timedelta(days=1)).isoformat(), pace_min_mile=8.)]
    later = snapshot(extended, np.r_[m, .01**2], np.vstack([g, [1.]]), cov, end)
    assert later == original
    for current, previous in zip((rows, m, g, cov), before):
        np.testing.assert_equal(current, previous)


def test_crop_is_presentation_only_and_retains_one_predecessor():
    end = datetime(2026, 10, 15, tzinfo=timezone.utc)
    curve = [dict(anchor=(end-timedelta(days=d)).isoformat(), level=d) for d in (100, 80, 30, 10, 0)]
    before = deepcopy(curve)
    assert crop(curve, end, 28) == curve[2:]
    assert crop(curve, end, 90) == curve
    assert crop(curve, end, 180) == curve
    assert curve == before


def test_comparison_attributes_the_entire_log_disagreement_without_adjusting_either_point():
    current = dict(available=True, observed_periods=[0, 1, 2],
        state=dict(period_log_pace=[2.3, 2.35, 2.32]), marginal_period_log_pace=[2.31, 2.34, 2.32])
    previous = dict(available=True, anchor='2026-09-17T00:00:00+00:00',
        state=dict(period_log_pace=[2.33]), marginal_period_log_pace=[2.335])
    before = deepcopy((current, previous))
    result = comparisons(current, previous, 28, 10., 10.)
    assert sum(result['log_pace_attribution'].values()) == pytest.approx(.02)
    assert result['joint_change_percent'] == pytest.approx(100*math.expm1(.05))
    assert result['curve_endpoint_change_percent'] == pytest.approx(100*math.expm1(.03))
    assert result['robust_control_disagreement_seconds_per_mile'] == 0
    assert (current, previous) == before
