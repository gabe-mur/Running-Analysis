from datetime import timedelta

import pytest

from run_analysis.movement import classify_movement
from run_analysis.tcx import parse_tcx
from scripts.audit_cardio_structure import audit_file, slice_interval, summarize
from test_tcx import make_tcx
from test_workload_profile import CONFIG


def test_clipping_hr_interval_preserves_additive_load(tmp_path):
    activity = parse_tcx(make_tcx(tmp_path/'run.tcx')).activities[0]
    activity.trackpoints[0].heart_rate_bpm = 130
    activity.trackpoints[1].heart_rate_bpm = 180
    item = classify_movement(activity.trackpoints, CONFIG['moving_time']).intervals[0]
    boundary = item.start.timestamp_utc+timedelta(seconds=7.35)
    pieces = [slice_interval(item, item.start.timestamp_utc, boundary),
              slice_interval(item, boundary, item.end.timestamp_utc)]
    before, after = summarize([item], CONFIG['zones']), summarize(pieces, CONFIG['zones'])
    assert before['points'] == pytest.approx(after['points'])
    assert before['zone_minutes'] == pytest.approx(after['zone_minutes'])
    assert before['moving_minutes'] == pytest.approx(after['moving_minutes'])


def test_without_lap_structure_total_still_reconciles(tmp_path):
    report = audit_file(make_tcx(tmp_path/'run.tcx'), CONFIG)
    assert list(report['phases']) == ['unresolved']
    assert report['phases']['unresolved']['points'] == report['total']['points']
    assert not report['planner_influence']
