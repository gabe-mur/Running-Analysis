from datetime import datetime, timezone
from pathlib import Path
import yaml
import pytest

from fit_builder import build_run
from run_analysis.fit import parse_fit
from run_analysis.tcx import parse_tcx
from test_tcx import make_tcx
from run_analysis.web.schemas import RecommendationResponse, WorkoutStep
from run_analysis.workload_profile import WorkBout, bout_summary, observed_profile, prescribed_profile

CONFIG = yaml.safe_load(Path('config.example.yaml').read_text())


def prescription(reps, minutes, recovery):
    at = datetime(2026, 9, 24, tzinfo=timezone.utc)
    return RecommendationResponse(generated_at=at, fitness_state_as_of=at, planned_for=at,
        workout_type='intervals', title='test', distance_range_miles=(4.,4.5),
        distance_target_miles=4.2, confidence='moderate', readiness='ready',
        structure=[WorkoutStep(instruction='work', phase='work', repetitions=reps,
            work_duration_minutes=minutes, recovery_duration_minutes=recovery, target_zones=['Z4'])])


def test_equal_work_retains_sustained_bout_and_recovery_differences():
    short = prescribed_profile(prescription(6, 2.5, 2))
    long = prescribed_profile(prescription(3, 5, 2))
    a, b = short['intensity_structure'], long['intensity_structure']
    assert a['total_work_seconds'] == b['total_work_seconds'] == 900
    assert a['duration_weighted_work_seconds'] == 150
    assert b['duration_weighted_work_seconds'] == 300
    assert a['between_work_recovery_seconds'] == 600
    assert b['between_work_recovery_seconds'] == 240
    assert short['cardiovascular']['zone_load'] is None
    assert not short['planner_influence']


def test_missing_recovery_is_unknown_not_zero():
    summary = bout_summary([WorkBout(120, None), WorkBout(120, None)])
    assert summary['between_work_recovery_seconds'] is None
    assert summary['work_fraction_of_work_and_between_recovery'] is None


def test_missing_prescribed_work_is_not_invented_from_workout_label():
    rec = prescription(3, 5, 2).model_copy(update={'structure': []})
    profile = prescribed_profile(rec)
    assert profile['intensity_structure']['bout_count'] == 0


def test_observed_profile_works_without_hrv_and_does_not_use_rr(tmp_path):
    path = tmp_path/'run.fit'
    build_run(path, start=datetime(2026,9,24,tzinfo=timezone.utc))
    activity = parse_fit(path).activities[0]
    before = observed_profile(activity, CONFIG)
    activity.sensor_stream = None
    assert observed_profile(activity, CONFIG) == before
    assert before['cardiovascular']['hr_coverage'] == 1.
    assert before['intensity_structure']['bout_count'] == 0
    for point in activity.trackpoints:
        point.heart_rate_bpm = None
    missing = observed_profile(activity, CONFIG)
    assert missing['cardiovascular']['zone_load'] is None
    assert missing['cardiovascular']['hr_coverage'] == 0.
    assert missing['running_exposure'] == before['running_exposure']


def test_invalid_durations_are_rejected():
    with pytest.raises(ValueError):
        bout_summary([WorkBout(float('nan'), None)])
    with pytest.raises(ValueError):
        bout_summary([WorkBout(120, -1)])


def test_tcx_profile_needs_no_sensor_stream(tmp_path):
    activity = parse_tcx(make_tcx(tmp_path/'run.tcx')).activities[0]
    profile = observed_profile(activity, CONFIG)
    assert activity.sensor_stream is None
    assert profile['running_exposure']['distance_meters'] == 40
    assert profile['cardiovascular']['hr_coverage'] == 1.
    assert not profile['planner_influence']


def test_read_only_audit_accepts_tcx(tmp_path):
    from scripts.audit_workload_profile import audit
    profile = audit(make_tcx(tmp_path/'run.tcx'), CONFIG)
    assert profile['activities'][0]['sensor_inventory']['rr_interval_count'] == 0
    assert not profile['planner_influence']
