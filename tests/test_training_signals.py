from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json

import pytest

from run_analysis.models import ActivitySensorStream, RRBatch
from run_analysis.movement import classify_movement
from run_analysis.recovery_response import ResponseExpectation
from run_analysis.response_model import Policy, Phase, dose, rate
from run_analysis.session_feedback import feedback_from_session_fields
from run_analysis.signal_structure import SignalSpan
from run_analysis.tcx import parse_tcx
from run_analysis.training_signals import (
    HeartRateReference, integrate_cardio_trace, observed_cardio_trace, observed_training_signals,
)
from run_analysis.workload_accounting import Quantity
from test_tcx import make_tcx
from test_workload_profile import CONFIG


def test_existing_numerical_probe_uses_the_shared_implementation():
    from scripts.numerical_response_mapping_probe import checks, dose as probe_dose
    assert probe_dose is dose
    assert len(checks()[0]) == 26


def test_observed_and_supplied_projected_trace_share_phase_accounting():
    phases = [Phase(10, .6, 1), Phase(5, .9, 1.4), Phase(3, .5, .8)]
    start, spans = 0., []
    for phase in phases:
        end = start+phase.minutes*60
        spans.append(SignalSpan(start, end, phase.hr_reserve, phase.hr_reserve))
        start = end
    expected = dose(phases)
    result = integrate_cardio_trace(spans, start)
    assert result['total_load'] == pytest.approx(expected.cardio)
    assert result['response_load_candidate'] == pytest.approx(expected.response_cardio)
    assert result['density_peak_candidate'] == pytest.approx(expected.density_peak)


@pytest.mark.parametrize('exponent', [0., 1.67, 1.92])
def test_linear_hr_integration_matches_independent_fine_grid(exponent):
    policy = replace(Policy(), intensity_exponent=exponent)
    result = integrate_cardio_trace([SignalSpan(0, 60, 0., 1.)], 60, policy)
    expected = sum(rate((i+.5)/20000, policy) for i in range(20000))/20000
    assert result['total_load'] == pytest.approx(expected, rel=1e-8)
    if exponent:
        assert result['total_load'] > result['mean_hr_shortcut_load']


def test_sampling_invariance_and_density_convergence():
    whole = integrate_cardio_trace([SignalSpan(0, 120, .4, .95)], 120)
    split = integrate_cardio_trace([SignalSpan(0, 47, .4, .4+.55*47/120),
                                   SignalSpan(47, 120, .4+.55*47/120, .95)], 120)
    fine = integrate_cardio_trace([SignalSpan(0, 120, .4, .95)], 120, density_step_seconds=.1)
    assert split['total_load'] == pytest.approx(whole['total_load'], rel=1e-9)
    assert split['response_load_candidate'] == pytest.approx(whole['response_load_candidate'])
    assert fine['response_load_candidate'] == pytest.approx(whole['response_load_candidate'], rel=1e-5)


def test_missing_span_is_not_zero_load_or_density_recovery():
    result = integrate_cardio_trace([SignalSpan(0, 30, .6, .6), SignalSpan(40, 60, .8, .8)], 60)
    assert result['unknown_seconds'] == 10
    assert result['observed_load'] > 0
    assert result['total_load'] is result['response_load_candidate'] is None
    assert result['load_bounds_within_reference']['low'] == result['observed_load']
    assert result['load_bounds_within_reference']['high'] == pytest.approx(
        result['observed_load']+10/60*rate(1))
    assert integrate_cardio_trace([], 60)['observed_load'] is None


@pytest.mark.parametrize('spans,total', [
    ([SignalSpan(0, 30, .6, .6), SignalSpan(20, 60, .6, .6)], 60),
    ([SignalSpan(-1, 30, .6, .6)], 60),
    ([SignalSpan(0, 90, .6, .6)], 60),
    ([SignalSpan(0, 30, .6, 1.1)], 60),
])
def test_invalid_trace_rejected(spans, total):
    with pytest.raises(ValueError):
        integrate_cardio_trace(spans, total)


def test_valid_observed_stopped_hr_counts_but_gap_does_not(tmp_path):
    activity = parse_tcx(make_tcx(tmp_path/'run.tcx')).activities[0]
    for point in activity.trackpoints:
        point.distance_m = 0
        point.speed_mps = 0
    intervals = classify_movement(activity.trackpoints, CONFIG['moving_time']).intervals
    assert intervals[0].moving_time_s == 0
    reference = HeartRateReference(50, 200, 'test')
    result = observed_cardio_trace(activity, intervals, reference)
    assert result['total_load'] > 0
    intervals[0].flags.append('large_recording_gap')
    gap = observed_cardio_trace(activity, intervals, reference)
    assert gap['unknown_seconds'] == 20
    assert gap['observed_load'] is None


def test_above_max_hr_remains_reviewable_not_clipped(tmp_path):
    activity = parse_tcx(make_tcx(tmp_path/'run.tcx')).activities[0]
    activity.trackpoints[-1].heart_rate_bpm = 220
    report = observed_training_signals(activity, CONFIG, workout_type='easy')
    assert report['cardiovascular']['exclusions']['outside_reference_seconds'] == 20
    assert report['cardiovascular']['total_load'] is None
    assert report['cardiovascular']['load_bounds_within_reference'] is None
    assert activity.trackpoints[-1].heart_rate_bpm == 220


def test_hrv_optional_and_does_not_change_demand_or_response(tmp_path):
    activity = parse_tcx(make_tcx(tmp_path/'run.tcx')).activities[0]
    before = observed_training_signals(activity, CONFIG, workout_type='easy')
    activity.sensor_stream = ActivitySensorStream(rr_batches=[RRBatch(1, (.8, None, .7))])
    after = observed_training_signals(activity, CONFIG, workout_type='easy')
    assert before['cardiovascular'] == after['cardiovascular']
    assert before['response'] == after['response']
    assert after['hrv']['rr_interval_count'] == 2
    assert not after['hrv']['model_influence']


def test_fit_feedback_is_not_soreness_or_inferred_expectation(tmp_path):
    activity = parse_tcx(make_tcx(tmp_path/'run.tcx')).activities[0]
    activity.sensor_stream = ActivitySensorStream(session_fields={'workout_rpe': 40, 'workout_feel': 75})
    report = observed_training_signals(activity, CONFIG, workout_type='intervals')
    response = report['response']
    assert response['feedback']['perceived_exertion'] == 4
    assert response['feedback']['workout_feel'] == 75
    assert response['leg_fatigue'] is None
    assert response['expected_comparison']['channels']['effort']['status'] == 'unknown'
    assert not response['cardiac_drift']['valid']
    assert not report['planner_influence']
    json.dumps(report, allow_nan=False)


def test_resolved_manual_effort_wins_and_issued_expectation_is_used(tmp_path):
    activity = parse_tcx(make_tcx(tmp_path/'run.tcx')).activities[0]
    activity.sensor_stream = ActivitySensorStream(session_fields={'workout_rpe': 40})
    expected = ResponseExpectation('issued_before_run', effort=Quantity(3, 5))
    report = observed_training_signals(activity, CONFIG, workout_type='easy',
        feedback=dict(perceived_exertion=8, effort_source='manual'), expected_response=expected)
    assert report['response']['feedback']['effort_source'] == 'manual'
    assert report['response']['expected_comparison']['review_needed']
    assert report['response']['expected_comparison']['channels']['effort']['status'] == 'more_demanding_than_expected'
    assert not report['response']['expected_comparison']['automatic_progression']


@pytest.mark.parametrize('value', [True, float('nan'), -1, 101, '40'])
def test_invalid_feedback_does_not_become_zero(value):
    assert feedback_from_session_fields({'workout_rpe': value})['perceived_exertion'] is None


def test_steady_drift_is_gated_by_coverage_and_app_role(tmp_path):
    activity = parse_tcx(make_tcx(tmp_path/'run.tcx')).activities[0]
    origin = datetime(2026, 9, 25, tzinfo=timezone.utc)
    template = activity.trackpoints[0]
    activity.trackpoints = [replace(template, timestamp_utc=origin+timedelta(seconds=s),
        distance_m=s*2, heart_rate_bpm=140+s/600, point_index=s//10) for s in range(0, 2401, 10)]
    activity.start_time_utc = origin
    activity.total_elapsed_time_s = activity.lap_recorded_time_s = 2400
    activity.total_distance_m = 4800
    valid = observed_training_signals(activity, CONFIG, workout_type='easy')
    assert valid['response']['cardiac_drift']['valid']
    quality = observed_training_signals(activity, CONFIG, workout_type='intervals')
    assert not quality['response']['cardiac_drift']['valid']
    activity.trackpoints[10].heart_rate_bpm = None
    partial = observed_training_signals(activity, CONFIG, workout_type='easy')
    assert not partial['response']['cardiac_drift']['valid']
    assert partial['cardiovascular']['total_load'] is None


def test_bare_pace_variation_does_not_create_intervals(tmp_path):
    activity = parse_tcx(make_tcx(tmp_path/'run.tcx')).activities[0]
    report = observed_training_signals(activity, CONFIG, workout_type='easy')
    assert report['external_and_structural_evidence']['pace_inferred_work'] is None
    assert report['response']['interval_response'] is None
    assert report['fast_exposure_score'] is None


def test_real_upload_bridge_resolves_feedback_and_audit_leaves_db_unchanged(tmp_path):
    from hashlib import sha256
    from run_analysis.config import load_config
    from scripts.audit_training_signals import audit
    from test_web_phase1 import _write_config
    from test_zip_feedback import archive, rated_fit, upload
    _write_config(tmp_path)
    zip_path = tmp_path/'source.zip'
    zip_path.write_bytes(archive([('export/run.fit', rated_fit())]))
    upload(tmp_path, 'source.zip', zip_path.read_bytes())
    database = tmp_path/'data/test.sqlite'
    before = sha256(database.read_bytes()).hexdigest()
    report = audit(database, load_config(tmp_path/'config.yaml'),
                   dates=('2024-07-01',), fit_zip=zip_path)
    assert sha256(database.read_bytes()).hexdigest() == before
    signal = report['sessions'][0]['signals']
    assert signal['response']['feedback']['perceived_exertion'] == 4
    assert signal['response']['feedback']['effort_source'].startswith('fit:')
    assert signal['response']['feedback']['workout_feel'] == 75
    assert signal['cardiovascular'] == report['additional_fit_sources'][0]['signals']['cardiovascular']
    assert not signal['performance_context']['additional_load_charge']
    assert not signal['planner_influence']


def test_bridge_rejects_mismatched_activity(tmp_path):
    from types import SimpleNamespace
    from run_analysis.training_signals import training_signals_from_run_feedback
    activity = parse_tcx(make_tcx(tmp_path/'run.tcx')).activities[0]
    feedback = SimpleNamespace(run=SimpleNamespace(start_time=activity.start_time_utc+timedelta(days=1)))
    with pytest.raises(ValueError, match='same start'):
        training_signals_from_run_feedback(activity, CONFIG, feedback)
