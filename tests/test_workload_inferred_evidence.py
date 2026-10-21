from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from run_analysis.workload_profile import inferred_bout_summary, observed_profile
from run_analysis.workload_expectation import observed_features, forecast, compare_distribution
from run_analysis.workout_scoring import analyze_intervals
from run_analysis.workout_detection import detect_structured_workout, inferred_work_groups
from test_workout_scoring import _interval, _laps_connection
from test_workload_expectation import AT, POLICY, profile, refs, intent
from test_workload_profile import CONFIG


def intervals():
    pattern = [(1.8, 20), (3.5, 7), (1.8, 18)]
    for _ in range(5):
        pattern.extend([(3.5, 23), (1.8, 20)])
    return [_interval(i, 0, speed, 5, 160)
            for i, speed in enumerate(speed for speed, n in pattern for _ in range(n))]


def inferred_profile():
    p = profile(reps=0)
    p['intensity_structure']['session_workout_type'] = 'intervals'
    p['intensity_structure']['pace_inferred_work'] = inferred_bout_summary(intervals())
    return p


def test_reuses_existing_detector_including_extra_short_candidate():
    raw = intervals()
    summary = inferred_bout_summary(raw)
    with _laps_connection() as db:
        existing = analyze_intervals(db, 1, raw)
        assert detect_structured_workout(db, 1, raw, z3_floor=154) is None
    assert summary['bout_count'] == existing.work_repetition_count == 6
    work = [r for r in existing.repetitions if r.kind == 'work']
    assert [b['work_seconds'] for b in summary['bouts']] == [r.duration_seconds for r in work]
    assert summary['bouts'][0]['work_seconds'] == 35
    assert summary['boundary_uncertainty'] == 'unquantified'
    assert not summary['confirmed_quality']


def test_inference_is_separate_and_recorded_control_unchanged():
    p = inferred_profile()
    assert observed_features(p)['work_seconds'] is None
    candidate = observed_features(p, include_inferred_work=True)
    assert candidate['work_seconds'].low == 610
    assert candidate['work_seconds'].basis == 'pace_stream_inference_unquantified'
    assert observed_features(p)['cardio_points'] == candidate['cardio_points']
    assert p['intensity_structure']['bout_count'] == 0


def test_recorded_work_takes_precedence_over_inferred_candidates():
    p = profile()
    p['intensity_structure']['pace_inferred_work'] = inferred_bout_summary(intervals())
    assert observed_features(p, include_inferred_work=True) == observed_features(p)


def test_inference_cannot_become_a_confident_execution_difference():
    p = inferred_profile()
    expected = forecast(intent(), refs(p), AT, POLICY, include_inferred_work=True)
    assert not expected['calibrated']
    assert expected['inferred_support']['work_seconds'] == ['0', '1', '2']
    assert 'unquantified' in expected['expected']['work_seconds'].basis
    before = deepcopy(expected)
    result = compare_distribution(expected, profile(rep_seconds=200), {})
    assert result['features']['work_seconds']['nominal_status'] == 'above_expectation'
    assert result['features']['work_seconds']['status'] == 'inferred_estimate_only'
    assert 'work_seconds' not in result['differences']
    assert not result['adaptation_authorized']
    assert expected == before


def test_future_inferred_session_cannot_enter_forecast():
    from datetime import timedelta
    future = [replace(r, available_at=AT+timedelta(seconds=1)) for r in refs(inferred_profile())]
    result = forecast(intent(), future, AT, POLICY, include_inferred_work=True)
    assert result['work_reference_keys'] == []


def test_recording_gap_candidates_remain_visible_but_not_usable():
    raw = intervals()
    start, end = inferred_work_groups(raw)[1]
    raw[start+3] = replace(raw[start+3], flags=('large_recording_gap',))
    summary = inferred_bout_summary(raw)
    assert summary['bout_count'] == 6
    assert summary['invalid_candidate_indexes']
    assert not summary['usable_for_comparison']
    p = profile(reps=0)
    p['intensity_structure']['session_workout_type'] = 'intervals'
    p['intensity_structure']['pace_inferred_work'] = summary
    assert observed_features(p, include_inferred_work=True)['work_seconds'] is None


def test_no_movement_does_not_invent_work():
    result = inferred_bout_summary([])
    assert result['bout_count'] == 0 and not result['usable_for_comparison']


@pytest.mark.parametrize('role', [None, 'unknown', 'easy', 'long', 'tempo_threshold', 'run_walk', 'intervals'])
def test_profile_reuses_app_session_gate_before_reconstructing_reps(monkeypatch, role):
    raw = intervals()
    activity = SimpleNamespace(trackpoints=[], laps=[],
        total_distance_m=sum(i.distance_m for i in raw),
        lap_recorded_time_s=sum(i.elapsed_s for i in raw),
        total_elapsed_time_s=sum(i.elapsed_s for i in raw))
    monkeypatch.setattr('run_analysis.workload_profile.classify_movement',
                        lambda *args: SimpleNamespace(intervals=raw))
    result = observed_profile(activity, CONFIG, workout_type=role)
    inferred = result['intensity_structure']['pace_inferred_work']
    assert (inferred is not None) == (role in {'intervals', 'run_walk'})
    work = observed_features(result, include_inferred_work=True)['work_seconds']
    assert (work is not None) == (role == 'intervals')
    easy = observed_profile(activity, CONFIG, workout_type='easy')
    assert result['cardiovascular'] == easy['cardiovascular']
    assert result['running_exposure'] == easy['running_exposure']


@pytest.mark.parametrize('role', [None, 'easy', 'long', 'run_walk'])
def test_ineligible_inferred_candidates_cannot_enter_work_reference_pool(role):
    p = inferred_profile()
    p['intensity_structure']['session_workout_type'] = role
    expected = forecast(intent(), refs(p), AT, POLICY, include_inferred_work=True)
    assert expected['work_reference_keys'] == []
    assert expected['expected']['work_seconds'] is None
