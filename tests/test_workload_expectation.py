from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from run_analysis.workload_accounting import Quantity
from run_analysis.workload_expectation import (
    ForecastPolicy, Intent, Reference, compare_distribution, forecast,
    intent_from_prescription,
)
from scripts.audit_workload_expectation import aligned_intents
from test_workload_profile import CONFIG, prescription


AT = datetime(2026, 9, 24, tzinfo=timezone.utc)
POLICY = ForecastPolicy(1.35, 2., 3, .98, 3., .1)


def profile(distance=7400., *, reps=6, rep_distance=400., rep_seconds=105.,
            minutes=45., hr=170., coverage=1., extra_unobserved=0.):
    seconds = minutes*60
    return dict(source='observed', limitations=[],
        running_exposure=dict(distance_meters=distance, moving_seconds=seconds,
            moving_seconds_range=(seconds, seconds+extra_unobserved)),
        cardiovascular=dict(hr_coverage=coverage,
            zone_seconds=dict(z2=seconds*coverage, unknown=seconds*(1-coverage))),
        intensity_structure=dict(bout_count=reps, total_work_seconds=reps*rep_seconds,
            longest_work_seconds=rep_seconds if reps else 0.,
            bouts=[dict(distance_meters=rep_distance) for _ in range(reps)],
            recorded_signals=dict(
                hr_duration_curve=[dict(window_seconds=300, peak_mean_bpm=hr)],
                speed_duration_curve=[dict(window_seconds=60, peak_mean_mps=4.)],
                coverage=dict(hr_seconds=seconds*coverage, speed_seconds=seconds))))


def intent():
    return Intent(Quantity(7100, 7800), None, None, None,
        work_distance_meters=Quantity.point(2400, 'chosen_6x400'),
        longest_work_distance_meters=Quantity.point(400, 'chosen_400m'))


def refs(p=None, count=3):
    return [Reference(str(i), AT-timedelta(days=i+1), deepcopy(p or profile()))
            for i in range(count)]


def test_forecast_is_frozen_before_actual_and_excludes_future():
    references = refs()+[Reference('target', AT+timedelta(hours=1), profile(hr=199))]
    result = forecast(intent(), references, AT, POLICY)
    before = deepcopy(result)
    assert result['selected_reference_keys'] == ['0', '1', '2']
    assert 'target' not in result['work_reference_keys']
    assert result['rejected'] == [dict(key='target', reason='not_yet_available')]
    actual = profile(hr=190)
    actual_before = deepcopy(actual)
    compared = compare_distribution(result, actual, {})
    assert compared['features']['hr_300s']['status'] == 'above_expectation'
    assert not compared['adaptation_authorized']
    assert not compared['planner_influence']
    assert result == before and actual == actual_before


def test_shorter_whole_run_still_supplies_matching_work_pace_only():
    result = forecast(intent(), refs(profile(distance=4150)), AT, POLICY)
    assert result['selected_reference_keys'] == []
    assert result['work_reference_keys'] == ['0', '1', '2']
    assert not result['meets_session_count_setting'] and result['meets_work_count_setting']
    assert result['expected']['work_seconds'].low == 630
    assert result['expected']['longest_work_seconds'].low == 105
    assert result['expected'].get('cardio_points') is None
    assert result['expected'].get('hr_300s') is None
    assert result['expected'].get('speed_60s') is None


def test_one_session_is_not_six_independent_references():
    result = forecast(intent(), refs(count=1), AT, POLICY)
    assert result['support']['work_seconds'] == ['0']
    assert result['expected']['work_seconds'] is None
    assert not result['meets_work_count_setting']
    row = compare_distribution(result, profile(), {})['features']['work_seconds']
    assert row['status'] == 'unknown'


def test_missing_lap_bouts_do_not_prove_zero_work():
    result = forecast(intent(), refs(profile(reps=0)), AT, POLICY)
    assert result['work_reference_keys'] == []
    compared = compare_distribution(result, profile(reps=0), {})
    assert compared['features']['work_seconds']['status'] == 'unknown'


def test_unknown_hr_does_not_discard_numeric_work_pace():
    result = forecast(intent(), refs(profile(coverage=0)), AT, POLICY)
    assert result['work_reference_keys'] == ['0', '1', '2']
    assert result['selected_reference_keys'] == []
    assert result['expected']['work_seconds'].low == 630
    assert result['expected'].get('cardio_points') is None


def test_no_label_or_generated_workout_dependency():
    references = refs()
    expected = forecast(intent(), references, AT, POLICY)
    for ref in references:
        ref.profile['title'] = 'rest / fartlek / long / intervals'
        ref.profile['workout_type'] = 'rest'
    assert forecast(intent(), references, AT, POLICY) == expected
    chosen = aligned_intents(CONFIG['zones'])['2026-09-24']
    assert chosen.work_distance_meters.low == 2400
    assert chosen.longest_work_distance_meters.low == 400
    assert chosen.work_seconds is None  # Not inferred from the target run.


def test_compliant_distance_band_not_rescaled_to_actual_run():
    expected = forecast(intent(), refs(), AT, POLICY)
    before = deepcopy(expected)
    for distance in (7100, 7800):
        result = compare_distribution(expected, profile(distance=distance), {})
        assert result['features']['distance_meters']['status'] == 'within_expectation'
    result = compare_distribution(expected, profile(distance=8500), {})
    assert result['features']['distance_meters']['status'] == 'above_expectation'
    assert expected == before


@pytest.mark.parametrize('coverage,extra', [(.9, 0), (1., 60)])
def test_partial_response_cannot_be_cleared_or_called_lower(coverage, extra):
    expected = forecast(intent(), refs(), AT, POLICY)
    for hr in (140, 170):
        actual = profile(hr=hr, coverage=coverage, extra_unobserved=extra)
        compared = compare_distribution(expected, actual, {})
        assert compared['features']['hr_300s']['status'] == 'partial_observation'
    higher = compare_distribution(expected,
        profile(hr=190, coverage=coverage, extra_unobserved=extra), {})
    assert higher['features']['hr_300s']['status'] == 'above_expectation'


def test_explicit_tolerance_handles_export_precision():
    expected = dict(expected={'speed_60s': Quantity.point(4., 'reference')})
    actual = profile()
    actual['intensity_structure']['recorded_signals']['speed_duration_curve'][0]['peak_mean_mps'] += .0000047
    compared = compare_distribution(expected, actual, {'speed_60s': .02})
    assert compared['features']['speed_60s']['status'] == 'within_expectation'


def test_numeric_time_prescription_ignores_title():
    rec = prescription(3, 5, 2)
    parsed = intent_from_prescription(rec, CONFIG['zones'])
    assert parsed.work_seconds.low == parsed.work_seconds.high == 900
    assert parsed.longest_work_seconds.low == 300
    assert intent_from_prescription(rec.model_copy(update={'title': '6x400'}), CONFIG['zones']) == parsed


def test_easy_matching_requires_observed_sustained_hr_evidence():
    easy = Intent(Quantity(7100, 7800), None, None, None, easy_hr_ceiling=154)
    result = forecast(easy, refs(profile(reps=0, hr=150)), AT, POLICY)
    assert result['meets_session_count_setting']
    assert result['work_reference_keys'] == []
    assert result['expected']['cardio_points'] is not None
    high = forecast(easy, refs(profile(reps=0, hr=175)), AT, POLICY)
    assert not high['meets_session_count_setting']


def test_naive_times_and_duplicate_sessions_rejected():
    with pytest.raises(ValueError, match='Aware'):
        forecast(intent(), refs(), AT.replace(tzinfo=None), POLICY)
    with pytest.raises(ValueError, match='Duplicate'):
        forecast(intent(), refs()+refs(), AT, POLICY)


@pytest.mark.parametrize('value', [0, 1.5, float('nan'), True])
def test_minimum_sessions_must_be_a_positive_integer(value):
    with pytest.raises(ValueError):
        replace(POLICY, minimum_sessions=value)


@pytest.mark.parametrize('value', [-1, float('nan'), float('inf')])
def test_invalid_tolerance_rejected(value):
    with pytest.raises(ValueError, match='tolerances'):
        compare_distribution(dict(expected={}), profile(), {'distance_meters': value})
