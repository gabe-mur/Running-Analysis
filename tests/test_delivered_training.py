from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from run_analysis.delivered_training import (
    DoseContract, DoseTarget, DeliveredDose, PurposeEvidence,
    assess_delivery, dose_from_profile, dose_metrics,
)
from run_analysis.workload_accounting import Quantity
from run_analysis.workload_profile import WorkBout

AT = datetime(2026, 9, 25, tzinfo=timezone.utc)


def q(low, high=None):
    return Quantity(low, low if high is None else high, basis='explicit_test_band')


def dose(bouts=(), **kwargs):
    fields = dict(key='run', start=AT, end=AT+timedelta(hours=1),
        available_at=AT+timedelta(hours=1), distance_meters=q(10000),
        moving_seconds=q(3600), bouts=tuple(bouts),
        structure_basis='explicit_segments' if bouts else 'unknown')
    fields.update(kwargs)
    return DeliveredDose(**fields)


def contract(purpose='long_endurance', **kwargs):
    fields = dict(issued_at=AT-timedelta(days=1), purpose=purpose,
        targets=(DoseTarget('distance_meters', q(9500, 10500)),), source='frozen_fixture')
    fields.update(kwargs)
    return DoseContract(**fields)


def quality():
    return contract('threshold', targets=(DoseTarget('work_seconds', q(870, 930)),),
                    bout_seconds=q(290, 310))


def assess(c, d, **kwargs):
    return assess_delivery(c, d, as_of=AT+timedelta(days=1), **kwargs)


def test_long_label_does_not_turn_short_run_into_full_delivery():
    full = assess(contract(), dose(role='long'))
    short = assess(contract(), dose(role='long', distance_meters=q(5000)))
    assert full['numerical_fit'] == 'within_all_issued_bands'
    assert short['dimensions']['distance_meters']['status'] == 'below'
    assert short['observed_metrics']['distance_meters'].low == 5000
    assert short['physiological_equivalence_established'] is False


@pytest.mark.parametrize('meters', [9500, 10000, 10500])
def test_all_band_values_are_compliant_without_midpoint_rewriting(meters):
    result = assess(contract(), dose(distance_meters=q(meters)))
    assert result['dimensions']['distance_meters']['status'] == 'within'
    assert result['observed_metrics']['distance_meters'].low == meters


def test_excess_is_retained_without_bonus_or_capping():
    result = assess(contract(), dose(distance_meters=q(12000)))
    assert result['dimensions']['distance_meters']['status'] == 'above'
    assert result['dimensions']['distance_meters']['ratio_to_target_range'][0] > 1
    assert not result['adaptation_authorized']


def test_partial_quality_is_not_zero_or_full_and_extra_miles_do_not_cancel_it():
    result = assess(quality(), dose([WorkBout(300, 120, 1000)]*2, distance_meters=q(16000)))
    assert result['dimensions']['work_seconds']['status'] == 'below'
    assert result['observed_metrics']['work_seconds'].low == 600
    assert result['observed_metrics']['distance_meters'].low == 16000


def test_equal_work_totals_keep_bout_and_recovery_differences():
    long = assess(quality(), dose([WorkBout(300, 120, 1000)]*3))
    short = assess(quality(), dose([WorkBout(150, 120, 500)]*6))
    assert long['observed_metrics']['work_seconds'] == short['observed_metrics']['work_seconds']
    assert long['numerical_fit'] == 'within_all_issued_bands'
    assert short['numerical_fit'] == 'measured_difference'
    assert long['observed_metrics']['between_recovery_seconds'].low == 240
    assert short['observed_metrics']['between_recovery_seconds'].low == 600
    assert short['per_bout'][0]['seconds']['status'] == 'below'


def test_one_long_bout_cannot_hide_other_short_bouts():
    result = assess(quality(), dose([WorkBout(300, 60, 1000)]+[WorkBout(100, 60, 300)]*6))
    assert result['dimensions']['work_seconds']['status'] == 'within'
    assert result['per_bout'][1]['seconds']['status'] == 'below'
    assert result['numerical_fit'] == 'measured_difference'


def test_novel_distance_work_needs_no_hr_or_historical_matches():
    c = contract('high_aerobic', targets=(DoseTarget('work_meters', q(1450, 1550)),), bout_meters=q(480, 520))
    result = assess(c, dose([WorkBout(140, 90, 500)]*3))
    assert result['numerical_fit'] == 'within_all_issued_bands'
    assert result['purpose_evidence'] == 'unverified'


def test_missing_work_is_unknown_and_inferred_boundaries_remain_provisional():
    result = assess(quality(), dose(role='intervals'))
    assert result['dimensions']['work_seconds']['status'] == 'unknown'
    assert result['observed_metrics']['work_seconds'] is None
    inferred = assess(quality(), dose([WorkBout(300, 120, 1000)]*3, structure_basis='pace_inferred'))
    assert inferred['numerical_fit'] == 'within_all_issued_bands'
    assert inferred['structure_certainty'] == 'provisional_unquantified_boundaries'
    assert not inferred['adaptation_authorized']


def test_independent_purpose_does_not_change_actual_work_or_override_difference():
    actual = dose([WorkBout(300, 120, 1000)]*3)
    evidence = PurposeEvidence('run', 'speed_skill', AT+timedelta(hours=2), 'athlete_description')
    result = assess(quality(), actual, purpose_evidence=evidence)
    assert result['purpose_evidence'] == 'different_purpose'
    assert result['numerical_fit'] == 'within_all_issued_bands'
    late = replace(evidence, available_at=AT+timedelta(days=2))
    assert assess(quality(), actual, purpose_evidence=late)['purpose_evidence'] == 'unverified'


def test_range_uncertainty_is_not_confirmed_fit():
    result = assess(contract(), dose(distance_meters=q(9000, 11000)))
    assert result['dimensions']['distance_meters']['status'] == 'overlapping_uncertainty'
    assert result['numerical_fit'] == 'unresolved'


def test_no_double_counting_work_inside_session_and_missing_recovery_is_unknown():
    actual = dose([WorkBout(300, None, 1000)]*3)
    measured = dose_metrics(actual)
    assert measured['distance_meters'].low == 10000
    assert measured['work_meters'].low == 3000
    assert measured['between_recovery_seconds'] is None
    # Descriptive recovery comparison requires an explicit issued constraint.
    c = replace(quality(), targets=quality().targets+(DoseTarget('between_recovery_seconds', q(200, 280)),))
    assert assess(c, actual)['dimensions']['between_recovery_seconds']['status'] == 'unknown'


def test_history_and_contract_are_immutable_and_chronological():
    c, actual = quality(), dose([WorkBout(300, 120, 1000)]*3)
    before = deepcopy((c, actual))
    assess(c, actual)
    assert (c, actual) == before
    with pytest.raises(ValueError, match='frozen'):
        assess(replace(c, issued_at=AT+timedelta(seconds=1)), actual)
    with pytest.raises(ValueError, match='not yet available'):
        assess(c, replace(actual, available_at=AT+timedelta(days=2)))
    with pytest.raises(ValueError, match='another session'):
        assess(c, actual, purpose_evidence=PurposeEvidence('other', 'threshold', AT, 'report'))


def test_contract_cannot_be_just_quality_label_or_no_total_work():
    with pytest.raises(ValueError, match='Quality needs'):
        contract('threshold')
    with pytest.raises(ValueError, match='Quality needs'):
        contract('threshold', bout_seconds=q(300))
    with pytest.raises(ValueError, match='session dose'):
        contract(targets=(DoseTarget('bout_count', q(3)),))


def test_impossible_work_totals_rejected():
    with pytest.raises(ValueError, match='Work exceeds session distance'):
        dose([WorkBout(300, 120, 4000)]*3)
    with pytest.raises(ValueError, match='Work exceeds session duration'):
        dose([WorkBout(1500, 120, 1000)]*3)


def test_adapter_reuses_recorded_features_and_does_not_mutate_profile():
    from test_workload_expectation import profile
    from run_analysis.workload_expectation import observed_features
    p = profile()
    # This older aggregate-only fixture omits individual timing; populate the
    # real observed_profile schema rather than fabricate timing in the adapter.
    for bout in p['intensity_structure']['bouts']:
        bout.update(work_seconds=105., recovery_after_seconds=90.)
    before = deepcopy(p)
    actual = dose_from_profile(p, key='run', start=AT, end=AT+timedelta(hours=1), available_at=AT+timedelta(hours=1))
    reference = observed_features(p)
    assert dose_metrics(actual)['work_seconds'].low == reference['work_seconds'].low
    assert dose_metrics(actual)['distance_meters'] == reference['distance_meters']
    assert p == before


def test_adapter_preserves_existing_inference_without_promoting_easy_jitter():
    from test_workload_inferred_evidence import inferred_profile
    p = inferred_profile()
    actual = dose_from_profile(p, key='run', start=AT, end=AT+timedelta(hours=1), available_at=AT+timedelta(hours=1))
    assert actual.structure_basis == 'pace_inferred'
    assert len(actual.bouts) == 6
    p['intensity_structure']['session_workout_type'] = 'easy'
    actual = dose_from_profile(p, key='run', start=AT, end=AT+timedelta(hours=1), available_at=AT+timedelta(hours=1))
    assert actual.bouts == ()
    assert actual.structure_basis == 'unknown'


def test_adapter_does_not_need_hr_to_retain_running_exposure(tmp_path):
    from test_tcx import make_tcx
    from test_workload_profile import CONFIG
    from run_analysis.tcx import parse_tcx
    from run_analysis.workload_profile import observed_profile
    activity = parse_tcx(make_tcx(tmp_path/'run.tcx')).activities[0]
    p = observed_profile(activity, CONFIG, workout_type='easy')
    before = dose_from_profile(p, key='run', start=AT, end=AT+timedelta(hours=1), available_at=AT+timedelta(hours=1))
    for point in activity.trackpoints:
        point.heart_rate_bpm = None
    missing = observed_profile(activity, CONFIG, workout_type='easy')
    after = dose_from_profile(missing, key='run', start=AT, end=AT+timedelta(hours=1), available_at=AT+timedelta(hours=1))
    assert before == after


def test_continuous_work_reuses_existing_detector_without_discarding_tempo():
    from types import SimpleNamespace
    from test_workload_expectation import profile
    from test_workload_profile import CONFIG
    from run_analysis.delivered_training import continuous_phase_from_activity
    activity = SimpleNamespace(laps=[SimpleNamespace(lap_index=i, total_time_s=seconds,
        distance_m=meters, average_hr_bpm=hr, maximum_hr_bpm=hr+5)
        for i, (seconds, meters, hr) in enumerate([(660, 1600, 140), (1080, 3400, 168), (900, 2000, 150)])])
    phase = continuous_phase_from_activity(activity, CONFIG)
    assert phase.duration_seconds == 1080
    assert phase.start_offset_seconds == 660
    p = profile(reps=0)
    p['intensity_structure']['session_workout_type'] = 'tempo_threshold'
    measured = dose_from_profile(p, key='run', start=AT, end=AT+timedelta(hours=1),
        available_at=AT+timedelta(hours=1), continuous_phase=phase)
    assert measured.structure_basis == 'recorded_continuous_lap'
    assert dose_metrics(measured)['work_seconds'].low == 1080
    assert dose_metrics(measured)['work_meters'].low == pytest.approx(3400)
    p['intensity_structure']['session_workout_type'] = 'easy'
    with pytest.raises(ValueError, match='eligible role'):
        dose_from_profile(p, key='run', start=AT, end=AT+timedelta(hours=1),
            available_at=AT+timedelta(hours=1), continuous_phase=phase)


def test_short_relaxed_reps_can_be_explicitly_measured_without_detector_minimum():
    c = contract('speed_skill', targets=(DoseTarget('work_seconds', q(75, 85)),), bout_seconds=q(18, 22))
    result = assess(c, dose([WorkBout(20, 90, 70)]*4))
    assert result['numerical_fit'] == 'within_all_issued_bands'
    assert result['purpose_evidence'] == 'unverified'


def test_recovery_cannot_extend_work_block_beyond_session():
    with pytest.raises(ValueError, match='Work and recovery'):
        dose([WorkBout(300, 1800, 1000)]*3)
