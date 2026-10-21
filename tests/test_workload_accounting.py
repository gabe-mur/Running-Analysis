from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from run_analysis.workload_accounting import (
    Quantity, Workload, continuous_zone_seconds, observed_workload,
    projected_workload, compare_execution,
)
from run_analysis.workload_profile import observed_profile, prescribed_profile
from run_analysis.tcx import parse_tcx
from test_tcx import make_tcx
from test_workload_profile import CONFIG, prescription


def interval(a, b, seconds=60, flags=()):
    return SimpleNamespace(start=SimpleNamespace(heart_rate_bpm=a),
        end=SimpleNamespace(heart_rate_bpm=b), moving_time_s=seconds, flags=flags)


def test_fractional_zone_boundary_has_no_unknown_hole():
    zones = continuous_zone_seconds([interval(138, 139)], CONFIG['zones'])
    assert zones['unknown'] == 0
    assert zones['z1'] == 60
    assert sum(zones.values()) == 60


def test_linear_crossings_conserve_time_and_sampling_invariance():
    whole = continuous_zone_seconds([interval(125, 190)], CONFIG['zones'])
    split = continuous_zone_seconds([interval(125, 150, 60*25/65),
                                    interval(150, 190, 60*40/65)], CONFIG['zones'])
    assert whole == pytest.approx(split)
    assert sum(whole.values()) == pytest.approx(60)
    assert whole['z2'] == pytest.approx(14/65*60)
    descending = continuous_zone_seconds([interval(190, 125)], CONFIG['zones'])
    assert descending == pytest.approx(whole)


@pytest.mark.parametrize('a,b,flags', [(None, 145, ()), (145, None, ()),
    (0, 145, ()), (float('nan'), 145, ()), (145, 145, ('large_recording_gap',))])
def test_missing_invalid_or_gap_hr_is_unknown(a, b, flags):
    zones = continuous_zone_seconds([interval(a, b, flags=flags)], CONFIG['zones'])
    assert zones['unknown'] == 60
    assert sum(zones.values()) == 60


def test_stopped_time_not_invented_as_moving_cardio():
    zones = continuous_zone_seconds([interval(170, 170, seconds=0)], CONFIG['zones'])
    assert sum(zones.values()) == 0


def test_sparse_hr_is_bounded_not_scaled_or_zero(tmp_path):
    activity = parse_tcx(make_tcx(tmp_path/'run.tcx', hr=False)).activities[0]
    done = observed_workload(observed_profile(activity, CONFIG))
    assert done.cardio_points.low == 0
    assert done.cardio_points.high == pytest.approx(20/60*5)
    assert done.cardio_points.estimate is None
    assert done.running_minutes.estimate == pytest.approx(20/60)


def test_missing_trackpoints_do_not_become_a_zero_load_session(tmp_path):
    activity = parse_tcx(make_tcx(tmp_path/'run.tcx')).activities[0]
    activity.trackpoints.clear()
    done = observed_workload(observed_profile(activity, CONFIG))
    assert done.running_minutes.low == 0
    assert done.running_minutes.high == pytest.approx(20/60)
    assert done.running_minutes.estimate is None
    assert done.cardio_points.high == pytest.approx(20/60*5)
    activity.lap_recorded_time_s = None
    missing = observed_workload(observed_profile(activity, CONFIG))
    assert missing.running_minutes is None
    assert missing.cardio_points is None


def test_phase_projection_and_compliance_preserve_exact_target():
    profile = prescribed_profile(prescription(3, 5, 2))
    expected = projected_workload(profile, duration_minutes=Quantity(40, 45, 42),
        base_rate=Quantity(1, 2), work_rate=Quantity(3, 4))
    assert expected.cardio_points.low == 70  # 15*3 + 25*1
    assert expected.cardio_points.high == 120  # 15*4 + 30*2
    assert expected.distance_meters.estimate == 4.2*1609.344
    actual = Workload('observed', Quantity.point(4.5*1609.344, 'measured'),
        Quantity.point(43, 'measured'), Quantity.point(100, 'measured'), {})
    before = deepcopy(actual)
    assert set(compare_execution(expected, actual).values()) == {'within'}
    harder = replace(actual, cardio_points=Quantity.point(140, 'measured'))
    result = compare_execution(expected, harder)
    assert result['distance_meters'] == 'within'
    assert result['cardio_points'] == 'above'
    assert actual == before


def test_unknown_overlapping_hr_is_not_compliance():
    expected = Workload('projected', None, None, Quantity(50, 80), {})
    actual = Workload('observed', None, None, Quantity(20, 100), {})
    assert compare_execution(expected, actual)['cardio_points'] == 'overlapping_uncertainty'


def test_no_response_rates_produces_explicit_unknown_projection():
    expected = projected_workload(prescribed_profile(prescription(3, 5, 2)),
        duration_minutes=Quantity(40, 45))
    assert expected.cardio_points == Quantity(0, 225, basis='explicit_phase_rate_bounds')


def test_recovery_hr_can_stay_higher_than_warmup_without_a_workout_multiplier():
    profile = prescribed_profile(prescription(3, 5, 2))
    expected = projected_workload(profile, duration_minutes=Quantity(40, 40),
        base_rate=Quantity(1, 1), work_rate=Quantity(4, 4), recovery_rate=Quantity(3, 3))
    assert expected.cardio_points.low == expected.cardio_points.high == 93
    # 15*4 work + 4*3 between-work recovery + 21*1 other time.
    assert expected.structure == profile['intensity_structure']


def test_separate_recovery_rate_needs_known_recovery_duration():
    profile = prescribed_profile(prescription(3, 5, None))
    with pytest.raises(ValueError, match='known recovery duration'):
        projected_workload(profile, duration_minutes=Quantity(40, 40), recovery_rate=Quantity(3, 3))


def test_impossible_prescribed_duration_rejected():
    with pytest.raises(ValueError, match='accommodate'):
        projected_workload(prescribed_profile(prescription(3, 5, 2)),
                           duration_minutes=Quantity(18, 18))


def test_warmup_and_cooldown_must_also_fit():
    from run_analysis.web.schemas import WorkoutStep
    rec = prescription(3, 5, 2)
    rec = rec.model_copy(update={'structure': [WorkoutStep(instruction='warmup', duration_minutes=12),
        *rec.structure, WorkoutStep(instruction='cooldown', duration_minutes=10)]})
    # Work/recoveries fit in 30 minutes, but the complete structure needs 41.
    with pytest.raises(ValueError, match='accommodate'):
        projected_workload(prescribed_profile(rec), duration_minutes=Quantity(30, 30))
    projected_workload(prescribed_profile(rec), duration_minutes=Quantity(41, 45))


def test_work_duration_range_is_not_collapsed_to_midpoint():
    rec = prescription(3, 5, 2)
    step = rec.structure[0].model_copy(update={'work_duration_minutes': None,
                                              'work_duration_range_minutes': (4., 6.)})
    rec = rec.model_copy(update={'structure': [step]})
    expected = projected_workload(prescribed_profile(rec), duration_minutes=Quantity(40, 40),
        base_rate=Quantity(1, 1), work_rate=Quantity(4, 4))
    assert expected.cardio_points.low == 76
    assert expected.cardio_points.high == 94


@pytest.mark.parametrize('bounds', [(-1, 2), (2, 1), (0, float('inf')), (0, float('nan'))])
def test_invalid_quantities_rejected(bounds):
    with pytest.raises(ValueError):
        Quantity(*bounds)
