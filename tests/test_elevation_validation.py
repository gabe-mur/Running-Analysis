import numpy as np
import pytest

from run_analysis.elevation_validation import profile_comparison


def test_constant_altitude_offset_does_not_count_as_bad_profile():
    distances=list(range(0,501,10));terrain=[10+5*np.sin(d/100) for d in distances]
    result=profile_comparison(distances,[v+30 for v in terrain],terrain)
    assert result['constant_offset_m']==pytest.approx(30)
    assert result['p90_absolute_residual_m']==pytest.approx(0,abs=1e-8)
    assert result['profile_correlation']==pytest.approx(1)


def test_time_varying_drift_is_not_removed_as_a_constant_offset():
    distances=list(range(0,501,10));terrain=[10.]*len(distances)
    result=profile_comparison(distances,[v+d*.04 for v,d in zip(terrain,distances)],terrain)
    assert result['p90_absolute_residual_m']>8
    assert result['profile_correlation'] is None


def test_bridge_deck_and_ground_disagreement_is_reported_not_assigned_as_watch_error():
    distances=list(range(0,501,10))
    deck=[20+15*np.sin(np.pi*d/500) for d in distances]
    ground=[0.]*len(distances)
    result=profile_comparison(distances,deck,ground)
    assert result['p90_absolute_residual_m']>5
    assert 'watch_is_wrong' not in result
    assert 'replace_recorded' not in result


def test_dense_stationary_samples_do_not_dominate_offset_estimate():
    distances=[0.]*200+[float(d) for d in range(20,501,20)]
    recorded=[100.]*200+[15.]*25;reference=[10.]*len(distances)
    result=profile_comparison(distances,recorded,reference)
    assert result['constant_offset_m']==pytest.approx(5)
    assert result['samples']==26


def test_missing_map_coverage_is_not_zero_or_extrapolated():
    distances=list(range(0,501,20));raw=[20.]*len(distances)
    mapped=[None]*len(distances);mapped[10:14]=[10.]*4
    result=profile_comparison(distances,raw,mapped)
    assert result['samples']==4
    assert result['common_sample_fraction']<.2
    assert result['constant_offset_m']==pytest.approx(10)
    assert profile_comparison(distances,raw,[None]*len(distances))['status']=='insufficient_overlap'
