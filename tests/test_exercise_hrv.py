from copy import deepcopy
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from run_analysis.exercise_hrv import (
    dfa_alpha1, smoothness_detrend, correction_sensitivity, exercise_rr_windows,
)


def signal(n=300, seed=812):
    return .42+np.random.default_rng(seed).normal(0,.006,n)


def test_dfa_is_unit_and_offset_invariant_and_not_a_heart_rate_score():
    values=signal(); original=values.copy()
    alpha=dfa_alpha1(values)['alpha1']
    assert dfa_alpha1(values*1000)['alpha1']==pytest.approx(alpha,abs=1e-11)
    assert dfa_alpha1(values+2)['alpha1']==pytest.approx(alpha,abs=1e-11)
    assert .3<alpha<.8
    np.testing.assert_array_equal(values,original)


@pytest.mark.parametrize('n,expected',[(180,.4251174221299478),(257,.4622273598709407),
    (300,.48176026924201065),(1000,.5548927713850977)])
def test_against_fixed_nolds_062_reference_results(n,expected):
    # Generated independently by the unmodified nolds DFA implementation:
    # scales 4..16, overlap=True, order=1, fit_trend=fit_exp='poly'. No runtime
    # nolds dependency or reference implementation copied into production.
    assert dfa_alpha1(signal(n))['alpha1']==pytest.approx(expected,abs=1e-12)


def test_correlation_structure_changes_alpha_without_changing_mean_rr():
    noise=signal(4000)-.42
    correlated=.42+np.cumsum(noise)*.01
    assert dfa_alpha1(correlated)['alpha1']>dfa_alpha1(noise+.42)['alpha1']+.5


@pytest.mark.parametrize('values',[np.ones(300),[.4]*159,[float('nan')]*300])
def test_unidentifiable_dfa_is_not_a_fitness_or_threshold_result(values):
    with pytest.raises(ValueError):dfa_alpha1(values)


@pytest.mark.parametrize('strength',[250.,500.,1000.])
def test_banded_smoothness_prior_matches_independent_dense_linear_system(strength):
    values=signal(180);d=np.diff(np.eye(len(values)),n=2,axis=0)
    centered=values-values.mean()
    expected=centered-np.linalg.solve(np.eye(len(values))+strength**2*d.T@d,centered)
    np.testing.assert_allclose(smoothness_detrend(values,strength),expected,atol=2e-11)


def test_correction_keeps_original_and_reports_both_raw_and_preprocessing_choices():
    values=signal();values[100]=.60;before=values.copy()
    result=correction_sensitivity(values)
    assert len(result['variants'])==9
    assert result['raw_alpha1']!=result['alpha1']
    assert not result['clinical_artifact_rate_established']
    assert result['preprocessing_range'][0]<=result['alpha1']<=result['preprocessing_range'][1]
    np.testing.assert_array_equal(values,before)


@pytest.mark.parametrize('indexes',[[0],[299],[100,101],list(range(100,140))])
def test_edges_clusters_or_many_bad_beats_are_not_interpolated_into_acceptance(indexes):
    values=signal();values[indexes]=.02
    result=correction_sensitivity(values)
    assert not result['usable_for_comparison']
    assert result['status']=='screening_rejected'


def payload():
    origin=datetime(2026,1,1,tzinfo=timezone.utc)
    at=lambda s:(origin+timedelta(seconds=s)).isoformat()
    # Each 5-second bracket contains 12 complete RR intervals whose total
    # duration is 5 seconds. Bracket rounding isn't used to stretch them.
    values=signal(720);values=values.reshape(-1,12)
    values=values/values.sum(axis=1,keepdims=True)*5
    return origin,dict(rr_batches=[dict(intervals_seconds=list(rr),preceding_record_utc=at(i*5),
        following_record_utc=at((i+1)*5)) for i,rr in enumerate(values)],
        timer_events=[dict(timestamp=at(0),event_type='start'),dict(timestamp=at(300),event_type='stop_all')])


def test_source_windows_keep_pause_and_unanchored_tail_out_and_never_stitch():
    origin,p=payload();original=deepcopy(p)
    p['timer_events'][1:1]=[dict(timestamp=(origin+timedelta(seconds=150)).isoformat(),event_type='stop_all'),
        dict(timestamp=(origin+timedelta(seconds=160)).isoformat(),event_type='start')]
    p['rr_batches'].append(dict(intervals_seconds=[.4]*50,preceding_record_utc=(origin+timedelta(seconds=300)).isoformat(),following_record_utc=None))
    result=exercise_rr_windows(p,origin,300)
    assert result['unanchored_rr_count']==50
    assert all(not w['usable_for_comparison'] for w in result['windows'] if w['start_seconds']<160 and w['end_seconds']>150)
    assert not result['planner_influence'] and not result['fitness_influence']
    assert original['rr_batches']==p['rr_batches'][:-1]


def test_missing_rr_bracket_cannot_be_joined_into_an_apparently_complete_window():
    origin,p=payload();p['rr_batches'].pop(10)
    result=exercise_rr_windows(p,origin,300)
    assert result['windows'][0]['status']=='missing_or_discontinuous_rr'


def test_internal_invalid_interval_is_not_discarded_as_array_padding():
    origin,p=payload();p['rr_batches'][10]['intervals_seconds'][3]=None
    assert exercise_rr_windows(p,origin,300)['windows'][0]['status']=='missing_or_discontinuous_rr'


def test_large_rr_clock_mismatch_is_not_rescaled_to_record_brackets():
    origin,p=payload()
    for batch in p['rr_batches']:batch['intervals_seconds']=[v*1.1 for v in batch['intervals_seconds']]
    assert all(w['status']=='incomplete_or_clock_mismatch' for w in exercise_rr_windows(p,origin,300)['windows'])


def test_no_timer_or_no_rr_means_unavailable_not_a_normal_hrv_value():
    origin,p=payload();p['timer_events']=[]
    assert all(w['status']=='paused_or_timer_unknown' for w in exercise_rr_windows(p,origin,300)['windows'])
    p['timer_events']=payload()[1]['timer_events'];p['rr_batches']=[]
    assert all(not w['usable_for_comparison'] for w in exercise_rr_windows(p,origin,300)['windows'])
