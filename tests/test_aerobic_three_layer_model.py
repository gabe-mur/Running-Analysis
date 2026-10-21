from copy import deepcopy
import math

import numpy as np
import pytest

from scripts.aerobic_three_layer_model import fit_layers,fit_period_layers,student_cdf,student_critical


@pytest.mark.parametrize('df,expected',[(1,12.706204736),(5,2.570581836),(10,2.228138852),(30,2.042272456)])
def test_student_intervals_match_reference_quantiles(df,expected):
    assert student_critical(df)==pytest.approx(expected,abs=1e-7)
    assert student_cdf(student_critical(df),df)==pytest.approx(.975,abs=1e-10)
    assert student_cdf(-1,df)==pytest.approx(1-student_cdf(1,df))


def fixture(noise=.03):
    rng=np.random.default_rng(44);groups=np.array([0]*10+[1]*10)
    y=math.log(10.)+.03*groups+rng.normal(0,noise,20)
    return y,np.full(20,.01**2),np.ones((20,1)),np.array([[.0004]]),groups


def test_measurement_day_and_mean_uncertainties_remain_distinct():
    result=fit_layers(*fixture())
    assert result['extra_day_log_sd']>0
    assert result['next_day_true_performance']['se']>result['periods'][0]['se']
    for p in result['periods']+[result['change']]:
        assert p['se']**2==pytest.approx(sum(p[k] for k in ('measurement_variance','day_variance','shared_variance'))*p['small_sample_scale'])
    assert result['change']['shared_variance']<1e-20
    assert result['periods'][0]['shared_variance']>0


def test_zero_scatter_keeps_measurement_floor_without_inventing_day_variation():
    y,m,g,c,groups=fixture()
    y=math.log(10.)+.03*groups
    result=fit_layers(y,m,g,c,groups)
    assert result['extra_day_variance']==0
    assert result['periods'][0]['se']>0
    assert result['change']['estimate']==pytest.approx(.03)


def test_declared_larger_measurement_error_is_not_added_again_as_day_noise():
    y,m,g,c,groups=fixture()
    precise=fit_layers(y,m,g,c,groups)
    imprecise=fit_layers(y,np.full(20,.08**2),g,c,groups)
    assert imprecise['extra_day_variance']==0
    assert precise['extra_day_variance']>0


def test_invalid_inputs_and_sparse_contrasts_cannot_create_precision():
    y,m,g,c,groups=fixture()
    with pytest.raises(ValueError):fit_layers(y,np.zeros(20),g,c,groups)
    with pytest.raises(ValueError):fit_layers(y[:2],m[:2],g[:2],c,[0,1])
    with pytest.raises(ValueError):fit_layers(y,m,g,-c,groups)
    before=deepcopy(y);fit_layers(y,m,g,c,groups)
    np.testing.assert_equal(y,before)


def test_period_model_does_not_force_noisy_current_days_to_match_quiet_prior_days():
    rng=np.random.default_rng(1);groups=np.array([0]*20+[1]*20)
    y=math.log(10)+rng.normal(0,np.where(groups==0,.08,.008))
    result=fit_period_layers(y,np.full(40,.005**2),np.zeros((40,1)),np.array([[0.]]),groups)
    assert result['period_day_log_sd'][0]>3*result['period_day_log_sd'][1]
    assert result['periods'][0]['se']>result['periods'][1]['se']
    assert result['next_day_true_performance']['se']>result['periods'][0]['se']
    assert result['next_day_true_performance']['df']<=len(y)-2


def test_all_contrasts_use_one_current_level_and_shared_error_is_retained():
    y,m,g,c,groups=fixture();before=deepcopy(y)
    a=fit_period_layers(y,m,g,c,groups,contrasts={'a':[-1,1]})
    b=fit_period_layers(y,m,g,c,groups,contrasts={'b':[1,-1],'a':[-1,1]})
    assert a['periods']==b['periods']
    assert a['contrasts']['a']==b['contrasts']['a']
    assert a['periods'][0]['shared_variance']>0
    assert a['change']['shared_variance']<1e-20
    np.testing.assert_equal(y,before)


def test_period_model_rejects_unidentified_day_variation_not_silently_sets_it_to_zero():
    with pytest.raises(ValueError,match='two observations'):
        fit_period_layers([1.,2.,2.1],[.01]*3,[[0.]]*3,[[0.]],[0,1,1])


def test_same_adjustment_error_cancels_in_measured_prediction_not_true_level():
    result=fit_period_layers(*fixture())
    assert result['periods'][0]['shared_variance']>0
    measured=result['next_comparable_measured_run']
    assert measured['shared_variance']<1e-20
    assert measured['measurement_variance']>result['periods'][0]['measurement_variance']
    assert measured['day_variance']>result['periods'][0]['day_variance']
    assert measured['range_95'][0]<measured['estimate']<measured['range_95'][1]


def test_single_available_period_needs_no_eight_run_gate_or_fake_prior():
    result=fit_period_layers([2.30,2.32,2.29],[.0001]*3,[[0.]]*3,[[0.]],[0,0,0])
    assert len(result['periods'])==1
    assert 'change' not in result and 'contrasts' not in result
    assert result['periods'][0]['se']>0
    assert result['periods'][0]['df']<=2
