from copy import deepcopy

import numpy as np
import pytest

from scripts.aerobic_three_layer_model import fit_layers,fit_period_layers
from scripts.aerobic_serial_model import correlation_matrix,fit_serial_layers,fit_serial_sensitivity,kernel_grid,_period_variance


def fixture():
    rng=np.random.default_rng(51);groups=np.repeat([1,0],10);times=np.arange(20)*3.
    y=2.3+.03*groups+rng.normal(0,.03,20)
    return y,np.full(20,.01**2),np.ones((20,1)),np.array([[.02**2]]),groups,times


def test_calendar_gaps_and_run_index_are_distinct_positive_kernels():
    times=np.array([0.,1.,30.]);r=correlation_matrix(times,half_life_days=3)
    assert r[0,1]>r[1,2] and r[0,2]<.002
    assert np.linalg.eigvalsh(r).min()>0
    indexed=correlation_matrix(times,run_rho=.7)
    assert indexed[0,1]==indexed[1,2]==pytest.approx(.7)
    for invalid in ([0,0],[1,0],[float('nan'),1]):
        with pytest.raises(ValueError):correlation_matrix(invalid,half_life_days=3)


def test_identity_recovers_period_variance_model_without_mutation():
    y,m,g,c,groups,times=fixture();before=deepcopy(y)
    expected=fit_period_layers(y,m,g,c,groups,contrasts={'monthly':[-1,1]})
    actual=fit_serial_layers(y,m,g,c,groups,correlation=np.eye(len(y)),contrasts={'monthly':[-1,1]})
    for a,b in zip(expected['periods'],actual['periods']):
        assert a['estimate']==pytest.approx(b['estimate'])
        assert a['range_95']==pytest.approx(b['range_95'])
    # Integer df flooring can differ by one at an exact integer because the
    # two algebraically equivalent covariance calculations round differently.
    assert actual['next_comparable_measured_run']['se']==pytest.approx(expected['next_comparable_measured_run']['se'])
    assert abs(actual['next_comparable_measured_run']['df']-expected['next_comparable_measured_run']['df'])<=1
    np.testing.assert_equal(y,before)


def test_future_covariance_is_used_and_identical_adjustment_cancels():
    y,m,g,c,groups,times=fixture()
    r=correlation_matrix(np.r_[times,times[-1]+1],half_life_days=7)
    independent=fit_serial_layers(y,m,g,c,groups,correlation=r[:-1,:-1])
    correlated=fit_serial_layers(y,m,g,c,groups,correlation=r[:-1,:-1],future_correlation=r[:-1,-1])
    assert correlated['periods']==independent['periods']
    assert correlated['next_comparable_measured_run']['se']<independent['next_comparable_measured_run']['se']
    assert correlated['next_comparable_measured_run']['shared_variance']<1e-20
    assert correlated['periods'][0]['shared_variance']>0
    with pytest.raises(ValueError,match='PSD'):
        fit_serial_layers(y,m,g,c,groups,correlation=np.eye(len(y)),future_correlation=np.ones(len(y)))


def test_envelope_does_not_hide_kernel_selection_uncertainty():
    y,m,g,c,groups,times=fixture()
    fit=fit_serial_sensitivity(y,m,g,c,groups,times=times,future_time=times[-1]+2,contrasts={'monthly':[-1,1]})
    assert len(kernel_grid(times,times[-1]+2))==9
    selected=fit['conditional_selected_fit']['contrasts']['monthly']['range_95']
    envelope=fit['contrasts']['monthly']['range_95']
    assert envelope[0]<=selected[0] and envelope[1]>=selected[1]
    assert 'not an exact' in fit['contrasts']['monthly']['interval_kind']


def test_fast_scalar_solver_preserves_original_moment_equation():
    y,m,g,c,groups,times=fixture()
    for rho in (0,.3,.7,.9):
        r=correlation_matrix(times,run_rho=rho)
        expected=fit_layers(y,m,g,c,np.zeros(len(y),int),day_correlation=r)
        actual=_period_variance(y,np.diag(m)+g@c@g.T,r)
        assert actual==pytest.approx(expected['extra_day_variance'],rel=1e-6,abs=1e-10)


def test_zero_day_variance_does_not_invent_persistence_and_bad_inputs_fail():
    y,m,g,c,groups,times=fixture();y=2.3+.03*groups
    r=correlation_matrix(times,half_life_days=7)
    result=fit_serial_layers(y,m,g,c,groups,correlation=r)
    assert result['period_day_log_sd']==[0.,0.]
    for measurement,gradient,cov,group in [(m*0,g,c,groups),(m,g,-c,groups),(m,g,c,np.ones(len(y),int))]:
        with pytest.raises(ValueError):fit_serial_layers(y,measurement,gradient,cov,group,correlation=r)
    with pytest.raises(ValueError):fit_serial_layers(y,m,g,c,groups,correlation=r,future_correlation=[0.])


def test_support_screen_retains_selected_fit_and_does_not_invent_an_envelope_se():
    y,m,g,c,groups,times=fixture();kwargs=dict(times=times,future_time=times[-1]+2,contrasts={'monthly':[-1,1]})
    zero=fit_serial_sensitivity(y,m,g,c,groups,**kwargs,support_deviance=0.)
    screened=zero['supported_kernel_fit']
    assert screened['supported_kernels']==[zero['selected_kernel']]
    assert screened['contrasts']['monthly']['range_95']==zero['conditional_selected_fit']['contrasts']['monthly']['range_95']
    assert 'se' not in screened['contrasts']['monthly'] and 'probability_positive' not in screened['contrasts']['monthly']
    all_supported=fit_serial_sensitivity(y,m,g,c,groups,**kwargs,support_deviance=1e6)
    assert all_supported['supported_kernel_fit']['contrasts']['monthly']['range_95']==all_supported['contrasts']['monthly']['range_95']
    for invalid in (-1.,float('nan')):
        with pytest.raises(ValueError):fit_serial_sensitivity(y,m,g,c,groups,**kwargs,support_deviance=invalid)


def test_coverage_strata_preserve_denominators_and_supported_kernel_counts():
    from scripts.audit_aerobic_three_layers import summarize
    samples=[]
    for kernel,covered,support in [('a',True,['a','b']),('a',False,['a']),('b',True,['b'])]:
        samples.append(dict(day_sd=.02,selected_kernel=kernel,supported_kernels=support,
            **{k:dict(covered=covered,estimate=0.,truth=0.,range=[-.1,.1]) for k in ('level','monthly','ninety_day','next_day')}))
    result=summarize(samples)
    assert result['by_selected_kernel']['a']['ninety_day']['n']==2
    assert result['by_selected_kernel']['a']['ninety_day']['rate']==.5
    assert result['by_selected_kernel']['b']['ninety_day']['n']==1
    assert result['support_inclusion']==dict(a=2,b=2)


def test_serial_control_cases_added_without_removing_original_cases():
    from scripts.audit_aerobic_three_layers import inputs,simulations
    from scripts.audit_progress_responsiveness import at
    from tests.test_aerobic_three_layer_audit import fixture as input_fixture
    report,inventory,end=input_fixture()
    rows,m,g,groups,end,used,eligible=inputs(report,inventory,end)
    fit=simulations(rows,m,g,np.zeros((4,4)),groups,end,eligible,2,42,serial_controls=True)
    assert len(fit['cells'])==48
    assert {'persistent_days','calendar_days_3','calendar_days_7'}<={v['scenario'] for v in fit['cells']}
