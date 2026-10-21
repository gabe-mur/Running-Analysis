from copy import deepcopy
from itertools import product
import math

import numpy as np
import pytest

from scripts.aerobic_joint_profile import GaussianChangeProfile,minimize_positive,polish_coordinates,LR_95


def fixture(mode='ml'):
    rng=np.random.default_rng(54);groups=np.repeat([1,0],10)
    y=2.3+.025*groups+rng.normal(0,.04,20);m=np.full(20,.01**2)
    return GaussianChangeProfile(y,m,np.ones((20,1)),np.array([[.005**2]]),groups,
        times=np.arange(20)*3.,future_time=60.,mode=mode)


@pytest.mark.parametrize('mode',['ml','nuisance_residual'])
@pytest.mark.parametrize('value',[None,.02])
def test_profile_analytic_gradients_match_finite_differences(mode,value):
    model=fixture(mode);f=model.objective('run_rho_0.7',[-1,1],value=value);theta=np.array([.7,1.1])
    objective,gradient=f(theta);numeric=[]
    for i in range(2):
        step=np.zeros(2);step[i]=1e-5
        numeric.append((f(theta+step)[0]-f(theta-step)[0])/2e-5)
    np.testing.assert_allclose(gradient,numeric,rtol=1e-6,atol=1e-6)
    assert np.isfinite(objective)


def test_positive_optimizer_handles_boundaries_and_reports_failed_convergence():
    def objective(x):return float(np.sum((x-np.array([-2.,3.]))**2)),2*(x-np.array([-2.,3.]))
    fit=minimize_positive(objective,[1.,1.])
    assert fit['converged'];np.testing.assert_allclose(fit['x'],[0.,3.],atol=1e-5)
    assert not minimize_positive(objective,[1.,1.],max_iterations=1)['converged']
    polished=polish_coordinates(objective,[.1,.1])
    assert polished['converged'];np.testing.assert_allclose(polished['x'],[0.,3.],atol=1e-4)


def test_iid_profile_matches_analytic_gaussian_unknown_variance_likelihood():
    y=np.array([2.28,2.35,2.30,2.26,2.33,2.29,2.36,2.31]);n=len(y);m=.005**2
    model=GaussianChangeProfile(y,np.full(n,m),np.zeros((n,1)),np.array([[0.]]),np.zeros(n,int),times=np.arange(n),future_time=n)
    model.kernels={'independent':np.eye(n)}
    free=model.fit([1.]);value=float(y.mean()+.02)
    assert free['converged']
    assert free['estimate']==pytest.approx(y.mean(),abs=1e-10)
    assert free['period_day_log_sd'][0]**2==pytest.approx(float(np.var(y))-m,rel=1e-5)
    result=model.likelihood_ratio([1.],value,free=free)
    expected=n*math.log(float(np.mean((y-value)**2))/float(np.var(y)))
    assert result['statistic']==pytest.approx(expected,abs=1e-7)
    assert LR_95==pytest.approx(3.841458820694124)


def test_profile_refits_nuisance_parameters_and_recovers_endpoints_without_mutation():
    model=fixture();before=deepcopy(model.y);contrast=[-1,1];free=model.fit(contrast)
    assert not free['unconverged_kernels']
    constrained=model.fit(contrast,value=.1,warm=free['kernel_fits'])
    assert constrained['period_day_log_sd']!=free['period_day_log_sd']
    interval=model.interval(contrast,free=free,iterations=12)
    assert interval['range_95'][0]<interval['estimate']<interval['range_95'][1]
    for endpoint in interval['range_95']:
        assert model.likelihood_ratio(contrast,endpoint,free=free)['statistic']==pytest.approx(LR_95,abs=.01)
    np.testing.assert_equal(model.y,before)


def test_interval_restarts_both_bounds_if_a_nesting_repair_moves_the_free_anchor(monkeypatch):
    model=fixture();free=model.fit([-1,1]);old=free['estimate'];new=old+.04
    calls=[]
    def repaired_ratio(contrast,value,*,free):
        if not calls:
            free['estimate']=new;free['objective']-=1.
        calls.append(value)
        return dict(statistic=((value-new)/.01)**2,unconverged_kernels=[])
    monkeypatch.setattr(model,'likelihood_ratio',repaired_ratio)
    result=model.interval([-1,1],free=free,iterations=20)
    assert result['anchor_restarts']==1
    assert result['estimate']==new
    np.testing.assert_allclose(result['range_95'],[new-math.sqrt(LR_95)*.01,new+math.sqrt(LR_95)*.01],atol=1e-7)


@pytest.mark.parametrize('mode',['ml','nuisance_residual'])
def test_joint_optimizer_beats_coarse_variance_grid_for_free_and_constrained_fits(mode):
    model=fixture(mode);model.kernels={k:v for k,v in model.kernels.items() if k in ('calendar_half_life_0d','calendar_half_life_7d')}
    for value in (None,.1):
        best=model.fit([-1,1],value=value)
        assert not best['unconverged_kernels']
        for kernel in model.kernels:
            objective=model.objective(kernel,[-1,1],value=value)
            for sd in product((0.,.01,.04,.1),repeat=2):
                assert best['objective']<=objective(np.array(sd)/model.scale)[0]+1e-6


def test_same_shared_scale_error_cancels_in_change_likelihood():
    first=fixture();second=fixture()
    first.known+=np.ones((first.n,first.n))*.1
    first.kernels=second.kernels={'independent':np.eye(first.n)}
    # The shared mean error changes level uncertainty, not a difference whose
    # coefficients sum to zero. Use residual likelihood to integrate that mean.
    first.mode=second.mode='nuisance_residual'
    a=first.interval([-1,1],iterations=10);b=second.interval([-1,1],iterations=10)
    assert a['estimate']==pytest.approx(b['estimate'],abs=1e-6)
    assert a['range_95']==pytest.approx(b['range_95'],abs=1e-5)


def test_bootstrap_null_simulation_is_reproducible_and_preserves_monte_carlo_rank_rule():
    model=fixture('nuisance_residual');model.kernels={'independent':np.eye(model.n)}
    # This single-kernel numerical control is retained by every clone. Normal
    # production-independent experiments search the entire nine-kernel grid.
    first=model.bootstrap_test([-1,1],0.,draws=19,seed=12)
    second=model.bootstrap_test([-1,1],0.,draws=19,seed=12)
    assert first==second
    assert first['pvalue']*20==pytest.approx(round(first['pvalue']*20))
    assert first['covered_95']==(first['pvalue']>.05)
    assert first['pvalue']>=1/20
    with pytest.raises(ValueError):model.bootstrap_test([-1,1],0.,draws=18)


def test_constrained_starts_repair_a_missed_free_optimum_without_dropping_the_draw():
    model=fixture('nuisance_residual');model.kernels={'independent':np.eye(model.n)}
    free=model.fit([-1,1]);expected=free['objective']
    # Simulate a local-optimization failure in the cached free candidate.
    free['kernel_fits']['independent']['objective']+=10.
    free['objective']+=10.
    result=model.likelihood_ratio([-1,1],0.,free=free)
    assert result['nesting_refits']>0
    assert free['objective']==pytest.approx(expected,abs=1e-6)
    assert result['statistic']>=0 and not result['unconverged_kernels']


@pytest.mark.parametrize('value',[0.,.5,.0466])
def test_early_bootstrap_decision_is_lossless_and_does_not_claim_an_exact_pvalue(value):
    model=fixture('nuisance_residual');model.kernels={'independent':np.eye(model.n)}
    full=model.bootstrap_test([-1,1],value,draws=63,seed=19)
    early=model.bootstrap_test([-1,1],value,draws=63,seed=19,decision_only=True)
    assert full['covered_95']==early['covered_95']
    assert early['pvalue_bounds'][0]<=full['pvalue']<=early['pvalue_bounds'][1]
    assert early['evaluated_draws']<=63
    if early['evaluated_draws']<63:
        assert early['pvalue'] is None and early['bootstrap_statistic_95'] is None


def test_observer_reuses_original_draws_and_restricted_scope_keeps_original_scenario_seed():
    from scripts.audit_aerobic_three_layers import simulations,inputs
    from tests.test_aerobic_three_layer_audit import fixture as input_fixture
    report,inventory,end=input_fixture();rows,m,g,groups,end,used,eligible=inputs(report,inventory,end)
    seen=[]
    result=simulations(rows,m,g,np.zeros((4,4)),groups,end,eligible,2,123,serial_controls=True,
        only_scenarios=['independent_days'],gains=(0.,),observer=lambda scenario,gain,y,fit,targets:seen.append((scenario,gain,targets['ninety_day'][1])))
    all_cases=simulations(rows,m,g,np.zeros((4,4)),groups,end,eligible,2,123,serial_controls=True)
    assert result['cells']==[c for c in all_cases['cells'] if c['scenario']=='independent_days' and c['gain_percent']==0]
    assert len(seen)==2 and all(v[:2]==('independent_days',0.) for v in seen)


def test_invalid_profile_requests_are_not_silently_fitted():
    model=fixture()
    for contrast in ([0,0],[1],[float('nan'),1]):
        with pytest.raises(ValueError):model.fit(contrast)
    with pytest.raises(ValueError):model.fit([-1,1],value=float('nan'))
