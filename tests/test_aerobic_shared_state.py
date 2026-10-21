from copy import deepcopy

import numpy as np
import pytest

from scripts.aerobic_joint_profile import GaussianChangeProfile
from scripts.aerobic_shared_state import fit_shared_state, state_contrast
from scripts.audit_aerobic_shared_influence import robust
from scripts.audit_aerobic_shared_outlier_inference import selected_cases
from scripts.audit_aerobic_shared_profile_compatibility import cases, summarize_compatibility
from datetime import datetime, timedelta, timezone


def fixture():
    rng = np.random.default_rng(61); groups = np.repeat([2,3,1,0], 6)
    return dict(y=2.3+.01*groups+rng.normal(0,.025,len(groups)),
        measurement=np.full(len(groups),.01**2), gradients=np.ones((len(groups),1)),
        covariance=np.array([[.005**2]]), groups=groups, times=np.arange(len(groups))*3.-69)


def test_shared_point_fits_once_and_contrasts_are_read_only(monkeypatch):
    data=fixture();before=deepcopy(data);original=GaussianChangeProfile.fit;calls=[]
    def record(self,*args,**kwargs):
        calls.append(1)
        return original(self,*args,**kwargs)
    monkeypatch.setattr(GaussianChangeProfile,'fit',record)
    state=fit_shared_state(**data);snapshot=deepcopy(state)
    first=state_contrast(state,[-1,1,0,0]);second=state_contrast(state,[-1,0,1,0])
    assert first['current_pace_min_mile']==second['current_pace_min_mile']==state['current_pace_min_mile']
    assert len(calls)==1 and state==snapshot
    for key in data:np.testing.assert_equal(data[key],before[key])
    assert not state['unconverged_kernels']


def test_shared_point_matches_existing_full_ml_not_a_new_location_estimator():
    data=fixture();state=fit_shared_state(**data)
    model=GaussianChangeProfile(data['y'],data['measurement'],data['gradients'],data['covariance'],
        data['groups'],times=data['times'],future_time=2.,mode='ml')
    for contrast in ([-1,1,0,0],[-1,0,1,0]):
        actual=model.fit(contrast)
        assert state_contrast(state,contrast)['estimate']==pytest.approx(actual['estimate'],abs=1e-12)
        np.testing.assert_allclose(state['period_log_pace'],actual['beta'],atol=1e-12)


def test_conditional_mean_weights_reconstruct_one_shared_state():
    data=fixture();state=fit_shared_state(**data)
    weights=np.array(state['conditional_mean_weights'])
    np.testing.assert_allclose(weights@data['y'],state['period_log_pace'],atol=1e-12)
    np.testing.assert_allclose(weights@np.eye(4)[data['groups']],np.eye(4),atol=1e-12)
    with pytest.raises(ValueError):state_contrast(state,[0,0,0,0])
    with pytest.raises(ValueError):state_contrast(state,[-1,1])


def test_shared_point_supports_two_observed_periods_without_invented_older_history():
    data=fixture();mask=data['groups']<2
    for key in ('y','measurement','gradients','groups','times'):data[key]=data[key][mask]
    state=fit_shared_state(**data)
    assert len(state['period_log_pace'])==2
    assert state_contrast(state,[-1,1])['current_pace_min_mile']>0


def test_real_period_wide_improvement_is_preserved_not_downweighted_as_an_outlier():
    data=fixture();baseline=fit_shared_state(**data);improved=deepcopy(data)
    improved['y'][data['groups']==0] -= np.log1p(.02)
    changed=fit_shared_state(**improved)
    assert baseline['current_pace_min_mile']/changed['current_pace_min_mile']==pytest.approx(1.02,rel=1e-8)
    for vector in ([-1,1,0,0],[-1,0,1,0]):
        gain=state_contrast(changed,vector)['estimate']-state_contrast(baseline,vector)['estimate']
        assert gain==pytest.approx(np.log1p(.02),abs=1e-8)


def test_influence_reference_keeps_original_window_anchor_after_latest_run_removal():
    end=datetime(2026,10,14,tzinfo=timezone.utc)
    rows=[dict(start_time=(end-timedelta(days=day)).isoformat(),pace_min_mile=10.+day/200,
        uncertainty_95_min_mile=.5,trend_weight=1.,shared_log_speed_gradient=[1.])
        for day in (115,105,95,50,40,30,27.5,20,10,0)]
    covariance=np.array([[.01**2]])
    expected=robust(rows[:-1],end,covariance)
    shifted=robust(rows[:-1],end-timedelta(days=10),covariance)
    assert expected['current_pace_min_mile'] != shifted['current_pace_min_mile']
    # The audit calls this explicitly with the unchanged original end, not
    # the last remaining row; deletion is not an accidental calendar rollover.
    assert expected==robust(rows[:-1],end,covariance)


def test_outlier_inference_selects_extremes_by_influence_not_outcome():
    cases=[dict(current_shift_seconds_per_mile=x,activity_id=i) for i,x in enumerate([2,-9,4,-1])]
    selected=selected_cases(dict(single_run_stress=cases))
    assert selected['worst_slower']==cases[2]
    assert selected['worst_faster']==cases[1]


def test_profile_compatibility_replays_aligned_masks_and_frozen_points_without_mutation():
    data=dict(actual_log_pace=[2.3,2.31,2.32])
    influence=dict(baseline=dict(contrasts=dict(monthly=dict(estimate=.03),ninety_day=dict(estimate=.01))),
        leave_one_out=[dict(activity_id=i,contrast_shift_log=dict(monthly=.001*i,ninety_day=.002*i)) for i in (0,1,2)],
        single_run_stress=[dict(activity_id=1,category='everyday_sensitivity',pace_change_percent=5.,
            contrast_shift_log=dict(monthly=.004,ninety_day=.003))])
    before=deepcopy((data,influence));result=list(cases(data,influence))
    assert len(result)==5
    np.testing.assert_equal(result[2]['mask'],[True,False,True])
    np.testing.assert_equal(result[2]['y'],[2.3,2.32])
    assert result[2]['proposals']['monthly']==pytest.approx(.031)
    np.testing.assert_allclose(result[-1]['y'],[2.3,2.31+np.log1p(.05),2.32])
    assert result[-1]['proposals']['ninety_day']==pytest.approx(.013)
    assert (data,influence)==before


def test_fixed_sensitivity_cases_are_counts_not_a_binomial_coverage_experiment():
    result=summarize_compatibility([dict(label='everyday',covered_95=accepted,
        peak_difference_log=.001,evaluated_draws=3,observed_unconverged_kernels=[],optimizer_warning_draws=0)
        for accepted in (True,False)])
    assert result['everyday']['point_in_confidence_set']==dict(accepted=1,tested=2,all_accepted=False)
    assert 'wilson' not in str(result).lower()
