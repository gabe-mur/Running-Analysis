import math

import numpy as np
import pytest

from scripts.aerobic_local_trend import LocalPaceTrend


def fit(y,times,**kwargs):
    return LocalPaceTrend(y,np.full(len(y),.01**2),np.zeros((len(y),1)),[[.01**2]],times,**kwargs)


def test_direct_fit_preserves_constant_and_linear_log_pace_including_endpoints():
    t=np.array([-70.,-60.,-40.,-35.,-20.,-5.,0.])
    for slope in (0.,-.001,.001):
        curve=fit(math.log(10)+slope*t,t)
        for at in (-70.,-47.5,-20.,0.):
            assert curve.level(at)['log_pace']==pytest.approx(math.log(10)+slope*at,abs=1e-12)


def test_local_trend_is_a_regression_not_a_connector_through_every_run():
    t=np.arange(-90.,1.,3.);y=np.full(len(t),math.log(10.));y[15]+=math.log(1.1)
    curve=fit(y,t)
    assert 10.<curve.level(t[15])['pace_min_mile']<10.2
    assert curve.level(t[15])['pace_min_mile']!=pytest.approx(11.)


def test_trend_does_not_extrapolate_or_bridge_unobserved_long_gaps():
    curve=fit(np.log([10.,10.,9.,9.]),[-100.,-95.,-5.,0.])
    assert not curve.level(-101.)['available']
    assert not curve.level(-50.)['available']
    assert not curve.level(1.)['available']
    assert curve.level(-98.)['pace_min_mile']==pytest.approx(10.)


def test_historical_trend_uses_later_evidence_and_is_explicitly_retrospective():
    curve=fit(np.log([10.,10.,9.]),[-20.,-10.,0.])
    assert curve.level(-10.)['future_observation_weight_present']
    assert not curve.level(0.)['future_observation_weight_present']


def test_single_initial_observation_can_be_shown_but_has_no_invented_trend():
    curve=fit([math.log(10.)],[0.])
    assert curve.level(0.)['pace_min_mile']==pytest.approx(10.)
    assert not curve.level(-1.)['available']


def test_evaluation_order_or_viewport_does_not_change_a_fitted_value():
    t=np.arange(-120.,1.,3.);curve=fit(math.log(10)+.02*np.sin(t/40),t)
    expected=curve.level(-14.)
    for at in (-110.,0.,-50.):curve.level(at)
    assert curve.level(-14.)==expected


@pytest.mark.parametrize('bandwidth',[0,-1,float('nan')])
def test_invalid_bandwidth_is_not_silently_accepted(bandwidth):
    with pytest.raises(ValueError):fit([math.log(10)],[0.],bandwidth_days=bandwidth)
