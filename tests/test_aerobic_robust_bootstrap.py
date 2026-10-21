from copy import deepcopy
from datetime import datetime, timedelta, timezone
import math

import numpy as np
import pytest

from run_analysis.aerobic_progress import build_aerobic_progress
from scripts.aerobic_robust_bootstrap import RobustCurveBootstrap, robust_statistic


def fixture():
    end = datetime(2026, 10, 14, tzinfo=timezone.utc)
    times=np.array([-115,-111,-107,-103,-99,-95,-87,-83,-79,-75,-71,-67,
        -55,-51,-47,-43,-39,-35,-27,-23,-19,-15,-11,0],float)
    groups=np.repeat([2,3,1,0],6)
    rng=np.random.default_rng(22)
    y=2.3+.012*groups+rng.normal(0,.025,len(times))
    rows=[dict(activity_id=i,start_time=(end+timedelta(days=float(day))).isoformat(),
        pace_min_mile=math.exp(float(value)),uncertainty_95_min_mile=.5,
        trend_weight=1.,included_in_trend=True,shared_log_speed_gradient=[1.])
        for i,(day,value) in enumerate(zip(times,y,strict=True))]
    covariance=np.array([[.005**2]])
    model=RobustCurveBootstrap(rows,np.full(len(rows),.01**2),np.ones((len(rows),1)),
        covariance,groups,times=times)
    return rows,covariance,model


def test_statistic_preserves_the_exact_production_curve_point_and_scale():
    rows,cov,model=fixture()
    state=build_aerobic_progress(rows,as_of=datetime(2026,10,14,tzinfo=timezone.utc),covariance=cov)
    for target,key in [('monthly','change'),('ninety_day','change_90d')]:
        result=model.evaluate(model.y,target,0.)
        expected=state[key]
        assert result['current_pace_min_mile']==state['current']['pace_min_mile']
        assert result['estimate']==pytest.approx(math.log1p(expected['speed_change_percent']/100))
        low,high=expected['estimated_change_range_95_percent']
        assert result['scale']==pytest.approx((math.log1p(high/100)-math.log1p(low/100))/3.92)


def test_robust_bootstrap_keeps_evidence_and_current_point_fixed_across_nulls():
    rows,cov,model=fixture();before=deepcopy(rows)
    # Keep this deterministic unit test small; the actual audit uses all nine.
    model.model.kernels={'independent':np.eye(len(rows))}
    first=model.test('ninety_day',0.,draws=19,seed=19)
    second=model.test('ninety_day',.01,draws=19,seed=19)
    assert first['estimate']==second['estimate']
    assert first['current_pace_min_mile']==second['current_pace_min_mile']
    assert first==model.test('ninety_day',0.,draws=19,seed=19)
    assert rows==before


@pytest.mark.parametrize('value',[0.,.5])
def test_early_robust_bootstrap_uses_the_same_fixed_budget_decision(value):
    rows,_,model=fixture();model.model.kernels={'independent':np.eye(len(rows))}
    full=model.test('ninety_day',value,draws=63,seed=21)
    early=model.test('ninety_day',value,draws=63,seed=21,decision_only=True)
    assert full['covered_95']==early['covered_95']
    assert early['pvalue_bounds'][0] <= full['pvalue'] <= early['pvalue_bounds'][1]
    if early['evaluated_draws']<63: assert early['pvalue'] is None


def test_robust_bootstrap_rejects_missing_periods_and_invalid_draw_budget():
    rows,cov,model=fixture()
    with pytest.raises(ValueError):robust_statistic(rows[-6:],cov,target='ninety_day')
    with pytest.raises(ValueError):model.test('ninety_day',0.,draws=18)
    with pytest.raises(ValueError):model.evaluate(model.y[:-1],'monthly',0.)
    with pytest.raises(ValueError):RobustCurveBootstrap(rows,np.full(len(rows),.01**2),
        np.ones((len(rows),1)),cov,np.roll(model.model.groups,1),times=model.model.template['times'])


def test_statistic_bootstrap_fits_only_the_null_not_a_new_point_in_each_replica(monkeypatch):
    rows,_,model=fixture();model.model.kernels={'independent':np.eye(len(rows))}
    original=model.model.fit;calls=[]
    def record(contrast,**kwargs):
        calls.append(kwargs)
        return original(contrast,**kwargs)
    monkeypatch.setattr(model.model,'fit',record)
    model.test('ninety_day',.012,draws=19,seed=31)
    assert calls == [dict(value=.012)]


def test_shared_covariance_and_full_budget_rank_match_explicit_seeded_draws():
    rows,_,model=fixture();model.model.kernels={'independent':np.eye(len(rows))}
    value=.01;contrast=[-1,0,1,0];null=model.model.fit(contrast,value=value)
    sd=np.array(null['period_day_log_sd'])[model.model.groups]
    covariance=model.model.known+np.diag(sd**2)
    assert covariance[0,1] > 0  # One coherent shared calibration, not independent errors.
    factor=np.linalg.cholesky(covariance);mean=model.model.x@np.array(null['beta'])
    observed=model.evaluate(model.y,'ninety_day',value)['statistic']
    rng=np.random.default_rng(33)
    exceedances=sum(model.evaluate(mean+factor@rng.normal(size=len(rows)),
        'ninety_day',value)['statistic']>=observed for _ in range(19))
    result=model.test('ninety_day',value,draws=19,seed=33)
    assert result['pvalue']==(1+exceedances)/20
    assert result['evaluated_draws']==19
