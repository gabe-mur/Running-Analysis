from copy import deepcopy
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from scripts.audit_aerobic_curve_compatibility import conditional_rows, robust_point, profile_points


def rows():
    end = datetime(2026, 10, 14, tzinfo=timezone.utc)
    return [dict(activity_id=i, start_time=(end-timedelta(days=117-i*3)).isoformat(),
        pace_min_mile=10.+.03*(i % 5), uncertainty_95_min_mile=.6,
        shared_log_speed_gradient=[1.], trend_weight=.5 if i % 2 else 1.,
        included_in_trend=True) for i in range(40)]


def test_conditional_precision_undoes_context_once_and_retains_shared_error():
    data=rows(); before=deepcopy(data); cov=np.array([[.01**2]])
    measurement=np.full(len(data),.02**2)
    updated=conditional_rows(data,measurement,cov)
    for source, changed, variance in zip(data,updated,measurement,strict=True):
        actual=(changed['uncertainty_95_min_mile']/1.96/source['pace_min_mile'])**2
        assert actual == pytest.approx(variance*source['trend_weight']+.01**2)
        assert changed['trend_weight']==source['trend_weight']
    assert data == before
    with pytest.raises(ValueError): conditional_rows(data,measurement[:-1],cov)


def test_existing_current_state_does_not_depend_on_history_selector():
    data=rows();before=deepcopy(data)
    result=robust_point(data,datetime(2026,10,14,tzinfo=timezone.utc),np.array([[.01**2]]))
    assert result['history_selector_invariant']
    assert result['current_pace_min_mile']>0
    assert data==before


def test_profile_compatibility_reports_both_contrasts_without_mutating_evidence():
    rng=np.random.default_rng(17);groups=np.repeat([2,3,1,0],6)
    y=2.3+.01*groups+rng.normal(0,.02,len(groups)); before=y.copy()
    result=profile_points(y,np.full(len(y),.01**2),np.ones((len(y),1)),
                          np.array([[.005**2]]),groups,np.arange(len(y))*3.-69)
    assert set(result)=={'ml','nuisance_residual'}
    assert result['ml']['current_level_disagreement_seconds_per_mile'] < 1e-7
    for mode in result.values():
        assert set(mode['comparisons'])=={'monthly','ninety_day'}
        assert all(not fit['unconverged_kernels'] for fit in mode['comparisons'].values())
    np.testing.assert_equal(y,before)
