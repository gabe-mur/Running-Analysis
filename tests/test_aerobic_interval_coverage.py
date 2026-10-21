from copy import deepcopy
from datetime import timedelta
import math

import numpy as np
import pytest

from scripts.audit_aerobic_interval_coverage import (
    checks, coverage, evaluate, period_parts, simulate_measurement, wilson,
)
from tests.test_small_aerobic_changes import rows
from scripts.audit_progress_responsiveness import at
from run_analysis.aerobic_progress import build_aerobic_progress


def test_decomposition_and_contrasts_match_production_without_mutation():
    source=rows(); before=deepcopy(source); end=at(source[-1]); cov=np.array([[.0004]])
    for r in source:r['shared_log_speed_gradient']=[.4]
    before=deepcopy(source)
    actual=evaluate(source,end,cov)
    oracle=build_aerobic_progress(source,as_of=end,covariance=cov,history_days=1)
    assert actual['level']['estimate']==oracle['current']['pace_min_mile']
    for key, field in (('monthly','change'),('ninety_day','change_90d')):
        assert 100*math.expm1(actual[key]['estimate'])==pytest.approx(oracle[field]['speed_change_percent'])
    p,parts=period_parts(source,end,cov)
    assert max(parts['measurement'],parts['day'])+parts['shared']==pytest.approx((p['uncertainty_95_min_mile']/1.96)**2)
    assert source==before


def test_truth_is_trailing_contrast_not_endpoint_gain():
    source=rows(); end=at(source[-1]); onset=end-timedelta(days=84)
    from scripts.audit_small_aerobic_changes import simulate_rows
    latent=simulate_rows(source,np.zeros(len(source)),10,onset,.02,84)
    result=evaluate(latent,end,np.zeros((1,1)))
    assert 0<100*math.expm1(result['monthly']['estimate'])<100*math.expm1(result['ninety_day']['estimate'])<2


def test_coverage_counts_known_truth_instead_of_positive_direction():
    samples=[dict(estimate=x,measurement=1.,day=0.,shared=0.,established=True) for x in (0.,1.,2.,3.)]
    result=coverage(samples,0.,('measurement','day','shared'))
    assert result['coverage']['0.95']['rate']==.5
    assert result['coverage']['0.8']['rate']==.5
    assert result['bias']==1.5
    assert wilson(95,100)[0]<.95<wilson(95,100)[1]


def test_one_shared_draw_is_coherent_across_the_entire_history():
    source=rows()
    for r in source:
        r['pace_min_mile']=10.
        r['shared_log_speed_gradient']=[.25]
        r['uncertainty_95_min_mile']=1.96*.625
    # All claimed variance is shared, leaving no independent measurement noise.
    observed=simulate_measurement(source,np.array([[.0625]]),np.random.default_rng(5),True)
    ratios=[r['pace_min_mile']/10 for r in observed]
    np.testing.assert_allclose(ratios,ratios[0],atol=1e-10)
    result=evaluate(observed,at(source[-1]),np.array([[.0625]]))
    assert abs(result['ninety_day']['estimate'])<1e-8
    assert result['ninety_day']['shared']<1e-20
    assert result['level']['shared']>0


def test_small_audit_is_reproducible_and_all_scenarios_are_retained():
    source=rows(); before=deepcopy(source); end=at(source[-1]); cov=np.zeros((1,1))
    a=checks(source,cov,end,draws=2,seed=123)
    b=checks(source,cov,end,draws=2,seed=123)
    assert a==b and source==before
    assert len(a['cells'])==80
    assert {c['shared_calibration_draw'] for c in a['cells']}=={False,True}
    assert {c['endpoint_gain_percent'] for c in a['cells']}=={0,.5,1,2}
