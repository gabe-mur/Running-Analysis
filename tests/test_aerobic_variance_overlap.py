from copy import deepcopy
from datetime import timedelta

import numpy as np
import pytest

from scripts.audit_aerobic_variance_overlap import candidate_period, evaluate_candidates, stress_checks, persistence_inventory, VARIANTS
from scripts.audit_aerobic_interval_coverage import evaluate, checks
from scripts.audit_progress_responsiveness import at
from tests.test_small_aerobic_changes import rows


def test_candidates_keep_production_point_estimates_and_variance_floors():
    source=rows();before=deepcopy(source);end=at(source[-1]);cov=np.array([[.0001]])
    actual=evaluate_candidates(source,end,cov);shipped=evaluate(source,end,cov)
    for key, value in actual.items():
        assert value['estimate']==shipped[key]['estimate']
        assert value['shared']==pytest.approx(shipped[key]['shared'])
        for name in VARIANTS:
            assert value[name]>=value['measurement']
            assert np.isfinite(value[name])
    assert source==before


def test_identical_runs_keep_measurement_floor_instead_of_zero_uncertainty():
    source=rows()
    for r in source:r['pace_min_mile']=10.
    p,parts=candidate_period(source,at(source[-1]),np.zeros((1,1)))
    assert parts['measurement']>0
    assert parts['variance_envelope']==parts['measurement']
    assert parts['excess_moments']==pytest.approx(parts['measurement'])


def test_large_scatter_is_not_ignored_when_measurement_error_is_small():
    source=rows()
    for i,r in enumerate(source):
        r['pace_min_mile']=10+(.9 if (i//2)%2 else -.9)
        r['uncertainty_95_min_mile']=.05
    p,parts=candidate_period(source,at(source[-1]),np.zeros((1,1)))
    assert parts['variance_envelope']>parts['measurement']
    assert parts['excess_moments']>parts['measurement']


def test_shared_error_remains_in_level_but_cancels_for_equal_period_exposure():
    source=rows()
    for r in source:r['pace_min_mile']=10.;r['shared_log_speed_gradient']=[.4]
    result=evaluate_candidates(source,at(source[-1]),np.array([[.0004]]))
    assert result['level']['shared']>0
    assert result['monthly']['shared']==pytest.approx(0,abs=1e-15)


def test_candidate_and_stress_controls_retain_predeclared_cases_and_are_repeatable():
    source=rows();cov=np.zeros((1,1))
    result=checks(source,cov,at(source[-1]),2,77,evaluator=evaluate_candidates,extra_variants=VARIANTS)
    assert len(result['cells'])==80
    for cell in result['cells']:
        for kind in ('level','monthly','ninety_day'):
            assert all(v in cell['results'][kind] for v in VARIANTS)
    a=stress_checks(2,123);b=stress_checks(2,123)
    assert a==b and len(a['cells'])==12
    assert {c['measurement_error_scale'] for c in a['cells']}=={.5,1.,2.}


def test_persistence_inventory_is_descriptive_and_does_not_cross_long_breaks():
    source=rows();before=deepcopy(source);end=at(source[-1])
    result=persistence_inventory(source,end)
    assert len(result['cases'])==4
    assert {c['lookback_days'] for c in result['cases']}=={90,180}
    assert all(c['lags'][0]['pairs']==c['runs']-1 for c in result['cases'])
    assert source==before
