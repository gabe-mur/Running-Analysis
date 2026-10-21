from copy import deepcopy

import numpy as np
import pytest

from scripts.audit_aerobic_history_cause import (
    gls, cross_period_controls, calibration_controls, fitted_geometry,
)
from scripts.audit_aerobic_shared_history import snapshot
from tests.test_aerobic_shared_history import fixture


def geometry():
    n = 8; groups = np.repeat([0, 1], 4)
    x = np.eye(2)[groups]
    g = np.column_stack([np.linspace(0., 1., n), np.sin(np.arange(n)),
        np.cos(np.arange(n)), np.linspace(8., 1., n)])
    c = np.diag([.02**2, .003**2, .01**2, .002**2])
    c[0, 1] = c[1, 0] = .00001
    day = .02**2*.6**abs(np.arange(n)[:, None]-np.arange(n))
    m = np.full(n, .01**2); shared = g@c@g.T
    y = x@np.array([2.3, 2.35])+.02*np.sin(np.arange(n)*1.3)
    beta, _ = gls(y, x, np.diag(m)+shared+day)
    return dict(y=y, x=x, groups=groups, gradients=g, calibration_covariance=c,
        m=m, shared=shared, day=day, fit=dict(beta=beta.tolist()))


def test_cross_period_controls_preserve_marginal_information_and_positive_covariance():
    data = geometry(); before = deepcopy(data)
    results = cross_period_controls(data)
    assert all(v['minimum_covariance_eigenvalue'] > 0 for v in results.values())
    expected = []
    full = np.diag(data['m'])+data['shared']+data['day']
    for group in (0, 1):
        mask = data['groups'] == group
        local, _ = gls(data['y'][mask], np.ones((mask.sum(), 1)), full[np.ix_(mask, mask)])
        expected.append(local[0])
    np.testing.assert_allclose(results['shared_cross_False_serial_cross_False']['period_log_pace'], expected, atol=1e-12)
    for key in data:
        np.testing.assert_equal(data[key], before[key])


def test_joint_fit_uses_outside_residual_patterns_not_the_other_period_mean():
    data = geometry(); v = np.diag(data['m'])+data['shared']+data['day']
    baseline, _ = gls(data['y'], data['x'], v)
    beta, _ = gls(data['y']+data['x']@np.array([0., .2]), data['x'], v)
    np.testing.assert_allclose(beta-baseline, [0., .2], atol=1e-12)


def test_calibration_random_effect_reconstructs_joint_mean_and_retains_the_core_block():
    data = geometry(); result = calibration_controls(data)
    assert result['conditional_correction_reconstructs_joint_mean']
    np.testing.assert_allclose(result['controls']['core_cross_True_heat_cross_True'], data['fit']['beta'], atol=1e-12)
    without_shared_cross = cross_period_controls(data)['shared_cross_False_serial_cross_True']['period_log_pace']
    np.testing.assert_allclose(result['controls']['core_cross_False_heat_cross_False'], without_shared_cross, atol=1e-12)
    data['calibration_covariance'][3, 0] = .001
    with pytest.raises(ValueError, match='Independent heat'):
        calibration_controls(data)


def test_full_geometry_is_the_existing_candidate_not_a_changed_estimator():
    rows, m, g, cov, end = fixture(); before = deepcopy((rows, m, g, cov))
    expected = snapshot(rows, m, g, cov, end)
    actual = fitted_geometry(rows, m, g, cov, end)
    np.testing.assert_allclose(actual['fit']['beta'], expected['state']['period_log_pace'], atol=1e-12)
    assert actual['fit']['kernel'] == expected['state']['kernel']
    for current, previous in zip((rows, m, g, cov), before):
        np.testing.assert_equal(current, previous)
