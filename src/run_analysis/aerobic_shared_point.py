"""Shared point fit using the audited full Gaussian likelihood.

No new covariance family or tuning coefficient. Contrasts read a single fitted
mean vector; they never refit the current level for a selected comparison.
Conditional covariance below is diagnostic, not a calibrated confidence band.
"""
import math

import numpy as np

from .aerobic_gaussian import GaussianChangeProfile


def fit_shared_state(y, measurement, gradients, covariance, groups, *, times, future_time=2.):
    model = GaussianChangeProfile(y, measurement, gradients, covariance, groups,
        times=times, future_time=future_time, mode='ml')
    current = np.eye(model.p)[0]
    fit = model.fit(current)
    sd = np.array(fit['period_day_log_sd'])[model.groups]
    v = model.known + sd[:, None]*model.kernels[fit['kernel']]*sd
    inv_x = np.linalg.solve(v, model.x)
    beta_cov = np.linalg.inv(model.x.T @ inv_x)
    weights = beta_cov @ inv_x.T
    return dict(version='aerobic-shared-full-ml-v1',
        period_log_pace=list(fit['beta']), current_pace_min_mile=math.exp(fit['beta'][0]),
        conditional_mean_covariance=beta_cov.tolist(), conditional_mean_weights=weights.tolist(),
        kernel=fit['kernel'], period_day_log_sd=list(fit['period_day_log_sd']),
        objective=fit['objective'], unconverged_kernels=list(fit['unconverged_kernels']),
        conditional_only='Covariance holds fitted day variances and kernel fixed; not a calibrated interval.')


def state_contrast(state, contrast):
    beta = np.asarray(state['period_log_pace'], float)
    vector = np.asarray(contrast, float)
    if vector.shape != beta.shape or not np.isfinite(vector).all() or not np.any(vector):
        raise ValueError('Finite nonzero contrast aligned with fitted periods required')
    cov = np.asarray(state['conditional_mean_covariance'], float)
    estimate = float(vector @ beta)
    return dict(estimate=estimate, change_percent=100*math.expm1(estimate),
        conditional_se=math.sqrt(max(0., float(vector @ cov @ vector))),
        current_pace_min_mile=state['current_pace_min_mile'])
