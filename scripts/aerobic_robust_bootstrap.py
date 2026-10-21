"""Offline fitted-null bootstrap for the unchanged production Huber statistic.

The Gaussian covariance model supplies null histories, not the displayed
point. This is a separate inference experiment, not a production dependency.
"""
from copy import deepcopy
from datetime import datetime, timedelta
import math

import numpy as np

from run_analysis.aerobic_progress import _period
from scripts.aerobic_joint_profile import GaussianChangeProfile
from scripts.audit_aerobic_three_layers import CONTRASTS


def robust_statistic(rows, covariance, *, target, value=0.):
    if target not in CONTRASTS or not math.isfinite(value):
        raise ValueError('Known contrast and finite proposed change required')
    end = max(datetime.fromisoformat(row['start_time']) for row in rows)
    current = _period(rows, end, covariance)
    prior = _period(rows, end-timedelta(days=28 if target == 'monthly' else 90), covariance)
    if current is None or prior is None:
        raise ValueError('Both comparison periods required')
    current_pace, prior_pace = current['pace_min_mile'], prior['pace_min_mile']
    estimate = math.log(prior_pace/current_pace)
    gradient = (np.array(prior['shared_pace_gradient'])/prior_pace
                - np.array(current['shared_pace_gradient'])/current_pace)
    variance = (prior['random_variance']/prior_pace**2
                + current['random_variance']/current_pace**2
                + max(0., float(gradient @ covariance @ gradient)))
    if variance <= 0:
        raise ValueError('Positive statistic scale required')
    se = math.sqrt(variance)
    return dict(estimate=estimate, scale=se, statistic=abs(estimate-value)/se,
                current_pace_min_mile=current_pace)


class RobustCurveBootstrap:
    def __init__(self, rows, measurement, gradients, covariance, groups, *, times, future_time=2.):
        self.rows = deepcopy(rows)
        self.covariance = np.asarray(covariance, float).copy()
        self.y = np.log([row['pace_min_mile'] for row in rows])
        dates = [datetime.fromisoformat(row['start_time']) for row in rows]
        if not dates or dates != sorted(dates):
            raise ValueError('Chronological observations required')
        days = np.array([(dates[-1]-day).total_seconds()/86400 for day in dates])
        expected = np.where(days < 28, 0, np.where(days < 56, 1, np.where(days >= 90, 2, 3)))
        if np.any(days >= 118) or set(groups) != {0, 1, 2, 3} or not np.array_equal(expected, groups):
            raise ValueError('Period groups must match the production 28-day comparison windows')
        self.model = GaussianChangeProfile(self.y, measurement, gradients, covariance, groups,
            times=times, future_time=future_time, mode='nuisance_residual')

    def evaluate(self, y, target, value):
        if len(y) != len(self.rows) or not np.isfinite(y).all():
            raise ValueError('Finite aligned simulated observations required')
        rows = [dict(row, pace_min_mile=math.exp(float(log_pace)))
                for row, log_pace in zip(self.rows, y, strict=True)]
        return robust_statistic(rows, self.covariance, target=target, value=value)

    def test(self, target, value, *, draws=127, seed=20261117, decision_only=False):
        if target not in CONTRASTS or not math.isfinite(value) or draws < 19:
            raise ValueError('Known contrast, finite null and at least 19 draws required')
        observed = self.evaluate(self.y, target, value)
        null = self.model.fit(CONTRASTS[target], value=value)
        sd = np.array(null['period_day_log_sd'])[self.model.groups]
        covariance = self.model.known + sd[:, None]*self.model.kernels[null['kernel']]*sd
        factor = np.linalg.cholesky(covariance)
        mean = self.model.x @ np.array(null['beta'])
        rng = np.random.default_rng(seed)
        exceedances = 0
        for evaluated in range(1, draws+1):
            y = mean + factor @ rng.normal(size=len(mean))
            simulated = self.evaluate(y, target, value)
            exceedances += simulated['statistic'] >= observed['statistic']
            lower = (1+exceedances)/(draws+1)
            upper = (1+exceedances+draws-evaluated)/(draws+1)
            if decision_only and (lower > .05 or upper <= .05):
                break
        return dict(**observed, pvalue=(1+exceedances)/(draws+1) if evaluated == draws else None,
            pvalue_bounds=[lower, upper], covered_95=lower > .05, bootstrap_draws=draws,
            evaluated_draws=evaluated, seed=seed, decision_only=decision_only,
            null_kernel=null['kernel'], null_day_log_sd=null['period_day_log_sd'],
            null_optimizer_warnings=null['unconverged_kernels'],
            scope='Original Huber statistic with frozen per-run absolute uncertainty/context; fitted Gaussian null, not a posterior probability.')
