"""Local log-mean prototype with one value per exact historical anchor.

The existing Gaussian fitter and covariance families are unchanged. A level
uses only its own trailing 28 days; comparison-specific inference may fit
nuisance parameters jointly, but cannot overwrite these point values.
"""
from copy import deepcopy
import math

import numpy as np

from .aerobic_shared_point import fit_shared_state


class CoherentHistory:
    def __init__(self, y, measurement, gradients, covariance, times):
        self.y = np.array(y, dtype=float, copy=True)
        self.m = np.array(measurement, dtype=float, copy=True)
        self.g = np.array(gradients, dtype=float, copy=True)
        self.c = np.array(covariance, dtype=float, copy=True)
        self.t = np.array(times, dtype=float, copy=True)
        n = len(self.y)
        if (self.y.shape != (n,) or self.m.shape != (n,) or self.t.shape != (n,)
            or self.g.ndim != 2 or self.g.shape[0] != n or self.c.shape != (self.g.shape[1], self.g.shape[1])
            or not all(np.isfinite(v).all() for v in (self.y, self.m, self.g, self.c, self.t))
            or np.any(self.m <= 0) or np.any(np.diff(self.t) <= 0)
            or not np.allclose(self.c, self.c.T) or np.linalg.eigvalsh(self.c).min() < -1e-10):
            raise ValueError('Finite aligned chronological data, positive precision and PSD covariance required')
        for value in (self.y, self.m, self.g, self.c, self.t):
            value.setflags(write=False)
        self._levels = {}

    def level(self, anchor):
        anchor = float(anchor)
        if not math.isfinite(anchor):
            raise ValueError('Finite exact calendar anchor required')
        if anchor not in self._levels:
            indices = np.flatnonzero((self.t > anchor-28) & (self.t <= anchor))
            result = dict(anchor=anchor, available=bool(len(indices)), run_count=len(indices), indices=indices.tolist())
            if len(indices) == 1:
                # Preserve the observation without inventing between-day noise
                # or requiring eight runs before showing any initial level.
                result.update(log_pace=float(self.y[indices[0]]), pace_min_mile=math.exp(self.y[indices[0]]),
                    evidence='single_observation', conditional_variance=None, unconverged_kernels=[])
            elif len(indices) > 1:
                state = fit_shared_state(self.y[indices], self.m[indices], self.g[indices], self.c,
                    np.zeros(len(indices), dtype=int), times=self.t[indices]-anchor)
                result.update(log_pace=state['period_log_pace'][0], pace_min_mile=state['current_pace_min_mile'],
                    evidence='local_period_fit', kernel=state['kernel'], day_log_sd=state['period_day_log_sd'][0],
                    weights=state['conditional_mean_weights'][0],
                    conditional_variance=state['conditional_mean_covariance'][0][0],
                    unconverged_kernels=state['unconverged_kernels'])
            self._levels[anchor] = result
        return deepcopy(self._levels[anchor])

    def compare(self, anchor, days):
        if not math.isfinite(days) or days < 28:
            raise ValueError('Nonoverlapping comparison at least 28 days apart required')
        current = self.level(anchor); previous = self.level(float(anchor)-days)
        result = dict(anchor=float(anchor), prior_anchor=float(anchor)-days, days=days,
            available=current['available'] and previous['available'])
        if result['available']:
            change = previous['log_pace']-current['log_pace']
            result.update(estimate=change, change_percent=100*math.expm1(change),
                current_pace_min_mile=current['pace_min_mile'], prior_pace_min_mile=previous['pace_min_mile'],
                provisional=min(current['run_count'], previous['run_count']) < 2)
        return result

    def series(self, anchors):
        return [self.level(t) for t in sorted(set(float(v) for v in anchors))]
