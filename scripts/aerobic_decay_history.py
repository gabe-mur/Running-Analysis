"""Offline positive-precision decay experiment; not imported by the app.

Evidence ages smoothly without fitting a new covariance/kernel at each date.
The half-life is an engineering assumption, not a physiological constant.
Shared calibration covariance is retained in a conditional sensitivity only;
there is no calibrated fitness interval or inferred detraining during a gap.
"""
from copy import deepcopy
import math

import numpy as np


class PrecisionDecayHistory:
    def __init__(self, y, measurement, gradients, covariance, times, *, half_life_days=14., boxcar_days=None):
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
            or not np.allclose(self.c, self.c.T) or np.linalg.eigvalsh(self.c).min(initial=0.) < -1e-10):
            raise ValueError('Finite chronological aligned observations, positive precision and PSD covariance required')
        if not math.isfinite(half_life_days) or half_life_days <= 0:
            raise ValueError('Positive finite half-life required')
        if boxcar_days is not None and (not math.isfinite(boxcar_days) or boxcar_days <= 0):
            raise ValueError('Positive finite boxcar width required')
        self.half_life = float(half_life_days)
        self.boxcar = boxcar_days
        for value in (self.y, self.m, self.g, self.c, self.t):
            value.setflags(write=False)

    def level(self, anchor):
        anchor = float(anchor)
        if not math.isfinite(anchor):
            raise ValueError('Finite calendar anchor required')
        mask = self.t <= anchor
        if self.boxcar is not None:
            mask &= self.t > anchor-self.boxcar
        ix = np.flatnonzero(mask)
        result = dict(anchor=anchor, available=bool(len(ix)), run_count=len(ix), indices=ix.tolist())
        if not len(ix):
            return result
        g = self.g[ix]
        marginal = self.m[ix]+np.einsum('ij,jk,ik->i', g, self.c, g)
        # Subtract the latest observed date, not the anchor: normalized weights
        # do not invent a changing performance level when no new run arrives.
        age = self.t[ix[-1]]-self.t[ix]
        log_w = -np.log(marginal)
        if self.boxcar is None:
            log_w -= math.log(2.)*age/self.half_life
        w = np.exp(log_w-log_w.max()); w /= w.sum()
        log_pace = float(w@self.y[ix])
        shared_gradient = w@g
        conditional = float((w*w)@self.m[ix]+shared_gradient@self.c@shared_gradient)
        return deepcopy(dict(result, log_pace=log_pace, pace_min_mile=math.exp(log_pace), weights=w.tolist(),
            conditional_variance=conditional, effective_run_count=float(1/(w@w)),
            maximum_run_weight=float(w.max()), weighted_age_days=float(w@(anchor-self.t[ix])),
            old_evidence_weight=float(w[anchor-self.t[ix] >= 28].sum()),
            days_since_observation=float(anchor-self.t[ix[-1]]),
            uncertainty_kind='Fixed-weight measurement/calibration sensitivity; excludes day noise and state uncertainty'))
