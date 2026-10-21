"""Offline Q5 candidate: shared speed/elapsed-time response, not live recovery.

One extra equilibrium covariate (elapsed minutes), learned only from preceding
runs. It is an empirical finite-duration approximation, not a physiological
drift law, load multiplier, phase correction or performance/readiness signal.
"""
from dataclasses import dataclass
from datetime import timedelta
from math import exp, expm1, isfinite

import numpy as np

from run_analysis.athlete_projection import aware
from run_analysis.hr_response import TAUS, _check
from run_analysis.signal_structure import SignalSpan

VERSION = 'experimental_elapsed_hr_response_v1'


def elapsed_statistics(rows, taus=TAUS):
    """Same equal-session endpoint moments, with an exact ramp-input response."""
    _check(rows)
    if not rows or not taus or any(not isfinite(t) or t <= 0 for t in taus):
        raise ValueError('Nonempty observations and positive response times required')
    tau = np.asarray(taus)
    total = np.zeros((len(taus), 4, 4))
    feature, offset, end = np.zeros((len(taus), 3)), np.zeros(len(taus)), None
    duration = sum(r.end-r.start for r in rows)
    for r in rows:
        if end is None or abs(r.start-end)>1e-8:
            feature[:] = 0
            offset[:] = r.first
        dt = r.end-r.start
        decay = np.exp(-dt/tau)
        response = -np.expm1(-dt/tau)
        next_feature = feature*decay[:, None]
        next_feature[:, :2] += response[:, None]*np.array((1., r.speed))
        # Elapsed time does not restart at phase changes or unknown-input gaps.
        next_feature[:, 2] += (r.start*response+dt-tau*response)/60
        next_offset = offset*decay
        for fraction in (.5-1/(2*np.sqrt(3)), .5+1/(2*np.sqrt(3))):
            values = np.empty((len(taus), 4))
            values[:, :3] = feature+(next_feature-feature)*fraction
            values[:, 3] = r.first+(r.last-r.first)*fraction-(offset+(next_offset-offset)*fraction)
            total += np.einsum('ni,nj->nij', values, values)*(dt/(2*duration))
        feature, offset, end = next_feature, next_offset, r.end
    return total


@dataclass(frozen=True)
class ElapsedEvidence:
    key: str
    end: object
    available_at: object
    accounting_key: str
    statistics: np.ndarray
    speed_domain: tuple[float, float]
    elapsed_domain_minutes: tuple[float, float]
    eligible: bool = True
    taus: tuple[float, ...] = TAUS

    def __post_init__(self):
        aware(self.end)
        aware(self.available_at)
        if (not self.key or not self.accounting_key or self.end > self.available_at
                or not self.taus or any(not isfinite(t) or t<=0 for t in self.taus)
                or any(not all(isfinite(v) for v in d) or not 0<=d[0]<=d[1]
                       for d in (self.speed_domain, self.elapsed_domain_minutes))):
            raise ValueError('Causal identified evidence and finite domains required')
        stats = np.array(self.statistics, dtype=float, copy=True)
        if stats.shape != (len(self.taus), 4, 4) or not np.isfinite(stats).all():
            raise ValueError('Finite elapsed-response statistics matching grid required')
        stats.setflags(write=False)
        object.__setattr__(self, 'statistics', stats)


def fit_elapsed_response(history, *, as_of, accounting_key, lookback_days=30., taus=TAUS):
    aware(as_of)
    if not accounting_key or not isfinite(lookback_days) or lookback_days<=0:
        raise ValueError('Accounting identity and positive history window required')
    accepted = sorted((e for e in history if e.eligible and e.accounting_key==accounting_key
        and as_of-timedelta(days=lookback_days)<e.end<=e.available_at<=as_of), key=lambda e:e.key)
    if len({e.key for e in accepted})!=len(accepted):
        raise ValueError('Duplicate source session')
    if any(e.taus!=taus for e in accepted):
        raise ValueError('Incompatible response grids')
    result = dict(version=VERSION, issued_at=as_of, accounting_key=accounting_key,
        source_keys=tuple(e.key for e in accepted), dynamic=None,
        status='no_supported_elapsed_response', uncertainty_calibrated=False, planner_influence=False)
    if not accepted:
        return result
    stats = sum(e.statistics for e in accepted)/len(accepted)
    candidates = []
    for index, (tau, s) in enumerate(zip(taus, stats)):
        x, xy = s[:3, :3], s[:3, 3]
        if np.linalg.matrix_rank(x)<3:
            continue
        coefficients = np.linalg.solve(x, xy)
        if not np.isfinite(coefficients).all() or coefficients[1]<=0:
            continue
        mse = max(0., float(s[3, 3]-2*coefficients@xy+coefficients@x@coefficients))
        model = dict(version=VERSION, intercept=float(coefficients[0]), slope=float(coefficients[1]),
            elapsed_coefficient_per_minute=float(coefficients[2]), tau_seconds=tau,
            training_rmse_reserve=float(np.sqrt(mse)), tau_at_search_boundary=index in (0, len(taus)-1),
            speed_domain=(min(e.speed_domain[0] for e in accepted), max(e.speed_domain[1] for e in accepted)),
            elapsed_domain_minutes=(min(e.elapsed_domain_minutes[0] for e in accepted),
                                    max(e.elapsed_domain_minutes[1] for e in accepted)))
        candidates.append((mse, index, model))
    if candidates:
        result.update(dynamic=min(candidates, key=lambda c:(c[0], c[1]))[2], status='experimental_fit')
    return result


def predict_elapsed_response(rows, model):
    """Exact response to constant speed and elapsed-time ramp in each interval."""
    _check(rows)
    if (model.get('version')!=VERSION
            or not all(isfinite(model[k]) for k in ('intercept','slope','elapsed_coefficient_per_minute','tau_seconds'))
            or model['slope']<=0 or model['tau_seconds']<=0):
        raise ValueError('Explicit finite experimental response required')
    spans, state, end, starts, speed_outside, time_outside = [], None, None, 0, 0., 0.
    for r in rows:
        if end is None or abs(r.start-end)>1e-8:
            state = r.first
            starts += 1
        dt, tau = r.end-r.start, model['tau_seconds']
        decay, response = exp(-dt/tau), -expm1(-dt/tau)
        target = model['intercept']+model['slope']*r.speed
        last = state*decay+target*response+model['elapsed_coefficient_per_minute']*(r.start*response+dt-tau*response)/60
        spans.append(SignalSpan(r.start, r.end, state, last))
        if not model['speed_domain'][0]<=r.speed<=model['speed_domain'][1]:
            speed_outside += dt
        lo, hi = (x*60 for x in model['elapsed_domain_minutes'])
        time_outside += dt-max(0., min(r.end, hi)-max(r.start, lo))
        state, end = last, r.end
    return tuple(spans), dict(segment_initializations=starts, speed_extrapolation_seconds=speed_outside,
        elapsed_extrapolation_seconds=time_outside,
        hr_outside_reference=any(not (0<=s.first<=1 and 0<=s.last<=1) for s in spans))
