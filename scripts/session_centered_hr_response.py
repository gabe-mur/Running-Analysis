"""Offline fixed-effects HR-response experiment; not a production predictor.

Separate session baselines while estimating a common within-session speed slope
and response time. Forecast baseline is the equal-session historical mean, NEVER
a held-out session's fitted intercept. No regularization/phase correction weights.
"""
from datetime import timedelta
from math import isfinite
import numpy as np

from run_analysis.athlete_projection import aware
from run_analysis.hr_response import TAUS


def fit_session_centered(history, *, as_of, accounting_key, lookback_days=30., taus=TAUS):
    aware(as_of)
    if not isfinite(lookback_days) or lookback_days <= 0:
        raise ValueError('Positive history window required')
    accepted = sorted((e for e in history if e.eligible and e.accounting_key == accounting_key
        and as_of-timedelta(days=lookback_days) < e.end <= e.available_at <= as_of),key=lambda e:e.key)
    if len({e.key for e in accepted}) != len(accepted):
        raise ValueError('Duplicate source session')
    if any(e.taus != taus for e in accepted):
        raise ValueError('Incompatible response grids')
    result = dict(issued_at=as_of,accounting_key=accounting_key,source_keys=tuple(e.key for e in accepted),
        dynamic=None,status='no_supported_within_session_relation',uncertainty_calibrated=False)
    if not accepted:
        return result
    candidates = []
    n = len(accepted)
    for i,tau in enumerate(taus,start=1):
        stats = [e.statistics[i] for e in accepted]
        # One nuisance intercept per training session, one common speed slope.
        # Duration-normalized per-session stats give each run one vote.
        normal, rhs = np.zeros((n+1,n+1)),np.zeros(n+1)
        for j,s in enumerate(stats):
            normal[j,j] = s[0,0]
            normal[j,n] = normal[n,j] = s[0,1]
            normal[n,n] += s[1,1]
            rhs[j] = s[0,2]
            rhs[n] += s[1,2]
        if np.linalg.matrix_rank(normal) < n+1:
            continue
        coef = np.linalg.solve(normal,rhs)
        if not np.isfinite(coef).all() or coef[-1] <= 0:
            continue
        mse = max(0.,(sum(s[2,2] for s in stats)-2*coef@rhs+coef@normal@coef)/n)
        model = dict(intercept=float(np.mean(coef[:-1])),slope=float(coef[-1]),tau_seconds=tau,
            training_rmse_reserve=float(np.sqrt(mse)),tau_at_search_boundary=i in (1,len(taus)),
            speed_domain=(min(e.speed_domain[0] for e in accepted),max(e.speed_domain[1] for e in accepted)),
            session_intercepts={e.key:float(a) for e,a in zip(accepted,coef[:-1])})
        candidates.append((mse,i,model))
    if candidates:
        result.update(dynamic=min(candidates,key=lambda c:(c[0],c[1]))[2],status='experimental_fit')
    return result
