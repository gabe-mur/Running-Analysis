"""Predeclared dependence controls for the Progress prototype, not recovery."""
import math
import numpy as np


def correlation_matrix(times, *, half_life_days=None, run_rho=None):
    times=np.asarray(times,float)
    if times.ndim!=1 or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError('Finite strictly increasing run dates required')
    if (half_life_days is None)==(run_rho is None):
        raise ValueError('Specify exactly one calendar or run-index kernel')
    if run_rho is not None:
        if not 0<=run_rho<1:raise ValueError('Run correlation must be in [0,1)')
        return run_rho**np.abs(np.arange(len(times))[:,None]-np.arange(len(times)))
    if not math.isfinite(half_life_days) or half_life_days<0:raise ValueError('Nonnegative finite half life required')
    return np.eye(len(times)) if half_life_days==0 else np.exp2(-np.abs(times[:,None]-times)/half_life_days)


def kernel_grid(times,future_time):
    augmented=np.r_[times,future_time]
    kernels={f'calendar_half_life_{h}d':correlation_matrix(augmented,half_life_days=h) for h in (0,1,3,7,14)}
    kernels.update({f'run_rho_{rho}':correlation_matrix(augmented,run_rho=rho) for rho in (.3,.5,.7,.9)})
    return kernels
