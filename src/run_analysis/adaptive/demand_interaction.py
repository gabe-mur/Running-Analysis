"""Explicit fatigue-cost hypothesis; does NOT change measured training load.

Independent normalized duration/cardio/work axes remain the archived default.
The shared-running alternative treats harder-work dose as a surcharge on running
exposure: (running + work)^2 rather than running^2 + work^2. Its cross terms
are derived from that grouping, not a fitted physiological coupling coefficient.
Both orders, explicit sessions and continuation use the same bilinear contract.
This assumption needs sensitivity/athlete validation; it is not a recovery clock.
"""
import numpy as np

MODES=('independent_v1','running_work_shared_v1')


def interaction(left,right,mode='independent_v1'):
    if mode not in MODES:raise ValueError('Explicit supported demand interaction required')
    result=np.asarray(left)*np.asarray(right)
    if mode=='running_work_shared_v1':
        result=result.copy()
        result[2]+=left[0]*right[2]+left[2]*right[0]
    return result


def session_cost(own,prior,compression,mode='independent_v1'):
    return .5*interaction(own,own,mode)+compression*interaction(own,prior,mode)
