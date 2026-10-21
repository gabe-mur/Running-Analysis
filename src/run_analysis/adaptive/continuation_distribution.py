"""Named continuation survival contracts; no athlete-dependent tuning.

The cubic is the lowest-degree curve with unit/zero endpoint values and zero
slope at both ends. It smooths truncation, not physical fatigue or readiness.
"""
from math import isfinite


def c1_survival(fraction):
    if not isfinite(fraction):
        raise ValueError('Finite normalized continuation position required')
    u=min(1.,max(0.,fraction))
    return (1-u)**2*(1+2*u)


def c1_handoff_probabilities(count):
    if type(count) is not int or count<1:
        raise ValueError('Positive integer continuation handoff count required')
    return tuple(c1_survival(i/count)-c1_survival((i+1)/count) for i in range(count))
