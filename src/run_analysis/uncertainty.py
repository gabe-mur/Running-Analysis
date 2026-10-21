"""Variance accounting shared by progress estimates, not training decisions."""
from math import isfinite


def overlapping_variance(measurement: float, observed: float) -> float:
    """Two estimates of the same random error are floors, not additive causes.

    Observed residual scatter already contains measurement variation. Preserve
    either estimate when it is larger. Independent shared/calibration terms
    belong outside this envelope and must still be propagated by the caller.
    This does not by itself calibrate serial dependence or robust-fit error.
    """
    if not all(isfinite(v) and v >= 0 for v in (measurement, observed)):
        raise ValueError('Finite nonnegative variance components required')
    return max(measurement, observed)
