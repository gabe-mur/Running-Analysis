"""Offline speed-to-HR system identification, not a recovery or fitness score.

One equilibrium relation and one response time, with no workout-type factors.
Fit on earlier sessions only. Recorded speed is an input, NOT a forecast of
sustainable pace. Unknown trace gaps reset state and remain unscored.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import exp, isfinite

import numpy as np

from .athlete_projection import aware
from .signal_structure import SignalSpan
from .training_signals import HeartRateReference


TAUS = tuple(float(x) for x in np.geomspace(1., 600., 41))


@dataclass(frozen=True)
class ResponseInterval:
    start: float
    end: float
    speed: float
    first: float
    last: float

    def __post_init__(self):
        if (not all(isfinite(x) for x in (self.start, self.end, self.speed, self.first, self.last))
                or not 0 <= self.start < self.end or self.speed < 0
                or not 0 <= self.first <= 1 or not 0 <= self.last <= 1):
            raise ValueError('Valid time, speed and HR reserve required')


def intervals_from_activity(activity, config):
    from .movement import classify_movement
    reference = HeartRateReference(config['resting_hr'], config['max_hr'], 'audit_configuration')
    origin = activity.start_time_utc
    result = []
    for i in classify_movement(activity.trackpoints, config['moving_time']).intervals:
        a, b = reference.reserve(i.start.heart_rate_bpm), reference.reserve(i.end.heart_rate_bpm)
        if (a is None or b is None or i.distance_source == 'unknown'
                or {'large_recording_gap', 'distance_reset', 'implausible_distance_speed'} & set(i.flags)):
            continue
        start, end = ((p.timestamp_utc-origin).total_seconds() for p in (i.start, i.end))
        if start >= 0 and end > start and i.distance_m >= 0:
            result.append(ResponseInterval(start, end, i.distance_m/i.elapsed_s, a, b))
    return tuple(result)


def _check(rows):
    if any(b.start < a.end-1e-8 for a, b in zip(rows, rows[1:])):
        raise ValueError('Ordered nonoverlapping intervals required')


def sufficient_statistics(rows, taus=TAUS):
    """Per-session integrated least squares; each source session gets one vote.

First matrix is the static speed/HR control. Others fit a first-order response.
Integrate squared residuals of linearly interpolated measured/model endpoints
exactly with two Gauss nodes. No endpoint is repeatedly teacher-forced from HR;
only the start of a contiguous valid segment initializes the response state.
"""
    _check(rows)
    if not rows or any(not isfinite(t) or t <= 0 for t in taus):
        raise ValueError('Nonempty observations and positive response times required')
    tau = np.asarray(taus)
    total = np.zeros((len(taus)+1, 3, 3))
    feature = np.zeros((len(taus), 2))
    offset = np.zeros(len(taus))
    end = None
    duration = sum(r.end-r.start for r in rows)
    for r in rows:
        if end is None or abs(r.start-end) > 1e-8:
            feature[:] = 0
            offset[:] = r.first
        dt = r.end-r.start
        a = np.exp(-dt/tau)
        next_feature = feature*a[:, None]+(1-a[:, None])*np.array([1., r.speed])
        next_offset = offset*a
        for fraction in (.5-1/(2*np.sqrt(3)), .5+1/(2*np.sqrt(3))):
            y = r.first+(r.last-r.first)*fraction
            values = np.empty((len(taus)+1, 3))
            values[0] = (1., r.speed, y)
            values[1:, :2] = feature+(next_feature-feature)*fraction
            values[1:, 2] = y-(offset+(next_offset-offset)*fraction)
            total += np.einsum('ni,nj->nij', values, values)*(dt/(2*duration))
        feature, offset, end = next_feature, next_offset, r.end
    return total


@dataclass(frozen=True)
class ResponseEvidence:
    key: str
    end: datetime
    available_at: datetime
    accounting_key: str
    statistics: np.ndarray
    speed_domain: tuple[float, float]
    eligible: bool = True
    taus: tuple[float, ...] = TAUS

    def __post_init__(self):
        aware(self.end)
        aware(self.available_at)
        if self.end > self.available_at:
            raise ValueError('Evidence cannot arrive before completion')
        if (not self.key or not self.accounting_key
                or not all(isfinite(x) for x in self.speed_domain)
                or not 0 <= self.speed_domain[0] <= self.speed_domain[1]
                or not self.taus or any(not isfinite(t) or t <= 0 for t in self.taus)):
            raise ValueError('Source, units, speed domain and positive response grid required')
        stats = np.array(self.statistics, dtype=float, copy=True)
        if stats.shape != (len(self.taus)+1, 3, 3) or not np.isfinite(stats).all():
            raise ValueError('Finite statistics matching response grid required')
        stats.setflags(write=False)
        object.__setattr__(self, 'statistics', stats)


def fit_response(history, *, as_of, accounting_key, lookback_days=84., taus=TAUS):
    aware(as_of)
    if not isfinite(lookback_days) or lookback_days <= 0:
        raise ValueError('Positive lookback required')
    accepted = sorted((e for e in history if e.eligible and e.accounting_key == accounting_key
        and as_of-timedelta(days=lookback_days) < e.end <= e.available_at <= as_of), key=lambda e: e.key)
    if len({e.key for e in accepted}) != len(accepted):
        raise ValueError('Duplicate source session')
    if any(e.taus != taus for e in accepted):
        raise ValueError('Incompatible response grids')
    result = dict(issued_at=as_of, accounting_key=accounting_key,
        source_keys=tuple(e.key for e in accepted), static=None, dynamic=None,
        status='no_supported_positive_speed_relation', uncertainty_calibrated=False)
    if not accepted:
        return result
    stats = sum((e.statistics for e in accepted))/len(accepted)
    candidates = []
    for index, s in enumerate(stats):
        x, xy = s[:2, :2], s[:2, 2]
        if np.linalg.matrix_rank(x) < 2:
            continue
        coefficients = np.linalg.solve(x, xy)
        if not np.isfinite(coefficients).all() or coefficients[1] <= 0:
            continue
        mse = max(0., float(s[2, 2]-2*coefficients@xy+coefficients@x@coefficients))
        model = dict(intercept=float(coefficients[0]), slope=float(coefficients[1]),
            tau_seconds=None if index == 0 else taus[index-1], training_rmse_reserve=float(np.sqrt(mse)),
            tau_at_search_boundary=index in (1, len(taus)),
            speed_domain=(min(e.speed_domain[0] for e in accepted), max(e.speed_domain[1] for e in accepted)))
        if index == 0:
            result['static'] = model
        else:
            candidates.append((mse, index, model))
    if candidates:
        result['dynamic'] = min(candidates, key=lambda c: (c[0], c[1]))[2]
        result['status'] = 'experimental_fit'
    return result


def predict_response(rows, model):
    """Conditional on recorded speed and segment-initial HR only; no clipping."""
    _check(rows)
    result, end, state, starts, outside = [], None, None, 0, 0.
    for r in rows:
        if end is None or abs(r.start-end) > 1e-8:
            state = r.first
            starts += 1
        target = model['intercept']+model['slope']*r.speed
        tau = model['tau_seconds']
        first = state if tau is not None else target
        last = target+(first-target)*exp(-(r.end-r.start)/tau) if tau is not None else target
        result.append(SignalSpan(r.start, r.end, first, last))
        if not model['speed_domain'][0] <= r.speed <= model['speed_domain'][1]:
            outside += r.end-r.start
        state, end = last, r.end
    return tuple(result), dict(segment_initializations=starts, speed_extrapolation_seconds=outside,
        hr_outside_reference=any(not (0 <= s.first <= 1 and 0 <= s.last <= 1) for s in result))


def predict_speed_program(program, model, *, initial_reserve, step_seconds=.5):
    """Forecast supplied (seconds, m/s) phases without completed-run inputs.

    Share the conditional predictor's state transition. Synthetic interval HR
    values are placeholders: only the very first initializes state; no gaps or
    phase resets. Fine sampling is for the shared piecewise-linear load integral.
    """
    if (not isfinite(initial_reserve) or not 0 <= initial_reserve <= 1
            or not isfinite(step_seconds) or step_seconds <= 0 or not program):
        raise ValueError('Program, initial HR reserve and positive integration step required')
    rows, at = [], 0.
    for seconds, speed in program:
        if not isfinite(seconds) or seconds <= 0 or not isfinite(speed) or speed < 0:
            raise ValueError('Positive phase duration and nonnegative speed required')
        end = at+seconds
        while at < end:
            next_at = min(end, at+step_seconds)
            if next_at <= at:
                raise ValueError('Integration step below time precision')
            rows.append(ResponseInterval(at, next_at, speed, initial_reserve, initial_reserve))
            at = next_at
    return predict_response(rows, model)
