"""Retrospective HR response alignment for comparable-performance experiments.

First-order effort response is approximate, not a measurement of oxygen uptake.
No target run selects its own response time. Original HR/pace stays unchanged.
"""
from bisect import bisect_left
from collections import defaultdict
from datetime import datetime
import math

import numpy as np

from .physiology import grade_energy_ratio


VERSION = 'effort-response-alignment-v1'
# A fixed search grid including the null; engineering candidates, not universal
# physiological time constants. Selection is held out by physical run.
TAUS = (0., 15., 30., 45., 60., 90., 120., 180., 240., 300.)


def response_step(state, effort, seconds, tau):
    """Exact mean and endpoint under piecewise constant effort."""
    if seconds <= 0 or tau < 0 or not all(math.isfinite(v) for v in (state, effort, seconds, tau)):
        raise ValueError('Finite effort/state, positive interval and nonnegative response time required')
    if tau == 0:
        return effort, effort
    decay = math.exp(-seconds/tau)
    average = effort+(state-effort)*tau/seconds*(-math.expm1(-seconds/tau))
    return average, effort+(state-effort)*decay


def context_for_windows(intervals, rows, config):
    """Extract all response candidates before HR-domain/score selection.

    HR above the fitness window ceiling can inform response fitting. A quick
    effort is not allowed to hide its delayed HR cost just by crossing Z3/Z4.
    Long missing moving intervals are not assumed to contain constant running.
    """
    if not intervals or not rows:
        return
    states=None;areas=np.zeros(len(TAUS));cumulative=[areas.copy()]
    times=[intervals[0].start.timestamp_utc.timestamp()]
    training=defaultdict(lambda: dict(seconds=0.,hr=0.,minute=0.,effort=np.zeros(len(TAUS))))
    moving=0.;unknown=[]
    maximum_grade=float(config['elevation']['maximum_plausible_grade_percent'])
    for interval in intervals:
        dt=interval.elapsed_s
        if dt <= 0:
            continue
        speed=interval.distance_m/dt
        grade=1.
        if interval.elevation_delta_m is not None and interval.distance_m >= 1.:
            slope=100*interval.elevation_delta_m/interval.distance_m
            if abs(slope)<=maximum_grade:
                grade=grade_energy_ratio(slope,maximum_grade)
        effort=max(0.,speed*grade)
        if states is None:
            states=np.full(len(TAUS),effort)
        means=[]
        for index,tau in enumerate(TAUS):
            mean,states[index]=response_step(states[index],effort,dt,tau)
            means.append(mean)
        means=np.array(means)
        areas+=means*dt
        cumulative.append(areas.copy())
        times.append(interval.end.timestamp_utc.timestamp())
        start=interval.start.timestamp_utc.timestamp()
        if dt>30 and interval.moving_time_s>0:
            # Unknown within-gap timing is not identified by distance endpoints.
            unknown.append((start,interval.end.timestamp_utc.timestamp()))
        values=[v for v in (interval.start.heart_rate_bpm,interval.end.heart_rate_bpm) if v is not None]
        minute=(moving+interval.moving_time_s/2)/60
        if (values and dt<=15 and interval.moving_time_s>0 and speed>0
            and 5<=minute<=60 and 128<=sum(values)/len(values)<=float(config['max_hr'])):
            # Fit at the SAME two-minute resolution used by the fitness
            # observations. Otherwise a longer response can win merely by
            # smoothing noisy instantaneous GPS speeds, not by aligning HR.
            block=training[int((start-times[0])/120)]
            block['seconds']+=dt;block['hr']+=sum(values)/len(values)*dt
            block['minute']+=minute*dt;block['effort']+=means*dt
        moving+=interval.moving_time_s
    times=np.array(times);cumulative=np.array(cumulative)
    samples=[]
    for block in training.values():
        sec=block['seconds']
        if sec>=90 and np.all(block['effort']>0):
            samples.append(dict(seconds=sec,hr=block['hr']/sec,minute=block['minute']/sec,
                                effort=(block['effort']/sec).tolist()))
    def integral(at):
        i=bisect_left(times,at)
        if i<len(times) and abs(times[i]-at)<1e-5:
            return cumulative[i]
        if i==0 or i==len(times):
            return None
        fraction=(at-times[i-1])/(times[i]-times[i-1])
        return cumulative[i-1]+fraction*(cumulative[i]-cumulative[i-1])
    for row in rows:
        low=datetime.fromisoformat(row['window_start_time_utc']).timestamp()
        high=datetime.fromisoformat(row['window_end_time_utc']).timestamp()
        a,b=integral(low),integral(high)
        if a is None or b is None:
            continue
        values=(b-a)/max(1e-9,high-low)
        valid=values[0]>0
        # An unknown gap and its response memory cannot become precise evidence.
        ratios=[float(v/values[0]) if valid and not any(low<hi+3*tau and high>lo for lo,hi in unknown) else None
                for v,tau in zip(values,TAUS)]
        row['hr_response_context']=dict(tau_seconds=list(TAUS),speed_ratios=ratios)
    # Shared in memory only; callers remove this before serializing windows.
    for row in rows:
        row['_hr_response_training']=samples


def training_losses(samples, config, calibration):
    """Equal-run centered loss: no run-level fitness target or score ranking."""
    if len(samples)<4:
        return None
    beta=calibration['parameters']
    # Use the established HR elasticity/drift, not a new fitted fitness slope.
    y=np.log([s['effort'] for s in samples])
    h=beta[0]*np.log([(s['hr']-config['resting_hr'])/(config['target_hr']-config['resting_hr']) for s in samples])
    drift=beta[1]*np.array([(s['minute']-config['reference_conditions']['within_run_minutes'])/10 for s in samples])
    residual=y-h[:,None]-drift[:,None]
    weights=np.array([s['seconds'] for s in samples]);weights/=weights.sum()
    residual-=np.average(residual,axis=0,weights=weights)
    # Each run supplies one loss per candidate, not hundreds of independent
    # votes. Squared residuals retain acceleration and recovery costs alike.
    return np.average(residual**2,axis=0,weights=weights).tolist()


def held_out_tau(losses, identity):
    earlier=[v for key,v in losses.items() if key!=identity and v is not None]
    if len(earlier)<3:
        return 0
    return int(np.argmin(np.mean(earlier,axis=0)))


def corrected_windows(rows, tau_index):
    result=[]
    for row in rows:
        context=row.get('hr_response_context')
        ratio=context['speed_ratios'][tau_index] if context else None
        if ratio is None or ratio<=0:
            continue
        copy={k:v for k,v in row.items() if k!='_hr_response_training'}
        copy['moving_pace_min_mile']=row['moving_pace_min_mile']/ratio
        result.append(copy)
    return result
