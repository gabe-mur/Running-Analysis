"""Unconnected Q5 executed-demand diagnostic under the existing response model.

The integrated first-order identity avoids differentiating noisy HR samples.
Inferred demand is NOT saved intention, lactate threshold or a validated physical
measurement. All queries using completed HR are diagnostic, not prospective.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import exp, fsum, isfinite, log

import numpy as np

from run_analysis.athlete_projection import aware
from run_analysis.phase_calibration import clip_trace
from run_analysis.performance_reference import geometric_mean
from .quality_history_policy import QualityHistoryPolicy

VERSION = 'lag_demand_reference_v1'


def executed_demand(spans, left, right, *, tau_seconds, bpm_span):
    """Mean input u = mean HRR + tau*(last HRR-first HRR)/duration.

    For dHRR/dt=(u-HRR)/tau this identity also holds for varying input u.
    No HR is invented across missing intervals or clipped into a plausible band.
    Model input demand is NOT measured HR: a signed or above-one inferred input
    is retained with a domain warning, not erased as if actual HR were invalid.
    The one-bpm sensitivity bound is a perturbation diagnostic, NOT assumed
    device error, confidence bounds or a physiological tolerance.
    """
    if (not all(isfinite(x) for x in (left,right,tau_seconds,bpm_span))
            or not 0 <= left < right or tau_seconds <= 0 or bpm_span <= 0):
        raise ValueError('Ordered interval, positive causal response time and HR scale required')
    spans = tuple(spans)
    if any(b.start < a.end for a,b in zip(spans,spans[1:])):
        raise ValueError('Ordered nonoverlapping measured HR required')
    if any(not 0 <= min(s.first,s.last) <= max(s.first,s.last) <= 1 for s in spans):
        raise ValueError('Supported observed normalized HR required')
    known = tuple(clip_trace(spans,left,right))
    duration = right-left
    seconds = fsum(s.end-s.start for s in known)
    result = dict(version=VERSION,status='incomplete_hr',mean_hrr=None,demand=None,
        known_seconds=seconds,seconds=duration,tau_seconds=tau_seconds,
        endpoint_term=None,first_hrr=None,last_hrr=None,
        outside_hr_reference_range=None,
        demand_change_bound_per_one_bpm=(1+2*tau_seconds/duration)/bpm_span,
        uncertainty_calibrated=False,
        assumption='Shared first-order response with constant tau; integrated identity, not intention')
    if (not known or abs(known[0].start-left)>1e-8 or abs(known[-1].end-right)>1e-8
            or abs(seconds-duration)>1e-8
            or any(abs(b.start-a.end)>1e-8 for a,b in zip(known,known[1:]))):
        return result
    mean = fsum((s.end-s.start)*(s.first+s.last)/2 for s in known)/duration
    term = tau_seconds*(known[-1].last-known[0].first)/duration
    demand = mean+term
    return dict(result,status='inferred_model_demand',outside_hr_reference_range=not 0 <= demand <= 1,
        mean_hrr=mean,demand=demand,endpoint_term=term,
        first_hrr=known[0].first,last_hrr=known[-1].last)


@dataclass(frozen=True)
class DemandSample:
    key: str
    source_key: str
    start: datetime
    end: datetime
    available_at: datetime
    accounting_key: str
    seconds: float
    speed_mps: float
    mean_hrr: float
    demand: float
    kind: str  # provenance only; never a predictor feature or eligibility gate.

    def __post_init__(self):
        for at in (self.start,self.end,self.available_at):
            aware(at)
        if (not all((self.key,self.source_key,self.accounting_key,self.kind))
                or not self.start < self.end <= self.available_at
                or not all(isfinite(x) for x in (self.seconds,self.speed_mps,self.mean_hrr,self.demand))
                or self.seconds <= 0 or self.speed_mps <= 0
                or not 0 <= self.mean_hrr <= 1):
            raise ValueError('Dated supported measured demand and pace required')


def _accepted(history, as_of, accounting_key, policy):
    aware(as_of)
    history = tuple(history)
    if not accounting_key or len({r.key for r in history}) != len(history):
        raise ValueError('Accounting identity and unique sample keys required')
    return tuple(sorted((r for r in history if r.accounting_key == accounting_key
        and r.end < as_of and r.available_at <= as_of
        and r.start >= as_of-timedelta(days=policy.current_days)),key=lambda r:r.key))


def fit_pace(history, *, as_of, accounting_key, channel, policy=QualityHistoryPolicy()):
    """Prior-only log-speed regression, equal total ownership per source run.

    Controls: duration (two coefficients), mean HR + duration (three), and
    inferred demand + duration (three). Coefficients are learned, not prescribed.
    Positive intensity slope is the existing direction assumption; no forced
    duration slope, family effects, regularization/window/weight sweep or fallback.
    A per-athlete intercept supplies the athlete's speed scale without VO2max
    input or pooling different athletes. Extrapolation is disclosed, not banned.
    """
    if channel not in {'duration','mean_hrr','demand'}:
        raise ValueError('Explicit fixed comparison channel required')
    rows = _accepted(history,as_of,accounting_key,policy)
    keys = tuple(sorted({r.source_key for r in rows}))
    result = dict(version=VERSION,issued_at=as_of,accounting_key=accounting_key,channel=channel,
        status='insufficient_identifiable_evidence',source_keys=keys,sample_keys=tuple(r.key for r in rows),
        coefficients=None,domain=None,training_log_rmse=None,uncertainty_calibrated=False,
        feature_order=('intercept','log_seconds') if channel=='duration' else
            ('intercept',channel,'log_seconds'))
    if not rows:
        return result
    counts = {key:sum(r.source_key==key for r in rows) for key in keys}
    x = np.array([[1.,log(r.seconds)] if channel=='duration' else
        [1.,getattr(r,channel),log(r.seconds)] for r in rows])
    y = np.log([r.speed_mps for r in rows])
    w = np.sqrt([1/counts[r.source_key] for r in rows])
    matrix = x*w[:,None]
    coefficients,_,rank,_ = np.linalg.lstsq(matrix,y*w,rcond=None)
    if rank < x.shape[1] or not np.isfinite(coefficients).all():
        return result
    result['domain'] = dict(seconds=(min(r.seconds for r in rows),max(r.seconds for r in rows)),
        intensity=None if channel=='duration' else
            (min(getattr(r,channel) for r in rows),max(getattr(r,channel) for r in rows)))
    if channel!='duration' and coefficients[1] <= 0:
        return dict(result,status='nonpositive_intensity_relation')
    residuals = x@coefficients-y
    return dict(result,status='fitted_diagnostic',coefficients=tuple(float(c) for c in coefficients),
        training_log_rmse=float(np.sqrt(np.sum((residuals*w)**2)/np.sum(w**2))))


def predict_pace(model, *, issued_at, seconds, intensity=None):
    aware(issued_at)
    if model['issued_at'] > issued_at:
        raise ValueError('Future fitted pace reference')
    if (not isfinite(seconds) or seconds <= 0 or model['channel']!='duration'
            and (intensity is None or not isfinite(intensity)
                 or model['channel']=='mean_hrr' and not 0 <= intensity <= 1)):
        raise ValueError('Positive duration and supported explicit intensity required')
    result = dict(status=model['status'],speed_mps=None,source_keys=model['source_keys'],
        extrapolation=None,uncertainty_calibrated=False)
    if model['coefficients'] is None:
        return result
    features = (1.,log(seconds)) if model['channel']=='duration' else (1.,intensity,log(seconds))
    log_speed = fsum(a*b for a,b in zip(features,model['coefficients']))
    try:
        speed = exp(log_speed)
    except OverflowError:
        return dict(result,status='nonfinite_speed_prediction')
    if not isfinite(speed) or speed <= 0:
        return dict(result,status='nonfinite_speed_prediction')
    domain = model['domain']
    outside = not domain['seconds'][0] <= seconds <= domain['seconds'][1]
    if model['channel']!='duration':
        outside |= not domain['intensity'][0] <= intensity <= domain['intensity'][1]
    return dict(result,status='predicted_diagnostic',speed_mps=speed,extrapolation=outside,
        demand_outside_hr_reference_range=model['channel']=='demand' and not 0 <= intensity <= 1)


def duration_reference(history, *, as_of, accounting_key, seconds, policy=QualityHistoryPolicy()):
    """Existing log-duration kernel on the identical supported sample cohort."""
    if not isfinite(seconds) or seconds <= 0:
        raise ValueError('Positive queried duration required')
    rows = _accepted(history,as_of,accounting_key,policy)
    values = []
    for key in sorted({r.source_key for r in rows}):
        group = [r for r in rows if r.source_key==key]
        weights = [1/(1+log(r.seconds/seconds)**2) for r in group]
        values.append((key,geometric_mean([r.speed_mps for r in group],weights),fsum(weights)/len(weights)))
    return dict(status='predicted_diagnostic' if values else 'no_recent_reference',
        speed_mps=geometric_mean([v for _,v,_ in values],[w for _,_,w in values]) if values else None,
        source_keys=tuple(k for k,_,_ in values),uncertainty_calibrated=False)


def fit_exposure_pace(history, *, as_of, accounting_key, channel,
                      policy=QualityHistoryPolicy()):
    """Corrected observation-only relation, NOT a capacity/duration model.

    Sample.seconds is known measurement support, never a performance-duration
    feature. Each source owns one vote, allocated by observed time. Keeping an
    identifiable intercept/intensity relation does not identify duration transfer
    or remove between-session confounding. Old fit_pace stays a frozen comparator.
    """
    if channel not in {'mean_hrr', 'demand'}:
        raise ValueError('Explicit observed exposure channel required')
    rows = _accepted(history, as_of, accounting_key, policy)
    keys = tuple(sorted({r.source_key for r in rows}))
    result = dict(version='exposure_pace_reference_v1', issued_at=as_of,
        accounting_key=accounting_key, channel=channel, status='insufficient_identifiable_evidence',
        source_keys=keys, sample_keys=tuple(r.key for r in rows), coefficients=None,
        feature_order=('intercept', channel), duration_effect_identified=False,
        measurement_duration_is_capacity=False, training_log_rmse=None,
        uncertainty_calibrated=False, domain=None)
    if not rows:
        return result
    seconds = {k:fsum(r.seconds for r in rows if r.source_key == k) for k in keys}
    weights = np.array([r.seconds/seconds[r.source_key] for r in rows])
    x = np.array([[1., getattr(r, channel)] for r in rows])
    y = np.log([r.speed_mps for r in rows])
    coefficients, _, rank, _ = np.linalg.lstsq(x*np.sqrt(weights)[:, None], y*np.sqrt(weights), rcond=None)
    if rank < 2 or not np.isfinite(coefficients).all():
        return result
    result['domain'] = dict(intensity=(min(getattr(r, channel) for r in rows),
        max(getattr(r, channel) for r in rows)))
    if coefficients[1] <= 0:
        return dict(result, status='nonpositive_intensity_relation')
    residuals = x@coefficients-y
    return dict(result, status='fitted_diagnostic', coefficients=tuple(float(c) for c in coefficients),
        training_log_rmse=float(np.sqrt(np.sum(weights*residuals**2)/np.sum(weights))))


def predict_exposure_pace(model, *, issued_at, intensity):
    """Descriptive pace at inferred/observed exposure; no capacity-time query."""
    aware(issued_at)
    if model['issued_at'] > issued_at or not isfinite(intensity):
        raise ValueError('Causal fit and finite exposure required')
    if model['channel'] == 'mean_hrr' and not 0 <= intensity <= 1:
        raise ValueError('Measured HR must fit its reference')
    result = dict(status=model['status'], speed_mps=None, source_keys=model['source_keys'],
        duration_effect_identified=False, uncertainty_calibrated=False, extrapolation=None)
    if model['coefficients'] is None:
        return result
    try:
        speed = exp(model['coefficients'][0]+model['coefficients'][1]*intensity)
    except OverflowError:
        return dict(result, status='nonfinite_speed_prediction')
    if not isfinite(speed) or speed <= 0:
        return dict(result, status='nonfinite_speed_prediction')
    return dict(result, status='predicted_diagnostic', speed_mps=speed,
        extrapolation=not model['domain']['intensity'][0] <= intensity <= model['domain']['intensity'][1])
