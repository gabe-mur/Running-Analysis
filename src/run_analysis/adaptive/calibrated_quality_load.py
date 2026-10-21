"""Experimental shared response/phase calibration; not installed in planning.

Requested pace drives the existing HR-response model. Prior out-of-sample phase
load ratios correct its load bias, with the existing duration similarity and one
vote per session. No family coefficients, target HR observations or tuned blend.
"""
from dataclasses import dataclass
from datetime import timedelta
from math import fsum, isfinite

from run_analysis.athlete_projection import aware
from run_analysis.hr_response import predict_speed_program
from run_analysis.phase_calibration import clip_trace, mask_trace
from run_analysis.training_signals import integrate_cardio_trace
from .direct_quality_load import POLICY, checked_layout, similarity


@dataclass(frozen=True)
class PhaseLoadRatio:
    kind: str
    seconds: float
    previous_seconds: float | None
    known_seconds: float
    load_ratio: float | None

    def __post_init__(self):
        if (self.kind not in {'warmup','work','recovery','cooldown'}
                or not isfinite(self.seconds) or self.seconds<=0
                or not isfinite(self.known_seconds) or not 0<=self.known_seconds<=self.seconds+1e-6
                or (self.known_seconds==0)!=(self.load_ratio is None)
                or self.load_ratio is not None and (not isfinite(self.load_ratio) or self.load_ratio<0)
                or self.previous_seconds is not None and (not isfinite(self.previous_seconds) or self.previous_seconds<=0)):
            raise ValueError('Explicit dimensionless load ratio and known-time mask required')


@dataclass(frozen=True)
class PhaseCalibrationHistory:
    key: str
    start: object
    end: object
    available_at: object
    accounting_key: str
    phases: tuple[PhaseLoadRatio, ...]

    def __post_init__(self):
        for at in (self.start,self.end,self.available_at):
            aware(at)
        if not self.key or not self.accounting_key or not self.start < self.end <= self.available_at:
            raise ValueError('Completed causal calibration source required')
        if not isinstance(self.phases,tuple) or any(not isinstance(p,PhaseLoadRatio) for p in self.phases):
            raise ValueError('Immutable phase calibration required')


def calibration_phases(layout, predicted, actual):
    """Ratio on identical known-HR time. Missing or zero model load is unknown.

    The supplied prediction must have been frozen BEFORE observing this source.
    Origin/provenance is owned by the chronological caller, not reconstructed
    from the observed HR. This function cannot make a fitted-in-sample curve
    into an honest out-of-sample prediction.
    """
    previous=None; result=[]
    for kind,left,right in checked_layout(layout):
        observed=clip_trace(actual,left,right)
        model=mask_trace(clip_trace(predicted,left,right),observed)
        observed=mask_trace(observed,model)
        seconds=fsum(s.end-s.start for s in observed)
        a=integrate_cardio_trace(observed,right,compute_density=False)['observed_load']
        p=integrate_cardio_trace(model,right,compute_density=False)['observed_load']
        supported=seconds if p is not None and p>0 else 0.
        result.append(PhaseLoadRatio(kind,right-left,previous,supported,a/p if supported else None))
        previous=right-left
    return tuple(result)


def forecast_calibrated(layout, program, response, history, *, as_of, accounting_key, initial_reserve):
    """Forward query accepts requested pace/timing, never completed target HR.

    Identity correction without prior residuals is declared, not a tuned fallback
    to whichever predictor wins. Ratios are not quality-benefit multipliers.
    Raw predicted HR and its validation remain visible; invalid HR is not clipped.
    """
    aware(as_of)
    layout=checked_layout(layout)
    if not accounting_key:
        raise ValueError('Accounting identity required')
    if len(layout)!=len(program) or any(abs(seconds-(right-left))>1e-6
            for (_,left,right),(seconds,speed) in zip(layout,program)):
        raise ValueError('Requested speed program must match the exact phase layout')
    history=tuple(history)
    if len({h.key for h in history})!=len(history):
        raise ValueError('Duplicate calibration session')
    accepted=sorted((h for h in history if h.accounting_key==accounting_key
        and as_of-timedelta(days=POLICY.current_days)<=h.start<h.end<as_of
        and h.available_at<=as_of),key=lambda h:(h.start,h.key))
    result=dict(version='shared_response_phase_calibration_v1',status='no_supported_response',
        issued_at=as_of,accounting_key=accounting_key,source_keys=[h.key for h in accepted],
        phases=[],predicted_load=None,uncertainty_calibrated=False,planner_influence=False)
    if not layout or response is None:
        return result
    raw,diagnostics=predict_speed_program(program,response,initial_reserve=initial_reserve)
    result.update(raw_response_diagnostics=diagnostics)
    if diagnostics['hr_outside_reference']:
        return dict(result,status='predicted_hr_outside_reference')
    previous=None
    for kind,left,right in layout:
        matching={h.key:[p for p in h.phases if p.kind==kind and p.load_ratio is not None] for h in accepted}
        fallback=kind!='work' and not any(matching.values())
        if fallback:
            matching={h.key:[p for p in h.phases if p.kind!='work' and p.load_ratio is not None] for h in accepted}
        sessions=[]
        for h in accepted:
            obs=matching[h.key]
            if not obs:continue
            weights=[similarity(p,right-left,previous,True) for p in obs]
            total=fsum(weights)
            sessions.append(dict(key=h.key,ratio=fsum(w*p.load_ratio for w,p in zip(weights,obs))/total,
                relevance=total/len(obs)))
        weight=fsum(s['relevance'] for s in sessions)
        correction=fsum(s['ratio']*s['relevance'] for s in sessions)/weight if weight else 1.
        raw_load=integrate_cardio_trace(clip_trace(raw,left,right),right,compute_density=False)['observed_load']
        amount=raw_load*correction
        if not isfinite(amount) or amount<0:
            raise ValueError('Finite nonnegative forward load required')
        result['phases'].append(dict(kind=kind,start=left,end=right,rate=amount/(right-left),load=amount,
            raw_load=raw_load,correction=correction,identity_correction=not sessions,
            fallback_pooled_nonwork=fallback,source_sessions=sessions))
        previous=right-left
    return dict(result,status='predicted',predicted_load=fsum(p['load'] for p in result['phases']))
