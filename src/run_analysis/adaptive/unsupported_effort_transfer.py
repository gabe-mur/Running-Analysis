"""Offline upward effort extrapolation around an unchanged supported forecast.

Only absent band overlap permits transfer. Empirical costs are multiplied by
requested/anchor response-load ratios under identical timing, initial state and
tau. This is an explicit unvalidated transfer assumption, not identified intent.
Supported prescriptions return exactly unchanged. No production callers.
"""
from copy import deepcopy
from dataclasses import dataclass
from math import fsum, isfinite

from run_analysis.athlete_projection import aware
from run_analysis.continuous_training_dose import ContinuousDoseReference
from run_analysis.phase_calibration import clip_trace
from run_analysis.response_model import rate
from run_analysis.training_signals import integrate_cardio_trace
from .direct_effort_load import effort_trace
from .direct_quality_load import checked_layout
from .work_exposure_matching import WorkExposure, forecast_work_exposure, forecast_work_for_ranking


VERSION = 'unsupported_upward_effort_contrast_v1'


@dataclass(frozen=True)
class ResponseAssumption:
    tau_seconds: float
    available_at: object
    source_keys: tuple[str, ...]
    basis: str
    sensitivity_taus: tuple[float, ...] = ()

    def __post_init__(self):
        aware(self.available_at)
        if (not self.basis.strip() or not isinstance(self.source_keys,tuple) or not self.source_keys
                or not isinstance(self.sensitivity_taus,tuple)
                or any(not isinstance(k,str) or not k.strip() for k in self.source_keys)
                or len(set(self.source_keys)) != len(self.source_keys)
                or any(isinstance(t,bool) or not isfinite(t) or t <= 0
                    for t in (self.tau_seconds,*self.sensitivity_taus))):
            raise ValueError('Positive explicit response times, availability and source provenance required')


def equivalent_reserve(per_minute, policy):
    """Inverse accounting rate, not an inferred measured or equilibrium HR."""
    if not isfinite(per_minute) or not 0 <= per_minute <= rate(1.,policy):
        raise ValueError('Empirical rate outside normalized accounting bounds')
    left,right = 0.,1.
    for _ in range(60):
        mid = (left+right)/2
        if rate(mid,policy) < per_minute:
            left = mid
        else:
            right = mid
    return (left+right)/2


def midpoint(command):
    return tuple(fsum(b)/2 for b in (command.start_band,command.finish_band))


def lower_anchor(anchor, requested):
    return all(a <= b for ab,rb in zip((anchor.start_band,anchor.finish_band),
        (requested.start_band,requested.finish_band)) for a,b in zip(ab,rb)) and midpoint(anchor)!=midpoint(requested)


def contrast(anchor, reference_inputs, requested_inputs, *, tau, initial, policy, step_seconds=.5):
    layout = tuple((p['kind'],p['start'],p['end']) for p in anchor['phases'])
    reference,valid_a = effort_trace(layout,reference_inputs,tau_seconds=tau,initial_reserve=initial,step_seconds=step_seconds)
    requested,valid_b = effort_trace(layout,requested_inputs,tau_seconds=tau,initial_reserve=initial,step_seconds=step_seconds)
    if not valid_a or not valid_b:
        return dict(status='response_outside_hr_reference',phases=[],predicted_load=None)
    result = []
    for p in anchor['phases']:
        a,b = p['start'],p['end']
        old = integrate_cardio_trace(clip_trace(reference,a,b),b,policy,compute_density=False)['observed_load']
        new = integrate_cardio_trace(clip_trace(requested,a,b),b,policy,compute_density=False)['observed_load']
        if old is None or old <= 0:
            return dict(status='unsupported_response_denominator',phases=[],predicted_load=None)
        factor = new/old
        amount = p['load']*factor
        if factor < 1.-1e-10 or amount/(b-a)*60 > rate(1.,policy)+1e-10:
            return dict(status='contrast_outside_accounting_bounds',phases=[],predicted_load=None)
        result.append(dict(kind=p['kind'],start=a,end=b,load=amount,rate=amount/(b-a),
            adjustment=factor,reference_model_load=old,requested_model_load=new))
    return dict(status='predicted',phases=result,predicted_load=fsum(p['load'] for p in result),
        final_reference_hrr=reference[-1].last,final_requested_hrr=requested[-1].last)


def forecast_extrapolated_work(layout, history, profiles, requested, *, anchors, response,
                               reference, as_of, accounting_key):
    """Use supported lower work bands without changing any supported query.

    Recovery/cooldown cost can increase after extrapolated work: same HR state
    carries through the complete phase sequence. Unknown cues, missing traces,
    absent lower anchors and cold start do not invent a forecast. No exact
    workout-format gate. Scenario ranges are sensitivities, NOT confidence bounds.
    """
    layout = checked_layout(layout); history = tuple(history); requested = tuple(requested)
    original = forecast_work_for_ranking(layout,history,profiles,requested,
        as_of=as_of,accounting_key=accounting_key)
    if original['work_exposure_matching']['requested_intensity_rankable']:
        return original
    if not isinstance(reference,ContinuousDoseReference) or reference.accounting_key!=accounting_key:
        raise ValueError('The same declared HR-load accounting reference is required')
    if not isinstance(response,ResponseAssumption) or response.available_at>as_of:
        raise ValueError('Explicit causal response assumption required')
    anchors = tuple(anchors)
    if not anchors or any(not isinstance(a,WorkExposure) for a in anchors):
        raise ValueError('Explicit candidate lower work bands required')
    # Keep the original unknown forecast until every missing work query has a
    # supported lower anchor. Never promote a timing-only proxy along this path.
    result = deepcopy(original)
    result['effort_extrapolation'] = dict(version=VERSION,status='no_supported_lower_anchor',
        response=response,reference=reference,planner_influence=False,uncertainty_calibrated=False,
        command_as_equilibrium_input_calibrated=False,transfer_calibrated=False)
    changed = []
    anchored = list(requested)
    for i,p in enumerate(original['phases']):
        if p['kind']!='work' or p['work_exposure']['supported']:
            continue
        if p['work_exposure']['fallback_reason']!='no_observed_band_overlap' or requested[i] is None:
            result['effort_extrapolation']['status'] = p['work_exposure']['fallback_reason']
            return result
        ordered = sorted((a for a in anchors if lower_anchor(a,requested[i])),
            key=lambda a:(-fsum(midpoint(a)),a.start_band,a.finish_band,a.basis))
        for command in ordered:
            trial = list(anchored);trial[i] = command
            f = forecast_work_exposure(layout,history,profiles,trial,as_of=as_of,accounting_key=accounting_key)
            if f['phases'][i]['work_exposure']['supported']:
                anchored[i] = command;changed.append(i)
                break
        else:
            return result
    if not changed:
        return result
    anchor = forecast_work_for_ranking(layout,history,profiles,anchored,as_of=as_of,accounting_key=accounting_key)
    if anchor['status']!='predicted':
        return result
    inputs = []
    for p,command in zip(anchor['phases'],anchored):
        inputs.append(midpoint(command) if p['kind']=='work' else
            (equivalent_reserve(p['rate']*60,reference.policy),)*2)
    # Cost-equivalent opening HR is a common declared starting-state proxy,
    # never the held-out run's first HR sample or a fitted Z5 response.
    initial = equivalent_reserve(anchor['phases'][0]['rate']*60,reference.policy)
    def transferred(fraction,tau):
        desired = list(inputs)
        for i in changed:
            q = requested[i]
            desired[i] = tuple(b[0]+fraction*(b[1]-b[0]) for b in (q.start_band,q.finish_band))
            if any(b<a for a,b in zip(inputs[i],desired[i])):
                return dict(status='sensitivity_below_anchor',phases=[],predicted_load=None)
        return contrast(anchor,inputs,desired,tau=tau,initial=initial,policy=reference.policy)
    nominal = transferred(.5,response.tau_seconds)
    result['effort_extrapolation'].update(anchor=anchor,anchor_commands=tuple(anchored),
        changed_work_indices=changed,initial_hrr_proxy=initial,nominal=nominal,
        source_keys=anchor['source_keys'],status=nominal['status'])
    if nominal['status']!='predicted':
        return result
    scenarios = [dict(target_fraction=f,tau_seconds=t,forecast=transferred(f,t))
        for t in sorted(set((response.tau_seconds,*response.sensitivity_taus))) for f in (0.,1.)]
    if any(s['forecast']['status']!='predicted' for s in scenarios):
        result['effort_extrapolation'].update(status='incomplete_sensitivity_support',scenarios=scenarios)
        return result
    costs = [nominal['predicted_load'],*(s['forecast']['predicted_load'] for s in scenarios)]
    result['effort_extrapolation'].update(status='extrapolated',scenarios=scenarios,
        load_sensitivity=dict(low=min(costs),nominal=nominal['predicted_load'],high=max(costs),
            basis='Declared target-band and causal response-time scenarios, not confidence bounds'))
    result.update(status='predicted',prediction_basis='extrapolated_effort_not_observed_support',
        predicted_load=nominal['predicted_load'])
    for p,a,n in zip(result['phases'],anchor['phases'],nominal['phases']):
        if n['adjustment']==1.:
            p.update(rate=a['rate'],load=a['load'])
        else:
            p.update(rate=n['rate'],load=n['load'])
        p['effort_transfer'] = dict(anchor_rate=a['rate'],adjustment=n['adjustment'],
            modeled_carryover=p['kind']!='work' and n['adjustment']>1.)
    result['work_exposure_matching']['requested_intensity_rankable'] = True
    result['work_exposure_matching']['rankable_basis'] = 'declared_extrapolation_not_band_support'
    return result
