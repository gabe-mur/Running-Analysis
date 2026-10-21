"""Explicit offline issue/reprice adapter for the reviewed work-band predictor."""
from datetime import datetime
from math import fsum

from run_analysis.continuous_training_dose import ContinuousDoseReference
from run_analysis.response_model import rate
from run_analysis.signal_structure import SignalSpan
from .quality_block_contract import read_quality_contract
from .unsupported_effort_transfer import ResponseAssumption, forecast_extrapolated_work
from .work_exposure_matching import PhaseTrace, WorkExposure, forecast_work_for_ranking, requested_work_bands


VERSION='quality_catalogue_runtime_v1'


def read_profiles(value):
    return {key:tuple(PhaseTrace(p['start'],p['end'],tuple(SignalSpan(**s) for s in p['spans']))
        for p in phases) for key,phases in value.items()}


def quality_target(intent,history,*,layout=None,prepared_profiles=None):
    """Frozen cues and causal raw profiles; same path at search and publication."""
    evidence=intent['quality_evidence']
    if evidence['version']!=VERSION:
        raise ValueError('Unknown explicit quality runtime contract')
    p=read_quality_contract(intent['prescription_contract'])
    issued=datetime.fromisoformat(intent['issued_at'])
    if datetime.fromisoformat(evidence['available_at'])>issued:
        raise ValueError('Future quality evidence cannot enter issuance')
    from .execution_forecast_contract import layout as clock
    layout=layout if layout is not None else clock((s['kind'],s['seconds']['target']) for s in intent['phases'])
    requested=requested_work_bands(p.phases,evidence['config'])
    if len(layout)!=len(requested) or tuple(k for k,_,_ in layout)!=tuple(s.kind for s in p.phases):
        return dict(status='unresolved_work_alignment',predicted_load=None,source_keys=[],phases=[])
    profiles=read_profiles(evidence['profiles']) if prepared_profiles is None else prepared_profiles
    kwargs=dict(as_of=issued,accounting_key=intent['accounting_key'])
    response=evidence.get('response');anchors=evidence.get('anchors')
    if response is None or not anchors:
        return forecast_work_for_ranking(layout,history,profiles,requested,**kwargs)
    response=ResponseAssumption(**dict(response,available_at=datetime.fromisoformat(response['available_at']),
        source_keys=tuple(response['source_keys']),sensitivity_taus=tuple(response['sensitivity_taus'])))
    anchors=tuple(WorkExposure(tuple(a['start_band']),tuple(a['finish_band']),a['basis']) for a in anchors)
    reference=ContinuousDoseReference(**{k:v for k,v in evidence['reference'].items() if k!='quality_mapping'})
    return forecast_extrapolated_work(layout,history,profiles,requested,
        anchors=anchors,response=response,reference=reference,**kwargs)


def quality_handoff(phases,target,history,scope,evidence):
    from .forecast_handoff import planning_handoff
    result=planning_handoff(phases,target,history,scope)
    result['predictor_version']=VERSION
    result['requested_effort_conditioned']=True
    result['prediction_basis']=target.get('prediction_basis','observed_work_band_matching')
    result['limits'][0]='Work HR-band matching uses measured whole-phase costs; commanded effort is not verified by observed overlap'
    if target.get('effort_extrapolation') and target['predicted_load'] is not None:
        # No pretend convex observed-rate bounds around extrapolated work. The
        # analytic transfer already rejects HR outside [0,1]. Use explicit full
        # accounting bounds over the declared duration box, not a confidence CI.
        reference=ContinuousDoseReference(**{k:v for k,v in evidence['reference'].items() if k!='quality_mapping'})
        high=fsum(p.seconds.high for p in phases)/60*rate(1.,reference.policy)
        result.update(usable=True,status='provisional_extrapolation',reason=None,
            cardio=dict(low=0.,high=high,estimate=target['predicted_load'],
                basis='full HRR accounting bounds over duration box; not observed support or confidence'),
            running_minutes=dict(low=fsum(p.seconds.low for p in phases)/60,
                high=fsum(p.seconds.high for p in phases)/60,estimate=fsum(p.seconds.target for p in phases)/60,
                basis='declared duration scenarios'),scope_basis='explicit_upward_effort_extrapolation',
            effort_extrapolation=target['effort_extrapolation'])
    return result
