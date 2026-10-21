from dataclasses import replace
from datetime import timedelta

import pytest

from run_analysis.adaptive.direct_effort_load import (
    ExposureHistory,effort_trace,forecast_effort_contrast,input_reference,phase_exposures)
from run_analysis.adaptive.direct_quality_load import PhaseHistory,phase_observations,score_direct
from run_analysis.signal_structure import SignalSpan
from test_quality_dose_policy import AT

LAYOUT=(('warmup',0.,300.),('work',300.,360.),('recovery',360.,480.),('cooldown',480.,600.))
INPUTS=((.6,.6),(.95,.95),(.55,.55),(.5,.5))


def histories():
    spans,valid=effort_trace(LAYOUT,INPUTS,tau_seconds=45,initial_reserve=.45)
    assert valid
    start,end=AT-timedelta(days=2),AT-timedelta(days=2)+timedelta(seconds=600)
    return (PhaseHistory('source',start,end,end,'units',phase_observations(LAYOUT,spans)),),(
        ExposureHistory('source',start,end,end,'units',phase_exposures(LAYOUT,spans)),)


def forecast(inputs,phase_history=(),exposure_history=()):
    return forecast_effort_contrast(LAYOUT,inputs,phase_history,exposure_history,
        as_of=AT,accounting_key='units',tau_seconds=45,initial_reserve=.45)


def test_supported_moments_are_invariant_to_trace_partition_and_gaps_not_filled():
    whole=(SignalSpan(0,600,.5,.8),)
    split=(SignalSpan(0,300,.5,.65),SignalSpan(300,600,.65,.8))
    layout=(('work',0,600),)
    assert phase_exposures(layout,whole)==phase_exposures(layout,split)
    gap=(SignalSpan(0,200,.5,.6),SignalSpan(400,600,.7,.8))
    p=phase_exposures(layout,gap)[0]
    assert p.known_seconds==400 and p.seconds==600
    assert p.endpoint_change_sum==pytest.approx(.2)
    assert p.mean_hrr==pytest.approx(.65)


def test_matching_reference_inputs_preserve_baseline_not_a_second_load():
    ph,ex=histories()
    refs=input_reference(LAYOUT,ex,as_of=AT,accounting_key='units',tau_seconds=45)
    f=forecast(tuple((p['demand'],p['demand']) for p in refs),ph,ex)
    assert f['status']=='predicted'
    assert all(p['adjustment']==pytest.approx(1) for p in f['phases'])
    assert f['predicted_load']==pytest.approx(f['baseline']['predicted_load'])


def test_numeric_effort_changes_work_and_recovery_without_family_constants():
    ph,ex=histories()
    lower=forecast(INPUTS,ph,ex)
    harder=list(INPUTS);harder[1]=(1.,1.)
    higher=forecast(harder,ph,ex)
    assert higher['predicted_load'] > lower['predicted_load']
    assert higher['phases'][2]['load'] > lower['phases'][2]['load']
    assert higher['phases'][0]['load']==pytest.approx(lower['phases'][0]['load'])


def test_no_quality_history_keeps_explicit_forward_prior_not_zero_or_family_ban():
    f=forecast(INPUTS)
    assert f['status']=='predicted' and f['predicted_load'] > 0
    assert f['mode']=='declared_forward_prior_without_phase_history'
    assert not f['command_mapping_calibrated']


def test_single_current_tau_recomputes_history_and_future_sources_cannot_enter():
    ph,ex=histories()
    future=replace(ex[0],key='future',start=AT,end=AT+timedelta(seconds=600),available_at=AT+timedelta(seconds=600))
    a=input_reference(LAYOUT,ex,as_of=AT,accounting_key='units',tau_seconds=45)
    b=input_reference(LAYOUT,(*ex,future),as_of=AT,accounting_key='units',tau_seconds=45)
    assert a==b
    c=input_reference(LAYOUT,ex,as_of=AT,accounting_key='units',tau_seconds=90)
    assert a[1]['demand']!=c[1]['demand']


def test_repetition_carryover_and_very_short_work_are_not_reset_or_peak_scored():
    layout=(('work',0,15),('recovery',15,45),('work',45,60))
    spans,valid=effort_trace(layout,((.95,.95),(.5,.5),(.95,.95)),tau_seconds=45,initial_reserve=.5)
    assert valid and spans[-1].last > next(s.last for s in spans if s.end==15)
    assert max(s.last for s in spans) < .95


def test_invalid_predicted_hr_is_a_failure_not_clipped():
    f=forecast(((.6,.6),(5.,5.),(.5,.5),(.5,.5)))
    assert f['status']=='requested_hr_outside_reference' and f['predicted_load'] is None


def test_unknown_support_never_becomes_zero_cost():
    ph,ex=histories(); ex=()
    f=forecast(INPUTS,ph,ex)
    assert f['status']=='unresolved_reference_input' and f['predicted_load'] is None
