from dataclasses import replace
from datetime import timedelta

import pytest

from run_analysis.adaptive.unsupported_effort_transfer import (
    ResponseAssumption, contrast, equivalent_reserve, forecast_extrapolated_work)
from run_analysis.adaptive.work_exposure_matching import WorkExposure, forecast_work_for_ranking
from run_analysis.continuous_training_dose import ContinuousDoseReference
from run_analysis.response_model import rate
from test_work_exposure_matching import AT, LAYOUT, query, record


REF = ContinuousDoseReference('test fixed accounting')
ANCHOR = WorkExposure((.75,.85),(.75,.85),'explicit prior Z4-like band')
HIGH = WorkExposure((.9,1.),(.9,1.),'explicit harder band')
RESPONSE = ResponseAssumption(64.,AT-timedelta(days=1),('past',),'prior-only test fit',(75.,))


def inputs():
    source,traces = record('past',.8)
    source = replace(source,accounting_key=REF.accounting_key)
    return (source,),{'past':traces}


def forecast(layout=LAYOUT, command=HIGH, response=RESPONSE, anchors=(ANCHOR,)):
    history,profiles = inputs()
    requested = tuple(command if k=='work' else None for k,_,_ in layout)
    return forecast_extrapolated_work(layout,history,profiles,requested,anchors=anchors,
        response=response,reference=REF,as_of=AT,accounting_key=REF.accounting_key)


def test_supported_predictions_return_exactly_without_using_extrapolation_assumptions():
    history,profiles = inputs()
    expected = forecast_work_for_ranking(LAYOUT,history,profiles,
        tuple(ANCHOR if k=='work' else None for k,_,_ in LAYOUT),
        as_of=AT,accounting_key=REF.accounting_key)
    assert forecast(command=ANCHOR,response=None,anchors=()) == expected


def test_harder_work_adds_load_and_recovery_carryover_not_a_second_whole_dose():
    f = forecast()
    assert f['status']=='predicted'
    detail = f['effort_extrapolation'];anchor = detail['anchor']
    assert detail['status']=='extrapolated'
    assert f['predicted_load'] > anchor['predicted_load']
    assert f['predicted_load'] == pytest.approx(sum(p['load'] for p in f['phases']))
    assert f['phases'][0]['load']==anchor['phases'][0]['load']
    assert f['phases'][2]['load'] > anchor['phases'][2]['load']
    assert f['phases'][2]['effort_transfer']['modeled_carryover']
    assert not detail['transfer_calibrated'] and not detail['uncertainty_calibrated']
    assert not f['phases'][1]['work_exposure']['supported']
    assert f['work_exposure_matching']['requested_intensity_rankable']
    assert detail['load_sensitivity']['low'] < f['predicted_load'] < detail['load_sensitivity']['high']


def test_short_effort_gets_less_increment_than_instantaneous_zone_pricing():
    f = forecast((('work',0.,15.),))
    detail = f['effort_extrapolation']
    multiplier = f['phases'][0]['effort_transfer']['adjustment']
    assert 1 < multiplier < rate(.95,REF.policy)/rate(.8,REF.policy)
    assert detail['nominal']['final_requested_hrr'] < .95
    slow = forecast((('work',0.,15.),),response=replace(RESPONSE,tau_seconds=1000.,sensitivity_taus=()))
    assert 1 < slow['phases'][0]['effort_transfer']['adjustment'] < multiplier


def test_harder_and_longer_single_effort_are_not_cheaper_and_never_exceed_hr_max():
    outputs = []
    for seconds in (15.,45.,137.,300.):
        layout=(('warmup',0.,60.),('work',60.,60.+seconds),('cooldown',60.+seconds,120.+seconds))
        a=forecast(layout,command=WorkExposure((.88,.92),(.88,.92),'hard'))
        b=forecast(layout)
        assert b['predicted_load'] > a['predicted_load']
        assert 0 <= b['effort_extrapolation']['nominal']['final_requested_hrr'] <= 1
        outputs.append(b['predicted_load'])
    assert outputs == sorted(outputs)


def test_unknown_cue_missing_trace_and_missing_lower_anchor_are_not_extrapolated():
    history,profiles = inputs()
    unknown = forecast(command=None)
    assert unknown['predicted_load'] is None
    missing = forecast_extrapolated_work(LAYOUT,history,{},query(LAYOUT,(.9,1.)),
        anchors=(ANCHOR,),response=RESPONSE,reference=REF,as_of=AT,accounting_key=REF.accounting_key)
    assert missing['predicted_load'] is None
    assert forecast(anchors=(WorkExposure((.99,1.),(.99,1.),'not a lower supported band'),))['predicted_load'] is None


def test_future_response_and_wrong_accounting_are_rejected():
    with pytest.raises(ValueError,match='causal'):
        forecast(response=replace(RESPONSE,available_at=AT+timedelta(seconds=1)))
    history,profiles = inputs()
    with pytest.raises(ValueError,match='same declared'):
        forecast_extrapolated_work(LAYOUT,history,profiles,query(LAYOUT,(.9,1.)),anchors=(ANCHOR,),
            response=RESPONSE,reference=REF,as_of=AT,accounting_key='different')


def test_accounting_inverse_does_not_claim_measured_hr_and_limits_are_explicit():
    for r in (0.,.5,.85,1.):
        assert equivalent_reserve(rate(r,REF.policy),REF.policy)==pytest.approx(r)
    with pytest.raises(ValueError,match='bounds'):
        equivalent_reserve(rate(1.,REF.policy)+1.,REF.policy)
    with pytest.raises(ValueError):
        replace(RESPONSE,tau_seconds=0.)


def test_future_and_old_phase_history_cannot_supply_extrapolation_or_change_it():
    history,profiles = inputs()
    source = history[0]
    future = replace(source,key='future',start=AT,end=AT+timedelta(minutes=5),available_at=AT+timedelta(minutes=5))
    old = replace(source,key='old',start=AT-timedelta(days=31),end=AT-timedelta(days=31,minutes=-5),
        available_at=AT-timedelta(days=31,minutes=-5))
    requested = query(LAYOUT,(.9,1.))
    a = forecast_extrapolated_work(LAYOUT,history,profiles,requested,anchors=(ANCHOR,),response=RESPONSE,
        reference=REF,as_of=AT,accounting_key=REF.accounting_key)
    b = forecast_extrapolated_work(LAYOUT,(*history,future,old),profiles,requested,anchors=(ANCHOR,),response=RESPONSE,
        reference=REF,as_of=AT,accounting_key=REF.accounting_key)
    assert a==b
    absent = forecast_extrapolated_work(LAYOUT,(future,old),{},requested,anchors=(ANCHOR,),response=RESPONSE,
        reference=REF,as_of=AT,accounting_key=REF.accounting_key)
    assert absent['predicted_load'] is None


def test_contrast_failures_remain_explicit_not_clipped_to_pass_physiological_bounds():
    p = dict(kind='work',start=0.,end=300.,load=5*rate(.99,REF.policy),rate=rate(.99,REF.policy)/60)
    bad = contrast(dict(phases=[p]),((.8,.8),),((1.,1.),),tau=64.,initial=.8,policy=REF.policy)
    assert bad['status']=='contrast_outside_accounting_bounds'
    assert bad['predicted_load'] is None


def test_half_second_integration_agrees_with_finer_response_grid():
    phases = [dict(kind=k,start=a,end=b,load=(b-a)/60,rate=1/60) for k,a,b in LAYOUT]
    base = ((.6,.6),(.8,.8),(.6,.6),(.8,.8),(.6,.6))
    desired = ((.6,.6),(.95,.95),(.6,.6),(.95,.95),(.6,.6))
    a = contrast(dict(phases=phases),base,desired,tau=64.,initial=.6,policy=REF.policy)
    b = contrast(dict(phases=phases),base,desired,tau=64.,initial=.6,policy=REF.policy,step_seconds=.25)
    assert a['predicted_load']==pytest.approx(b['predicted_load'],rel=1e-5)
