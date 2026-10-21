from dataclasses import replace
from datetime import timedelta
from math import exp, log

import pytest

from run_analysis.adaptive.lag_demand_reference import (
    DemandSample,duration_reference,executed_demand,fit_pace,predict_pace,
)
from run_analysis.signal_structure import SignalSpan
from test_quality_dose_policy import AT


def trace(target,seconds,*,initial=.45,tau=45.,step=.1):
    rows=[];at=0.;state=initial
    while at < seconds:
        end=min(seconds,at+step)
        last=target+(state-target)*exp(-(end-at)/tau)
        rows.append(SignalSpan(at,end,state,last));state=last;at=end
    return tuple(rows)


@pytest.mark.parametrize('seconds',[15.,60.,300.,900.])
def test_integrated_identity_recovers_constant_input_across_short_and_long_efforts(seconds):
    result=executed_demand(trace(.85,seconds),0.,seconds,tau_seconds=45.,bpm_span=145.)
    assert result['status']=='inferred_model_demand'
    assert result['demand']==pytest.approx(.85,abs=2e-7)
    assert result['mean_hrr'] < result['demand']


def test_recovery_decline_removes_inherited_hr_not_actual_cardio_load():
    result=executed_demand(trace(.45,90,initial=.85),0,90,tau_seconds=45,bpm_span=145)
    assert result['mean_hrr'] > .45 and result['endpoint_term'] < 0
    assert result['demand']==pytest.approx(.45,abs=2e-7)


def test_varying_demand_identity_does_not_require_constant_effort_or_phase_reset():
    first=trace(.9,60)
    second=tuple(SignalSpan(s.start+60,s.end+60,s.first,s.last)
        for s in trace(.5,120,initial=first[-1].last))
    result=executed_demand((*first,*second),0,180,tau_seconds=45,bpm_span=145)
    assert result['demand']==pytest.approx((.9*60+.5*120)/180,abs=2e-7)


def test_missing_hr_is_not_filled_or_averaged_over_a_partial_bout():
    rows=(SignalSpan(0,25,.5,.6),SignalSpan(30,60,.6,.7))
    result=executed_demand(rows,0,60,tau_seconds=45,bpm_span=145)
    assert result['status']=='incomplete_hr' and result['demand'] is None
    assert result['known_seconds']==55


def test_noise_sensitivity_is_greater_for_short_bouts_and_not_a_confidence_interval():
    rows=(SignalSpan(0,60,.5,.6),)
    short=executed_demand(rows,0,60,tau_seconds=45,bpm_span=145)
    long=executed_demand((SignalSpan(0,600,.5,.6),),0,600,tau_seconds=45,bpm_span=145)
    assert short['demand_change_bound_per_one_bpm'] > long['demand_change_bound_per_one_bpm']
    perturbed=executed_demand((SignalSpan(0,60,.5-1/145,.6+1/145),),0,60,
                             tau_seconds=45,bpm_span=145)
    assert abs(perturbed['demand']-short['demand']) <= short['demand_change_bound_per_one_bpm']
    assert not short['uncertainty_calibrated']


def test_above_one_model_input_is_not_misclassified_as_invalid_measured_hr():
    result=executed_demand((SignalSpan(0,15,.5,.9),),0,15,tau_seconds=45,bpm_span=145)
    assert result['status']=='inferred_model_demand' and result['demand'] > 1
    assert result['outside_hr_reference_range']
    row=sample('above',15,result['demand'],4.5,mean=result['mean_hrr'])
    assert row.demand==result['demand']


def test_signed_model_input_is_retained_with_warning_without_creating_negative_actual_hr():
    result=executed_demand((SignalSpan(0,15,.9,.5),),0,15,tau_seconds=45,bpm_span=145)
    assert result['demand'] < 0 and result['outside_hr_reference_range']
    assert result['mean_hrr'] > 0


def sample(key,seconds,demand,speed,*,source=None,mean=None):
    return DemandSample(key,source or key,AT-timedelta(days=2),AT-timedelta(days=1),
        AT-timedelta(days=1),'units',seconds,speed,demand if mean is None else mean,demand,'any provenance')


def synthetic():
    return [sample(str(i),seconds,demand,exp(.3+.8*demand-.04*log(seconds)))
        for i,(seconds,demand) in enumerate(((60,.5),(60,.8),(180,.6),(300,.9),(600,.7)))]


def fit(rows,channel='demand'):
    return fit_pace(rows,as_of=AT,accounting_key='units',channel=channel)


def test_prior_per_athlete_coefficients_recover_shared_relation_without_workout_labels():
    model=fit(synthetic())
    assert model['coefficients']==pytest.approx((.3,.8,-.04))
    prediction=predict_pace(model,issued_at=AT,seconds=500,intensity=.75)
    assert prediction['speed_mps']==pytest.approx(exp(.3+.8*.75-.04*log(500)))
    renamed=[replace(r,kind='new workout') for r in synthetic()]
    assert fit(renamed)==model


def test_duplicate_sampling_does_not_increase_a_source_runs_ownership():
    rows=synthetic();model=fit(rows)
    copies=[replace(rows[0],key='copy'+str(i)) for i in range(30)]
    assert fit([*rows,*copies])['coefficients']==pytest.approx(model['coefficients'])


@pytest.mark.parametrize('change',[
    {'available_at':AT+timedelta(seconds=1)},
    {'end':AT,'available_at':AT},
    {'start':AT-timedelta(days=32),'end':AT-timedelta(days=31),'available_at':AT-timedelta(days=31)},
    {'accounting_key':'other'},
])
def test_future_unavailable_old_and_different_normalization_do_not_change_fit(change):
    rows=synthetic();extra=replace(rows[0],key='excluded',source_key='excluded',**change)
    assert fit([*rows,extra])==fit(rows)


def test_no_source_count_or_identical_workout_gate_but_rank_failure_is_explicit():
    rows=[replace(r,source_key='one source') for r in synthetic()]
    assert fit(rows)['status']=='fitted_diagnostic'
    assert fit([rows[0]])['status']=='insufficient_identifiable_evidence'
    assert predict_pace(fit([]),issued_at=AT,seconds=300,intensity=.7)['speed_mps'] is None


def test_nonpositive_intensity_relation_is_not_silently_repaired():
    rows=[replace(r,speed_mps=exp(1.-.5*r.demand+.01*log(r.seconds))) for r in synthetic()]
    assert fit(rows)['status']=='nonpositive_intensity_relation'


def test_duration_control_uses_same_sources_and_transfer_is_disclosed_not_banned():
    rows=synthetic()
    models=[fit(rows,channel) for channel in ('duration','mean_hrr','demand')]
    assert all(m['source_keys']==models[0]['source_keys'] for m in models)
    reference=duration_reference(rows,as_of=AT,accounting_key='units',seconds=1000)
    assert reference['speed_mps'] is not None
    answer=predict_pace(models[2],issued_at=AT,seconds=1000,intensity=.95)
    assert answer['extrapolation'] and answer['speed_mps'] is not None
    with pytest.raises(ValueError,match='Future'):
        predict_pace(dict(models[2],issued_at=AT+timedelta(seconds=1)),issued_at=AT,seconds=60,intensity=.7)


def test_invalid_or_duplicate_inputs_are_rejected():
    with pytest.raises(ValueError,match='unique'):
        fit([synthetic()[0],synthetic()[0]])
    with pytest.raises(ValueError,match='response time'):
        executed_demand((),0,60,tau_seconds=0,bpm_span=145)
    with pytest.raises(ValueError,match='nonoverlapping'):
        executed_demand((SignalSpan(0,40,.5,.6),SignalSpan(20,60,.6,.7)),0,60,
                        tau_seconds=45,bpm_span=145)


def test_same_shared_relation_scales_to_a_different_athlete_without_retuning_coefficients():
    first=fit(synthetic())
    second=fit([replace(r,speed_mps=r.speed_mps*2) for r in synthetic()])
    assert second['coefficients'][0]==pytest.approx(first['coefficients'][0]+log(2))
    assert second['coefficients'][1:]==pytest.approx(first['coefficients'][1:])
    a,b=(predict_pace(m,issued_at=AT,seconds=240,intensity=.7)['speed_mps'] for m in (first,second))
    assert b==pytest.approx(a*2)


def test_above_one_inferred_demand_can_enter_fit_and_query_but_actual_mean_hr_stays_bounded():
    rows=[*synthetic(),sample('high',45,1.08,exp(.3+.8*1.08-.04*log(45)),mean=.82)]
    model=fit(rows)
    prediction=predict_pace(model,issued_at=AT,seconds=45,intensity=1.08)
    assert prediction['speed_mps']==pytest.approx(rows[-1].speed_mps)
    assert prediction['demand_outside_hr_reference_range']
    with pytest.raises(ValueError,match='supported explicit intensity'):
        predict_pace(fit(rows,'mean_hrr'),issued_at=AT,seconds=45,intensity=1.08)
