"""Offline routing/clock/unknown controls, not recipe efficacy validation."""
from dataclasses import replace
from datetime import timedelta

import pytest

from run_analysis.adaptive.quality_block_contract import read_quality_contract
from run_analysis.adaptive.quality_catalogue_forecast import (
    forecast_prescription, forecast_prescription_point, prescription_query)
from run_analysis.adaptive.unsupported_effort_transfer import ResponseAssumption
from run_analysis.adaptive.work_exposure_matching import forecast_work_for_ranking, requested_work_bands
from run_analysis.continuous_training_dose import ContinuousDoseReference
from test_quality_block_contract import block, prescription
from test_quality_dose_policy import AT, CONFIG as PARTIAL_CONFIG, menu
from test_quality_recipe_catalogue import catalogue
from test_work_exposure_matching import record


CONFIG=dict(PARTIAL_CONFIG,zones=dict(PARTIAL_CONFIG['zones'],z1=(100,135),z2=(136,153),z5=(181,194)))
REF=ContinuousDoseReference('catalogue routing fixed test accounting')


def evidence():
    source,profile=record('past',.85)
    source=replace(source,start=AT-timedelta(days=1),end=AT-timedelta(days=1,minutes=-5),
        available_at=AT-timedelta(days=1,minutes=-5),accounting_key=REF.accounting_key)
    return (source,),{'past':profile}


def example():
    return prescription(block('work',120,cues=('Z4',)))


def test_supported_nominal_is_exact_existing_prediction_and_phase_sum():
    p=example();history,profiles=evidence();layout,requested=prescription_query(p,CONFIG)
    expected=forecast_work_for_ranking(layout,history,profiles,requested,as_of=AT,accounting_key=REF.accounting_key)
    f=forecast_prescription(p,iter(history),profiles,CONFIG,reference=REF)
    assert f['nominal']==expected
    assert f['all_scenarios_predicted']
    assert f['nominal']['predicted_load']==pytest.approx(sum(q['load'] for q in f['nominal']['phases']))
    assert not f['planner_influence'] and not f['purpose_credit_assigned'] and not f['uncertainty_calibrated']
    assert f['scenario_envelope']['low']==f['scenario_envelope']['high']


def test_time_clock_fixed_distance_clock_native_and_fartlek_range_preserved():
    rows=catalogue()['candidates']
    for key in ('aerobic_timed','pickups_distance','fartlek_surges'):
        p=next(c.prescription for c in rows if c.recipe_key==key)
        layouts={point:prescription_query(p,CONFIG,point=point)[0]
            for point in ('lower_speed','nominal','higher_speed')}
        seconds=lambda point:layouts[point][1][2]-layouts[point][1][1]
        if key=='aerobic_timed':
            assert seconds('lower_speed')==seconds('nominal')==seconds('higher_speed')
        elif key=='pickups_distance':
            assert seconds('lower_speed')>seconds('nominal')>seconds('higher_speed')
        else:
            assert [seconds(p) for p in layouts]==[90,60,45]
        for layout in layouts.values():
            assert layout[0][1]==0 and all(a[2]==b[1] for a,b in zip(layout,layout[1:]))


def test_mixed_blocks_use_independent_cues_no_family_gate_and_restart_exact():
    p=prescription(block('work',90,cues=('Z3',)),block('work',200,unit='distance',cues=('Z4',)),
        family='novel_mixed')
    layout,requested=prescription_query(p,CONFIG)
    assert requested==requested_work_bands(p.phases,CONFIG)
    assert requested[1]!=requested[2] and requested[0] is None and requested[-1] is None
    restored=read_quality_contract(p.definition()['numeric_contract'])
    assert prescription_query(restored,CONFIG)==(layout,requested)
    h,t=evidence()
    assert forecast_prescription(restored,h,t,CONFIG,reference=REF)==forecast_prescription(p,h,t,CONFIG,reference=REF)


def test_missing_support_is_not_zero_or_diagnostic_timing_load():
    h,t=evidence()
    p=prescription(block('work',90,cues=('Z5',)))
    f=forecast_prescription(p,h,t,CONFIG,reference=REF)
    assert f['nominal']['predicted_load'] is None
    assert f['nominal']['work_exposure_matching']['diagnostic_predicted_load']>0
    assert f['scenario_envelope'] is None and not f['all_scenarios_predicted']
    cold=forecast_prescription(example(),(),{},CONFIG,reference=REF)
    assert cold['nominal']['predicted_load'] is None
    unknown=prescription(block('work',90,cues=('unknown effort',)))
    assert forecast_prescription(unknown,h,t,CONFIG,reference=REF)['nominal']['predicted_load'] is None


def test_extrapolation_remains_explicit_with_carryover_and_causal_assumptions():
    h,t=evidence()
    anchor=requested_work_bands(example().phases,CONFIG)[1]
    response=ResponseAssumption(64,AT-timedelta(days=1),('past',),'frozen prior test fit')
    p=prescription(block('work',90,cues=('Z5',)))
    f=forecast_prescription(p,h,t,CONFIG,reference=REF,anchors=(anchor,),response=response)
    assert f['all_scenarios_predicted']
    assert f['nominal']['prediction_basis']=='extrapolated_effort_not_observed_support'
    assert not f['nominal']['phases'][1]['work_exposure']['supported']
    assert f['nominal']['phases'][-1]['effort_transfer']['modeled_carryover']
    with pytest.raises(ValueError,match='causal'):
        forecast_prescription(p,h,t,CONFIG,reference=REF,anchors=(anchor,),
            response=replace(response,available_at=AT+timedelta(seconds=1)))


def test_v2_supported_clock_and_cost_equal_and_future_sources_ignored():
    p=next(p for p in menu()['prescriptions'] if p.request.family=='short_intervals')
    h,t=evidence();layout,requested=prescription_query(p,CONFIG)
    assert [b-a for _,a,b in layout]==pytest.approx([q.seconds.estimate for q in p.phases])
    expected=forecast_work_for_ranking(layout,h,t,requested,as_of=p.request.issued_at,accounting_key=REF.accounting_key)
    assert forecast_prescription_point(p,h,t,CONFIG,reference=REF)==expected
    future=replace(h[0],key='future',start=AT,end=AT+timedelta(minutes=5),available_at=AT+timedelta(minutes=5))
    assert forecast_prescription(p,(*h,future),t,CONFIG,reference=REF)==forecast_prescription(p,h,t,CONFIG,reference=REF)
    with pytest.raises(TypeError,match='Frozen'):
        prescription_query({},CONFIG)
    with pytest.raises(ValueError,match='scenario'):
        prescription_query(p,CONFIG,point='invented')


def test_all_v2_nominal_commands_reuse_exact_compiled_clock_including_progression():
    for p in menu()['prescriptions']:
        clock=0.;expected=[]
        for phase in p.phases:
            end=clock+phase.seconds.estimate
            expected.append((phase.kind,clock,end));clock=end
        assert prescription_query(p,CONFIG)[0]==tuple(expected)
