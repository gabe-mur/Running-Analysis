from datetime import datetime,timezone
from xml.etree import ElementTree as ET

import pytest

from scripts.new_model_daily_harness import Dose
from scripts.recorded_trace_execution import transformed_payload,recorded_phase_bounds


AT=datetime(2026,9,27,tzinfo=timezone.utc)


def source():
    return dict(key='original',role='quality',source_sha256='original-hash',
        phases=(('warmup',0,60),('work',60,120),('recovery',120,180),('cooldown',180,240)),
        points=((0,0,120),(30,60,125),(60,120,130),(90,240,160),(120,360,170),
            (150,420,None),(180,480,140),(210,540,135),(240,600,130)))


def test_phase_scaling_preserves_distance_timing_hr_gaps_and_source_identity():
    dose=Dose('quality','quality',3.,(('warmup',120.),('work',30.),('recovery',60.),
        ('work',30.),('cooldown',120.)))
    original=source()
    payload,phases,end,provenance=transformed_payload(original,dose,AT,distance_factor=1.1,duration_factor=.9)
    xml=ET.fromstring(payload.content)
    points=xml.findall('.//Trackpoint')
    times=[datetime.fromisoformat(p.findtext('Time')) for p in points]
    distances=[float(p.findtext('DistanceMeters')) for p in points]
    assert all(b>a for a,b in zip(times,times[1:]))
    assert all(b>=a for a,b in zip(distances,distances[1:]))
    assert distances[-1]==pytest.approx(3.3*1609.344)
    assert (end-AT).total_seconds()==pytest.approx(sum(s for _,s in phases))
    assert times[-1]==end
    assert any(p.find('HeartRateBpm') is None for p in points)
    assert any(p.findtext('HeartRateBpm/Value')=='160' for p in points)
    assert provenance['source_sha256']=='original-hash' and provenance['synthetic_not_physiological_prediction']
    assert original==source()


@pytest.mark.parametrize('factor',[0.,-1.,float('nan'),float('inf')])
def test_invalid_factors_rejected(factor):
    with pytest.raises(ValueError,match='factors'):
        transformed_payload(source(),Dose('e','easy',1.,(('warmup',60.),)),AT,distance_factor=factor)


def test_absent_quality_phase_cannot_be_invented():
    data=source()
    data['phases']=(('warmup',0,240),)
    with pytest.raises(ValueError,match='No original work'):
        transformed_payload(data,Dose('q','quality',1.,(('work',60.),)),AT)


def test_fractional_lap_timer_boundary_trims_to_recorded_samples_without_extrapolation():
    phases,trimmed=recorded_phase_bounds((('warmup',0.,2557.212),),0.,2557.)
    assert phases==(('warmup',0.,2557.),)
    assert trimmed==pytest.approx(.212)
    data=source(); data['phases']=(('warmup',0.,240.),); data['source_boundary_trim_seconds']=.212
    payload,_,_,provenance=transformed_payload(data,Dose('e','easy',2.,(('warmup',300.),)),AT)
    assert provenance['source_boundary_trim_seconds']==.212
    assert len(ET.fromstring(payload.content).findall('.//Trackpoint'))==len(data['points'])


@pytest.mark.parametrize('phases,first,last',[
    ((('warmup',0.,240.),),0.,238.),
    ((('work',0.,1.),),1.,240.),
])
def test_missing_source_coverage_cannot_be_hidden_as_timer_rounding(phases,first,last):
    with pytest.raises(ValueError,match='covered'):
        recorded_phase_bounds(phases,first,last)


def test_donor_selection_does_not_use_progression_for_interval_recovery(monkeypatch):
    from types import SimpleNamespace
    from scripts.recorded_trace_execution import compatible_donor
    dose=Dose('quality:ordinary','quality',3.,(('warmup',60.),('work',60.),('recovery',30.),('cooldown',60.)))
    progression=SimpleNamespace(key='progression',miles=3.,start=AT)
    intervals=SimpleNamespace(key='intervals',miles=3.2,start=AT)
    data=dict(progression=dict(phases=(('warmup',0,60),('work',60,120),('cooldown',120,180))),intervals=source())
    monkeypatch.setattr('scripts.recorded_trace_execution.donor',lambda db,run,*args:data[run.key])
    selected,trace=compatible_donor(None,(progression,intervals),dose,None,None,{})
    assert selected==intervals and trace==source()
    with pytest.raises(ValueError,match='No original donor'):
        compatible_donor(None,(progression,),dose,None,None,{})


def test_alias_requires_unique_exact_original_identity_not_nearest_run():
    import sqlite3
    from datetime import timedelta
    from types import SimpleNamespace
    from scripts.recorded_trace_execution import original_activity_id
    db=sqlite3.connect(':memory:')
    db.execute('CREATE TABLE activities(id,start_time_utc,start_time_utc_epoch,total_distance_m,total_elapsed_time_s)')
    run=SimpleNamespace(key='sandbox:2',start=AT,end=AT+timedelta(seconds=2000),miles=5195.92/1609.344)
    values=(387,AT.isoformat(),AT.timestamp(),5195.92,2000.)
    db.execute('INSERT INTO activities VALUES (?,?,?,?,?)',values)
    assert original_activity_id(db,run)==387
    db.execute('UPDATE activities SET total_distance_m=total_distance_m+1')
    with pytest.raises(ValueError,match='exact original'):original_activity_id(db,run)
    db.execute('UPDATE activities SET total_distance_m=5195.92')
    db.execute('INSERT INTO activities VALUES (?,?,?,?,?)',(388,*values[1:]))
    with pytest.raises(ValueError,match='exact original'):original_activity_id(db,run)
    db.close()


def measured_segments(payload, phases):
    from bisect import bisect_right
    points=ET.fromstring(payload.content).findall('.//Trackpoint')
    times=[(datetime.fromisoformat(p.findtext('Time'))-AT).total_seconds() for p in points]
    meters=[float(p.findtext('DistanceMeters')) for p in points]
    def value(t):
        i=max(0,min(len(times)-2,bisect_right(times,t)-1))
        return meters[i]+(meters[i+1]-meters[i])*(t-times[i])/(times[i+1]-times[i])
    cursor=0.; result=[]
    for _,seconds in phases:
        result.append(value(cursor+seconds)-value(cursor));cursor+=seconds
    return result


@pytest.mark.parametrize('recipe',['threshold_continuous','threshold_repeats',
    'aerobic_timed','aerobic_distance','pickups_timed','pickups_distance',
    'fartlek_surges','progression','hill_repeats'])
@pytest.mark.parametrize('factor',[1.,.94,1.08])
def test_native_commands_commit_primary_units_not_forecast_meters(recipe,factor):
    from test_quality_recipe_catalogue import catalogue
    rows=catalogue()['candidates']
    if recipe=='progression':
        from run_analysis.adaptive.quality_recipe_catalogue import progression_options
        from run_analysis.adaptive.quality_block_contract import adapt_v2
        from test_quality_recipe_catalogue import menu,window
        base=adapt_v2(next(p for p in menu()['prescriptions'] if p.request.family=='progression'))
        options=progression_options(base,(4.2,),window=window(1,3600))
        p=options[0].prescription
    else:p=next(c.prescription for c in rows if c.recipe_key==recipe)
    phases=tuple((b.kind,b.seconds.estimate) for b in p.phases)
    dose=Dose(p.key,'quality',p.miles,phases,
        phase_scenarios=tuple((b.seconds.low,b.seconds.high) for b in p.phases),
        workout_definition=p.definition())
    payload,execution,_,provenance=transformed_payload(source(),dose,AT,
        distance_factor=factor,duration_factor=factor)
    measured=measured_segments(payload,execution)
    assert sum(measured)==pytest.approx(p.miles*1609.344*factor,abs=1e-5)
    for b,(_,seconds),distance in zip(p.phases,execution,measured):
        if b.primary.unit=='time':assert seconds==pytest.approx(b.primary.amount.estimate*factor)
        elif b.primary.unit=='distance':assert distance==pytest.approx(b.primary.amount.estimate*factor,abs=1e-5)
        elif b.primary.unit=='portion':assert distance==pytest.approx(b.primary.amount.estimate*p.miles*1609.344*factor,abs=1e-5)
    assert provenance['execution_contract']=='primary_units_v2'


def test_forecast_distance_for_timed_work_is_not_forced_and_bad_contract_rejected():
    from types import SimpleNamespace
    dose=SimpleNamespace(miles=1.,phases=(('warmup',60.),('work',60.),),
        workout_definition=dict(numeric_contract=dict(phases=[
            dict(kind=k,primary=dict(unit='time',amount=dict(estimate=60.)),meters=dict(estimate=800.))
            for k in ('warmup','work')])))
    payload,phases,_,_=transformed_payload(source(),dose,AT)
    assert measured_segments(payload,phases)[1]!=pytest.approx(800.)
    dose.workout_definition['numeric_contract']['phases'][1]['primary']=dict(unit='distance',amount=dict(estimate=2000.))
    with pytest.raises(ValueError,match='cannot fit'):transformed_payload(source(),dose,AT)
