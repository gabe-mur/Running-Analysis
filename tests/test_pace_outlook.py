from copy import deepcopy
from datetime import date,datetime,timedelta,timezone
import math
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

from run_analysis import pace_outlook as po


AT=datetime(2026,10,20,20,tzinfo=timezone.utc)


def constant_fits():
    return {policy:dict(calendar=dict(coefficients=dict(elapsed=math.log(.98))),
        log_miles=dict(coefficients=dict(log_miles=-.04))) for policy in po.POLICIES}


def test_curve_anchors_once_and_center_is_scenario_midpoint_without_a_time_taper():
    fits=constant_fits()
    rows=po.projected_curve(fits,start=AT.date(),end=AT.date()+timedelta(days=42),
        current_pace=10,prior_miles=500,volume=20,growth=.04,target_volume=20)
    assert rows[0]['pace_min_mile']==pytest.approx(10)
    sustained=10*.98**1.5;diminishing=10*(1+120/500)**-.04
    assert rows[-1]['low_pace_min_mile']==pytest.approx(min(sustained,diminishing))
    assert rows[-1]['high_pace_min_mile']==pytest.approx(max(sustained,diminishing))
    assert rows[-1]['pace_min_mile']==pytest.approx((sustained+diminishing)/2)
    assert rows[-1]['date']=='2026-12-01'
    assert all(r['low_pace_min_mile']<=r['pace_min_mile']<=r['high_pace_min_mile'] for r in rows)
    for variants in fits.values():
        for fit in variants.values():
            for key in fit['coefficients']:fit['coefficients'][key]=0
    flat=po.projected_curve(fits,start=AT.date(),end=AT.date()+timedelta(days=42),
        current_pace=10,prior_miles=500,volume=20,growth=.04,target_volume=42.5)
    assert all(r['pace_min_mile']==10 for r in flat)
    with pytest.raises(ValueError):
        po.projected_curve(fits,start=AT.date(),end=AT.date()+timedelta(days=500),current_pace=10,
            prior_miles=500,volume=20,growth=.04,target_volume=42.5)


def test_future_changes_are_not_forced_to_be_positive_or_floored_at_current_pace():
    fits=constant_fits()
    for v in fits.values():
        v['calendar']['coefficients']['elapsed']=.01
        v['log_miles']['coefficients']['log_miles']=.02
    rows=po.projected_curve(fits,start=AT.date(),end=AT.date()+timedelta(days=28),
        current_pace=10,prior_miles=500,volume=20,growth=.04,target_volume=42.5)
    assert rows[-1]['low_pace_min_mile']>10


def fixture(monkeypatch):
    c=sqlite3.connect(':memory:');c.row_factory=sqlite3.Row
    c.execute('CREATE TABLE activities(id INTEGER,start_time_utc TEXT,total_elapsed_time_s REAL)')
    rows=[];sessions=[];details={}
    for i in range(90):
        at=AT-timedelta(days=(89-i)*3+1,hours=8)
        pace=10.8*math.exp(-.001*i)
        rows.append(dict(activity_id=i+1,start_time=at.isoformat(),pace_min_mile=pace,
            included_in_trend=True,trend_weight=1,workout_type='long' if i%4==0 else 'easy'))
        c.execute('INSERT INTO activities VALUES (?,?,?)',(i+1,at.isoformat(),3600))
        sessions.append(SimpleNamespace(activity_id=i+1,start_time=at,distance_miles=8 if i%4==0 else 4,moving_minutes=50))
        details[i+1]=dict(workout=SimpleNamespace(value=rows[-1]['workout_type']))
    # A quality activity is training even though it is not an aerobic outcome.
    qtime=AT-timedelta(days=2)
    c.execute('INSERT INTO activities VALUES (?,?,?)',(501,qtime.isoformat(),2400))
    sessions.append(SimpleNamespace(activity_id=501,start_time=qtime,distance_miles=4,moving_minutes=40))
    details[501]=dict(workout=SimpleNamespace(value='intervals'))
    monkeypatch.setattr(po,'prepare_progress_data',lambda connection:SimpleNamespace(sessions=sessions,details=details))
    monkeypatch.setattr(po,'comparison_covariance',lambda c:np.zeros((1,1)))
    monkeypatch.setattr(po,'measurement_inputs',lambda rows,c:(np.full(len(rows),.001),np.zeros((len(rows),1))))
    report=dict(runs=rows,calibration={})
    performance=dict(available=True,current=dict(available=True,pace_min_mile=10.),last_observation=rows[-1]['start_time'],
        target_hr_bpm=148,progress_trend=dict(log_pace_change_per_day=-.0005),hidden_activity_ids=[],days_since_observation=1)
    goal=dict(status='supporting_volume',volume_per_7=19.4,volume_growth_per_7=.04,target_volume_per_7=42.5,
        projection=dict(target_date='2027-05-19',points=[dict(date='2026-10-20'),dict(date='2027-05-19')]))
    return c,report,performance,goal,sessions,details


def test_uses_all_history_and_quality_training_without_writing(monkeypatch):
    c,r,p,g,_,_=fixture(monkeypatch);before=c.total_changes;original=deepcopy(r)
    value=po.pace_outlook(c,r,p,g,as_of=AT,timezone='America/New_York')
    assert value['available'] and value['training_runs']==91
    assert value['latest_activity_id']==90 and value['date']=='2027-05-19'
    assert value['current_pace_min_mile']==pytest.approx(10.8*math.exp(-.089))
    assert value['smoothing_days']==42
    assert value['error_growth_exponent']>=0 and value['calibration_origins']>=8
    assert value['reference_hr_bpm']==148
    assert value['range_kind'].endswith('not a calibrated prediction interval')
    assert c.total_changes==before and r==original
    # Cropping the display history does not crop the training inputs or refit.
    cropped={**p,'history':[],'observations':[]}
    assert po.pace_outlook(c,r,cropped,g,as_of=AT,timezone='America/New_York')['points']==value['points']


def test_new_completed_recording_updates_current_anchor_and_forecast(monkeypatch):
    c,r,p,g,sessions,details=fixture(monkeypatch)
    first=po.pace_outlook(c,r,p,g,as_of=AT,timezone='America/New_York')
    at=AT-timedelta(hours=4)
    c.execute('INSERT INTO activities VALUES (?,?,?)',(91,at.isoformat(),3600))
    sessions.append(SimpleNamespace(activity_id=91,start_time=at,distance_miles=9,moving_minutes=90))
    details[91]=dict(workout=SimpleNamespace(value='long'))
    r['runs'].append(dict(r['runs'][-1],activity_id=91,start_time=at.isoformat(),pace_min_mile=9.6))
    p={**p,'last_observation':at.isoformat(),'current':dict(available=True,pace_min_mile=9.8)}
    second=po.pace_outlook(c,r,p,g,as_of=AT,timezone='America/New_York')
    assert second['latest_activity_id']==91 and second['training_runs']==92
    assert second['points'][0]['pace_min_mile']==second['current_pace_min_mile']
    assert second['current_pace_min_mile']<first['current_pace_min_mile']
    assert second['latest_calibration_outcome']>first['latest_calibration_outcome']
    assert second['error_growth_exponent']!=first['error_growth_exponent']
    assert second['points']!=first['points']


def test_live_range_is_from_actual_future_error_not_the_response_scenario_spread(monkeypatch):
    c,r,p,g,_,_=fixture(monkeypatch)
    fitted=po._fit_history(c,r,p,AT,timezone.utc)
    calibration=fitted['error_calibration']
    rows=po.projected_curve(fitted['fits'],start=AT.date(),end=date(2027,5,19),
        current_pace=fitted['current_pace'],prior_miles=fitted['prior_miles'],volume=g['volume_per_7'],
        growth=g['volume_growth_per_7'],target_volume=g['target_volume_per_7'],error_calibration=calibration)
    last=rows[-1];radius=po.error_radius(calibration,211)
    assert last['low_pace_min_mile']==pytest.approx(last['pace_min_mile']*math.exp(-radius))
    assert last['high_pace_min_mile']==pytest.approx(last['pace_min_mile']*math.exp(radius))
    assert rows[0]['pace_min_mile']==fitted['current_pace']
    for point in rows:
        elapsed=(date.fromisoformat(point['date'])-AT.date()).days
        if elapsed<calibration['minimum_horizon_days']:
            assert point['low_pace_min_mile'] is None and point['high_pace_min_mile'] is None
        else:
            assert point['low_pace_min_mile']<point['pace_min_mile']<point['high_pace_min_mile']
    baseline=po.projected_curve(fitted['fits'],start=AT.date(),end=date(2027,5,19),
        current_pace=fitted['current_pace'],prior_miles=fitted['prior_miles'],volume=g['volume_per_7'],
        growth=g['volume_growth_per_7'],target_volume=g['target_volume_per_7'])
    assert [r['pace_min_mile'] for r in rows]==[r['pace_min_mile'] for r in baseline]
    assert last['low_pace_min_mile']!=pytest.approx(baseline[-1]['low_pace_min_mile'])
    with pytest.raises(ValueError,match='earlier than the tested'):
        po.projected_curve(fitted['fits'],start=AT.date(),end=AT.date()+timedelta(days=1),
            current_pace=fitted['current_pace'],prior_miles=fitted['prior_miles'],volume=g['volume_per_7'],
            growth=g['volume_growth_per_7'],target_volume=g['target_volume_per_7'],error_calibration=calibration)


def test_health_context_changes_training_response_cohort_not_recorded_training(monkeypatch):
    c,r,p,g,_,details=fixture(monkeypatch)
    first=po.pace_outlook(c,r,p,g,as_of=AT,timezone='America/New_York')
    for i in range(40,46):details[i]['health_tag']='illness_recovery'
    before=c.total_changes
    second=po.pace_outlook(c,r,p,g,as_of=AT,timezone='America/New_York')
    assert second['available'] and second['training_runs']==first['training_runs']
    assert second['response_examples']<first['response_examples']
    assert c.total_changes==before


def test_live_helpers_are_the_same_pure_functions_used_in_the_audits():
    from run_analysis import training_response as shared
    from scripts import audit_empirical_training_response as empirical
    from scripts import audit_consistent_training_response as consistent
    assert po.benchmark_examples is empirical.benchmark_examples is shared.benchmark_examples
    assert po.fit_response is empirical.fit_response is shared.fit_response
    assert po.annotate is consistent.annotate is shared.annotate


@pytest.mark.parametrize('status',['holding','review','update_pending','taper','at_target'])
def test_no_invented_forecast_when_progression_is_not_active(monkeypatch,status):
    c,r,p,g,_,_=fixture(monkeypatch)
    assert not po.pace_outlook(c,r,p,{**g,'status':status},as_of=AT,timezone='America/New_York')['available']


def test_new_user_gets_no_fictional_pace_and_long_gaps_do_not_reuse_old_era(monkeypatch):
    c,r,p,g,_,_=fixture(monkeypatch)
    assert not po.pace_outlook(c,{**r,'runs':r['runs'][-3:]},p,g,as_of=AT,timezone='America/New_York')['available']
    isolated={**r,'runs':r['runs'][:20]+r['runs'][-3:]}
    assert not po.pace_outlook(c,isolated,p,g,as_of=AT,timezone='America/New_York')['available']
