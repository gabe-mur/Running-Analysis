from copy import deepcopy
from datetime import datetime,timedelta,timezone
import math

import numpy as np
import pytest

from scripts.audit_empirical_training_response import (
    benchmark_examples,choose,disjoint,evaluate,features,fit_response,predict,residual_range,scenario,
)

AT=datetime(2026,1,1,tzinfo=timezone.utc)


def examples(effect=-.04):
    rng=np.random.default_rng(7);rows=[]
    for i in range(65):
        origin=AT+timedelta(days=i*7)
        miles=float(rng.uniform(.3,2));elapsed=float(rng.uniform(.7,1.2))
        change=effect*miles
        rows.append(dict(origin=origin.isoformat(),origin_id=i,outcome_end=(origin+timedelta(days=28)).isoformat(),
            horizon=28,target_id=str(i),outcome_ids=[i],response_variance=.001,elapsed=elapsed,
            miles=miles,minutes=miles*1.02,log_miles=math.log1p(miles),change=change,
            start_pace=10,actual=10*math.exp(change)))
    return rows


def test_direct_fit_recovers_training_response_without_forcing_improvement():
    rows=examples();fit=fit_response(rows,'miles')
    assert fit['coefficients']['miles']==pytest.approx(-.04)
    bad=fit_response(examples(effect=.02),'miles')
    assert bad['coefficients']['miles']==pytest.approx(.02)
    forecasts,_=evaluate(rows)
    dose=[r for r in forecasts if r['model']=='miles']
    clock=[r for r in forecasts if r['model']=='calendar']
    assert max(abs(r['predicted']-r['actual']) for r in dose)<1e-10
    assert np.mean([abs(r['predicted']-r['actual']) for r in clock])>.05


def test_chronological_fit_selection_and_calibration_do_not_see_future_labels():
    rows=examples();altered=deepcopy(rows);cutoff=AT+timedelta(days=300)
    for r in altered:
        if datetime.fromisoformat(r['outcome_end'])>=cutoff:
            r['change']+=.4;r['actual']*=math.exp(.4)
    a,sa=evaluate(rows);b,sb=evaluate(altered)
    before=lambda r:datetime.fromisoformat(r['origin'])<=cutoff
    assert [r['predicted'] for r in a if before(r)]==[r['predicted'] for r in b if before(r)]
    assert [(r['model'],r['predicted'],r['range']) for r in sa if before(r)]==[(r['model'],r['predicted'],r['range']) for r in sb if before(r)]
    for r in sa:
        assert r['latest_training_outcome']<r['origin'] and r['latest_selection_outcome']<r['origin']
        if r['range']['available']:assert r['range']['latest_calibration_outcome']<r['origin']


def test_features_match_weighted_log_outcomes_and_origin_completed_training():
    origin=AT+timedelta(days=10)
    training=[dict(time=AT,available_at=AT+timedelta(hours=1),miles=10),
        dict(time=origin-timedelta(hours=1),available_at=origin,miles=5),
        dict(time=origin+timedelta(days=1),available_at=origin+timedelta(days=1,hours=1),miles=100)]
    e=dict(origin=origin.isoformat(),delivered_total_miles=5,
        training_by_outcome=[dict(weight=.25,total_miles=2,minutes=70,days=2),
            dict(weight=.75,total_miles=6,minutes=140,days=3)])
    result=features([e],training,AT)[0]
    assert result['origin_cumulative_miles']==15
    assert result['log_miles']==pytest.approx(.25*math.log1p(2/15)+.75*math.log1p(6/15))
    assert result['minutes']==pytest.approx((.25*70*2/7+.75*140*3/7)/1000)


def residual(i,error):
    return dict(origin=(AT+timedelta(days=i*28)).isoformat(),
        outcome_end=(AT+timedelta(days=i*28+21)).isoformat(),outcome_ids=[i],log_error=error)


def test_ranges_use_actual_forecast_misses_not_current_pace_or_model_spread():
    errors=[residual(i,error) for i,error in enumerate([.02,.04,.06,.08])]
    interval=residual_range(10,errors)
    assert interval['available'] and interval['high_pace']<10
    assert interval['low_pace']==pytest.approx(10*math.exp(-.074))
    assert not residual_range(10,errors[:2])['available']
    duplicate={**errors[0],'origin':(AT+timedelta(days=1)).isoformat()}
    assert len(disjoint([*errors,duplicate]))==4


def test_no_time_taper_is_imposed_on_learned_linear_dose_response():
    fit=dict(coefficients=dict(miles=-.1))
    goal=dict(volume_per_7=20,volume_growth_per_7=.04,target_volume_per_7=20)
    points=scenario(fit,goal,at=AT,current_pace=10,prior_miles=200,minutes_per_mile=10,days=56)
    assert points[-1]['pace']==pytest.approx(10*math.exp(-.1*160/100))
    changes=np.diff(np.log([r['pace'] for r in points]))
    assert np.ptp(changes)<1e-12
    assert predict(fit,dict(miles=0))==0


def test_history_candidates_filter_origins_not_just_recent_outcome_labels():
    rows=examples();at=AT+timedelta(days=400)
    fit=fit_response(rows,'miles_recent84',as_of=at)
    assert fit and fit['origins']<len(rows)
    assert fit['coefficients']['miles']==pytest.approx(-.04)


def test_model_selection_uses_only_common_hindcast_cases():
    rows=[]
    for i in range(10):
        for model,error in [('calendar',.02),('miles',.01)]:
            rows.append(dict(model=model,origin=str(i),target_id=str(i),horizon=28,
                predicted=10*math.exp(error),actual=10,log_error=error,outcome_ids=[i]))
    assert choose(rows)=='miles'
    # Add very good unmatched calendar cases: those cannot change the winner.
    rows.extend(dict(model='calendar',origin=str(i),target_id=str(i),horizon=28,
        predicted=10,actual=10,log_error=0,outcome_ids=[i]) for i in range(10,100))
    assert choose(rows)=='miles'


def test_benchmark_target_rebuilds_prefix_without_later_chart_information():
    rows=[dict(activity_id=i,start_time=(AT+timedelta(days=i*3)).isoformat(),
        pace_min_mile=10*math.exp(-i*.001)) for i in range(40)]
    e=dict(origin_id=10,outcome_ids=[17,18],
        training_by_outcome=[dict(weight=.5),dict(weight=.5)])
    m=np.full(40,.001);g=np.zeros((40,0));cov=np.zeros((0,0))
    first=benchmark_examples([e],rows,m,g,cov)[0]
    changed=deepcopy(rows)
    for r in changed[20:]:r['pace_min_mile']*=1.5
    assert benchmark_examples([e],changed,m,g,cov)==[first]
    assert first['start_pace']==pytest.approx(10*math.exp(-.01))
    assert first['actual']==pytest.approx(10*math.exp(-.0175))
