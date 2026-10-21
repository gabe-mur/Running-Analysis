"""Estimator contract tests, not proof of physiological fitness accuracy."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import math

import numpy as np
import pytest

from run_analysis.aerobic_progress import build_aerobic_progress, progress_from_comparability
from run_analysis.analytics import _window_estimate


ORIGIN=datetime(2026,1,1,tzinfo=timezone.utc)


def observations(current=9.5,prior=10.,gradient=(1.,),uncertainty=.8):
    return [dict(activity_id=i,start_time=(ORIGIN+timedelta(days=day)).isoformat(),
                 pace_min_mile=prior if day<28 else current,uncertainty_95_min_mile=uncertainty,
                 trend_weight=1.,included_in_trend=True,shared_log_speed_gradient=list(gradient))
            for i,day in enumerate(range(0,56,4))]


def build(rows,**kwargs):
    return build_aerobic_progress(rows,as_of=kwargs.pop('as_of',ORIGIN+timedelta(days=54)),
        covariance=kwargs.pop('covariance',[[.001]]),**kwargs)


def test_same_robust_level_as_existing_estimator_without_mutating_observations():
    rows=observations();before=deepcopy(rows)
    result=build(rows)
    mapped=[dict(start_time_utc=r['start_time'],standardized_pace=r['pace_min_mile'],
                 uncertainty_95=r['uncertainty_95_min_mile'],trend_weight=r['trend_weight']) for r in rows]
    old=_window_estimate(mapped,ORIGIN+timedelta(days=52),28)
    assert result['current']['pace_min_mile']==pytest.approx(old['pace_min_mile'])
    assert rows==before


def test_recovers_sustained_performance_change_in_interpretable_units():
    result=build(observations(9.5,10.))
    assert result['change']['speed_change_percent']==pytest.approx(100*(10/9.5-1))
    assert result['change']['pace_change_seconds_per_mile']==pytest.approx(-30)
    assert result['change']['direction']=='improving'


def test_display_range_does_not_change_current_state_or_comparison():
    views=[build(observations(),history_days=days) for days in (14,28,90,365)]
    assert all(v['current']==views[0]['current'] and v['change']==views[0]['change'] for v in views)
    assert len(views[0]['history'])<len(views[-1]['history'])


def test_display_retains_one_exact_predecessor_without_changing_estimates():
    rows=observations()
    short=build(rows,history_days=13)
    full=build(rows,history_days=365)
    boundary=ORIGIN+timedelta(days=41)
    older=[r for r in short['history'] if datetime.fromisoformat(r['as_of'])<boundary]
    assert len(older)==1
    assert older[0]['as_of']==(ORIGIN+timedelta(days=40)).isoformat()
    full_by_date={r['as_of']:r for r in full['history']}
    assert all(r==full_by_date[r['as_of']] for r in short['history'])
    assert short['current']==full['current']
    assert short['change']==full['change']


def test_identical_calibration_exposure_cancels_in_change_not_absolute_level():
    result=build(observations())
    assert result['current']['shared_calibration_variance']>0
    assert result['change']['shared_calibration_log_variance']==pytest.approx(0,abs=1e-15)
    assert result['change']['shared_calibration_log_variance_if_periods_were_independent']>0


def test_different_exposure_does_not_receive_fake_shared_error_cancellation():
    rows=observations()
    for row in rows[7:]:row['shared_log_speed_gradient']=[2.]
    result=build(rows)
    assert result['change']['shared_calibration_log_variance']>0


def test_more_runs_cannot_average_away_shared_calibration_error():
    sparse=observations(current=10,prior=10)
    dense=[]
    for i,row in enumerate(sparse):
        for j in range(5):
            dense.append(dict(row,activity_id=i*5+j,start_time=(datetime.fromisoformat(row['start_time'])+timedelta(minutes=j)).isoformat()))
    a,b=build(sparse),build(dense)
    assert b['current']['shared_calibration_variance']==pytest.approx(a['current']['shared_calibration_variance'])
    assert b['current']['uncertainty_95_min_mile']>=1.96*math.sqrt(b['current']['shared_calibration_variance'])


def test_single_bad_run_does_not_become_a_persistent_fitness_decline():
    rows=observations(current=10,prior=10);rows[-1]['pace_min_mile']=13
    result=build(rows)
    assert abs(result['change']['speed_change_percent'])<1
    assert result['change']['direction']=='stable_or_uncertain'


def test_persistent_decline_is_not_robustly_hidden():
    result=build(observations(current=11.,prior=10.))
    assert result['change']['direction']=='declining'
    assert result['change']['speed_change_percent']< -8


def test_training_load_alone_cannot_manufacture_fitness_improvement():
    rows=observations(current=10,prior=10)
    for i,row in enumerate(rows):row['training_load']=100+i*100
    assert build(rows)['change']['speed_change_percent']==pytest.approx(0)


def test_no_new_evidence_holds_last_observation_but_reports_staleness():
    rows=observations();early=build(rows);late=build(rows,as_of=ORIGIN+timedelta(days=120))
    assert late['current']==early['current'] and late['change']==early['change']
    assert late['days_since_observation']>early['days_since_observation']


def test_new_runner_gets_provisional_level_without_eight_run_gate():
    result=build(observations()[:1])
    assert result['available'] and result['current']['evidence']=='provisional'
    assert result['change'] is None


def test_future_and_quality_context_cannot_change_level():
    rows=observations();expected=build(rows)
    rows.extend([dict(rows[-1],activity_id=100,pace_min_mile=5,included_in_trend=False),
                 dict(rows[-1],activity_id=101,pace_min_mile=5,start_time=(ORIGIN+timedelta(days=70)).isoformat())])
    assert build(rows)['current']==expected['current']


def test_duplicate_or_invalid_uncertainty_cannot_manufacture_precision():
    rows=observations()
    with pytest.raises(ValueError,match='Duplicate'):build(rows+[rows[0]])
    with pytest.raises(ValueError,match='positive semidefinite'):build(rows,covariance=[[-1]])
    rows[0]['pace_min_mile']=float('nan')
    with pytest.raises(ValueError,match='Finite positive'):build(rows)


def test_weekly_sensitivity_floor_keeps_serially_related_variation_visible():
    rows=observations(current=10,prior=10)
    for i,row in enumerate(rows):row['pace_min_mile']+=.3 if (i//2)%2 else -.3
    result=build(rows)
    assert result['current']['between_run_spread_seconds_per_mile']>10
    assert result['current']['random_variance']>0


def test_longer_progress_comparison_does_not_change_with_history_crop():
    rows=[]
    for i in range(40):
        rows.append(dict(observations()[0],activity_id=i,
            start_time=(ORIGIN+timedelta(days=i*4)).isoformat(),pace_min_mile=11-i*.02))
    at=ORIGIN+timedelta(days=160)
    a=build(rows,as_of=at,history_days=14);b=build(rows,as_of=at,history_days=365)
    assert a['change_90d']==b['change_90d']
    assert a['change_90d']['speed_change_percent']>a['change']['speed_change_percent']>0
    assert '90 days earlier' in a['change_90d']['comparison']


def test_absolute_fitness_level_uses_exact_adjusted_level_and_fixed_current_window():
    rows=observations(gradient=(.1,.1,.1,.1))
    report=dict(runs=rows,calibration=dict(covariance=np.eye(3).tolist(),heat=dict(sd=.1)))
    config=dict(max_hr=195,resting_hr=46,target_hr=148)
    a=progress_from_comparability(report,as_of=ORIGIN+timedelta(days=54),history_days=14,config=config)
    b=progress_from_comparability(report,as_of=ORIGIN+timedelta(days=54),history_days=365,config=config)
    assert a['fitness_level']['value']==b['fitness_level']['value']
    assert a['fitness_level']['estimated_range_95']==b['fitness_level']['estimated_range_95']
    assert a['fitness_level']['adjusted_pace_min_mile']==a['current']['pace_min_mile']
    assert len(a['fitness_level']['history'])<len(b['fitness_level']['history'])
    assert a['fitness_level']['history'][-1]['value']==a['fitness_level']['value']
    missing=progress_from_comparability(report,as_of=ORIGIN+timedelta(days=54),config={})
    assert missing['available'] and not missing['fitness_level']['available']
