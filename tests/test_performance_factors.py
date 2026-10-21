from copy import deepcopy
from datetime import datetime, timedelta, timezone
import math

import numpy as np
import pytest

from run_analysis.performance_factors import analyze_performance_factors, _design, _fit, _prepare


def sample(coefficient=.012, seasonal_only=False):
    rng=np.random.default_rng(18)
    origin=datetime(2024,1,1,tzinfo=timezone.utc)
    rows=[]
    for i in range(160):
        day=i*3
        heat=12+5*math.sin(day/90)+(0 if seasonal_only else rng.normal(0,4))
        hr=145+rng.normal(0,4)
        minutes=25+rng.normal(0,4)
        block=math.floor(origin.timestamp()/86400/56+day/56)
        if seasonal_only: heat=block*2.
        log_pace=math.log(10)-day*.0002-.007*(hr-145)+.001*(minutes-25)+coefficient*heat+rng.normal(0,.008)
        rows.append(dict(activity_id=i,start_time_utc=(origin+timedelta(days=day)).isoformat(),
            observed_pace_min_mile=math.exp(log_pace),sampled_hr_bpm=hr,sampled_minutes=minutes,
            heat_c=heat,workout_type='easy',device='watch',health_tag='unreported'))
    return rows


def test_recovers_planted_association_and_chronological_benefit():
    result=analyze_performance_factors(sample())
    heat=next(f for f in result['primary']['factors'] if f['key']=='heat_c')
    assert heat['log_pace_coefficient_standardized']/heat['scale']==pytest.approx(.012,abs=.002)
    assert heat['range_95_seconds_per_mile'][0]>0
    check=result['chronological_check']
    assert check['factors_mae_seconds_per_mile']<check['baseline_mae_seconds_per_mile']
    assert result['adjusted_trend_promoted'] is False


def test_missing_optional_context_does_not_delete_runs_or_become_measured_zero():
    rows=sample()
    for r in rows[::3]:r['heat_c']=None
    rows[1]['grade_percent']=None
    result=analyze_performance_factors(rows)
    assert result['observations']==len(rows)
    coverage=next(c for c in result['coverage'] if c['key']=='heat_c')
    assert coverage['missing_runs']==54
    matrix,spec=_design(rows)
    heat=next(i for i,f in enumerate(spec) if f['key']=='heat_c' and f['kind']=='numeric')
    missing=next(i for i,f in enumerate(spec) if f['key']=='heat_c' and f['kind']=='missing')
    assert matrix[0,heat]==0  # centered design, not a recorded temperature
    assert matrix[0,missing]==1


def test_pure_calendar_confounding_is_not_a_precise_zero_or_causal_effect():
    fit=_fit(sample(seasonal_only=True))
    heat=next(f for f in fit['factors'] if f['key']=='heat_c')
    assert heat['status']=='confounded_with_time_or_effort'
    assert heat['seconds_per_mile'] is None
    assert heat['range_95_seconds_per_mile'] is None


def test_negative_effect_not_clipped_to_expected_direction():
    result=analyze_performance_factors(sample(coefficient=-.012))
    heat=next(f for f in result['primary']['factors'] if f['key']=='heat_c')
    assert heat['seconds_per_mile']<0
    assert heat['range_95_seconds_per_mile'][1]<0


def test_reproducible_order_independent_and_no_input_mutation():
    rows=sample();before=deepcopy(rows)
    result=analyze_performance_factors(rows)
    assert result==analyze_performance_factors(list(reversed(rows)))
    assert rows==before


def test_invalid_core_is_disclosed_optional_feedback_not_required():
    rows=sample();rows[0]['sampled_hr_bpm']=None;rows[1]['start_time_utc']='broken'
    valid,skipped=_prepare(rows)
    assert len(valid)==158
    assert skipped=={'missing_pace_hr_or_position':1,'invalid_time':1}
    result=analyze_performance_factors(valid)
    assert all(f['runs']==0 for f in result['feedback'])
    assert analyze_performance_factors([])['status']=='more_observations_needed'


def test_duplicate_activity_cannot_multiply_evidence():
    rows=sample()
    with pytest.raises(ValueError,match='only once'):analyze_performance_factors(rows+[rows[0]])


def test_design_uses_training_scales_not_future_weather_or_device():
    train=sample();matrix,spec=_design(train)
    first=dict(train[0],heat_c=1000,device='new watch')
    transformed,same=_design([first],spec)
    index=next(i for i,f in enumerate(spec) if f['key']=='heat_c' and f['kind']=='numeric')
    assert same is spec
    assert transformed[0,index]==pytest.approx((1000-spec[index]['center'])/spec[index]['scale'])
    assert not any(f.get('level')=='new watch' for f in same)


def test_feedback_is_retained_but_cannot_explain_away_the_completed_performance():
    rows=sample();base=analyze_performance_factors(rows)
    for i,r in enumerate(rows):
        r['perceived_exertion']=i%5+2
        r['workout_feel']=50
    result=analyze_performance_factors(rows)
    assert result['primary']==base['primary']
    assert result['feedback'][0]['runs']==160


def test_one_calendar_block_can_show_an_estimate_but_not_false_block_uncertainty():
    rows=sample()[:10]
    origin=datetime(2024,1,4,tzinfo=timezone.utc)
    for i,row in enumerate(rows):row['start_time_utc']=(origin+timedelta(hours=i*12)).isoformat()
    result=_fit(rows)
    assert result['status']=='estimated_associations'
    assert result['calendar_blocks']==1
    assert all(f['range_95_seconds_per_mile'] is None for f in result['factors'])


def test_categorical_context_includes_quality_and_recorded_health_not_only_easy():
    rows=sample()
    for i,r in enumerate(rows):
        r['workout_type']='intervals' if i%4==0 else 'easy'
        r['health_tag']='illness_recovery' if i%10==0 else 'unreported'
    result=analyze_performance_factors(rows)
    assert result['observations']==160
    assert any(f.get('level')=='intervals' for f in result['primary']['factors'])
    assert any(f.get('level')=='illness_recovery' for f in result['primary']['factors'])


def test_reader_retains_short_quality_runs_and_partial_terrain_without_mutating_settings(monkeypatch):
    import sqlite3
    import run_analysis.performance_factors as module
    from run_analysis.config import load_config
    from run_analysis.db import initialize
    from test_progress import _insert_run
    config=load_config('config.yaml');before=deepcopy(config)
    db=sqlite3.connect(':memory:');db.row_factory=sqlite3.Row;initialize(db)
    identity=_insert_run(db,1,datetime(2026,10,1,tzinfo=timezone.utc),2,12,100)
    def loader(connection,settings,**kwargs):
        assert settings['model']['maximum_reliable_segment_minutes']>60
        return [dict(activity_id=identity,start_time_utc='2026-10-01T00:00:00+00:00',
            moving_pace_min_mile=10,average_hr_bpm=160,moving_minutes_into_run=8,
            heart_rate_change_bpm=1,speed_change_fraction=.01,grade_energy_ratio=1.02,
            average_grade_percent=1,workout_type='intervals',previous_7d_miles=10,
            previous_28d_miles=40,days_since_previous_run=1,days_since_previous_hard_run=5),
            dict(activity_id=identity,start_time_utc='2026-10-01T00:00:00+00:00',
            moving_pace_min_mile=10,average_hr_bpm=160,moving_minutes_into_run=9,
            grade_energy_ratio=None,average_grade_percent=None,workout_type='intervals',
            previous_7d_miles=10,previous_28d_miles=40,days_since_previous_run=1,
            days_since_previous_hard_run=5)],{}
    monkeypatch.setattr(module,'load_overlapping_model_windows',loader)
    changes=db.total_changes
    rows,coverage=module.load_factor_observations(db,config)
    assert db.total_changes==changes
    assert config==before
    assert len(rows)==1 and rows[0]['workout_type']=='intervals'
    assert rows[0]['grade_percent']==1
    assert rows[0]['grade_window_coverage']==.5
    assert rows[0]['heat_c'] is None
    assert coverage['usable_runs']==1
