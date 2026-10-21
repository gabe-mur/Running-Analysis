from copy import deepcopy
from datetime import datetime, timedelta, timezone
import math
from pathlib import Path

import numpy as np
import pytest

from run_analysis.config import load_config
from run_analysis.pace_comparability import fit_comparison_calibration,compare_run_windows,comparison_view,_features
from run_analysis.physiology import grade_energy_ratio


@pytest.fixture
def config():
    return load_config(Path(__file__).parents[1]/'config.example.yaml')


def row(config,**kwargs):
    result=dict(activity_id=1,start_time_utc='2026-10-01T00:00:00+00:00',moving_pace_min_mile=10.,
        average_hr_bpm=config['target_hr'],moving_minutes_into_run=config['reference_conditions']['within_run_minutes'],
        grade_energy_ratio=1.,grade_complete_fraction=1.,headwind_signed_mph=0.,crosswind_mph=0.,
        temperature_f=config['reference_conditions']['temperature_f'],dewpoint_f=config['reference_conditions']['dewpoint_f'],
        workout_type='easy',health_tag='normal',heart_rate_change_bpm=0.,speed_change_fraction=0.)
    result.update(kwargs)
    return result


def test_reference_conditions_identity_and_no_minimum_history_gate(config):
    calibration=fit_comparison_calibration([],config)
    result=compare_run_windows([row(config)],[1.],config,calibration)
    assert result['pace_min_mile']==pytest.approx(10.)
    assert all(abs(value)<1e-8 for value in result['contributions_seconds_per_mile'].values())
    assert calibration['contributing_runs']==0
    assert len(result['shared_log_speed_gradient'])==4
    assert all(abs(value)<1e-8 for value in result['shared_log_speed_gradient'])


def test_progress_prototype_reuses_calibration_and_does_not_mutate_pace_report(config):
    from run_analysis.aerobic_progress import progress_from_comparability
    calibration=fit_comparison_calibration([],config)
    at=datetime(2026,10,13,tzinfo=timezone.utc)
    runs=[]
    for identity,days in enumerate(range(60,-1,-4)):
        run=compare_run_windows([row(config,moving_pace_min_mile=10 if days>28 else 9.5)],[1.],config,calibration)
        runs.append(dict(activity_id=identity,start_time=(at-timedelta(days=days)).isoformat(),
            distance_miles=4,included_in_trend=True,trend_weight=1.,workout_type='easy',**run))
    report=dict(calibration=calibration,runs=runs);before=deepcopy(report)
    short=progress_from_comparability(report,as_of=at,history_days=14)
    long=progress_from_comparability(report,as_of=at,history_days=365)
    assert short['available']
    assert short['current']==long['current'] and short['change']==long['change']
    assert short['change']['speed_change_percent']>0
    assert all(point['change'] is not None for point in short['history'])
    assert report==before


@pytest.mark.parametrize('grade',[-8.,-3.,0.,3.,8.])
def test_incline_and_decline_convert_same_energy_rate_to_same_pace(config,grade):
    calibration=fit_comparison_calibration([],config)
    ratio=grade_energy_ratio(grade)
    r=row(config,grade_energy_ratio=ratio,moving_pace_min_mile=10.*ratio)
    result=compare_run_windows([r],[1.],config,calibration)
    assert result['pace_min_mile']==pytest.approx(10.)
    if grade>0:assert result['contributions_seconds_per_mile']['terrain']<0
    else:assert result['contributions_seconds_per_mile']['terrain']>=-1e-8


def test_out_and_back_hills_do_not_cancel_before_cost_calculation(config):
    calibration=fit_comparison_calibration([],config)
    ratios=[grade_energy_ratio(v) for v in [-5.,5.]]
    assert np.mean(ratios)>1
    result=compare_run_windows([row(config,grade_energy_ratio=float(np.mean(ratios)))],[1.],config,calibration)
    assert result['pace_min_mile']<10.


def test_hr_normalization_treats_lower_and_higher_effort_symmetrically(config):
    calibration=fit_comparison_calibration([],config)
    for hr in [config['target_hr']-5,config['target_hr']+5]:
        reserve=(hr-config['resting_hr'])/(config['target_hr']-config['resting_hr'])
        r=row(config,average_hr_bpm=hr,moving_pace_min_mile=10./reserve)
        assert compare_run_windows([r],[1.],config,calibration)['pace_min_mile']==pytest.approx(10.)


def test_missing_context_retains_estimate_and_widens_range(config):
    calibration=fit_comparison_calibration([],config)
    complete=compare_run_windows([row(config)],[1.],config,calibration)
    missing=row(config,grade_energy_ratio=None,temperature_f=None,headwind_signed_mph=None,crosswind_mph=None)
    result=compare_run_windows([missing],[1.],config,calibration)
    assert result['pace_min_mile']==pytest.approx(10.)
    assert result['terrain_coverage']==result['wind_coverage']==result['weather_coverage']==0
    assert result['uncertainty_95_min_mile']>complete['uncertainty_95_min_mile']


def test_rejected_terrain_cannot_earn_credit_even_with_a_supplied_grade(config):
    calibration=fit_comparison_calibration([],config)
    bad=row(config,grade_energy_ratio=1.5,terrain_quality={'usable':False,'status':'rejected'})
    estimate=compare_run_windows([bad],[1.],config,calibration)
    assert estimate['pace_min_mile']==pytest.approx(10.)
    assert estimate['contributions_seconds_per_mile']['terrain']==pytest.approx(0.)
    assert estimate['terrain_coverage']==0
    assert estimate['uncertainty_95_min_mile']>0
    assert fit_comparison_calibration([bad],config)['contributing_runs']==0


def test_common_uncertainty_does_not_disappear_with_duplicate_windows(config):
    calibration=fit_comparison_calibration([],config)
    r=row(config,average_hr_bpm=config['target_hr']+4)
    first=compare_run_windows([r],[1.],config,calibration)
    repeated=compare_run_windows([r]*100,[1.]*100,config,calibration)
    assert first['uncertainty_95_min_mile']==pytest.approx(repeated['uncertainty_95_min_mile'])


def test_wind_humidity_directions_and_additive_attribution(config):
    calibration=fit_comparison_calibration([],config)
    head=compare_run_windows([row(config,headwind_signed_mph=8.)],[1.],config,calibration)
    tail=compare_run_windows([row(config,headwind_signed_mph=-4.)],[1.],config,calibration)
    hot=compare_run_windows([row(config,temperature_f=85.,dewpoint_f=75.)],[1.],config,calibration)
    assert head['pace_min_mile']<10.<tail['pace_min_mile']
    assert hot['pace_min_mile']<10.
    assert sum(hot['contributions_seconds_per_mile'].values())==pytest.approx((hot['pace_min_mile']-10.)*60)


def test_joint_calibration_recovers_hr_and_drift_without_learning_fitness_trend(config):
    rng=np.random.default_rng(18);rows=[]
    for identity in range(50):
        for minute,hr in zip([10,15,20,25,30,35,40,45],rng.uniform(config['target_hr']-8,config['target_hr']+8,8)):
            reserve=(hr-config['resting_hr'])/(config['target_hr']-config['resting_hr'])
            speed=(2.4+identity*.01)*reserve**.8*math.exp(-.012*(minute-20)/10)
            rows.append(row(config,activity_id=identity,average_hr_bpm=hr,moving_minutes_into_run=minute,
                moving_pace_min_mile=1609.344/(60*speed)))
    prior=deepcopy(config);calibration=fit_comparison_calibration(rows,config)
    assert config==prior
    assert calibration['parameters'][0]==pytest.approx(.8,abs=.03)
    assert calibration['parameters'][1]==pytest.approx(-.012,abs=.003)
    assert calibration['parameters'][2]==pytest.approx(.5)


def test_quality_or_abnormal_health_cannot_recalibrate_aerobic_relationship(config):
    calibration=fit_comparison_calibration([row(config,workout_type='intervals'),row(config,health_tag='illness')],config)
    assert calibration['contributing_runs']==0


def test_invalid_reference_or_weights_fail_explicitly(config):
    calibration=fit_comparison_calibration([],config)
    with pytest.raises(ValueError):compare_run_windows([row(config)],[0],config,calibration)
    with pytest.raises(ValueError):_features(row(config,average_hr_bpm=config['resting_hr']),config)
    with pytest.raises(ValueError):_features(row(config,moving_pace_min_mile=float('nan')),config)


def test_chronological_checks_keep_both_tasks_and_recover_known_comparability(config):
    from run_analysis.pace_comparability import validate_comparability
    rng=np.random.default_rng(31);rows=[]
    origin=datetime(2026,1,1,tzinfo=timezone.utc)
    for identity in range(24):
        for minute,hr in zip([10,15,20,25,30,35,40,45],rng.uniform(config['target_hr']-8,config['target_hr']+8,8)):
            reserve=(hr-config['resting_hr'])/(config['target_hr']-config['resting_hr'])
            rows.append(row(config,activity_id=identity,start_time_utc=(origin+timedelta(days=identity*2)).isoformat(),
                average_hr_bpm=hr,moving_minutes_into_run=minute,moving_pace_min_mile=10/reserve))
    result=validate_comparability(rows,config)
    assert result['held_out_runs']==12
    assert result['mean_absolute_error_seconds_per_mile']['weather']<1e-8
    between=result['between_run_check']
    assert between['pairs']==12
    assert between['mean_absolute_disagreement_seconds_per_mile']['weather']<1e-8
    assert all(f['last_training_start']<f['first_test_start'] for f in result['folds'])


def test_view_does_not_leak_future_runs_or_mix_raw_and_adjusted_run_sets(config):
    calibration=fit_comparison_calibration([],config)
    at=datetime(2026,10,13,tzinfo=timezone.utc);runs=[]
    for identity,days in enumerate([10,5,1,-1]):
        result=compare_run_windows([row(config)],[1.],config,calibration)
        runs.append(dict(activity_id=identity,start_time=(at-timedelta(days=days)).isoformat(),
            distance_miles=4,trend_weight=1.,included_in_trend=True,workout_type='easy',**result))
    result=comparison_view(dict(runs=runs),28,at)
    assert len(result['runs'])==len(result['series'])==3
    assert [v['run_count'] for v in result['trend']]==[v['run_count'] for v in result['recorded_trend']]
    assert result['performance']['current']['run_count']==3
    assert result['performance']['current']['pace_min_mile']==pytest.approx(10.)
    assert result['performance']['last_observation']==runs[2]['start_time']


def test_pace_display_retains_previous_trend_point_but_only_visible_run_marks(config):
    calibration=fit_comparison_calibration([],config)
    at=datetime(2026,10,13,tzinfo=timezone.utc)
    runs=[]
    for identity,days in enumerate([40,30,20,10,2]):
        estimate=compare_run_windows([row(config,moving_pace_min_mile=9+days/100)],[1.],config,calibration)
        runs.append(dict(activity_id=identity,start_time=(at-timedelta(days=days)).isoformat(),
            distance_miles=4,trend_weight=1.,included_in_trend=True,workout_type='easy',**estimate))
    short=comparison_view(dict(runs=runs),14,at)
    full=comparison_view(dict(runs=runs),90,at)
    assert [r['activity_id'] for r in short['series']]==[3,4]
    for key in ('trend','recorded_trend'):
        assert datetime.fromisoformat(short[key][0]['as_of'])==datetime.fromisoformat(runs[2]['start_time'])
        full_by_date={r['as_of']:r for r in full[key]}
        assert all(r==full_by_date[r['as_of']] for r in short[key])


def test_progress_and_comparison_share_read_only_cache_without_changing_vo2_or_planner(tmp_path,monkeypatch):
    import sqlite3
    from hashlib import sha256
    from fastapi.testclient import TestClient
    from run_analysis.db import connect,initialize
    from run_analysis.web.app import create_app
    import run_analysis.pace_comparability as module
    import run_analysis.web.app as app_module
    from test_web_phase1 import _write_config
    _write_config(tmp_path)
    path=tmp_path/'data/test.sqlite';path.parent.mkdir()
    with connect(path) as db:initialize(db)
    db.close()
    original=module.build_pace_comparability;calls=[]
    def wrapped(connection,config):
        calls.append(1);before=connection.total_changes
        result=original(connection,config)
        assert connection.total_changes==before
        return result
    monkeypatch.setattr(module,'build_pace_comparability',wrapped)
    original_progress=app_module.build_progress_wip;canonical=[]
    def capture_progress(*args,**kwargs):
        result=original_progress(*args,**kwargs)
        canonical.append(deepcopy(result))
        return result
    monkeypatch.setattr(app_module,'build_progress_wip',capture_progress)
    for name in ('generate_weekly_schedule','ensure_current_weekly_schedule','fit_models'):
        monkeypatch.setattr('run_analysis.web.app.'+name,lambda *args,**kwargs:pytest.fail('Preview invoked live model/planner'))
    # Pin WAL lifecycle, as the running app does. Closing the last SQLite
    # connection legitimately invalidates the file-version cache, even with no writes.
    keeper=sqlite3.connect(path);keeper.execute('SELECT count(*) FROM activities').fetchone()
    digest=sha256(path.read_bytes()).hexdigest()
    try:
        with TestClient(create_app(tmp_path)) as client:
            for days in (28,90):
                response=client.get(f'/api/pace-comparability?window_days={days}')
                assert response.status_code==200
                assert response.json()['series']==[]
                assert not response.json()['planner_changed']
                response=client.get(f'/api/progress-wip?window_days={days}&include_exploratory=false')
                assert response.status_code==200
                result=response.json()
                assert result['pace_comparison']['series']==[]
                assert result['pace_comparison']['performance']['current'] is None
                assert result['pace_comparison']['as_of']==result['as_of']
                # No substitution into the underlying scores, VO2 or load outputs.
                assert result['aerobic_performance']['version']=='direct-aerobic-performance-v9-headline-amount'
                assert result['aerobic_performance']['available'] is False
                assert {k:v for k,v in result.items() if k not in {'pace_comparison','aerobic_progress','aerobic_performance'}}==canonical[-1]
                assert result['aerobic_progress']['available'] is False
    finally:keeper.close()
    assert calls==[1]
    assert sha256(path.read_bytes()).hexdigest()==digest


def test_reader_scores_one_short_run_outside_z2_without_old_model_score(config,monkeypatch):
    import sqlite3
    import run_analysis.pace_comparability as module
    from run_analysis.db import initialize
    from test_progress import _insert_run
    db=sqlite3.connect(':memory:');db.row_factory=sqlite3.Row;initialize(db)
    identity=_insert_run(db,1,datetime(2026,10,1,tzinfo=timezone.utc),2,12,100)
    window=row(config,activity_id=identity,average_hr_bpm=160,moving_minutes_into_run=8)
    monkeypatch.setattr(module,'load_model_window_sets',lambda connection,cfg,sizes,**kwargs:({size:[window] for size in sizes},{}))
    monkeypatch.setattr(module,'load_overlapping_model_windows',lambda *args,**kwargs:([window],{}))
    before=db.total_changes
    result=module.build_pace_comparability(db,config)
    assert len(result['runs'])==1
    assert result['runs'][0]['included_in_trend']
    assert result['runs'][0]['trend_weight']==1.
    assert result['runs'][0]['pace_min_mile']>10  # Higher observed HR -> slower equivalent at target HR.
    assert db.total_changes==before
