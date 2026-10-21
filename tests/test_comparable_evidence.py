from copy import deepcopy
from datetime import datetime, timedelta, timezone
import math
import sqlite3

import numpy as np
import pytest

from run_analysis.comparable_evidence import build_comparable_evidence, comparable_progress_data
from run_analysis.pace_comparability import compare_run_windows, fit_comparison_calibration
from run_analysis.pace_temperature import fit_temperature, hr_shift, comparison_covariance
from scripts.thermal_hr_trial import fit_weather, hr_shift as trial_shift
from tests.test_pace_comparability import config, row
from tests.test_thermal_hr_trial import synthetic


def temperature_calibration(config, scale=1.):
    cal=fit_comparison_calibration([],config)
    cal['heat'].update(mean=0.,sd=0.)
    cal['temperature_hr']=dict(fit_temperature([]),scale=scale)
    return cal


def test_production_fit_reproduces_reviewed_air_trial_and_sparse_prior():
    rows=synthetic(scale=1.6)
    candidate=fit_temperature(rows); previous=fit_weather(rows,model='air')
    assert candidate['scale']==pytest.approx(previous['scale'],abs=1e-12)
    assert candidate['conditional_scale_sd']==pytest.approx(previous['conditional_scale_sd'],abs=1e-12)
    assert fit_temperature([])['scale']==1.
    assert fit_temperature(rows[:1])['scale']==pytest.approx(1.)
    assert not candidate['humidity_fitted']


def test_temperature_matches_reviewed_shape_and_preserves_weather_attribution(config):
    for temp in (46.,55.,85.):
        r=row(config,temperature_f=temp,average_hr_bpm=150.)
        assert hr_shift(r,config)==pytest.approx(trial_shift(r,config,exposure='air'))
        cal=temperature_calibration(config)
        candidate=compare_run_windows([r],[1.],config,cal)
        shifted=dict(r,average_hr_bpm=r['average_hr_bpm']-hr_shift(r,config))
        neutral=deepcopy(cal);neutral.pop('temperature_hr')
        expected=compare_run_windows([shifted],[1.],config,neutral)
        assert candidate['pace_min_mile']==pytest.approx(expected['pace_min_mile'])
        assert sum(candidate['contributions_seconds_per_mile'].values())==pytest.approx(
            60*(candidate['pace_min_mile']-candidate['observed_pace_min_mile']))
        assert candidate['temperature_response']['applied_windows']==1
        # Altering unused legacy heat cannot charge the temperature cost twice.
        cal['heat']['mean']=.1
        assert compare_run_windows([r],[1.],config,cal)['pace_min_mile']==candidate['pace_min_mile']


def test_no_separate_humidity_credit_and_missing_temperature_is_retained(config):
    cal=temperature_calibration(config)
    dry=row(config,temperature_f=80.,relative_humidity_percent=20.,dewpoint_f=25.)
    wet=dict(dry,relative_humidity_percent=95.,dewpoint_f=78.)
    assert compare_run_windows([dry],[1.],config,cal)['pace_min_mile']==compare_run_windows([wet],[1.],config,cal)['pace_min_mile']
    missing=dict(dry,temperature_f=None)
    result=compare_run_windows([missing],[1.],config,cal)
    neutral=deepcopy(cal);neutral['temperature_hr']['scale']=0.
    assert result['pace_min_mile']==pytest.approx(compare_run_windows([dry],[1.],config,neutral)['pace_min_mile'])
    assert result['weather_coverage']==0
    assert result['temperature_response']['applied_windows']==0


def test_temperature_gradient_and_real_speed_gain_are_preserved(config):
    cal=temperature_calibration(config,2.1);r=row(config,temperature_f=85.,average_hr_bpm=160.)
    a=compare_run_windows([r],[1.],config,cal)
    for i in range(4):
        changed=deepcopy(cal);eps=1e-6
        if i<3:changed['parameters'][i]+=eps
        else:changed['temperature_hr']['scale']+=eps
        b=compare_run_windows([r],[1.],config,changed)
        numerical=-math.log(b['pace_min_mile']/a['pace_min_mile'])/eps
        assert a['shared_log_speed_gradient'][i]==pytest.approx(numerical,abs=1e-6)
    better=compare_run_windows([dict(r,moving_pace_min_mile=r['moving_pace_min_mile']/1.02)],[1.],config,cal)
    assert a['pace_min_mile']/better['pace_min_mile']==pytest.approx(1.02)


def test_personalization_scales_the_aggregated_run_burden_not_window_modes(config):
    windows=[row(config,moving_pace_min_mile=pace,moving_minutes_into_run=minute,
        temperature_f=temperature,average_hr_bpm=hr) for pace,minute,temperature,hr in
        ((8.,5.,45.,140.),(12.,15.,60.,160.),(10.,30.,80.,150.),(9.,40.,80.,145.))]
    population=temperature_calibration(config);neutral=deepcopy(population);neutral.pop('temperature_hr')
    personal=temperature_calibration(config,2.18)
    p=compare_run_windows(windows,[1.]*4,config,population)['pace_min_mile']
    n=compare_run_windows(windows,[1.]*4,config,neutral)['pace_min_mile']
    result=compare_run_windows(windows,[1.]*4,config,personal,precision_seed=42)
    assert result['pace_min_mile']==pytest.approx(n*math.exp(-2.18*math.log(n/p)),abs=1e-12)
    assert result['shared_log_speed_gradient'][3]==pytest.approx(math.log(n/p))
    assert result['comparison_precision']


def test_shared_reader_retains_all_evidence_and_recomputes_precision(config,monkeypatch):
    from run_analysis.db import initialize
    from test_progress import _insert_run
    from run_analysis.progress import prepare_progress_data, build_progress
    from run_analysis.aerobic_performance import _fit
    db=sqlite3.connect(':memory:');db.row_factory=sqlite3.Row;initialize(db)
    origin=datetime(2026,10,1,tzinfo=timezone.utc);basecal=fit_comparison_calibration([],config)
    contexts={};runs=[]
    for i in range(4):
        at=origin+timedelta(days=i*3)
        identity=_insert_run(db,i+1,at,4,40,100)
        windows=[row(config,activity_id=identity,temperature_f=50.+i*8,moving_minutes_into_run=minute)
            for minute in (10.,20.,30.)]
        context=dict(coverage=dict(moving_seconds=2400.,stopped_seconds=0.))
        contexts[identity]=(windows,[1.,1.,1.],context)
        value=compare_run_windows(windows,[1.,1.,1.],config,basecal,precision_seed=5+i)
        runs.append(dict(activity_id=identity,start_time=at.isoformat(),distance_miles=4.,health_tag='normal',workout_type='easy',
            included_in_trend=True,trend_weight=1.,**value))
    base=dict(calibration=basecal,runs=runs)
    before=deepcopy(base)
    def aligned(connection,report,cfg,*,window_collector,include_precision=True):
        for identity,(windows,weights,context) in contexts.items():
            window_collector(identity,windows,weights,context)
        return deepcopy(report)
    monkeypatch.setattr('run_analysis.comparable_evidence.build_hr_aligned_comparability',aligned)
    changes=db.total_changes;result=build_comparable_evidence(db,base,config)
    assert db.total_changes==changes and base==before
    assert [r['activity_id'] for r in result['runs']]==[r['activity_id'] for r in runs]
    assert comparison_covariance(result['calibration']).shape==(5,5)
    assert all(len(r['shared_log_speed_gradient'])==5 and r['comparison_precision'] for r in result['runs'])
    assert all(r['temperature_response']['original_windows']==3 for r in result['runs'])
    assert _fit(result)['available']
    from run_analysis.aerobic_progress import progress_from_comparability as older_progress
    assert older_progress(result,as_of=origin+timedelta(days=10))['available']
    import json
    json.dumps(result,allow_nan=False)  # Native API-safe scalars, not numpy bool/int.
    raw=prepare_progress_data(db);adapted=comparable_progress_data(raw,result)
    assert adapted.sessions==raw.sessions and adapted.details==raw.details
    assert len(adapted.analytics_rows)==4
    progress=build_progress(db,28,as_of=origin+timedelta(days=10),config=config,prepared=raw,comparison_report=result)
    assert all(p.pace_basis=='shared-aerobic-comparison-v1' for p in progress.series)
    assert len(progress.activity_coverage)==4
    assert all(p.score_status=='trend_evidence' for p in progress.activity_coverage)


def test_missing_response_retains_original_and_separate_legacy_uncertainty(config,monkeypatch):
    from run_analysis.db import initialize
    db=sqlite3.connect(':memory:');db.row_factory=sqlite3.Row;initialize(db)
    cal=fit_comparison_calibration([],config)
    value=compare_run_windows([row(config,temperature_f=80.)],[1.],config,cal,precision_seed=2)
    original=dict(activity_id=1,included_in_trend=True,trend_weight=1.,**value)
    base=dict(runs=[original],calibration=cal)
    monkeypatch.setattr('run_analysis.comparable_evidence.build_hr_aligned_comparability',lambda *args,**kwargs:deepcopy(base))
    result=build_comparable_evidence(db,base,config);r=result['runs'][0]
    assert r['pace_min_mile']==original['pace_min_mile']
    assert r['shared_log_speed_gradient'][3]==0.
    assert r['shared_log_speed_gradient'][4]==original['shared_log_speed_gradient'][3]
    before=np.array(original['shared_log_speed_gradient']);after=np.array(r['shared_log_speed_gradient'])
    assert after@comparison_covariance(result['calibration'])@after==pytest.approx(before@comparison_covariance(cal)@before)


def test_ui_consumers_share_scores_without_refreshing_model_or_planner(tmp_path,monkeypatch,config):
    from fastapi.testclient import TestClient
    from run_analysis.config import load_config
    from run_analysis.db import connect,initialize
    from run_analysis.web.app import create_app
    from test_web_phase1 import _write_config
    from test_progress import _insert_run
    defaults=config['model']
    _write_config(tmp_path);config=load_config(tmp_path/'config.yaml')
    config['model']={**defaults,**config['model']}
    path=tmp_path/'data/test.sqlite';path.parent.mkdir()
    end=datetime.now(timezone.utc);cal=temperature_calibration(config)
    cal['comparison_legacy_heat_sd']=.002
    runs=[]
    with connect(path) as db:
        initialize(db)
        for i in range(4):
            at=end-timedelta(days=10-i*2);identity=_insert_run(db,i+1,at,4,40,100)
            value=compare_run_windows([row(config,moving_pace_min_mile=10-i*.1)],[1.],config,cal,precision_seed=i)
            value['shared_log_speed_gradient'].append(0.)
            runs.append(dict(activity_id=identity,start_time=at.isoformat(),distance_miles=4.,health_tag='normal',
                workout_type='easy',included_in_trend=True,trend_weight=1.,**value))
        db.commit()
    report=dict(version='shared-aerobic-comparison-v1',calibration=cal,runs=runs,planner_changed=False)
    monkeypatch.setattr('run_analysis.comparable_evidence.cached_comparable_evidence',lambda *args:deepcopy(report))
    def forbidden(*args,**kwargs):pytest.fail('Display invoked model fit or plan refresh')
    for name in ('fit_models','generate_weekly_schedule','ensure_current_weekly_schedule'):
        monkeypatch.setattr('run_analysis.web.app.'+name,forbidden)
    with TestClient(create_app(tmp_path)) as client:
        progress=client.get('/api/progress?window_days=28').json()
        wip=client.get('/api/progress-wip?window_days=28&include_exploratory=false').json()
        dashboard=client.get('/api/dashboard?fitness_window_days=28').json()
        listed=client.get('/api/runs?sort_by=standardized&sort_order=asc').json()
        detail=client.get('/api/runs/'+str(runs[0]['activity_id'])).json()
    expected={r['activity_id']:r['pace_min_mile'] for r in runs}
    assert {r['activity_id']:r['standardized_pace_min_mile'] for r in progress['series']}==expected
    assert wip['baseline']['series']==progress['series']
    assert {r['activity_id']:r['standardized_pace_min_mile'] for r in dashboard['progress']['series']}==expected
    assert [r['activity_id'] for r in listed]==[r['activity_id'] for r in reversed(runs)]
    assert all(r['fitness_observation']['standardized_pace_at_target_hr']['minutes_per_mile']==expected[r['activity_id']] for r in listed)
    assert detail['run']['fitness_observation']['standardized_pace_at_target_hr']['minutes_per_mile']==expected[runs[0]['activity_id']]
    assert len(wip['aerobic_performance']['observations'])==4
