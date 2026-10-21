from datetime import date, datetime, timedelta, timezone
from math import exp
from types import SimpleNamespace

import pytest

from run_analysis.db import connect, initialize
from run_analysis.progress_training_load import ewma, recorded_cardio, training_load_history
from run_analysis.response_model import rate
from run_analysis.continuous_training_dose import ContinuousDoseReference
from test_progress import _insert_run
from test_workload_profile import CONFIG


def row(day, low=30., high=None):
    return dict(activity_id=day.toordinal(), day=day, start_time=f'{day}T12:00:00+00:00',
        low=low, high=low if high is None else high, covered_seconds=1200., elapsed_seconds=1200.)


def history():
    end = datetime(2026, 10, 10, 20, tzinfo=timezone.utc)
    return end, [row(end.date()-timedelta(days=i)) for i in range(600)]


@pytest.mark.parametrize('tau', [2, 7, 42])
def test_unit_gain_and_rest_day_decay(tau):
    end, rows = history()
    before = ewma(rows, end.date(), tau)
    assert before['value'] == pytest.approx(30., abs=.0001)
    after = ewma(rows, end.date()+timedelta(days=1), tau)
    assert after['value'] == pytest.approx(before['value']*exp(-1/tau))


def test_not_a_hard_window_and_missing_hr_bounds_not_zero():
    end, rows = history()
    old = row(end.date()-timedelta(days=30), 100., 200.)
    base = ewma(rows, end.date(), 7)
    result = ewma(rows+[old], end.date(), 7)
    assert result['low'] > base['low']
    assert result['high'] > result['value'] > result['low']
    assert not ewma([row(end.date()-timedelta(days=1))], end.date(), 42)['complete']
    assert not ewma(rows+[row(end.date(), None)], end.date(), 42)['complete']


def test_window_invariance_future_exclusion_and_pre_run_local_date():
    end, rows = history()
    at = datetime(2026, 10, 10, 2, tzinfo=timezone.utc)  # October 9 in New York
    s = SimpleNamespace(activity_id=1, start_time=at)
    p = SimpleNamespace(included_in_trend=True, standardized_pace_min_mile=10., uncertainty_95_min_mile=.4)
    points, details = {1:p}, {1:{'health_tag':'normal'}}
    extra = row(date(2026,10,9), 900.)
    args = (rows+[extra], [s], points, details, end)
    short = training_load_history(*args, 14, 'America/New_York')
    long = training_load_history(*args, 180, 'America/New_York')
    assert short['current'] == long['current']
    assert short['ratio'] == long['ratio']
    assert short['runs'] == long['runs']
    assert short['runs'][0]['load'] == ewma(rows, date(2026,10,8),7)['value']
    future = training_load_history(rows+[extra,row(date(2026,10,11),1000.)], [s], points, details, end,14,'America/New_York')
    assert future['current'] == short['current']
    details[1]['health_tag']='illness'
    assert not training_load_history(*args,14,'America/New_York')['runs']


def test_ratio_bounds_and_no_baseline_verdict_for_short_history():
    end, rows = history()
    result = training_load_history(rows+[row(end.date(),60.,100.)],[],{},{},end,28,'UTC')
    assert result['ratio_bounds']['low'] < result['ratio'] < result['ratio_bounds']['high']
    short = training_load_history(rows[:50],[],{},{},end,28,'UTC')
    assert short['current']['recent']['complete']
    assert not short['current']['baseline']['complete']
    assert short['ratio'] is None


def test_reader_shared_trace_units_readonly_cache_invalidates(tmp_path, monkeypatch):
    import run_analysis.progress_training_load as module
    config = {**CONFIG, 'resting_hr':46, 'max_hr':194, 'timezone_default':'UTC'}
    path = tmp_path/'history.sqlite'
    at = datetime(2026,10,9,12,tzinfo=timezone.utc)
    with connect(path) as db:
        initialize(db)
        id = _insert_run(db,1,at,1.,20.,20.)
        db.execute('UPDATE activities SET total_elapsed_time_s=20,trackpoint_count=21 WHERE id=?',(id,))
        for i in range(21):
            db.execute('''INSERT INTO trackpoints(activity_id,lap_index,track_index,point_index,
                timestamp_utc,gps_valid,distance_m,heart_rate_bpm,parse_flags_json)
                VALUES (?,0,0,?,?,0,?,150,'[]')''',(id,i,(at+timedelta(seconds=i)).isoformat(),i*3.))
        db.execute("INSERT INTO source_files(id,path,display_path,size_bytes,mtime_ns,sha256,parse_status,imported_at_utc) VALUES(1,'run.fit','run.fit',1,1,'source-one','ok','now')")
        db.execute('INSERT INTO activity_sources VALUES(?,1,0,1,NULL)',(id,))
        db.commit()
    calls=[]
    real=module._load_points
    monkeypatch.setattr(module,'_load_points',lambda db,id: (calls.append(id) or real(db,id)))
    with connect(path) as db:
        n=db.total_changes
        result=recorded_cardio(db,config,at+timedelta(days=1))
        assert result[0]['low'] == pytest.approx(20/60*rate((150-46)/(194-46),ContinuousDoseReference('test').policy))
        assert result[0]['low'] == result[0]['high']
        assert recorded_cardio(db,config,at+timedelta(days=1)) == result
        assert len(calls)==1
        assert db.total_changes==n
        # Simulate a server restart: derived results survive without loading HR
        # samples again, and preserve the full output exactly.
        module._TRACE_CACHE.clear()
        assert recorded_cardio(db,config,at+timedelta(days=1))==result
        assert len(calls)==1
        from run_analysis.derived_load_cache import cache_path
        import sqlite3
        with sqlite3.connect(cache_path(path)) as cache:
            cache.execute("UPDATE traces SET digest='corrupt'")
        module._TRACE_CACHE.clear()
        assert recorded_cardio(db,config,at+timedelta(days=1))==result
        assert len(calls)==2
        # A request midway through this recorded activity cannot use its ending.
        assert recorded_cardio(db,config,at+timedelta(seconds=10)) == []
        recorded_cardio(db,{**config,'max_hr':200},at+timedelta(days=1))
        assert len(calls)==3
        db.execute("UPDATE activities SET updated_at_utc='changed' WHERE id=?",(id,))
        recorded_cardio(db,config,at+timedelta(days=1))
        assert len(calls)==4
        db.rollback()
        db.execute("UPDATE source_files SET sha256='source-two' WHERE id=1");db.commit()
        recorded_cardio(db,config,at+timedelta(days=1))
        assert len(calls)==5
