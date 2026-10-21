"""The Dashboard number, curves and strip must describe the same quantity."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import sqlite3
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from fastapi.testclient import TestClient
import pytest

from run_analysis.training_overview import (
    LABELS, build_training_overview, classify_load, smooth_load_history, load_status,
    recommended_load_history,
)
from run_analysis.web.app import create_app
from test_progress_wip import project, _score
from test_progress import _insert_run
from test_progress_training_load import history, row


END = datetime(2026, 10, 10, 20, tzinfo=timezone.utc)


def sample(value, spread=0):
    return dict(complete=True, value=value, low=value-spread, high=value+spread)


def load(days=14, ratio=1.2, spread=0):
    return dict(unit='HR-reserve equivalent minutes per day', daily=[
        dict(as_of=(END-timedelta(days=days-i)).isoformat(),
        recent=sample(30*ratio, spread), baseline=sample(30, spread))
        for i in range(days+1)])


@pytest.mark.parametrize('ratio,expected', [(1.2,'above'),(1.05,'near'),(1,'near'),(.95,'near'),(.8,'below')])
def test_labels_follow_plotted_values_and_inclusive_boundaries(ratio, expected):
    result=classify_load(sample(30*ratio), sample(30))
    assert result['status']==expected
    assert result['ratio']==pytest.approx(ratio)
    assert result['ratio_bounds']['low']==pytest.approx(ratio)
    assert load_status({**result,'as_of':END.isoformat()})['label']==LABELS[expected]


def test_missing_zero_and_ambiguous_bounds_are_not_a_verdict():
    missing=dict(complete=False, value=None, low=None, high=None)
    assert classify_load(missing,sample(30))==dict(status='unknown',ratio=None,ratio_bounds=None)
    assert classify_load(sample(30),sample(0))['ratio'] is None
    assert classify_load(sample(31.5,3),sample(30,3))['status']=='unknown'
    assert set(LABELS)=={'above','near','below','unknown'}


def test_smoothing_is_causal_not_cosmetic_or_mean_of_ratios():
    source=load()
    source['daily'][-1]['recent']=sample(80,4)
    source['daily'][-1]['baseline']=sample(40,2)
    before=deepcopy(source)
    result=smooth_load_history(source)
    current=result['current']
    assert current['recent']['value']==pytest.approx((6*36+80)/7)
    assert current['baseline']['value']==pytest.approx((6*30+40)/7)
    assert current['ratio']==current['recent']['value']/current['baseline']['value']
    assert current['ratio']!=pytest.approx((6*1.2+2)/7)
    assert current['ratio_bounds']['low']==current['recent']['low']/current['baseline']['high']
    assert source==before  # Raw cardio accounting is never replaced.
    future={**source['daily'][-1],'as_of':(END+timedelta(days=1)).isoformat(),'recent':sample(10000)}
    assert smooth_load_history({**source,'daily':source['daily']+[future]})['daily'][:-1]==result['daily']


def test_gap_and_sparse_history_remain_unavailable():
    source=load()
    assert all(not r['recent']['complete'] for r in smooth_load_history(source)['daily'][:6])
    source['daily'][-2]['recent']=dict(complete=False,value=None,low=None,high=None)
    assert smooth_load_history(source)['current']['status']=='unknown'
    assert not smooth_load_history(source)['current']['recent']['complete']
    source=load()
    del source['daily'][-3]
    assert smooth_load_history(source)['current']['status']=='unknown'
    assert smooth_load_history(load(days=5))['current']['ratio'] is None


def test_partial_today_uses_athletes_local_date_not_utc_date():
    zone=ZoneInfo('Pacific/Auckland')
    source=load(days=6)
    for i,r in enumerate(source['daily']):
        r['as_of']=datetime(2026,10,4+i,12,tzinfo=zone).isoformat()
    source['daily'][-1]['as_of']=datetime(2026,10,10,3,tzinfo=zone).astimezone(timezone.utc).isoformat()
    assert smooth_load_history(source,'Pacific/Auckland')['current']['status']=='above'


def test_crossing_and_rebounding_strip_describes_relation_not_slope():
    source=load(ratio=.8)
    # Rebounding but still below baseline is correctly called Less load than usual,
    # not Easing off. No label claims the load is falling.
    source['daily'][-1]['recent']=sample(27)
    below=smooth_load_history(source)['current']
    assert below['status']=='below' and below['recent']['value']>source['daily'][0]['recent']['value']
    source['daily'][-7:]=[{**r,'recent':sample(60)} for r in source['daily'][-7:]]
    above=smooth_load_history(source)['current']
    assert above['status']=='above'
    assert load_status(above)['label']=='More load than usual'
    assert load_status(above)['detail']=='Recent training load is 100% higher than your six-week training trend.'


def test_window_invariance_shared_raw_load_and_no_performance_fit(project, monkeypatch):
    import run_analysis.training_overview as module
    import run_analysis.analytics as analytics
    _, path, config=project
    end, rows=history()
    rows += [row(end.date(),100)]
    monkeypatch.setattr(module,'prepare_progress_data',lambda db:SimpleNamespace(sessions=[],points=[],details={}))
    monkeypatch.setattr(module,'recorded_cardio',lambda db,config,at:[r for r in rows if r['day']<=at.date()])
    monkeypatch.setattr(analytics,'build_fitness_analytics',lambda *a,**k:pytest.fail('Unplotted pace fitted'))
    with sqlite3.connect(path) as db:
        short=build_training_overview(db,config,as_of=end,window_days=14)
        long=build_training_overview(db,config,as_of=end,window_days=90)
    assert short['load_display']['daily']==long['load_display']['daily'][-15:]
    assert short['current_status']==long['current_status']
    assert short['load_guidance']['daily']==long['load_guidance']['daily'][-15:]
    assert short['load_display']['current']['ratio']==short['current_status']['ratio']
    assert short['training_load']['current']==long['training_load']['current']
    for plotted,status in zip(long['load_display']['daily'],long['status_history']):
        assert status['status']==classify_load(plotted['recent'],plotted['baseline'])['status']
        assert status['ratio']==plotted['ratio']
        assert status['as_of']==plotted['as_of']


def test_reader_and_endpoint_leave_evidence_unchanged(project, monkeypatch):
    root,path,config=project
    with sqlite3.connect(path) as db:
        id=_insert_run(db,1,END-timedelta(days=1),3,30,100)
        _score(db,id,10)
        db.commit()
    digest=sha256(path.read_bytes()).hexdigest()
    with sqlite3.connect(f'{path.as_uri()}?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        result=build_training_overview(db,config,as_of=END)
        assert db.total_changes==0
        assert len(result['status_history'])==91
        assert result['current_status']['status']=='unknown'
        assert result['current_status']['ratio'] is None
        assert 'runs' not in result['training_load'] and 'comparison' not in result['training_load']
    import run_analysis.web.app as module
    monkeypatch.setattr(module,'build_dashboard',lambda *a,**k:pytest.fail('Legacy Dashboard invoked'))
    with TestClient(create_app(root)) as client:
        response=client.get('/api/training-overview')
        assert response.status_code==200,response.text
        assert response.json()['version']=='dashboard_load_balance_v2'
        assert 'not a readiness score' in response.json()['assumptions']['scope']
    assert sha256(path.read_bytes()).hexdigest()==digest


@pytest.mark.parametrize('ratio,status',[(.79,'below'),(.8,'within'),(1,'within'),(1.3,'within'),(1.31,'above')])
def test_recommended_range_uses_same_plotted_units_and_inclusive_edges(ratio,status):
    display=smooth_load_history(load(ratio=ratio))
    current=recommended_load_history(display)['current']
    assert current['status']==status
    assert current['recommended_range']==dict(low=24,high=39)
    assert current['ratio']==pytest.approx(display['current']['ratio'])


def test_personal_range_scales_and_falls_with_history_not_page_window():
    source=load()
    first=recommended_load_history(smooth_load_history(source))
    larger=deepcopy(source)
    for r in larger['daily']:
        for k in ('recent','baseline'):
            for field in ('value','low','high'): r[k][field]*=3
    second=recommended_load_history(smooth_load_history(larger))
    assert second['current']['status']==first['current']['status']
    assert second['current']['recommended_range']==dict(low=72,high=117)
    # Actual EWMA history: sustained rest lowers the personal reference gradually.
    end,rows=history()
    from run_analysis.progress_training_load import training_load_history
    initial=training_load_history(rows,[],{}, {},end,90,'UTC')
    later=training_load_history(rows,[],{}, {},end+timedelta(days=30),90,'UTC')
    early=recommended_load_history(smooth_load_history(initial))['current']
    late=recommended_load_history(smooth_load_history(later))['current']
    assert late['recommended_range']['high'] < early['recommended_range']['high']


def test_low_patch_is_not_catchup_and_boundary_overlap_is_not_unclear():
    display=smooth_load_history(load(days=6,ratio=.7))
    short=recommended_load_history(display)['current']
    assert short['consecutive_low_days']==1
    assert short['action']=='Keep to your planned recovery and training.'
    sustained=recommended_load_history(smooth_load_history(load(ratio=.7)))['current']
    assert sustained['consecutive_low_days']==7
    assert sustained['action']=='Rebuild consistent training.'
    assert 'do not cram' in sustained['detail']
    ambiguous=recommended_load_history(smooth_load_history(load(ratio=1.3,spread=3)))['current']
    assert ambiguous['status']=='within'
    assert ambiguous['boundary_overlap']
    assert 'unclear' not in ambiguous['label'].lower()
    missing=recommended_load_history(smooth_load_history(load(days=5)))['current']
    assert missing['status']=='unavailable' and missing['recommended_range'] is None


def test_guidance_is_causal_and_missing_dates_break_low_streak():
    display=smooth_load_history(load(ratio=.7))
    before=deepcopy(display)
    result=recommended_load_history(display)
    future={**display['daily'][-1],'as_of':(END+timedelta(days=1)).isoformat(),'ratio':3}
    assert recommended_load_history({**display,'daily':display['daily']+[future]})['daily'][:-1]==result['daily']
    assert display==before
    del display['daily'][-2]
    assert recommended_load_history(display)['current']['consecutive_low_days']==1


def test_low_streak_is_window_invariant_with_six_prior_guidance_dates():
    display=smooth_load_history(load(days=120,ratio=.7))
    full=recommended_load_history(display)
    short=recommended_load_history({**display,'daily':display['daily'][-21:]})
    assert short['daily'][-15:]==full['daily'][-15:]


def test_boundary_uncertainty_never_prescribes_catchup_training():
    current=recommended_load_history(smooth_load_history(load(ratio=.79,spread=3)))['current']
    assert current['status']=='below' and current['consecutive_low_days']==7
    assert current['boundary_overlap']
    assert current['action']=='Keep to your planned recovery and training.'


def test_health_check_in_is_separate_not_a_load_override(project,monkeypatch):
    import run_analysis.training_overview as module
    _,path,config=project
    report=dict(status='pain_or_injury_concern',reported_at=END-timedelta(hours=2))
    monkeypatch.setattr(module,'read_health_reports',lambda db,as_of:(report,))
    with sqlite3.connect(path) as db:
        db.row_factory=sqlite3.Row
        fresh=build_training_overview(db,config,as_of=END)
        assert fresh['current_check_in']['status']=='pain_or_injury_concern'
        assert all(r['status']=='unknown' for r in fresh['status_history'])
        assert build_training_overview(db,config,as_of=END+timedelta(days=2))['current_check_in'] is None
