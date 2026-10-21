from copy import deepcopy
from datetime import datetime,timedelta,timezone
import sqlite3

from run_analysis.db import initialize
from run_analysis.progress import prepare_progress_data
from run_analysis.progress_chart_context import build_chart_context
from run_analysis.aerobic_performance import _fit,progress_from_comparability
from tests.test_aerobic_candidate import report
from tests.test_pace_comparability import config
from test_progress import _insert_run


def test_chart_context_hides_manual_exclusions_without_erasing_load_or_auto_hiding_injury():
    db=sqlite3.connect(':memory:');db.row_factory=sqlite3.Row;initialize(db)
    end=datetime(2026,10,14,tzinfo=timezone.utc);identities=[]
    for i,(role,include,health) in enumerate((('easy',0,'normal'),('hike',0,'normal'),
        ('run_walk',None,'injury_affected'),('intervals',None,'normal'),
        ('tempo_threshold',0,'normal'),('run_walk',1,'normal'))):
        identity=_insert_run(db,i+1,end-timedelta(days=i),3,30,100);identities.append(identity)
        db.execute('''INSERT INTO run_overrides(activity_id,workout_type,include_in_model,health_tag)
            SELECT activity_id,?,?,? FROM activities WHERE id=?''',(role,include,health,identity))
    before=deepcopy(prepare_progress_data(db));changes=db.total_changes
    result=build_chart_context(db)
    assert set(result['hidden_activity_ids'])=={identities[0],identities[1],identities[4]}
    assert set(result['manually_excluded_activity_ids'])==set(result['hidden_activity_ids'])
    assert [r['activity_id'] for r in result['quality_sessions']]==[identities[3]]
    assert result['quality_sessions'][0]['observed_pace_min_mile']==10.
    assert not result['quality_sessions'][0]['included_in_trend']
    assert prepare_progress_data(db)==before
    assert db.total_changes==changes
    assert identities[0] in {s.activity_id for s in before.sessions}


def test_context_is_cropped_but_never_changes_trend_or_score_weights(config):
    evidence,end=report(config);before=deepcopy(evidence);native=_fit(evidence)
    context=dict(hidden_activity_ids=[999],manually_excluded_activity_ids=[999],quality_sessions=[
        dict(activity_id=90+i,as_of=(end+timedelta(days=days)).isoformat(),workout_type='intervals',
            observed_pace_min_mile=100.) for i,days in enumerate((-40,-5,5))])
    with sqlite3.connect(':memory:') as db:
        result=progress_from_comparability(db,evidence,as_of=end,history_days=14,chart_context=context)
    assert result['current']==native['current']
    assert [r['activity_id'] for r in result['quality_sessions']]==[91]
    assert result['manually_excluded_activity_ids']==[999]
    assert result['observations']==native['observations'][-len(result['observations']):]
    assert evidence==before


def test_explicit_hidden_run_cannot_become_a_scored_outlier_when_metrics_are_stale(config):
    evidence,end=report(config);hidden=evidence['runs'][-1]['activity_id']
    evidence['runs'][-1]['pace_min_mile']=50.
    result=_fit({**evidence,'chart_context':dict(hidden_activity_ids=[hidden])})
    assert hidden not in {r['activity_id'] for r in result['observations']}
    assert result['current']['run_count']==len(evidence['runs'])-1
    assert result['current']['pace_min_mile']<11.


def test_quality_history_stays_visible_without_any_comparable_run():
    at=datetime(2026,10,14,tzinfo=timezone.utc)
    context=dict(hidden_activity_ids=[],quality_sessions=[dict(activity_id=1,as_of=at.isoformat(),workout_type='intervals')])
    with sqlite3.connect(':memory:') as db:
        result=progress_from_comparability(db,{'runs':[]},as_of=at,chart_context=context)
    assert result['available'] is False
    assert len(result['quality_sessions'])==1
