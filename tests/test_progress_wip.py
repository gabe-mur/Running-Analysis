from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from run_analysis.config import load_config
from run_analysis.db import connect, initialize
from run_analysis.progress import AVAILABLE_WINDOWS, build_progress
from run_analysis.progress_wip import build_progress_wip, _drift, _load_total, _vo2, _effort_history, _canonical_load
from run_analysis.training_load import calculate_session_load
from run_analysis.vo2_estimation import estimate_local_vo2
from run_analysis.web.app import create_app
from run_analysis.web.schemas import WorkoutType, FitnessTrend, PaceValue
from test_progress import _insert_run
from test_web_phase1 import _write_config


@pytest.fixture
def project(tmp_path):
    _write_config(tmp_path)
    database=tmp_path/'data/test.sqlite'
    database.parent.mkdir()
    with connect(database) as db:
        initialize(db)
    db.close()  # Checkpoint fixture setup before taking a byte-level baseline.
    config=load_config(tmp_path/'config.yaml')
    config['profile']=dict(birth_date='1990-01-01',height_in=70,weight_lb=140,sex='male',max_hr_source='measured')
    return tmp_path,database,config


def _score(db,id,pace):
    db.execute('INSERT INTO model_runs(activity_id,model_name,model_version,result_json) VALUES (?,?,?,?)',
        (id,'standardized_pace_at_target_hr','test',json.dumps(dict(
            standardized_pace_at_target_hr_min_mile=pace,raw_pace_at_target_hr_min_mile=pace+.3,
            uncertainty_95_min_mile=.3,effective_window_count=6,reference_time_support='interpolation'))))


def test_pace_retains_native_anchor_volume_ends_today_and_future_data_excluded(project):
    root,path,config=project
    end=datetime(2026,10,4,12,tzinfo=timezone.utc)
    with connect(path) as db:
        for i,days in enumerate((50,40,35,30,20,15,10,-1),1):
            id=_insert_run(db,i,end-timedelta(days=days),3+i/10,40,100)
            _score(db,id,10+i/10)
        db.commit()
        result=build_progress_wip(db,config,root,28,as_of=end)
        # Match the established pace semantics even when its anchor differs
        # from the request-time physical volume window.
        native=build_progress(db,28,config=config,as_of=end).model_dump(mode='json')
        assert result['performance']['current']['run_count']==5
        assert result['performance']['prior']['run_count']==2
        assert result['performance']['current']['pace_min_mile']==native['current_pace']['minutes_per_mile']
        assert result['performance']['period_change']==native['period_change']
        assert result['performance']['within_window_trend']==native['within_window_trend']
        assert result['exposure']['current']['run_count']==3
        assert len(result['activities'])==3
        assert len(result['exposure']['daily'])==29
        assert result['performance']['last_observation']==end-timedelta(days=10)
        assert result['performance']['trajectory']['distance_adjusted'] is True
        assert all(datetime.fromisoformat(a['start_time'])<=end for a in result['activities'])
        # Same physical HR accounting as released Progress; the WIP intensity
        # headline remains 28-day even if the displayed history is shorter.
        assert result['effort']['current']['easy_percent'] == native['intensity']['easy_percent']
        assert result['effort']['current']['known_hr_minutes'] == native['intensity']['known_hr_minutes']
        assert result['effort']['current']['above_easy_percent'] == pytest.approx(
            native['intensity']['moderate_percent'] + native['intensity']['hard_percent'])
        shorter = build_progress_wip(db,config,root,14,as_of=end)
        assert shorter['effort']['current'] == result['effort']['current']


def test_reader_leaves_old_progress_and_database_unchanged(project):
    root,path,config=project
    end=datetime.now(timezone.utc)
    with connect(path) as db:
        id=_insert_run(db,1,end-timedelta(days=1),4,40,100)
        _score(db,id,10)
        db.commit()
        before=build_progress(db,28,config=config,as_of=end).model_dump(mode='json')
    digest=sha256(path.read_bytes()).hexdigest()
    with sqlite3.connect(f'{path.as_uri()}?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        db.execute('BEGIN')
        build_progress_wip(db,config,root,28,as_of=end)
        after=build_progress(db,28,config=config,as_of=end).model_dump(mode='json')
        assert db.total_changes==0
    assert before==after
    assert sha256(path.read_bytes()).hexdigest()==digest


def test_visible_progress_payload_is_identical_without_removed_panel_work(project, monkeypatch):
    import run_analysis.progress_wip as reader
    root,path,config=project
    end=datetime(2026,10,4,12,tzinfo=timezone.utc)
    with connect(path) as db:
        for i,days in enumerate((40,20,15,10,2),1):
            activity_id=_insert_run(db,i,end-timedelta(days=days),4,40,100)
            _score(db,activity_id,10+i/10)
        db.commit()
        full=build_progress_wip(db,config,root,28,as_of=end)
        def unwanted(*args,**kwargs):
            raise AssertionError('Removed exploratory panel was computed')
        for name in ('get_run_feedback','_load_points','_quality_performance'):
            monkeypatch.setattr(reader,name,unwanted)
        lean=build_progress_wip(db,config,root,28,as_of=end,include_exploratory=False)
        for key in ('baseline','performance','exposure','effort','load','activities',
                    'training_load','fatigue','race_goal','signals'):
            assert lean[key]==full[key],key
        assert lean['quality']==lean['durability']==[]


def test_api_is_read_only_and_original_route_remains_available(project):
    root,path,config=project
    digest=sha256(path.read_bytes()).hexdigest()
    with TestClient(create_app(root)) as client:
        response=client.get('/api/progress-wip')
        assert response.status_code==200,response.text
        data=response.json()
        assert data['performance']['current'] is None
        assert data['load']['current']['cardio'] is None
        assert data['load']['current']['complete'] is False
        assert data['signals']['planner_influence'] is False
        assert client.get('/api/progress-wip?window_days=15').status_code==422
        assert sha256(path.read_bytes()).hexdigest()==digest
        assert client.get('/api/progress').status_code==200


def test_wip_can_skip_undisplayed_legacy_fit_without_changing_visible_results(project,monkeypatch):
    from run_analysis import aerobic_candidate
    root,path,config=project
    with TestClient(create_app(root)) as client:
        full=client.get('/api/progress-wip?include_exploratory=false').json()
        def forbidden(*args,**kwargs):
            pytest.fail('WIP invoked the undisplayed legacy fit')
        monkeypatch.setattr(aerobic_candidate,'progress_from_comparability',forbidden)
        lean=client.get('/api/progress-wip?include_exploratory=false&include_legacy_aerobic=false').json()
    assert 'aerobic_progress' in full and 'aerobic_progress' not in lean
    for key in ('aerobic_performance','pace_comparison','performance','exposure','effort'):
        # Only the request clock changes; identical evidence and definitions.
        if key=='aerobic_performance':
            assert lean[key]['observations']==full[key]['observations']
        elif key=='performance':
            assert lean[key]['current']==full[key]['current']
        else:
            assert lean[key].keys()==full[key].keys()


def test_no_runtime_created_when_committed_ledger_is_missing(project):
    root,path,config=project
    missing=root/'data/nonexistent-runtime'
    with connect(path) as db:
        db.execute("INSERT INTO app_state(key,value_json,updated_at_utc) VALUES ('adaptive_runtime',?,'now')",
            (json.dumps(dict(root=str(missing))),))
        db.commit()
        result=build_progress_wip(db,config,root)
    assert result['load']['status']=='unavailable'
    assert not missing.exists()


def test_invalid_snapshot_is_not_adopted_or_repaired(project):
    root,path,config=project
    runtime=root/'data/bad-runtime'
    runtime.mkdir()
    journal=runtime/'sandbox.sqlite'
    with sqlite3.connect(journal) as db:
        db.execute('CREATE TABLE sandbox_decisions(sequence INTEGER,event TEXT,snapshot TEXT)')
        db.execute('INSERT INTO sandbox_decisions VALUES (1,?,?)',
            (json.dumps(dict(snapshot_fingerprint='not-a-match')),'{}'))
    digest=sha256(journal.read_bytes()).hexdigest()
    with connect(path) as db:
        db.execute("INSERT INTO app_state(key,value_json,updated_at_utc) VALUES ('adaptive_runtime',?,'now')",
            (json.dumps(dict(root=str(runtime))),))
        db.commit()
        result=build_progress_wip(db,config,root)
    assert result['load']['status']=='unavailable'
    assert 'ValueError' in result['load']['reason']
    assert sha256(journal.read_bytes()).hexdigest()==digest


def test_synced_physical_keys_map_to_app_ids_read_only(project, monkeypatch):
    from dataclasses import asdict
    from run_analysis.workload_accounting import Quantity
    from run_analysis.adaptive.program_objective_trial import DemandScenario
    root,path,config=project
    end=datetime(2026,10,10,20,tzinfo=timezone.utc)
    run=SimpleNamespace(key='sandbox:1', source='observed', start=end-timedelta(days=1),
        end=end-timedelta(days=1)+timedelta(minutes=40), miles=4., role='quality',
        demand=tuple(Quantity.point(v,'test') for v in (40,60,10)), work_evidence_basis='recorded', accounting_key='units')
    state=SimpleNamespace(history=[run], planner_preferences=None, as_of=end, stage_key='stage',
        timezone='UTC',context=SimpleNamespace(scales=(40.,60.,10.), scenario=DemandScenario(),demand_accounting_key='units'))
    monkeypatch.setattr('run_analysis.adaptive.decision_snapshot.restore',lambda payload,expected_fingerprint:state)
    monkeypatch.setattr('run_analysis.adaptive.planner_preferences.latest_request',lambda runtime,as_of:None)
    runtime=root/'data/runtime'; runtime.mkdir()
    journal=runtime/'sandbox.sqlite'
    with sqlite3.connect(journal) as db:
        db.execute('CREATE TABLE sandbox_decisions(sequence INTEGER,event TEXT,snapshot TEXT)')
        db.execute('INSERT INTO sandbox_decisions VALUES (1,?,?)',(json.dumps({'snapshot_fingerprint':'test'}),'{}'))
        db.execute('CREATE TABLE adaptive_app_sources(activity_id TEXT,physical_key TEXT)')
        db.execute("INSERT INTO adaptive_app_sources VALUES ('42','sandbox:1')")
    digest=sha256(journal.read_bytes()).hexdigest()
    with connect(path) as db:
        db.execute("INSERT INTO app_state(key,value_json,updated_at_utc) VALUES ('adaptive_runtime',?,'now')",
            (json.dumps(dict(root=str(runtime))),))
        db.commit()
        result=_canonical_load(db,root,[SimpleNamespace(activity_id=42,start_time=run.start,distance_miles=4.)],end)
        assert result['rows'][0]['activity_id']==42
        assert result['rows'][0]['cardio']==asdict(run.demand[1])
        assert result['fatigue_model']['scales']==(40.,60.,10.)
        # Mapping alone cannot validate a mismatched physical run.
        assert not _canonical_load(db,root,[SimpleNamespace(activity_id=42,start_time=run.start,distance_miles=5.)],end)['rows']
    assert sha256(journal.read_bytes()).hexdigest()==digest


def test_partial_and_different_unit_loads_are_not_complete():
    sessions=[SimpleNamespace(activity_id=1),SimpleNamespace(activity_id=2)]
    q=dict(low=8,high=10)
    row=dict(activity_id=1,cardio=q,work_minutes=q,accounting_key='a')
    result=_load_total([row],sessions)
    assert result['cardio']==q
    assert result['covered_runs']==1 and result['complete'] is False
    mixed=_load_total([row,{**row,'activity_id':2,'accounting_key':'b'}],sessions)
    assert mixed['cardio'] is None and mixed['complete'] is False


def test_effort_uses_duration_weighted_fixed_28_days_not_selected_period():
    end = datetime(2026, 10, 9, 12, tzinfo=timezone.utc)
    sessions = [SimpleNamespace(activity_id=i, start_time=end-timedelta(days=days))
        for i, days in enumerate((28, 20, 1, -1))]
    # The twenty-day-old long easy run matters even in a fourteen-day view.
    # Unknown minutes are not added to easy; Z3 is above easy, not all "hard".
    details = {i: {'load': calculate_session_load(zones, seconds)}
        for i, (zones, seconds) in enumerate((
            ({'z4': 6000}, 6000),
            ({'z2': 5400}, 5400),
            ({'z3': 300, 'z4': 300, 'unknown': 600}, 1200),
            ({'z4': 6000}, 6000)))}
    short = _effort_history(sessions, details, end, 14)
    long = _effort_history(sessions, details, end, 90)
    assert short['current'] == long['current']
    assert len(short['daily']) == 15 and len(long['daily']) == 91
    current = short['current']
    assert current['known_hr_minutes'] == 100
    assert current['missing_hr_minutes'] == 10
    assert current['easy_percent'] == 90
    assert current['above_easy_percent'] == 10
    assert current['hr_coverage'] == pytest.approx(100/110)
    assert short['trailing_days'] == 28


def test_effort_without_hr_is_unknown_not_zero_harder_time():
    end = datetime(2026, 10, 9, 12, tzinfo=timezone.utc)
    run = SimpleNamespace(activity_id=1, start_time=end)
    details = {1: {'load': calculate_session_load({}, 1800)}}
    current = _effort_history([run], details, end, 14)['current']
    assert current['above_easy_percent'] is None
    assert current['easy_percent'] is None
    assert current['hr_coverage'] == 0
    assert current['missing_hr_minutes'] == 30
    empty = _effort_history([], {}, end, 14)['current']
    assert empty['hr_coverage'] is None


def test_quality_session_share_uses_purpose_not_hr_and_excludes_nonrunning():
    end = datetime(2026, 10, 9, 12, tzinfo=timezone.utc)
    roles = [WorkoutType.INTERVALS, WorkoutType.TEMPO_THRESHOLD, WorkoutType.RACE,
        WorkoutType.EASY, WorkoutType.RECOVERY, WorkoutType.LONG, WorkoutType.RUN_WALK,
        WorkoutType.UNKNOWN, WorkoutType.OTHER, WorkoutType.BIKE, WorkoutType.HIKE, WorkoutType.REST]
    sessions = [SimpleNamespace(activity_id=i, start_time=end-timedelta(days=1)) for i in range(len(roles))]
    # Missing HR does not erase a classified quality run; Z3 doesn't promote easy.
    details = {i: dict(workout=role, load=calculate_session_load(
        {'z3': 1800} if role == WorkoutType.EASY else {}, 1800)) for i, role in enumerate(roles)}
    short = _effort_history(sessions, details, end, 14)['session_balance']
    long = _effort_history(sessions, details, end, 90)['session_balance']
    assert short['current'] == long['current']
    assert short['current'] == dict(as_of=end.isoformat(), quality_sessions=3,
        easy_sessions=4, classified_runs=7, unknown_runs=2, completed_runs=9, quality_percent=3/7*100)
    assert short['trailing_runs'] == 20 and 'trailing_days' not in short
    assert short['reference']['low'] == 20 and short['reference']['high'] == 25


def test_quality_session_window_retains_old_runs_excludes_future_unknown_and_zero():
    end = datetime(2026, 10, 9, 12, tzinfo=timezone.utc)
    sessions = [SimpleNamespace(activity_id=i, start_time=end-timedelta(days=days))
        for i, days in enumerate((28, 20, 1, -1))]
    details = {i: dict(workout=role, load=calculate_session_load({}, 1800))
        for i, role in enumerate((WorkoutType.INTERVALS, WorkoutType.LONG, WorkoutType.UNKNOWN, WorkoutType.RACE))}
    current = _effort_history(sessions, details, end, 14)['session_balance']['current']
    assert current['classified_runs'] == 2 and current['unknown_runs'] == 1
    assert current['completed_runs'] == 3
    assert current['quality_percent'] == 50
    empty = _effort_history([], {}, end, 14)['session_balance']['current']
    assert empty['quality_percent'] is None
    unknown = _effort_history([sessions[2]], details, end, 14)['session_balance']['current']
    assert unknown['quality_percent'] is None and unknown['unknown_runs'] == 1


def test_quality_share_changes_only_when_a_recorded_run_enters_not_on_rest_days():
    end = datetime(2026, 10, 9, 12, tzinfo=timezone.utc)
    # A long break must not silently age out easy runs or quality runs.
    sessions = [SimpleNamespace(activity_id=i, start_time=end-timedelta(days=90-i))
        for i in range(21)]
    roles = [WorkoutType.INTERVALS] + [WorkoutType.EASY] * 15 + [WorkoutType.INTERVALS] * 5
    details = {i: dict(workout=role, load=calculate_session_load({}, 1800))
        for i, role in enumerate(roles)}
    # Unsorted history, a newer bike, and a future quality run cannot alter the
    # last twenty completed run slots.
    sessions.reverse()
    sessions += [SimpleNamespace(activity_id=21, start_time=end),
        SimpleNamespace(activity_id=22, start_time=end+timedelta(days=8))]
    details[21] = dict(workout=WorkoutType.BIKE, load=calculate_session_load({}, 1800))
    details[22] = dict(workout=WorkoutType.INTERVALS, load=calculate_session_load({}, 1800))
    balance = _effort_history(sessions, details, end, 14)['session_balance']
    assert balance['current']['quality_sessions'] == 5
    assert balance['current']['completed_runs'] == 20
    assert balance['current']['quality_percent'] == 25
    assert {r['quality_percent'] for r in balance['daily']} == {25}
    later = _effort_history(sessions, details, end+timedelta(days=7), 90)['session_balance']['current']
    assert {k: v for k, v in later.items() if k != 'as_of'} == {
        k: v for k, v in balance['current'].items() if k != 'as_of'}
    next_run = _effort_history(sessions, details, end+timedelta(days=8), 14)['session_balance']['current']
    assert next_run['quality_sessions'] == 6
    assert next_run['quality_percent'] == 30


def test_unknown_run_occupies_one_of_twenty_slots_without_becoming_easy():
    end = datetime(2026, 10, 9, 12, tzinfo=timezone.utc)
    sessions = [SimpleNamespace(activity_id=i, start_time=end-timedelta(days=40-i))
        for i in range(21)]
    roles = [WorkoutType.INTERVALS] + [WorkoutType.EASY] * 19 + [WorkoutType.UNKNOWN]
    details = {i: dict(workout=role, load=calculate_session_load({}, 1800))
        for i, role in enumerate(roles)}
    current = _effort_history(sessions, details, end, 14)['session_balance']['current']
    assert current['completed_runs'] == 20
    assert current['classified_runs'] == 19 and current['unknown_runs'] == 1
    assert current['quality_sessions'] == 0 and current['quality_percent'] == 0


def _intervals(heart_rate=150,endpoint_missing=False):
    output=[]
    for index in range(360):
        hr=100 if index<30 else heart_rate
        output.append(SimpleNamespace(moving_time_s=10.,elapsed_s=10.,distance_m=30.,stopped_time_s=0.,
            start=SimpleNamespace(heart_rate_bpm=hr),end=SimpleNamespace(heart_rate_bpm=None if endpoint_missing else hr),flags=[]))
    return output


def test_drift_excludes_warmup_and_requires_paired_hr():
    result=_drift(_intervals(),WorkoutType.EASY)
    assert result['available'] is True
    assert result['decoupling_percent']==pytest.approx(0)
    assert result['legacy_decoupling_percent']>0
    assert result['first_hr']==150
    missing=_drift(_intervals(endpoint_missing=True),WorkoutType.EASY)
    assert missing['available'] is False
    assert '80%' in missing['reason']


@pytest.mark.parametrize('role',[WorkoutType.INTERVALS,WorkoutType.TEMPO_THRESHOLD,WorkoutType.RACE])
def test_quality_is_not_a_steady_drift_test(role):
    assert _drift(_intervals(),role)['available'] is False


def test_short_postwarmup_duration_is_explicitly_missing():
    result=_drift(_intervals()[:200],WorkoutType.EASY)
    assert result['available'] is False
    assert result['decoupling_percent'] is None


@pytest.mark.parametrize('pace,profile,hr',[
    (10,{},{}), (10,{'max_hr_source':'estimated'},{}),
    (6,{},{}), (20,{},{}), (10,{},dict(max_hr=60,resting_hr=65)),
    (10,{},dict(target_hr=190)), (10,None,{}),
])
def test_vo2_adapter_preserves_shared_pooling_uncertainty_and_unavailable_reasons(project,pace,profile,hr):
    root,path,config=project
    end=datetime(2026,10,4,tzinfo=timezone.utc)
    config={**config,**hr,'profile':{**config['profile'],**profile} if profile is not None else None}
    with connect(path) as db:
        baseline=build_progress(db,28,config=config,as_of=end)
    estimate=dict(pace_min_mile=pace,uncertainty_95_min_mile=.3)
    actual=_vo2(estimate,config,as_of=end,baseline=baseline)
    expected=estimate_local_vo2(current_pace=PaceValue(minutes_per_mile=pace,display=''),
        current_pace_uncertainty_95=.3,as_of=end,recent_load=baseline.current_load,
        fitness_trend=FitnessTrend.UNCERTAIN,config=config)
    assert actual==expected
    assert _vo2(None,config,as_of=end,baseline=baseline).value_ml_kg_min is None


def test_quality_not_lost_without_aerobic_score(project):
    root,path,config=project
    end=datetime.now(timezone.utc)
    with connect(path) as db:
        id=_insert_run(db,1,end-timedelta(days=1),4,40,100)
        db.execute("INSERT INTO run_overrides(activity_id,workout_type,health_tag) VALUES ('external-1','tempo_threshold','normal')")
        for index,(seconds,distance,hr) in enumerate(((720,1600,140),(1080,2900,168),(840,1800,150))):
            db.execute('INSERT INTO laps(activity_id,lap_index,total_time_s,distance_m,average_hr_bpm,maximum_hr_bpm) VALUES (?,?,?,?,?,?)',
                (id,index,seconds,distance,hr,hr+5))
        db.commit()
        result=build_progress_wip(db,config,root,as_of=end)
    assert result['performance']['current'] is None
    assert result['activities'][0]['aerobic_eligible'] is False
    assert len(result['quality'])==1
    assert result['quality'][0]['continuous']['duration_seconds']==1080
    assert result['quality'][0]['continuous']['average_hr_bpm']==168
    assert result['durability']==[]


@pytest.mark.parametrize('window',AVAILABLE_WINDOWS)
def test_native_retention_contract_across_windows_and_sparse_periods(project,window):
    root,path,config=project
    end=datetime(2026,10,4,12,tzinfo=timezone.utc)
    with connect(path) as db:
        for i,days in enumerate((370,190,100,50,40,32,24,20,16,12,8,4),1):
            id=_insert_run(db,i,end-timedelta(days=days),4+i/20,40,100)
            _score(db,id,10+i/30)
        db.commit()
        old=build_progress(db,window,config=config,as_of=end).model_dump(mode='json')
        result=build_progress_wip(db,config,root,window,as_of=end)
    perf=result['performance']
    assert result['baseline']==old
    assert perf['period_change']==old['period_change']
    assert perf['within_window_trend']==old['within_window_trend']
    assert perf['confidence']==old['fitness_confidence']
    assert perf['fitness_trend']==old['fitness_trend']
    assert perf['rolling_28d']==old['trend_28d']
    assert perf['current']['pace_min_mile']==old['current_pace']['minutes_per_mile']
    assert perf['current']['uncertainty_95_min_mile']==old['uncertainty_95_min_mile']
    assert result['exposure']['current']['distance_miles']==old['period_comparison']['current']['distance_miles']
    assert result['exposure']['current']['run_count']==old['period_comparison']['current']['run_count']


def test_preserves_native_chart_data_and_vo2_curve_matches_headline(project):
    root,path,config=project
    end=datetime(2026,10,4,12,tzinfo=timezone.utc)
    with connect(path) as db:
        for i,days in enumerate((24,20,16,12,8,4),1):
            id=_insert_run(db,i,end-timedelta(days=days),4,40,100)
            _score(db,id,10)
        db.commit()
        old=build_progress(db,28,config=config,as_of=end).model_dump(mode='json')
        result=build_progress_wip(db,config,root,28,as_of=end)
    assert result['baseline']==old
    # Shared history keeps two decimals, shared headline one decimal.
    assert round(result['performance']['vo2_series'][-1]['value_ml_kg_min'],1)==result['performance']['vo2']['value_ml_kg_min']
    assert round(result['performance']['vo2_series'][-1]['uncertainty_95_ml_kg_min'],1)==result['performance']['vo2']['uncertainty_95_ml_kg_min']
    assert 'George' in result['performance']['vo2']['method']


def test_current_levels_and_insights_do_not_change_with_display_window(project):
    root,path,config=project
    end=datetime(2026,10,4,12,tzinfo=timezone.utc)
    with connect(path) as db:
        for i,days in enumerate((100,24,12,4),1):
            id=_insert_run(db,i,end-timedelta(days=days),4,40,100)
            _score(db,id,10+i/10)
        db.commit()
        results=[build_progress_wip(db,config,root,days,as_of=end) for days in (14,28,180)]
    assert all(r['performance']['vo2_input']==results[0]['performance']['vo2_input'] for r in results)
    assert all({k:v for k,v in r['performance']['vo2'].items() if k!='series'}==
        {k:v for k,v in results[0]['performance']['vo2'].items() if k!='series'} for r in results)
    assert all(r['insights']==results[0]['insights'] for r in results)
    assert results[0]['exposure']['current']['run_count']==2
    assert len(results[0]['insights']['activities'])==4
    assert results[0]['performance']['rolling_raw_28d'][-1]['pace_min_mile']>results[0]['performance']['rolling_28d'][-1]['pace_min_mile']
    for row in results[0]['exposure']['daily']:
        assert sum(row['distance_by_role'].values())==pytest.approx(row['distance_7d'])


def test_current_vo2_does_not_reuse_old_or_single_run_evidence(project):
    root,path,config=project
    end=datetime(2026,10,4,12,tzinfo=timezone.utc)
    with connect(path) as db:
        for i,days in enumerate((24,4),1):
            id=_insert_run(db,i,end-timedelta(days=days),4,40,100)
            _score(db,id,10)
        db.commit()
        result=build_progress_wip(db,config,root,28,as_of=end)
    assert result['performance']['current'] is not None
    assert result['performance']['vo2_input'] is None
    assert result['performance']['vo2']['value_ml_kg_min'] is None
    assert result['performance']['vo2']['confidence']=='unavailable'


def test_missing_current_vo2_keeps_supported_history(project):
    root,path,config=project
    end=datetime(2026,10,4,12,tzinfo=timezone.utc)
    with connect(path) as db:
        for i,days in enumerate((24,20),1):
            id=_insert_run(db,i,end-timedelta(days=days),4,40,100)
            _score(db,id,10)
        db.commit()
        result=build_progress_wip(db,config,root,28,as_of=end)
    assert result['performance']['vo2']['value_ml_kg_min'] is None
    assert result['performance']['vo2_series']
    assert all(datetime.fromisoformat(p['as_of']) < end-timedelta(days=10)
        for p in result['performance']['vo2_series'])


def test_saved_work_block_not_discarded_when_continuous_detector_has_no_result(project):
    root,path,config=project
    end=datetime(2026,10,4,12,tzinfo=timezone.utc)
    with connect(path) as db:
        id=_insert_run(db,1,end-timedelta(days=1),4,40,100)
        db.execute("INSERT INTO run_overrides(activity_id,workout_type,health_tag) VALUES ('external-1','tempo_threshold','normal')")
        db.execute('INSERT INTO model_runs(activity_id,model_name,model_version,result_json) VALUES (?,?,?,?)',
            (id,'standardized_pace_at_target_hr','test',json.dumps(dict(quality_performance=dict(
                source='detected_work_block',duration_minutes=18,pace_min_mile=8.5,average_hr_bpm=165)))))
        db.commit()
        result=build_progress_wip(db,config,root,as_of=end)
    quality=result['quality'][0]
    assert quality['continuous'] is None
    assert quality['recorded_performance']==result['baseline']['quality_performance'][0]
    # Unknown structure must not be silently reclassified as comparable tempo.
    assert result['insights']['quality_comparisons']['sustained']['total_sessions']==0


def test_insight_buffer_preserves_pre_display_quality_without_showing_old_session(project):
    root,path,config=project
    end=datetime(2026,10,4,12,tzinfo=timezone.utc)
    with connect(path) as db:
        id=_insert_run(db,1,end-timedelta(days=190),4,40,100)
        db.execute("INSERT INTO run_overrides(activity_id,workout_type,health_tag) VALUES ('external-1','tempo_threshold','normal')")
        for index,(seconds,distance,hr) in enumerate(((720,1600,140),(1080,2900,168),(840,1800,150))):
            db.execute('INSERT INTO laps(activity_id,lap_index,total_time_s,distance_m,average_hr_bpm,maximum_hr_bpm) VALUES (?,?,?,?,?,?)',(id,index,seconds,distance,hr,hr+5))
        db.commit()
        result=build_progress_wip(db,config,root,28,as_of=end)
    assert result['quality']==[]
    assert result['insights']['activities']==[]
    assert len(result['insights']['quality_history'])==1
    assert result['insights']['quality_history'][0]['continuous']['duration_seconds']==1080
