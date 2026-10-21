"""Real-data cloned route controls; never writes the application's database.

The one protected-date search is real. Forecasts are explicitly unavailable in
this offline control, not forged normal-weather evidence. Adoption and every
saved winner are independently repriced before exercising the app boundary.
"""
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
import json
import shutil
import sqlite3
from time import perf_counter
from unittest.mock import patch
import yaml
from fastapi.testclient import TestClient

from run_analysis.config import load_config, resolve_project_path
from run_analysis.db import connect, initialize
from run_analysis.adaptive.live_bridge import attach_runtime, ENGINE
from run_analysis.web.app import create_app
from scripts.replay_adaptive_search import load_decision


def smoke(staged, output):
    output.mkdir(parents=True,exist_ok=False)
    config=load_config()
    original=resolve_project_path(Path.cwd(),config['paths']['database'])
    database=output/'app-copy.sqlite'
    with sqlite3.connect(original.resolve().as_uri()+'?mode=ro',uri=True) as source, sqlite3.connect(database) as dest:
        source.backup(dest)
    runtime=output/'runtime'
    shutil.copytree(staged,runtime)
    checked=load_decision(runtime,1)
    config['paths']['database']=str(database.resolve())
    config['adaptive_planner']={'engine':ENGINE}
    config['weather']['forecast_enabled']=False
    config['weather']['emergency_alerts_enabled']=False
    config_path=output/'config.yaml'
    config_path.write_text(yaml.safe_dump(config))
    at=datetime.now(timezone.utc)
    with connect(database) as db:
        initialize(db)
        attach_runtime(db,runtime,config,at)
        activity_count=db.execute('SELECT COUNT(*) FROM activities').fetchone()[0]
        latest=dict(db.execute('''SELECT s.path FROM activities a JOIN activity_sources l ON l.activity_id=a.id
            AND l.is_primary=1 JOIN source_files s ON s.id=l.source_file_id
            WHERE a.sport='Running' ORDER BY a.start_time_utc DESC LIMIT 1''').fetchone())
    def unavailable(*args):
        return [[(at,None) for at in group] for group in args[-1]]
    report=dict(archived_cost=checked['archived_cost'],original_database_modified=False,
        forecast_control='unavailable, not invented normal weather',routes={},timings={})
    with patch('run_analysis.forecast.planned_forecast_options',unavailable), TestClient(create_app(Path.cwd(),config_path.resolve())) as client:
        for route in ('/api/health','/api/weekly-schedule/latest','/api/weekly-schedule/latest',
                      '/api/recommendation/latest','/api/progress','/api/dashboard','/api/fitness-state','/api/settings'):
            began=perf_counter();response=client.get(route)
            assert response.status_code==200,(route,response.text)
            report['routes'][route]=response.status_code
            report['timings'].setdefault(route,[]).append(perf_counter()-began)
        initial=client.get('/api/weekly-schedule/latest').json()
        assert initial['planner_version']==201 and initial['planner_status']=='ready'
        assert len(list(runtime.glob('decision-*.json')))==1,'Identical GET performed a search'
        path=Path(latest['path'])
        response=client.post('/api/uploads',files={'files':(path.name,path.read_bytes(),'application/octet-stream')})
        assert response.status_code==200,response.text
        assert len(list(runtime.glob('decision-*.json')))==1,'Duplicate upload changed the planner'
        report['routes']['duplicate_original_upload']=response.status_code
        picked=next(d for d in initial['days'] if d['recommendation'])['date']
        began=perf_counter()
        response=client.post('/api/weekly-schedule/rest-day',json={'date':picked,'is_rest_day':True})
        assert response.status_code==200,response.text
        blocked=response.json()
        assert blocked['planner_status']=='ready',blocked['summary']
        day=next(d for d in blocked['days'] if d['date']==picked)
        assert day['forced_rest'] and day['recommendation'] is None
        report['timings']['protected_date_search']=perf_counter()-began
        report['routes']['protected_date']=response.status_code
        decisions=len(list(runtime.glob('decision-*.json')))
        assert client.get('/api/weekly-schedule/latest').json()==blocked
        assert len(list(runtime.glob('decision-*.json')))==decisions
        # A real current report is an explicit safety override, not fake rest
        # optimization and not another expensive search.
        response=client.post('/api/weekly-schedule',json={'health_status':'pain_or_injury_concern'})
        assert response.status_code==200,response.text
        paused=response.json()
        assert paused['planner_status']=='review_required'
        assert all(d['recommendation'] is None for d in paused['days'])
        assert len(list(runtime.glob('decision-*.json')))==decisions
        assert client.get('/api/weekly-schedule/latest').json()['planner_status']=='review_required'
        report['routes']['health_guard']=response.status_code
    with connect(database) as db:
        assert db.execute('SELECT COUNT(*) FROM activities').fetchone()[0]==activity_count
    report['decisions']=[]
    for path in sorted(runtime.glob('decision-*.json')):
        saved=load_decision(runtime,int(path.stem.split('-')[1]))
        report['decisions'].append(dict(index=saved['event']['index'],cost=saved['archived_cost'],eligible=True))
    with (output/'report.json').open('x') as stream:
        json.dump(report,stream,indent=2)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    parser=ArgumentParser()
    parser.add_argument('--staged',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    smoke(args.staged,args.output)
