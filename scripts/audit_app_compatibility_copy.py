"""Baseline API smoke on a SQLite backup, never the live running app/database."""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from time import perf_counter
import yaml

from fastapi.testclient import TestClient
from run_analysis.config import load_config
from run_analysis.web.app import create_app
from scripts.research_manifest import freeze_sources


def run(root,database):
    root.mkdir(parents=True,exist_ok=False)
    config=load_config()
    manifest=freeze_sources(root/'sources.zip',config=config,
        entrypoint='scripts.audit_app_compatibility_copy',inputs=(database,))
    target=root/'app.sqlite'
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as source:
        with sqlite3.connect(target) as copy:
            source.backup(copy)
    config=dict(config,paths=dict(config['paths'],database='app.sqlite',
        overrides='overrides.csv',weather_cache='weather-cache',report='report.html'))
    with (root/'config.yaml').open('x') as stream:
        yaml.safe_dump(config,stream)
    client=TestClient(create_app(root,'config.yaml'),raise_server_exceptions=False)
    results=[]
    for endpoint in ('/api/progress','/api/runs?limit=3','/api/current-status','/api/recommendation/latest'):
        at=perf_counter()
        response=client.get(endpoint)
        result=dict(endpoint=endpoint,status=response.status_code,seconds=perf_counter()-at,
            body_sha256=sha256(response.content).hexdigest())
        if response.status_code>=400:
            result['error']=response.text[:2000]
        results.append(result)
        print(json.dumps(result),flush=True)
        if endpoint.startswith('/api/runs') and response.status_code==200:
            for row in response.json():
                detail=client.get('/api/runs/'+str(row['activity_id']))
                results.append(dict(endpoint='/api/runs/'+str(row['activity_id']),status=detail.status_code))
    report=dict(manifest=manifest,results=results,scope='unchanged current API on an isolated source backup; not adaptive planner route acceptance',
        source_opened_read_only=True,live_server_touched=False)
    with (root/'summary.json').open('x') as stream:
        json.dump(report,stream,indent=2)
    return report


if __name__=='__main__':
    p=ArgumentParser()
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--database',type=Path,default=Path('data/run_analysis.sqlite'))
    a=p.parse_args()
    run(a.output_dir,a.database)
