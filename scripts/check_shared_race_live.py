"""Actual race Apply/search/restart on a separate application/runtime copy."""
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
import sqlite3
from unittest.mock import patch

from fastapi.testclient import TestClient
from run_analysis.config import load_config
from run_analysis.db import connect
from run_analysis.adaptive.live_bridge import ENGINE
from run_analysis.adaptive.new_model_daily_harness import DailyHarness
from run_analysis.web.app import create_app
from scripts.replay_adaptive_search import load_decision
from scripts.research_manifest import freeze_sources
from scripts.release_c1_live_planner import journal_digest


def check(prepared, output):
    report=json.loads((prepared/'report.json').read_text())
    source=Path(report['runtime'])
    output.mkdir(parents=True,exist_ok=False)
    root=output/'runtime'
    original_digest=journal_digest(source/'sandbox.sqlite')
    shutil.copytree(source,root,ignore=shutil.ignore_patterns('*.sqlite','*.sqlite-wal','*.sqlite-shm','*.lock'))
    for src,dst in ((source/'sandbox.sqlite',root/'sandbox.sqlite'),
                    (prepared/'app-copy.sqlite',output/'app.sqlite')):
        with sqlite3.connect(src.resolve().as_uri()+'?mode=ro',uri=True) as old, sqlite3.connect(dst) as new:
            old.backup(new)
    cfg=deepcopy(load_config());cfg['paths']['database']=str((output/'app.sqlite').resolve())
    for key in ('uploads','outputs'):
        if key in cfg['paths']:cfg['paths'][key]=str((output/key).resolve())
    with connect(output/'app.sqlite') as app:
        app.execute("UPDATE app_state SET value_json=? WHERE key='adaptive_runtime'",
            (json.dumps(dict(engine=ENGINE,root=str(root.resolve()))),))
        app.commit()
    cert_path=root/'coached-release.json'
    cert=json.loads(cert_path.read_text());cert['database']=cfg['paths']['database']
    cert_path.write_text(json.dumps(cert))
    manifest=freeze_sources(output/'check.sources.zip',config=cfg,entrypoint=__name__,
        inputs=(prepared/'report.json',))
    # Only configuration ownership is substituted. The app handlers, request
    # journal, accounting, predictor, search, publication and restart are real.
    with patch('run_analysis.web.app.load_config',return_value=cfg),TestClient(create_app(Path.cwd())) as client:
        response=client.get('/api/settings/race-preparation/status');response.raise_for_status()
        status=response.json()
        if not status['race_activation_available']:
            raise ValueError('Shared V7 race Apply is not enabled: '+repr(status))
        goal=dict(distance='marathon',preparation_mode='balanced',
            race_date=(datetime.now(timezone.utc).date()+timedelta(days=210)).isoformat())
        response=client.post('/api/settings/race-preparation/apply',json=dict(
            goal=goal,expected_revision=status['expected_revision']))
        response.raise_for_status()
        response=client.get('/api/weekly-schedule/latest');response.raise_for_status()
        result=response.json()
        if result['planner_status']!='ready' or result['planner_version']!=217:
            raise ValueError('Race application did not produce a ready shared plan: '+repr(result))
    event=json.loads(max(root.glob('decision-*.json')).read_text())
    state=event['stage_runtime']['general_progression_state']
    coaching=state['coaching_progression']
    assert coaching['maximum_growth_per_7']==__import__('pytest').approx(.06)
    assert coaching['target_volume_per_7']>=40 and coaching['target_long_miles']>=20
    assert event['race_preparation']['distance']=='marathon'
    assert state['realized_reference']==report['progression_state']['realized_reference']
    load_decision(root,event['index']+1)
    digest=journal_digest(root/'sandbox.sqlite')
    def no_search(*args,**kwargs):raise AssertionError('Race restart unexpectedly repeated search')
    with patch('run_analysis.web.app.load_config',return_value=cfg),patch.object(DailyHarness,'replan',no_search),TestClient(create_app(Path.cwd())) as client:
        response=client.get('/api/weekly-schedule/latest');response.raise_for_status()
        assert response.json()['planner_status']=='ready' and response.json()['planner_version']==217
        response=client.get('/api/settings/race-preparation/status');response.raise_for_status()
        assert response.json()['race_activation_available']
    assert journal_digest(root/'sandbox.sqlite')==digest
    assert journal_digest(source/'sandbox.sqlite')==original_digest
    result=dict(status='race_apply_and_restart_passed',manifest=manifest,live_changed=False,
        prepared_source_unchanged=True,independent_reprice=True,restart_no_search=True,
        decision_index=event['index'],seconds=event['seconds'],coaching_progression=coaching,
        scope='One real copied-app marathon opening; no claim of a new 21-day race replay')
    with (output/'report.json').open('x') as handle:json.dump(result,handle,indent=2)
    print(json.dumps({k:result[k] for k in ('status','seconds','coaching_progression','live_changed')}))


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('prepared',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args();check(args.prepared,args.output)
