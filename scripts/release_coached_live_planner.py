"""Prepare and attach coached v206 on preserved real app/runtime evidence.

No replay activities or stage transplantation. The native input supplies only
previously admitted immutable commands and measured seed profiles. Publication,
upload reconciliation, weather, inputs and search use the actual live bridge.
"""
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime,timedelta,timezone
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile
import fcntl
import json
import shutil
import sqlite3

import yaml

from run_analysis.config import load_config,resolve_project_path
from run_analysis.db import connect
from run_analysis.adaptive.live_bridge import ENGINE,schedule
from run_analysis.adaptive.factory import resume_service
from run_analysis.adaptive.new_model_daily_harness import DailyHarness
from run_analysis.adaptive.coached_runtime import enable
from run_analysis.adaptive.quality_forecast_runtime import VERSION
from scripts.release_c1_live_planner import journal_digest,app_evidence,rows
from scripts.replay_adaptive_search import load_decision
from scripts.research_manifest import freeze_sources


def install_native(h,native,at):
    event=json.loads(max(native.glob('decision-*.json')).read_text())
    n=DailyHarness.resume(native,deepcopy(h.config),as_of=datetime.fromisoformat(event['as_of']))
    try:
        if (h.stage!=n.stage or h.training_policy!=n.training_policy or h.scales!=n.scales
                or h.demand_scenario!=n.demand_scenario or h.quality_runtime is not None):
            raise ValueError('Native commands cannot replace the live stage, normalization or existing quality runtime')
        additions={k:d for k,d in n.catalog.items() if d.workout_definition and
            d.workout_definition.get('calendar_integration')==VERSION}
        if not additions:raise ValueError('No admitted native quality commands')
        for k,d in additions.items():
            if k in h.catalog and h.catalog[k]!=d:
                raise ValueError('Cannot overwrite an already issued command identity')
        h.catalog={**h.catalog,**additions}
        h.coaching_contracts={**h.coaching_contracts,**{k:n.coaching_contracts[k] for k in additions}}
        h.quality_runtime=deepcopy(n.quality_runtime)
        from run_analysis.adaptive.quality_catalogue_runtime import json_value
        if (h.quality_runtime['evidence']['config']!=json_value({k:h.config[k] for k in ('resting_hr','max_hr','zones')})
                or datetime.fromisoformat(h.quality_runtime['evidence']['available_at'])>at):
            raise ValueError('Native evidence normalization or causal cutoff differs from current inputs')
        enable(h)
    finally:n.sandbox.close()


def certificate(h,database):
    return dict(version='coached_live_v206',database=str(database.resolve()),
        coaching_base_key=h.coaching_base.key,validated_app_copy=True,
        scope='Explicit copied-app bridge, restart and route acceptance; not physiological clearance')


def prepare(output,runtime,native):
    config=load_config();database=resolve_project_path(Path.cwd(),config['paths']['database'])
    at=datetime.now(timezone.utc)
    output.mkdir(parents=True,exist_ok=False)
    with connect(database) as app:
        pointer=json.loads(app.execute("SELECT value_json FROM app_state WHERE key='adaptive_runtime'").fetchone()[0])
        source=Path(pointer['root'])
        with (source/'live-bridge.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            original_digest=journal_digest(source/'sandbox.sqlite')
            shutil.copytree(source,runtime,ignore=shutil.ignore_patterns('*.sqlite','*.sqlite-wal','*.sqlite-shm','*.lock'))
            with sqlite3.connect((source/'sandbox.sqlite').resolve().as_uri()+'?mode=ro',uri=True) as original:
                with sqlite3.connect(runtime/'sandbox.sqlite') as copied:original.backup(copied)
                prior={t:rows(original,t) for t in ('sandbox_decisions','sandbox_issues','sandbox_completions',
                    'adaptive_app_sources','stages')}
            with sqlite3.connect(output/'app-before.sqlite') as copied:app.backup(copied)
            evidence=app_evidence(app,config,at)
    shutil.copy2(output/'app-before.sqlite',output/'app-copy.sqlite')
    manifest=freeze_sources(output/'release.sources.zip',config=config,entrypoint=__name__,
        inputs=(source/'scenario.json',native/'scenario.json',max(native.glob('decision-*.json'))))
    with connect(output/'app-copy.sqlite') as copied:
        copied.execute("UPDATE app_state SET value_json=? WHERE key='adaptive_runtime'",
            (json.dumps(dict(engine=ENGINE,root=str(runtime.resolve()))),));copied.commit()
        def upgrade(h,when):
            install_native(h,native,when)
        result=schedule(copied,config,Path.cwd(),as_of=at,_upgrade_coaching=upgrade)
        if result.planner_status!='ready' or result.planner_version!=206:
            raise ValueError('Copied-app release not ready: '+str(result.planner_review_reason))
    latest=json.loads(max(runtime.glob('decision-*.json')).read_text())
    checked=load_decision(runtime,latest['index']+1)
    # New family selection is the explicit one-time release; existing dates and
    # roles must still be preserved through the near-term handoff.
    from scripts.run_adaptive_lifecycle import fast_fail_edits
    edits=fast_fail_edits(latest,4)
    if edits:raise ValueError('Coached release four-day gate: '+json.dumps(edits))
    service=resume_service(runtime,config,as_of=at)
    try:
        h=service.harness
        for table,old in prior.items():
            if rows(h.sandbox.db,table)[:len(old)]!=old:raise ValueError('Original journal changed: '+table)
        base=h.coaching_base.key
        if base!=checked['state'].quality_coaching.key:
            raise ValueError('Opening coaching ownership differs from committed snapshot')
        if h.policy.evaluations!=10000 or h.policy.calendar_merge_evaluations!=32:
            raise ValueError('Live search allowances changed')
        (runtime/'coached-release.json').write_text(json.dumps(certificate(h,output/'app-copy.sqlite'),indent=2))
    finally:service.harness.sandbox.close()
    report=dict(prepared=True,attached=False,validated=False,release='coached_live_v206',as_of=at.isoformat(),
        source_pointer=pointer,source_journal_digest=original_digest,app_evidence_digest=evidence,
        runtime=str(runtime.resolve()),manifest=manifest,native_input=str(native.resolve()),
        decision_index=latest['index'],snapshot_fingerprint=latest['snapshot_fingerprint'],
        coaching_base_key=base,original_journal_preserved=True,replay_completions_imported=False,
        seconds=latest['seconds'],miles21=latest['volume'],churn=latest['churn'],
        independent_reprice=True,four_day_gate_passed=True,after=result.model_dump(mode='json'))
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({k:report[k] for k in ('prepared','validated','attached','seconds','miles21','churn')},indent=2),flush=True)


def verify_sources(output,report):
    archive_path=output/'release.sources.zip'
    if sha256(archive_path.read_bytes()).hexdigest()!=report['manifest']['sha256']:
        raise ValueError('Prepared source archive changed')
    with ZipFile(archive_path) as archive:
        manifest=json.loads(archive.read('manifest.json'))
    for path,digest in manifest['source_sha256'].items():
        if sha256(Path(path).read_bytes()).hexdigest()!=digest:
            raise ValueError('Prepared release source changed: '+path)
    if json.loads(json.dumps(load_config(),default=lambda x:x.isoformat()))!=manifest['config']:
        raise ValueError('Prepared release configuration changed')
    checked=load_decision(Path(report['runtime']),report['decision_index']+1)
    if checked['state'].fingerprint!=report['snapshot_fingerprint']:
        raise ValueError('Prepared release snapshot changed')


def validate(output):
    from fastapi.testclient import TestClient
    from run_analysis.web.app import create_app
    report=json.loads((output/'report.json').read_text())
    verify_sources(output,report)
    config=load_config();config['paths']['database']=str((output/'app-copy.sqlite').resolve())
    for name in ('uploads','outputs'):
        if name in config['paths']:config['paths'][name]=str((output/name).resolve())
    config_path=output/'app-copy.config.yaml'
    config_path.write_text(yaml.safe_dump(config))
    root=Path(report['runtime']);before=journal_digest(root/'sandbox.sqlite')
    calls=[]
    def no_search(*args,**kwargs):raise AssertionError('Unchanged copied-app request unexpectedly reran the planner')
    # Actual route handlers, copied database and production guard. No forecast,
    # scoring, receipt, or source resolver is mocked. Unchanged requests must
    # render the durable decision without spending another search budget.
    with patch.object(DailyHarness,'replan',no_search),TestClient(create_app(Path.cwd(),config_path)) as client:
        for url in ('/api/health','/api/settings','/api/settings/race-preparation/status'):
            response=client.get(url);response.raise_for_status();calls.append(dict(route=url,status=response.status_code))
        response=client.get('/api/weekly-schedule/latest');response.raise_for_status()
        if response.json()['planner_version']!=206 or response.json()['planner_status']!='ready':
            raise ValueError('Restarted app route does not render coached v206')
        calls.append(dict(route='/api/weekly-schedule/latest',status=response.status_code))
        with connect(output/'app-copy.sqlite') as app:
            source=app.execute('''SELECT s.path FROM activities a JOIN activity_sources l
                ON l.activity_id=a.id AND l.is_primary=1 JOIN source_files s ON s.id=l.source_file_id
                WHERE a.sport='Running' ORDER BY a.start_time_utc DESC LIMIT 1''').fetchone()[0]
        path=Path(source)
        response=client.post('/api/uploads',files={'files':(path.name,path.read_bytes(),'application/octet-stream')})
        response.raise_for_status();upload=response.json()
        if (not upload['files'] or any(f['status']!='unchanged' for f in upload['files'])
                or any(s['status'] not in ('complete','skipped') for s in upload['stages'])):
            raise ValueError('Original-file duplicate upload did not complete unchanged')
        calls.append(dict(route='/api/uploads',status=response.status_code,result=upload))
    if journal_digest(root/'sandbox.sqlite')!=before:
        raise ValueError('Unchanged route or duplicate upload changed the decision/completion journal')
    report.update(validated=True,route_checks=calls,restart_and_duplicate_exact=True,
        limitation='Duplicate real FIT/upload and unchanged routes; new measured uploads retain prior AS200 seam validation')
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(dict(validated=True,route_checks=[(r['route'],r['status']) for r in calls]),indent=2),flush=True)


def validate_race(output):
    """An actual race Apply/search/restart on a separate app/runtime copy.

    No race request enters the release or real app. Phase admission controls
    cover all four distances; this copied 10k opening is integration acceptance,
    not a new race-specific daily stability or physiological accuracy claim.
    """
    from fastapi.testclient import TestClient
    from run_analysis.web.app import create_app
    from run_analysis.adaptive.race_accounting import allowed_dose
    report=json.loads((output/'report.json').read_text());verify_sources(output,report)
    if not report['validated']:raise ValueError('App-route validation required first')
    source=Path(report['runtime']);before=journal_digest(source/'sandbox.sqlite')
    probe=output/'race-check';probe.mkdir(exist_ok=False);runtime=probe/'runtime'
    shutil.copytree(source,runtime,ignore=shutil.ignore_patterns('*.sqlite','*.sqlite-wal','*.sqlite-shm','*.lock'))
    for original,destination in ((source/'sandbox.sqlite',runtime/'sandbox.sqlite'),
            (output/'app-copy.sqlite',probe/'app.sqlite')):
        with sqlite3.connect(original.resolve().as_uri()+'?mode=ro',uri=True) as db:
            with sqlite3.connect(destination) as copy:db.backup(copy)
    config=load_config();config['paths']['database']=str((probe/'app.sqlite').resolve())
    path=probe/'config.yaml';path.write_text(yaml.safe_dump(config))
    with connect(probe/'app.sqlite') as app:
        app.execute("UPDATE app_state SET value_json=? WHERE key='adaptive_runtime'",
            (json.dumps(dict(engine=ENGINE,root=str(runtime.resolve()))),));app.commit()
        activities=app.execute('SELECT COUNT(*) FROM activities').fetchone()[0]
    release=json.loads((runtime/'coached-release.json').read_text())
    release['database']=config['paths']['database']
    (runtime/'coached-release.json').write_text(json.dumps(release,indent=2))
    goal=dict(distance='10k',race_date=(datetime.now(timezone.utc).date()+timedelta(days=90)).isoformat(),
        preparation_mode='balanced')
    with TestClient(create_app(Path.cwd(),path)) as client:
        status=client.get('/api/settings/race-preparation/status');status.raise_for_status()
        assert status.json()['race_activation_available'],status.text
        requested=client.post('/api/settings/race-preparation/apply',json=dict(
            goal=goal,expected_revision=status.json()['expected_revision']))
        requested.raise_for_status()
        assert requested.json()['saved'] and not requested.json()['calendar_changed'],requested.text
        response=client.get('/api/weekly-schedule/latest');response.raise_for_status()
        assert response.json()['planner_status']=='ready' and response.json()['planner_version']==206,response.text
    latest=json.loads(max(runtime.glob('decision-*.json')).read_text())
    checked=load_decision(runtime,latest['index']+1)
    service=resume_service(runtime,config,as_of=datetime.fromisoformat(latest['as_of']))
    try:
        h=service.harness
        assert h.coaching_base.key==report['coaching_base_key']
        assert h.race_goal.payload['distance']==goal['distance']
        assert all(allowed_dose(h.race_goal,datetime.fromisoformat(day).date(),
            h.catalog[key],h.stage,h.race_event) for day,key in latest['program'])
    finally:service.harness.sandbox.close()
    digest=journal_digest(runtime/'sandbox.sqlite')
    def no_search(*a,**kw):raise AssertionError('Race app restart unexpectedly searched again')
    with patch.object(DailyHarness,'replan',no_search),TestClient(create_app(Path.cwd(),path)) as client:
        restarted=client.get('/api/weekly-schedule/latest');restarted.raise_for_status()
        status=client.get('/api/settings/race-preparation/status');status.raise_for_status()
        assert restarted.json()['planner_status']=='ready' and status.json()['applied']['distance']=='10k'
    with connect(probe/'app.sqlite') as app:
        assert app.execute('SELECT COUNT(*) FROM activities').fetchone()[0]==activities
    assert journal_digest(runtime/'sandbox.sqlite')==digest
    assert journal_digest(source/'sandbox.sqlite')==before
    report['race_check']=dict(passed=True,goal=goal,runtime=str(runtime.resolve()),
        decision_index=latest['index'],snapshot_fingerprint=checked['state'].fingerprint,
        seconds=latest['seconds'],independent_reprice=True,dated_admission=True,
        unchanged_restart=True,release_journal_unchanged=True,activities_unchanged=True,
        scope='Actual copied 10k Apply/opening/restart; not a combined race/coaching daily stability screen')
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report['race_check'],indent=2),flush=True)


def attach(output):
    report=json.loads((output/'report.json').read_text())
    if (not report['prepared'] or not report['validated'] or report['attached']
            or not report.get('race_check',{}).get('passed')):
        raise ValueError('Verified unattached copied-app release required')
    verify_sources(output,report)
    config=load_config();database=resolve_project_path(Path.cwd(),config['paths']['database'])
    runtime=Path(report['runtime']);source=Path(report['source_pointer']['root'])
    with (source/'live-bridge.lock').open('a') as lock,connect(database) as app:
        fcntl.flock(lock,fcntl.LOCK_EX);app.execute('BEGIN IMMEDIATE')
        pointer=json.loads(app.execute("SELECT value_json FROM app_state WHERE key='adaptive_runtime'").fetchone()[0])
        if pointer!=report['source_pointer'] or journal_digest(source/'sandbox.sqlite')!=report['source_journal_digest']:
            raise ValueError('Live runtime changed; do not discard newer decisions')
        if app_evidence(app,config,datetime.fromisoformat(report['as_of']))!=report['app_evidence_digest']:
            raise ValueError('Live evidence changed since preparation')
        released=json.loads((runtime/'coached-release.json').read_text())
        if released['coaching_base_key']!=report['coaching_base_key'] or not released['validated_app_copy']:
            raise ValueError('Release certificate changed')
        released['database']=str(database.resolve())
        released['acceptance_report_sha256']=sha256((output/'report.json').read_bytes()).hexdigest()
        (runtime/'coached-release.json').write_text(json.dumps(released,indent=2))
        app.execute("UPDATE app_state SET value_json=?,updated_at_utc=? WHERE key='adaptive_runtime'",
            (json.dumps(dict(engine=ENGINE,root=str(runtime.resolve()))),datetime.now(timezone.utc).isoformat()))
        app.commit()
    report.update(attached=True,attached_at=datetime.now(timezone.utc).isoformat(),
        rollback='Old pointer only before new live evidence; never restore the application database over uploads')
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(dict(attached=True,runtime=str(runtime),version=206)),flush=True)


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('mode',choices=('prepare','validate','race-check','attach'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--runtime',type=Path);p.add_argument('--native',type=Path)
    a=p.parse_args()
    if a.mode=='prepare':
        if a.runtime is None or a.native is None:p.error('prepare requires --runtime and --native')
        prepare(a.output,a.runtime,a.native)
    elif a.mode=='validate':validate(a.output)
    elif a.mode=='race-check':validate_race(a.output)
    else:attach(a.output)
