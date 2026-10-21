"""One copied-live quality preference search; no live settings/plan writes."""
from argparse import ArgumentParser
from dataclasses import asdict
from datetime import datetime,timedelta
import fcntl
import json
from pathlib import Path
import shutil
import sqlite3
from time import perf_counter

from run_analysis.config import load_config,resolve_project_path
from run_analysis.adaptive.factory import resume_service
from run_analysis.adaptive.quality_choices import allowed,dose_type
from run_analysis.adaptive.web_materialization import render
from run_analysis.web.schemas import QualitySessionType
from scripts.replay_adaptive_search import load_decision
from scripts.research_manifest import freeze_sources


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    p.add_argument('--only',choices=[k.value for k in QualitySessionType])
    args=p.parse_args()
    config=load_config()
    database=resolve_project_path(Path.cwd(),config['paths']['database'])
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as db:
        source=Path(json.loads(db.execute("SELECT value_json FROM app_state WHERE key='adaptive_runtime'").fetchone()[0])['root'])
    args.output.mkdir(parents=True,exist_ok=False)
    clone=args.output/'runtime'
    with (source/'live-bridge.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        shutil.copytree(source,clone,ignore=shutil.ignore_patterns('*.sqlite','*.sqlite-wal','*.sqlite-shm','*.lock'))
        with sqlite3.connect((source/'sandbox.sqlite').resolve().as_uri()+'?mode=ro',uri=True) as original:
            old=original.execute('SELECT * FROM sandbox_decisions ORDER BY sequence').fetchall()
            with sqlite3.connect(clone/'sandbox.sqlite') as copied:original.backup(copied)
    prior=json.loads(old[-1][1]); at=datetime.fromisoformat(prior['as_of'])+timedelta(seconds=2)
    if args.only:
        config={**config,'coaching':{**config['coaching'],'quality_sessions':{
            k.value:k.value==args.only for k in QualitySessionType}}}
    manifest=freeze_sources(args.output/'sources.zip',config=config,entrypoint=__name__,
        inputs=(max(clone.glob('decision-*.json')),clone/'scenario.json'))
    service=resume_service(clone,config,as_of=at)
    try:
        before=tuple(service.harness.history); stage=service.harness.stage
        stamp=perf_counter()
        event=service.refresh(at,reason='copied quality preferences validation')['event']
        elapsed=perf_counter()-stamp
        h=service.harness
        assert tuple(h.history)==before and h.stage==stage
        assert [tuple(r) for r in h.sandbox.db.execute('SELECT * FROM sandbox_decisions ORDER BY sequence')][:-1]==old
        assert all(allowed(h.catalog[k],h.coaching_contracts[k],config) for _,k in event['program'])
        quality=[dict(day=d,family=dose_type(h.catalog[k],h.coaching_contracts[k]),miles=h.catalog[k].miles,
            work_seconds=sum(s for phase,s in h.catalog[k].phases if phase=='work'))
            for d,k in event['program'] if h.catalog[k].role=='quality']
        assert quality
        if args.only:assert all(r['family']==args.only for r in quality)
        schedule=render(h,event,config)
        report=dict(seconds=elapsed,search_seconds=event['seconds'],quality=quality,
            manifest=manifest,
            preferences=config['coaching']['quality_sessions'],churn=event['churn'],
            hard_21_miles=event['volume'],history_unchanged=True,stage_unchanged=True,
            prior_decisions_unchanged=True,scope='One copied-live 21-day projection, not daily stability or physiological validation',
            schedule=schedule.model_dump(mode='json'))
        definitions={k:asdict(d) for k,d in h.catalog.items() if d.workout_definition}
    finally:service.harness.sandbox.close()
    load_decision(clone,event['index']+1)
    report['independent_reprice_passed']=True
    reopened=resume_service(clone,config,as_of=at+timedelta(seconds=1))
    try:
        assert definitions=={k:asdict(d) for k,d in reopened.harness.catalog.items() if d.workout_definition}
        report['restart_passed']=True
    finally:reopened.harness.sandbox.close()
    (args.output/'report.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='schedule'},default=str),flush=True)


if __name__=='__main__':main()
