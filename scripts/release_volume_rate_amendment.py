"""Prepare a prospective rate change on coherent live copies; attach only guarded proof."""
from argparse import ArgumentParser
from datetime import datetime,timedelta,timezone
from pathlib import Path
import fcntl
import json
import shutil
import sqlite3

from run_analysis.config import load_config,resolve_project_path
from run_analysis.db import connect
from run_analysis.adaptive.live_bridge import schedule,ENGINE
from run_analysis.adaptive.factory import resume_service
from run_analysis.adaptive.new_model_daily_harness import churn
from scripts.replay_adaptive_search import load_decision
from scripts.research_manifest import freeze_sources
from scripts.run_adaptive_lifecycle import fast_fail_edits
from scripts.release_c1_live_planner import journal_digest,app_evidence,app_sources_from_copy,rows,attach


def prepare(output,runtime):
    cfg=load_config(); database=resolve_project_path(Path.cwd(),cfg['paths']['database'])
    at=datetime.now(timezone.utc); output.mkdir(parents=True,exist_ok=False)
    with connect(database) as app:
        pointer=json.loads(app.execute("SELECT value_json FROM app_state WHERE key='adaptive_runtime'").fetchone()[0])
        source=Path(pointer['root'])
        with (source/'live-bridge.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            digest=journal_digest(source/'sandbox.sqlite')
            shutil.copytree(source,runtime,ignore=shutil.ignore_patterns('*.sqlite','*.sqlite-wal','*.sqlite-shm','*.lock'))
            with sqlite3.connect((source/'sandbox.sqlite').resolve().as_uri()+'?mode=ro',uri=True) as original, sqlite3.connect(runtime/'sandbox.sqlite') as cloned:
                original.backup(cloned)
                prior={table:rows(original,table) for table in ('sandbox_decisions','sandbox_issues','sandbox_completions','stages','adaptive_app_sources')}
            with sqlite3.connect(output/'app-before.sqlite') as copied: app.backup(copied)
            evidence=app_evidence(app,cfg,at)
            before=json.loads(app.execute("SELECT value_json FROM app_state WHERE key='weekly_schedule'").fetchone()[0])
    shutil.copy2(output/'app-before.sqlite',output/'app-copy.sqlite')
    manifest=freeze_sources(output/'release.sources.zip',config=cfg,entrypoint=__name__,inputs=(source/'scenario.json',))
    with connect(output/'app-copy.sqlite') as copied:
        copied.execute("UPDATE app_state SET value_json=? WHERE key='adaptive_runtime'",(json.dumps(dict(engine=ENGINE,root=str(runtime.resolve()))),)); copied.commit()
        # The actual bridge captures current evidence/weather before the policy revision.
        baseline=schedule(copied,cfg,Path.cwd(),as_of=at)
        if baseline.planner_status!='ready': raise ValueError('Current live evidence requires review')
    amended_at=at+timedelta(microseconds=1)
    service=resume_service(runtime,cfg,amended_at); original=service.harness.stage
    try:
        def gate(h,preview):
            edits=churn(h.previous,preview['program'],preview['state'].context.decision,h.catalog)
            failures=fast_fail_edits(dict(churn=edits),7)
            if failures: raise ValueError('Rate amendment visible churn gate: '+json.dumps(failures))
        result=service.amend_volume_rate(amended_at,preview_check=gate)
        event=result['event']; h=service.harness
        if event is None: raise ValueError('Rate amendment not accepted: '+result['stage_review']['status'])
        for table,old in prior.items(): assert rows(h.sandbox.db,table)[:len(old)]==old,table
        assert h.stage.issued_at==original.issued_at and h.stage.review_at==original.review_at
        assert h.stage.weekly_equivalent_miles==original.weekly_equivalent_miles and h.stage.long_miles==original.long_miles
        checked=load_decision(runtime,event['index']+1)
        # Update the app COPY through the real bridge without another search.
        with connect(output/'app-copy.sqlite') as copied:
            after=schedule(copied,cfg,Path.cwd(),as_of=amended_at+timedelta(microseconds=1))
        assert after.generated_at==datetime.fromisoformat(event['as_of'])
        report=dict(as_of=at.isoformat(),source_pointer=pointer,runtime=str(runtime.resolve()),
            source_journal_digest=digest,app_evidence_digest=evidence,
            source_activity_set=app_sources_from_copy(output/'app-before.sqlite'),manifest=manifest,
            prepared=True,attached=False,original_decisions_issues_completions_preserved=True,
            independent_reprice=True,snapshot_fingerprint=checked['state'].fingerprint,
            decision_index=event['index'],seconds=event['seconds'],evaluations=event['evaluations'],
            churn=event['churn'],current_volume=event['progression_target']['planning_volume_per_7'],
            old_phase_goal=original.policy.volume_goal_miles,new_phase_goal=h.stage.policy.volume_goal_miles,
            original_review_at=original.review_at.isoformat(),new_review_at=h.stage.review_at.isoformat(),
            long_miles=h.stage.long_miles,delivery_confirmed=False,before=before,after=after.model_dump(mode='json'),
            scope='One prospective policy amendment, not daily stability or phase-completion acceptance')
    finally:
        service.harness.sandbox.close()
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('prepared','seconds','evaluations','churn','current_volume','new_phase_goal','new_review_at')},indent=2),flush=True)


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','attach'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--runtime',type=Path)
    args=parser.parse_args()
    if args.mode=='prepare':
        if args.runtime is None: parser.error('prepare requires --runtime')
        prepare(args.output,args.runtime)
    else: attach(args.output)
