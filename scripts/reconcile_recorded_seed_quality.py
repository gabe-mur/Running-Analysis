"""Review/reconcile recorded seed quality; clone first, live only explicitly."""
from argparse import ArgumentParser
from dataclasses import replace
from datetime import datetime,timedelta,timezone
from hashlib import sha256
import fcntl
import json
from pathlib import Path
import shutil
import sqlite3

from run_analysis.config import load_config
from run_analysis.adaptive.adaptive_candidate_pipeline import load_athlete
from run_analysis.adaptive.factory import resume_service
from run_analysis.adaptive.execution_upload_sandbox import encode
from run_analysis.adaptive.seed_reconciliation import reconcile
from run_analysis.adaptive.live_bridge import app_sources,apply_app_inputs,_prior_capture
from run_analysis.adaptive.decision_snapshot import capture_app_inputs,restore
from scripts.research_manifest import freeze_sources


def reconcile_record(database,activity_id,output,*,live=False,refresh=False):
    output.mkdir(parents=True,exist_ok=False)
    config=load_config();at=datetime.now(timezone.utc)
    app=sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True);app.row_factory=sqlite3.Row
    pointer=json.loads(app.execute("SELECT value_json FROM app_state WHERE key='adaptive_runtime'").fetchone()[0])
    live_root=Path(pointer['root'])
    manifest=freeze_sources(output/'sources.zip',config=config,entrypoint=__name__)
    with (live_root/'live-bridge.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        root=live_root if live else output/'runtime'
        if not live:
            shutil.copytree(live_root,root,ignore=shutil.ignore_patterns('*.sqlite','*.sqlite-wal','*.sqlite-shm','live-bridge.lock'))
        # SQLite backup includes committed WAL without copying an inconsistent file.
        with sqlite3.connect((live_root/'sandbox.sqlite').resolve().as_uri()+'?mode=ro',uri=True) as src:
            with sqlite3.connect(output/'before-runtime.sqlite' if live else root/'sandbox.sqlite') as dst:
                src.backup(dst)
        if not live:
            # The coherent copy is now isolated; its search must not hold up
            # live GET/upload requests while exercising the cloned ledger.
            fcntl.flock(lock,fcntl.LOCK_UN)
        service=resume_service(root,config,as_of=at);h=service.harness
        try:
            key=str(activity_id)
            source=app_sources(app)[key]
            tracked=h.sandbox.db.execute('SELECT * FROM adaptive_app_sources WHERE activity_id=?',(key,)).fetchone()
            assert tracked is not None and tracked['physical_key']==key and key in h.seed_keys
            assert tracked['source_sha256']==source['sha256']
            assert sha256(Path(source['path']).read_bytes()).hexdigest()==source['sha256']
            old=next(r for r in h.history if r.key==key)
            inputs=load_athlete(database,config,at)
            new=next(r for r in inputs['history'] if r.key==key)
            phase=next(r for r in inputs['phase_history'] if r.key==key)
            before_history={r.key:encode(r) for r in h.history}
            before_phases={p.key:encode(p) for p in h.phase_history}
            before_issues=[tuple(r) for r in h.sandbox.db.execute('SELECT * FROM sandbox_issues ORDER BY fingerprint')]
            before_decisions=[tuple(r) for r in h.sandbox.db.execute('SELECT * FROM sandbox_decisions ORDER BY sequence')]
            old_exports={p.name:sha256(p.read_bytes()).hexdigest() for p in root.glob('*.json')
                if p.name.startswith(('scenario','snapshot-','decision-'))}
            correction=reconcile(h,new,phase,expected_fingerprint=sha256(encode(old).encode()).hexdigest(),
                available_at=at,source_sha256=source['sha256'],
                reason='Recorded progression finish recognized from original FIT; repair prior easy seed classification')
            assert before_issues==[tuple(r) for r in h.sandbox.db.execute('SELECT * FROM sandbox_issues ORDER BY fingerprint')]
            assert before_decisions==[tuple(r) for r in h.sandbox.db.execute('SELECT * FROM sandbox_decisions ORDER BY sequence')]
            assert len(before_history)==len(h.history)
            assert all(before_history[r.key]==encode(r) for r in h.history if r.key!=key)
            assert all(before_phases[p.key]==encode(p) for p in h.phase_history if p.key!=key)
            keys={int(r[0]):r[1] for r in h.sandbox.db.execute('SELECT * FROM adaptive_app_sources')}
            captured=capture_app_inputs(app,captured_at=at,timezone=config['timezone_default'],activity_keys=keys,
                previous=_prior_capture(h),expectations={r.key:(r.expected_at,r.expectation) for r in h.responses})
            apply_app_inputs(h,captured)
            decision=service.refresh(at+timedelta(microseconds=1),reason='recorded_progression_evidence_reconciliation') if refresh else None
            assert all(sha256((root/name).read_bytes()).hexdigest()==digest for name,digest in old_exports.items())
            for event,content in h.sandbox.decisions():
                restore(content,expected_fingerprint=event['snapshot_fingerprint'])
            report=dict(at=at.isoformat(),live=live,activity_id=activity_id,root=str(root),manifest=manifest,
                correction=correction,before=json.loads(encode(old)),after=json.loads(encode(next(r for r in h.history if r.key==key))),
                original_decisions_preserved=True,original_issues_preserved=True,other_history_preserved=True,
                physical_history_count=len(h.history),app_input_role_check_passed=True,
                decision=decision,original_source_sha256=source['sha256'])
            with (output/'report.json').open('x') as f:f.write(encode(report))
            print(json.dumps(dict(activity_id=activity_id,live=live,correction=correction,
                physical_history_count=len(h.history),before_role=old.role,after_role=new.role,
                unchanged_running_minutes=old.demand[0]==new.demand[0],unchanged_cardio=old.demand[1]==new.demand[1],
                app_input_role_check_passed=True,decision_churn=decision['event']['churn'] if decision else None,
                seconds=decision['event']['seconds'] if decision else None),indent=2))
        finally: h.sandbox.close();app.close()


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--database',type=Path,default=Path('data/run_analysis.sqlite'))
    p.add_argument('--activity-id',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--live',action='store_true');p.add_argument('--refresh',action='store_true')
    args=p.parse_args();reconcile_record(args.database,args.activity_id,args.output,live=args.live,refresh=args.refresh)
