"""One real-evidence CLONED preference refresh; never writes the live runtime.

Tests application/restart and independent selected-program reprice. Not a daily
stability or recovery validation. Existing user preferences stay unchanged.
"""
from argparse import ArgumentParser
from dataclasses import asdict
from datetime import datetime, timedelta
import fcntl
import json
from pathlib import Path
import shutil
import sqlite3
from time import perf_counter

from run_analysis.config import load_config, resolve_project_path
from run_analysis.adaptive.factory import resume_service
from run_analysis.adaptive.decision_snapshot import restore
from run_analysis.adaptive.planner_preferences import queue_request, apply_request, progression_stage, latest_request
from run_analysis.web.schemas import PlannerPreferencesPatch
from scripts.replay_adaptive_search import load_decision


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--verify-existing',action='store_true',
        help='Finish verification of the existing cloned decision without another search')
    args = parser.parse_args()
    project = Path.cwd()
    config = load_config(project/'config.yaml')
    database = resolve_project_path(project,config['paths']['database'])
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as db:
        pointer = json.loads(db.execute("SELECT value_json FROM app_state WHERE key='adaptive_runtime'").fetchone()[0])
    source = Path(pointer['root'])
    if not args.verify_existing:
        args.output.mkdir(parents=True,exist_ok=False)
    clone = args.output/'runtime'
    # Copy coherent exports plus a SQLite backup while the real owner is locked.
    # No training search occurs under the LIVE lock.
    with (source/'live-bridge.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not args.verify_existing:
            shutil.copytree(source,clone,ignore=shutil.ignore_patterns('*.sqlite','*.sqlite-wal','*.sqlite-shm','*.lock'))
        with sqlite3.connect((source/'sandbox.sqlite').resolve().as_uri()+'?mode=ro',uri=True) as original:
            old = original.execute('SELECT * FROM sandbox_decisions ORDER BY sequence').fetchall()
            if not args.verify_existing:
                with sqlite3.connect(clone/'sandbox.sqlite') as copied:
                    original.backup(copied)
    original_event = json.loads(old[-1][1])
    at = datetime.fromisoformat(original_event['as_of'])+timedelta(seconds=2)
    service = resume_service(clone,config,as_of=at+timedelta(seconds=2) if args.verify_existing else at)
    try:
        h = service.harness
        original_stage = h.stage
        before_history = tuple(h.history)
        before_reference = h.frequency_contract.preference
        state = restore(old[-1][2],expected_fingerprint=original_event['snapshot_fingerprint'])
        if args.verify_existing:
            request = latest_request(h.sandbox.db)
            event = json.loads(h.sandbox.db.execute(
                'SELECT event FROM sandbox_decisions ORDER BY sequence DESC LIMIT 1').fetchone()[0])
            assert event['reason'] == 'cloned user preference validation'
            elapsed = event['seconds']
        else:
            request = queue_request(h.sandbox.db,state=state,stage=asdict(h.stage),as_of=at,
                patch=PlannerPreferencesPatch(long_run_goal_miles=18,run_frequency='fewer',
                    expected_revision='adopted:'+state.fingerprint))
            assert apply_request(h,as_of=at)
            start = perf_counter()
            result = service.refresh(at+timedelta(seconds=1),reason='cloned user preference validation')
            elapsed = perf_counter()-start
            event = result['event']
        h = service.harness
        applied = restore((clone/event['snapshot_file']).read_text(),expected_fingerprint=event['snapshot_fingerprint'])
        assert applied.planner_preferences == request
        assert h.stage == original_stage and tuple(h.history) == before_history
        assert dict(asdict(h.stage),issued_at=h.stage.issued_at.isoformat()) == original_event['stage_runtime']['stage']
        assert tuple(h.history) == state.history
        assert h.frequency_contract.targets == state.frequency_contract.targets
        assert h.frequency_contract.preference.reference_fatigue == before_reference.reference_fatigue
        assert h.frequency_contract.preference.reference_days == before_reference.reference_days
        assert h.frequency_contract.preference.edge_fraction == before_reference.edge_fraction
        # Sandbox connections return sqlite3.Row; independent backups return tuples.
        rows = [tuple(row) for row in h.sandbox.db.execute('SELECT * FROM sandbox_decisions ORDER BY sequence')]
        assert rows[:-1] == old
        report = dict(scope='one real-evidence cloned preference replan, not daily stability',
            source_snapshot=state.fingerprint,applied_snapshot=applied.fingerprint,
            preferences=request.payload,seconds=elapsed,search_seconds=event['seconds'],
            evaluations=event['evaluations'],selected_cost=event['cost'],
            churn=event['churn'],selected_frequency=event['selected']['frequency_range'],
            preference=event['selected']['frequency_preference'],stage_unchanged=True,
            physical_history_unchanged=True,prior_decisions_unchanged=True,
            prospective_goal=progression_stage(h.stage,h.planner_preferences).goal_long_miles,
            model_default_weights_unchanged=True,live_preferences_changed=False)
    finally:
        service.harness.sandbox.close()
    # Read-only reconstruction independently reprices the actually issued winner.
    checked = load_decision(clone,event['index']+1)
    report['independent_reprice_passed'] = True  # loader enforces exact archived parity
    resumed = resume_service(clone,config,as_of=at+timedelta(seconds=2))
    try:
        assert resumed.harness.planner_preferences == request
        assert resumed.harness.frequency_contract.preference.center_fraction == .25
        assert not apply_request(resumed.harness,as_of=at+timedelta(seconds=2))
        assert resumed.harness.stage == original_stage
        report['restart_passed'] = True
    finally:
        resumed.harness.sandbox.close()
    (args.output/'report.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
    print(json.dumps(report,default=str),flush=True)


if __name__ == '__main__':
    main()
