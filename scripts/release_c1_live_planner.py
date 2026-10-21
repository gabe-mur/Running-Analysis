"""Prepare/attach explicit shared-search releases without cold adoption or history rewrites.

Preparation runs the actual live bridge against coherent application/runtime
copies. Attachment is a guarded pointer change, not restoration of an app DB.
No replay executions or user race selection enter this release.
"""
from argparse import ArgumentParser
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import fcntl
import json
import shutil
import sqlite3

from run_analysis.config import load_config, resolve_project_path
from run_analysis.db import connect
from run_analysis.adaptive.live_bridge import ENGINE, schedule, app_sources
from run_analysis.adaptive.decision_snapshot import capture_app_inputs
from run_analysis.adaptive.factory import resume_service
from scripts.replay_adaptive_search import load_decision
from scripts.research_manifest import freeze_sources


def journal_digest(database):
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True) as db:
        return sha256('\n'.join(db.iterdump()).encode()).hexdigest()


def app_evidence(db, config, at):
    # Resolve with the identical cutoff; later uploads/reports/settings must
    # prevent attachment, while read-only cache/weather writes may continue.
    captured = capture_app_inputs(db, captured_at=at,
        timezone=config['timezone_default'],
        activity_keys={int(key):key for key in app_sources(db)})
    state = [(r['key'], r['value_json']) for r in db.execute(
        "SELECT key,value_json FROM app_state WHERE key NOT LIKE 'weekly_schedule%' ORDER BY key")]
    return sha256(repr((app_sources(db), captured, state)).encode()).hexdigest()


def rows(db, table):
    return [tuple(r) for r in db.execute(f'SELECT * FROM {table} ORDER BY rowid')]


def prepare(output, runtime, *, calendar_merge=False):
    config = load_config()
    database = resolve_project_path(Path.cwd(), config['paths']['database'])
    at = datetime.now(timezone.utc)
    output.mkdir(parents=True, exist_ok=False)
    with connect(database) as app:
        pointer = json.loads(app.execute("SELECT value_json FROM app_state WHERE key='adaptive_runtime'").fetchone()[0])
        source = Path(pointer['root'])
        with (source/'live-bridge.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            original_digest = journal_digest(source/'sandbox.sqlite')
            shutil.copytree(source, runtime, ignore=shutil.ignore_patterns(
                '*.sqlite', '*.sqlite-wal', '*.sqlite-shm', '*.lock'))
            with sqlite3.connect((source/'sandbox.sqlite').resolve().as_uri()+'?mode=ro', uri=True) as original, sqlite3.connect(runtime/'sandbox.sqlite') as cloned:
                original.backup(cloned)
                prior = {table:rows(original,table) for table in
                    ('sandbox_decisions','sandbox_issues','sandbox_completions','adaptive_app_sources','stages')}
            with sqlite3.connect(output/'app-before.sqlite') as copied:
                app.backup(copied)
            evidence = app_evidence(app, config, at)
            before = json.loads(app.execute("SELECT value_json FROM app_state WHERE key='weekly_schedule'").fetchone()[0])
    shutil.copy2(output/'app-before.sqlite', output/'app-copy.sqlite')
    manifest = freeze_sources(output/'release.sources.zip', config=config,
        entrypoint=__name__, inputs=(source/'scenario.json',))
    with connect(output/'app-copy.sqlite') as copied:
        copied.execute("UPDATE app_state SET value_json=? WHERE key='adaptive_runtime'",
            (json.dumps(dict(engine=ENGINE, root=str(runtime.resolve()))),))
        copied.commit()
        result = schedule(copied, config, Path.cwd(), as_of=at,
            _upgrade_continuation=not calendar_merge,_upgrade_calendar_merge=calendar_merge)
        expected_version=205 if calendar_merge else 202
        if result.planner_status != 'ready' or result.planner_version != expected_version:
            raise ValueError(f'Live release not ready: {result.planner_review_reason}')
    latest = json.loads(sorted(runtime.glob('decision-*.json'))[-1].read_text())
    checked = load_decision(runtime, latest['index']+1)
    resumed = resume_service(runtime, config, at)
    try:
        h = resumed.harness
        assert h.policy.continuation_distribution == 'c1_v1'
        assert h.policy.complete_first_search and h.policy.final_winner_evaluations == 1000
        assert h.policy.coordinated_final_evaluations == 64
        if calendar_merge:
            assert h.policy.calendar_merge_evaluations==32
            from scripts.run_adaptive_lifecycle import fast_fail_edits
            failures=fast_fail_edits(latest,4)
            if failures:
                raise ValueError('Shared-search release four-day gate: '+json.dumps(failures))
        for table, old in prior.items():
            assert rows(h.sandbox.db, table)[:len(old)] == old, table
        assert ':c1_handoff_v1' in latest['model_version']
        scenario_policy = json.loads((runtime/'scenario.json').read_text())['search_policy']
    finally:
        h.sandbox.close()
    report = dict(as_of=at.isoformat(), source_pointer=pointer, runtime=str(runtime.resolve()),
        source_journal_digest=original_digest, app_evidence_digest=evidence,
        source_activity_set=app_sources_from_copy(output/'app-before.sqlite'),
        manifest=manifest, prepared=True, attached=False,
        original_decisions_issues_completions_preserved=True, independent_reprice=True,
        restart_policy_verified=True, search_policy=scenario_policy,
        model_version=latest['model_version'], snapshot_fingerprint=latest['snapshot_fingerprint'],
        decision_index=latest['index'], seconds=latest['seconds'], evaluations=latest['evaluations'],
        hard_miles=latest['volume'], cost=checked['archived_cost'], churn=latest['churn'],
        before=before, after=result.model_dump(mode='json'),
        race_activation_enabled=False, user_race_selected=False)
    if calendar_merge:
        report.update(release='shared_calendar_merge_v205',four_day_gate_passed=True,
            acceptance='Current-evidence live copy, not proof of zero seven-day churn or global optimality')
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('prepared','attached','runtime','seconds','evaluations','hard_miles','churn')}, indent=2), flush=True)


def app_sources_from_copy(database):
    with connect(database) as db:
        return app_sources(db)


def race_closeout_proof(paths):
    """Require the three actual daily ledgers, not three copies of one result."""
    if len(paths)!=3 or len({p.parent.resolve() for p in paths})!=3:
        raise ValueError('Race closeout requires separate race, omission and general daily audits')
    proof=[];kinds=set()
    for path in paths:
        audit=json.loads(path.read_text());root=path.parent
        reports=sorted(root.glob('resume-*.summary.json'))
        final_path=reports[-1] if reports else root/'lifecycle-summary.json'
        final=json.loads(final_path.read_text())
        four=audit['churn']['4']
        if (audit['status']!='bounded_screen_complete' or audit['completed_days']!=21
                or final['completed_days']!=21 or final['status']!='bounded_screen_complete'
                or audit['failures'] or final['failures'] or not audit['score_replay_verified']
                or four['date_edits'] or four['type_edits']
                or four['largest_absolute_dose_edit']>.5+1e-8
                or audit['decisions']!=len(final['events'])
                or not all(d['eligibility'] for d in audit['decision_diagnostics'])
                or final['search_policy']['calendar_merge_evaluations']!=32
                or final['events'][-1]['refinement']['calendar_merge_doses']['max_full_evaluations']!=32):
            raise ValueError(f'Current-contract race closeout gate did not pass: {path}')
        kind=('omission' if final.get('missed_runs') else
            'race' if final['events'][0].get('race_preparation') else 'general')
        kinds.add(kind)
        proof.append(dict(kind=kind,audit=str(path.resolve()),audit_sha256=sha256(path.read_bytes()).hexdigest(),
            summary=str(final_path.resolve()),summary_sha256=sha256(final_path.read_bytes()).hexdigest(),
            completed_days=21,four_day_churn=four,seven_day_churn=audit['churn']['7'],
            visible_prescription_history={k:{field:value for field,value in v.items() if field!='changed_dates'}
                for k,v in audit.get('visible_prescription_history',{}).items()},
            delivered_funding_ratio=audit['delivered_funding_ratio'],
            driver_recovery=audit.get('driver_recovery')))
    if kinds!={'race','omission','general'}:
        raise ValueError('The three required closeout scenarios are not distinct')
    return proof


def attach(output, *, race_audits=()):
    report = json.loads((output/'report.json').read_text())
    if not report['prepared'] or report['attached']:
        raise ValueError('A prepared, unattached release is required')
    race_proof=race_closeout_proof(race_audits) if report.get('release')=='shared_calendar_merge_v205' else []
    config = load_config()
    source = Path(report['source_pointer']['root'])
    runtime = Path(report['runtime'])
    database = resolve_project_path(Path.cwd(), config['paths']['database'])
    # Source code may not change between verification and attachment.
    from zipfile import ZipFile
    with ZipFile(output/'release.sources.zip') as archive:
        manifest = json.loads(archive.read('manifest.json'))
        archived_activation=archive.read('src/run_analysis/adaptive/race_settings.py')
    attachment_script_changes = []
    for path, digest in manifest['source_sha256'].items():
        if sha256(Path(path).read_bytes()).hexdigest() != digest:
            # An attachment-only repair may follow preparation; all model,
            # service, bridge, search and fixture bytes must remain identical.
            # Archive the actual attachment code separately, never overwrite
            # provenance of the already-completed preparation search.
            activation_path='src/run_analysis/adaptive/race_settings.py'
            activation_only=(race_proof and path==activation_path and
                Path(path).read_bytes()==archived_activation.replace(
                    b'RACE_ACTIVATION_READY = False\n',b'RACE_ACTIVATION_READY = True\n',1))
            if not activation_only and path != str(Path(__file__).resolve().relative_to(Path.cwd())):
                raise ValueError(f'Prepared release source changed: {path}')
            attachment_script_changes.append(path)
    if json.loads(json.dumps(config, default=lambda value:value.isoformat())) != manifest['config']:
        raise ValueError('Prepared release configuration changed')
    checked = load_decision(runtime, report['decision_index']+1)
    assert checked['state'].fingerprint == report['snapshot_fingerprint']
    with (source/'live-bridge.lock').open('a') as lock, connect(database) as db:
        fcntl.flock(lock, fcntl.LOCK_EX)
        db.execute('BEGIN IMMEDIATE')
        pointer = json.loads(db.execute("SELECT value_json FROM app_state WHERE key='adaptive_runtime'").fetchone()[0])
        if pointer != report['source_pointer'] or journal_digest(source/'sandbox.sqlite') != report['source_journal_digest']:
            raise ValueError('Live runtime changed since preparation; do not discard its newer decisions')
        if app_evidence(db, config, datetime.fromisoformat(report['as_of'])) != report['app_evidence_digest']:
            raise ValueError('Live application evidence changed since preparation')
        attachment_manifest = freeze_sources(output/'attachment.sources.zip', config=config,
            entrypoint=__name__, inputs=(output/'report.json',))
        db.execute("UPDATE app_state SET value_json=?,updated_at_utc=? WHERE key='adaptive_runtime'",
            (json.dumps(dict(engine=ENGINE, root=str(runtime))), datetime.now(timezone.utc).isoformat()))
        db.commit()
    report['attached'] = True
    report['attachment_manifest'] = attachment_manifest
    report['attachment_only_source_changes'] = attachment_script_changes
    if race_proof:
        report['race_daily_acceptance']=race_proof
        from run_analysis.adaptive.race_settings import RACE_ACTIVATION_READY
        report['race_activation_enabled']=RACE_ACTIVATION_READY
    report['attached_at'] = datetime.now(timezone.utc).isoformat()
    report['rollback'] = 'Reattach the old pointer only before any new live evidence/decisions; afterward migrate the retained journal, never restore the application database.'
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(attached=True, runtime=str(runtime))), flush=True)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare','attach'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runtime', type=Path)
    parser.add_argument('--calendar-merge',action='store_true',
        help='Prepare the explicit v205 shared search on preserved live copies')
    parser.add_argument('--acceptance-audit',type=Path,action='append',default=[],
        help='Attach v205 only with independently repriced race/omission/general21-day audits')
    args = parser.parse_args()
    if args.mode == 'prepare':
        if args.runtime is None:
            parser.error('prepare requires --runtime')
        prepare(args.output, args.runtime,calendar_merge=args.calendar_merge)
    else:
        attach(args.output,race_audits=args.acceptance_audit)
