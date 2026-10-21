"""Release the tested V7 build through coherent copies and guarded attachment.

Never copy replay completions to live, replace its app database, or change HR
normalization. The fast smoothing probe is not part of this release.
"""
from argparse import ArgumentParser
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import fcntl
import json
import shutil
import sqlite3
from unittest.mock import patch
from zipfile import ZipFile

from run_analysis.config import load_config, resolve_project_path
from run_analysis.db import connect
from run_analysis.adaptive.factory import resume_service
from run_analysis.adaptive.hr_reference import frozen_live_config
from run_analysis.adaptive.live_bridge import (ENGINE, continuation_release_policy,
    persist_release_policy, runtime_version, schedule)
from scripts.release_c1_live_planner import (app_evidence, journal_digest, rows,
    app_sources_from_copy, attach)
from scripts.replay_adaptive_search import load_decision
from scripts.research_manifest import freeze_sources
from scripts.run_adaptive_lifecycle import fast_fail_edits


def apply_build(h, at):
    """Explicit copy-only hook; preserve physical history and accepted targets."""
    if (h.policy != continuation_release_policy(h.policy)
            or h.policy.calendar_merge_evaluations != 32
            or h.policy.progression_admission != 'observed_rolling_v2'
            or h.policy.progression_credit != 'actual_distance_v2'
            or h.stage is None or h.stage.policy.volume_path != 'gradual'
            or h.stage.policy.growth_focus != 'easy_volume'
            or getattr(h, 'race_goal', None) is not None):
        raise ValueError('Release requires the accepted general-fitness live contracts')
    if h.policy.general_progression == 'continuous_general_v7':
        raise ValueError('V7 is already adopted; do not reset its measurement epoch')
    if h.frequency_contract is None or h.frequency_contract.version != 'frequency_midpoint_v3':
        raise ValueError('Accepted frequency contract required')
    if h.frequency_contract.preference.edge_fraction not in (.02, .03):
        raise ValueError('Unexpected frequency weight requires explicit reconciliation')
    h.policy = replace(h.policy, general_progression='continuous_general_v7',
        volume_shortfall_policy='forgone_issued_v2')
    h.frequency_contract = replace(h.frequency_contract,
        preference=replace(h.frequency_contract.preference, edge_fraction=.03))


def acceptance(root):
    audit_path = root/'perfect-21d/daily-audit.json'
    audit = json.loads(audit_path.read_text())
    summary = json.loads((root/'perfect-21d/lifecycle-summary.json').read_text())
    four = audit['churn']['4']
    if (audit['status'] != 'bounded_screen_complete' or audit['completed_days'] != 21
            or summary['status'] != 'bounded_screen_complete' or summary['completed_days'] != 21
            or audit['failures'] or summary['failures'] or not audit['score_replay_verified']
            or four['date_edits'] or four['type_edits']
            or four['largest_absolute_dose_edit'] > .5 + 1e-8):
        raise ValueError('Required perfect-adherence 21-day acceptance is missing')
    parity = root/'final-signal-parity/measurement.json'
    check = json.loads(parity.read_text())
    if len(check['rows']) != audit['decisions'] or not all(
            r['original_winner_reproduced'] and r['old_normalized_volume'] == r['new_measured_volume']
            for r in check['rows']):
        raise ValueError('Final measurement/source equivalence is missing')
    archive = root/'final-signal-parity/measurement.sources.zip'
    with ZipFile(archive) as bundle:
        manifest = json.loads(bundle.read('manifest.json'))
    changed = [p for p, digest in manifest['source_sha256'].items()
               if p.startswith('src/') and sha256(Path(p).read_bytes()).hexdigest() != digest]
    if changed:
        raise ValueError('Model sources differ from acceptance: '+repr(changed))
    return dict(audit=str(audit_path.resolve()), audit_sha256=sha256(audit_path.read_bytes()).hexdigest(),
        parity=str(parity.resolve()), parity_sha256=sha256(parity.read_bytes()).hexdigest(),
        production_sources_match=True, completed_days=21, four_day_churn=four,
        scope='Perfect-adherence sandbox plus final source parity; not imperfect adherence or race migration')


def prepare(output, runtime, proof):
    accepted = acceptance(proof)
    config = load_config()
    database = resolve_project_path(Path.cwd(), config['paths']['database'])
    at = datetime.now(timezone.utc)
    output.mkdir(parents=True, exist_ok=False)
    with connect(database) as app:
        pointer = json.loads(app.execute("SELECT value_json FROM app_state WHERE key='adaptive_runtime'").fetchone()[0])
        source = Path(pointer['root'])
        with (source/'live-bridge.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            digest = journal_digest(source/'sandbox.sqlite')
            shutil.copytree(source, runtime, ignore=shutil.ignore_patterns(
                '*.sqlite', '*.sqlite-wal', '*.sqlite-shm', '*.lock'))
            with sqlite3.connect((source/'sandbox.sqlite').resolve().as_uri()+'?mode=ro', uri=True) as original:
                with sqlite3.connect(runtime/'sandbox.sqlite') as copied:
                    original.backup(copied)
                prior = {t: rows(original, t) for t in ('sandbox_decisions', 'sandbox_issues',
                    'sandbox_completions', 'adaptive_app_sources', 'stages')}
            with sqlite3.connect(output/'app-before.sqlite') as copied:
                app.backup(copied)
            evidence = app_evidence(app, config, at)
            before = json.loads(app.execute("SELECT value_json FROM app_state WHERE key='weekly_schedule'").fetchone()[0])
    shutil.copy2(output/'app-before.sqlite', output/'app-copy.sqlite')
    manifest = freeze_sources(output/'release.sources.zip', config=config, entrypoint=__name__,
        inputs=(source/'scenario.json', Path(accepted['audit']), Path(accepted['parity'])))
    with connect(output/'app-copy.sqlite') as app:
        app.execute("UPDATE app_state SET value_json=? WHERE key='adaptive_runtime'",
            (json.dumps(dict(engine=ENGINE, root=str(runtime.resolve()))),))
        app.commit()
        result = schedule(app, config, Path.cwd(), as_of=at, _upgrade_coaching=apply_build)
    if result.planner_status != 'ready' or result.planner_version != 217:
        raise ValueError('Copied live bridge did not produce V7: '+str(result.planner_review_reason))
    event = json.loads(max(runtime.glob('decision-*.json')).read_text())
    failures = fast_fail_edits(event, 4)
    if failures:
        raise ValueError('Release handoff four-day gate failed: '+json.dumps(failures))
    checked = load_decision(runtime, event['index']+1)
    if not checked['evaluate_details'](tuple(map(tuple, event['program'])))['decision_eligible']:
        raise ValueError('Prepared winner is not independently eligible')
    cfg = frozen_live_config(config, root=runtime)
    service = resume_service(runtime, cfg, at)
    try:
        h = service.harness
        for table, old in prior.items():
            if rows(h.sandbox.db, table)[:len(old)] != old:
                raise ValueError('Original journal changed: '+table)
        if runtime_version(h) != 217 or h.frequency_contract.preference.edge_fraction != .03:
            raise ValueError('Restart did not retain the tested policy')
        persist_release_policy(h)
        controller = h.general_progression_state
    finally:
        service.harness.sandbox.close()
    # Re-enter the real bridge with the exact same evidence/time. No second
    # search is allowed; this also materializes the app copy's saved schedule.
    from run_analysis.adaptive.new_model_daily_harness import DailyHarness
    def no_search(*args, **kwargs):
        raise AssertionError('Unchanged bridge repeated a planner search')
    with patch.object(DailyHarness, 'replan', no_search), connect(output/'app-copy.sqlite') as app:
        cached = schedule(app, config, Path.cwd(), as_of=at)
        if cached.planner_version != 217 or cached.planner_status != 'ready':
            raise ValueError('Restarted bridge did not render the prepared build')
    report = dict(prepared=True, attached=False, release='calendar_mass_v217',
        as_of=at.isoformat(), source_pointer=pointer, runtime=str(runtime.resolve()),
        source_journal_digest=digest, app_evidence_digest=evidence,
        source_activity_set=app_sources_from_copy(output/'app-before.sqlite'), manifest=manifest,
        acceptance=accepted, decision_index=event['index'],
        snapshot_fingerprint=event['snapshot_fingerprint'], seconds=event['seconds'],
        evaluations=event['evaluations'], churn=event['churn'],
        independent_reprice=True, original_decisions_issues_completions_preserved=True,
        restart_policy_verified=True, unchanged_bridge_no_search=True,
        progression_state=controller, before=before, after=result.model_dump(mode='json'),
        scope='Real-evidence copied live bridge; no replay receipts or fabricated history imported')
    with (output/'report.json').open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps({k: report[k] for k in ('prepared', 'attached', 'runtime', 'seconds', 'churn')}), flush=True)


def recertify(prepared, output, parity_path, race_path, tests_path):
    """Retain the original search provenance; verify the behavior-preserving wiring.

    This is not a fresh opening search or a new race stability pass. All old
    reports/archives survive; only a new guarded attachment report is created.
    """
    import xml.etree.ElementTree as ET
    from fastapi.testclient import TestClient
    from run_analysis.web.app import create_app
    from run_analysis.adaptive.new_model_daily_harness import DailyHarness
    original=json.loads((prepared/'report.json').read_text())
    parity=json.loads(parity_path.read_text());race=json.loads(race_path.read_text())
    if (not original['prepared'] or original['attached'] or parity['status']!='identical_general_reviews'
            or parity['decisions']!=34 or race['status']!='race_apply_and_restart_passed'
            or not race['restart_no_search'] or not race['independent_reprice']):
        raise ValueError('General equivalence and real race-route verification are required')
    cases=ET.parse(tests_path).findall('.//testcase')
    if not cases or any(c.find('failure') is not None or c.find('error') is not None for c in cases):
        raise ValueError('Passing shared-progression regressions required')
    for proof in (parity,race):
        archive=proof['manifest']
        if sha256(Path(archive['path']).read_bytes()).hexdigest()!=archive['sha256']:
            raise ValueError('Verification archive changed')
        with ZipFile(archive['path']) as bundle: manifest=json.loads(bundle.read('manifest.json'))
        if any(sha256(Path(p).read_bytes()).hexdigest()!=digest
               for p,digest in manifest['source_sha256'].items() if p.startswith('src/')):
            raise ValueError('Verified production sources changed')
    cfg=load_config();database=resolve_project_path(Path.cwd(),cfg['paths']['database'])
    source=Path(original['source_pointer']['root']);runtime=Path(original['runtime'])
    with connect(database) as app:
        pointer=json.loads(app.execute("SELECT value_json FROM app_state WHERE key='adaptive_runtime'").fetchone()[0])
        if (pointer!=original['source_pointer'] or journal_digest(source/'sandbox.sqlite')!=original['source_journal_digest']
                or app_evidence(app,cfg,datetime.fromisoformat(original['as_of']))!=original['app_evidence_digest']):
            raise ValueError('Live evidence changed; prepare a new current-evidence opening')
    load_decision(runtime,original['decision_index']+1)
    output.mkdir(parents=True,exist_ok=False)
    with (sqlite3.connect((prepared/'app-copy.sqlite').resolve().as_uri()+'?mode=ro',uri=True) as old,
            sqlite3.connect(output/'app-copy.sqlite') as copied):
        old.backup(copied)
    from copy import deepcopy
    test_cfg=deepcopy(cfg);test_cfg['paths']['database']=str((output/'app-copy.sqlite').resolve())
    for key in ('uploads','outputs'):
        if key in test_cfg['paths']:test_cfg['paths'][key]=str((output/key).resolve())
    certificate=runtime/'coached-release.json';old_certificate=certificate.read_bytes()
    adapted=json.loads(old_certificate);adapted['database']=test_cfg['paths']['database']
    certificate.write_text(json.dumps(adapted))
    def no_search(*args,**kwargs):raise AssertionError('Recertification unexpectedly repeated search')
    try:
        with patch('run_analysis.web.app.load_config',return_value=test_cfg),patch.object(DailyHarness,'replan',no_search),TestClient(create_app(Path.cwd())) as client:
            response=client.get('/api/weekly-schedule/latest');response.raise_for_status()
            current=response.json()
            if current['planner_version']!=217 or current['planner_status']!='ready':
                raise ValueError('General app route did not retain the prepared plan')
            response=client.get('/api/settings/race-preparation/status');response.raise_for_status()
            if not response.json()['race_activation_available']:
                raise ValueError('Race Apply would be disabled by deployment')
    finally:certificate.write_bytes(old_certificate)
    manifest=freeze_sources(output/'release.sources.zip',config=cfg,entrypoint=__name__,
        inputs=(prepared/'report.json',parity_path,race_path,tests_path))
    report=dict(original,manifest=manifest,after=current,opening_source_manifest=original['manifest'],
        original_opening_report=str((prepared/'report.json').resolve()),
        recertified_at=datetime.now(timezone.utc).isoformat(),
        general_equivalence=dict(path=str(parity_path.resolve()),sha256=sha256(parity_path.read_bytes()).hexdigest()),
        race_verification=dict(path=str(race_path.resolve()),sha256=sha256(race_path.read_bytes()).hexdigest()),
        shared_regressions=dict(path=str(tests_path.resolve()),tests=len(cases)),
        race_support_verified=True,recertification_searches=0,
        scope='Original opening retained by 34-review exact equivalence; copied-app race Apply/search/restart separately verified')
    with (output/'report.json').open('x') as handle:json.dump(report,handle,indent=2,allow_nan=False)
    print(json.dumps(dict(prepared=True,attached=False,race_support_verified=True,
        recertification_searches=0,tests=len(cases))))


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'recertify', 'attach'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runtime', type=Path)
    parser.add_argument('--proof', type=Path, default=Path('output/cadence-mass-20261019'))
    parser.add_argument('--prepared',type=Path)
    parser.add_argument('--parity',type=Path)
    parser.add_argument('--race-check',type=Path)
    parser.add_argument('--tests',type=Path)
    args = parser.parse_args()
    if args.mode == 'prepare':
        if args.runtime is None:
            parser.error('prepare requires --runtime')
        prepare(args.output, args.runtime, args.proof)
    elif args.mode=='recertify':
        if any(x is None for x in (args.prepared,args.parity,args.race_check,args.tests)):
            parser.error('recertify requires --prepared, --parity, --race-check and --tests')
        recertify(args.prepared,args.output,args.parity,args.race_check,args.tests)
    else:
        report = json.loads((args.output/'report.json').read_text())
        if (report.get('release') != 'calendar_mass_v217' or not report.get('unchanged_bridge_no_search')
                or not report.get('race_support_verified')):
            raise ValueError('Verified V7 copied bridge report required')
        attach(args.output)
