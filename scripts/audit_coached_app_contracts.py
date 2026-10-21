"""Read-only live-source clone audit of coaching integration boundaries.

No calendar searches or publication. Probe the real harness up to its ordinary
search boundary with current live evidence, then change only enabled families.
This is an integration audit, not a stability pass or release authorization.
"""
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import fcntl
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sqlite3
from unittest.mock import patch

from run_analysis.config import load_config, resolve_project_path
from run_analysis.adaptive.new_model_daily_harness import DailyHarness
from run_analysis.adaptive.coached_runtime import enable, offline_guard
from run_analysis.adaptive.quality_forecast_runtime import VERSION
from scripts.research_manifest import freeze_sources


class SearchBoundary(Exception):
    pass


def audit(native, output):
    output.mkdir(parents=True, exist_ok=False)
    config = load_config(); project = Path.cwd()
    database = resolve_project_path(project, config['paths']['database']).resolve()
    with sqlite3.connect(database.as_uri()+'?mode=ro', uri=True) as app:
        pointer_raw = app.execute("SELECT value_json FROM app_state WHERE key='adaptive_runtime'").fetchone()[0]
        source = Path(json.loads(pointer_raw)['root'])
        with (source/'live-bridge.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            shutil.copytree(source, output/'runtime', ignore=shutil.ignore_patterns(
                '*.sqlite','*.sqlite-wal','*.sqlite-shm','*.lock'))
            with sqlite3.connect((source/'sandbox.sqlite').resolve().as_uri()+'?mode=ro', uri=True) as journal:
                with sqlite3.connect(output/'runtime/sandbox.sqlite') as copied: journal.backup(copied)
            with sqlite3.connect(output/'app-copy.sqlite') as copied: app.backup(copied)
            journal_digest = sha256('\n'.join(journal_rows(source/'sandbox.sqlite')).encode()).hexdigest()
    latest = json.loads(max((output/'runtime').glob('decision-*.json')).read_text())
    native_event = json.loads(max(native.glob('decision-*.json')).read_text())
    at = max(datetime.now(timezone.utc), datetime.fromisoformat(latest['as_of'])+timedelta(microseconds=1))
    n = DailyHarness.resume(native, config, as_of=datetime.fromisoformat(native_event['as_of']))
    # Native input must be a separately frozen offline runtime; never modify it.
    try:
        additions = {k:d for k,d in n.catalog.items() if d.workout_definition and
                     d.workout_definition.get('calendar_integration') == VERSION}
        contracts = {k:n.coaching_contracts[k] for k in additions}
        native_runtime = deepcopy(n.quality_runtime)
        stage, policy, scales = n.stage, n.training_policy, n.scales
    finally: n.sandbox.close()
    manifest = freeze_sources(output/'sources.zip', config=config, entrypoint=__name__,
        inputs=(output/'app-copy.sqlite', output/'runtime/scenario.json',
                native/'scenario.json', native/max(native.glob('decision-*.json')).name))
    probes = []
    for change in (None, 'threshold', 'progression', 'short_intervals'):
        h = DailyHarness.resume(output/'runtime', deepcopy(config), as_of=at)
        try:
            if (h.stage != stage or h.training_policy != policy or h.scales != scales):
                raise ValueError('Native commands cannot replace the current accepted stage/normalization')
            h.catalog.update(additions); h.coaching_contracts.update(contracts)
            h.quality_runtime = deepcopy(native_runtime)
            enable(h)
            original_sequence = h.coaching_sequence
            if change is not None: h.config['coaching']['quality_sessions'][change] = False
            captured = {}
            def boundary(**values):
                captured.update(values)
                raise SearchBoundary()
            try:
                with patch('run_analysis.adaptive.new_model_daily_harness.migrated_search', boundary):
                    h.replan(at, reason='copied-app integration boundary audit', publish=False)
            except SearchBoundary:
                result = dict(reached_search=True, error=None, budget=captured['budget'])
            except (KeyError, ValueError) as error:
                result = dict(reached_search=False, error=repr(error), error_type=type(error).__name__)
            else: raise AssertionError('Audit unexpectedly performed or skipped a search')
            probes.append(dict(disabled_family=change, **result,
                frozen_sequence_unchanged=h.coaching_sequence == original_sequence))
        finally: h.sandbox.close()
    # Confirm the production guard, not an unguarded pointer repoint, is the
    # remaining app-attachment seam. Nothing is attached to the real database.
    with sqlite3.connect(output/'app-copy.sqlite') as copied:
        copied.execute("UPDATE app_state SET value_json=? WHERE key='adaptive_runtime'",
            (json.dumps(dict(engine=json.loads(pointer_raw)['engine'], root=str((output/'runtime').resolve()))),))
    copied_config = deepcopy(config); copied_config['paths']['database'] = str((output/'app-copy.sqlite').resolve())
    h = DailyHarness.resume(output/'runtime', config, as_of=at)
    try:
        h.config = copied_config
        try: offline_guard(h)
        except ValueError as error: attachment_guard = str(error)
        else: raise AssertionError('Unverified app pointer bypassed the offline guard')
    finally: h.sandbox.close()
    with sqlite3.connect(database.as_uri()+'?mode=ro', uri=True) as app:
        assert app.execute("SELECT value_json FROM app_state WHERE key='adaptive_runtime'").fetchone()[0] == pointer_raw
    assert sha256('\n'.join(journal_rows(source/'sandbox.sqlite')).encode()).hexdigest() == journal_digest
    report = dict(manifest=manifest, live_runtime=str(source), live_decision=latest['index']+1,
        as_of=at.isoformat(), probes=probes, attachment_guard=attachment_guard,
        native_stage_matches=True, live_pointer_and_journal_unchanged=True,
        new_calendar_searches=0, publication=False, full_app_acceptance=False,
        scope='Current coherent live app/runtime copy; supported-input and policy-transition boundaries only')
    (output/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k != 'manifest'}, indent=2), flush=True)


def journal_rows(path):
    with sqlite3.connect(path.resolve().as_uri()+'?mode=ro', uri=True) as db:
        return list(db.iterdump())


if __name__ == '__main__':
    p=ArgumentParser(description=__doc__); p.add_argument('native', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args(); audit(a.native, a.output)
