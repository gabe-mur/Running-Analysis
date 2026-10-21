"""Exercise full upload no-op orchestration on a disk backup, never live data.

Pass previously downloaded original FITs; no network or planner runs are needed.
Every downstream stage is instrumented to fail if invoked, not merely hidden.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import tempfile
from unittest.mock import patch

import yaml

from run_analysis.config import load_config, resolve_project_path
from run_analysis.privacy import private_directory, private_file
from run_analysis.web.upload_service import UploadPayload, run_upload_pipeline


def _fingerprint(connection):
    digest = sha256()
    for line in connection.iterdump():
        digest.update(line.encode())
        digest.update(b'\n')
    return digest.hexdigest()


def run(root, files):
    config = load_config(root / 'config.yaml')
    database = resolve_project_path(root, config['paths']['database']).resolve()
    before_hash = sha256(database.read_bytes()).hexdigest()
    audit_root = Path(tempfile.mkdtemp(prefix='duplicate-upload-audit-', dir=private_directory(root / 'output')))
    private_directory(audit_root)
    sandbox_path = audit_root / 'sandbox.sqlite'
    report = dict(scope='full upload orchestration on live-database backup; cached original FITs',
                  automatic_sync_enabled=False, live_import_performed=False, results=[])
    with sqlite3.connect(f'{database.as_uri()}?mode=ro', uri=True) as live:
        live.execute('BEGIN')
        with sqlite3.connect(sandbox_path) as sandbox:
            live.backup(sandbox)
            before = _fingerprint(sandbox)
            count = sandbox.execute('SELECT count(*) FROM activities').fetchone()[0]
        private_file(sandbox_path)
        config['paths']['database'] = str(sandbox_path)
        config_path = audit_root / 'config.yaml'
        config_path.write_text(yaml.safe_dump(config))
        private_file(config_path)
        with ExitStack() as stack:
            guards = {name: stack.enter_context(patch(
                f'run_analysis.web.upload_service.{name}',
                side_effect=AssertionError(f'No-op upload called {name}'),
            )) for name in ('sync_overrides', 'process_activities', 'update_weather',
                            'fit_models', 'load_current_status', 'generate_weekly_schedule',
                            'match_activities_to_prescriptions')}
            for filename in files:
                path = Path(filename)
                for attempt in ('renamed-download', 'repeat-download'):
                    result = run_upload_pipeline(audit_root, config_path, [
                        UploadPayload(f'intervals-{path.name}', path.read_bytes())])
                    report['results'].append(dict(file=path.name, attempt=attempt,
                                                  response=result.model_dump(mode='json')))
                    assert result.files[0].status == 'unchanged', 'Cached FIT was not a completed-content hit'
                    assert result.files[0].activity_ids, 'Existing run link missing'
                    assert all(s.status == 'skipped' for s in result.stages
                               if s.name in ('process', 'weather', 'model', 'schedule'))
            report['downstream_calls'] = {name: guard.call_count for name, guard in guards.items()}
            assert not any(report['downstream_calls'].values()), 'Downstream work was requested'
        with sqlite3.connect(sandbox_path) as sandbox:
            report['all_database_rows_unchanged'] = _fingerprint(sandbox) == before
            report['activity_count_before'] = count
            report['activity_count_after'] = sandbox.execute('SELECT count(*) FROM activities').fetchone()[0]
        assert report['all_database_rows_unchanged'], 'Sandbox rows changed'
        report['live_connection_writes'] = live.total_changes
    report['live_database_bytes_unchanged'] = sha256(database.read_bytes()).hexdigest() == before_hash
    assert report['live_database_bytes_unchanged'], 'Live database changed during audit'
    report['passed'] = True
    report_path = audit_root / 'summary.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    private_file(report_path)
    print(json.dumps({key: value for key, value in report.items() if key != 'results'}, indent=2))
    print(f'Report: {report_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='+', type=Path)
    args = parser.parse_args()
    run(Path.cwd(), args.files)
