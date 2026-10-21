"""Desktop initialization and migration must never depend on a developer's data."""
from importlib.util import module_from_spec, spec_from_file_location
import hashlib
from pathlib import Path
import sqlite3

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    spec = spec_from_file_location(path.stem, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_empty_private_workspace_is_independent_and_idempotent(tmp_path):
    server = load(ROOT / 'desktop/server_entry.py')
    first, second = tmp_path / 'first', tmp_path / 'second'
    config, database = server.prepare_workspace(first, ROOT)
    assert database == first / 'data/run_analysis.sqlite'
    with sqlite3.connect(database) as db:
        assert db.execute('SELECT count(*) FROM activities').fetchone()[0] == 0
    # Updating the application must not replace a person's settings.
    overlay = first / 'config.local.yaml'
    overlay.write_text('max_hr: 187\n')
    assert server.prepare_workspace(first, ROOT)[0]['max_hr'] == 187
    assert server.prepare_workspace(second, ROOT)[0]['max_hr'] == config['max_hr']
    assert not (second / 'config.local.yaml').exists()
    assert overlay.stat().st_mode & 0o777 == 0o600
    assert not (first / 'data/credentials').exists()


def test_migration_copies_recordings_without_changing_originals(tmp_path):
    from run_analysis.db import connect
    from run_analysis.importer import import_files
    from test_tcx import make_tcx
    server = load(ROOT / 'desktop/server_entry.py')
    migration = load(ROOT / 'scripts/migrate_desktop_data.py')
    source, destination = tmp_path / 'source', tmp_path / 'app-data'
    config, database = server.prepare_workspace(source, ROOT)
    recording = make_tcx(source / 'TCX/run.tcx')
    with connect(database) as db:
        import_files(db, source, config['timezone_default'], paths=[recording])
    before = hashlib.sha256(recording.read_bytes()).hexdigest()
    summary = migration.migrate(source, destination)
    assert summary['counts']['activities'] == 1
    assert summary['source_modified'] is False
    assert not summary['active_planner_copied']
    assert hashlib.sha256(recording.read_bytes()).hexdigest() == before
    with sqlite3.connect(destination / 'data/run_analysis.sqlite') as db:
        path = Path(db.execute('SELECT path FROM source_files').fetchone()[0])
        assert path.is_relative_to(destination)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    with pytest.raises(ValueError, match='new destination'):
        migration.migrate(source, destination)


def test_bundle_is_built_from_explicit_resources_not_the_checkout():
    spec = (ROOT / 'desktop/RunningCoachServer.spec').read_text()
    assert 'include_py_files=True' in spec  # Implementation fingerprints survive freezing.
    assert "root/'config.yaml'" not in spec
    assert "root/'data'" not in spec
    assert "collect_data_files('certifi')" in spec
    assert 'static/app/views/*.js' in (ROOT / 'pyproject.toml').read_text()


def test_migration_rebinds_release_location_but_preserves_verified_policy(tmp_path):
    import json
    from dataclasses import asdict
    from test_coached_live_integration import sequence
    migration = load(ROOT / 'scripts/migrate_desktop_data.py')
    runtime = tmp_path / 'runtime'; runtime.mkdir()
    old, new = tmp_path / 'old.sqlite', tmp_path / 'new.sqlite'
    base = sequence()
    decision = json.dumps({'stage_runtime': {'coaching_base': asdict(base)}})
    (runtime / 'decision-0001.json').write_text(decision)
    certificate = dict(version='coached_live_v206', database=str(old),
        coaching_base_key=base.key, validated_app_copy=True, acceptance_report_sha256='retained')
    (runtime / 'coached-release.json').write_text(json.dumps(certificate))
    assert migration.rebind_release_owner(runtime, old, new)
    rebound = json.loads((runtime / 'coached-release.json').read_text())
    assert rebound == dict(certificate, database=str(new))
    assert (runtime / 'decision-0001.json').read_text() == decision
    with pytest.raises(ValueError, match='unverified'):
        migration.rebind_release_owner(runtime, old, new)
