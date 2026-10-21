"""Copy an existing local project to a NEW desktop workspace, never move it.

SQLite backup includes committed WAL state. The active planner's file lock
protects its journal/export set. Immutable snapshots are never rewritten: only
source-file locations and the app's runtime pointer are rebased in the copy.
Research corpora, old runtimes and audit output are deliberately not copied.
"""
from argparse import ArgumentParser
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile

from run_analysis.config import load_config, resolve_project_path


@contextmanager
def runtime_guard(root):
    if root is None:
        yield
        return
    import fcntl
    with (root/'live-bridge.lock').open('r') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        yield


def backup(source, target):
    target.parent.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect(source.resolve().as_uri()+'?mode=ro',uri=True) as old, sqlite3.connect(target) as new:
        old.backup(new)
        assert new.execute('PRAGMA quick_check').fetchone()[0]=='ok'


def rebind_release_owner(runtime, original_database, destination_database):
    """Rebind only the verified database location, not the coaching contract.

    The certificate belongs to a database path. Copying it byte-for-byte leaves
    the new private workspace owned by the old development database. Immutable
    decisions, snapshots, acceptance evidence and policy keys remain untouched.
    """
    path = runtime / 'coached-release.json'
    if not path.exists():
        return False
    from run_analysis.adaptive.quality_coaching_policy import read_sequence
    release = json.loads(path.read_text())
    event = json.loads(max(runtime.glob('decision-*.json')).read_text())
    base = read_sequence(event['stage_runtime']['coaching_base'])
    if (release.get('version') != 'coached_live_v206' or
            Path(release.get('database', '')).resolve() != original_database.resolve() or
            release.get('coaching_base_key') != base.key or not release.get('validated_app_copy')):
        raise ValueError('Cannot migrate an unverified coached release owner')
    release['database'] = str(destination_database.resolve())
    temporary = path.with_suffix('.desktop-owner.tmp')
    temporary.write_text(json.dumps(release, indent=2) + '\n')
    temporary.chmod(0o600)
    temporary.replace(path)
    return True


def migrate(source, destination):
    source=source.resolve();destination=destination.expanduser().resolve()
    if destination.exists() or destination==source or source in destination.parents:
        raise ValueError('Choose a new destination outside the source project; no existing data will be overwritten')
    config=load_config(source/'config.yaml')
    original=resolve_project_path(source,config['paths']['database'])
    with sqlite3.connect(original.resolve().as_uri()+'?mode=ro',uri=True) as db:
        row=db.execute("SELECT value_json FROM app_state WHERE key='adaptive_runtime'").fetchone()
        pointer=json.loads(row[0]) if row else None
    runtime=Path(pointer['root']) if pointer else None
    config_files={name:hashlib.sha256((source/name).read_bytes()).hexdigest()
        for name in ('config.yaml','config.local.yaml') if (source/name).exists()}
    destination.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix='.running-coach-migration-',dir=destination.parent))
    runtime_relative=Path('data')/'planner'
    final_runtime=destination/runtime_relative
    with runtime_guard(runtime):
        # Reject a pointer switch while waiting for a running search to finish.
        with sqlite3.connect(original.resolve().as_uri()+'?mode=ro',uri=True) as db:
            latest=db.execute("SELECT value_json FROM app_state WHERE key='adaptive_runtime'").fetchone()
            if (json.loads(latest[0]) if latest else None)!=pointer:
                raise ValueError('The active planner changed during migration; retry with the current runtime')
        for name in ('config.yaml','config.local.yaml','config.example.yaml','run_overrides.example.csv'):
            if (source/name).is_file():
                shutil.copyfile(source/name,staging/name)
        override=resolve_project_path(source,config['paths']['overrides'])
        if override.exists():
            shutil.copyfile(override,staging/'run_overrides.csv')
        new_database=staging/'data/run_analysis.sqlite'
        backup(original,new_database)
        copied={}
        def copy_sources(database, *, is_runtime=False):
            with sqlite3.connect(database) as db:
                for key,path,digest in db.execute('SELECT id,path,sha256 FROM source_files').fetchall():
                    old=Path(path)
                    if not old.is_absolute():old=source/old
                    old=old.resolve()
                    if old not in copied:
                        if not old.is_file() or hashlib.sha256(old.read_bytes()).hexdigest()!=digest:
                            raise ValueError(f'Original activity {key} is missing or changed; migration stopped without editing the source')
                        if is_runtime and old.is_relative_to(runtime):
                            relative=runtime_relative/old.relative_to(runtime)
                        else:
                            suffix=''.join(old.suffixes[-2:]) if old.suffix=='.gz' else old.suffix
                            relative=Path('uploads/migrated')/(f'{len(copied)+1}-{digest[:16]}'+suffix)
                        target=staging/relative;target.parent.mkdir(parents=True,exist_ok=True)
                        if not target.exists():shutil.copyfile(old,target)
                        copied[old]=destination/relative
                    db.execute('UPDATE source_files SET path=? WHERE id=?',(str(copied[old]),key))
                if not is_runtime and pointer:
                    changed=dict(pointer,root=str(final_runtime))
                    db.execute("UPDATE app_state SET value_json=? WHERE key='adaptive_runtime'",(json.dumps(changed),))
        copy_sources(new_database)
        if runtime:
            shutil.copytree(runtime,staging/runtime_relative,
                ignore=shutil.ignore_patterns('*.sqlite','*.sqlite-*','*.lock','*.server.lock'))
            backup(runtime/'sandbox.sqlite',staging/runtime_relative/'sandbox.sqlite')
            copy_sources(staging/runtime_relative/'sandbox.sqlite',is_runtime=True)
            rebind_release_owner(staging/runtime_relative, original, destination/'data/run_analysis.sqlite')
        for name in ('credentials','elevation_tiles','bridge_geometry','weather_cache'):
            directory=source/'data'/name
            if directory.is_dir():shutil.copytree(directory,staging/'data'/name)
        for name in ('intervals-sync.json','intervals-wellness.json','weather_privacy_salt'):
            if (source/'data'/name).is_file():shutil.copyfile(source/'data'/name,staging/'data'/name)
        if (source/'data/.hr-load-cache.sqlite').is_file():
            backup(source/'data/.hr-load-cache.sqlite',staging/'data/.hr-load-cache.sqlite')
        if (source/'output/model_results.json').is_file():
            (staging/'output').mkdir(exist_ok=True)
            shutil.copyfile(source/'output/model_results.json',staging/'output/model_results.json')
        # Rebase explicit file paths in the NEW overlay, not the original files.
        import yaml
        overlay=yaml.safe_load((staging/'config.local.yaml').read_text()) if (staging/'config.local.yaml').exists() else {}
        overlay=overlay or {}
        overlay['paths']={**config['paths'],'database':'data/run_analysis.sqlite',
            'overrides':'run_overrides.csv','report':'output/report.html','weather_cache':'data/weather_cache'}
        (staging/'config.local.yaml').write_text(yaml.safe_dump(overlay,sort_keys=False))
        if any(hashlib.sha256((source/name).read_bytes()).hexdigest()!=digest for name,digest in config_files.items()):
            raise ValueError('Settings changed while migration was copying; retry. The source is untouched.')
        with sqlite3.connect(new_database) as db:
            counts={name:db.execute('SELECT COUNT(*) FROM '+name).fetchone()[0]
                for name in ('activities','source_files','trackpoints')}
        summary=dict(created_at=datetime.now(timezone.utc).isoformat(),counts=counts,
            copied_original_files=len(copied),active_planner_copied=bool(pointer),
            immutable_snapshots_unchanged=True,source_modified=False,private_reference_material_copied=False)
        (staging/'migration.json').write_text(json.dumps(summary,indent=2))
        for item in staging.rglob('*'):
            if item.is_symlink():raise ValueError('Unexpected symbolic link in private migration')
            item.chmod(0o700 if item.is_dir() else 0o600)
        staging.chmod(0o700)
        staging.rename(destination)
    return summary


if __name__=='__main__':
    parser=ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(migrate(args.source,args.destination),indent=2))
