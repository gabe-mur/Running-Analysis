"""Freeze the quality workstream baseline without search or live publication."""
from argparse import ArgumentParser
from hashlib import sha256
import fcntl
import json
from pathlib import Path
import shutil
import sqlite3
from zipfile import ZipFile, ZIP_DEFLATED

from run_analysis.config import load_config, resolve_project_path
from scripts.replay_adaptive_search import load_decision
from scripts.research_manifest import freeze_sources


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def freeze(output):
    project = Path.cwd().resolve()
    config = load_config()
    database = resolve_project_path(project, config['paths']['database'])
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True) as db:
        pointer = db.execute("SELECT value_json FROM app_state WHERE key='adaptive_runtime'").fetchone()[0]
    source = Path(json.loads(pointer)['root']).resolve()
    output = output.resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('Baseline destination must be separate from the live runtime')
    output.mkdir(parents=True, exist_ok=False)
    clone = output/'runtime'
    with (source/'live-bridge.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        shutil.copytree(source, clone, ignore=shutil.ignore_patterns('*.sqlite', '*.sqlite-wal', '*.sqlite-shm', '*.lock'))
        with sqlite3.connect((source/'sandbox.sqlite').as_uri()+'?mode=ro', uri=True) as original:
            with sqlite3.connect(clone/'sandbox.sqlite') as copied:
                original.backup(copied)
        latest = max(clone.glob('decision-*.json'))
        event = json.loads(latest.read_text())
        inputs = (latest, clone/event['snapshot_file'], clone/'scenario.json')
        source_hashes = {str(p.relative_to(clone)): digest(source/p.relative_to(clone)) for p in inputs}
    manifest = freeze_sources(output/'sources.zip', config=config, entrypoint=__name__, inputs=inputs)
    # The shared manifest archives Python. Preserve exact frontend bytes too.
    assets = sorted(p for p in (project/'src/run_analysis/web/static').rglob('*')
                    if p.is_file() and p.suffix in {'.js', '.css', '.html'})
    with ZipFile(output/'frontend.zip', 'x', compression=ZIP_DEFLATED) as archive:
        for path in assets:
            archive.write(path, str(path.relative_to(project)))
    case = load_decision(clone, event['index']+1)
    assert all(digest(source/relative) == value for relative, value in source_hashes.items())
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True) as db:
        assert db.execute("SELECT value_json FROM app_state WHERE key='adaptive_runtime'").fetchone()[0] == pointer
    report = dict(version=1, live_runtime=source.as_posix(), cloned_runtime=clone.as_posix(),
        decision=latest.name, as_of=event['as_of'], source_manifest=manifest,
        frontend_sha256=digest(output/'frontend.zip'),
        enabled_menu=config['coaching']['quality_sessions'],
        archived_cost=event['cost'], repriced_cost=case['archived_cost'],
        archived_candidate_eligible=True, live_pointer_unchanged=True,
        archived_inputs_unchanged=True, search_performed=False,
        quality_catalog=[dict(key=d.key, miles=d.miles, phases=d.phases,
            workout_definition=d.workout_definition) for d in case['catalog'].values() if d.role=='quality'],
        stage=event['stage_runtime']['stage'],
        prototype_status='Frozen source includes quality_choices_v1. Wiring tested; predictor and daily acceptance open.',
        release_scope='No live plan/settings write. New replacement stays unconnected until checklist gates pass.')
    (output/'baseline.json').write_text(json.dumps(report, indent=2, default=str)+'\n')
    return report


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    report = freeze(parser.parse_args().output)
    print(json.dumps({k: report[k] for k in ('decision', 'as_of', 'archived_cost', 'repriced_cost',
        'live_pointer_unchanged', 'archived_inputs_unchanged', 'search_performed')}))


if __name__ == '__main__':
    main()
