"""Paired complete harness searches with exact pre-edit source controls.

No publication, uploads, live-pointer changes, or search-policy changes. Each
mode gets a fresh coherent runtime clone and fresh evaluator caches. Hash every
full accounted evaluation in order, plus the complete final search result.
"""
from argparse import ArgumentParser
import ast
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import asdict, is_dataclass
from datetime import datetime, date, timedelta
import fcntl
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sqlite3
from time import perf_counter
from unittest.mock import patch
from zipfile import ZipFile

from run_analysis.config import load_config
from run_analysis.adaptive.factory import resume_service
from run_analysis.adaptive.identity_json import identity_fields
from scripts.research_manifest import freeze_sources


def encode(value):
    if is_dataclass(value): return identity_fields(value)
    if isinstance(value, (date, datetime)): return value.isoformat()
    raise TypeError(type(value).__name__)


@contextmanager
def original_records(reference):
    from run_analysis.adaptive import program_evidence_adapter as adapter
    from run_analysis.adaptive import shared_program_evaluator as evaluator
    from run_analysis.adaptive import continuous_program_adapter as continuous
    from run_analysis.adaptive import quality_coaching_policy as coaching
    project = Path(__file__).resolve().parents[1]
    reviewed = {f'src/run_analysis/adaptive/{name}.py' for name in
                ('identity_json', 'program_evidence_adapter', 'shared_program_evaluator', 'program_objective_trial')}
    with ZipFile(reference) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        for name, digest in manifest['source_sha256'].items():
            if name.startswith('src/') and name not in reviewed:
                assert sha256((project/name).read_bytes()).hexdigest() == digest, name
        assert not ({str(p.relative_to(project)) for p in (project/'src').rglob('*.py')}
                    - set(manifest['source_sha256']) - reviewed)
        def function(module, name, **bindings):
            path = 'src/'+module.__name__.replace('.', '/')+'.py'
            raw = archive.read(path)
            assert sha256(raw).hexdigest() == manifest['source_sha256'][path]
            node = next(n for n in ast.parse(raw).body
                        if isinstance(n, ast.FunctionDef) and n.name == name)
            scope = dict(vars(module), **bindings)
            exec(compile(ast.Module(body=[node], type_ignores=[]), module.__file__, 'exec'), scope)
            return scope[name]
        sessions = function(evaluator, '_sessions')
        program = function(evaluator, 'evaluate_program', _sessions=sessions)
        accounted = function(adapter, 'evaluate_accounted', evaluate_program=program)
        with ExitStack() as stack:
            for module, name, value in (
                (evaluator, '_sessions', sessions), (evaluator, 'evaluate_program', program),
                (adapter, 'evaluate_program', program), (adapter, 'evaluate_accounted', accounted),
                (continuous, 'evaluate_accounted', accounted), (coaching, 'evaluate_program', program)):
                stack.enter_context(patch.object(module, name, value))
            yield


def clone(root, destination):
    lock = root/'live-bridge.lock'
    with lock.open('r') if lock.exists() else nullcontext() as locked:
        if locked is not None: fcntl.flock(locked, fcntl.LOCK_SH)
        shutil.copytree(root, destination, ignore=shutil.ignore_patterns(
            '*.sqlite', '*.sqlite-wal', '*.sqlite-shm', '*.lock'))
        with sqlite3.connect((root.resolve()/'sandbox.sqlite').as_uri()+'?mode=ro', uri=True) as source:
            with sqlite3.connect(destination/'sandbox.sqlite') as copied: source.backup(copied)


def without_wall_times(value, path='', removed=None):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            at = path+'.'+key if path else key
            if key in {'seconds','seconds_before_refinement'}:
                removed.append(at)
            else: result[key] = without_wall_times(item, at, removed)
        return result
    if isinstance(value, (tuple, list)):
        return [without_wall_times(v, path+f'[{i}]', removed) for i, v in enumerate(value)]
    return value


def attribute_first_comparison(output):
    """Preserve the initial failed comparator; attribute its wall-time-only diff."""
    original, optimized = (json.loads((output/(m+'.json')).read_text())
                           for m in ('original','optimized'))
    omitted = []
    a = without_wall_times(original['final'], removed=omitted)
    b = without_wall_times(optimized['final'], removed=[])
    assert a == b
    assert original['ordered_full_digest'] == optimized['ordered_full_digest']
    assert original['ordered_full_evaluations'] == optimized['ordered_full_evaluations']
    report = dict(original_comparator_stopped=True,
        attribution='Only opening.seconds_before_refinement remained unexcluded; no score/calendar mismatch',
        exact_ordered_full_outputs=True, exact_final_excluding_wall_time=True,
        additional_excluded_wall_time_fields=omitted, performance_accepted=False,
        implementation_retained=False,
        inputs={m:sha256((output/(m+'.json')).read_bytes()).hexdigest() for m in ('original','optimized')})
    with (output/'attribution.json').open('x') as stream: json.dump(report, stream, indent=2)
    return report


def run(root, output, reference, order):
    from run_analysis.adaptive import continuous_program_adapter as continuous
    output.mkdir(parents=True, exist_ok=False)
    config = load_config()
    manifest = freeze_sources(output/'sources.zip', config=config, entrypoint=__name__,
        inputs=(reference, root/'scenario.json', *sorted(root.glob('decision-*.json')),
                *sorted(root.glob('snapshot-*.json'))))
    latest = json.loads(sorted(root.glob('decision-*.json'))[-1].read_text())
    at = datetime.fromisoformat(latest['as_of'])+timedelta(microseconds=1)
    records = []
    for mode in order.split(','):
        destination = output/(mode+'-runtime')
        clone(root, destination)
        with original_records(reference) if mode == 'original' else nullcontext():
            service = resume_service(destination, config, as_of=at)
            digest = sha256(); calls = 0; hash_seconds = 0.
            original = continuous.evaluate_accounted
            def tracked(context, history, planned, **kwargs):
                nonlocal calls, hash_seconds
                result = original(context, history, planned, **kwargs)
                stamp = perf_counter()
                candidate = [(r.key, r.start.isoformat(), r.miles, r.role) for r in planned]
                raw = json.dumps((candidate, result), sort_keys=True, separators=(',', ':'),
                                 allow_nan=False, default=encode)
                digest.update(raw.encode()); digest.update(b'\n'); calls += 1
                hash_seconds += perf_counter()-stamp
                return result
            try:
                stamp = perf_counter()
                with patch.object(continuous, 'evaluate_accounted', tracked):
                    result = service.harness.replan(at, reason='exact fixed-record benchmark', publish=False)
                elapsed = perf_counter()-stamp
                removed = []
                selected = dict(program=result['program'], snapshot=result['state'].fingerprint,
                    search=without_wall_times(result['search'], removed=removed))
                row = dict(mode=mode, seconds=elapsed, trace_overhead_seconds=hash_seconds,
                    ordered_full_evaluations=calls, ordered_full_digest=digest.hexdigest(),
                    final=selected, removed_wall_time_fields=removed,
                    policy=asdict(service.harness.policy))
                (output/(mode+'.json')).write_text(json.dumps(row, indent=2, default=encode))
                print(json.dumps({k:v for k,v in row.items() if k not in ('final','policy') }), flush=True)
                records.append(row)
            finally: service.harness.sandbox.close()
    before, after = ({r['mode']:r for r in records}[m] for m in ('original','optimized'))
    for key in ('ordered_full_evaluations','ordered_full_digest','final','policy','removed_wall_time_fields'):
        assert before[key] == after[key], ('Exact search parity failed', key)
    report = dict(manifest=manifest, source=str(root), results=records,
        exact_full_search=True, live_changed=False, publication=False,
        savings_fraction=1-after['seconds']/before['seconds'])
    (output/'report.json').write_text(json.dumps(report, indent=2, default=encode))
    print(json.dumps(dict(exact_full_search=True, original_seconds=before['seconds'],
        optimized_seconds=after['seconds'], savings_fraction=report['savings_fraction'],
        ordered_full_evaluations=after['ordered_full_evaluations'])), flush=True)


if __name__ == '__main__':
    p = ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--order', default='original,optimized', choices=('original,optimized','optimized,original'))
    a = p.parse_args(); run(a.root, a.output, a.reference, a.order)
