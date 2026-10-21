"""Isolated exact-search comparison, not a different search/model experiment.

Capture the real harness's complete search arguments without publication, then
run both implementations against the SAME frozen evaluator and candidates.
Prototype patches are confined to this process. The production server remains
untouched. Record every evaluation's candidate, full score/eligibility result,
and fingerprint in an ordered digest, not merely the selected winner.
"""
from argparse import ArgumentParser
from contextlib import ExitStack,contextmanager,nullcontext
from contextvars import ContextVar
from dataclasses import asdict
from datetime import datetime,timedelta
from functools import cached_property
from hashlib import sha256
import cProfile
import json
from pathlib import Path
import pstats
import shutil
import inspect
import ast
import fcntl
import sqlite3
from functools import lru_cache
from copy import deepcopy
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile

from run_analysis.config import load_config
from run_analysis.adaptive.factory import resume_service
from run_analysis.adaptive.continuous_search_adapter import search
from run_analysis.adaptive.identity_json import identity_fields,identity_dumps
from scripts.research_manifest import freeze_sources

_in_prototype=ContextVar('lossless_identity_prototype',default=False)


@contextmanager
def fast_identities():
    if _in_prototype.get():
        yield
        return
    token=_in_prototype.set(True)
    try:
        with _prototype_patches():
            yield
    finally:
        _in_prototype.reset(token)


@contextmanager
def _prototype_patches():
    from run_analysis.adaptive import program_evidence_adapter as adapter,shared_program_evaluator as evaluator
    with ExitStack() as stack:
        for module in (adapter,evaluator):
            if hasattr(module,'asdict'):
                stack.enter_context(patch.object(module,'asdict',identity_fields))
                stack.enter_context(patch.object(module,'json',SimpleNamespace(dumps=identity_dumps)))
        for name in ('taper_start','volume_smoothing_days'):
            old=getattr(evaluator.ProgramContext,name)
            if isinstance(old,property):
                cached=cached_property(old.fget)
                cached.__set_name__(evaluator.ProgramContext,name)
                stack.enter_context(patch.object(evaluator.ProgramContext,name,cached))
        # Compile only these trusted repository functions for the isolated
        # prototype. This is not activity/snapshot-supplied executable data and
        # is never imported by the live scheduler. Exact source substitutions
        # let us test the tiny loop changes before editing a served core module.
        from run_analysis.adaptive import program_objective_trial as objective,continuation_boundary_trial as boundary
        stack.enter_context(patch.object(evaluator,'rolling_volume_components',_fast_kernel(objective.rolling_volume_components)))
        stack.enter_context(patch.object(boundary,'continuation_boundary_cost',_fast_kernel(boundary.continuation_boundary_cost)))
        yield


@lru_cache(maxsize=4)
def _fast_kernel(original):
    source=inspect.getsource(original)
    if original.__name__=='rolling_volume_components':
        if 'window_start=day-timedelta(days=window-1)' in source:
            return original
        old='        active=[r for r in rows if day-timedelta(days=window-1)<=r.day<=day]'
        new='        window_start=day-timedelta(days=window-1)\n        active=[r for r in rows if window_start<=r.day<=day]'
    elif original.__name__=='continuation_boundary_cost':
        if 'nominal_rows=' in source:
            return original
        old='    ledger = evaluation.conditional_ledger'
        new=old+'\n    nominal_rows={id(row):nominal(row.demand) for row in ledger}'
        assert source.count('nominal(r.demand) * exp(')==1
        source=source.replace('nominal(r.demand) * exp(','nominal_rows[id(r)] * exp(')
    else:
        raise ValueError('Only explicitly reviewed lossless kernels may be prototyped')
    assert source.count(old)==1,'Prototype must match the reviewed original source exactly'
    namespace=dict(original.__globals__)
    exec(compile(source.replace(old,new),original.__code__.co_filename,'exec'),namespace)
    return namespace[original.__name__]


@contextmanager
def original_identities(reference_sources):
    """Restore ONLY the reviewed original hot functions from our source archive.

    For this optimization's before/after diagnostic, not a general historical
    runner. Never accepts uploaded activity code. Verify the source bytes against
    the archive's manifest and retain the archive hash in the comparison report.
    Other model/search dependencies must be unchanged between the two modes.
    """
    from run_analysis.adaptive import program_evidence_adapter as adapter,shared_program_evaluator as evaluator
    from run_analysis.adaptive import program_objective_trial as objective,continuation_boundary_trial as boundary
    from run_analysis.adaptive import continuous_program_adapter as continuous
    with ZipFile(reference_sources) as archive:
        manifest=json.loads(archive.read('manifest.json'))
        project=Path(__file__).resolve().parents[1]
        reviewed={f'src/run_analysis/adaptive/{name}.py' for name in (
            'identity_json','program_evidence_adapter','shared_program_evaluator',
            'program_objective_trial','continuation_boundary_trial')}
        for name,digest in manifest['source_sha256'].items():
            if name.startswith('src/') and name not in reviewed:
                assert sha256((project/name).read_bytes()).hexdigest()==digest,('Unreviewed model dependency changed',name)
        assert not ({str(p.relative_to(project)) for p in (project/'src').rglob('*.py')}
                    -set(manifest['source_sha256'])-reviewed),'Unreviewed new model dependencies'
        def namespace(module):
            path='src/'+module.__name__.replace('.','/')+'.py'
            raw=archive.read(path)
            assert sha256(raw).hexdigest()==manifest['source_sha256'][path],path
            return ast.parse(raw),dict(vars(module),asdict=asdict,json=json)
        def function(module,name,**bindings):
            tree,scope=namespace(module)
            scope.update(bindings)
            node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
            exec(compile(ast.Module(body=[node],type_ignores=[]),module.__file__,'exec'),scope)
            return scope[name]
        volume=function(objective,'rolling_volume_components')
        cost=function(boundary,'continuation_boundary_cost')
        program=function(evaluator,'evaluate_program',rolling_volume_components=volume)
        accounted=function(adapter,'evaluate_accounted',evaluate_program=program)
        reference=function(adapter,'reference_key')
        tree,scope=namespace(evaluator)
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ProgramContext')
        getters={}
        for name in ('taper_start','volume_smoothing_days'):
            node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
            node.decorator_list=[]
            exec(compile(ast.Module(body=[node],type_ignores=[]),evaluator.__file__,'exec'),scope)
            getters[name]=property(scope[name])
        with ExitStack() as stack:
            for module,name,value in ((evaluator,'evaluate_program',program),
                (evaluator,'rolling_volume_components',volume),
                (boundary,'continuation_boundary_cost',cost),
                (adapter,'evaluate_program',program),(adapter,'evaluate_accounted',accounted),
                (adapter,'reference_key',reference),(continuous,'evaluate_accounted',accounted),
                (continuous,'reference_key',reference)):
                stack.enter_context(patch.object(module,name,value))
            for name,value in getters.items():
                stack.enter_context(patch.object(evaluator.ProgramContext,name,value))
            yield


def normalize_search_result(result):
    normalized=deepcopy(result)
    normalized.pop('seconds',None)
    for name in ('pre_calendar_distance','post_calendar_distance'):
        normalized.get('refinement',{}).get(name,{}).pop('seconds',None)
    return normalized


class CapturedSearch(Exception):
    pass


def capture(root,output):
    # An actively served SQLite database may have a WAL. Freeze only its clone;
    # do not copy open DB bytes or mutate/checkpoint the original. Cooperate with
    # the existing bridge writer lock while copying journal mirrors and backing
    # up SQLite, so they represent the same publication.
    lock=root/'live-bridge.lock'
    with lock.open('r') if lock.exists() else nullcontext() as locked:
        if locked is not None:
            fcntl.flock(locked,fcntl.LOCK_SH)
        shutil.copytree(root,output/'runtime-copy',
            ignore=shutil.ignore_patterns('sandbox.sqlite','sandbox.sqlite-wal','sandbox.sqlite-shm'))
        with sqlite3.connect((root.resolve()/'sandbox.sqlite').as_uri()+'?mode=ro',uri=True) as source:
            with sqlite3.connect(output/'runtime-copy'/'sandbox.sqlite') as destination:
                source.backup(destination)
    event=json.loads(sorted((output/'runtime-copy').glob('decision-*.json'))[-1].read_text())
    at=datetime.fromisoformat(event['as_of'])+timedelta(microseconds=1)
    service=resume_service(output/'runtime-copy',load_config(),as_of=at)
    kwargs={}
    def captured(**values):
        kwargs.update(values)
        raise CapturedSearch()
    try:
        with patch('run_analysis.adaptive.new_model_daily_harness.migrated_search',captured):
            try:
                service.harness.replan(at,reason='lossless exact-search diagnostic',publish=False)
            except CapturedSearch:
                pass
        assert kwargs,'Actual harness did not reach the search boundary'
        return service,kwargs
    except Exception:
        service.harness.sandbox.close()
        raise


def main(root,output,profile_candidates=0,order='original,lossless',reference_sources=None):
    from run_analysis.adaptive import shared_program_evaluator as evaluator
    if not hasattr(evaluator,'asdict') and reference_sources is None:
        raise ValueError('Production is optimized: provide --reference-sources for an ORIGINAL comparator')
    output.mkdir(parents=True,exist_ok=False)
    manifest=freeze_sources(output/'sources.zip',config=load_config(),entrypoint=__name__,
        inputs=(root/'scenario.json',*sorted(root.glob('decision-*.json')),*sorted(root.glob('snapshot-*.json')),
                *((reference_sources,) if reference_sources else ())))
    service,kwargs=capture(root,output)
    try:
        evaluate=kwargs['evaluate']
        if profile_candidates:
            proposals=kwargs['seed_programs'][:profile_candidates]
            profile=cProfile.Profile()
            profile.enable()
            for candidate in proposals:
                evaluate(candidate)
            profile.disable()
            profile.dump_stats(output/'baseline.profile')
            pstats.Stats(profile).strip_dirs().sort_stats('cumtime').print_stats(30)
        rows=[]
        for name in order.split(','):
            trace=sha256();count=0
            def tracked(candidate):
                nonlocal count
                result=evaluate(candidate)
                trace.update(json.dumps((candidate,result),sort_keys=True,allow_nan=False,separators=(',',':'),default=str).encode())
                trace.update(b'\n');count+=1
                return result
            stamp=perf_counter()
            with ExitStack() as stack:
                if name=='lossless':
                    stack.enter_context(fast_identities())
                elif reference_sources is not None:
                    stack.enter_context(original_identities(reference_sources))
                result=search(**dict(kwargs,evaluate=tracked))
            duration=perf_counter()-stamp
            normalized=normalize_search_result(result)
            row=dict(mode=name,seconds=duration,evaluations=count,
                ordered_evaluation_digest=trace.hexdigest(),result=normalized)
            rows.append(row)
            with (output/(name+'.json')).open('x') as stream:
                json.dump(row,stream,indent=2,default=str,allow_nan=False)
            print(json.dumps({k:row[k] for k in ('mode','seconds','evaluations','ordered_evaluation_digest')}),flush=True)
        by_mode={r['mode']:r for r in rows}
        original,optimized=by_mode['original'],by_mode['lossless']
        assert original['ordered_evaluation_digest']==optimized['ordered_evaluation_digest'],'Candidate order or exact evaluation changed'
        assert original['result']==optimized['result'],'Search output/coverage changed'
        report=dict(manifest=manifest,source=str(root),lossless=True,
            original_seconds=original['seconds'],optimized_seconds=optimized['seconds'],
            speedup=original['seconds']/optimized['seconds'],
            savings_fraction=1-optimized['seconds']/original['seconds'],
            evaluations=original['evaluations'],ordered_evaluation_digest=original['ordered_evaluation_digest'],
            order=order,excluded_wall_clock_fields=['seconds','refinement.pre_calendar_distance.seconds','refinement.post_calendar_distance.seconds'],
            same_full_search_result=True,live_changed=False)
        if reference_sources:
            report['original_source_archive']=str(reference_sources)
        with (output/'report.json').open('x') as stream:
            json.dump(report,stream,indent=2,allow_nan=False)
        print(json.dumps(report,indent=2),flush=True)
    finally:
        service.harness.sandbox.close()


if __name__=='__main__':
    parser=ArgumentParser()
    parser.add_argument('root',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--profile-candidates',type=int,default=0)
    parser.add_argument('--order',choices=('original,lossless','lossless,original'),default='original,lossless')
    parser.add_argument('--reference-sources',type=Path)
    args=parser.parse_args()
    main(args.root,args.output,args.profile_candidates,args.order,args.reference_sources)
