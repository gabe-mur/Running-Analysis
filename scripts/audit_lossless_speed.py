"""Matched cold reader and full planner searches. Copies only; no live writes.

The old hot functions come solely from our trusted, hashed source archive.
No uploaded activities can supply executable code. All other dependencies are
the same in both runs. Compare exact payloads / ordered score fingerprints.
"""
from argparse import ArgumentParser
from contextlib import ExitStack,contextmanager
from datetime import datetime,timezone
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from unittest.mock import patch
from zipfile import ZipFile
import ast
import json
import sqlite3
import sys

from run_analysis.config import load_config
from run_analysis import read_cache


class NoReuse:
    def get(self): return None
    def set(self,value): return None
    def reset(self,token): pass


@contextmanager
def old_planner(archive_path):
    from run_analysis.adaptive import program_objective_trial,work_exposure_matching
    from run_analysis.adaptive import quality_forecast_runtime,execution_forecast_contract
    from run_analysis.adaptive import coached_pipeline
    from run_analysis.adaptive import continuation_boundary_trial
    from run_analysis.adaptive import program_evidence_adapter,shared_program_evaluator
    from run_analysis import program_delivery
    originals={}
    with ZipFile(archive_path) as archive:
        manifest=json.loads(archive.read('manifest.json'))
        for module,name in ((program_objective_trial,'rolling_volume_components'),
            (program_delivery,'_target_total'),(work_exposure_matching,'band_fraction'),
            (quality_forecast_runtime,'quality_target'),(execution_forecast_contract,'issue_execution'),
            (coached_pipeline,'coached_evaluator'),(continuation_boundary_trial,'continuation_boundary_cost'),
            (program_evidence_adapter,'evaluate_accounted'),(shared_program_evaluator,'evaluate_program')):
            path='src/'+module.__name__.replace('.','/')+'.py'
            raw=archive.read(path)
            assert sha256(raw).hexdigest()==manifest['source_sha256'][path]
            tree=ast.parse(raw)
            node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
            namespace=dict(vars(module))
            exec(compile(ast.Module(body=[node],type_ignores=[]),module.__file__,'exec'),namespace)
            originals[getattr(module,name)]=namespace[name]
    with ExitStack() as stack:
        # Include established imported aliases, especially the harness's issuer.
        for module in list(sys.modules.values()):
            if module is None or not getattr(module,'__name__','').startswith('run_analysis.'):
                continue
            for name,value in list(vars(module).items()):
                if callable(value) and not isinstance(value,type):
                    try: old=originals.get(value)
                    except TypeError: continue
                    if old is not None:
                        stack.enter_context(patch.object(module,name,old))
        yield


def encode(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False,default=str).encode()


def reader(output,archive_path):
    from run_analysis import model_windows,elevation_quality
    from run_analysis import comparable_evidence
    from run_analysis.comparable_evidence import cached_comparable_evidence
    from run_analysis.progress_wip import build_progress_wip
    from run_analysis.aerobic_performance import progress_from_comparability
    from run_analysis.progress_chart_context import cached_chart_context
    root=Path.cwd();config=load_config();end=datetime.now(timezone.utc)
    with ZipFile(archive_path) as archive:
        source_path='src/run_analysis/comparable_evidence.py'
        raw=archive.read(source_path);manifest=json.loads(archive.read('manifest.json'))
        assert sha256(raw).hexdigest()==manifest['source_sha256'][source_path]
        node=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name=='build_comparable_evidence')
        namespace=dict(vars(comparable_evidence))
        exec(compile(ast.Module(body=[node],type_ignores=[]),comparable_evidence.__file__,'exec'),namespace)
        original_builder=namespace['build_comparable_evidence']
    path=root/config['paths']['database'];records=[];baseline=None
    with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        with read_cache.invariant_snapshot(db):
            for mode in ('before','after'):
                read_cache._ENTRIES.clear();elevation_quality._read_tile.cache_clear()
                with ExitStack() as stack:
                    if mode=='before':
                        stack.enter_context(patch.object(model_windows,'_PREPARATION',NoReuse()))
                        stack.enter_context(patch.object(elevation_quality,'_TILE_LOOKUPS',NoReuse()))
                        stack.enter_context(patch.object(comparable_evidence,'build_comparable_evidence',original_builder))
                    started=perf_counter()
                    evidence=cached_comparable_evidence(db,config)
                    payload=build_progress_wip(db,config,root,90,as_of=end,
                        include_exploratory=False,comparison_report=evidence)
                    payload['aerobic_performance']=progress_from_comparability(db,evidence,
                        as_of=end,history_days=90,chart_context=cached_chart_context(db))
                    duration=perf_counter()-started
                data=encode(dict(evidence=evidence,payload=payload))
                if baseline is None: baseline=data
                else: assert data==baseline,'Reader output changed'
                record=dict(mode=mode,seconds=duration,sha256=sha256(data).hexdigest())
                records.append(record);print('reader',record,flush=True)
            assert db.total_changes==0
    result=dict(exact_payload_parity=True,live_writes=0,rows=records)
    (output/'reader.json').write_text(json.dumps(result,indent=2)+'\n')


def normalize_result(result):
    from scripts.benchmark_lossless_identities import normalize_search_result
    result=normalize_search_result(result)
    # An observed stopwatch, not search coverage, a score or a selection.
    result.get('refinement',{}).get('opening',{}).pop('seconds_before_refinement',None)
    return json.loads(encode(result))


def verify_planner_rows(rows,output):
    for row in rows:row['result']=normalize_result(row['result'])
    assert rows[0]['digest']==rows[1]['digest'],'Candidate order, score or admission changed'
    assert rows[0]['result']==rows[1]['result'],'Search result changed'
    report=dict(rows=rows,exact_ordered_evaluations=True,exact_full_search_result=True,live_writes=0,
        excluded_stopwatch_fields=['seconds','refinement.pre_calendar_distance.seconds',
            'refinement.post_calendar_distance.seconds','refinement.opening.seconds_before_refinement'])
    (output/'planner.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
    return report


def planner(output,archive_path,*,modes=('before','after'),compare_to=None):
    from scripts.benchmark_lossless_identities import capture,normalize_search_result
    from run_analysis.adaptive.hr_reference import frozen_live_config
    from run_analysis.adaptive.continuous_search_adapter import search
    from run_analysis.adaptive import program_objective_trial,work_exposure_matching
    from run_analysis.adaptive import continuation_boundary_trial
    from run_analysis import program_delivery
    config=load_config();path=Path(config['paths']['database']).resolve()
    with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as db:
        source=Path(json.loads(db.execute("SELECT value_json FROM app_state WHERE key='adaptive_runtime'").fetchone()[0])['root'])
    config=frozen_live_config(config,root=source)
    rows=[json.loads(compare_to.read_text())] if compare_to else []
    if rows:rows[0]['result']=normalize_result(rows[0]['result'])
    for mode in modes:
        work=output/mode;work.mkdir()
        for memo in (program_objective_trial._rolling_frame,work_exposure_matching._band_fraction_cached,
                     program_delivery._target_total_cached,continuation_boundary_trial._continuation_frame):
            memo.cache_clear()
        with ExitStack() as stack:
            if mode=='before':stack.enter_context(old_planner(archive_path))
            stack.enter_context(patch('scripts.benchmark_lossless_identities.load_config',return_value=config))
            start=perf_counter();service,args=capture(source,work)
            prepared=perf_counter()-start
            evaluate=args['evaluate'];trace=sha256();count=0
            def tracked(program):
                nonlocal count
                result=evaluate(program)
                trace.update(encode((program,result)));trace.update(b'\n');count+=1
                return result
            try:
                print('planner',mode,'prepared',prepared,'seconds; starting identical full search',flush=True)
                start=perf_counter();result=search(**dict(args,evaluate=tracked));seconds=perf_counter()-start
                normalized=normalize_result(result)
                row=dict(mode=mode,preparation_seconds=prepared,search_seconds=seconds,
                    total_seconds=prepared+seconds,evaluations=count,digest=trace.hexdigest(),result=normalized)
                rows.append(row)
                (work/'search.json').write_text(json.dumps(row,indent=2,default=str)+'\n')
                print('planner',{k:v for k,v in row.items() if k!='result'},flush=True)
            finally: service.harness.sandbox.close()
    verify_planner_rows(rows,output)


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('part',choices=('reader','planner','verify'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--before-sources',type=Path)
    p.add_argument('--modes',default='before,after');p.add_argument('--compare-to',type=Path)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.part=='reader':reader(args.output,args.before_sources)
    elif args.part=='planner':planner(args.output,args.before_sources,modes=tuple(args.modes.split(',')),compare_to=args.compare_to)
    else:verify_planner_rows([json.loads(args.compare_to.read_text()),
        json.loads((args.output/'after/search.json').read_text())],args.output)
