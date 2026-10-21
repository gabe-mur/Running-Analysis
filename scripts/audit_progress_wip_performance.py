"""Read-only cold-reader timing and numerical parity; no planner calls."""
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sqlite3
from time import perf_counter

import numpy as np

from run_analysis import aerobic_sampling, read_cache
from run_analysis.config import load_config
from run_analysis.comparable_evidence import cached_comparable_evidence
from run_analysis.aerobic_performance import progress_from_comparability
from run_analysis.aerobic_candidate import progress_from_comparability as legacy_progress
from run_analysis.objective_modeling import _robust_weighted_speed
from run_analysis.progress_chart_context import cached_chart_context
from run_analysis.progress_wip import build_progress_wip


def main():
    root=Path.cwd();config=load_config(root/'config.yaml');end=datetime.now(timezone.utc)
    database=root/config['paths']['database']
    native=aerobic_sampling.robust_speed_samples
    def sequential(values,weights,samples):
        values=np.asarray(values);weights=np.asarray(weights)
        return np.array([_robust_weighted_speed(values[ix].tolist(),weights[ix].tolist()) for ix in samples])
    results={};timings={}
    with sqlite3.connect(database.as_uri()+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        with read_cache.invariant_snapshot(db):
            for name,implementation in [('before',sequential),('after',native)]:
                read_cache._ENTRIES.clear()
                aerobic_sampling.robust_speed_samples=implementation
                start=perf_counter();evidence=cached_comparable_evidence(db,config)
                evidence_seconds=perf_counter()-start
                visible=build_progress_wip(db,config,root,90,as_of=end,include_exploratory=False,comparison_report=evidence)
                visible['aerobic_performance']=progress_from_comparability(db,evidence,as_of=end,
                    history_days=90,chart_context=cached_chart_context(db))
                if name=='before':
                    legacy_progress(db,evidence,as_of=end,history_days=90,config=config)
                timings[name]=dict(evidence_seconds=evidence_seconds,total_seconds=perf_counter()-start)
                results[name]=dict(evidence=evidence,visible=visible)
                print(name,timings[name],flush=True)
            assert db.total_changes==0
    aerobic_sampling.robust_speed_samples=native
    max_delta=0.;numbers=0
    def compare(a,b,path='root'):
        nonlocal max_delta,numbers
        assert type(a) is type(b),(path,type(a),type(b))
        if isinstance(a,dict):
            assert a.keys()==b.keys(),path
            for key in a:compare(a[key],b[key],path+'.'+key)
        elif isinstance(a,(list,tuple)):
            assert len(a)==len(b),path
            for i,(v,w) in enumerate(zip(a,b)):compare(v,w,path+f'[{i}]')
        elif isinstance(a,(float,int)) and not isinstance(a,bool):
            numbers+=1;max_delta=max(max_delta,abs(a-b))
            assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10),(path,a,b)
        else:assert a==b,(path,a,b)
    compare(results['before'],results['after'])
    report=dict(as_of=end.isoformat(),timings=timings,numeric_values_compared=numbers,
        max_absolute_delta=max_delta,parity=True,planner_called=False,database_writes=0,
        current_pace=results['after']['visible']['aerobic_performance']['current'],
        progress_trend=results['after']['visible']['aerobic_performance']['progress_trend'])
    destination=root/'output/progress-wip-cold-performance-20261020.json'
    destination.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
