"""Read-only broad context audit; no sync, model publication or planner search."""
import argparse
import json
from pathlib import Path
import sqlite3
from time import perf_counter

from run_analysis.config import load_config
from run_analysis.performance_factors import analyze_performance_factors, load_factor_observations


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    config=load_config('config.yaml')
    with sqlite3.connect('file:data/run_analysis.sqlite?mode=ro',uri=True) as c:
        c.row_factory=sqlite3.Row;c.execute('BEGIN')
        start=perf_counter();rows,coverage=load_factor_observations(c,config)
        loaded=perf_counter();result=analyze_performance_factors(rows)
        result['coverage_summary']=coverage
        result['timing_seconds']=dict(load=round(loaded-start,3),analyze=round(perf_counter()-loaded,3))
        assert c.total_changes==0
    rendered=json.dumps(result,indent=2,allow_nan=False)
    if args.output:
        from run_analysis.privacy import private_file
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(rendered+'\n');private_file(args.output)
        print(f"Saved {result['observations']} run observations to {args.output}")
        for factor in result['primary'].get('factors',[]):
            value=factor['seconds_per_mile']
            print(f"{factor['label']}: {value:+.1f} sec/mi" if value is not None else f"{factor['label']}: not separable from time/effort")
        print('Chronological check:',json.dumps(result['chronological_check']))
    else:
        print(rendered)


if __name__=='__main__': main()
