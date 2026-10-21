"""Audit the usable comparable-pace preview, without changing live outputs."""
import argparse
import json
from pathlib import Path
import sqlite3
from time import perf_counter

from run_analysis.config import load_config,resolve_project_path
from run_analysis.model_windows import load_model_window_sets
from run_analysis.pace_comparability import build_pace_comparability,validate_comparability
from run_analysis.privacy import private_file


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args()
    config=load_config('config.yaml');database=resolve_project_path(Path.cwd(),config['paths']['database'])
    start=perf_counter()
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as c:
        c.row_factory=sqlite3.Row;c.execute('BEGIN')
        result=build_pace_comparability(c,config)
        size=int(config['model'].get('window_seconds',300))
        windows,_=load_model_window_sets(c,config,(size,),check_elevation=True)
        result['chronological_validation']=validate_comparability(windows[size],config)
        assert c.total_changes==0
    result['elapsed_seconds']=round(perf_counter()-start,3)
    rendered=json.dumps(result,indent=2,allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(rendered+'\n');private_file(args.output)
    print('Comparable run estimates:',len(result['runs']))
    print('Calibration:',result['calibration']['parameters'])
    print('Heat:',result['calibration']['heat'])
    print('Chronological check:',result['chronological_validation'])
    print('Elapsed seconds:',result['elapsed_seconds'])
    if args.output:print('Saved:',args.output)


if __name__=='__main__':main()
