"""Compare local recorded elevation with cached maps, with no network/writes to athlete evidence."""
import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3
from time import perf_counter

from run_analysis.config import load_config,resolve_project_path
from run_analysis.elevation_validation import audit_elevation_profiles
from run_analysis.privacy import private_file
from run_analysis.read_cache import invariant_snapshot


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args()
    config=load_config('config.yaml');path=resolve_project_path(Path.cwd(),config['paths']['database'])
    start=perf_counter()
    with closing(sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True)) as c:
        c.row_factory=sqlite3.Row
        with invariant_snapshot(c):
            result=audit_elevation_profiles(c,config)
            assert c.total_changes==0
    result['elapsed_seconds']=round(perf_counter()-start,3)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');private_file(args.output)
    print(json.dumps({key:result[key] for key in ('by_source_and_format','recent_90_days_by_source','elapsed_seconds','scope')},indent=2))


if __name__=='__main__':main()
