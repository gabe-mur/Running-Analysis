"""Build the fixed weather-attribution prototype without modifying live data."""
import argparse
import json
from pathlib import Path
import sqlite3
from time import perf_counter

from run_analysis.config import load_config
from run_analysis.weather_impact import analyze_weather_impact, load_weather_observations


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args()
    config=load_config('config.yaml')
    with sqlite3.connect('file:data/run_analysis.sqlite?mode=ro',uri=True) as c:
        c.row_factory=sqlite3.Row
        c.execute('BEGIN') # One coherent snapshot, even if an upload completes.
        start=perf_counter()
        rows,coverage=load_weather_observations(c,config)
        extracted=perf_counter()
        result=analyze_weather_impact(rows)
        finished=perf_counter()
        assert c.total_changes==0
    result['coverage']=coverage
    result['live_database_modified']=False
    result['timing_seconds']={'load_observations':round(extracted-start,3),'analyze':round(finished-extracted,3)}
    rendered=json.dumps(result,indent=2);print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(rendered+'\n')


if __name__=='__main__': main()
