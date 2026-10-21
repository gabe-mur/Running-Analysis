"""Read-only comparison integration audit; never refreshes the planner."""
from argparse import ArgumentParser
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sqlite3
from time import perf_counter

from run_analysis.comparable_evidence import cached_comparable_evidence, comparable_progress_data
from run_analysis.config import load_config, resolve_project_path
from run_analysis.progress import prepare_progress_data, build_progress
from run_analysis.aerobic_performance import progress_from_comparability
from run_analysis.pace_comparability import comparison_view
from run_analysis.read_cache import invariant_snapshot


def main():
    p=ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference-trial',type=Path);args=p.parse_args()
    if args.output.exists():p.error('Use a new output path')
    config=load_config('config.yaml');database=resolve_project_path(Path.cwd(),config['paths']['database'])
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        with invariant_snapshot(db):
            started=perf_counter();report=cached_comparable_evidence(db,config);cold=perf_counter()-started
            started=perf_counter();cached=cached_comparable_evidence(db,config);warm=perf_counter()-started
            assert report==cached
            as_of=datetime.now(timezone.utc)
            short=progress_from_comparability(db,report,as_of=as_of,history_days=28)
            long=progress_from_comparability(db,report,as_of=as_of,history_days=365)
            assert short.get('current')==long.get('current')
            raw=prepare_progress_data(db);adapted=comparable_progress_data(raw,report)
            assert raw.sessions==adapted.sessions and raw.details==adapted.details
            progress=build_progress(db,90,as_of=as_of,config=config,prepared=raw,comparison_report=report)
            comparison=comparison_view(report,90,as_of)
            assert progress.current_pace.minutes_per_mile==comparison['performance']['current']['pace_min_mile']
            by_id={r['activity_id']:r for r in report['runs']}
            assert all(v.standardized_pace_min_mile==by_id[v.activity_id]['pace_min_mile']
                for v in progress.series if v.standardized_pace_min_mile is not None)
            assert db.total_changes==0
            result=dict(version=report['version'],source_read_only=True,database_writes=db.total_changes,
                runs=len(report['runs']),trend_runs=sum(r['included_in_trend'] for r in report['runs']),
                original_windows=sum(r.get('hr_response',{}).get('original_windows',0) for r in report['runs']),
                retained_windows=sum(r.get('hr_response',{}).get('retained_windows',0) for r in report['runs']),
                temperature=report['temperature_response'],current=short.get('current'),
                current_invariant_across_viewports=True,recorded_sessions_unchanged=True,
                progress_comparison_equal=True,cold_seconds=cold,cached_seconds=warm,
                observations=report['runs'],report=report)
            if args.reference_trial:
                previous=json.loads(args.reference_trial.read_text())
                old={r['activity_id']:r for r in previous['observations'] if r['available']}
                scale=report['temperature_response']['scale']
                assert abs(scale-previous['fits']['air']['scale'])<1e-10
                errors=[60*(r['pace_min_mile']-old[r['activity_id']]['pace_no_weather']*
                    math.exp(-scale*old[r['activity_id']]['air_burden'])) for r in report['runs'] if r['activity_id'] in old]
                result['offline_parity']=dict(runs=len(errors),maximum_absolute_difference_seconds=max(map(abs,errors)))
                assert result['offline_parity']['maximum_absolute_difference_seconds']<1e-8
    fd=os.open(args.output,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as out:json.dump(result,out,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k not in ('observations','report')},indent=2))


if __name__=='__main__':main()
