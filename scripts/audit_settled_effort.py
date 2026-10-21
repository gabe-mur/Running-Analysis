"""All-run read-only audit of settled context before HR/pace standardization."""
from argparse import ArgumentParser
from collections import Counter,defaultdict
from copy import deepcopy
import json
from pathlib import Path
import sqlite3

import numpy as np

from run_analysis.config import load_config,resolve_project_path
from run_analysis.model_windows import load_overlapping_model_windows
from run_analysis.objective_modeling import _primary_window_weight
from run_analysis.pace_comparability import build_pace_comparability,compare_run_windows
from run_analysis.privacy import private_file
from run_analysis.read_cache import invariant_snapshot
from run_analysis.settled_effort import independent_sections
from scripts.audit_standardization_sampling import support_minutes


def audit(connection,config):
    base=build_pace_comparability(connection,config)
    originals={r['activity_id']:r for r in base['runs'] if r['included_in_trend'] and r['trend_weight']>0}
    windows,counters=load_overlapping_model_windows(connection,config,window_seconds=120,stride_seconds=60,
        check_elevation=True,settled_context=True)
    groups=defaultdict(list)
    for w in windows:
        if w['activity_id'] in originals:groups[w['activity_id']].append(w)
    results=[]
    for identity,rows in groups.items():
        before=originals[identity]
        result=compare_run_windows(rows,[_primary_window_weight(w,config) for w in rows],config,base['calibration'])
        np.testing.assert_allclose(result['pace_min_mile'],before['pace_min_mile'],rtol=1e-12,atol=1e-12)
        selected=[w for w in rows if w['settled_context']['usable']]
        count=independent_sections(selected)
        after=compare_run_windows(selected,[_primary_window_weight(w,config) for w in selected],config,base['calibration']) if count>=2 else None
        gain=[]
        if after:
            for fraction in (-.03,.03):
                shifted=[dict(w,moving_pace_min_mile=w['moving_pace_min_mile']/(1+fraction)) for w in selected]
                v=compare_run_windows(shifted,[_primary_window_weight(w,config) for w in shifted],config,base['calibration'])
                gain.append(dict(planted=100*fraction,recovered=100*(after['pace_min_mile']/v['pace_min_mile']-1)))
        reasons=Counter(reason for w in rows if not w['settled_context']['usable'] for reason in w['settled_context']['reasons'])
        results.append(dict(activity_id=identity,date=before['start_time'],before=before['pace_min_mile'],
            after=after['pace_min_mile'] if after else None,
            delta_seconds=60*(after['pace_min_mile']-before['pace_min_mile']) if after else None,
            retained_minutes=support_minutes(selected),original_minutes=support_minutes(rows),
            retained_windows=len(selected),original_windows=len(rows),reasons=dict(reasons),planted=gain,
            independent_sections=count,
            sections=[dict(minute=w['moving_minutes_into_run'],pace=w['moving_pace_min_mile'],hr=w['average_hr_bpm'],
                **w['settled_context']) for w in rows]))
    return dict(version='settled-effort-audit-v1',live_changed=False,runs=results,
        total=len(results),retained_runs=sum(r['after'] is not None for r in results),
        recent_runs=sum(r['date']>='2026-03-01' for r in results),
        recent_retained=sum(r['date']>='2026-03-01' and r['after'] is not None for r in results),
        calibration=base['calibration'],assumptions=[
            'Two-minute preceding and one-minute following context; 30-second terrain-effort and HR bins.',
            'Existing steady-benchmark tolerances: effort span 15%, HR change 6 bpm, HR range 15 bpm, stops 5 seconds.',
            'Context HR is not restricted to the score HR domain; delayed high HR can invalidate an earlier transient.',
            'The score still uses the original HR, movement, normalization, role and health rules after context checks.',
            'A failed section remains in activity history/load; insufficient settled evidence is missing, never an invented poor score.',
            'No fitness-outcome threshold or fast-run penalty; steady fast and slow inputs face identical relative checks.',
            'These context widths and steady-state tolerances are explicit engineering assumptions, not clinical cutoffs.',
            'At least two disjoint two-minute sections are needed to support the run score; overlapping windows do not satisfy repeatability.',
            'No fitted lag correction accepted merely for moving the April examples in a desired direction.'
        ])


def main():
    p=ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():p.error('New output path required')
    c=load_config('config.yaml');database=resolve_project_path(Path.cwd(),c['paths']['database'])
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as d:
        d.row_factory=sqlite3.Row
        with invariant_snapshot(d):r=audit(d,c);assert d.total_changes==0
    with args.output.open('x') as out:json.dump(r,out,indent=2,allow_nan=False)
    private_file(args.output)
    print(json.dumps({k:v for k,v in r.items() if k not in ('calibration','runs')},indent=2))
    print(json.dumps([{k:v for k,v in row.items() if k!='sections'} for row in r['runs'] if row['activity_id'] in (352,353,356,33,34,373,377,390)],indent=2))


if __name__=='__main__':main()
