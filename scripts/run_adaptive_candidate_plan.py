"""One real-evidence offline build; later lifecycle tests use this same caller."""
from argparse import ArgumentParser
from datetime import datetime,timezone
import json
from pathlib import Path
from time import perf_counter

from run_analysis.config import load_config
from scripts.adaptive_candidate_pipeline import load_athlete,compile_athlete
from run_analysis.adaptive.adaptive_candidate_pipeline import compile_easy_volume_phase
from scripts.research_manifest import freeze_sources
from run_analysis.adaptive.factory import build


if __name__=='__main__':
    p=ArgumentParser()
    p.add_argument('--database',type=Path,default=Path('data/run_analysis.sqlite'))
    p.add_argument('--as-of',type=datetime.fromisoformat,default=datetime(2026,9,27,8,tzinfo=timezone.utc))
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--evaluations',type=int,default=3000)
    p.add_argument('--easy-volume-phase',action='store_true')
    p.add_argument('--continuation-boundary',action='store_true')
    p.add_argument('--single-volume',action='store_true')
    p.add_argument('--purpose-continuation',action='store_true',help='Use the v19 purpose-handoff accounting contract')
    p.add_argument('--volume-linear-coefficient',type=float,default=None,
        help='Explicit volume utility coefficient; new single-volume builds default to 0.1, zero is the baseline')
    p.add_argument('--soft-horizon-taper',action='store_true')
    a=p.parse_args()
    stamp=perf_counter()
    config=load_config()
    # Freeze before computation; later workspace edits must not relabel the
    # already-running process as if it had used newer bytes.
    manifest=freeze_sources(a.output_dir.with_suffix('.sources.zip'),config=config,
        entrypoint='scripts.run_adaptive_candidate_plan',inputs=(a.database,))
    inputs=load_athlete(a.database,config,a.as_of)
    baseline=None
    if a.easy_volume_phase:
        baseline,candidate,_=compile_easy_volume_phase(inputs,a.as_of,timezone=config['timezone_default'])
    else:
        candidate=compile_athlete(inputs,a.as_of)
    harness=build(a.output_dir,config,inputs,candidate,a.as_of,evaluations=a.evaluations,
        baseline_candidate=baseline,continuation_boundary=a.continuation_boundary,
        single_volume=a.single_volume,soft_horizon_taper=a.soft_horizon_taper,
        volume_linear_coefficient=a.volume_linear_coefficient,purpose_continuation=a.purpose_continuation)
    try:
        event=harness.replan(a.as_of,reason='first real-candidate whole-calendar build')
        report=dict(seconds=perf_counter()-stamp,search_seconds=event['seconds'],
            cost=event['cost'],volume=event['volume'],program=event['program'],
            beam=event['calendar_beam'],evaluations=event['evaluations'],
            budget_exhausted=event['budget_exhausted'],manifest=manifest)
        report.update(as_of=a.as_of.isoformat(),model_version=event['model_version'],
            decision=event['decision'],hard_end=event['hard_end'],end=event['end'],
            planned_training=event['planned_training'],
            volume_linear_coefficient=harness.policy.volume_linear_coefficient,
            scope='Current live history, isolated connected planner. Explicit volume curve persisted; compression reward not connected; not published to live server.')
        with (a.output_dir/'build-summary.json').open('x') as stream:
            json.dump(report,stream,indent=2,default=lambda value:value.isoformat())
        print(json.dumps(report,default=lambda value:value.isoformat()))
    finally:
        harness.sandbox.close()
