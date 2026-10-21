"""One prospective-volume sandbox build plus cheap fixed-block retention cases."""
from argparse import ArgumentParser
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from run_analysis.config import load_config
from run_analysis.adaptive.adaptive_candidate_pipeline import load_athlete, compile_easy_volume_phase
from run_analysis.adaptive.factory import build
from run_analysis.volume_retention import VolumeReviewBlock, review_volume_retention
from scripts.research_manifest import freeze_sources


def encode(value):
    return asdict(value) if is_dataclass(value) else value.isoformat()


def run(root,database,as_of,*,evaluations=10000):
    if root.exists():
        raise ValueError('New output directory required')
    config=load_config()
    manifest=freeze_sources(root.with_suffix('.sources.zip'),config=config,
        entrypoint='scripts.prototype_volume_target',inputs=(database,))
    inputs=load_athlete(database,config,as_of)
    baseline,candidate,proposal=compile_easy_volume_phase(inputs,as_of,timezone=config['timezone_default'])
    harness=build(root,config,inputs,candidate,as_of,evaluations=evaluations,
                  soft_horizon_taper=True,baseline_candidate=baseline)
    try:
        event=harness.replan(as_of,reason='prospective volume bootstrap experiment')
        scenarios={}
        stage=candidate['stage']
        for name,fraction,intent,complete,issued_fraction in (
            ('compliant',1.,'ordinary',True,1.),
            ('half_volume',.5,'ordinary',True,1.),
            ('confirmed_break',0.,'confirmed_break',True,0.),
            ('unknown_uploads',0.,'ordinary',False,1.),
            ('planner_under_delivery',.5,'ordinary',True,.5),
            ('planned_deload',.5,'planned_deload',True,1.)):
            rows=[]
            for i in range(3):
                start=as_of+timedelta(days=14*i)
                end=start+timedelta(days=14)
                rows.append(VolumeReviewBlock(start,end,stage.weekly_equivalent_miles*2*issued_fraction,
                    stage.weekly_equivalent_miles*2*fraction,stage.weekly_equivalent_miles*2*fraction,
                    complete,end,'explicit hypothetical scenario, not live athlete evidence',intent))
            scenarios[name]=[review_volume_retention(stage,rows,as_of+timedelta(days=day)) for day in (7,14,28,42)]
        report=dict(manifest=manifest,proposal=proposal,target_history=harness.target_history,
            planned_training=event['planned_training'],cost=event['cost'],seconds=event['seconds'],
            evaluations=event['evaluations'],local_complete=event['local_neighborhood_complete'],
            retention=scenarios,scope='one sandbox initial search; no daily-stability or reduction-stage acceptance')
        with (root/'volume-prototype.json').open('x') as stream:
            json.dump(report,stream,default=encode,indent=2)
        print(json.dumps({k:report[k] for k in ('cost','seconds','evaluations','local_complete')}))
        print(json.dumps(dict(volume=stage.weekly_equivalent_miles,long=stage.long_miles,
            planned=event['planned_training']['role_sequence']),default=encode))
        return report
    finally:
        harness.sandbox.close()


if __name__=='__main__':
    p=ArgumentParser()
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--database',type=Path,default=Path('data/run_analysis.sqlite'))
    p.add_argument('--as-of',type=datetime.fromisoformat,default=datetime(2026,9,27,8,tzinfo=timezone.utc))
    p.add_argument('--evaluations',type=int,default=10000)
    a=p.parse_args()
    run(a.output_dir,a.database,a.as_of,evaluations=a.evaluations)
