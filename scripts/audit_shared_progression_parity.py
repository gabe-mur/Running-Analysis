"""Read-only comparison of all archived general V7 reviews, without searches."""
from argparse import ArgumentParser
from dataclasses import asdict
from datetime import datetime, timedelta
import json
from pathlib import Path

from run_analysis.config import load_config
from run_analysis.adaptive.factory import resume_service
from run_analysis.adaptive.hr_reference import frozen_live_config
from run_analysis.adaptive.adaptive_stage_lifecycle import review_entries
from run_analysis.adaptive.smoothed_general_progression import propose
from scripts.trial_smoothed_progression import target
from scripts.run_adaptive_lifecycle import fork_seed_plan
from scripts.replay_adaptive_search import load_decision
from scripts.research_manifest import freeze_sources


def audit(source, output):
    output.mkdir(parents=True, exist_ok=False)
    root=output/'read-copy'
    at=fork_seed_plan(source,root)
    cfg=frozen_live_config(load_config(),root=source)
    summary=json.loads((source/'lifecycle-summary.json').read_text())
    indexes={e['index'] for e in summary['events']}
    service=resume_service(root,cfg,at)
    h=service.harness
    all_history=tuple(h.history); all_events=tuple(h.events)
    results=[]
    try:
        for i,event in enumerate(all_events):
            if event['index'] not in indexes:continue
            previous=all_events[i-1]['stage_runtime']
            clock=datetime.fromisoformat(event['as_of'])
            h.history=[r for r in all_history if r.available_at<=clock]
            h.events=list(all_events[:i])
            h.general_progression_state=previous.get('general_progression_state')
            h.target_history=tuple(map(target,previous['target_history']))
            recent=[r.miles for r in h.history if clock-r.start<=timedelta(days=30)
                and r.start<clock and h.capacity_eligibility.get(r.key,True)]
            review=propose(h,clock,review_entries(h,clock,original_schedule=True),longest=max(recent,default=None))
            expected=event['stage_runtime']['general_progression_state']
            if review['continuous_state'] != expected:
                differences={k:(expected.get(k),v) for k,v in review['continuous_state'].items()
                    if expected.get(k)!=v}
                raise ValueError('General review changed: '+repr((event['index'],differences)))
            if asdict(review['target']) != asdict(target(event['stage_runtime']['target_history'][-1])):
                raise ValueError('General target changed at '+str(event['index']))
            load_decision(source,event['index']+1)
            results.append(dict(event=event['index'],identical_state=True,identical_target=True,
                winner_repriced=True))
    finally:h.sandbox.close()
    manifest=freeze_sources(output/'parity.sources.zip',config=cfg,entrypoint=__name__,
        inputs=(source/'lifecycle-summary.json',))
    result=dict(status='identical_general_reviews',decisions=len(results),rows=results,
        planner_searches=0,live_changed=False,manifest=manifest,
        scope='Archived perfect-adherence general course, not new race stability evidence')
    with (output/'parity.json').open('x') as handle:json.dump(result,handle,indent=2)
    print(json.dumps(dict(status=result['status'],decisions=len(results),planner_searches=0)))


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args();audit(args.source,args.output)
