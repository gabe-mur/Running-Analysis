"""Reprice a later native-workout edit in every earlier frozen input state.

Hindsight attribution only: no search, publication, physiological claim or live
access. Each archived winner must reproduce before its edited counterfactual.
"""
from argparse import ArgumentParser
import json
from pathlib import Path
from time import perf_counter

from scripts.replay_adaptive_search import load_decision
from scripts.research_manifest import freeze_sources
from run_analysis.adaptive.continuous_search_adapter import canonical


def run(root,output,*,event_number):
    output.mkdir(exist_ok=False)
    event=json.loads((root/f'decision-{event_number:04d}.json').read_text())
    edits=event['churn']['7']
    if not edits:raise ValueError('Declared event has no visible witness edits')
    stamp=perf_counter();rows=[]
    for i in range(1,event_number+1):
        case=load_decision(root,i)
        original=canonical(case['event']['program'])
        before,after=dict(original),dict(original)
        for edit in edits:
            day=edit['day']
            for target,key in ((before,edit['before']),(after,edit['after'])):
                target.pop(day,None)
                if key is not None:target[day]=key
        def price(p):
            result=case['evaluate_details'](canonical(p.items()))
            e=result['evaluation']
            return dict(eligible=result['decision_eligible'],cost=float(e.total_cost),
                fatigue=e.weighted_demand_cost,delivery=e.delivery_cost,short=e.short_run_cost)
        a,b=price(before),price(after)
        rows.append(dict(event=i,as_of=case['event']['as_of'],before=a,after=b,
            delta={k:b[k]-a[k] for k in ('cost','fatigue','delivery','short')}))
    report=dict(witness_event=event_number,edits=edits,rows=rows,
        seconds=perf_counter()-stamp,scope=__doc__,live_changed=False)
    report['manifest']=freeze_sources(output/'sources.zip',config={},entrypoint=__name__,
        inputs=(root/'scenario.json',*sorted(root.glob('decision-*.json')),
            *sorted(root.glob('snapshot-*.json'))))
    with (output/'report.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps(dict(seconds=report['seconds'],rows=rows),indent=2))


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path);p.add_argument('--event',type=int,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    run(a.root,a.output,event_number=a.event)
