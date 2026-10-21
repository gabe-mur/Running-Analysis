"""Frozen score decomposition, not search or a valid alternative publication.

Hold the earlier whole calendar fixed and exchange only the witnessed doses.
Cross earlier/later forecasts with earlier/later clock context to attribute a
ranking reversal. No model coefficients, future HR or continuity adjustments.
"""
from argparse import ArgumentParser
from dataclasses import replace
from datetime import timedelta
import json
from pathlib import Path

from scripts.replay_adaptive_search import load_decision
from scripts.research_manifest import freeze_sources


def run(root,output,before_number,after_number):
    output.mkdir(exist_ok=False)
    before=load_decision(root,before_number);after=load_decision(root,after_number)
    base=tuple(tuple(p) for p in before['event']['program'])
    edited=dict(base)
    for row in after['event']['churn']['7']:
        if row['after'] is None:edited.pop(row['day'],None)
        else:edited[row['day']]=row['after']
    alternative=tuple(sorted(edited.items()))
    rows=[]
    # Clock/target/accounting fields move together. Purpose forecast evidence
    # stays with its physical forecast; do not mix the two representations.
    for forecast_name,forecast in (('earlier',before),('later',after)):
        for clock_name,clock in (('earlier',before),('later',after)):
            context=replace(clock['state'].context,
                quality_programme=forecast['state'].context.quality_programme,
                continuation_boundary_references=forecast['state'].context.continuation_boundary_references,
                steady_continuation_references=forecast['state'].context.steady_continuation_references)
            state=replace(forecast['state'],context=context)
            priced=[]
            for program in (base,alternative):
                result=forecast['evaluate_details'](program,state_override=state)
                e=result['evaluation'];p=e.parts
                priced.append(dict(cost=e.total_cost,fatigue=e.weighted_demand_cost,
                    delivery=e.delivery_cost,short=e.short_run_cost,
                    purpose=p['shortfall_by_purpose'],
                    quality={k:p['quality_programme'][k] for k in
                        ('shared_opportunity_loss','descriptor_loss','loss')},
                    eligible=result['decision_eligible']))
            rows.append(dict(forecasts=forecast_name,clock=clock_name,before=priced[0],after=priced[1],
                delta={k:priced[1][k]-priced[0][k] for k in ('cost','fatigue','delivery','short')}))
    report=dict(before_event=before_number,after_event=after_number,rows=rows,base=base,
        alternative=alternative,scope=__doc__,live_changed=False)
    from run_analysis.adaptive.direct_quality_load import POLICY
    sources={}
    for name,case in (('earlier',before),('later',after)):
        at=case['state'].as_of
        sources[name]={h.key:h for h in case['state'].phase_history
            if at-timedelta(days=POLICY.current_days)<=h.start<h.end<at and h.available_at<=at and h.phases}
    physical={r.key:r for r in before['state'].history}
    report['evidence_change']=dict(physical_history_changed=before['state'].history!=after['state'].history,
        phase_history_changed=before['state'].phase_history!=after['state'].phase_history,
        window_days=POLICY.current_days,
        retired=[dict(key=k,start=sources['earlier'][k].start.isoformat(),role=physical[k].role)
            for k in sorted(sources['earlier'].keys()-sources['later'].keys())],
        added=sorted(sources['later'].keys()-sources['earlier'].keys()))
    report['manifest']=freeze_sources(output/'sources.zip',config={},entrypoint=__name__,
        inputs=(root/'scenario.json',root/f'decision-{before_number:04d}.json',
            root/f'decision-{after_number:04d}.json',root/before['event']['snapshot_file'],
            root/after['event']['snapshot_file']))
    with (output/'report.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps(rows,indent=2))


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--before',type=int,required=True);p.add_argument('--after',type=int,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    run(a.root,a.output,a.before,a.after)
