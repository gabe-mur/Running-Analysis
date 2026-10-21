"""Frozen sustained-volume ablation; optional one ordinary search, no publication."""
from argparse import ArgumentParser
from dataclasses import replace
from datetime import date,datetime,time,timedelta
from itertools import product
import json
from math import ceil
from pathlib import Path
from zoneinfo import ZoneInfo

from run_analysis.config import load_config
from run_analysis.continuous_training_dose import ContinuousDoseReference
from run_analysis.adaptive.decision_snapshot import migrate_snapshot_defaults,evaluate_snapshot
from run_analysis.adaptive.forecast_handoff import read_scope
from run_analysis.adaptive.new_model_daily_harness import Dose,planned_accounted,churn
from run_analysis.adaptive.sustained_volume_trial import sustained_volume_cost
from run_analysis.adaptive.continuous_search_adapter import search
from run_analysis.adaptive.stage_calendar_seeds import complete_stage_seeds
from scripts.research_manifest import freeze_sources


def load_case(root,index):
    event=json.loads((root/f'decision-{index:04d}.json').read_text())
    state=migrate_snapshot_defaults((root/event['snapshot_file']).read_text(),
        expected_fingerprint=event['snapshot_fingerprint'])['state']
    catalog={d['key']:Dose(**dict(d,phases=tuple(map(tuple,d['phases'])),
        phase_scenarios=tuple(map(tuple,d['phase_scenarios'])),
        forecast_scope=read_scope(d['forecast_scope']))) for d in event['stage_runtime']['catalog']}
    context=state.context
    # Existing longest purpose cadence, not a newly fitted fatigue constant.
    smoothing=ceil(max(7/context.policy.long_per_7,7/context.policy.quality_per_7))
    start=context.decision-timedelta(days=context.policy.volume_window_days+smoothing-2)
    changes={}
    extensions=[]
    for name in ('volume_targets','support_targets'):
        segments=getattr(context,name)
        if segments and segments[0].start>start:
            extensions.append(dict(series=name,old_start=segments[0].start,new_start=start,
                assumption='extend the same provisional historical reference; never raise it to a later stage'))
            segments=(replace(segments[0],start=start),*segments[1:])
        changes[name]=segments
    trial_context=replace(context,**changes)
    ref=ContinuousDoseReference('sandbox upload accounting')
    handoffs=event['forecast_handoff_by_dose']
    dates=tuple((context.decision+timedelta(days=i)).isoformat()
        for i in range((context.end-context.decision).days))
    physical={}
    def evaluate(program):
        for d,k in program:
            if (d,k) not in physical:
                physical[d,k]=planned_accounted(f'candidate:{d}:{k}',
                    datetime.combine(date.fromisoformat(d),time(18),ZoneInfo(state.timezone)),
                    catalog[k],handoffs[k],ref,available=state.as_of)
        r=evaluate_snapshot(state,tuple(physical[p] for p in program),tuple(k for _,k in program),reference=ref)
        scored=r['evaluation']
        trial=sustained_volume_cost(trial_context,scored,smoothing_days=smoothing)
        parts=scored.parts
        support=parts['easy_duration']['supporting_easy_volume']
        aggregate_funded=all(parts['delivery'][n]+1e-8>=parts['targets'][n] for n in ('volume','long','quality'))
        aggregate_funded=aggregate_funded and support['delivered_miles']+1e-8>=support['target_miles']
        return dict(cost=trial['cost'],eligible=r['decision_eligible'],original_cost=scored.total_cost,
            volume_cost=trial['volume_cost'],original_volume_cost=trial['previous_volume_cost'],
            losses=trial['losses'],aggregate_funded=aggregate_funded,
            first14_miles=sum(catalog[k].miles for d,k in program if date.fromisoformat(d)<context.decision+timedelta(days=14)),
            hard_miles=sum(catalog[k].miles for d,k in program if date.fromisoformat(d)<context.hard_end),
            total_miles=sum(catalog[k].miles for d,k in program))
    baseline=tuple(map(tuple,event['program']))
    priced=evaluate(baseline)
    if not priced['eligible'] or abs(priced['original_cost']-event['cost'])>1e-12:
        raise ValueError('Original archived score must reproduce before ablation')
    return dict(event=event,state=state,catalog=catalog,dates=dates,handoffs=handoffs,
        evaluate=evaluate,program=baseline,baseline=priced,smoothing_days=smoothing,extensions=extensions)


def contrasts(case):
    c=case['catalog'];ctx=case['state'].context
    options=[]
    for d,k in case['program']:
        if c[k].role!='easy' or date.fromisoformat(d)>=ctx.decision+timedelta(days=14):
            continue
        larger=[j for j in case['handoffs'] if c[j].role=='easy' and c[j].miles>=c[k].miles+.5-1e-8]
        if larger:
            options.append((d,k,min(larger,key=lambda j:(c[j].miles,j))))
    rows=[]
    for bits in product((False,True),repeat=len(options)):
        swaps={d:(b if bit else a) for (d,a,b),bit in zip(options,bits)}
        p=tuple((d,swaps.get(d,k)) for d,k in case['program'])
        rows.append(dict(program=p,**case['evaluate'](p)))
    return dict(index=case['event']['index']+1,as_of=case['event']['as_of'],
        smoothing_days=case['smoothing_days'],extensions=case['extensions'],
        baseline=case['baseline'],alternatives=rows)


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--search',action='store_true')
    a=p.parse_args()
    a.output_dir.mkdir(exist_ok=False)
    root=Path('output/easy-volume-phase-daily-21-20260928')
    pair=Path('output/funded-stage-repair-20260928/sandbox')
    specs=[(root,i) for i in (1,10,17,23,26)]+[(pair,i) for i in (27,28)]
    manifest=freeze_sources(a.output_dir/'sources.zip',config=load_config(),
        entrypoint='scripts.audit_sustained_volume',inputs=tuple(r/f'{kind}-{i:04d}.json'
            for r,i in specs for kind in ('decision','snapshot')))
    cases=[load_case(r,i) for r,i in specs]
    result=dict(manifest=manifest,contrasts=[contrasts(c) for c in cases[:5]])
    result['same_evidence_pair']=[dict(state_index=c['event']['index']+1,
        growth=c['evaluate'](cases[-2]['program']),refresh=c['evaluate'](cases[-1]['program'])) for c in cases[-2:]]
    for r in result['contrasts']:
        print(json.dumps(dict(index=r['index'],baseline=r['baseline'],
            best=min(r['alternatives'],key=lambda v:v['cost'])['cost'])),flush=True)
    print(json.dumps(dict(same_evidence_pair=result['same_evidence_pair'])),flush=True)
    if a.search:
        case=cases[-1];ctx=case['state'].context;c=case['catalog'];keys=tuple(case['handoffs'])
        seed=set()
        for target in (ctx.policy.ordinary_miles,ctx.policy.ordinary_miles/2):
            seed.add(min((k for k in keys if c[k].role=='easy'),key=lambda k:(abs(c[k].miles-target),k)))
        for role in ('long','quality'):
            seed.add(max((k for k in keys if c[k].role==role),key=lambda k:(
                sum(s for phase,s in c[k].phases if phase=='work') if role=='quality' else c[k].miles,k)))
        found=search(dates=case['dates'],keys=keys,previous=cases[-2]['program'],
            evaluate=case['evaluate'],budget=10000,hard_end=ctx.hard_end.isoformat(),
            roles={k:c[k].role for k in keys},seed_keys=tuple(sorted(seed)),
            previous_end=case['event']['end'],seed_programs=complete_stage_seeds(case['dates'],c,ctx.policy,
                supporting_miles=ctx.easy_duration_policy.supporting_easy_miles_per_7))
        result['ordinary_search']=dict(found,churn=churn(cases[-2]['program'],found['program'],ctx.decision,c))
        print(json.dumps({k:result['ordinary_search'][k] for k in ('cost','seconds','evaluations','local_neighborhood_complete','churn','selected')}),flush=True)
    with (a.output_dir/'audit.json').open('x') as f:
        json.dump(result,f,indent=2,default=lambda v:v.isoformat())


if __name__=='__main__':
    main()
