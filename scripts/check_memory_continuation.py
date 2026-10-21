"""Frozen real-HR attribution for memory-covered continuation. No publication."""
from argparse import ArgumentParser
from dataclasses import replace
from datetime import date,datetime,time,timedelta
from itertools import combinations
import json
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

from scripts.check_decimal_refinement import expand
from scripts.replay_adaptive_search import load_decision
from scripts.research_manifest import freeze_sources
from run_analysis.config import load_config
from run_analysis.adaptive.new_model_daily_harness import HarnessPolicy,planned_accounted
from run_analysis.adaptive.decision_snapshot import evaluate_snapshot
from run_analysis.continuous_training_dose import ContinuousDoseReference


def memory_context(ctx,*,guard=7,end=None):
    end=end or ctx.decision+timedelta(days=max((ctx.hard_end-ctx.decision).days,
        guard+ctx.policy.volume_window_days+ctx.volume_smoothing_days-2)+14)
    def extend(segments):
        if not segments:return ()
        assert segments[-1].end==ctx.end
        return (*segments[:-1],replace(segments[-1],end=end))
    return replace(ctx,continuation_guard_days=guard,end=end,
        volume_targets=extend(ctx.volume_targets),support_targets=extend(ctx.support_targets))


def main():
    parser=ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--decimal-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();a.output.mkdir(exist_ok=False)
    manifest=freeze_sources(a.output/'sources.zip',config=load_config(),entrypoint=__name__,
        inputs=(a.root/'scenario.json',a.root/'decision-0006.json',a.root/'decision-0007.json',
                a.decimal_root/'replay-0007.json'))
    stamp=perf_counter()
    before=load_decision(a.root,6);after=load_decision(a.root,7)
    state,catalog,hs,_=expand(after,HarnessPolicy(**json.loads((a.root/'scenario.json').read_text())['search_policy']))
    old=state.context
    new=memory_context(old)
    states={
        'old_before':replace(state,context=replace(old,hard_end=before['state'].context.hard_end,
            end=before['state'].context.end)),
        'old_after':state,
        # Same post-upload evidence/decision, emulate earlier endpoints. Six
        # remaining protected days correspond to yesterday's seven-day prefix.
        'guard_before':replace(state,context=replace(memory_context(old,guard=6,end=new.end-timedelta(days=1)),
            hard_end=before['state'].context.hard_end)),
        'guard_after':replace(state,context=new)}
    ref=ContinuousDoseReference('sandbox upload accounting');zone=ZoneInfo(state.timezone)
    dates=[old.decision+timedelta(days=i) for i in range((new.end-old.decision).days)]
    physical={(d.isoformat(),k):planned_accounted(f'candidate:{d}:{k}',datetime.combine(d,time(18),zone),
        catalog[k],h,ref,available=state.as_of) for d in dates for k,h in hs.items()}
    calls=0
    def evaluate(s,program):
        nonlocal calls
        calls+=1
        r=evaluate_snapshot(s,tuple(physical[p] for p in program),tuple(k for _,k in program),reference=ref)
        e=r['evaluation']
        return dict(cost=e.total_cost,eligible=r['decision_eligible'],fatigue=e.weighted_demand_cost,
            short=e.short_run_cost,losses=e.parts['shortfall_by_purpose'])
    # Common dated extension is an explicit comparison scaffold, not inferred
    # evidence or a searched tail. No scenario may modify it to favor its prefix.
    start=old.end
    support=old.support_targets[-1].miles_per_7_days
    tenths=round(support*10);sizes=[(tenths//3+(i<tenths%3))/10 for i in range(3)]
    extension=[]
    for day in dates:
        if not start<=day<new.end-timedelta(days=1):continue
        phase=(day-start).days%7
        key={0:'quality:ordinary',4:'long:8.4',1:f'easy:{sizes[0]:.1f}',
             2:f'easy:{sizes[1]:.1f}',6:f'easy:{sizes[2]:.1f}'}.get(phase)
        if key:extension.append((day.isoformat(),key))
    prior=tuple((d,k) for d,k in before['event']['program'] if d>=old.decision.isoformat())
    def transfer(program,n):
        return tuple((d,{'2026-10-02':f'easy:{n/10:.1f}',
            '2026-10-06':f'easy:{(70-n)/10:.1f}'}.get(d,k)) for d,k in program)
    controls={}
    for name,s in states.items():
        p=tuple(sorted((*prior,*extension))) if name.startswith('guard') else prior
        rows=[dict(first=n/10,second=(70-n)/10,**evaluate(s,transfer(p,n))) for n in range(26,45)]
        fixed={r['first']:r for r in rows}
        span=(s.context.end-s.context.stage_start).days
        controls[name]=dict(best=min((r for r in rows if r['eligible']),key=lambda r:r['cost']),
            eligible=sum(r['eligible'] for r in rows),pair_delta=fixed[2.6]['cost']-fixed[3.5]['cost'],
            raw_volume_delta={k:span*(fixed[2.6]['losses'][k]-fixed[3.5]['losses'][k])
                for k in ('volume','ordinary_easy')},curve=rows)
    assert controls['old_before']['pair_delta']>0>controls['old_after']['pair_delta']
    for k in ('volume','ordinary_easy'):
        assert abs(controls['guard_before']['raw_volume_delta'][k]-controls['guard_after']['raw_volume_delta'][k])<1e-12
    assert controls['guard_before']['best']['first']==controls['guard_after']['best']['first']
    # Reprice the same fine-search schedule and consolidate only three near easies.
    baseline=tuple(map(tuple,json.loads((a.decimal_root/'replay-0007.json').read_text())['program']))
    comparisons={}
    block=('2026-10-02','2026-10-03','2026-10-04')
    for label,s,p in (('old',state,baseline),('guard',states['guard_after'],tuple(sorted((*baseline,*extension))))):
        base=evaluate(s,p); rest=tuple(row for row in p if row[0] not in block)
        alternatives=[]
        for days in combinations(block,2):
            for n in range(11,60):
                row=tuple(sorted((*rest,(days[0],f'easy:{n/10:.1f}'),(days[1],f'easy:{(70-n)/10:.1f}'))))
                value=evaluate(s,row)
                if value['eligible']:alternatives.append(dict(program=[r for r in row if r[0] in block],**value))
        best=min(alternatives,key=lambda r:r['cost'])
        comparisons[label]=dict(baseline=base,best_consolidated=best,eligible=len(alternatives),
            delta={k:best[k]-base[k] for k in ('cost','fatigue','short')})
    report=dict(manifest=manifest,controls=controls,comparisons=comparisons,
        shared_extension=extension,evaluations=calls,seconds=perf_counter()-stamp,
        scope='Frozen full evaluator and eligibility with explicit common dated extension; not a search or daily-stability proof')
    (a.output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(controls={k:{n:v for n,v in r.items() if n!='curve'} for k,r in controls.items()},
        comparisons=comparisons,seconds=report['seconds'],evaluations=calls),indent=2))


if __name__=='__main__':main()
