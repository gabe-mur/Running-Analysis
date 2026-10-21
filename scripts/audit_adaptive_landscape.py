"""Frozen full-score landscape probe, not a production optimizer or daily test.

Equal proposal budgets compare multiple starting regions. Exhaustive blocks are
conditional on their frozen exterior. Neither is a global optimality claim.
"""
from argparse import ArgumentParser
from collections import Counter
from dataclasses import replace
from datetime import date, datetime, time
from itertools import product
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from random import Random
from time import perf_counter
from zoneinfo import ZoneInfo
from zipfile import ZipFile

from run_analysis.config import load_config
from run_analysis.continuous_training_dose import ContinuousDoseReference
from run_analysis.adaptive.decision_snapshot import evaluate_snapshot
from run_analysis.adaptive.new_model_daily_harness import neighbors, planned_accounted, churn
from run_analysis.adaptive.stage_calendar_seeds import complete_stage_seeds
from scripts.replay_adaptive_search import load_decision
from scripts.research_manifest import freeze_sources


def canonical(program):
    p=tuple(sorted(tuple(x) for x in program))
    if len(dict(p))!=len(p):
        raise ValueError('Duplicate date')
    return p


def families(dates,catalog,policy,supporting,anchor,*,count=256,seed=20260928):
    """Count-stratified complete calendars; RNG is seeded research sampling only."""
    rng=Random(seed)
    groups={r:sorted((k for k,d in catalog.items() if d.role==r)) for r in ('easy','long','quality')}
    if count<1 or len(dates)<7 or not all(groups.values()):
        raise ValueError('Positive sample count, >=7 days and all roles required')
    n=len(dates)
    def dose(role,target):
        return min(groups[role],key=lambda k:(abs(catalog[k].miles-target),k))
    def funded(occupied,easy_days,mode):
        target=max(supporting*n/7,policy.weekly_miles*n/7-sum(catalog[k].miles for k in occupied.values()))
        weights=[1. if mode==0 else rng.uniform(.3,1.7)**mode for _ in easy_days]
        for day,w in zip(easy_days,weights):
            occupied[day]=dose('easy',target*w/sum(weights))
        return canonical(occupied.items())
    output={'stage_lattice':list(complete_stage_seeds(dates,catalog,policy,
        supporting_miles=supporting,limit=count)),
        'nonperiodic_spaced':[], 'uniform_counts':[], 'coupled_incumbent':[]}
    for i in range(count):
        # Cover 2--6 runs/7 days, independently perturb key-session counts.
        runs=min(n,max(3,round(n*(2+(i%9)/2)/7)))
        longs=min(runs-2,max(1,round(n*policy.long_per_7/7)+(i//9)%3-1))
        qualities=min(runs-longs-1,max(1,round(n*policy.quality_per_7/7)+(i//27)%3-1))
        for family in ('nonperiodic_spaced','uniform_counts'):
            occupied={}
            if family=='uniform_counts':
                selected=rng.sample(list(dates),runs)
                for j,day in enumerate(selected[:longs+qualities]):
                    role='long' if j<longs else 'quality'
                    occupied[day]=rng.choice(groups[role])
                easy_days=selected[longs+qualities:]
            else:
                for role,total in (('long',longs),('quality',qualities)):
                    for j in range(total):
                        wanted=int((j+rng.random())*n/total)
                        at=min((x for x in range(n) if dates[x] not in occupied),key=lambda x:(abs(x-wanted),x))
                        occupied[dates[at]]=rng.choice(groups[role])
                easy_days=rng.sample([d for d in dates if d not in occupied],runs-len(occupied))
            output[family].append(funded(occupied,easy_days,i%3))
        changed=dict(anchor)
        for _ in range(2+i%5):
            if not changed:
                break
            old=rng.choice(sorted(changed))
            new=rng.choice(list(dates))
            key=changed.pop(old)
            if new in changed:
                changed[old]=changed[new]
            changed[new]=key
        # Both dates and doses change together; not just a +/-1-day search.
        for day in sorted(changed):
            if catalog[changed[day]].role!='easy' and rng.random()<.3:
                changed[day]=rng.choice(groups[catalog[changed[day]].role])
        easy_days=[d for d in sorted(changed) if catalog[changed[d]].role=='easy']
        fixed={d:k for d,k in changed.items() if d not in easy_days}
        output['coupled_incumbent'].append(funded(fixed,easy_days,i%3) if easy_days else canonical(fixed.items()))
    return {name:tuple(dict.fromkeys(rows)) for name,rows in output.items()}


def descend(start,dates,keys,price,quota):
    """Same full-neighborhood steepest-descent budget for every starting lane.

    Unique proposals count even if shared physical caching makes them free.
    A candidate discovered in another lane never silently replaces this start.
    """
    current=canonical(start)
    value=price(current)
    visited={current}
    path=[dict(cost=value['cost'],program=current)]
    used=0
    while True:
        best=current
        best_value=value
        complete=True
        for p in neighbors(current,dates,keys):
            p=canonical(p)
            if p in visited:
                continue
            if used>=quota:
                complete=False
                break
            visited.add(p)
            used+=1
            v=price(p)
            if v['eligible'] and v['cost']<best_value['cost']:
                best,best_value=p,v
        if best!=current:
            current,value=best,best_value
            path.append(dict(cost=value['cost'],program=current))
        elif complete:
            return dict(program=current,cost=value['cost'],path=path,proposals=used,quota=quota,
                status='coordinate_complete')
        if not complete:
            return dict(program=current,cost=value['cost'],path=path,proposals=used,quota=quota,
                status='budget_exhausted')


def block_candidates(program,days,keys):
    """Every rest/dose assignment for these dates, with the exterior unchanged."""
    fixed=tuple(p for p in program if p[0] not in days)
    for assignment in product((None,*keys),repeat=len(days)):
        yield canonical((*fixed,*((d,k) for d,k in zip(days,assignment) if k is not None)))


def shape(program,catalog,dates):
    indices={d:i for i,d in enumerate(dates)}
    offsets=[indices[d] for d,_ in program]
    gaps=[b-a for a,b in zip(offsets,offsets[1:])]
    counts=Counter(catalog[k].role for _,k in program)
    return dict(runs=len(program),miles=sum(catalog[k].miles for _,k in program),
        counts=dict(counts),longest_internal_gap=max(gaps,default=None),
        consecutive_pairs=sum(g==1 for g in gaps),
        largest_run_share=max((catalog[k].miles for _,k in program),default=0)/
            max(1e-12,sum(catalog[k].miles for _,k in program)))


def backward(earlier,later,programs):
    """Retrospective candidate injection, with each origin's original evidence."""
    reference=ContinuousDoseReference('sandbox upload accounting')
    prefix=[tuple(p) for p in earlier['event']['program'] if p[0]<later['dates'][0]]
    old,new=earlier['state'],later['state']
    contexts=(('earlier',old,earlier),
        ('later_fixed_end',replace(new,context=replace(new.context,
            hard_end=old.context.hard_end,end=old.context.end)),later),
        ('later_rolling_end',new,later))
    result={}
    for label,state,case in contexts:
        result[label]={}
        for name,program in programs.items():
            combined=canonical({**dict(prefix),**dict(program)}.items())
            included=tuple((d,k) for d,k in combined if state.context.decision.isoformat()<=d<state.context.end.isoformat())
            rows=tuple(planned_accounted(f'candidate:{d}:{k}',
                datetime.combine(date.fromisoformat(d),time(18),ZoneInfo(state.timezone)),
                case['catalog'][k],case['handoffs'][k],reference,available=state.as_of) for d,k in included)
            v=evaluate_snapshot(state,rows,tuple(k for _,k in included),reference=reference)
            score=v['evaluation']
            result[label][name]=dict(cost=score.total_cost,eligible=v['decision_eligible'],
                demand=score.weighted_demand_cost,delivery=score.delivery_cost)
    return result


def verify_finalist(root,audit_output):
    """Finish the new winner's local check within the original unused call cap.

    Keep the original result intact and archive this follow-up separately.
    Reuse only the same frozen-origin physical evaluations, never later evidence.
    """
    report=json.loads((audit_output/'result.json').read_text())
    with ZipFile(report['manifest']['path']) as archive:
        inputs=json.loads(archive.read('manifest.json'))['input_sha256']
    expected={Path(p).resolve():digest for p,digest in inputs.items()}
    for p in (root/'scenario.json',*(root/f'decision-{i:04d}.json' for i in report['events'])):
        if expected.get(p.resolve())!=sha256(p.read_bytes()).hexdigest():
            raise ValueError('Cannot reuse cached scores from a different frozen decision')
    for p,digest in expected.items():
        if sha256(p.read_bytes()).hexdigest()!=digest:
            raise ValueError('Frozen experiment inputs changed')
    early,case=[load_decision(root,i) for i in report['events']]
    output=audit_output/'finalist-check'
    output.mkdir(exist_ok=False)
    manifest=freeze_sources(output/'sources.zip',config=load_config(),
        entrypoint='scripts.audit_adaptive_landscape.verify_finalist',
        inputs=(audit_output/'result.json',audit_output/'evaluations.jsonl',root/'scenario.json',
            *(root/f'decision-{i:04d}.json' for i in report['events']),
            *(root/c['event']['snapshot_file'] for c in (early,case))))
    cache={}
    for line in (audit_output/'evaluations.jsonl').read_text().splitlines():
        row=json.loads(line)
        cache[canonical(row['program'])]={k:row[k] for k in ('cost','eligible')}
    before=len(cache)
    started=perf_counter()
    with (output/'evaluations.jsonl').open('x') as journal:
        def price(p):
            p=canonical(p)
            if p not in cache:
                if len(cache)>=report['budget']:
                    raise RuntimeError('Original global cap reached')
                v=case['evaluate'](p)
                cache[p]=v
                journal.write(json.dumps(dict(program=p,**v))+'\n')
            return cache[p]
        original=canonical(report['best_program'])
        verified=case['evaluate'](original)
        if not verified['eligible'] or abs(verified['cost']-report['best_cost'])>1e-12:
            raise ValueError('Frozen winner failed to reproduce')
        result=descend(original,case['dates'],tuple(sorted(case['handoffs'])),price,
            min(2000,report['budget']-len(cache)))
        result.update(manifest=manifest,additional_evaluations=len(cache)-before,
            combined_evaluations=len(cache),seconds=perf_counter()-started,
            churn=churn(early['event']['program'],result['program'],case['state'].context.decision,case['catalog']),
            backward=backward(early,case,dict(archived_winner=case['event']['program'],
                prior=early['event']['program'],best_found=result['program'])))
        with (output/'result.json').open('x') as stream:
            json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ('cost','status','additional_evaluations','combined_evaluations','seconds','churn')}),flush=True)
    return result


def run(root,output,*,events=(3,4),per_family=256,lane_budget=2000,budget=30000,seed=20260928):
    started=perf_counter()
    early,case=[load_decision(root,i) for i in events]
    output.mkdir(exist_ok=False)
    inputs=[root/'scenario.json',*(root/f'decision-{i:04d}.json' for i in events),
        *(root/c['event']['snapshot_file'] for c in (early,case))]
    manifest=freeze_sources(output/'sources.zip',config=load_config(),
        entrypoint='scripts.audit_adaptive_landscape',inputs=inputs)
    dates,catalog=case['dates'],case['catalog']
    keys=tuple(sorted(case['handoffs']))
    catalog={k:catalog[k] for k in keys}
    incumbent=canonical(case['event']['program'])
    prior=canonical(p for p in early['event']['program'] if p[0] in dates)
    cache={}
    journal=(output/'evaluations.jsonl').open('x')
    phase='controls'
    def price(p):
        p=canonical(p)
        if p not in cache:
            if len(cache)>=budget:
                raise RuntimeError('Global cap reached; journal retained, no acceptance claim')
            if any(d not in dates or k not in keys for d,k in p):
                raise ValueError('Outside frozen domain')
            v=case['evaluate'](p)
            if not isfinite(v['cost']):
                raise ValueError('Nonfinite evaluator result')
            cache[p]=v
            journal.write(json.dumps(dict(index=len(cache),phase=phase,program=p,**v))+'\n')
        return cache[p]
    try:
        price(incumbent)
        prior_options=[prior]
        for d in dates:
            if d>=early['event']['end'] and d not in dict(prior):
                prior_options.extend(canonical((*prior,(d,k))) for k in keys)
        prior_valid=[p for p in prior_options if price(p)['eligible']]
        controls={'archived_winner':incumbent}
        if prior_valid:
            controls['prior_extension']=min(prior_valid,key=lambda p:(price(p)['cost'],p))
        sampled=families(dates,catalog,case['state'].context.policy,
            case['state'].context.easy_duration_policy.supporting_easy_miles_per_7,
            incumbent,count=per_family,seed=seed)
        summaries={}
        starts=[(name,p) for name,p in controls.items()]
        for name,programs in sampled.items():
            phase='sample:'+name
            valid=[p for p in programs if price(p)['eligible']]
            ordered=sorted(valid,key=lambda p:(price(p)['cost'],p))
            summaries[name]=dict(sampled=len(programs),eligible=len(valid),
                best_cost=price(ordered[0])['cost'] if ordered else None)
            chosen=[]
            for p in ordered:
                calendar={(d,catalog[k].role) for d,k in p}
                if any(len(calendar.symmetric_difference({(d,catalog[k].role) for d,k in q}))<4 for q in chosen):
                    continue
                chosen.append(p)
                if len(chosen)==2:
                    break
            starts.extend((name,p) for p in chosen)
        print(json.dumps(dict(stage='sampled',evaluations=len(cache),families=summaries)),flush=True)
        lanes=[]
        for i,(name,p) in enumerate(starts):
            phase=f'lane:{i}:{name}'
            before=len(cache)
            lane=descend(p,dates,keys,price,lane_budget)
            lane.update(family=name,evaluations=len(cache)-before,
                shape=shape(lane['program'],catalog,dates))
            lanes.append(lane)
            print(json.dumps(dict(stage=phase,cost=lane['cost'],status=lane['status'],
                evaluations=len(cache),seconds=perf_counter()-started)),flush=True)
            journal.flush()
        valid=lambda: (p for p,v in cache.items() if v['eligible'])
        best=min(valid(),key=lambda p:(price(p)['cost'],p))
        blocks=[]
        hard=sum(d<case['state'].context.hard_end.isoformat() for d in dates)
        pair_starts=sorted(set(x for x in (0,1,2,3,4,6,hard-1,hard,len(dates)-3,len(dates)-2) if x+1<len(dates)))
        groups={r:[k for k in keys if catalog[k].role==r] for r in ('easy','long','quality')}
        representative=tuple(dict.fromkeys((
            min(groups['easy'],key=lambda k:abs(catalog[k].miles-case['state'].context.policy.ordinary_miles/2)),
            min(groups['easy'],key=lambda k:abs(catalog[k].miles-case['state'].context.policy.ordinary_miles)),
            max(groups['long'],key=lambda k:catalog[k].miles),
            max(groups['quality'],key=lambda k:catalog[k].miles))))
        panels=[(dates[i:i+2],keys) for i in pair_starts]+[
            (dates[i:i+3],representative) for i in range(0,len(dates)-2,2)]
        for block,menu in panels:
            phase='block:'+','.join(block)
            start=best
            count=0
            before=len(cache)
            for p in block_candidates(start,block,menu):
                v=price(p)
                count+=1
                if v['eligible'] and v['cost']<price(best)['cost']:
                    best=p
            blocks.append(dict(days=block,menu=menu,assignments=count,evaluations=len(cache)-before,
                before_cost=price(start)['cost'],after_cost=price(best)['cost'],
                complete=True,program=best))
        winners={**controls,'best_found':best}
        for i,lane in enumerate(lanes):
            winners[f'lane_{i}_{lane["family"]}']=lane['program']
        comparisons=backward(early,case,winners)
        report=dict(manifest=manifest,seed=seed,events=events,per_family=per_family,
            lane_budget=lane_budget,budget=budget,evaluations=len(cache),seconds=perf_counter()-started,
            families=summaries,lanes=lanes,blocks=blocks,backward=comparisons,
            archived_cost=case['archived_cost'],best_cost=price(best)['cost'],best_program=best,
            best_shape=shape(best,catalog,dates),churn=churn(early['event']['program'],best,
                case['state'].context.decision,catalog),
            forecast_nominal_changed=early['event']['forecast_load_by_dose']!=case['event']['forecast_load_by_dose'],
            scope='Frozen stratified sampling, equal-budget descent, conditional exhaustive blocks. '
                'No global optimum, physiological validation or daily-stability claim; backward injection is retrospective.')
        with (output/'result.json').open('x') as stream:
            json.dump(report,stream,indent=2,allow_nan=False)
        print(json.dumps({k:report[k] for k in ('evaluations','seconds','archived_cost','best_cost','best_shape','churn')}),flush=True)
        return report
    finally:
        journal.close()


if __name__=='__main__':
    p=ArgumentParser()
    p.add_argument('root',type=Path)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--events',type=int,nargs=2,default=(3,4))
    p.add_argument('--per-family',type=int,default=256)
    p.add_argument('--lane-budget',type=int,default=2000)
    p.add_argument('--budget',type=int,default=30000)
    p.add_argument('--seed',type=int,default=20260928)
    a=p.parse_args()
    run(a.root,a.output_dir,events=tuple(a.events),per_family=a.per_family,
        lane_budget=a.lane_budget,budget=a.budget,seed=a.seed)
