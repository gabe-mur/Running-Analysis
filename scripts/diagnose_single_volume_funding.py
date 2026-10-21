"""Frozen funding contrasts; no policy changes, uploads, or live publication."""
from argparse import ArgumentParser
from datetime import date, datetime, time, timedelta
import json
from math import ceil
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

from scripts.replay_adaptive_search import load_decision
from scripts.research_manifest import freeze_sources
from run_analysis.config import load_config
from run_analysis.adaptive.decision_snapshot import evaluate_snapshot
from run_analysis.adaptive.new_model_daily_harness import planned_accounted
from run_analysis.continuous_training_dose import ContinuousDoseReference


def diagnose(root, index):
    case = load_decision(root, index)
    state, catalog, handoffs = (case[k] for k in ('state', 'catalog', 'handoffs'))
    ctx = state.context
    base = tuple(map(tuple, case['event']['program']))
    reference = ContinuousDoseReference('sandbox upload accounting')
    physical, cache = {}, {}
    stamp = perf_counter()

    def evaluate(program):
        program = tuple(sorted(program))
        if program in cache:
            return cache[program]
        for day, key in program:
            if (day, key) not in physical:
                physical[day, key] = planned_accounted(f'candidate:{day}:{key}',
                    datetime.combine(date.fromisoformat(day), time(18), ZoneInfo(state.timezone)),
                    catalog[key], handoffs[key], reference, available=state.as_of)
        r = evaluate_snapshot(state, tuple(physical[p] for p in program),
            tuple(k for _, k in program), reference=reference)
        e = r['evaluation']
        hard_miles = sum(catalog[k].miles for d, k in program if d < ctx.hard_end.isoformat())
        out = dict(program=program, hard_miles=hard_miles, eligible=r['decision_eligible'],
            cost=e.total_cost, fatigue=e.weighted_demand_cost, delivery=e.delivery_cost,
            short=e.short_run_cost, losses=e.parts['shortfall_by_purpose'],
            rolling=e.parts['rolling_volume']['daily'],
            constraints=r.get('constraints'), suitability=r.get('suitability'))
        assert abs(out['cost']-out['fatigue']-out['delivery']-out['short']) < 1e-12
        cache[program] = out
        return out

    baseline = evaluate(base)
    assert abs(baseline['cost']-case['event']['cost']) < 1e-12
    target = sum(max(0, (min(ctx.hard_end, t.end)-max(ctx.decision, t.start)).days)
        * t.miles_per_7_days/7 for t in ctx.volume_targets)

    def contrast(r):
        delta = {k:r[k]-baseline[k] for k in ('cost', 'fatigue', 'delivery', 'short', 'hard_miles')}
        # Exact crossover for THIS fixed pair, not a calibrated recommendation.
        raw_demand_delta = delta['fatigue']/ctx.exchange
        crossover = -(delta['delivery']+delta['short'])/raw_demand_delta if raw_demand_delta else None
        return dict(**{k:v for k,v in r.items() if k!='rolling'}, delta=delta,
            pairwise_exchange_crossover=crossover)

    variants = []
    for day, key in base:
        if day >= ctx.hard_end.isoformat() or catalog[key].role != 'easy':
            continue
        for change in (-.1, .1, .5, 1.):
            newkey = f'easy:{catalog[key].miles+change:.1f}'
            if newkey in handoffs:
                variants.append(evaluate(tuple((d,newkey if d==day else k) for d,k in base)))
    # Bounded new-day alternatives: every free hard-window date and every
    # supported one-decimal easy dose. Nothing is invented beyond the catalog.
    used = {d for d,_ in base}
    for offset in range((ctx.hard_end-ctx.decision).days):
        day = (ctx.decision+timedelta(days=offset)).isoformat()
        if day in used:
            continue
        for key in handoffs:
            if catalog[key].role == 'easy':
                variants.append(evaluate((*base,(day,key))))

    # Dose-only constrained path: best eligible +0.1-mile increment at each step,
    # even when it raises cost. Dates, key sessions and tail all stay fixed.
    # A greedy witness, not a claim of optimal allocation/global optimality.
    current = base
    path = [baseline]
    for _ in range(max(0,ceil((target-baseline['hard_miles'])*10-1e-9))):
        choices = []
        for day,key in current:
            if day >= ctx.hard_end.isoformat() or catalog[key].role != 'easy':
                continue
            newkey = f'easy:{catalog[key].miles+.1:.1f}'
            if newkey in handoffs:
                trial = evaluate(tuple((d,newkey if d==day else k) for d,k in current))
                if trial['eligible']:
                    choices.append(trial)
        if not choices:
            break
        best = min(choices,key=lambda r:(r['cost'],r['program']))
        path.append(best)
        current = best['program']

    eligible = [r for r in variants if r['eligible']]
    added = [r for r in eligible if r['hard_miles']>baseline['hard_miles']+1e-8]
    funded = [r for r in eligible if r['hard_miles']>=target]
    blocks = []
    for left,right in ((0,7),(7,21),(21,40),(40,54)):
        rows = [r['volume'] for r in baseline['rolling']
            if left <= (r['day']-ctx.decision).days < right]
        blocks.append(dict(offsets=[left,right],mean_smoothed_28day_fraction=
            sum(r['delivered']/r['target'] for r in rows)/len(rows) if rows else None))
    return dict(index=index,reason=case['event']['reason'],snapshot_fingerprint=case['event']['snapshot_fingerprint'],
        exchange=ctx.exchange,target_integral=target,baseline=contrast(baseline),
        higher_volume_best=contrast(min(added,key=lambda r:r['cost'])) if added else None,
        funded_insertion_best=contrast(min(funded,key=lambda r:r['cost'])) if funded else None,
        dose_path=[contrast(r) for r in path],
        better_candidates=[contrast(r) for r in eligible if r['cost']<baseline['cost']-1e-12],
        variants=[contrast(r) for r in variants],baseline_rolling=baseline['rolling'],blocks=blocks,
        candidate_count=len(variants),eligible_count=len(eligible),evaluations=len(cache),seconds=perf_counter()-stamp,
        limitations=['Fixed forecast, target and history; tail held fixed',
            'No full calendar reoptimization or long-term realized funding claim',
            'Hard-window integral is diagnostic, not the implemented rolling objective',
            'Exchange crossover is pairwise sensitivity, not selected new policy'])


def main():
    p=ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('output/single-volume-daily-2-20260929'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--events',type=int,nargs='+',default=[1,3])
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    inputs=[a.root/'scenario.json']
    for i in a.events:
        path=a.root/f'decision-{i:04d}.json'
        inputs.extend((path,a.root/json.loads(path.read_text())['snapshot_file']))
    manifest=freeze_sources(a.output/'sources.zip',config=load_config(),entrypoint=__name__,inputs=tuple(inputs))
    results=[]
    for i in a.events:
        result=diagnose(a.root,i);results.append(result)
        with (a.output/f'event-{i:04d}.json').open('x') as f:
            json.dump(result,f,indent=2,default=str)
        print(json.dumps(dict(index=i,evaluations=result['evaluations'],seconds=result['seconds'],
            baseline=result['baseline']['cost'],best_added=result['higher_volume_best']['delta'],
            funded=result['dose_path'][-1]['delta'],better=len(result['better_candidates'])),default=str),flush=True)
    with (a.output/'manifest.json').open('x') as f:
        json.dump(manifest,f,indent=2)


if __name__=='__main__':
    main()
