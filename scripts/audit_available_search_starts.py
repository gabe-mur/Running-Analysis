"""Causal archived search-start comparison; no later plans or live publication.

Only finalists actually recorded in this decision and its predecessor are
available. Missing discarded candidates are not reconstructed as observed.
Native-calendar and winner-calendar dose-only comparisons isolate coverage.
"""
from argparse import ArgumentParser
from datetime import date, datetime
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

from run_analysis.adaptive.compiled_calendar_doses import compile_calendar, refine
from run_analysis.adaptive.race_accounting import allowed_dose
from scripts.replay_adaptive_search import load_decision
from scripts.research_manifest import freeze_sources


def available_programs(event, catalog, dates, *, prior=None):
    """Recover named recorded finalists, not all candidates a search evaluated."""
    domain = set(dates)
    found = {}
    def add(program, source):
        p = tuple(sorted(tuple(row) for row in program))
        if len(dict(p)) != len(p) or any(d not in domain or k not in catalog for d,k in p):
            raise ValueError('Archived search start is outside its physical domain')
        found.setdefault(p, []).append(source)
    add(event['program'], 'archived_winner')
    def walk(value, path):
        if (isinstance(value, list) and value and all(isinstance(p, list) and len(p)==2
                and isinstance(p[0], str) and len(p[0])==10 and p[1] in catalog for p in value)):
            for d,k in value:
                date.fromisoformat(d)
            add(value, path)
        elif isinstance(value, dict):
            for k,v in value.items(): walk(v, path+'.'+k)
        elif isinstance(value, list):
            for n,v in enumerate(value): walk(v, f'{path}[{n}]')
    walk(event.get('refinement', {}), 'refinement')
    if prior is not None:
        if (prior['index'] >= event['index'] or
                datetime.fromisoformat(prior['as_of']) > datetime.fromisoformat(event['as_of'])):
            raise ValueError('A search start must not come from a later decision')
        p = tuple((d,k) for d,k in prior['program'] if d in domain and k in catalog)
        if p: add(p, 'actual_prior_plan_current_domain')
    return tuple(dict(program=p, sources=s) for p,s in found.items())


def carry_easy_doses(winner, alternative, catalog):
    """Borrow overlapping easy doses only; preserve winner dates and all roles."""
    other = dict(alternative)
    return tuple((d,other[d] if d in other and catalog[k].role=='easy'
        and catalog[other[d]].role=='easy' else k) for d,k in winner)


def shortlist_captured(captured, winner, catalog, *, per_group=8):
    """Best dose allocations on winner's calendar plus best distinct calendars."""
    def calendar(p): return tuple((d,catalog[k].role) for d,k in p)
    target = calendar(winner)
    ranked = sorted((r for r in captured if r['eligible']),key=lambda r:(r['cost'],r['program']))
    same = [r for r in ranked if calendar(r['program'])==target][:per_group]
    others = []; seen = {target}
    for row in ranked:
        identity = calendar(row['program'])
        if identity not in seen:
            others.append(row); seen.add(identity)
        if len(others)>=per_group: break
    return tuple(dict(program=r['program'],sources=[f"captured_search_evaluation:{r['evaluation']}"])
        for r in (*same,*others))


def run(root, *, index, output, full_check_cap=64, capture_search=False):
    case = load_decision(root,index)  # Mandatory archived cost/admission reproduction.
    event,state,catalog = (case[k] for k in ('event','state','catalog'))
    output.mkdir(parents=True,exist_ok=False)
    prior_path = root/f'decision-{index-1:04d}.json'
    prior = json.loads(prior_path.read_text()) if index>1 else None
    manifest = freeze_sources(output/'sources.zip',config=dict(root=str(root),index=index,
        full_check_cap=full_check_cap),entrypoint=__name__,inputs=(root/f'decision-{index:04d}.json',
        root/event['snapshot_file'],root/'scenario.json',*((prior_path,) if prior else ())))
    starts = available_programs(event,catalog,case['dates'],prior=prior)
    winner = tuple(map(tuple,event['program']))
    capture = None
    if capture_search:
        if prior is None:
            raise ValueError('Full capture requires the actual archived predecessor, not a fabricated opening')
        from scripts.benchmark_complete_first_search import run as benchmark
        scenario = json.loads((root/'scenario.json').read_text())
        policy = scenario['search_policy']
        captured = []
        def observe(program,checked):
            captured.append(dict(program=tuple(program),evaluation=len(captured)+1,**checked))
        search_result = benchmark(root,output/'captured-search',index=index,prior_root=root,
            prior_index=index-1,complete_first=policy.get('complete_first_search',False),
            evaluations=policy['evaluations'],final_winner_evaluations=policy.get('final_winner_evaluations',0),
            evaluation_observer=observe)
        if (tuple(search_result['program'])!=winner or abs(search_result['cost']-event['cost'])>1e-12
                or len(captured)!=search_result['evaluations']):
            raise ValueError('Captured search must reproduce the archived winner and evaluation count')
        selected = shortlist_captured(captured,winner,catalog)
        starts = (*starts,*selected)
        pool_path = output/'captured-search'/'candidate-pool.json'
        pool_path.write_text(json.dumps(captured,allow_nan=False))
        capture = dict(evaluations=len(captured),seconds=search_result['seconds'],
            admitted=sum(r['eligible'] for r in captured),shortlisted=len(selected),
            pool=str(pool_path),manifest=search_result['manifest'],winner_reproduced=True)
    raw_cache = {}; full_calls = 0
    stage = SimpleNamespace(**event['stage_runtime']['stage'])
    def details(program):
        nonlocal full_calls
        program = tuple(program)
        if program not in raw_cache:
            raw = case['evaluate_details'](program)
            raw_cache[program] = dict(raw,decision_eligible=raw['decision_eligible'] and all(
                allowed_dose(state.context.race_goal,date.fromisoformat(d),catalog[k],stage) for d,k in program))
            full_calls += 1
        return raw_cache[program]
    def evaluate(program):
        raw = details(program)
        return dict(cost=raw['evaluation'].total_cost,eligible=raw['decision_eligible'])
    modes = []
    stamp = perf_counter()
    for mode in ('native_calendar','winner_calendar_easy_doses'):
        seeds = {}
        for row in starts:
            p = row['program'] if mode=='native_calendar' else carry_easy_doses(winner,row['program'],catalog)
            seeds.setdefault(p,[]).extend(row['sources'])
        results = []
        for program,sources in seeds.items():
            started = perf_counter(); before_calls = full_calls
            raw = details(program)
            result = dict(seed_program=program,sources=sources,initial_cost=raw['evaluation'].total_cost,
                initial_eligible=raw['decision_eligible'])
            if raw['decision_eligible']:
                compiled = compile_calendar(state,program,physical=case['physical'],result=raw)
                polished = refine(compiled,evaluate,max_full_evaluations=full_check_cap)
                result.update(polished)
            result.update(actual_new_full_evaluations=full_calls-before_calls,
                total_seconds=perf_counter()-started)
            results.append(result)
        admitted = [r for r in results if r['initial_eligible']]
        best = min(admitted,key=lambda r:(r['cost'],r['program']))
        baseline = next(r for r in results if r['seed_program']==winner)
        modes.append(dict(mode=mode,starts=results,best_program=best['program'],best_cost=best['cost'],
            best_sources=best['sources'],winner_only_cost=baseline['cost'],
            gain_over_winner_only=baseline['cost']-best['cost'],
            all_admitted_neighborhoods_complete=all(r['neighborhood_complete'] for r in admitted)))
    result = dict(source_decision=str(root/f'decision-{index:04d}.json'),
        snapshot_fingerprint=state.fingerprint,as_of=state.as_of.isoformat(),manifest=manifest,
        archived_cost=case['archived_cost'],available_start_count=len(starts),modes=modes,captured_search=capture,
        seconds=perf_counter()-stamp,actual_full_evaluations=full_calls,
        full_check_cap_per_start=full_check_cap,live_changed=False,
        scope='Same-decision finalists, causal prior and optional causal captured shortlist; not exhaustive refinement or daily stability')
    (output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({k:result[k] for k in ('seconds','actual_full_evaluations','available_start_count')}
        | dict(modes=[{k:m[k] for k in ('mode','best_cost','winner_only_cost','gain_over_winner_only',
            'best_sources','all_admitted_neighborhoods_complete')} for m in modes]),indent=2),flush=True)
    return result


if __name__=='__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--index',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--full-check-cap',type=int,default=64)
    parser.add_argument('--capture-search',action='store_true',
        help='Replay unchanged search once, archive all evaluated candidates, refine a bounded causal shortlist')
    args = parser.parse_args()
    run(args.root,index=args.index,output=args.output,full_check_cap=args.full_check_cap,
        capture_search=args.capture_search)
