"""Bounded fixed-calendar continuation experiment; no search or publication.

Refine each archived state from its own existing refined winner only. Later
plans serve solely as hindsight score witnesses, never earlier search starts.
"""
from argparse import ArgumentParser
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

from scripts.replay_adaptive_search import load_decision
from scripts.research_manifest import freeze_sources
from run_analysis.adaptive.compiled_calendar_doses import compile_calendar, refine
from run_analysis.adaptive.continuation_survival_trial import distribution,evaluate_trial
from run_analysis.adaptive.soft_horizon_trial import day_weight
from run_analysis.adaptive.new_model_daily_harness import churn
from run_analysis.adaptive.decision_snapshot import serialize,restore


def run(root,*,artifact,output,full_check_cap=64,basis='c1_soft_handoff',integrated=False):
    total_started=perf_counter()
    if integrated and basis!='c1_soft_handoff':
        raise ValueError('Only the retained C1 experiment has a named persisted contract')
    saved=json.loads(artifact.read_text())
    if len(saved['cases'])!=2:
        raise ValueError('One frozen paired witness required')
    cases=[load_decision(root,r['index']) for r in saved['cases']]
    for case,row in zip(cases,saved['cases']):
        if case['state'].fingerprint!=row['snapshot']:
            raise ValueError('Witness snapshot mismatch')
    output.mkdir(parents=True,exist_ok=False)
    manifest=freeze_sources(output/'sources.zip',config=dict(root=str(root),artifact=str(artifact),
        full_check_cap=full_check_cap,basis=basis,integrated=integrated),entrypoint=__name__,inputs=(artifact,*[
            path for c,r in zip(cases,saved['cases']) for path in (root/f"decision-{r['index']:04d}.json",
                root/c['event']['snapshot_file'])],root/'scenario.json'))
    started=perf_counter();results=[];calls=0;parity=[]
    for case,row in zip(cases,saved['cases']):
        source_state=case['state']
        state=replace(source_state,context=replace(source_state.context,continuation_distribution='c1_v1')) if integrated else source_state
        spec=distribution(state.context,basis=basis)
        seed=tuple(map(tuple,row['program']))
        baseline=case['evaluate_details'](seed)
        if not baseline['decision_eligible']:
            raise ValueError('Current witness is no longer eligible')
        n=len(spec['probabilities'])
        uniform=evaluate_trial(source_state.context,baseline['evaluation'],probabilities=(1/n,)*n,
            weight=lambda d:day_weight(source_state.context,d))
        error=abs(uniform.total_cost-baseline['evaluation'].total_cost)
        if error>1e-12:
            raise ValueError('Uniform control did not reproduce archived objective')
        parity.append(error)
        if integrated:
            restored=restore(serialize(state),expected_fingerprint=state.fingerprint)
            if restored!=state:
                raise ValueError('New scoring contract did not round-trip')
            actual=case['evaluate_details'](seed,state_override=restored)['evaluation']
            diagnostic=evaluate_trial(source_state.context,baseline['evaluation'],
                probabilities=spec['probabilities'],weight=spec['weight'])
            if abs(actual.total_cost-diagnostic.total_cost)>1e-12:
                raise ValueError('Named contract differs from frozen trial')
        cache={}
        def details(p):
            nonlocal calls
            p=tuple(p)
            if p not in cache:
                raw=case['evaluate_details'](p,state_override=state);calls+=1
                cache[p]=raw if integrated else dict(raw,evaluation=evaluate_trial(state.context,raw['evaluation'],
                    probabilities=spec['probabilities'],weight=spec['weight']))
            return cache[p]
        def evaluate(p):
            raw=details(p)
            return dict(cost=raw['evaluation'].total_cost,eligible=raw['decision_eligible'])
        compiled=compile_calendar(state,seed,physical=case['physical'],result=details(seed),
            day_weight_function=spec['weight'],handoff_probabilities=spec['probabilities'])
        validation=[]
        for i in range(len(compiled.days)):
            levels=compiled.original_levels.copy()
            levels[i]=(levels[i]+1)%len(compiled.keys)
            p=compiled.physical(levels)
            err=abs(compiled.cost(levels)-evaluate(p)['cost']);validation.append(err)
            if err>1e-10:
                raise ValueError('Trial compiler mismatch outside the anchored winner')
        polished=refine(compiled,evaluate,max_full_evaluations=full_check_cap)
        winner=tuple(polished['program'])
        ev=details(winner)['evaluation']
        future=tuple(r for r in ev.conditional_ledger if r.source=='planned')
        results.append(dict(index=row['index'],snapshot=state.fingerprint,source_snapshot=source_state.fingerprint,
            model_version=ev.model_version,as_of=state.as_of.isoformat(),
            original_program=seed,baseline_cost=baseline['evaluation'].total_cost,
            maximum_score_error=max(validation),memory_days=spec['memory_days'],
            endpoint_probability=spec['endpoint_probability'],
            hard21_miles=sum(r.miles for r in future if r.day<state.context.hard_end),
            hard21_runs=sum(r.day<state.context.hard_end for r in future),
            **polished))
    a,b=cases
    original_p,original_q=[tuple(map(tuple,r['program'])) for r in saved['cases']]
    ranking={}
    if all(d in a['dates'] and d in b['dates'] for d,_ in (*original_p,*original_q)):
        # Freeze forecast and mean-continuation references as well as the two
        # explicit allocations. This is attribution, not a causal start.
        later=replace(b['state'],context=replace(b['state'].context,
            continuation_boundary_references=a['state'].context.continuation_boundary_references))
        for name,state in (('before',a['state']),('after_frozen_forecasts',later)):
            if integrated:
                state=replace(state,context=replace(state.context,continuation_distribution='c1_v1'))
            values=[]
            for p in (original_p,original_q):
                raw=a['evaluate_details'](p,state_override=state)
                dist=distribution(state.context,basis=basis)
                values.append(raw['evaluation'].total_cost if integrated else evaluate_trial(state.context,raw['evaluation'],
                    probabilities=dist['probabilities'],weight=dist['weight']).total_cost)
            ranking[name]=dict(before_allocation=values[0],after_allocation=values[1],delta=values[1]-values[0])
    old=tuple(results[0]['program']);new=tuple(results[1]['program'])
    edits=churn(old,new,b['state'].context.decision,catalog=b['catalog'])
    auxiliary_calls=len(cases)*(3 if integrated else 2)+2*len(ranking)
    result=dict(scope=('Offline named C1 score, snapshot round-trip and fixed-calendar refinement; not daily/live acceptance'
        if integrated else 'Offline survival fixed-calendar prototype; not daily acceptance or a publication identity'),
        cases=results,old_objective_uniform_parity_errors=parity,
        fixed_allocation_ranking=ranking,visible_comparison=edits,
        basis=basis,integrated=integrated,actual_trial_full_evaluations=calls,seconds=perf_counter()-started,
        auxiliary_full_evaluations=auxiliary_calls,total_full_evaluations=calls+auxiliary_calls,
        total_seconds=perf_counter()-total_started,
        timing_scope='seconds excludes source reconstruction/archiving; total_seconds includes them',
        manifest=manifest,live_changed=False,
        assumptions=[('Handoff hazard uses W+S-2 volume-memory days; accounting assumption, not physiological calibration'
            if basis=='memory_survival' else 'Cubic survival has zero slope at both existing taper endpoints; accounting assumption, not physiology'),
            'Remaining probability collapses at unchanged explicit end; no extra search dates',
            'No new HR model, volume utility, admission or continuity weight',
            'Only existing fixed calendars are refined; no global calendar or daily stability claim',
            'HR bounds are scenarios; driver recovery provenance unchanged'])
    with (output/'result.json').open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False,default=str)
    print(json.dumps(dict(seconds=result['seconds'],total_seconds=result['total_seconds'],
        full_evaluations=result['total_full_evaluations'],trial_full_evaluations=calls,
        ranking=ranking,visible_comparison=edits,cases=[{k:r[k] for k in
            ('index','hard21_miles','hard21_runs','neighborhood_complete','budget_exhausted','seconds')} for r in results]),indent=2))
    return result


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--artifact',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--full-check-cap',default=64,type=int)
    parser.add_argument('--basis',default='c1_soft_handoff',choices=('memory_survival','c1_soft_handoff'))
    parser.add_argument('--integrated',action='store_true')
    run(**vars(parser.parse_args()))
