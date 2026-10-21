"""Fixed-calendar cadence investigation, NOT a production scheduler/repair.

Both physical cadence families use the SAME eight-day policy and forecasts.
Omit only terminal purpose while searching, then verify a known-phase nominal
continuation has the same price. Match total mileage, not arbitrary hard/tail
sections; the original prospective-volume gate remains in force. Calendar
periodicity is a diagnostic control only. No database or model defaults change.
"""
from argparse import ArgumentParser
from dataclasses import replace
from datetime import date, datetime, time
from itertools import combinations
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from scripts.audit_matched_key_cadence import matched_program, conditioned_tail_control
from scripts.audit_progression_rollover_instability import seed_library, score
from scripts.audit_key_cadence import summarize
from scripts.check_rollover_search_repair import cadence_case
from scripts.replay_adaptive_search import load_decision
from scripts.research_manifest import freeze_sources
from run_analysis.config import load_config
from run_analysis.adaptive.continuous_search_adapter import canonical, fingerprint
from run_analysis.adaptive.decision_snapshot import evaluate_snapshot
from run_analysis.adaptive.finalized_distance_search import refine_distances
from run_analysis.adaptive.new_model_daily_harness import planned_accounted
from run_analysis.adaptive.program_objective_trial import PURPOSES
from run_analysis.continuous_training_dose import ContinuousDoseReference


def adjacent_pair_moves(program, catalog):
    """All +/-0.1 easy transfers across ALL dates, with exact total preservation.

    This is deliberately a smaller diagnostic neighborhood than the existing
    full pair-line search. Complete scans certify only this neighborhood, not
    arbitrary-distance transfers or calendar optimality. No convexity assumed.
    """
    units = {round(v.miles*10): k for k,v in sorted(catalog.items(), reverse=True) if v.role=='easy'}
    easy = [(d,k) for d,k in program if catalog[k].role=='easy']
    for (a,ka),(b,kb) in combinations(easy,2):
        ua,ub = round(catalog[ka].miles*10),round(catalog[kb].miles*10)
        for delta in (-1,1):
            if ua+delta in units and ub-delta in units:
                yield tuple((d,units[ua+delta] if d==a else units[ub-delta] if d==b else k) for d,k in program)


def run(root, comparison, matched, output, *, index=10, dose_budget=2000):
    if dose_budget<1: raise ValueError('Positive independent per-family dose budget required')
    start=perf_counter(); original=load_decision(root,index)
    case=cadence_case(original,8); state=case['state']; catalog=case['catalog']
    neutral_state=replace(state,context=replace(state.context,purpose_continuation=False))
    prior=json.loads((comparison/'cadence-8'/'result.json').read_text())
    baseline=canonical(prior['program'])
    assert state.fingerprint==prior['snapshot_fingerprint']
    assert abs(score(case,baseline)['cost']-prior['cost'])<1e-12
    previous_eight=canonical(json.loads((matched/'report.json').read_text())['program'])
    total=sum(round(catalog[k].miles*10) for _,k in baseline)
    checkpoint=state.prospective_volume_review.end.astimezone(ZoneInfo(state.timezone)).date().isoformat()
    first=sum(round(catalog[k].miles*10) for d,k in baseline if d<checkpoint)
    limits=(checkpoint,state.context.end.isoformat()); targets=(first,total-first)
    output.mkdir(exist_ok=False)
    manifest=freeze_sources(output/'sources.zip',config=load_config(),entrypoint=__name__,inputs=(
        root/f'decision-{index:04d}.json',root/original['event']['snapshot_file'],root/'scenario.json',
        comparison/'cadence-8'/'result.json',matched/'report.json'))
    cache={}
    def evaluate(program):
        program=canonical(program)
        if program not in cache: cache[program]=original['evaluate'](program,state_override=neutral_state)
        return cache[program]
    families={}; winners={}; output_rows={}
    for cadence,prior_program in ((7,baseline),(8,previous_eight)):
        library=seed_library(cadence_case(original,cadence),100000)
        controls={v for p in library for reverse in (False,True)
            if (v:=matched_program(p,catalog,limits,targets,reverse=reverse)) is not None}
        controls.add(prior_program)
        assert all(sum(round(catalog[k].miles*10) for _,k in p)==total for p in controls)
        ranked=sorted((evaluate(p)['cost'],p) for p in controls if evaluate(p)['eligible'])
        if not ranked: raise ValueError('No admitted control for cadence '+str(cadence))
        families[str(cadence)]=dict(library=len(library),matched_controls=len(controls),
            admitted=len(ranked),screen_cost=ranked[0][0],screen_program=ranked[0][1])
        winners[cadence]=ranked[0][1]
        print(json.dumps(dict(milestone='coarse_family_complete',cadence=cadence,
            **{k:families[str(cadence)][k] for k in ('library','matched_controls','admitted','screen_cost')},
            seconds=perf_counter()-start)),flush=True)
    screen_evaluations=len(cache)
    for cadence in (7,8):
        def item(program):
            value=evaluate(program)
            return dict(days=program,physical=program,tail_fingerprint='',stable=False,
                score=SimpleNamespace(total_cost=value['cost'],offsets=()),
                row=dict(hard_fingerprint=fingerprint(program)))
        candidate=item(winners[cadence]); fronts=[candidate]
        def price(seed):
            if not evaluate(seed['days'])['eligible']: return None
            revised=item(seed['days']); fronts.append(revised); return revised
        trace=refine_distances([candidate],price,{},None,budget=dose_budget,
            visible_days=len(case['dates']),seed_candidates=[candidate],
            neighbor_adapter=lambda current:adjacent_pair_moves(current['days'],catalog),
            identity_adapter=fingerprint)
        winner=min(fronts,key=lambda r:(r['score'].total_cost,r['physical']))['physical']
        assert sum(round(catalog[k].miles*10) for _,k in winner)==total
        details=summarize(dict(case,state=neutral_state),winner)
        assert details['score']['eligible'] and details['score']['funded']
        reference=ContinuousDoseReference('sandbox upload accounting')
        physical=tuple(planned_accounted(f'candidate:{d}:{k}',datetime.combine(
            date.fromisoformat(d),time(18),ZoneInfo(state.timezone)),catalog[k],case['handoffs'][k],
            reference,available=state.as_of) for d,k in winner)
        checked=evaluate_snapshot(state,physical,tuple(k for _,k in winner),reference=reference)
        evaluation=checked['evaluation']; boundary=evaluation.parts['continuation_boundary']
        conditioned=conditioned_tail_control(state.context,evaluation.conditional_ledger,
            tuple(r['handoff'] for r in boundary['scenarios']))
        tail=sum(row['loss']*state.context.policy.purpose_weights[PURPOSES.index(role)] for role,row in conditioned.items())
        conditioned_cost=evaluation.total_cost-boundary['purpose_tail_cost']+tail
        output_rows[str(cadence)]=dict(program=winner,details=details,refinement=trace,
            full_original_score=score(case,winner),conditioned_terminal_cost=tail,
            conditioned_score=conditioned_cost,conditioned_handoffs=conditioned,
            total_miles=total/10)
        print(json.dumps(dict(milestone='dose_family_complete',cadence=cadence,
            cost=details['score']['cost'],conditioned_tail=tail,
            evaluations=trace['evaluations'],status=[r['status'] for r in trace['lanes']],
            seconds=perf_counter()-start)),flush=True)
    report=dict(manifest=manifest,source=str(root),decision=index,as_of=original['event']['as_of'],
        policy_cadence_days=8,physical_families=(7,8),match=dict(total_miles=total/10,
            checkpoint_initial_miles=first/10,hard_tail_matched=False,
            checkpoint_matching_initialization_only=True),families=families,results=output_rows,
        screen_evaluations=screen_evaluations,evaluations=len(cache),seconds=perf_counter()-start,
        dose_budget_per_family=dose_budget,default_policy_modified=False,live_server_modified=False,
        limitations=['Fixed periodic key/easy-date families are diagnostic controls, not forced production patterns',
            'Terminal-purpose ablation isolates the demonstrated unknown-phase bias; it is not a proposed model repair',
            'Known-phase nominal control repriced after search; nonzero fallback is disclosed, not silently ignored',
            'Full coarse libraries searched but only best balanced-dose lane in each family refined; no global optimality',
            'Dose convergence applies only to +/-0.1 equal-total easy pairs, not full pair lines or calendar edits',
            'Equal full physical volume does not imply equal hard-volume, tapered score exposure or training effectiveness',
            'Same synthetic history/uncertain HR forecasts; not live advice, measured fatigue, daily stability or medical clearance'])
    with (output/'report.json').open('x') as stream: json.dump(report,stream,indent=2,default=str,allow_nan=False)
    print(json.dumps(dict(status='complete',seconds=report['seconds'],evaluations=len(cache),
        cost_gap_eight_minus_seven=output_rows['8']['conditioned_score']-output_rows['7']['conditioned_score'])),flush=True)
    return report


def finish_existing(root, output, *, index=10):
    """A handful of fixed reprices and attribution; NEVER rerun the screen."""
    saved=json.loads((output/'report.json').read_text())
    case=cadence_case(load_decision(root,index),8);context=case['state'].context
    neutral=replace(context,purpose_continuation=False);catalog=case['catalog']
    rows=saved['results'];a=rows['7']['details']['score'];b=rows['8']['details']['score']
    for name,row in rows.items():
        checked=score(case,canonical(row['program']),neutral)
        assert checked['eligible'] and checked['funded']
        assert abs(checked['cost']-row['conditioned_score'])<1e-12
        assert row['conditioned_terminal_cost']==0.
    deltas={k:b[k]-a[k] for k in ('cost','weighted_demand','delivery','short','linear_volume')}
    assert abs(deltas['cost']-sum(deltas[k] for k in ('weighted_demand','delivery','short')))<1e-12
    # Isolate the original 21-day matching restriction on the NEW calendar.
    # Balanced initialization only: not another search or an optimized control.
    baseline=rows['7']['program'];cut=case['state'].prospective_volume_review.end.astimezone(
        ZoneInfo(case['state'].timezone)).date().isoformat()
    limits=(cut,context.hard_end.isoformat(),context.end.isoformat());targets=[];lower=''
    for upper in limits:
        targets.append(sum(round(catalog[k].miles*10) for d,k in baseline if lower<=d<upper));lower=upper
    rematched=[]
    for reverse in (False,True):
        p=matched_program(rows['8']['program'],catalog,limits,targets,reverse=reverse)
        if p is None: raise ValueError('Unexpected capacity failure in fixed-date matching control')
        rematched.append(dict(reverse=reverse,program=p,score=score(case,p,neutral)))
    inventory={name:{role:dict(runs=sum(catalog[k].role==role for _,k in row['program']),
        miles=sum(catalog[k].miles for _,k in row['program'] if catalog[k].role==role))
        for role in ('easy','long','quality')} for name,row in rows.items()}
    result=dict(manifest=freeze_sources(output/'attribution.sources.zip',config=load_config(),
        entrypoint=__name__,inputs=(output/'report.json',root/f'decision-{index:04d}.json',
            root/case['event']['snapshot_file'],root/'scenario.json')),
        deltas_eight_minus_seven=deltas,
        corrected_cost_difference_percent=100*(b['cost']/a['cost']-1),
        demand_difference_percent=100*(b['weighted_demand']/a['weighted_demand']-1),
        full_inventory=inventory,same_eight_day_dates_old_three_section_totals=rematched,
        both_known_phase_tail_costs_zero=True,
        caveats=['The seven-day comparator has seven-day hard-prefix key spacing but an unrestricted irregular tail',
            'The new eight-day family meets eight-day purpose delivery, not identical long/quality session counts',
            'Lower full objective is largely volume accounting; key-entry axes must be inspected independently',
            'Restoring exact hard/tail matching here uses balanced doses only, not a complete optimized attribution'])
    with (output/'attribution.json').open('x') as stream:json.dump(result,stream,indent=2,default=str,allow_nan=False)
    print(json.dumps({k:result[k] for k in ('deltas_eight_minus_seven','corrected_cost_difference_percent','demand_difference_percent')}),flush=True)
    return result


def verify_corrected(root, output, *, index=10):
    """Verify integrated v20 against prior controls; fixed reprices ONLY."""
    from run_analysis.adaptive.shared_program_evaluator import PHASE_CONTINUATION_MODEL_VERSION
    saved=json.loads((output/'report.json').read_text());original=load_decision(root,index)
    reference=ContinuousDoseReference('sandbox upload accounting');matrix={}
    for cadence in (7,8):
        case=cadence_case(original,cadence);state=case['state']
        revised=replace(state,context=replace(state.context,purpose_continuation_phase='known_full_phase_v1'))
        rows={}
        for name,prior in saved['results'].items():
            program=canonical(prior['program'])
            physical=tuple(planned_accounted(f'candidate:{d}:{k}',datetime.combine(
                date.fromisoformat(d),time(18),ZoneInfo(state.timezone)),case['catalog'][k],case['handoffs'][k],
                reference,available=state.as_of) for d,k in program)
            old=evaluate_snapshot(state,physical,tuple(k for _,k in program),reference=reference)
            new=evaluate_snapshot(revised,physical,tuple(k for _,k in program),reference=reference)
            a,b=old['evaluation'],new['evaluation']
            assert old['decision_eligible']==new['decision_eligible']==True
            assert a.weighted_demand_cost==b.weighted_demand_cost and a.short_run_cost==b.short_run_cost
            assert a.conditional_ledger==b.conditional_ledger
            assert a.parts['delivery']==b.parts['delivery'] and a.parts['targets']==b.parts['targets']
            assert b.model_version==PHASE_CONTINUATION_MODEL_VERSION
            assert a.fingerprint!=b.fingerprint
            tail=b.parts['continuation_boundary']['purpose_tail_cost']
            if cadence==8:
                assert tail==0. and abs(b.total_cost-prior['conditioned_score'])<1e-12
            rows[name]=dict(old_cost=a.total_cost,new_cost=b.total_cost,new_terminal_purpose_cost=tail,
                model_version=b.model_version,eligible=new['decision_eligible'],
                purpose_continuation=b.parts['continuation_boundary']['purpose_continuation'],
                physical_costs_and_delivery_unchanged=True)
        matrix[str(cadence)]=rows
    result=dict(matrix=matrix,manifest=freeze_sources(output/'v20-verification.sources.zip',config=load_config(),
        entrypoint=__name__,inputs=(output/'report.json',root/f'decision-{index:04d}.json',
            root/original['event']['snapshot_file'],root/'scenario.json')),
        scope='Integrated scoring verification only; no new search, daily replay or live deployment')
    with (output/'v20-verification.json').open('x') as stream:json.dump(result,stream,indent=2,default=str,allow_nan=False)
    print(json.dumps(dict(status='integrated_v20_verified',costs={c:{k:v['new_cost'] for k,v in rows.items()}
        for c,rows in matrix.items()})),flush=True)
    return result


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('root',type=Path);p.add_argument('comparison',type=Path)
    p.add_argument('matched',type=Path);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--event',type=int,default=10);p.add_argument('--dose-budget',type=int,default=2000)
    p.add_argument('--finish-existing',action='store_true')
    p.add_argument('--verify-corrected',action='store_true')
    a=p.parse_args()
    if a.verify_corrected:verify_corrected(a.root,a.output_dir,index=a.event)
    elif a.finish_existing:finish_existing(a.root,a.output_dir,index=a.event)
    else:run(a.root,a.comparison,a.matched,a.output_dir,index=a.event,dose_budget=a.dose_budget)
