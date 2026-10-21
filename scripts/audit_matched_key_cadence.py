"""Bounded fixed-key cadence control, not a new scheduler or live policy.

Enumerate the existing full stage-seed library, commit matched decimal mileage,
then reuse the existing lane/dose refiner. Full physical evaluation and admission
are unchanged. Exact key dates are diagnostic controls ONLY; search may miss
better calendars and this does not accept a repeating production pattern.
"""
from argparse import ArgumentParser
from collections import Counter
from dataclasses import replace
from datetime import datetime,time,timedelta
from math import ceil
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from scripts.replay_adaptive_search import load_decision
from scripts.check_rollover_search_repair import cadence_case
from scripts.audit_progression_rollover_instability import seed_library, score
from scripts.audit_key_cadence import summarize
from scripts.research_manifest import freeze_sources
from run_analysis.config import load_config
from run_analysis.adaptive.continuous_search_adapter import canonical, fingerprint
from run_analysis.adaptive.decimal_dose_search import decimal_pair_neighbors
from run_analysis.adaptive.finalized_distance_search import refine_distances
from run_analysis.adaptive.continuation_boundary_trial import purpose_handoff_losses
from run_analysis.adaptive.timely_purpose_trial import envelope_day
from run_analysis.adaptive.program_objective_trial import PURPOSES,credits
from run_analysis.adaptive.decision_snapshot import evaluate_snapshot
from run_analysis.adaptive.new_model_daily_harness import planned_accounted
from run_analysis.continuous_training_dose import ContinuousDoseReference


def balanced_units(total,count,available,*,reverse=False):
    """Exact tenths, near-equal initialization only; no preferred-size score."""
    if count==0:return () if total==0 else None
    low,high=min(available),max(available)
    if not count*low<=total<=count*high:return None
    quotient,remainder=divmod(total,count)
    result=tuple(quotient+int(i<remainder) for i in range(count))
    if any(v not in available for v in result):return None
    return tuple(reversed(result)) if reverse else result


def matched_program(program,catalog,limits,totals,*,reverse=False):
    """Preserve all dates/roles; fill easy commands to three physical budgets.

    First-review matching is an initialization control, not an added evaluator
    restriction. Later refinement may increase checkpoint credit while keeping
    hard21 and tail totals equal to the comparator.
    """
    by_unit={}
    for key,dose in sorted(catalog.items()):
        if dose.role=='easy':by_unit.setdefault(round(dose.miles*10),key)
    result=[];lower=''
    for upper,target in zip(limits,totals):
        group=tuple((d,k) for d,k in program if lower<=d<upper)
        fixed=tuple((d,k) for d,k in group if catalog[k].role!='easy')
        easy=tuple(d for d,k in group if catalog[k].role=='easy')
        remaining=target-sum(round(catalog[k].miles*10) for _,k in fixed)
        allocated=balanced_units(remaining,len(easy),by_unit,reverse=reverse)
        if allocated is None:return None
        result.extend((*fixed,*zip(easy,(by_unit[v] for v in allocated))))
        lower=upper
    if len(result)!=len(program):raise ValueError('Matching limits must cover every prescribed date')
    return canonical(result)


def same_volume_sections(a,b,miles,hard_end):
    def sums(p):
        return tuple(sum(round(miles[k]*10) for d,k in p if (d<hard_end)==hard) for hard in (True,False))
    return sums(a)==sums(b)


def conditioned_tail_control(context,ledger,handoffs):
    """Diagnostic only: preserve known full-session phase at a fluid handoff.

    This is a nominal perfect-continuation assumption, NOT a calibrated arrival
    forecast or implementation proposal. Missing, partial or already overdue
    phase retains the existing uniform-phase control, never fabricated credit.
    """
    uniform=purpose_handoff_losses(context,ledger,handoffs)
    span=(context.end-context.stage_start).days;result={}
    for role in context.timing.purposes:
        index=PURPOSES.index(role);cadence=7/getattr(context.policy,role+'_per_7')
        losses=[];rows=[]
        for i,handoff in enumerate(handoffs):
            events=sorted((r.day,float(credits(r,context.policy)[index])) for r in ledger
                if r.day<handoff and float(credits(r,context.policy)[index])>0)
            latest=events[-1] if events else None
            age=(handoff-latest[0]).days if latest else None
            if latest is None or latest[1]<1.-1e-12 or age>cadence:
                loss=uniform[role]['by_handoff'][i];delay=None;basis='uniform_unknown_partial_or_overdue_phase'
            else:
                delay=max(0,ceil(cadence-age));basis='known_full_reference_nominal_phase'
                loss=sum(envelope_day(events,handoff+timedelta(days=d),
                    grace=cadence*context.timing.grace_fraction,memory=cadence*context.timing.memory_scale,
                    initial=dict(context.opening_coverage).get(role,0.),initial_day=context.decision)['loss']
                    for d in range(delay))/span
            losses.append(loss);rows.append(dict(handoff=handoff.isoformat(),last_credit=latest,
                age_days=age,arrival_delay_days=delay,basis=basis,loss=loss))
        result[role]=dict(loss=sum(losses)/len(losses),handoffs=rows)
    return result


def attribute(root,comparison,matched,output,*,index=10):
    """Few fixed reprices only; never repeats the bounded cadence screen."""
    original=load_decision(root,index);case=cadence_case(original,8);state=case['state']
    reference=ContinuousDoseReference('sandbox upload accounting')
    saved=json.loads((matched/'report.json').read_text())
    selected=json.loads((comparison/'cadence-8'/'result.json').read_text())
    assert state.fingerprint==selected['snapshot_fingerprint']
    output.mkdir(exist_ok=False)
    manifest=freeze_sources(output/'sources.zip',config=load_config(),entrypoint=__name__,
        inputs=(matched/'report.json',comparison/'cadence-8'/'result.json',
            root/f'decision-{index:04d}.json',root/original['event']['snapshot_file'],root/'scenario.json'))
    result={}
    for name,program,expected in (('existing',canonical(selected['program']),selected['cost']),
            ('matched',canonical(saved['program']),saved['matched_eight_day']['score']['cost'])):
        physical=tuple(planned_accounted(f'candidate:{d}:{k}',datetime.combine(
            datetime.fromisoformat(d).date(),time(18),ZoneInfo(state.timezone)),case['catalog'][k],
            case['handoffs'][k],reference,available=state.as_of) for d,k in program)
        checked=evaluate_snapshot(state,physical,tuple(k for _,k in program),reference=reference)
        evaluation=checked['evaluation'];parts=evaluation.parts;b=parts['continuation_boundary']
        assert checked['decision_eligible'] and abs(evaluation.total_cost-expected)<1e-12
        omitted=score(case,program,replace(state.context,purpose_continuation=False))
        assert omitted['weighted_demand']==evaluation.weighted_demand_cost
        assert abs(evaluation.total_cost-omitted['cost']-b['purpose_tail_cost'])<1e-12
        conditioned=conditioned_tail_control(state.context,evaluation.conditional_ledger,
            tuple(r['handoff'] for r in b['scenarios']))
        tail=sum(conditioned[role]['loss']*state.context.policy.purpose_weights[PURPOSES.index(role)] for role in conditioned)
        result[name]=dict(full=score(case,program),without_terminal_purpose=omitted,
            explicit_prefix_purpose_loss={role:data['loss'] for role,data in parts['timing'].items()},
            terminal_purpose_cost=b['purpose_tail_cost'],uniform_terminal=parts['continuation_boundary']['purpose_continuation'],
            conditioned_control=conditioned,conditioned_terminal_cost=tail,
            conditioned_full_cost=evaluation.total_cost-b['purpose_tail_cost']+tail,
            borrowed_future_credit=False)
    gap=result['matched']['full']['cost']-result['existing']['full']['cost']
    tail_gap=result['matched']['terminal_purpose_cost']-result['existing']['terminal_purpose_cost']
    report=dict(manifest=manifest,controls=result,full_gap=gap,terminal_purpose_gap=tail_gap,
        terminal_share_of_gap=tail_gap/gap,remaining_gap=gap-tail_gap,
        conditioned_gap=result['matched']['conditioned_full_cost']-result['existing']['conditioned_full_cost'],
        scope='Fixed-program attribution only; cadence, volume, physical demand and admission unchanged; no new search or default policy',
        limitations=['Conditioning on known full phase is a perfect nominal-continuation control, not a justified universal future forecast',
            'Removing the term is an ablation, not a proposed return to finite-horizon blindness',
            'Remaining dose/calendar search is uncertified; no daily stability, physiology or deployment acceptance'])
    with (output/'report.json').open('x') as stream:json.dump(report,stream,indent=2,default=str,allow_nan=False)
    print(json.dumps({k:report[k] for k in ('full_gap','terminal_purpose_gap','terminal_share_of_gap','remaining_gap','conditioned_gap')}),flush=True)
    return report


def run(root,comparison,output,*,index=10,budget=6000):
    if budget<4000:raise ValueError('Budget must cover the declared full coarse-control screen')
    original=load_decision(root,index)
    case=cadence_case(original,8);state=case['state'];context=state.context;catalog=case['catalog']
    selected=json.loads((comparison/'cadence-8'/'result.json').read_text())
    baseline=canonical(selected['program'])
    assert state.fingerprint==selected['snapshot_fingerprint']
    assert abs(score(case,baseline)['cost']-selected['cost'])<1e-12
    checkpoint=state.prospective_volume_review.end.astimezone(ZoneInfo(state.timezone)).date().isoformat()
    limits=(checkpoint,context.hard_end.isoformat(),context.end.isoformat())
    lower='';totals=[]
    for upper in limits:
        totals.append(sum(round(catalog[k].miles*10) for d,k in baseline if lower<=d<upper));lower=upper
    library=seed_library(case,100000)
    controls=[];seen=set();unmatchable=0
    for program in library:
        variants=[matched_program(program,catalog,limits,totals,reverse=r) for r in (False,True)]
        if all(v is None for v in variants):unmatchable+=1
        for v in variants:
            if v is not None and v not in seen:controls.append(v);seen.add(v)
    if len(controls)+100>budget:raise ValueError('Control screen would leave no meaningful refinement budget')
    output.mkdir(exist_ok=False)
    manifest=freeze_sources(output/'sources.zip',config=load_config(),entrypoint=__name__,
        inputs=(root/f'decision-{index:04d}.json',root/original['event']['snapshot_file'],root/'scenario.json',
            comparison/'cadence-7'/'result.json',comparison/'cadence-8'/'result.json'))
    start=perf_counter();cache={};reasons=Counter()
    def evaluate(program):
        program=canonical(program)
        if program not in cache:
            if len(cache)>=budget:return None
            cache[program]=original['evaluate'](program,state_override=state)
        return cache[program]
    ranked=[]
    for program in controls:
        checked=evaluate(program)
        if checked['eligible']:ranked.append((checked['cost'],program))
        else:reasons['rejected_by_unchanged_full_evaluator']+=1
    ranked.sort()
    if not ranked:raise ValueError('No matched eight-day controls admitted; inspect eligibility before proceeding')
    # Keep independent long/quality phases rather than three almost-identical doses.
    lanes=[];phases=set()
    for cost,program in ranked:
        phase=tuple((d,catalog[k].role) for d,k in program if catalog[k].role!='easy')
        if phase not in phases:
            phases.add(phase);lanes.append(program)
        if len(lanes)==3:break
    def item(program):
        checked=evaluate(program)
        return dict(days=program,physical=program,tail_fingerprint='',stable=False,
            score=SimpleNamespace(total_cost=checked['cost'],offsets=()),
            row=dict(hard_fingerprint=fingerprint(program)))
    candidates=[item(p) for p in lanes]
    miles={k:v.miles for k,v in catalog.items()};roles={k:v.role for k,v in catalog.items()}
    domain={d:i for i,d in enumerate(case['dates'])};fronts=[]
    def proposals(current):
        for proposed in decimal_pair_neighbors(current['days'],roles,miles,index=domain,radius=len(domain)):
            if same_volume_sections(current['days'],proposed,miles,limits[1]):yield proposed
    def price(seed):
        checked=evaluate(seed['days'])
        if checked is None or not checked['eligible']:return None
        revised=item(seed['days']);fronts.append(revised)
        return revised
    screen_evaluations=len(cache)
    trace=refine_distances(candidates,price,{},None,budget=budget-screen_evaluations,
        visible_days=len(domain),seed_candidates=candidates,max_sweeps=1,
        neighbor_adapter=proposals,identity_adapter=fingerprint)
    winner=min([*candidates,*fronts],key=lambda r:(r['score'].total_cost,r['physical']))['physical']
    assert same_volume_sections(baseline,winner,miles,limits[1])
    details=summarize(case,winner)
    baseline_details=summarize(case,baseline)
    # Matched-prefix initialization can be relaxed by ordinary dose refinement;
    # the unchanged prospective volume gate continues to govern acceptance.
    assert details['score']['eligible'] and details['score']['funded']
    report=dict(source=str(root),decision=index,as_of=original['event']['as_of'],manifest=manifest,
        hypothesis='Matched eight-day calendar/dose coordination may be missed by coarse seeds and short-range incumbent descent',
        match=dict(limits=limits,initial_section_miles=[v/10 for v in totals],
            hard_and_tail_exact=True,checkpoint_exact_initialization_only=True),
        screen=dict(full_library=len(library),unmatchable_calendar_shapes=unmatchable,
            unique_matched_controls=len(controls),admitted=len(ranked),rejected=dict(reasons),
            evaluations=screen_evaluations,best_cost=ranked[0][0],best_program=ranked[0][1]),
        refinement=trace,budget=budget,evaluations=len(cache),seconds=perf_counter()-start,
        prior_search=baseline_details,matched_eight_day=details,program=winner,
        improvement_over_prior=baseline_details['score']['cost']-details['score']['cost'],
        cross_policy={str(c):score(cadence_case(original,c),winner) for c in (7,8)},
        prior_same_eight_day_policy=score(case,baseline),
        default_policy_modified=False,live_server_modified=False,daily_replay_started=False,
        limitations=['Diagnostic eight-day key-date controls, not a mandatory production calendar',
            'Budgeted one-sweep dose comparison, not globally optimized eight-day calendars',
            'Equal hard and tail mileage do not imply identical rolling delivery, purpose credit or fatigue',
            'Actual old history/unknown readiness and conditional HR forecasts preserved; no medical recovery claim'])
    with (output/'report.json').open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False)
    print(json.dumps({k:report[k] for k in ('screen','evaluations','seconds','improvement_over_prior')},default=str),flush=True)
    return report


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('root',type=Path);p.add_argument('comparison',type=Path)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--event',type=int,default=10)
    p.add_argument('--budget',type=int,default=6000)
    p.add_argument('--attribute-matched',type=Path)
    a=p.parse_args()
    if a.attribute_matched:attribute(a.root,a.comparison,a.attribute_matched,a.output_dir,index=a.event)
    else:run(a.root,a.comparison,a.output_dir,index=a.event,budget=a.budget)
