"""Frozen coordinated-funding search; no upload, model-weight change or deployment."""
from argparse import ArgumentParser
from dataclasses import asdict,replace
from datetime import date,datetime,time,timedelta
import json
from pathlib import Path
import sqlite3
from time import perf_counter
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from scripts.replay_adaptive_search import load_decision,replayed_prior
from scripts.research_manifest import freeze_sources
from run_analysis.config import load_config
from run_analysis.continuous_training_dose import ContinuousDoseReference
from run_analysis.staged_progression import DevelopmentStage,ProgressionPolicy
from run_analysis.stage_target_history import StageTarget
from run_analysis.adaptive.prospective_progression import freeze_volume_review
from run_analysis.adaptive.adaptive_stage_lifecycle import review_entries
from run_analysis.adaptive.execution_upload_sandbox import ExecutionUploadSandbox
from run_analysis.adaptive.decision_snapshot import evaluate_snapshot,serialize,restore
from run_analysis.adaptive.new_model_daily_harness import planned_accounted,churn
from run_analysis.adaptive.continuous_search_adapter import search
from run_analysis.adaptive.decimal_dose_search import calendar_representatives
from run_analysis.adaptive.stage_calendar_seeds import complete_stage_seeds


def coordinated_case(root,index,*,review_hour):
    case=load_decision(root,index)  # Archive parity before changing admission.
    state,event,catalog=(case[k] for k in ('state','event','catalog'))
    scenario=json.loads((root/'scenario.json').read_text())
    runtime=event.get('stage_runtime',scenario)
    row=runtime['stage']
    stage=DevelopmentStage(**dict(row,issued_at=datetime.fromisoformat(row['issued_at']),
        policy=ProgressionPolicy(**row['policy'])))
    targets=tuple(StageTarget(**dict(t,available_at=datetime.fromisoformat(t['available_at']),
        effective_on=date.fromisoformat(t['effective_on']))) for t in runtime['target_history'])
    zone=ZoneInfo(state.timezone)
    checkpoint=datetime.combine(stage.review_at.astimezone(zone).date(),time(review_hour),zone)
    if checkpoint<stage.review_at:checkpoint+=timedelta(days=1)
    # Reuse durable causal completion/version/issued-band accounting through a
    # read-only handle. Do not run sandbox __init__/resume, which may recover files.
    journal=ExecutionUploadSandbox.__new__(ExecutionUploadSandbox)
    journal.db=sqlite3.connect(f'file:{(root/"sandbox.sqlite").resolve()}?mode=ro',uri=True)
    journal.db.row_factory=sqlite3.Row
    try:
        events=[json.loads(p.read_text()) for p in sorted(root.glob('decision-*.json'))
            if int(p.stem.split('-')[-1])<=index]
        harness=SimpleNamespace(sandbox=journal,events=events,catalog=catalog,zone=zone)
        entries=review_entries(harness,state.as_of)
    finally:journal.close()
    contract=freeze_volume_review(stage,targets,entries,state.as_of,
        review_at=checkpoint,timezone=state.timezone)
    if checkpoint>datetime.combine(state.context.hard_end,time.min,zone):
        raise ValueError('First-review trial must be inside the hard projection, not funded by speculative tail')
    return dict(case,state=replace(state,prospective_volume_review=contract),
        original_state=state,stage=stage,checkpoint=checkpoint,entries=entries)


def run(root,index,output,*,evaluations=10000,review_hour=4,prior_result=None,empty_rollover=False):
    case=coordinated_case(root,index,review_hour=review_hour)
    prior=None
    if prior_result is not None:
        prior=tuple(map(tuple,replayed_prior(prior_result,root/f'decision-{index:04d}.json')))
    if empty_rollover:
        if prior_result is None:raise ValueError('Expected empty-day rollover requires the preceding coordinated plan')
        saved=json.loads(prior_result.read_text())
        before=restore((prior_result.parent/'snapshot.json').read_text(),expected_fingerprint=saved['snapshot_fingerprint'])
        decision=before.context.decision+timedelta(days=1)
        if any(date.fromisoformat(d)<decision for d,k in prior):
            raise ValueError('Empty-day control cannot silently consume a prescribed run')
        cutoff=datetime.combine(decision,time(review_hour),ZoneInfo(before.timezone))
        # A real refresh re-resolves the persisted path over its newly exposed
        # dates. Extend only the already-flat bounded endpoint, not the ramp.
        def extended(segments):
            if not segments:raise ValueError('Expected rollover requires the explicit dated target path')
            return (*segments[:-1],replace(segments[-1],end=segments[-1].end+timedelta(days=1)))
        ctx=replace(before.context,decision=decision,hard_end=before.context.hard_end+timedelta(days=1),
            end=before.context.end+timedelta(days=1),volume_targets=extended(before.context.volume_targets),
            support_targets=extended(before.context.support_targets))
        state=replace(before,as_of=cutoff,context=ctx,
            prospective_volume_review=replace(before.prospective_volume_review,available_at=cutoff))
        dates=tuple((decision+timedelta(days=i)).isoformat() for i in range((ctx.end-decision).days))
        case=dict(case,state=state,dates=dates)
    state,event,catalog,dates,handoffs=(case[k] for k in ('state','event','catalog','dates','handoffs'))
    output.mkdir(exist_ok=False)
    sources=(root/f'decision-{index:04d}.json',root/event['snapshot_file'],root/'scenario.json',root/'sandbox.sqlite')
    if prior_result is not None:sources=(*sources,prior_result,prior_result.parent/'snapshot.json')
    manifest=freeze_sources(output/'sources.zip',config=load_config(),entrypoint=__name__,inputs=sources)
    (output/'snapshot.json').write_text(serialize(state))
    reference=ContinuousDoseReference('sandbox upload accounting');physical={};cache={}
    started=perf_counter()
    def evaluate(program):
        program=tuple(sorted(program))
        if program not in cache:
            for day,key in program:
                if (day,key) not in physical:
                    physical[day,key]=planned_accounted(f'candidate:{day}:{key}',
                        datetime.combine(date.fromisoformat(day),time(18),ZoneInfo(state.timezone)),
                        catalog[key],handoffs[key],reference,available=state.as_of)
            r=evaluate_snapshot(state,tuple(physical[p] for p in program),tuple(k for _,k in program),reference=reference)
            e=r['evaluation']
            cache[program]=dict(cost=e.total_cost,eligible=r['decision_eligible'],
                construction_eligible=r['construction_eligible'],progression_delivery=r['progression_delivery'],
                fatigue=e.weighted_demand_cost,delivery_cost=e.delivery_cost,short=e.short_run_cost,
                score_fingerprint=e.fingerprint)
        return cache[program]
    archived=tuple((d,k) for d,k in event['program'] if d in dates)
    old=evaluate(archived)
    if not empty_rollover:assert abs(old['cost']-event['cost'])<1e-12 and old['construction_eligible']
    previous=prior if prior is not None else archived
    policy=replace(state.context.policy,weekly_miles=event['progression_target']['phase_goal'])
    support=state.context.easy_duration_policy.supporting_easy_miles_per_7+(policy.weekly_miles-state.context.policy.weekly_miles)
    seeds=list(complete_stage_seeds(dates,catalog,policy,supporting_miles=support,limit=max(1,min(168,evaluations//10))))
    # Construction witnesses extend existing early easies at one-decimal steps.
    # These are fully repriced, unprivileged seeds, not pinned dates or overrides.
    witness=previous
    attempts=0
    while not evaluate(witness)['progression_delivery']['funded']:
        if len(cache)>=evaluations//2:break
        candidates=[]
        for day,key in witness:
            if catalog[key].role!='easy':continue
            larger=f'easy:{catalog[key].miles+.1:.1f}'
            if larger not in handoffs:continue
            proposal=tuple((d,larger if d==day else k) for d,k in witness)
            price=evaluate(proposal)
            if (price['construction_eligible'] and
                    price['progression_delivery']['shortfall_miles']<evaluate(witness)['progression_delivery']['shortfall_miles']-1e-8):
                candidates.append((price['cost'],proposal))
        if not candidates:break
        witness=min(candidates)[1];attempts+=1
        if attempts>1000:raise ValueError('Funding witness construction failed to terminate')
    seeds.append(witness)
    seed_keys=set()
    for target in (policy.ordinary_miles,policy.ordinary_miles/2):
        seed_keys.add(min((k for k in handoffs if catalog[k].role=='easy'),key=lambda k:(abs(catalog[k].miles-target),k)))
    for role in ('long','quality'):
        seed_keys.add(max((k for k in handoffs if catalog[k].role==role),key=lambda k:(
            sum(s for p,s in catalog[k].phases if p=='work') if role=='quality' else catalog[k].miles,k)))
    construction_evaluations=len(cache)
    result=search(dates=dates,keys=tuple(sorted(handoffs)),previous=previous,previous_end=event['end'],
        evaluate=evaluate,budget=evaluations-construction_evaluations,hard_end=state.context.hard_end.isoformat(),
        roles={k:catalog[k].role for k in handoffs},seed_keys=tuple(sorted(seed_keys)),
        dose_miles={k:catalog[k].miles for k in handoffs},seed_programs=tuple(seeds),
        exploration_keys=calendar_representatives({k:catalog[k] for k in handoffs},
            ordinary=policy.ordinary_miles,support=support),frontier_repair=True,stable_days=7)
    winner=tuple(map(tuple,result['program']))
    # Fresh independent reprice; no search cache for acceptance evidence.
    reprice=evaluate_snapshot(state,tuple(physical[p] for p in winner),tuple(k for _,k in winner),reference=reference)
    assert reprice['decision_eligible'] and abs(reprice['evaluation'].total_cost-result['cost'])<1e-12
    hard=[(d,catalog[k].role,catalog[k].miles) for d,k in winner if d<state.context.hard_end.isoformat()]
    result.update(manifest=manifest,archived=old,witness=evaluate(witness),
        witness_construction_evaluations=construction_evaluations,
        total_uncached_full_evaluations=len(cache),total_seconds=perf_counter()-started,
        checkpoint=case['checkpoint'].isoformat(),review_contract=asdict(state.prospective_volume_review),
        selection_contract='prospective_volume_review_v1',
        snapshot_fingerprint=state.fingerprint,training=dict(hard_miles=sum(m for _,_,m in hard),
            hard_runs=len(hard),schedule=hard),edits_vs_archived=churn(archived,winner,state.context.decision,catalog),
        edits_vs_prior=churn(previous,winner,state.context.decision,catalog),
        scope='Frozen first-checkpoint admission prototype; no daily stability, missed-run or stage acceptance claim',
        expected_empty_rollover=empty_rollover,
        expected_rollover_scope='No opening run prescribed; unchanged real history/forecasts, moved horizon and decision date only' if empty_rollover else None,
        live_changed=False,weights_changed=False)
    with (output/'result.json').open('x') as stream:json.dump(result,stream,indent=2,default=str)
    print(json.dumps(dict(seconds=result['seconds'],total_seconds=result['total_seconds'],evaluations=result['evaluations'],
        extra_construction_evaluations=result['witness_construction_evaluations'],
        old=old,selected=evaluate(winner),training=result['training'],edits=result['edits_vs_prior']),default=str),flush=True)


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--root',type=Path,default=Path('output/current-v18-volume-curve-daily-2-20260930'))
    p.add_argument('--event',type=int,default=1);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--evaluations',type=int,default=10000);p.add_argument('--review-hour',type=int,default=4)
    p.add_argument('--prior-result',type=Path)
    p.add_argument('--empty-rollover',action='store_true')
    args=p.parse_args()
    if not 0<=args.review_hour<=23:p.error('Review hour must be 0–23')
    run(args.root,args.event,args.output,evaluations=args.evaluations,review_hour=args.review_hour,
        prior_result=args.prior_result,empty_rollover=args.empty_rollover)
