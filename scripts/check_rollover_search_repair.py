"""Bounded frozen search check; unchanged evidence/objective, no daily replay."""
from argparse import ArgumentParser
from dataclasses import replace
from datetime import date, datetime, time, timedelta
import json
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

from scripts.replay_adaptive_search import load_decision, replayed_prior
from scripts.research_manifest import freeze_sources
from scripts.audit_progression_rollover_instability import seed_library, splice_program, score
from run_analysis.config import load_config
from run_analysis.continuous_training_dose import ContinuousDoseReference
from run_analysis.adaptive.decision_snapshot import evaluate_snapshot, serialize
from run_analysis.adaptive.new_model_daily_harness import planned_accounted, churn
from run_analysis.adaptive.continuous_search_adapter import search
from run_analysis.adaptive.decimal_dose_search import calendar_representatives


def cadence_case(case, days):
    """Explicit frozen cadence experiment; preserve original evidence/targets.

    Reallocate only future supporting-volume references. The single-volume
    contract has no separate supporting-volume penalty. Existing cadence-owned
    smoothing/taper and mean continuation follow the new rate; end dates stay
    fixed so this is the same selectable projection, not a longer search.
    """
    if type(days) is not int or days < 2:
        raise ValueError('Explicit integer cadence of at least two days required')
    state=case['state']; context=state.context; policy=context.policy
    if policy.purpose_weights[1]!=0 or not context.purpose_continuation:
        raise ValueError('Cadence experiment requires the enabled single-volume purpose contract')
    refs={role:miles for role,miles,_ in context.continuation_boundary_references}
    rate=7/days
    if policy.long_per_7==rate and policy.quality_per_7==rate:
        return case
    freed=(refs['long']*(policy.long_per_7-rate)+refs['quality']*(policy.quality_per_7-rate))
    support=context.easy_duration_policy.supporting_easy_miles_per_7+freed
    future=tuple(segment if segment.end<=context.decision else
        replace(segment,miles_per_7_days=segment.miles_per_7_days+freed)
        if segment.start>=context.decision else segment
        for segment in context.support_targets)
    # Never rewrite the historical portion of a segment crossing the decision.
    from run_analysis.program_delivery import TargetSegment
    future=tuple(part for segment in future for part in (
        (TargetSegment(segment.start,context.decision,segment.miles_per_7_days),
         TargetSegment(context.decision,segment.end,segment.miles_per_7_days+freed))
        if segment.start<context.decision<segment.end else (segment,)))
    changed=replace(context,policy=replace(policy,long_per_7=rate,quality_per_7=rate),
        easy_duration_policy=replace(context.easy_duration_policy,supporting_easy_miles_per_7=support),
        support_targets=future)
    paths=case['event']['stage_runtime']['target_history']
    event=dict(case['event'],stage_runtime=dict(case['event']['stage_runtime'],target_history=[
        *paths[:-1],dict(paths[-1],support_goal=paths[-1]['support_goal']+freed)]))
    return dict(case,state=replace(state,context=changed),event=event)


def check_empty_rollover(before, after, previous):
    """A clock-only pair cannot silently omit a newly prescribed opening run."""
    a,b=before['state'],after['state']
    if b.context.decision-a.context.decision!=timedelta(days=1):
        raise ValueError('Causal rollover requires consecutive decision dates')
    if any(d==a.context.decision.isoformat() for d,k in previous):
        raise ValueError('Repaired opening plan assigns a run; execute/upload it before rollover')
    fields=('history','phase_history','current','responses','restrictions','contracts',
            'resting_hr','maximum_hr','references_available_at','stage_key')
    if any(getattr(a,f)!=getattr(b,f) for f in fields) or before['handoffs']!=after['handoffs']:
        raise ValueError('Empty rollover must retain physical history, health and forecast evidence')
    settings=('policy','scales','scenario','exchange','timing','easy_duration_policy',
              'delivery_contract','soft_horizon_taper','continuation_guard_days','volume_linear_coefficient',
              'purpose_continuation','purpose_continuation_phase')
    if any(getattr(a.context,f)!=getattr(b.context,f) for f in settings):
        raise ValueError('Empty rollover must retain objective settings and normalization')
    return dict(opening_day=a.context.decision.isoformat(),opening_is_rest=True,
                unchanged_evidence_fields=fields,unchanged_forecasts=True,unchanged_settings=settings)


def run(root, output, *, earlier=22, later=23, evaluations=10000, seed_limit=1680, previous_result=None,
        purpose_continuation=False, cadence_days=None, conditioned_purpose_phase=False):
    before=load_decision(root,earlier); case=load_decision(root,later)
    # Verify historical pricing above BEFORE explicitly migrating the new
    # experimental objective. Never relabel an archived score as a new one.
    if purpose_continuation:
        before=dict(before,state=replace(before['state'],context=replace(
            before['state'].context,purpose_continuation=True)))
        case=dict(case,state=replace(case['state'],context=replace(
            case['state'].context,purpose_continuation=True)))
    if conditioned_purpose_phase:
        before=dict(before,state=replace(before['state'],context=replace(
            before['state'].context,purpose_continuation=True,purpose_continuation_phase='known_full_phase_v1')))
        case=dict(case,state=replace(case['state'],context=replace(
            case['state'].context,purpose_continuation=True,purpose_continuation_phase='known_full_phase_v1')))
    if cadence_days is not None:
        if previous_result is not None or earlier!=later:
            raise ValueError('Cadence comparison must be a single frozen frame, not a causal replay')
        before=cadence_case(before,cadence_days)
        case=cadence_case(case,cadence_days)
    previous=None; causal=None
    if previous_result is not None:
        previous=tuple(map(tuple,replayed_prior(previous_result,root/f'decision-{earlier:04d}.json')))
        saved=json.loads(previous_result.read_text())
        checked=score(before,previous)
        if not checked['eligible'] or abs(checked['cost']-saved['cost'])>1e-12:
            raise ValueError('Repaired prior must reproduce on its original decision evidence')
        causal=check_empty_rollover(before,case,previous)
    output.mkdir(exist_ok=False)
    manifest=freeze_sources(output/'sources.zip',config=load_config(),entrypoint=__name__,
        inputs=(root/f'decision-{earlier:04d}.json',root/f'decision-{later:04d}.json',
                root/before['event']['snapshot_file'],root/case['event']['snapshot_file'],root/'scenario.json',
                *((previous_result,) if previous_result is not None else ())))
    state,dates,catalog,handoffs=(case[k] for k in ('state','dates','catalog','handoffs'))
    supported={k:catalog[k] for k in handoffs}
    prior=tuple(tuple(p) for p in (previous if previous is not None else before['event']['program']) if p[0] in dates)
    archived=tuple(tuple(p) for p in case['event']['program'])
    seeds=seed_library(case,min(seed_limit,max(1,evaluations//5)))
    reference=ContinuousDoseReference('sandbox upload accounting')
    physical={}; cache={}
    def evaluate(program):
        program=tuple(sorted(program))
        if program not in cache:
            for day,key in program:
                if (day,key) not in physical:
                    physical[day,key]=planned_accounted(f'candidate:{day}:{key}',
                        datetime.combine(date.fromisoformat(day),time(18),ZoneInfo(state.timezone)),
                        catalog[key],handoffs[key],reference,available=state.as_of)
            result=evaluate_snapshot(state,tuple(physical[p] for p in program),
                                     tuple(k for _,k in program),reference=reference)
            cache[program]=dict(cost=result['evaluation'].total_cost,
                eligible=result['decision_eligible'],construction_eligible=result['construction_eligible'],
                progression_delivery=result.get('progression_delivery'))
        return cache[program]
    # Comparators are audited, NOT injected as later-outcome-informed seeds.
    archived_current=evaluate(archived)['cost']
    if not purpose_continuation and not conditioned_purpose_phase and cadence_days is None:
        assert abs(archived_current-case['archived_cost'])<1e-12
    known_splice=splice_program(prior,archived,before['state'].context.taper_start.isoformat())
    known_price=score(case,known_splice)
    seed_keys=set()
    for target in (state.context.policy.ordinary_miles,state.context.policy.ordinary_miles/2):
        seed_keys.add(min((k for k in handoffs if catalog[k].role=='easy'),
                         key=lambda k:(abs(catalog[k].miles-target),k)))
    for role in ('long','quality'):
        seed_keys.add(max((k for k in handoffs if catalog[k].role==role),key=lambda k:(
            sum(s for p,s in catalog[k].phases if p=='work') if role=='quality' else catalog[k].miles,k)))
    started=perf_counter()
    result=search(dates=dates,keys=tuple(sorted(handoffs)),previous=prior,
        previous_end=before['event']['end'],evaluate=evaluate,budget=evaluations,
        hard_end=state.context.hard_end.isoformat(),roles={k:catalog[k].role for k in handoffs},
        seed_keys=tuple(sorted(seed_keys)),seed_programs=seeds,
        dose_miles={k:catalog[k].miles for k in handoffs},
        exploration_keys=calendar_representatives(supported,ordinary=state.context.policy.ordinary_miles,
            support=state.context.easy_duration_policy.supporting_easy_miles_per_7),
        recombination_dates=(state.context.taper_start.isoformat(),))
    fresh=score(case,result['program'])
    assert fresh['eligible'] and abs(fresh['cost']-result['cost'])<1e-12
    assert result['evaluations']<=evaluations
    report=dict(result,manifest=manifest,archived_cost=case['archived_cost'],
        archived_cost_current_contract=archived_current,purpose_continuation=state.context.purpose_continuation,
        purpose_continuation_phase=state.context.purpose_continuation_phase,
        cadence_days=cadence_days,
        snapshot_fingerprint=state.fingerprint,causal_empty_rollover=causal,
        previous_plan_source=str(previous_result) if previous_result is not None else str(root/f'decision-{earlier:04d}.json'),
        complete_seed_count=len(seeds),complete_seed_limit=seed_limit,
        original_seconds=case['event']['seconds'],fresh_reprice=fresh,
        known_stable_splice=known_price,comparators_not_seeded=True,
        meets_known_comparator=bool(result['cost']<=min(archived_current,known_price['cost'])+1e-12),
        churn_vs_prior=churn(prior,result['program'],state.context.decision,catalog),
        hard_schedule=[(d,catalog[k].role,catalog[k].miles) for d,k in result['program']
                       if d<state.context.hard_end.isoformat()],
        total_seconds=perf_counter()-started,
        scope='Frozen original evidence search check, not daily/adherence/stability acceptance; live unchanged')
    if causal is not None:
        from scripts.run_adaptive_lifecycle import fast_fail_edits
        report['causal_fast_fail_edits']=fast_fail_edits(dict(churn=report['churn_vs_prior']))
        report['causal_four_day_gate_pass']=not report['causal_fast_fail_edits']
    with (output/'result.json').open('x') as stream:json.dump(report,stream,indent=2,default=str)
    with (output/'snapshot.json').open('x') as stream:stream.write(serialize(state))
    print(json.dumps({k:report[k] for k in ('cost','archived_cost','seconds','evaluations',
        'local_neighborhood_complete','churn_vs_prior','fresh_reprice','known_stable_splice')},default=str),flush=True)
    return report


if __name__=='__main__':
    parser=ArgumentParser()
    parser.add_argument('root',type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--events',type=int,nargs=2,default=(22,23))
    parser.add_argument('--evaluations',type=int,default=10000)
    parser.add_argument('--seed-limit',type=int,default=1680)
    parser.add_argument('--previous-result',type=Path)
    parser.add_argument('--purpose-continuation',action='store_true')
    parser.add_argument('--conditioned-purpose-phase',action='store_true')
    args=parser.parse_args()
    run(args.root,args.output_dir,earlier=args.events[0],later=args.events[1],evaluations=args.evaluations,
        seed_limit=args.seed_limit,previous_result=args.previous_result,
        purpose_continuation=args.purpose_continuation,conditioned_purpose_phase=args.conditioned_purpose_phase)
