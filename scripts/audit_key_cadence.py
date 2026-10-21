"""Two bounded unrestricted searches on one frozen v19 frame. No deployment."""
from argparse import ArgumentParser
from datetime import date, datetime, time, timedelta
import json
from pathlib import Path
from statistics import mean
from zoneinfo import ZoneInfo

from scripts.check_rollover_search_repair import run as search_frame, cadence_case
from scripts.replay_adaptive_search import load_decision, recurrence_summary
from scripts.audit_progression_rollover_instability import score, seed_library
from scripts.audit_adaptive_daily_replay import target_integral
from scripts.check_explicit_distribution_tail import incoming_demand_trace
from scripts.research_manifest import freeze_sources
from run_analysis.config import load_config
from run_analysis.adaptive.decision_snapshot import evaluate_snapshot
from run_analysis.adaptive.new_model_daily_harness import planned_accounted
from run_analysis.adaptive.program_objective_trial import AXES
from run_analysis.continuous_training_dose import ContinuousDoseReference


def mean_key_entry(keys):
    return {role:{axis:mean(r['preceding_by_axis'][axis] for r in keys if r['role']==role)
        for axis in AXES} for role in ('long','quality') if any(r['role']==role for r in keys)}


def summarize(case,program):
    state=case['state']; context=state.context; catalog=case['catalog']
    reference=ContinuousDoseReference('sandbox upload accounting')
    physical=tuple(planned_accounted(f'candidate:{d}:{k}',datetime.combine(
        date.fromisoformat(d),time(18),ZoneInfo(state.timezone)),catalog[k],case['handoffs'][k],
        reference,available=state.as_of) for d,k in program)
    checked=evaluate_snapshot(state,physical,tuple(k for _,k in program),reference=reference)
    trace=incoming_demand_trace(checked['evaluation'],context)
    hard=tuple((date.fromisoformat(d),catalog[k]) for d,k in program if date.fromisoformat(d)<context.hard_end)
    active={d for d,_ in hard}; streak=maximum=0; rests=[]
    schedule=[]
    for offset in range((context.hard_end-context.decision).days):
        day=context.decision+timedelta(days=offset)
        dose=next((v for d,v in hard if d==day),None)
        streak=streak+1 if dose else 0; maximum=max(maximum,streak)
        if dose is None:rests.append(day.isoformat())
        schedule.append(dict(day=day.isoformat(),role=dose.role if dose else 'rest',miles=dose.miles if dose else 0.))
    gaps={}
    observed=[r for r in checked['evaluation'].conditional_ledger if r.source=='observed']
    for role in ('long','quality'):
        days=sorted([r.day for r in observed if r.role==role]+[d for d,v in hard if v.role==role])
        gaps[role]=[dict(previous=a.isoformat(),next=b.isoformat(),days=(b-a).days)
            for a,b in zip(days,days[1:]) if b>=context.decision]
    byrole={role:dict(runs=sum(v.role==role for _,v in hard),miles=sum(v.miles for _,v in hard if v.role==role))
        for role in ('easy','long','quality')}
    keys=[r for r in trace if r['role'] in ('long','quality')]
    target=target_integral(context.volume_targets,context.decision,context.hard_end)
    all_days=sorted({r.day for r in observed}|active)
    rolling=[dict(day=(context.decision+timedelta(days=n)).isoformat(),miles=sum(
        r.miles for r in checked['evaluation'].conditional_ledger if context.decision+timedelta(days=n-6)<=r.day<=context.decision+timedelta(days=n)))
        for n in range((context.hard_end-context.decision).days)]
    return dict(score=score(case,program),schedule=schedule,by_role=byrole,
        miles21=sum(v.miles for _,v in hard),runs21=len(hard),rest_dates=rests,maximum_running_streak=maximum,
        raw_hard_session_demand_cost=sum(r['raw_weighted_own']+r['raw_weighted_overlap'] for r in trace),
        raw_hard_key_overlap_cost=sum(r['raw_weighted_overlap'] for r in keys),
        target_integral21=target,funding_ratio21=sum(v.miles for _,v in hard)/target,
        maximum_bracketed_run_gap=max(((b-a).days for a,b in zip(all_days,all_days[1:]) if b>=context.decision),default=None),
        key_gaps=gaps,key_entry=keys,
        mean_key_entry_by_role=mean_key_entry(keys),
        daily_rolling_7_miles=rolling,mean_rolling_7_miles=mean(r['miles'] for r in rolling),
        recurrence=recurrence_summary(program,catalog,tuple(r['day'] for r in schedule)),
        full_projection_recurrence=recurrence_summary(program,catalog,case['dates']),
        derived=dict(volume_smoothing_days=context.volume_smoothing_days,taper_start=context.taper_start.isoformat(),
            end=context.end.isoformat(),hard_end=context.hard_end.isoformat(),
            supporting_easy_miles_per_7=context.easy_duration_policy.supporting_easy_miles_per_7,
            mean_continuation=checked['evaluation'].parts['continuation_boundary']['expected_support_per_7']))


def run(root,output,*,index=10,evaluations=10000):
    original=load_decision(root,index)  # Mandatory unmodified archived parity.
    output.mkdir(exist_ok=False)
    trials={};cases={}
    for cadence in (7,8):
        cases[cadence]=cadence_case(original,cadence)
        trials[cadence]=search_frame(root,output/f'cadence-{cadence}',earlier=index,later=index,
            evaluations=evaluations,cadence_days=cadence)
        print(json.dumps(dict(milestone='cadence_search_complete',days=cadence,
            cost=trials[cadence]['cost'],seconds=trials[cadence]['total_seconds'])),flush=True)
    return finish(root,output,original,cases,trials,index=index)


def finish(root,output,original,cases,trials,*,index,recovered=False,revision=1):
    if type(revision) is not int or revision<1:raise ValueError('Positive report revision required')
    summaries={str(c):summarize(cases[c],tuple(map(tuple,trials[c]['program']))) for c in trials}
    for c in trials:
        assert summaries[str(c)]['score']['eligible']
        assert abs(summaries[str(c)]['score']['cost']-trials[c]['cost'])<1e-12
        assert trials[c]['snapshot_fingerprint']==cases[c]['state'].fingerprint
    matrix={str(model):{str(winner):score(cases[model],tuple(map(tuple,trials[winner]['program'])))
        for winner in trials} for model in trials}
    report=dict(source=str(root),event=index,as_of=original['event']['as_of'],
        archived_parity_verified=True,searches={str(c):dict(seconds=t['total_seconds'],evaluations=t['evaluations'],
            local_neighborhood_complete=t['local_neighborhood_complete'],complete_seed_count=t['complete_seed_count'],
            manifest=t['manifest'],snapshot_fingerprint=t['snapshot_fingerprint']) for c,t in trials.items()},
        summaries=summaries,cross_price_matrix=matrix,report_only_recovery=recovered,report_revision=revision,
        complete_seed_library_sizes={str(c):len(seed_library(cases[c],100000)) for c in trials},
        audit_manifest=freeze_sources(output/('audit.sources.zip' if revision==1 else f'audit-v{revision}.sources.zip'),config=load_config(),entrypoint=__name__,
            inputs=(root/f'decision-{index:04d}.json',root/original['event']['snapshot_file'],root/'scenario.json',
                *(output/f'cadence-{c}'/name for c in trials for name in ('result.json','snapshot.json')))),
        limitations=['Frozen projection comparison, not daily/adherence/stability acceptance or live recommendation',
            'Seven and eight-day utility scores are different policy contracts; compare physical demand and delivery separately',
            'Same archived physical incumbent for both searches; no continuity penalty, date lock or forced repeating schedule',
            'Cadence also owns volume smoothing and continuation assumptions; these are disclosed, not independent ablations',
            'Forecast HR/duration/work uncertainty retained; modeled carryover is not measured recovery or tendon clearance',
            'Finite 10000-evaluation search is not a global optimality certificate'],live_server_unchanged=True)
    with (output/('report.json' if revision==1 else f'report-v{revision}.json')).open('x') as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(status='frozen_comparison_complete',output=str(output))),flush=True)
    return report


def summarize_existing(root,output,*,index,revision=1):
    """Reporting-only repair: verify saved search outputs, never repeat searches."""
    original=load_decision(root,index)
    trials={c:json.loads((output/f'cadence-{c}'/'result.json').read_text()) for c in (7,8)}
    return finish(root,output,original,{c:cadence_case(original,c) for c in trials},trials,
        index=index,recovered=True,revision=revision)


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--event',type=int,default=10);p.add_argument('--evaluations',type=int,default=10000)
    p.add_argument('--summarize-existing',action='store_true')
    p.add_argument('--report-revision',type=int,default=1)
    a=p.parse_args()
    if a.summarize_existing:summarize_existing(a.root,a.output_dir,index=a.event,revision=a.report_revision)
    else:run(a.root,a.output_dir,index=a.event,evaluations=a.evaluations)
