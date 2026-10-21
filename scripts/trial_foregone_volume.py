"""Explicit copied-runtime catch-up repair; preserve all earlier comparisons."""
from argparse import ArgumentParser
from dataclasses import replace
from datetime import date,datetime,timedelta
import json
from pathlib import Path
from unittest.mock import patch

from run_analysis.config import load_config
from run_analysis.adaptive.factory import resume_service
from run_analysis.adaptive.hr_reference import frozen_live_config
from run_analysis.adaptive.foregone_volume import VERSION,LEGACY_VERSION
from scripts.run_adaptive_lifecycle import fork_seed_plan,run
from scripts.replay_adaptive_search import load_decision
from scripts.research_manifest import freeze_sources


def opening(source,output):
    output.mkdir(exist_ok=True,parents=True)
    root=output/'opening';at=fork_seed_plan(source,root)+timedelta(microseconds=1)
    config=frozen_live_config(load_config(),root=source)
    manifest=freeze_sources(output/'opening.sources.zip',config=config,entrypoint=__name__,
        inputs=tuple(source.glob('decision-*.json'))+(source/'scenario.json',))
    service=resume_service(root,config,at);h=service.harness
    try:
        h.policy=replace(h.policy,volume_shortfall_policy=VERSION)
        before=h.stage,h.catalog,h.scales,h.target_history,h.training_policy
        result=service.refresh(at);new=service.harness;event=result['event']
        assert (new.stage,new.catalog,new.scales,new.training_policy)==(before[0],before[1],before[2],before[4])
        state=load_decision(root,len(new.events))['state']
        assert state.context.volume_shortfall_policy==VERSION and not state.context.foregone_volume
        report=dict(status='opening_only',manifest=manifest,as_of=event['as_of'],
            seconds=event['seconds'],churn=event['churn'],miles21=event['volume'],
            policy=VERSION,live_changed=False)
        with (output/'opening.json').open('x') as stream:json.dump(report,stream,indent=2)
        print(json.dumps(report),flush=True)
    finally:service.harness.sandbox.close()


def daily(source,output,days,scenario,label,missed,*,resume=False,checkpoint_days=None):
    config=frozen_live_config(load_config(),root=source)
    from zoneinfo import ZoneInfo
    e=json.loads(sorted((output/'opening').glob('decision-*.json'))[-1].read_text())
    start=datetime.fromisoformat(e['as_of']).astimezone(ZoneInfo(config['timezone_default'])).date()+timedelta(days=1)
    with patch('scripts.run_adaptive_lifecycle.load_config',return_value=config):
        result=run(output/label,Path('data/run_analysis.sqlite'),days=days,scenario=scenario,resume=resume,
            seed_plan=None if resume else output/'opening',screen_start=start.isoformat(),stability_window=4,
            checkpoint_days=checkpoint_days,missed_run_offsets=tuple(missed))
    print(json.dumps({k:result[k] for k in ('status','completed_days','requested_days','failures')}),flush=True)


def parity(source,output,*,overshoot=False):
    """Exact no-miss equivalence across every archived perfect daily frame."""
    filename='lifecycle-summary.json' if overshoot else 'resume-0001.summary.json'
    report=json.loads((source/filename).read_text())
    if not overshoot:
        assert report['completed_days']==report['requested_days']==21 and not report['failures']
    # Perfect execution is verified, not assumed just because a scenario says so.
    for upload in report['uploads']:
        fingerprint=upload['comparison']['issued_fingerprint']
        owner=next(e for e in reversed(report['events']) if any(
            x['fingerprint']==fingerprint for x in e['active'].values()))
        day=upload['day'];key=owner['active'][day]['dose']
        catalog={x['key']:x for x in owner['stage_runtime']['catalog']}
        delta=upload['comparison']['actual_distance_miles']-catalog[key]['miles']
        assert delta>=-1e-8 if overshoot else abs(delta)<1e-8
    rows=[]
    since=date.fromisoformat(report['events'][0]['decision'])
    for event in report['events']:
        case=load_decision(source,event['index']+1)
        state=replace(case['state'],context=replace(case['state'].context,
            volume_shortfall_policy=LEGACY_VERSION,volume_shortfall_since=since,foregone_volume=()))
        result=case['evaluate_details'](tuple(map(tuple,event['program'])),state_override=state)
        e=result['evaluation']
        assert result['decision_eligible'] and abs(e.total_cost-event['cost'])<1e-12
        rows.append(dict(event=event['index'],eligible=True,identical_score=True))
    output.mkdir(exist_ok=True,parents=True)
    stem='overshoot-parity' if overshoot else 'perfect-parity'
    manifest=freeze_sources(output/(stem+'.sources.zip'),config=load_config(),entrypoint=__name__,
        inputs=(source/filename,)+tuple(source.glob('snapshot-*.json'))+tuple(source.glob('decision-*.json')))
    with (output/(stem+'.json')).open('x') as stream:
        json.dump(dict(no_search=True,verified_perfect_uploads=len(report['uploads']),
            source_report=filename,overshoot_control=overshoot,manifest=manifest,
            decisions=rows,live_changed=False),stream,indent=2)
    print(json.dumps(dict(no_search=True,verified_perfect_uploads=len(report['uploads']),
        identical_eligible_decisions=len(rows))),flush=True)


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('opening','daily','parity','overshoot-parity'))
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--days',type=int,default=7)
    p.add_argument('--scenario',choices=('prescribed','imperfect'),default='prescribed')
    p.add_argument('--label',default='missed-easy-7d')
    p.add_argument('--missed-run-offset',type=int,action='append',default=[])
    p.add_argument('--resume',action='store_true')
    p.add_argument('--checkpoint-days',type=int)
    a=p.parse_args()
    if not a.label or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in a.label):
        p.error('Label must contain lowercase letters, numbers, hyphens or underscores')
    if a.mode=='opening':opening(a.source,a.output)
    elif a.mode in {'parity','overshoot-parity'}:parity(a.source,a.output,overshoot=a.mode=='overshoot-parity')
    else:daily(a.source,a.output,a.days,a.scenario,a.label,a.missed_run_offset,
        resume=a.resume,checkpoint_days=a.checkpoint_days)
