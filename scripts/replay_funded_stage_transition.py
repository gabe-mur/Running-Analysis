"""Replay failed growth and a same-evidence refresh in a copied sandbox only."""
from argparse import ArgumentParser
from datetime import datetime,timedelta,date,time
from dataclasses import asdict,replace
import json
from pathlib import Path
import shutil
import sqlite3

from run_analysis.config import load_config
from run_analysis.adaptive.factory import resume_service
from run_analysis.adaptive.stage_acceptance import StageLedger
from run_analysis.adaptive.decision_snapshot import restore,evaluate_snapshot
from run_analysis.adaptive.new_model_daily_harness import planned_accounted
from scripts.research_manifest import freeze_sources


def funding(h,event,*,program_override=None):
    state=restore((h.sandbox.root/event['snapshot_file']).read_text(),
        expected_fingerprint=event['snapshot_fingerprint'])
    program=tuple(map(tuple,event['program'] if program_override is None else program_override))
    runs=tuple(planned_accounted(f'candidate:{d}:{k}',
        datetime.combine(date.fromisoformat(d),time(18),h.zone),h.catalog[k],
        event['forecast_handoff_by_dose'][k],h.ref,available=state.as_of) for d,k in program)
    result=evaluate_snapshot(state,runs,tuple(k for _,k in program),reference=h.ref)
    parts=result['evaluation'].parts
    if program_override is None and abs(result['evaluation'].total_cost-event['cost'])>1e-12:
        raise AssertionError('Archived score mismatch')
    support=parts['easy_duration']['supporting_easy_volume']
    return dict(cost=result['evaluation'].total_cost,eligible=result['decision_eligible'],
        delivery=parts['delivery'],targets=parts['targets'],supporting=support,
        funded=all(parts['delivery'][p]+1e-8>=parts['targets'][p] for p in ('volume','long','quality'))
        and support['delivered_miles']+1e-8>=support['target_miles'])


def run(root,out,*,sustained=False):
    out.mkdir(exist_ok=False)
    config=load_config()
    manifest=freeze_sources(out/'sources.zip',config=config,
        entrypoint='scripts.replay_funded_stage_transition',inputs=(root/'sandbox.sqlite',root/'scenario.json'))
    clone=out/'sandbox'
    clone.mkdir()
    for name in ('scenario.json','execution-sandbox.json'):
        shutil.copy2(root/name,clone/name)
    with sqlite3.connect(f'file:{root.resolve()}/sandbox.sqlite?mode=ro',uri=True) as source:
        with sqlite3.connect(clone/'sandbox.sqlite') as target:
            source.backup(target)
        at=datetime.fromisoformat(source.execute('SELECT max(available_at) FROM sandbox_completions').fetchone()[0])+timedelta(seconds=1)
    service=resume_service(clone,config,at)
    report=dict(manifest=manifest,scope='Copied failed transition and same-evidence refresh; not daily acceptance')
    if sustained:
        h=service.harness
        h.policy=replace(h.policy,delivery_contract='sustained_v1')
        from math import ceil
        smoothing=ceil(max(7/h.training_policy.long_per_7,7/h.training_policy.quality_per_7))
        start=h.origin.astimezone(h.zone).date()-timedelta(days=h.training_policy.volume_window_days+smoothing-2)
        first=h.target_history[0]
        h.target_history=(replace(first,effective_on=min(first.effective_on,start)),*h.target_history[1:])
        report['explicit_migration']=dict(delivery_contract='sustained_v1',historical_reference_unchanged=True,
            first_target_start=h.target_history[0].effective_on.isoformat(),smoothing_days=smoothing)
    try:
        for label,stamp in (('growth',at),('same_evidence_refresh',at+timedelta(seconds=1))):
            decision=service.refresh(stamp,reason=label)
            h=service.harness
            event=decision['event']
            row=dict(stage=asdict(h.stage),review_status=decision['stage_review']['status'],
                funding=funding(h,event),event_index=event['index'],
                delivery_contract=h.policy.delivery_contract,
                **{k:event[k] for k in ('cost','seconds','evaluations','local_neighborhood_complete','churn','volume')})
            report[label]=row
            print(json.dumps(dict(label=label,**{k:row[k] for k in ('review_status','funding','cost','seconds','evaluations','churn')})),flush=True)
            with (out/f'{label}.json').open('x') as f:
                json.dump(row,f,indent=2,default=lambda v:v.isoformat())
        current=StageLedger.for_sandbox(h.sandbox).current
        assert current==h.stage
        report['committed_stage']=asdict(current)
        # Verify the new contract survives restart instead of reverting to the
        # copied original scenario's historical selection policy.
        service.harness.sandbox.close()
        reopened=resume_service(clone,config,at+timedelta(seconds=2))
        service=reopened
        report['restart_contract']=reopened.harness.policy.delivery_contract
        assert report['restart_contract']==report['growth']['delivery_contract']
    finally:
        service.harness.sandbox.close()
    with (out/'report.json').open('x') as f:
        json.dump(report,f,indent=2,default=lambda v:v.isoformat())


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--sustained',action='store_true')
    a=p.parse_args()
    run(a.root,a.output_dir,sustained=a.sustained)
