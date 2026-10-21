"""Copied-live full-path publication, original-HR upload, restart and replan seam.

One engineering execution with actual donor HR and explicit transformed work
windows. Not a real adherence or predictor-accuracy claim; no live writes.
"""
from argparse import ArgumentParser
from datetime import datetime,time,timedelta
from pathlib import Path
import json
import sqlite3

from run_analysis.config import load_config,resolve_project_path
from run_analysis.adaptive.new_model_daily_harness import DailyHarness
from run_analysis.adaptive.quality_forecast_runtime import VERSION
from run_analysis.adaptive.coached_runtime import enable
from run_analysis.adaptive.service import AdaptivePlannerService
from run_analysis.adaptive.quality_assignment_accounting import completion_json
from scripts.run_adaptive_lifecycle import fork_seed_plan,fast_fail_edits,fast_fail_quality_edits,verify_prescribed_quality
from scripts.recorded_trace_execution import compatible_donor,transformed_payload
from scripts.replay_adaptive_search import load_decision
from scripts.research_manifest import freeze_sources


def run(baseline,native,output,*,resume_opening=False):
    if not resume_opening:output.mkdir(exist_ok=False)
    config=load_config()
    if resume_opening:
        old=json.loads((output/'stopped.json').read_text())
        if old['phase']!='restart_before_upload':raise ValueError('Only an attributed pre-upload driver stop can resume here')
        last=datetime.fromisoformat(json.loads(max((output/'runtime').glob('decision-*.json')).read_text())['as_of'])
        native_at=datetime.fromisoformat(json.loads(max((output/'native-input').glob('decision-*.json')).read_text())['as_of'])
    else:
        last=fork_seed_plan(baseline/'runtime',output/'runtime')
        native_at=fork_seed_plan(native,output/'native-input')
    h=DailyHarness.resume(output/'runtime',config,as_of=last)
    n=DailyHarness.resume(output/'native-input',config,as_of=native_at)
    phase='opening'
    try:
        if (h.stage.key!=n.stage.key or h.training_policy!=n.training_policy or h.scales!=n.scales):
            raise ValueError('Cannot replace live stage or fatigue normalization')
        if not resume_opening:
            additions={k:d for k,d in n.catalog.items() if d.workout_definition and
                d.workout_definition.get('calendar_integration')==VERSION}
            h.catalog.update(additions);h.coaching_contracts.update({k:n.coaching_contracts[k] for k in additions})
            h.quality_runtime=n.quality_runtime;enable(h)
        elif h.coaching_sequence is None or h.coaching_sequence.consumed!=0 or h.events[-1]['reason']!='coached_sandbox_opening':
            raise ValueError('Resume must retain the committed opening before its first upload')
        manifest=freeze_sources(output/('resume-0001.sources.zip' if resume_opening else 'sources.zip'),config=config,entrypoint=__name__,
            inputs=(baseline/'baseline.json',baseline/'runtime/scenario.json',native/'scenario.json'))
        opening=h.events[-1] if resume_opening else h.replan(last+timedelta(microseconds=1),reason='coached_sandbox_opening')
        index=len(h.events);frozen=load_decision(output/'runtime',index)
        if abs(frozen['archived_cost']-opening['cost'])>1e-12:raise ValueError('Published snapshot no longer reproduces')
        print(json.dumps(dict(phase=phase,seconds=opening['seconds'],cost=opening['cost'],
            independently_reproduced=True,assignments=len(opening['coached_assignments']))),flush=True)
        phase='restart_before_upload'
        prior_sequence=h.coaching_sequence;h.sandbox.close()
        h=DailyHarness.resume(output/'runtime',config,as_of=datetime.fromisoformat(opening['as_of']))
        if h.coaching_sequence!=prior_sequence:raise ValueError('Restart reset issued sequence')
        day=opening['decision'];chosen=h.active.get(day)
        if not chosen or h.catalog[chosen['dose']].role!='quality':raise ValueError('Opening needs its assigned quality seam')
        dose=h.catalog[chosen['dose']]
        database=resolve_project_path(Path.cwd(),config['paths']['database'])
        _,donor=compatible_donor(database,[r for r in h.history if r.role=='quality'],dose,config,h.ref,{})
        start=datetime.combine(datetime.fromisoformat(day).date(),time(18),h.zone)
        payload,phases,end,provenance=transformed_payload(donor,dose,start)
        phase='upload'
        uploaded=h.upload(payload,day=day,available_at=end,observed_role='quality',observed_phases=phases)
        provenance['primary_fulfillment_check']=verify_prescribed_quality(h,
            uploaded['comparison']['actual_key'],quality=True)
        receipts=[completion_json(r) for r in h.coaching_receipts]
        consumed=h.coaching_sequence.consumed
        if consumed!=1:raise ValueError('Primary-compliant donor work did not verify its issued purpose; inspect measured evidence')
        retry=h.upload(payload,day=day,available_at=end+timedelta(seconds=1),observed_role='quality',observed_phases=phases)
        if retry['status']!='duplicate' or consumed!=h.coaching_sequence.consumed:
            raise ValueError('Retry changed issued-sequence ownership')
        print(json.dumps(dict(phase=phase,status=uploaded['status'],consumed=consumed,
            receipt=json.loads(receipts[0]),hr_provenance=provenance)),flush=True)
        phase='restart_after_upload'
        h.sandbox.close();h=DailyHarness.resume(output/'runtime',config,as_of=end+timedelta(seconds=1))
        if h.coaching_sequence.consumed!=consumed or [completion_json(r) for r in h.coaching_receipts]!=receipts:
            raise ValueError('Durable upload receipt or sequence changed on restart')
        phase='post_upload_search'
        inputs=dict(history=tuple(r for r in h.history if r.key in h.seed_keys),
            sessions=tuple(r for r in h.session_evidence() if r.key in h.seed_keys),phase_history=h.phase_history,reference=h.ref)
        service=AdaptivePlannerService(h,inputs)
        post=service.refresh(end+timedelta(seconds=2),reason='post_upload')['event'];h=service.harness
        checked=load_decision(output/'runtime',len(h.events))
        if abs(checked['archived_cost']-post['cost'])>1e-12:raise ValueError('Post-upload frozen score does not reproduce')
        gates={str(window):dict(date_type_dose=fast_fail_edits(post,window),
            quality_family_recipe=fast_fail_quality_edits(post,h.catalog,window)) for window in (4,7)}
        clean=all(not v['date_type_dose'] and not v['quality_family_recipe'] for v in gates.values())
        report=dict(status='bounded_seams_clean' if clean else 'stopped_visible_edit_requires_attribution',
            opening=dict(index=index,seconds=opening['seconds'],cost=opening['cost'],miles21=opening['volume']),
            post_upload=dict(index=len(h.events),seconds=post['seconds'],cost=post['cost'],miles21=post['volume'],churn=post['churn']),
            gates=gates,receipts=[json.loads(r) for r in receipts],sequence_consumed=h.coaching_sequence.consumed,
            retry_status=retry['status'],restart_exact=True,full_snapshot_repricing=True,
            hr_provenance=provenance,manifest=manifest,live_changed=False,
            quality_cadence_contract=h.coaching_sequence.cadence_contract,
            driver_recovery=(dict(original_stop=old,opening_search_reused=True,
                correction='Pass original database path to existing donor helper; no planner/input change') if resume_opening else None),
            scope='One transformed original-HR primary-compliance engineering seam; no daily stability or measured adherence claim')
        (output/'report.json').write_text(json.dumps(report,default=str,indent=2,allow_nan=False))
        print(json.dumps(report,default=str,indent=2),flush=True)
    except Exception as error:
        (output/('stopped-0002.json' if resume_opening else 'stopped.json')).write_text(
            json.dumps(dict(phase=phase,error=repr(error),live_changed=False),indent=2))
        raise
    finally:
        h.sandbox.close();n.sandbox.close()


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('baseline',type=Path);p.add_argument('native',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume-opening',action='store_true')
    a=p.parse_args();run(a.baseline,a.native,a.output,resume_opening=a.resume_opening)
