"""Reprice the genuine omitted-run prefix, never the already failed calendar.

Causal pruning is confined to a newly created diagnostic copy. Original reports,
physical evidence and live state are preserved. One refresh is not daily acceptance.
"""
from argparse import ArgumentParser
from datetime import datetime
import json
from pathlib import Path
import shutil
import sqlite3

from run_analysis.config import load_config
from run_analysis.adaptive.factory import resume_service
from scripts.replay_adaptive_search import load_decision
from scripts.research_manifest import freeze_sources
from scripts.run_adaptive_lifecycle import fast_fail_edits


def prepare_continuation(output):
    """Retain the original completed prefix and original transformed HR bytes."""
    report=json.loads((output/'report.json').read_text())
    if report['status']!='one_refresh_clean':raise ValueError('Attribute the new refresh before continuing')
    source=Path(report['source']);root=output/'sandbox'
    prefix=json.loads((source/'lifecycle-summary.json').read_text())
    at=datetime.fromisoformat(report['event']['as_of'])
    if prefix['status']!='bounded_screen_complete':raise ValueError('Original executed prefix was not clean')
    config=load_config();service=resume_service(root,config,at);h=service.harness
    try:
        observed=(at.astimezone(h.zone).date()-h.origin.astimezone(h.zone).date()).days
        if prefix['completed_days']!=observed:raise ValueError('Repair must be the next original daily boundary')
        known={r.key for _,r in h.sandbox.completed_accounted(at)}
        for path in (*source.glob('execution-[0-9][0-9][0-9][0-9].json'),*source.glob('missed-run-*.json')):
            value=json.loads(path.read_text())
            if path.name.startswith('execution-'):
                if value['comparison']['actual_key'] not in known:
                    raise ValueError('Future execution cannot enter the repaired prefix')
            target=root/path.name
            if target.exists():
                if target.read_bytes()!=path.read_bytes():raise ValueError('Original execution bytes changed')
            else:shutil.copy2(path,target)
        result=h.summary(prefix['completed_days'],prefix['uploads'])
        result.update(status='bounded_screen_complete',scenario=prefix['scenario'],
            completed_days=prefix['completed_days'],requested_days=prefix['requested_days'],
            failures=[],final_stage=h.stage,stage_reviews=(*prefix['stage_reviews'],report['review']),
            stability_window=4,initial_search_reused=True,refresh_clock=prefix['refresh_clock'],
            missed_run_offsets=prefix['missed_run_offsets'],missed_runs=prefix['missed_runs'],
            recovery=dict(kind='attributed_omission_prefix_repair',original_root=str(source),
                original_failed_resume=str(source/'resume-0002.summary.json'),
                explanation='Same clean original prefix; failed later calendar excluded before one admitted repair; no future execution or HR donor'),
            acceptance_scope='Repaired prefix with original-HR evidence; extended acceptance pending')
        with (root/'lifecycle-summary.json').open('x') as stream:
            from dataclasses import asdict
            json.dump(result,stream,indent=2,default=lambda v:asdict(v) if hasattr(v,'__dataclass_fields__') else v.isoformat())
    finally:h.sandbox.close()


def run(source,output,*,index=4,as_of):
    case=load_decision(source,index);event=case['event'];stamp=datetime.fromisoformat(event['as_of'])
    if as_of<=stamp:raise ValueError('Refresh must follow the original prefix')
    output.mkdir(exist_ok=False);root=output/'sandbox';root.mkdir()
    config=load_config()
    manifest=freeze_sources(output/'sources.zip',config=config,entrypoint=__name__,
        inputs=(source/'scenario.json',source/'sandbox.sqlite',source/f'decision-{index:04d}.json'))
    for name in ('scenario.json','execution-sandbox.json'):
        shutil.copy2(source/name,root/name)
    for number in range(1,index+1):
        for prefix in ('decision','snapshot'):
            name=f'{prefix}-{number:04d}.json';shutil.copy2(source/name,root/name)
    with sqlite3.connect((source/'sandbox.sqlite').resolve().as_uri()+'?mode=ro',uri=True) as original:
        with sqlite3.connect(root/'sandbox.sqlite') as copy:
            original.backup(copy)
    with sqlite3.connect(root/'sandbox.sqlite') as copy:
        copy.execute('DELETE FROM sandbox_decisions WHERE sequence>=?',(index,))
        for key,at in copy.execute('SELECT fingerprint,issued_at FROM sandbox_issues').fetchall():
            if datetime.fromisoformat(at)>stamp:
                copy.execute('DELETE FROM sandbox_issues WHERE fingerprint=?',(key,))
        for key,at in copy.execute('SELECT activity_id,available_at FROM sandbox_completions').fetchall():
            if datetime.fromisoformat(at)>stamp:
                copy.execute('DELETE FROM sandbox_completions WHERE activity_id=?',(key,))
        if copy.execute('SELECT count(*) FROM sandbox_completion_versions').fetchone()[0]:
            raise ValueError('Corrections require a separate prefix adapter')
        keep=set();key=event['stage_runtime']['stage']['key']
        while key:
            keep.add(key);row=copy.execute('SELECT parent FROM stages WHERE key=?',(key,)).fetchone()
            if row is None:raise ValueError('Original stage ancestry missing')
            key=row[0]
        for (key,) in copy.execute('SELECT key FROM stages').fetchall():
            if key not in keep:copy.execute('DELETE FROM stages WHERE key=?',(key,))
        copy.execute('UPDATE current_stage SET key=?',(event['stage_runtime']['stage']['key'],))
    service=resume_service(root,config,stamp)
    try:
        before=service.harness.stage
        result=service.refresh(as_of,reason='original missed-opportunity prefix refresh')
        h=service.harness;e=result['event']
        report=dict(manifest=manifest,source=str(source),source_index=index,original_cost=case['archived_cost'],
            status='one_refresh_clean' if not fast_fail_edits(e) else 'stopped_for_attribution',
            before_stage=before,after_stage=h.stage,review=result['stage_review'],event=e,
            four_day_failures=fast_fail_edits(e),live_changed=False,
            scope='Actual original-HR omission prefix and one new refresh; not extended imperfect acceptance')
        with (output/'report.json').open('x') as stream:
            from dataclasses import asdict
            json.dump(report,stream,indent=2,default=lambda v:asdict(v) if hasattr(v,'__dataclass_fields__') else v.isoformat())
        print(json.dumps(dict(status=report['status'],review=result['stage_review']['status'],
            miles21=e['volume'],seconds=e['seconds'],churn=e['churn'])),flush=True)
        return report
    finally:service.harness.sandbox.close()


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('source',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--index',type=int,default=4)
    p.add_argument('--as-of',type=datetime.fromisoformat,required=True);a=p.parse_args()
    run(a.source,a.output,index=a.index,as_of=a.as_of)
