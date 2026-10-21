"""Copied-runtime continuous general progression screen; no live activation."""
from argparse import ArgumentParser
from dataclasses import replace
from datetime import datetime,timedelta
import json
from pathlib import Path
from unittest.mock import patch

from run_analysis.config import load_config
from run_analysis.adaptive.hr_reference import frozen_live_config
from run_analysis.adaptive.factory import resume_service
from run_analysis.adaptive.continuous_general_progression import VERSION
from scripts.run_adaptive_lifecycle import fork_seed_plan,run
from scripts.replay_adaptive_search import load_decision


def opening(source,output):
    root=output/'opening'
    at=fork_seed_plan(source,root)+timedelta(microseconds=1)
    config=frozen_live_config(load_config(),root=source)
    service=resume_service(root,config,at)
    h=service.harness
    try:
        h.policy=replace(h.policy,general_progression=VERSION)
        before=h.stage,h.scales,dict(h.catalog),h.training_policy,h.context(at).stage_start
        result=service.refresh(at)
        new=service.harness
        assert (new.stage,new.scales,new.catalog,new.training_policy,new.context(at).stage_start)==before
        event=result['event']
        case=load_decision(root,len(new.events))
        assert case['evaluate_details'](tuple(map(tuple,event['program'])))['decision_eligible']
        report=dict(status='opening_only',review=result['stage_review'],seconds=event['seconds'],
            prior_churn=event['churn'],volume=event['volume'],
            schedule=[dict(day=d,role=new.catalog[k].role,miles=new.catalog[k].miles)
                for d,k in event['program'] if d<event['hard_end']],
            stage_catalog_scales_and_accounting_origin_unchanged=True,
            live_changed=False)
        from dataclasses import asdict,is_dataclass
        (output/'opening.json').write_text(json.dumps(report,indent=2,
            default=lambda v:asdict(v) if is_dataclass(v) else v.isoformat())+'\n')
        print(json.dumps({k:v for k,v in report.items() if k!='review'}),flush=True)
    finally:service.harness.sandbox.close()


def daily(source,output,database,days,*,resume=False,scenario='prescribed',label=None,
          checkpoint_days=None,missed_run_offsets=()):
    seed=output/'opening'
    config=frozen_live_config(load_config(),root=source)
    event=json.loads(sorted(seed.glob('decision-*.json'))[-1].read_text())
    from zoneinfo import ZoneInfo
    start=datetime.fromisoformat(event['as_of']).astimezone(ZoneInfo(config['timezone_default'])).date()+timedelta(days=1)
    root=output/(label or ('daily' if scenario=='prescribed' else 'imperfect'))
    with patch('scripts.run_adaptive_lifecycle.load_config',return_value=config):
        report=run(root,database,days=days,scenario=scenario,resume=resume,
            seed_plan=None if resume else seed,screen_start=start.isoformat(),stability_window=4,
            checkpoint_days=checkpoint_days,missed_run_offsets=missed_run_offsets)
    print(json.dumps(dict(status=report['status'],completed_days=report['completed_days'],
        failures=report['failures'],live_changed=False)),flush=True)


def review_parity(source,output):
    """Cheap replay of every saved review with current code; no planner search.

    The initial screen preceded extraction of the shared phase-neutral evidence
    gate. Prove identical target/state proposals, rather than rerun searches for
    a behavior-preserving refactor. Original source manifests remain untouched.
    """
    from run_analysis.adaptive.continuous_general_progression import propose
    from run_analysis.adaptive.adaptive_stage_lifecycle import review_entries
    from run_analysis.stage_target_history import StageTarget
    from datetime import date
    root=output/'daily'
    config=frozen_live_config(load_config(),root=source)
    event=json.loads(sorted(root.glob('decision-*.json'))[-1].read_text())
    service=resume_service(root,config,datetime.fromisoformat(event['as_of']))
    h=service.harness
    def target(row):
        return StageTarget(**dict(row,available_at=datetime.fromisoformat(row['available_at']),
            effective_on=date.fromisoformat(row['effective_on']),
            review_anchor=datetime.fromisoformat(row['review_anchor']) if row.get('review_anchor') else None))
    rows=[]
    try:
        all_history=tuple(h.history)
        for previous,event in zip(h.events,h.events[1:]):
            before=previous.get('stage_runtime',{})
            after=event.get('stage_runtime',{})
            if not before.get('general_progression_state') or event['reason']!='continuous_general_review':continue
            at=datetime.fromisoformat(event['as_of'])
            # decision() advances past an already completed date. Reconstruct
            # the causally available history, not today's later completions.
            h.history=[r for r in all_history if r.available_at<=at]
            h.general_progression_state=before['general_progression_state']
            h.target_history=tuple(target(row) for row in before['target_history'])
            longest=max((r.miles for r in h.history if at-timedelta(days=30)<=r.start<at
                and r.available_at<=at and h.capacity_eligibility.get(r.key,True)),default=None)
            result=propose(h,at,review_entries(h,at),longest=longest)
            expected=after['general_progression_state']
            differences={k:(expected.get(k),v) for k,v in result['continuous_state'].items() if expected.get(k)!=v}
            if differences:raise AssertionError(dict(index=event['index'],state_differences=differences))
            assert result['target']==target(after['target_history'][-1])
            assert result['candidate'] is None
            rows.append(dict(index=event['index'],at=event['as_of'],gate=result['gate_status'],
                identical_state_and_target=True))
        report=dict(no_search=True,review_count=len(rows),reviews=rows,live_changed=False)
        (output/'review-parity.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)
    finally:h.sandbox.close()


def regression_summary(output):
    """Resolve named JUnit cases and verify complete collection coverage.

    Keep failed/interrupted reports. A corrected test needs a later passing
    result; unnamed interruption placeholders never count as passing tests.
    """
    import subprocess
    import sys
    import xml.etree.ElementTree as ET
    from hashlib import sha256
    reports=('regression-interrupted.xml','regression.xml',
        'regression-integration.xml','regression-integration-followup.xml')
    latest={};history=[];placeholders=[];digests={}
    for name in reports:
        path=output/name
        digests[name]=sha256(path.read_bytes()).hexdigest()
        for case in ET.parse(path).findall('.//testcase'):
            key=case.get('classname'),case.get('name')
            if not all(key):
                placeholders.append(dict(report=name,attributes=case.attrib));continue
            status=('failed' if case.find('failure') is not None or case.find('error') is not None
                else 'skipped' if case.find('skipped') is not None else 'passed')
            latest[key]=dict(status=status,report=name)
            if status=='failed':history.append(dict(test='::'.join(key),report=name))
    files=sorted({module.replace('.','/')+'.py' for module,_ in latest})
    if not files or not all(Path(path).is_file() for path in files):
        raise ValueError('Named test modules must resolve before claiming coverage')
    collected=subprocess.run([sys.executable,'-m','pytest','--collect-only','-q','-o','addopts=',*files],
        capture_output=True,text=True,check=True)
    expected=set()
    for line in collected.stdout.splitlines():
        if line.startswith('tests/') and '::' in line:
            parts=line.strip().split('::')
            module=parts[0][:-3].replace('/','.')
            expected.add(('.'.join([module,*parts[1:-1]]),parts[-1]))
    missing=['::'.join(k) for k in sorted(expected-set(latest))]
    unresolved=[dict(test='::'.join(k),**latest[k]) for k in sorted(expected)
        if k in latest and latest[k]['status']!='passed']
    report=dict(status='passed' if expected and not missing and not unresolved else 'failed',
        files=files,collected_tests=len(expected),covered_tests=len(expected & set(latest)),
        missing=missing,unresolved=unresolved,historical_failures=history,
        excluded_runner_placeholders=placeholders,source_reports=digests,
        method='Latest named JUnit result; corrected cases need a later pass; complete module collection checked',
        scope='Selected progression/accounting/search/quality/race/app-integration modules, not the entire repository',
        live_changed=False)
    with (output/'regression-summary.json').open('x') as stream:
        json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps({k:report[k] for k in ('status','collected_tests','covered_tests','missing','unresolved')}),flush=True)


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('opening','daily','review-parity','regression-summary'))
    p.add_argument('--source',type=Path,default=Path('output/frequency-weight-3pct-20261018/weight-03'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--database',type=Path,default=Path('data/run_analysis.sqlite'))
    p.add_argument('--days',type=int,default=3)
    p.add_argument('--resume',action='store_true',help='Extend the existing screen without replaying its completed days')
    p.add_argument('--scenario',choices=('prescribed','imperfect'),default='prescribed')
    p.add_argument('--label',help='Separate screen directory within the output root')
    p.add_argument('--checkpoint-days',type=int)
    p.add_argument('--missed-run-offset',type=int,action='append',default=[])
    args=p.parse_args();args.output.mkdir(exist_ok=True)
    if args.label is not None and (not args.label or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in args.label)):
        p.error('Screen label must contain lowercase letters, numbers, hyphens or underscores')
    if args.mode=='opening':opening(args.source,args.output)
    elif args.mode=='daily':daily(args.source,args.output,args.database,args.days,
        resume=args.resume,scenario=args.scenario,label=args.label,
        checkpoint_days=args.checkpoint_days,missed_run_offsets=tuple(args.missed_run_offset))
    elif args.mode=='review-parity':review_parity(args.source,args.output)
    else:regression_summary(args.output)


if __name__=='__main__':main()
