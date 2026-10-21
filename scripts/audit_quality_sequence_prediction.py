"""Current direct accuracy and one phase-adjacency ablation; no search or writes to live data."""
from argparse import ArgumentParser
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import sqlite3
from statistics import mean, median
from time import perf_counter

from run_analysis.adaptive.direct_quality_load import forecast_direct, phase_observations, score_direct
from scripts.audit_direct_quality_load import DirectAudit, phase_errors
from scripts.audit_issued_quality_load import audit as issued_audit, matched_assignments
from scripts.audit_work_stimulus import encode
from scripts.research_manifest import freeze_sources
from scripts.validate_quality_load import audit


def stats(rows):
    errors = [r['fraction_error'] for r in rows if r and r['fraction_error'] is not None]
    return dict(n=len(errors),mae=mean(map(abs,errors)) if errors else None,
        bias=mean(errors) if errors else None,median_absolute_error=median(map(abs,errors)) if errors else None,
        max_absolute_error=max(map(abs,errors)) if errors else None)


def summarize(cases):
    supported=[c for c in cases if c['direct_load']['metrics']
               and c['direct_load']['metrics']['fraction_error'] is not None]
    paired=[c for c in supported if c['sequence_trial']['metrics']
            and c['sequence_trial']['metrics']['fraction_error'] is not None]
    result=dict(total=len(cases),baseline_supported=len(supported),paired=len(paired),
        baseline=stats([c['direct_load']['metrics'] for c in paired]),
        candidate=stats([c['sequence_trial']['metrics'] for c in paired]),
        unsupported=[dict(key=c['key'],start=c['start'],status=c['direct_load']['forecast']['status'])
                     for c in cases if c not in supported],phases={})
    for kind in ('warmup','work','recovery','cooldown'):
        both=[(c['direct_load']['detailed_phase_errors'].get(kind),
               c['sequence_trial']['phase_errors'].get(kind)) for c in paired]
        both=[(a,b) for a,b in both if a and b and a['fraction_error'] is not None and b['fraction_error'] is not None]
        result['phases'][kind]=dict(baseline=stats([a for a,b in both]),candidate=stats([b for a,b in both]))
    return result


def issued_summary(cases):
    """Actual issue-time whole errors include execution deviations.

    Keep the different-session assignment and unresolved prescriptions visible.
    Only the paired current-quality rows have resolved physical work alignment;
    this is not a new accuracy filter or evidence of full compliance.
    """
    rows=[]
    for row in cases:
        old=row.get('whole_plan_vs_execution')
        new=row.get('sequence_trial',{}).get('issued',{}).get('whole_plan_vs_execution')
        conditional=row.get('actual_timing_diagnostic',{}).get('whole_load_error')
        new_conditional=row.get('sequence_trial',{}).get('realized_timing',{}).get('metrics')
        rows.append(dict(key=row['key'],role=row['current_actual_role'],contract_status=row['contract']['status'],
            baseline=old,candidate=new,execution_differences=row.get('execution_differences'),
            baseline_realized_timing=conditional,candidate_realized_timing=new_conditional))
    comparable=[r for r in rows if r['role'] in {'intervals','tempo_threshold'} and
        r['baseline'] and r['candidate'] and r['baseline']['relative_error_bounds']
        and r['candidate']['relative_error_bounds']]
    def error_bounds(name):
        ranges=[r[name]['relative_error_bounds'] for r in comparable]
        if not ranges:
            return None
        return (mean(0. if low<=0<=high else min(abs(low),abs(high)) for low,high in ranges),
                mean(max(abs(low),abs(high)) for low,high in ranges))
    return dict(cases=rows,paired_quality=len(comparable),
        baseline_mae_bounds=error_bounds('baseline'),candidate_mae_bounds=error_bounds('candidate'),
        interpretation='Whole prescribed load versus actual whole load; includes actual extent/type deviations, not pure load-model error',
        prospective_acceptance=False)


class Comparison(DirectAudit):
    def __call__(self,case,run,layout,actual,*args):
        prior=tuple(self.history)
        super().__call__(case,run,layout,actual,*args)
        forecast=forecast_direct(layout,prior,as_of=run.start,
            accounting_key=case['fit']['accounting_key'],sequence_context=True)
        observations=phase_observations(layout,actual)
        case['sequence_trial']=dict(forecast=forecast,metrics=score_direct(forecast,actual),
            phase_errors=phase_errors(forecast,observations))
        # Reusable causal inputs, so further analysis need not parse originals
        # or run the unrelated old HR-curve fit again. Only prior history is a
        # predictor input; target observations are scored after construction.
        case['direct_validation_inputs']=dict(history=[asdict(h) for h in prior],
            target=asdict(self.history[-1]),layout=layout,observations=[asdict(o) for o in observations])


def issued_observer(row,run,layout,trace,history,reference):
    if row['issued_at'] is None or row['contract']['layout'] is None:
        return
    result={}
    for name,query in (('issued',row['contract']['layout']),('realized_timing',layout)):
        forecast=forecast_direct(query,history,as_of=row['issued_at'],
            accounting_key=reference.accounting_key,sequence_context=True)
        result[name]=dict(forecast=forecast)
        if name=='realized_timing':
            result[name].update(metrics=score_direct(forecast,trace['normalized_spans']),
                phase_errors=phase_errors(forecast,phase_observations(layout,trace['normalized_spans'])))
        elif forecast['predicted_load'] is not None:
            from scripts.audit_issued_quality_load import whole_load_error
            bounds=trace['load_bounds_within_reference']
            result[name]['whole_plan_vs_execution']=whole_load_error(forecast['predicted_load'],bounds['low'],bounds['high'])
    row['sequence_trial']=result


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument('--baseline-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--summarize-existing',action='store_true',
        help='Recover the final report from already-saved comparisons without parsing or predicting again')
    args=p.parse_args()
    if args.summarize_existing:
        result=json.loads((args.output/'comparison.json').read_text())
        issued=json.loads((args.output/'issued.json').read_text())
        report=dict(summary=result['sequence_summary'],conditional_development_gate_passed=result['development_gate_passed'],
            conditional_seconds=result['elapsed_seconds'],issued=issued_summary(issued['cases']),
            planner_influence=False,report_recovery='Original complete comparison and issued artifacts; final console serialization repaired without rerunning predictions',
            disposition='Conditional improvement only; no automatic deployment or all-family accuracy claim')
        with (args.output/'summary.json').open('x') as stream:
            json.dump(report,stream,indent=2,default=encode,allow_nan=False)
        print(json.dumps(report,indent=2,default=encode),flush=True)
        return
    args.output.mkdir(parents=True,exist_ok=False)
    parent=json.loads((args.baseline_root/'baseline.json').read_text())
    as_of=datetime.fromisoformat(parent['protocol']['cutoff'])
    # Use the archived config, not mutable live settings.
    from zipfile import ZipFile
    import yaml
    with ZipFile(args.baseline_root/'sources.zip') as archive:
        config=yaml.safe_load(archive.read('config.yaml'))
    database=args.baseline_root/'evidence.sqlite'
    protocol=dict(version='phase_adjacency_ablation_v1',
        hypothesis='Same-duration first work after warmup and later work after recovery may have different empirical load rates',
        change='Match immediately preceding physical phase kind when supported; otherwise retain unchanged direct estimate',
        fixed='30 days, phase-duration similarity, preceding-duration similarity, one session vote, nonlinear measured HR integrals',
        forbidden='No target HR/speed, family coefficients, zone mapping, window sweep or selected-case deletion',
        gate='Preserve supported cases; lower overall MAE; no worse MAE in each supported role and known-intent cohort; no worse work-phase MAE',
        scope='Previously inspected development data, conditional timing; actual issued forecasts reported separately; not fatigue accuracy or prospective acceptance',
        planner_influence=False)
    with (args.output/'protocol.json').open('x') as stream:
        json.dump(protocol,stream,indent=2)
    manifest=freeze_sources(args.output/'sources.zip',config=config,entrypoint=__name__,
        inputs=(database,args.baseline_root/'baseline.json',args.output/'protocol.json'))
    started=perf_counter()
    result=audit(database,config,as_of,case_observer=Comparison())
    # An unchanged comparator must actually be identical, not merely similar.
    originals={c['key']:c for c in parent['cases']}
    for case in result['cases']:
        if json.loads(json.dumps(case['direct_load'],default=encode))!=originals[case['key']]['direct_load']:
            raise AssertionError('Unchanged direct comparator differs')
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as db:
        known={str(r[0]) for r in matched_assignments(db,as_of)}
    groups={name:summarize(rows) for name,rows in (
        ('all',result['cases']),('intervals',[c for c in result['cases'] if c['label']=='intervals']),
        ('tempo_threshold',[c for c in result['cases'] if c['label']=='tempo_threshold']),
        ('saved_intent',[c for c in result['cases'] if c['key'] in known]),
        ('2026',[c for c in result['cases'] if c['start'].year==2026]))}
    supported=[g for g in groups.values() if g['paired']]
    gate=bool(supported) and all(g['paired']==g['baseline_supported'] and g['candidate']['mae']<=g['baseline']['mae']
        and (not g['phases']['work']['baseline']['n'] or
             g['phases']['work']['candidate']['mae']<=g['phases']['work']['baseline']['mae']) for g in supported)
    gate=gate and groups['all']['candidate']['mae']<groups['all']['baseline']['mae']
    result.update(sequence_summary=groups,protocol=protocol,manifest=manifest,development_gate_passed=gate,
        comparator_exact=True,planner_influence=False)
    with (args.output/'comparison.json').open('x') as stream:
        json.dump(result,stream,default=encode,indent=2,allow_nan=False)
    issued=issued_audit(database,config,as_of=as_of,execution_observer=issued_observer)
    with (args.output/'issued.json').open('x') as stream:
        json.dump(issued,stream,default=encode,indent=2,allow_nan=False)
    report=dict(summary=groups,conditional_development_gate_passed=gate,seconds=perf_counter()-started,
        issued=issued_summary(issued['cases']),planner_influence=False,
        disposition='Conditional improvement only; no automatic deployment or all-family accuracy claim')
    with (args.output/'summary.json').open('x') as stream:
        json.dump(report,stream,indent=2,default=encode,allow_nan=False)
    print(json.dumps(report,indent=2,default=encode),flush=True)


if __name__=='__main__':
    main()
