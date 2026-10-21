"""Frozen upward-extrapolation check, not accuracy validation on actual Z5 runs."""
from argparse import ArgumentParser
from datetime import datetime, timedelta
import json
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import yaml

from run_analysis.adaptive.prescribed_hr_target import prescribed_hr_target
from run_analysis.adaptive.unsupported_effort_transfer import ResponseAssumption, forecast_extrapolated_work
from run_analysis.adaptive.work_exposure_matching import WorkExposure, forecast_work_for_ranking
from run_analysis.workload_accounting import Quantity
from scripts.audit_program_evidence_adapter import load_history
from scripts.audit_work_exposure_matching import Inventory, menu, saved_queries, summary
from scripts.audit_work_stimulus import encode
from scripts.program_evidence_adapter import QualityReference
from scripts.research_manifest import freeze_sources


def assumption(parents, as_of):
    fits = [p['fit'] for p in parents if p.get('fit',{}).get('dynamic')
        and as_of-timedelta(days=30) <= datetime.fromisoformat(p['fit']['issued_at']) <= as_of]
    if not fits:
        raise ValueError('No frozen causal response time for this diagnostic origin')
    selected = max(fits,key=lambda f:f['issued_at'])
    return ResponseAssumption(selected['dynamic']['tau_seconds'],
        datetime.fromisoformat(selected['issued_at']),tuple(selected['source_keys']),
        'Latest already-frozen prior-only dynamic fit; no refitting for Z5; prior-fit extrema are sensitivities',
        tuple(sorted({f['dynamic']['tau_seconds'] for f in fits})))


def command(cue, config):
    p = prescribed_hr_target((cue,),config)
    return WorkExposure((p.low,p.high),(p.low,p.high),'declared counterfactual '+cue)


def audit(inventory, parents, as_of):
    anchors = tuple(command(z,inventory.config) for z in ('Z2','Z3','Z4'))
    origins = [inventory.origins[-1],dict(name='current_frozen_cutoff',as_of=as_of,history=tuple(inventory.history))]
    rows = []; problems = []
    for origin in origins:
        response = assumption(parents,origin['as_of'])
        for seconds in (15,30,45,60,90,120,180,300,600,900,1200,1800):
            layout = (('warmup',0.,600.),('work',600.,600.+seconds),
                ('cooldown',600.+seconds,1200.+seconds))
            local = []
            for cue in ('Z4','low Z5','Z5','upper Z5'):
                requested = (None,command(cue,inventory.config),None)
                f = forecast_extrapolated_work(layout,origin['history'],inventory.profiles,requested,
                    anchors=anchors,response=response,reference=inventory.reference,
                    as_of=origin['as_of'],accounting_key=inventory.reference.accounting_key)
                detail = f.get('effort_extrapolation')
                baseline = forecast_work_for_ranking(layout,origin['history'],inventory.profiles,requested,
                    as_of=origin['as_of'],accounting_key=inventory.reference.accounting_key)
                row = dict(origin=origin['name'],seconds=seconds,cue=cue,status=f['status'],
                    predicted_load=f['predicted_load'],supported_exact=f==baseline if cue=='Z4' else None,
                    extrapolation=detail,response=response,
                    work_load=f['phases'][1]['load'],cooldown_load=f['phases'][2]['load'])
                rows.append(row);local.append(row)
                if f['status']!='predicted':
                    problems.append(dict(origin=origin['name'],seconds=seconds,cue=cue,
                        problem=detail['status'] if detail else f['status']))
                if cue=='Z4' and f!=baseline:
                    problems.append(dict(problem='supported_forecast_changed',origin=origin['name'],seconds=seconds))
            for a,b in zip(local,local[1:]):
                if a['predicted_load'] is not None and b['predicted_load'] is not None and b['predicted_load']<a['predicted_load']-1e-10:
                    problems.append(dict(problem='harder_cheaper',origin=origin['name'],seconds=seconds,from_cue=a['cue'],to_cue=b['cue']))
        for cue in ('Z4','low Z5','Z5','upper Z5'):
            chosen = [r for r in rows if r['origin']==origin['name'] and r['cue']==cue]
            for a,b in zip(chosen,chosen[1:]):
                if a['predicted_load'] is not None and b['predicted_load'] is not None and b['predicted_load']<a['predicted_load']-1e-10:
                    problems.append(dict(problem='longer_cheaper',origin=origin['name'],cue=cue,from_seconds=a['seconds'],to_seconds=b['seconds']))
    return dict(rows=rows,problems=problems,mechanics_gate_passed=not problems,
        requested_forecasts=len(rows),extrapolated=sum(bool(r['extrapolation']) for r in rows),
        supported=sum(r['supported_exact'] is True for r in rows))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root',type=Path,required=True)
    parser.add_argument('--issued-report',type=Path,required=True)
    parser.add_argument('--partial-opening-key',action='append',default=[])
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    parents = json.loads((args.baseline_root/'baseline.json').read_text())
    issued = json.loads(args.issued_report.read_text())
    with ZipFile(args.baseline_root/'sources.zip') as z:
        config = yaml.safe_load(z.read('config.yaml'))
    as_of = datetime.fromisoformat(parents['protocol']['cutoff'])
    if not set(args.partial_opening_key)<={c['key'] for c in parents['cases']}:
        raise ValueError('Partial-opening key must resolve to frozen inventory')
    protocol = dict(version='unsupported_effort_transfer_audit_v1',
        hypothesis='Preserve supported forecasts exactly; upward absent-overlap queries use closest supported lower authored band and response-load contrast',
        fixed=['Existing phase costs, 30-day availability and source votes; no fitted new weights or expanded history',
            'Midpoint work cues are hypothetical equilibrium inputs; nonwork cost-equivalent HR is a reference proxy',
            'Same response tau/initial state/timing on both sides; carry state through work and nonwork',
            'Latest prior frozen response fit and prior-fit extrema; no fits to target Z5 runs',
            'Sensitivity across target-band endpoints and prior tau extrema, not prediction/confidence intervals',
            'Fail rather than clip or clamp an invalid ratio or accounting-bound violation'],
        acceptance='Supported outputs exact; numeric higher/longer checks; recording completeness preserved; failure statuses retained',
        limits=['Previously inspected development evidence; no actual independently prescribed Z5 accuracy cohort',
            'Same-on/off time constant, command-as-equilibrium and proportional cost transfer are assumptions',
            'Supported whole prescription unchanged; extrapolated work changes downstream nonwork cost through modeled carryover',
            'No calendar search, recovery-model change, database mutation or deployment'],
        partial_opening_keys=args.partial_opening_key,planner_influence=False)
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    database = args.baseline_root/'evidence.sqlite'
    manifest = freeze_sources(args.output/'sources.zip',config=config,entrypoint=__name__,
        inputs=(database,args.baseline_root/'baseline.json',args.issued_report,args.output/'protocol.json'))
    started = perf_counter()
    inventory = Inventory(config,{c['key']:c for c in parents['cases']},
        {c['key']:c for c in issued['cases']},set(args.partial_opening_key))
    reference = QualityReference('threshold','work_seconds',Quantity.point(900,'unused loading reference'),
        Quantity(290,310),'source loading only')
    history,missing = load_history(database,config,as_of,reference,
        since=min(datetime.fromisoformat(c['start']) for c in parents['cases'])-timedelta(days=1),run_transform=inventory)
    probe = audit(inventory,parents['cases'],as_of)
    unchanged = []
    for origin in inventory.origins:
        key = origin['name'].removeprefix('before_completed_')
        case = next(c for c in inventory.cases if c['key']==key)
        if not case['intent_known']:
            continue
        requested = saved_queries(inventory.issued[key]['payload'],
            tuple((p['kind'],p['start'],p['end']) for p in case['candidate']['phases']),config)
        f = forecast_extrapolated_work(tuple((p['kind'],p['start'],p['end']) for p in case['candidate']['phases']),
            origin['history'],inventory.profiles,requested,anchors=(),response=None,reference=None,
            as_of=origin['as_of'],accounting_key=inventory.reference.accounting_key)
        unchanged.append(dict(key=key,exact=f==case['candidate']))
    recipes = menu(inventory,history,as_of)
    recipe_checks = []; structure_probes = []
    for r in recipes['prescriptions']:
        f = forecast_extrapolated_work(tuple((p['kind'],p['start'],p['end']) for p in r['candidate']['phases']),
            inventory.history,inventory.profiles,
            tuple(p['work_exposure']['command'] if p['kind']=='work' else None for p in r['candidate']['phases']),
            anchors=(),response=None,reference=None,as_of=as_of,accounting_key=inventory.reference.accounting_key)
        recipe_checks.append(dict(family=r['family'],variant=r['variant'],exact=f==r['candidate']))
        if r['family'] in {'short_intervals','long_intervals','hill_repeats','fartlek'} and r['variant'].endswith('ordinary'):
            requested = tuple(command('Z5',config) if p['kind']=='work' else None for p in r['candidate']['phases'])
            x = forecast_extrapolated_work(tuple((p['kind'],p['start'],p['end']) for p in r['candidate']['phases']),
                inventory.history,inventory.profiles,requested,
                anchors=tuple(command(z,config) for z in ('Z2','Z3','Z4')),
                response=assumption(parents['cases'],as_of),reference=inventory.reference,
                as_of=as_of,accounting_key=inventory.reference.accounting_key)
            structure_probes.append(dict(family=r['family'],variant=r['variant'],forecast=x,
                baseline_work_cues_replaced_for_diagnostic_only=True))
    result = dict(protocol=protocol,manifest=manifest,probe=probe,known_intent_exact=unchanged,
        menu_exact=recipe_checks,native_structure_probes=structure_probes,
        known_intent_errors_unchanged=summary(inventory.cases),source_missing=missing,
        numerical_gate_passed=probe['mechanics_gate_passed'] and all(r['exact'] for r in (*unchanged,*recipe_checks))
            and all(r['forecast']['status']=='predicted' for r in structure_probes),
        elapsed_seconds=perf_counter()-started,planner_influence=False,
        actual_z5_accuracy_validated=False)
    with (args.output/'report.json').open('x') as s:
        json.dump(result,s,indent=2,default=encode,allow_nan=False)
    print(json.dumps(dict(numerical_gate=result['numerical_gate_passed'],problems=probe['problems'],
        probe_count=probe['requested_forecasts'],extrapolated=probe['extrapolated'],supported_exact=probe['supported'],
        known_intent_exact=unchanged,menu_count=len(recipe_checks),
        native_structure_statuses=[dict(family=r['family'],variant=r['variant'],status=r['forecast']['status']) for r in structure_probes],
        seconds=result['elapsed_seconds']),indent=2),flush=True)


if __name__=='__main__':
    main()
