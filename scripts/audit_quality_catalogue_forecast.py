"""Catalogue3 frozen mechanical screen, not prospective accuracy or calendar acceptance."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from hashlib import sha256
import json
from math import fsum
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import yaml

from run_analysis.adaptive.quality_block_contract import adapt_v2, read_quality_contract
from run_analysis.adaptive.quality_catalogue_forecast import forecast_prescription, prescription_query
from run_analysis.adaptive.quality_prescription_contract import read_prescription
from run_analysis.adaptive.quality_recipe_catalogue import DoseWindow, progression_options
from run_analysis.adaptive.unsupported_effort_transfer import forecast_extrapolated_work
from run_analysis.workload_accounting import Quantity
from scripts.audit_program_evidence_adapter import load_history
from scripts.audit_unsupported_effort_transfer import assumption, command
from scripts.audit_work_exposure_matching import Inventory, saved_queries
from scripts.audit_work_stimulus import encode
from scripts.program_evidence_adapter import QualityReference
from scripts.research_manifest import freeze_sources


def basis(f):
    return ('unresolved' if f['status']!='predicted' else 'extrapolated'
        if f.get('effort_extrapolation') else 'observed_band_supported')


def compact(f):
    return dict(status=f['status'], basis=basis(f), load=f['predicted_load'],
        work_load=fsum(p['load'] for p in f['phases'] if p['kind']=='work')
            if f['status']=='predicted' else None,
        phase_count=len(f['phases']), source_keys=f['source_keys'],
        unresolved=f['work_exposure_matching'].get('unresolved_phases'),
        extrapolation_status=f.get('effort_extrapolation',{}).get('status'))


def audit(args):
    args.output.mkdir(parents=True,exist_ok=False)
    paths = (args.baseline_root/'evidence.sqlite', args.baseline_root/'baseline.json',
        args.baseline_root/'sources.zip', args.issued_report, args.catalogue, args.menu)
    hashes = {str(p):sha256(p.read_bytes()).hexdigest() for p in paths}
    parents=json.loads(paths[1].read_text());issued=json.loads(args.issued_report.read_text())
    catalogue=json.loads(args.catalogue.read_text());menu=json.loads(args.menu.read_text())
    with ZipFile(paths[2]) as z:
        config=yaml.safe_load(z.read('config.yaml'))
    at=datetime.fromisoformat(parents['protocol']['cutoff'])
    protocol=dict(version='quality_catalogue_forecast_audit_v1', cutoff=at,
        partial_opening_keys=['36'],
        fixed=['Frozen measured HR accounting, 30-day causal sources, work-band matching and source votes',
            'Native compiler duration; per-block cues, no raw-speed model or recipe multipliers',
            'Latest already-frozen causal response time and prior-fit extrema only for absent higher-band support'],
        limits=['Engineering pace/window inputs from Catalogue1/2, not athlete-specific prescriptions or admission',
            'Three execution scenarios are not confidence bounds; HR lag extrapolation remains uncalibrated',
            'Mechanical checks and exact baseline parity, not independent accuracy, quality purpose allocation or calendar stability',
            'No live DB, search, parameter changes or deployment'], planner_influence=False)
    with (args.output/'protocol.json').open('x') as s:
        json.dump(protocol,s,indent=2,default=encode)
    manifest=freeze_sources(args.output/'sources.zip',config=config,entrypoint=__name__,
        inputs=(*paths,args.output/'protocol.json'))
    started=perf_counter()
    inventory=Inventory(config,{c['key']:c for c in parents['cases']},
        {c['key']:c for c in issued['cases']},{'36'})
    reference=QualityReference('threshold','work_seconds',Quantity.point(900,'loading only'),
        Quantity(290,310),'original frozen source loading only')
    _,missing=load_history(paths[0],config,at,reference,
        since=min(datetime.fromisoformat(c['start']) for c in parents['cases'])-timedelta(days=1),
        run_transform=inventory)
    loading_seconds=perf_counter()-started
    response=assumption(parents['cases'],at)
    anchors=tuple(command(z,config) for z in ('Z2','Z3','Z4'))
    kwargs=dict(reference=inventory.reference,anchors=anchors,response=response)
    previous=[read_prescription(r['prescription']) for r in menu['menu']['prescriptions']]
    candidates=[dict(recipe=c['recipe'],repetitions=c['repetitions'],
        prescription=read_quality_contract(c['prescription']['numeric_contract'])) for c in catalogue['candidates']]
    base=adapt_v2(next(p for p in previous if p.request.family=='progression'))
    window=DoseWindow(1,3600,10,7200,20,'structural gradual envelope; not athlete admission')
    candidates.extend(dict(recipe='progression',repetitions=1,prescription=c.prescription)
        for c in progression_options(base,catalogue['progression_targets'],window=window))
    results=[];forecasts={};groups=defaultdict(list);problems=[]
    for c in candidates:
        p=c['prescription']
        if p.request.issued_at!=at:
            raise ValueError('Frozen command and frozen evidence origin must agree')
        f=forecast_prescription(p,inventory.history,inventory.profiles,config,**kwargs)
        forecasts[p.key]=f
        target=p.miles if c['recipe']=='progression' else p.request.blocks[1].primary.amount.estimate
        row=dict(key=p.key,recipe=c['recipe'],repetitions=c['repetitions'],target=target,
            primary_unit=p.request.blocks[1].primary.unit,miles=p.miles,
            work_seconds=fsum(q.seconds.estimate for q in p.phases if q.kind=='work'),
            total_seconds=fsum(q.seconds.estimate for q in p.phases),
            scenarios={k:compact(v) for k,v in f['scenarios'].items()},
            scenario_envelope=f['scenario_envelope'],all_scenarios_predicted=f['all_scenarios_predicted'])
        results.append(row)
        groups[('repetitions',c['recipe'],target)].append(row)
        groups[('target',c['recipe'],c['repetitions'])].append(row)
        if not f['all_scenarios_predicted']:
            problems.append(dict(problem='unresolved_candidate',key=p.key,
                scenarios=row['scenarios']))
    compared=Counter()
    for (axis,recipe,fixed),group in groups.items():
        ordered=sorted(group,key=lambda r:r[axis])
        for a,b in zip(ordered,ordered[1:]):
            for point in ('lower_speed','nominal','higher_speed'):
                for metric in ('load','work_load'):
                    av,bv=(r['scenarios'][point][metric] for r in (a,b))
                    if av is None or bv is None:
                        continue
                    compared[axis]+=1
                    if bv<av-1e-9:
                        problems.append(dict(problem='more_work_cheaper',axis=axis,recipe=recipe,
                            fixed=fixed,point=point,metric=metric,before=a,after=b,
                            before_forecast=forecasts[a['key']]['scenarios'][point],
                            after_forecast=forecasts[b['key']]['scenarios'][point]))
    # Same clock, only authored effort changes; no pacing or structure confound.
    harder=[]
    for recipe in ('threshold_continuous','threshold_repeats','aerobic_timed','pickups_distance','hill_repeats','progression'):
        c=next(c for c in candidates if c['recipe']==recipe)
        layout,_=prescription_query(c['prescription'],config)
        local=[]
        for cue in ('upper Z3','low Z4','Z4','upper Z4','low Z5','Z5'):
            requested=tuple(command(cue,config) if k=='work' else None for k,_,_ in layout)
            f=forecast_extrapolated_work(layout,inventory.history,inventory.profiles,requested,
                as_of=at,accounting_key=inventory.reference.accounting_key,**kwargs)
            row=dict(recipe=recipe,cue=cue,forecast=f,summary=compact(f))
            local.append(row);harder.append(row)
            if f['status']!='predicted':
                problems.append(dict(problem='unresolved_intensity_probe',recipe=recipe,cue=cue,forecast=f))
        for a,b in zip(local,local[1:]):
            if a['summary']['load'] is not None and b['summary']['load'] is not None:
                compared['harder']+=1
                if b['summary']['load']<a['summary']['load']-1e-9:
                    problems.append(dict(problem='harder_cheaper',recipe=recipe,before=a,after=b))
    parity=[]
    for p,r in zip(previous,menu['menu']['prescriptions']):
        f=forecast_prescription(p,inventory.history,inventory.profiles,config,**kwargs)
        parity.append(dict(key=p.key,exact=json.loads(json.dumps(f['nominal'],default=encode))==r['candidate']))
    known=[]
    for origin in inventory.origins:
        key=origin['name'].removeprefix('before_completed_')
        case=next(c for c in inventory.cases if c['key']==key)
        if not case['intent_known']:
            continue
        layout=tuple((p['kind'],p['start'],p['end']) for p in case['candidate']['phases'])
        f=forecast_extrapolated_work(layout,origin['history'],inventory.profiles,
            saved_queries(inventory.issued[key]['payload'],layout,config),anchors=(),response=None,reference=None,
            as_of=origin['as_of'],accounting_key=inventory.reference.accounting_key)
        known.append(dict(key=key,exact=f==case['candidate']))
    parity_ok=all(r['exact'] for r in (*parity,*known))
    if not parity_ok:
        problems.append(dict(problem='baseline_parity_failed',menu=parity,known=known))
    if hashes!={str(p):sha256(p.read_bytes()).hexdigest() for p in paths}:
        raise AssertionError('Frozen input changed')
    result=dict(protocol=protocol,manifest=manifest,candidates=results,harder_probes=harder,
        problems=problems,problem_counts=dict(Counter(p['problem'] for p in problems)),
        comparisons=dict(compared),menu_exact=parity,known_intent_exact=known,
        scenario_basis_counts=dict(Counter(s['basis'] for r in results for s in r['scenarios'].values())),
        gate=not problems,response=response,source_missing=missing,input_hashes=hashes,
        loading_seconds=loading_seconds,elapsed_seconds=perf_counter()-started,
        planner_influence=False,prospective_accuracy_validated=False)
    with (args.output/'report.json').open('x') as s:
        json.dump(result,s,indent=2,default=encode,allow_nan=False)
    print(json.dumps(dict(candidate_count=len(results),scenario_count=3*len(results),
        gate=result['gate'],problem_counts=result['problem_counts'],comparisons=result['comparisons'],
        scenario_basis_counts=result['scenario_basis_counts'],menu_exact=sum(r['exact'] for r in parity),
        known_exact=sum(r['exact'] for r in known),loading_seconds=loading_seconds,
        seconds=result['elapsed_seconds']),indent=2),flush=True)


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument('--baseline-root',type=Path,required=True)
    p.add_argument('--issued-report',type=Path,required=True)
    p.add_argument('--catalogue',type=Path,required=True)
    p.add_argument('--menu',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    audit(p.parse_args())


if __name__=='__main__':
    main()
