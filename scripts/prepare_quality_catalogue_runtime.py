"""Freeze a new offline catalogue stage; no live access or calendar search by default."""
from argparse import ArgumentParser
from dataclasses import asdict, replace
from datetime import datetime, timedelta
import json
from math import fsum
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import yaml

from run_analysis.adaptive.quality_catalogue_admission import CatalogueAdmission
from run_analysis.adaptive.quality_catalogue_runtime import (
    catalogue_doses, delivery_from, exposure_pace, json_value,retained_progression)
from run_analysis.adaptive.quality_dose_policy import QualityDoseStage,purpose_contract
from run_analysis.adaptive.quality_forecast_runtime import VERSION
from run_analysis.adaptive.quality_catalogue_forecast import forecast_prescription_point
from run_analysis.adaptive.quality_prescription_contract import causal_paces
from run_analysis.adaptive.quality_purpose_contract import assess_quality_purposes
from run_analysis.adaptive.quality_purpose_forecast import forecast_purposes
from run_analysis.adaptive.quality_block_contract import read_quality_contract
from run_analysis.adaptive.new_model_daily_harness import DailyHarness
from run_analysis.adaptive.review_seed import completed_seed,phase_seed
from run_analysis.workload_accounting import Quantity
from scripts.audit_program_evidence_adapter import load_history
from scripts.audit_work_exposure_matching import Inventory
from scripts.audit_unsupported_effort_transfer import assumption,command
from scripts.program_evidence_adapter import QualityReference
from scripts.run_adaptive_lifecycle import fork_seed_plan
from scripts.research_manifest import freeze_sources


def prepare(args):
    args.output.mkdir(exist_ok=False)
    started=perf_counter()
    baseline=args.baseline
    parents=json.loads((baseline/'baseline.json').read_text())
    issued=json.loads(args.issued.read_text())
    with ZipFile(baseline/'sources.zip') as z:config=yaml.safe_load(z.read('config.yaml'))
    at=datetime.fromisoformat(parents['protocol']['cutoff'])
    inventory=Inventory(config,{c['key']:c for c in parents['cases']},{c['key']:c for c in issued['cases']},{'36'})
    reference=QualityReference('threshold','work_seconds',Quantity.point(900,'source loading only'),Quantity(290,310),'source loading only')
    physical,missing=load_history(baseline/'evidence.sqlite',config,at,reference,
        since=min(datetime.fromisoformat(c['start']) for c in parents['cases'])-timedelta(days=1),run_transform=inventory)
    if missing:raise ValueError('Frozen sources missing: '+str(missing))
    copied=args.output/'comparator'
    last=fork_seed_plan(args.seed,copied)
    h=DailyHarness.resume(copied,config,as_of=last)
    try:
        anchor=h.catalog['quality:ordinary'];coaching=h.coaching_contracts[anchor.key]
        stage=QualityDoseStage(h.stage.key,at,h.stage.ordinary_miles,
            fsum(s for k,s in anchor.phases if k=='work')/60,7/h.training_policy.quality_per_7,
            'Existing independently accepted stage; quality growth unchanged')
        admission=CatalogueAdmission(stage,anchor.miles,fsum(s for _,s in anchor.phases),
            max(1,int(stage.quality_work_minutes*2)),
            'Accepted complete-session distance/time and work envelope; recipe grid bounds repetition count')
        eligible={r.key for r in physical}
        easy=causal_paces(physical,as_of=at,eligible_keys=eligible)['easy']
        targets={'threshold':command('upper Z3 / low Z4',config),
            'aerobic':command('upper Z4',config),'fast':command('Z4',config)}
        paces={name:exposure_pace(physical,inventory.history,inventory.profiles,target,as_of=at)
            for name,target in targets.items()}
        if easy is None or any(p is None for p in paces.values()):
            raise ValueError('No causal extent scenario for a requested purpose; explicit prior required')
        doses,contracts,generation=catalogue_doses(admission,issued_at=at,anchor=anchor,coaching=coaching,
            easy_pace=easy,threshold_pace=paces['threshold'],aerobic_pace=paces['aerobic'],fast_pace=paces['fast'],
            open_recovery_seconds=90)
        progression,progression_contracts=retained_progression(h,issued_at=at)
        doses=(*doses,*progression);contracts={**contracts,**progression_contracts}
        purposes=purpose_contract(stage,config=config,issued_at=at,easy_pace=easy,faster_pace=paces['fast'])
        observed=[]
        required={r.key for r in h.history if r.role=='quality'}
        for key in sorted(required):
            signal=inventory.signals.get(key)
            if signal is None:raise ValueError('Unaligned seed purpose evidence: '+key)
            observed.append(delivery_from(assess_quality_purposes(signal,contract=purposes,role='quality',as_of=at),key))
        response=assumption(parents['cases'],at)
        anchors=tuple(command(z,config) for z in ('Z2','Z3','Z4'))
        evidence=json_value(dict(version=VERSION,available_at=at,
            config={k:config[k] for k in ('resting_hr','max_hr','zones')},profiles=inventory.profiles,
            reference=inventory.reference,response=response,anchors=anchors))
        rows=[]
        for dose in doses:
            p=read_quality_contract(dose.workout_definition['numeric_contract'])
            forecast=forecast_prescription_point(p,inventory.history,inventory.profiles,config,
                reference=inventory.reference,anchors=anchors,response=response)
            credit=forecast_purposes(p,forecast,inventory.history,inventory.profiles,config,contract=purposes)
            rows.append(dict(key=dose.key,recipe=dose.workout_definition['recipe_key'],miles=dose.miles,
                work_minutes=fsum(s for k,s in dose.phases if k=='work')/60,forecast=forecast,
                purposes=credit,definition=dose.workout_definition))
        runtime=json_value(dict(version=VERSION,evidence=evidence,purpose_contract=purposes,
            observed_purposes=observed,admission=admission,
            limits=['Uncalibrated causal whole-bout extent proxies, not adherence or physiological thresholds',
                'No active race-specific block or uphill extent prior supplied',
                'Stage-owned physical ceilings are distinct from purpose-normalization targets']))
        completed=completed_seed(h,at)
        report=json_value(dict(as_of=at,stage=stage,admission=admission,pace_references=dict(easy=easy,**paces),
            purposes=purposes,rows=rows,rejected=generation['rejected'],empty_recipes=generation['empty_recipes'],
            runtime=runtime,seed_review_entries=completed,
            seconds=perf_counter()-started,stability_tested=False,live_changed=False))
        report['manifest']=freeze_sources(args.output/'sources.zip',config=config,entrypoint=__name__,
            inputs=(baseline/'baseline.json',args.issued,copied/'scenario.json',*sorted(copied.glob('decision-*.json'))))
        (args.output/'admission-report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        (args.output/'runtime.json').write_text(json.dumps(runtime,indent=2,allow_nan=False))
        # New decisions require a new experiment identity. Retain the old exact
        # comparator bytes; do not relabel its plan as new-model stability.
        original={r.key:r for r in inventory.history}
        phases=tuple(original.get(r.key,r) for r in phase_seed(h,at))
        new=DailyHarness(args.output/'prototype',config,origin=at,
            # Archived generic anchors still own stage-review/support contracts.
            # The explicit runtime filters them from future selection; deleting
            # them would break growth review, not remove legacy scoring.
            catalog=(*h.catalog.values(),*doses),
            history=h.history,phase_history=phases,scales=h.scales,training_policy=h.training_policy,
            search_policy=h.policy,demand_scenario=h.demand_scenario,exchange=h.exchange,timing=h.timing,
            coaching_contracts={**h.coaching_contracts,**contracts},
            easy_duration_policy=h.easy_duration_policy,decision_source='Explicit offline catalogue stage '+VERSION,
            restrictions=h.restrictions,current_context=h.current_context,responses=h.responses,
            stage=h.stage,target_history=h.target_history,capacity_eligibility=h.capacity_eligibility,
            frequency_contract=h.frequency_contract,seed_review_entries=completed,quality_runtime=runtime)
        new.planner_preferences=h.planner_preferences
        try:
            if args.search:new.replan(at,reason='offline_catalogue_opening')
        finally:new.sandbox.close()
        print(json.dumps(dict(options=len(rows),recipes=sorted({r['recipe'] for r in rows}),
            max_credit={purpose:max(r['purposes']['credits'][purpose].high for r in rows)
                for purpose in ('sustained','faster')},seconds=perf_counter()-started,
            search=args.search,live_changed=False),indent=2),flush=True)
    finally:h.sandbox.close()


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--issued',type=Path,required=True)
    p.add_argument('--search',action='store_true')
    prepare(p.parse_args())


if __name__=='__main__':main()
