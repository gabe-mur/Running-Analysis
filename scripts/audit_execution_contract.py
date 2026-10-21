"""Persist/replay new offline contract on recent originals; never add old margins."""
from argparse import ArgumentParser
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import json

from run_analysis.config import load_config
from run_analysis.workload_accounting import Quantity
from scripts.audit_issued_quality_load import audit
from scripts.audit_work_stimulus import encode
from scripts.direct_quality_load import checked_layout
from scripts.execution_forecast_contract import (TargetBand,IntendedPhase,IntendedExecution,CompletedExecution,
    issue_execution,save_execution,read_execution,compare_execution)


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    config=load_config(Path('config.yaml'))
    context=json.dumps({k:config[k] for k in ('resting_hr','max_hr')},sort_keys=True)
    comparisons=[]
    def replay(row,run,observed_layout,trace,history,reference):
        contract=row['contract']
        intent=IntendedExecution(row['key'],row['issued_at'],datetime.fromisoformat(row['planned_for']),
            tuple(IntendedPhase(k,TargetBand(b-a,None,None,'saved target only; no historical adherence margin specified'))
                for k,a,b in contract['layout']),None,reference.accounting_key,context,None)
        frozen=issue_execution(intent,history)
        path=args.output_dir/f"issued-{row['key']}.json"
        save_execution(frozen,path)
        restored=read_execution(path)
        if restored != frozen:
            raise AssertionError('Persistence changed issued record')
        if frozen.payload['target_forecast']['predicted_load'] != row['forecast']['predicted_load']:
            raise AssertionError('Contract changed target forecast')
        bounds=trace['load_bounds_within_reference']
        observed=CompletedExecution(run.key,row['key'],run.start,run.end,row['actual']['available_at'],
            tuple((k,b-a) for k,a,b in checked_layout(observed_layout)) if observed_layout else None,
            run.miles,Quantity(bounds['low'],bounds['high'],trace['total_load'],'original continuous HR accounting'),
            reference.accounting_key,context,phase_evidence='inferred')
        before=asdict(observed)
        result=compare_execution(restored,observed,as_of=observed.available_at)
        if asdict(observed)!=before or result['actual_cardio']!=before['cardio']:
            raise AssertionError('Measured activity was changed')
        comparisons.append(dict(key=row['key'],saved_plan_id=row['plan_id'],comparison=result,
            legacy_distance_range=row['payload'].get('distance_range_miles'),
            note='Legacy target/margins not invented; plan distance range retained here, no new one-decimal target retrofitted',
            issued_record=str(path),fingerprint=frozen.fingerprint))
    result=audit(Path('data/run_analysis.sqlite'),config,execution_observer=replay)
    report=dict(version='execution_contract_original_replay_v1',cases=comparisons,
        inventory=[dict(key=c['key'],status=c.get('forecast_status'),contract=c['contract']['status']) for c in result['cases']],
        database_unchanged=result['database_unchanged'],planner_influence=False,
        policy='No historical duration margins, canonical distance target or load-error allowance fabricated; target and recorded load preserved')
    with (args.output_dir/'report.json').open('x') as stream:
        json.dump(report,stream,default=encode,indent=2,allow_nan=False)
    print(json.dumps(report,default=encode,indent=2))


if __name__=='__main__':
    main()
