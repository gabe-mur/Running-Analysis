"""Bounded A3/P1 original-data handoff audit; no calendar search or live writes."""
from argparse import ArgumentParser
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

from run_analysis.config import load_config
from run_analysis.session_decision import ProgramDecisionInput, PrescriptionBinding
from scripts.adaptive_candidate_pipeline import load_athlete, compile_athlete
from scripts.audit_work_stimulus import encode
from scripts.execution_forecast_contract import IntendedExecution, IntendedPhase, TargetBand, issue_execution
from scripts.execution_upload_sandbox import measurement_context
from scripts.new_model_daily_harness import planned_accounted
from scripts.research_manifest import freeze_sources
from scripts.continuous_program_adapter import evaluate_continuous
from scripts.shared_program_evaluator import ProgramContext


def audit(database,config,as_of):
    started=perf_counter()
    inputs=load_athlete(database,config,as_of)
    candidate=compile_athlete(inputs,as_of)
    zone=ZoneInfo(config['timezone_default'])
    today=as_of.astimezone(zone).date()
    decision=today+timedelta(days=1) if any(r.start.astimezone(zone).date()==today for r in inputs['history']) else today
    context=ProgramContext((),today,decision,decision+timedelta(days=21),decision+timedelta(days=35),
        candidate['scales'],candidate['training_policy'],candidate['demand_scenario'],candidate['exchange'],
        candidate['timing'],demand_accounting_key=inputs['reference'].accounting_key,
        easy_duration_policy=candidate['easy_duration_policy'])
    rows=[]
    for dose in candidate['catalog']:
        intent=IntendedExecution(dose.key,as_of,as_of+timedelta(days=1),
            tuple(IntendedPhase(k,TargetBand(s,lo,hi,'AS37 declared duration scenario'))
                for (k,s),(lo,hi) in zip(dose.phases,dose.phase_scenarios)),
            TargetBand(dose.miles,max(.1,dose.miles-.2),dose.miles+.2,'provisional distance tolerance'),
            inputs['reference'].accounting_key,measurement_context(config),forecast_scope=dose.forecast_scope)
        issued=issue_execution(intent,inputs['phase_history'])
        handoff=issued.payload['planning_handoff']
        physical=planned_accounted(dose.key,intent.suggested_start,dose,handoff,inputs['reference'],available=as_of) if handoff['usable'] else None
        contract=candidate['coaching_contracts'][dose.key]
        suitability=None
        if physical:
            evaluated=evaluate_continuous(context,inputs['history'],(physical,),reference=inputs['reference'],
                as_of=as_of,timezone=config['timezone_default'],purpose_fraction=.5,
                decision_input=ProgramDecisionInput(candidate['stage'].key,as_of,'real candidate audit',
                    (PrescriptionBinding(physical.key,contract.purpose,contract.intensity_key,contract),)))
            suitability=dict(eligible=evaluated['decision_eligible'],checks=evaluated['suitability'],
                constraints=evaluated['constraints']['status'],cost=evaluated['evaluation'].total_cost)
        rows.append(dict(key=dose.key,role=dose.role,miles=dose.miles,phases=dose.phases,
            forecast=handoff,issued_fingerprint=issued.fingerprint,physical=physical,
            contract=contract,suitability=suitability))
    return dict(as_of=as_of,stage=candidate['stage'],scales=candidate['scales'],
        source_sessions=len(inputs['sessions']),candidate_count=len(rows),
        all_handoffs_usable=all(r['forecast']['usable'] for r in rows),candidates=rows,
        all_single_candidates_eligible=all(r['suitability'] and r['suitability']['eligible'] for r in rows),
        seconds=perf_counter()-started,live_writes=False,calendar_search=False,
        response_allowance=None,physiological_validation=False)


if __name__=='__main__':
    parser=ArgumentParser()
    parser.add_argument('--database',type=Path,default=Path('data/run_analysis.sqlite'))
    parser.add_argument('--config',type=Path,default=Path('config.yaml'))
    parser.add_argument('--as-of',type=datetime.fromisoformat,default=datetime(2026,9,27,tzinfo=timezone.utc))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    config=load_config(args.config)
    manifest=freeze_sources(args.output.with_suffix('.sources.zip'),config=config,
        entrypoint='scripts.audit_adaptive_candidates',inputs=(args.database,))
    result=audit(args.database,config,args.as_of)
    result['manifest']=manifest
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result,stream,default=encode,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k not in {'candidates','scales'}},default=encode))
