"""P3/O1 fixed matched-volume controls. Not a new scheduler or tuned policy."""
from argparse import ArgumentParser
from dataclasses import replace
from datetime import datetime,timedelta,timezone,time
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from run_analysis.config import load_config
from run_analysis.session_decision import ProgramDecisionInput,PrescriptionBinding
from scripts.adaptive_candidate_pipeline import load_athlete,compile_athlete,resize_easy
from scripts.execution_forecast_contract import IntendedExecution,IntendedPhase,TargetBand,issue_execution
from scripts.execution_upload_sandbox import measurement_context
from scripts.new_model_daily_harness import planned_accounted,training_metrics
from scripts.continuous_program_adapter import evaluate_continuous
from scripts.shared_program_evaluator import ProgramContext
from scripts.audit_work_stimulus import encode
from scripts.research_manifest import freeze_sources


def audit(database,config,as_of):
    inputs=load_athlete(database,config,as_of)
    build=compile_athlete(inputs,as_of)
    zone=ZoneInfo(config['timezone_default'])
    today=as_of.astimezone(zone).date()
    decision=today+timedelta(days=1) if any(r.start.astimezone(zone).date()==today for r in inputs['history']) else today
    catalog={d.key:d for d in build['catalog']}
    contracts=build['coaching_contracts']
    easy=next(d for d in catalog.values() if d.role=='easy')
    for miles in (1.4,4.2):
        dose=resize_easy(easy,miles,contracts[easy.key])
        catalog[dose.key]=dose
        contracts[dose.key]=contracts[easy.key]
    long=max((d for d in catalog.values() if d.role=='long'),key=lambda d:d.miles).key
    # Explicit real-athlete fixed contrasts. Wider/generalized stages belong V5.
    cases={
        'two_runs_underfunded':((0,long),(4,'quality:ordinary')),
        'three_runs_concentrated':((0,long),(2,'easy:4.2'),(4,'quality:ordinary')),
        'four_runs_equal':((0,long),(2,'easy:2.1'),(4,'quality:ordinary'),(6,'easy:2.1')),
        'four_runs_unequal':((0,long),(2,'easy:1.1'),(4,'quality:ordinary'),(6,'easy:3.1')),
        'four_runs_unequal_reversed':((0,long),(2,'easy:3.1'),(4,'quality:ordinary'),(6,'easy:1.1')),
        'five_runs_fragmented':((0,long),(1,'easy:1.4'),(2,'easy:1.4'),(4,'quality:ordinary'),(6,'easy:1.4')),
        'four_runs_compressed_quality':((0,long),(1,'quality:ordinary'),(3,'easy:2.1'),(5,'easy:2.1')),
        'rest_only':(),
    }
    hands={}
    for key,dose in catalog.items():
        intent=IntendedExecution(key,as_of,as_of+timedelta(days=1),tuple(IntendedPhase(k,
            TargetBand(s,lo,hi,'frozen duration scenario')) for (k,s),(lo,hi) in zip(dose.phases,dose.phase_scenarios)),
            TargetBand(dose.miles,max(.1,dose.miles-.2),dose.miles+.2,'declared distance tolerance'),
            inputs['reference'].accounting_key,measurement_context(config),forecast_scope=dose.forecast_scope)
        hands[key]=issue_execution(intent,inputs['phase_history']).payload['planning_handoff']
    context=ProgramContext((),today,decision,decision+timedelta(days=21),decision+timedelta(days=35),
        build['scales'],build['training_policy'],build['demand_scenario'],build['exchange'],build['timing'],
        demand_accounting_key=inputs['reference'].accounting_key,easy_duration_policy=build['easy_duration_policy'])
    results={}
    for name,cycle in cases.items():
        runs=[]
        bindings=[]
        for week in range(5):
            for offset,key in cycle:
                start=datetime.combine(decision+timedelta(days=week*7+offset),time(18),zone)
                run=planned_accounted(f'{name}:{week}:{offset}',start,catalog[key],hands[key],inputs['reference'],available=as_of)
                runs.append(run)
                c=contracts[key]
                bindings.append(PrescriptionBinding(run.key,c.purpose,c.intensity_key,c))
        decision_input=ProgramDecisionInput(build['stage'].key,as_of,'fixed distribution contrasts',tuple(bindings))
        scenarios={}
        for memory in (1.,2.,3.):
            for exchange in (.005,.01,.02):
                ctx=replace(context,scenario=replace(context.scenario,memory_days=memory),exchange=exchange)
                result=evaluate_continuous(ctx,inputs['history'],runs,reference=inputs['reference'],as_of=as_of,
                    timezone=config['timezone_default'],purpose_fraction=.5,decision_input=decision_input)
                scenarios[f'{memory}:{exchange}']=dict(cost=result['evaluation'].total_cost,parts=result['evaluation'].parts,
                    eligible=result['decision_eligible'])
        results[name]=dict(scenarios=scenarios,metrics=training_metrics(runs,inputs['history'],decision,35,zone))
    ranking={k:sorted(results,key=lambda n:results[n]['scenarios'][k]['cost']) for k in next(iter(results.values()))['scenarios']}
    return dict(as_of=as_of,stage=build['stage'],cases=results,ranking=ranking,
                controls='predeclared matched 16.5-mile cycles except intentionally underfunded/rest cases',
                frozen_evidence=inputs,physiological_validation=False,calendar_search=False)


if __name__=='__main__':
    p=ArgumentParser()
    p.add_argument('--database',type=Path,default=Path('data/run_analysis.sqlite'))
    p.add_argument('--as-of',type=datetime.fromisoformat,default=datetime(2026,9,27,8,tzinfo=timezone.utc))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    config=load_config()
    manifest=freeze_sources(a.output.with_suffix('.sources.zip'),config=config,
        entrypoint='scripts.audit_stage_distribution',inputs=(a.database,))
    result=audit(a.database,config,a.as_of)
    result['manifest']=manifest
    with a.output.open('x') as stream:
        json.dump(result,stream,default=encode,indent=2,allow_nan=False)
    print(json.dumps(dict(ranking=result['ranking'],nominal={n:v['scenarios']['2.0:0.01']['cost'] for n,v in result['cases'].items()})))
