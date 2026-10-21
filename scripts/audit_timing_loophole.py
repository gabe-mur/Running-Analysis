"""Rescore the saved overload failure under unchanged evidence and new coverage."""
from argparse import ArgumentParser
from dataclasses import replace
from datetime import date,datetime,time,timedelta
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from scripts.decision_snapshot import migrate_snapshot_defaults,evaluate_snapshot
from scripts.forecast_handoff import read_scope
from scripts.new_model_daily_harness import Dose,planned_accounted
from run_analysis.continuous_training_dose import ContinuousDoseReference


def audit(root):
    event=json.loads((root/'decision-0001.json').read_text())
    scenario=json.loads((root/'scenario.json').read_text())
    migration=migrate_snapshot_defaults((root/event['snapshot_file']).read_text(),expected_fingerprint=event['snapshot_fingerprint'])
    state=migration.pop('state')
    ref=ContinuousDoseReference('sandbox upload accounting')
    catalog={d['key']:Dose(**dict(d,phases=tuple(tuple(p) for p in d['phases']),
        phase_scenarios=tuple(tuple(p) for p in d['phase_scenarios']),forecast_scope=read_scope(d['forecast_scope']))) for d in scenario['catalog']}
    start=date.fromisoformat(event['decision'])
    balanced=tuple(((start+timedelta(days=7*c+n)).isoformat(),key) for c in range(5)
        for n,key in ((0,'long:8.4'),(2,'easy:2.1'),(4,'quality:ordinary'),(6,'easy:2.1')))
    results={}
    for name,program in (('saved_overload',event['program']),('matched_stage_reference',balanced)):
        runs=tuple(planned_accounted(f'probe:{day}:{key}',datetime.combine(date.fromisoformat(day),time(18),ZoneInfo(state.timezone)),
            catalog[key],event['forecast_handoff_by_dose'][key],ref,available=state.as_of) for day,key in program)
        cases={}
        for grace in (0.,1.):
            updated=replace(state,context=replace(state.context,timing=replace(state.context.timing,grace_fraction=grace)))
            result=evaluate_snapshot(updated,runs,tuple(k for _,k in program),reference=ref)
            cases[str(grace)]=dict(cost=result['evaluation'].total_cost,
                demand=result['evaluation'].weighted_demand_cost,delivery=result['evaluation'].delivery_cost,
                eligible=result['decision_eligible'])
        results[name]=dict(cases=cases,runs=len(runs),miles=sum(r.miles for r in runs))
    return dict(results=results,snapshot_migration=migration)


if __name__=='__main__':
    p=ArgumentParser()
    p.add_argument('root',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    result=audit(a.root)
    with a.output.open('x') as stream:
        json.dump(result,stream,indent=2)
    print(json.dumps(result))
