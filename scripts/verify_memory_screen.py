"""Read-only numeric/persistence replay of memory-covered daily screens."""
from argparse import ArgumentParser
from datetime import date,datetime,time
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from run_analysis.adaptive.decision_snapshot import restore,evaluate_snapshot
from run_analysis.adaptive.forecast_handoff import read_scope
from run_analysis.adaptive.new_model_daily_harness import Dose,planned_accounted
from run_analysis.continuous_training_dose import ContinuousDoseReference


def verify(root):
    ref=ContinuousDoseReference('sandbox upload accounting');checks=[]
    for path in sorted(root.glob('decision-*.json')):
        e=json.loads(path.read_text())
        state=restore((root/e['snapshot_file']).read_text(),expected_fingerprint=e['snapshot_fingerprint'])
        assert state.context.continuation_guard_days==7
        assert (state.context.hard_end-state.context.decision).days==21
        assert (state.context.end-state.context.taper_start).days==14
        assert all(d<e['hard_end'] for d in e['active'])
        catalog={d['key']:Dose(**dict(d,phases=tuple(map(tuple,d['phases'])),
            phase_scenarios=tuple(map(tuple,d['phase_scenarios'])),forecast_scope=read_scope(d['forecast_scope'])))
            for d in e['stage_runtime']['catalog']}
        planned=tuple(planned_accounted(f'candidate:{day}:{key}',
            datetime.combine(date.fromisoformat(day),time(18),ZoneInfo(state.timezone)),
            catalog[key],e['forecast_handoff_by_dose'][key],ref,available=state.as_of) for day,key in e['program'])
        r=evaluate_snapshot(state,planned,tuple(k for _,k in e['program']),reference=ref)
        assert abs(r['evaluation'].total_cost-e['cost'])<1e-12
        assert r['evaluation'].fingerprint==e['selected']['score_fingerprint']
        assert r['decision_eligible']==e['selected']['eligible']
        assert r['suitability']==e['selected']['suitability']
        assert r['evaluation'].model_version==e['model_version']
        checks.append(dict(index=e['index'],as_of=e['as_of'],reason=e['reason'],seconds=e['seconds'],
            evaluations=e['evaluations'],churn=e['churn'],planned_training=e['planned_training']))
    return dict(replayed=len(checks),events=checks)


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
    result=verify(a.root)
    with (a.output or a.root/'verification.json').open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(dict(replayed=result['replayed'],events=[{k:v for k,v in r.items() if k!='planned_training'} for r in result['events']]),indent=2))
