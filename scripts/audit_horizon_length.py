"""Frozen-evidence horizon ablation using the connected full search, no uploads.

Not a daily lifecycle pass: isolate one witnessed boundary transition before
paying for another longitudinal run. No objectives/weights or live state change.
"""
from argparse import ArgumentParser
from dataclasses import replace
from datetime import date, datetime, time, timedelta
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from run_analysis.config import load_config
from run_analysis.continuous_training_dose import ContinuousDoseReference
from scripts.decision_snapshot import migrate_snapshot_defaults, evaluate_snapshot
from scripts.forecast_handoff import read_scope
from scripts.new_model_daily_harness import Dose, planned_accounted, churn
from scripts.continuous_search_adapter import search
from scripts.stage_calendar_seeds import complete_stage_seeds
from scripts.research_manifest import freeze_sources


def run(root, output, *, horizon_days=49, evaluations=10000, event_numbers=(9, 10),start_case=None):
    output.mkdir(exist_ok=False)
    manifest = freeze_sources(output/'sources.zip', config=load_config(),
        entrypoint='scripts.audit_horizon_length',
        inputs=tuple(root/f'decision-{n:04d}.json' for n in event_numbers))
    events = [json.loads((root/f'decision-{n:04d}.json').read_text()) for n in event_numbers]
    states = [migrate_snapshot_defaults((root/e['snapshot_file']).read_text(),
        expected_fingerprint=e['snapshot_fingerprint'])['state'] for e in events]
    scenario = json.loads((root/'scenario.json').read_text())
    catalog = {d['key']: Dose(**dict(d, phases=tuple(tuple(p) for p in d['phases']),
        phase_scenarios=tuple(tuple(p) for p in d['phase_scenarios']),
        forecast_scope=read_scope(d['forecast_scope']))) for d in scenario['catalog']}
    reference = ContinuousDoseReference('sandbox upload accounting')
    previous = tuple(tuple(p) for p in events[0]['program'])
    cases = []
    for index in range(2):
        # Hold evidence AND decision origin fixed. An enlarged first plan can
        # schedule a previously empty opening day; advancing the decision date
        # without completing that new run would accidentally test a miss.
        event,state=events[0],states[0]
        end = state.context.decision+timedelta(days=horizon_days+index)
        def extend(segments):
            return (*segments[:-1], replace(segments[-1], end=end)) if segments else ()
        state = replace(state, context=replace(state.context, end=end,
            volume_targets=extend(state.context.volume_targets), support_targets=extend(state.context.support_targets)))
        dates = tuple((state.context.decision+timedelta(days=n)).isoformat() for n in range(horizon_days+index))
        handoffs = event['forecast_handoff_by_dose']
        physical = {(d, k): planned_accounted(f'candidate:{d}:{k}',
            datetime.combine(date.fromisoformat(d), time(18), ZoneInfo(state.timezone)), catalog[k],
            handoffs[k], reference, available=state.as_of) for d in dates for k in handoffs}
        def evaluate(program):
            r = evaluate_snapshot(state, tuple(physical[p] for p in program),
                tuple(k for _, k in program), reference=reference)
            return dict(cost=r['evaluation'].total_cost, eligible=r['decision_eligible'],
                constraints=r['constraints']['status'], recovery_feasibility=r['constraints']['recovery_feasibility'],
                suitability=r['suitability'], score_fingerprint=r['evaluation'].fingerprint)
        keys = set()
        for target in (state.context.policy.ordinary_miles, state.context.policy.ordinary_miles/2):
            keys.add(min((k for k in handoffs if catalog[k].role=='easy'), key=lambda k:(abs(catalog[k].miles-target),k)))
        for role in ('long', 'quality'):
            keys.add(max((k for k in handoffs if catalog[k].role==role), key=lambda k:(
                sum(s for p,s in catalog[k].phases if p=='work') if role=='quality' else catalog[k].miles,k)))
        if index==0 and start_case is not None:
            result=json.loads(start_case.read_text())
            checked=evaluate(tuple(tuple(p) for p in result['program']))
            if not checked['eligible'] or abs(checked['cost']-result['selected']['cost'])>1e-12:
                raise ValueError('Saved candidate does not reproduce under frozen evidence')
            result['reused_candidate_from']=str(start_case)
        else:
            result = search(dates=dates, keys=tuple(sorted(handoffs)), previous=previous, evaluate=evaluate,
                budget=evaluations, hard_end=state.context.hard_end.isoformat(),
                roles={k:catalog[k].role for k in handoffs}, seed_keys=tuple(sorted(keys)),
                seed_programs=complete_stage_seeds(dates, catalog, state.context.policy,
                    supporting_miles=state.context.easy_duration_policy.supporting_easy_miles_per_7))
        edits = churn(previous, result['program'], state.context.decision, catalog)
        case = dict(result,as_of=state.as_of.isoformat(), horizon_days=horizon_days+index, churn=edits,
                    evidence_and_origin_frozen=True)
        with (output/f'case-{len(cases)+1}.json').open('x') as stream:
            json.dump(case, stream, indent=2)
        cases.append(case)
        previous = result['program']
        print(json.dumps({k:case[k] for k in ('as_of','horizon_days','churn','seconds','local_neighborhood_complete','evaluations')}), flush=True)
    with (output/'summary.json').open('x') as stream:
        json.dump(dict(manifest=manifest, event_numbers=event_numbers, cases=cases,
            scope='frozen-evidence paired horizon ablation, not longitudinal acceptance'), stream, indent=2)


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--horizon-days', type=int, default=49)
    p.add_argument('--evaluations', type=int, default=10000)
    p.add_argument('--events', type=int, nargs=2, default=(9,10))
    p.add_argument('--start-case',type=Path)
    a = p.parse_args()
    run(a.root,a.output_dir,horizon_days=a.horizon_days,evaluations=a.evaluations,event_numbers=tuple(a.events),start_case=a.start_case)
