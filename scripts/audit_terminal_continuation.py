"""Score-only candidate-conditioned continuation probe; no scheduler/deployment.

Repeat a candidate's final complete cycle as a declared background policy. This
is NOT a training recommendation or a calibrated value function. Check several
cycle/coverage lengths before deciding whether this approximation is usable.
"""
from argparse import ArgumentParser
from dataclasses import replace
from datetime import date, datetime, time, timedelta
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from run_analysis.continuous_training_dose import ContinuousDoseReference
from scripts.decision_snapshot import migrate_snapshot_defaults, evaluate_snapshot
from scripts.forecast_handoff import read_scope
from scripts.new_model_daily_harness import Dose, planned_accounted


def repeat_suffix(program, *, end, cycle_days, extra_days):
    if cycle_days < 1 or extra_days < 0:
        raise ValueError('Positive cycle and nonnegative extension required')
    rows = dict(program)
    suffix = {date.fromisoformat(d): k for d, k in program
              if end-timedelta(days=cycle_days) <= date.fromisoformat(d) < end}
    for offset in range(extra_days):
        day = end+timedelta(days=offset)
        source = end-timedelta(days=cycle_days)+timedelta(days=offset % cycle_days)
        if source in suffix:
            rows[day.isoformat()] = suffix[source]
    return tuple(sorted(rows.items()))


def audit(root, event_numbers=(9, 10)):
    events = [json.loads((root/f'decision-{n:04d}.json').read_text()) for n in event_numbers]
    states = [migrate_snapshot_defaults((root/e['snapshot_file']).read_text(),
              expected_fingerprint=e['snapshot_fingerprint'])['state'] for e in events]
    scenario = json.loads((root/'scenario.json').read_text())
    catalog = {d['key']: Dose(**dict(d, phases=tuple(tuple(p) for p in d['phases']),
        phase_scenarios=tuple(tuple(p) for p in d['phase_scenarios']),
        forecast_scope=read_scope(d['forecast_scope']))) for d in scenario['catalog']}
    reference = ContinuousDoseReference('sandbox upload accounting')
    consumed = {d: k for d, k in events[0]['program'] if d < states[1].context.decision.isoformat()}
    rows = []
    for cycle in (14, 21, 28):
        for extra in (0, 28, 56):
            comparisons = {}
            for label, state, event in zip(('earlier', 'later'), states, events):
                endpoint = state.context.end+timedelta(days=extra)
                def extend_segments(segments):
                    return (*segments[:-1], replace(segments[-1], end=endpoint)) if segments else ()
                state = replace(state, context=replace(state.context, end=endpoint,
                    volume_targets=extend_segments(state.context.volume_targets),
                    support_targets=extend_segments(state.context.support_targets)))
                candidates = {}
                for name, old_event in zip(('prior', 'later_winner'), events):
                    # Continue from each candidate's own original endpoint. A
                    # later comparison never silently interprets the missing day
                    # of the prior candidate as a newly chosen rest day.
                    own_end = date.fromisoformat(old_event['end'])
                    p = repeat_suffix(tuple({**consumed, **dict(old_event['program'])}.items()),
                        end=own_end, cycle_days=cycle, extra_days=max(0, (endpoint-own_end).days))
                    p = tuple((d, k) for d, k in p if state.context.decision.isoformat() <= d < endpoint.isoformat())
                    physical = tuple(planned_accounted(f'terminal:{d}:{k}',
                        datetime.combine(date.fromisoformat(d), time(18), ZoneInfo(state.timezone)),
                        catalog[k], event['forecast_handoff_by_dose'][k], reference,
                        available=state.as_of) for d, k in p)
                    r = evaluate_snapshot(state, physical, tuple(k for _, k in p), reference=reference)
                    v = r['evaluation']
                    candidates[name] = dict(cost=v.total_cost, eligible=r['decision_eligible'],
                        demand=v.weighted_demand_cost, delivery=v.delivery_cost,
                        timing={k: s['loss'] for k, s in v.parts['timing'].items()},
                        total_miles=sum(catalog[k].miles for _, k in p))
                comparisons[label] = candidates
            rows.append(dict(cycle_days=cycle, extra_days=extra, comparisons=comparisons))
    return dict(event_numbers=event_numbers, results=rows,
        scope='score-only fixed candidates; repeated suffix sensitivity, not rollout acceptance')


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('--events', type=int, nargs=2, default=(9, 10))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = audit(a.root, tuple(a.events))
    with a.output.open('x') as stream:
        json.dump(result, stream, indent=2)
    for row in result['results']:
        print(json.dumps(dict(cycle=row['cycle_days'], extra=row['extra_days'],
            differences={k: v['later_winner']['cost']-v['prior']['cost']
                         for k, v in row['comparisons'].items()})))
