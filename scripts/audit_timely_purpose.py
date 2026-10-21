"""Bounded timing-value controls and saved-frontier phase comparison. Not live."""

from argparse import ArgumentParser
from dataclasses import replace
from datetime import date, datetime, timedelta
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from run_analysis.config import load_config
from run_analysis.workload_accounting import Quantity
from scripts.program_objective_trial import TrialSession, TrialPolicy, DemandScenario, describe
from scripts.shared_program_evaluator import ProgramCandidate, ProgramContext, evaluate_program
from scripts.timely_purpose_trial import PurposeTimingPolicy


START = date(2026, 9, 26)
PATTERNS = dict(regular=(1, 8, 15), deferred=(7, 14, 21),
                clustered=(1, 2, 15), frontloaded=(0, 1, 2))
SCALES = (.5, 1., 2.)
MEMORIES = (1., 2., 4.)
EXCHANGES = (.002, .01, .02)


def split(rows, start, *, basis):
    return ProgramCandidate(tuple(r for r in rows if r.day < start+timedelta(days=21)),
                            tuple(r for r in rows if r.day >= start+timedelta(days=21)), basis)


def synthetic_session(key, offset, role, source='planned'):
    miles = {'long': 8.4, 'quality': 4.5, 'easy': 4.4}[role]
    return TrialSession(key, START+timedelta(days=offset), miles, role,
        (Quantity.point(miles*10, 'synthetic duration'),
         Quantity.point(miles*20, 'synthetic cardio'),
         Quantity.point(15. if role == 'quality' else 0., 'synthetic structured work')),
        1. if role == 'quality' else 0., source)


def control_plans(role, days):
    easy = tuple(synthetic_session(f'easy:{i}', i, 'easy') for i in range(4, days, 7))
    tail = tuple(synthetic_session(f'tail:{i}', i, role) for i in range(28, days, 7))
    return {name: split((*easy, *tail, *(synthetic_session(f'key:{j}', i, role)
                        for j, i in enumerate(offsets))), START, basis='fixed common continuation')
            for name, offsets in PATTERNS.items()}


def controls():
    rows = []
    for role, days, scale, memory in product(('long', 'quality'), (56, 112), SCALES, MEMORIES):
        plans = control_plans(role, days)
        for condition, age, factor in (('ordinary', 6, 1.), ('recent_high_demand', 1, 3.)):
            h = synthetic_session('actual', -age, role, 'observed')
            h = replace(h, demand=tuple(q.scale(factor) for q in h.demand))
            context = ProgramContext((h,), START, START, START+timedelta(days=21),
                START+timedelta(days=days), (44., 88., 15.), TrialPolicy(),
                DemandScenario(memory_days=memory), .01, PurposeTimingPolicy(scale, (role,)))
            evaluated = {name: evaluate_program(context, p) for name, p in plans.items()}
            legacy = {name: evaluate_program(replace(context, timing=None), p) for name, p in plans.items()}
            for exchange in EXCHANGES:
                costs = {name: v.delivery_cost+exchange*v.parts['demand_cost'] for name, v in evaluated.items()}
                previous = {name: v.delivery_cost+exchange*v.parts['demand_cost'] for name, v in legacy.items()}
                rows.append(dict(role=role, days=days, timing_scale=scale, demand_memory=memory,
                    exchange=exchange, condition=condition, winner=min(costs, key=costs.get),
                    aggregate_winner=min(previous, key=previous.get), costs=costs,
                    regular_beats={name: costs['regular'] < costs[name] for name in PATTERNS if name != 'regular'},
                    components={name: dict(delivery=v.delivery_cost, demand=v.parts['demand_cost'],
                        timing=v.parts['timing'][role]['loss']) for name, v in evaluated.items()}))
    ordinary = [r for r in rows if r['condition'] == 'ordinary']
    recent = [r for r in rows if r['condition'] == 'recent_high_demand']
    return dict(cases=len(rows), ordinary_cases=len(ordinary),
        ordinary_reversals=[r for r in ordinary if not all(r['regular_beats'].values())],
        recent_high_demand_winner_counts={name: sum(r['winner'] == name for r in recent) for name in PATTERNS},
        cases_detail=rows)


def saved_frontier(database, config, artifact):
    from scripts.audit_recency_demand import load_evidence
    from scripts.audit_volume_demand_frontier import materialize
    before = sha256(database.read_bytes()).hexdigest()
    saved = json.loads(artifact.read_text())
    origin = date.fromisoformat(saved['start'])
    evidence, unresolved, _, _ = load_evidence(database, config, datetime.fromisoformat(saved['as_of']))
    if unresolved or len(evidence) != saved['history_count']:
        raise ValueError('Saved comparison history cannot be reconstructed as declared')
    history = tuple(TrialSession('actual:'+r['key'], r['day'], r['miles'], r['role'],
        tuple(r['quantities'][c] for c in ('running_minutes', 'cardio', 'work_minutes')), 0., 'observed')
        for r in evidence)
    # The previous adapter's historical quality zero is a placeholder, not proof
    # of no quality completion. Do not use it to initialize quality timing.
    rates = {role: {axis: dict(bounds=Quantity(**entry['bounds']))
                    for axis, entry in channels.items()} for role, channels in saved['rates'].items()}
    last_long = max(r.day for r in history if r.role == 'long')
    results = []
    for frontier in saved['frontiers']['nominal']:
        states = [np.roll(frontier['state'], phase) for phase in range(21)]
        programs = [materialize(s, origin, 3, rates, 'rotated') for s in states]
        candidate_rows = [split(p, origin, basis='saved 21-day pattern repeated for 63 days') for p in programs]
        for scale, memory in product(SCALES, MEMORIES):
            ctx = ProgramContext(history, origin, origin, origin+timedelta(days=21),
                origin+timedelta(days=63), tuple(saved['scales']), TrialPolicy(),
                DemandScenario(memory_days=memory), .01, PurposeTimingPolicy(scale, ('long',)))
            evaluated = [evaluate_program(ctx, p) for p in candidate_rows]
            if scale == 1 and memory == 2:
                if abs(evaluated[0].parts['demand_cost'] - frontier['cost']) > 1e-8:
                    raise ValueError('Reconstructed demand no longer matches saved frontier')
            for exchange in EXCHANGES:
                costs = [v.delivery_cost+exchange*v.parts['demand_cost'] for v in evaluated]
                aggregate_costs = [v.parts['aggregate_delivery_loss'] + exchange*v.parts['demand_cost']
                                   for v in evaluated]
                winner = min(range(21), key=lambda i: (costs[i], i))
                aggregate_winner = min(range(21), key=lambda i: (aggregate_costs[i], i))
                first = min(r.day for r in programs[winner] if r.role == 'long')
                aggregate_first = min(r.day for r in programs[aggregate_winner] if r.role == 'long')
                results.append(dict(weekly_miles=frontier['weekly_equivalent_miles'],
                    timing_scale=scale, demand_memory=memory, exchange=exchange,
                    winner_phase=winner, first_long_gap_days=(first-last_long).days,
                    aggregate_winner_phase=aggregate_winner,
                    aggregate_winner_first_long_gap_days=(aggregate_first-last_long).days,
                    aggregate_winner_cost=aggregate_costs[aggregate_winner],
                    aggregate_original_cost=aggregate_costs[0],
                    original_first_long_gap_days=frontier['first_key_gaps_from_actual_history']['long']['days'],
                    winner_cost=costs[winner], original_cost=costs[0],
                    winning_calendar=[dict(day=r.day, role=r.role, miles=r.miles) for r in programs[winner]
                                      if r.day < origin+timedelta(days=21)],
                    training=describe(programs[winner], history, origin, 21)))
    after = sha256(database.read_bytes()).hexdigest()
    if after != before:
        raise ValueError('Database changed during read-only comparison; results require reconciliation')
    return dict(history_count=len(history), database_unchanged=True,
        purpose_scope='long timing only; quality aggregate unchanged because historical credit is unresolved',
        compared_phases=21, cases=len(results), details=results,
        nominal=[r for r in results if r['timing_scale'] == 1 and r['demand_memory'] == 2 and r['exchange'] == .01])


def audit(database=None, config=None, artifact=None):
    start = perf_counter()
    result = dict(status='offline_policy_sensitivity_not_scheduler_validation', controls=controls(),
        planner_influence=False, calibrated=False)
    if database is not None:
        result['saved_frontier'] = saved_frontier(database, config, artifact)
    result['elapsed_seconds'] = perf_counter()-start
    return result


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, default=Path('data/run_analysis.sqlite'))
    parser.add_argument('--config', type=Path, default=Path('config.yaml'))
    parser.add_argument('--artifact', type=Path, default=Path('output/volume-demand-frontier-20260926-expanded.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output path')
    result = audit(args.database, load_config(args.config), args.artifact)
    with args.output.open('x') as stream:
        from scripts.audit_recency_demand import json_ready
        json.dump(result, stream, default=json_ready,
                  indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(cases=result['controls']['cases'],
        ordinary_reversals=len(result['controls']['ordinary_reversals']),
        recent_high_demand_winners=result['controls']['recent_high_demand_winner_counts'],
        nominal=[dict(volume=r['weekly_miles'], old_gap=r['original_first_long_gap_days'],
            aggregate_best_gap=r['aggregate_winner_first_long_gap_days'],
            new_gap=r['first_long_gap_days'], phase=r['winner_phase']) for r in result['saved_frontier']['nominal']],
        seconds=result['elapsed_seconds']), indent=2))
