"""Hindsight-only path diagnosis between two saved plans, never a churn test.

Reprice exact plans under one frozen state. Enumerate one/two-date substitutions
toward the known better witness, plus contiguous splice controls. This uses
future search knowledge deliberately and must never be called a production
search result or supplied as an independent-discovery seed.
"""
from argparse import ArgumentParser
from datetime import date, datetime, time, timedelta
from itertools import combinations
import json
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

from run_analysis.continuous_training_dose import ContinuousDoseReference
from run_analysis.adaptive.decision_snapshot import restore, evaluate_snapshot
from run_analysis.adaptive.forecast_handoff import read_scope
from run_analysis.adaptive.new_model_daily_harness import Dose, planned_accounted
from run_analysis.adaptive.continuation_boundary_trial import continuation_boundary_cost
from run_analysis.adaptive.calendar_operators import nearby_prescription_neighbors, equal_volume_neighbors
from scripts.research_manifest import freeze_sources


def substitutions(program, target):
    a, b = dict(program), dict(target)
    return tuple(d for d in sorted(a.keys() | b.keys()) if a.get(d) != b.get(d))


def splice(program, target, days):
    a, b = dict(program), dict(target)
    for d in days:
        a.pop(d, None)
        if d in b:
            a[d] = b[d]
    return tuple(sorted(a.items()))


def translated_segments(program, dates, *, max_span=8):
    """Independent diagnostic: shift a short date segment ±1, keeping doses.

    No witness input, deletion at the boundary, or occupied-date overwrite.
    Eight days is only a bounded search radius, not a training cadence.
    """
    dates = tuple(dates)
    index = {d: i for i, d in enumerate(dates)}
    seen = {tuple(program)}
    for width in range(1, min(max_span, len(dates)) + 1):
        for left in range(len(dates) - width + 1):
            moving = [(d, k) for d, k in program if left <= index[d] < left + width]
            if not moving:
                continue
            kept = [(d, k) for d, k in program if not left <= index[d] < left + width]
            for step in (-1, 1):
                if any(not 0 <= index[d] + step < len(dates) for d, _ in moving):
                    continue
                candidate = tuple(sorted((*kept, *((dates[index[d] + step], k) for d, k in moving))))
                if len(dict(candidate)) == len(candidate) and candidate not in seen:
                    seen.add(candidate)
                    yield candidate


def run(root, baseline_path, witness_path, out):
    out.mkdir(exist_ok=False)
    manifest = freeze_sources(out / 'sources.zip', config={}, entrypoint=__name__,
        inputs=(root / 'decision-0006.json', root / 'snapshot-0006.json', baseline_path, witness_path))
    event = json.loads((root / 'decision-0006.json').read_text())
    state = restore((root / event['snapshot_file']).read_text(), expected_fingerprint=event['snapshot_fingerprint'])
    catalog = {d['key']: Dose(**dict(d, phases=tuple(map(tuple, d['phases'])),
        phase_scenarios=tuple(map(tuple, d['phase_scenarios'])), forecast_scope=read_scope(d['forecast_scope'])))
        for d in event['stage_runtime']['catalog']}
    reference = ContinuousDoseReference('sandbox upload accounting')
    keys = tuple(sorted(event['forecast_handoff_by_dose']))
    refs = {}
    for role, target in (('easy', state.context.policy.ordinary_miles), ('long', state.context.policy.long_miles), ('quality', 0)):
        key = 'quality:ordinary' if role == 'quality' else min((k for k in keys if catalog[k].role == role),
            key=lambda k: (abs(catalog[k].miles - target), k))
        sample = planned_accounted('reference:' + key,
            datetime.combine(state.context.decision, time(18), ZoneInfo(state.timezone)),
            catalog[key], event['forecast_handoff_by_dose'][key], reference, available=state.as_of)
        refs[role] = (sample.miles, sample.demand)
    physical, cache = {}, {}

    def evaluate(program):
        program = tuple(sorted(program))
        if program in cache:
            return cache[program]
        assert len(dict(program)) == len(program)
        for day, key in program:
            if (day, key) not in physical:
                physical[day, key] = planned_accounted(f'candidate:{day}:{key}',
                    datetime.combine(date.fromisoformat(day), time(18), ZoneInfo(state.timezone)),
                    catalog[key], event['forecast_handoff_by_dose'][key], reference, available=state.as_of)
        result = evaluate_snapshot(state, tuple(physical[p] for p in program), tuple(k for _, k in program), reference=reference)
        measured = result['evaluation']
        trial = continuation_boundary_cost(state.context, measured, refs)
        row = dict(program=program, cost=trial['cost'], eligible=result['decision_eligible'],
            volume_cost=trial['volume_cost'], demand_cost=measured.weighted_demand_cost + trial['cross_demand_cost'],
            short_run_cost=measured.short_run_cost, other_delivery=measured.delivery_cost - trial['previous_volume_cost'],
            hard_miles=sum(catalog[k].miles for d, k in program if date.fromisoformat(d) < state.context.hard_end),
            full_miles=sum(catalog[k].miles for _, k in program))
        assert abs(sum(row[k] for k in ('volume_cost', 'demand_cost', 'short_run_cost', 'other_delivery')) - row['cost']) < 1e-10
        cache[program] = row
        return row

    began = perf_counter()
    saved = [json.loads(p.read_text())['search'] for p in (baseline_path, witness_path)]
    baseline, witness = (tuple(map(tuple, s['program'])) for s in saved)
    for program, archived in zip((baseline, witness), saved):
        assert abs(evaluate(program)['cost'] - archived['cost']) < 1e-12
        assert evaluate(program)['eligible']
    changed = substitutions(baseline, witness)
    # All one/two-date edits toward this particular witness, not all catalogue
    # edits. Preserve date identity and never introduce duplicate sessions.
    paths = []
    for max_edits in (1, 2):
        current = baseline
        steps = []
        while True:
            remaining = substitutions(current, witness)
            options = [(days, evaluate(splice(current, witness, days)))
                       for count in range(1, max_edits + 1) for days in combinations(remaining, count)]
            valid = [(days, r) for days, r in options if r['eligible']]
            best = min(valid, key=lambda v: (v[1]['cost'], v[0]), default=((), evaluate(current)))
            if best[1]['cost'] >= evaluate(current)['cost']:
                paths.append(dict(max_edits=max_edits, steps=steps, final=evaluate(current),
                    remaining=remaining, examined_final=len(options)))
                break
            days, row = best
            before = evaluate(current)
            steps.append(dict(days=days, before=before['cost'], after=row['cost'],
                component_delta={k: row[k] - before[k] for k in ('volume_cost', 'demand_cost', 'short_run_cost', 'other_delivery')},
                program=row['program']))
            current = row['program']

    splices = []
    for i in range(len(changed)):
        for j in range(i + 1, len(changed) + 1):
            days = changed[i:j]
            splices.append(dict(days=days, **evaluate(splice(baseline, witness, days))))
    # Direct membership in actual public operators. Single-date substitutions
    # are already ordinary insert/delete/dose edits. Do not invent an optimizer.
    first = state.context.decision
    allowed = range((state.context.end - first).days)
    def membership(start, destination):
        occupied = {(date.fromisoformat(d) - first).days: k for d, k in start}
        wanted = tuple(((date.fromisoformat(d) - first).days, k) for d, k in destination)
        if len(substitutions(start, destination)) == 1:
            return 'coordinate'
        if wanted in nearby_prescription_neighbors(occupied, allowed):
            return 'nearby_prescription'
        if any(p == wanted for p in equal_volume_neighbors(occupied, allowed,
                {k: catalog[k].role for k in keys}, {k: catalog[k].miles for k in keys})):
            return 'equal_volume'
        return 'not_in_tested_families'  # Does not claim absence from every operator.
    for path in paths:
        current = baseline
        for step in path['steps']:
            step['operator'] = membership(current, step['program'])
            current = step['program']
    dates = tuple((first + timedelta(days=i)).isoformat() for i in allowed)
    shifts = [evaluate(p) for p in translated_segments(baseline, dates)]
    traced_checks = []
    for sweep in saved[0].get('refinement', {}).get('final_joint', ()):
        if sweep['phase'] != 'final_check' or 'best_program' not in sweep:
            continue
        program = tuple(map(tuple, sweep['best_program']))
        assert abs(evaluate(program)['cost'] - sweep['best_cost']) < 1e-12
        occupied = {(date.fromisoformat(d) - first).days: k for d, k in program}
        rows = [evaluate(tuple((dates[i], k) for i, k in p))
                for p in nearby_prescription_neighbors(occupied, allowed)]
        best = min((r for r in rows if r['eligible']), key=lambda r: r['cost'])
        traced_checks.append(dict(parent_cost=evaluate(program)['cost'], best=best,
            changes=substitutions(program, best['program']), candidates=len(rows)))
    report = dict(manifest=manifest, baseline=evaluate(baseline), witness=evaluate(witness),
        changed=changed, paths=paths, best_splice=min((r for r in splices if r['eligible']), key=lambda r: r['cost']),
        splices=splices, independent_segment_shifts=dict(candidates=len(shifts),
            best=min((r for r in shifts if r['eligible']), key=lambda r: r['cost'], default=None)),
        traced_checks=traced_checks,
        unique_evaluations=len(cache), seconds=perf_counter() - began,
        scope='Hindsight witness substitutions at one frozen state. NOT churn, daily lifecycle, independent discovery or global optimality.')
    with (out / 'report.json').open('x') as stream:
        json.dump(report, stream, indent=2, default=str)
    print(json.dumps(dict(baseline=evaluate(baseline)['cost'], witness=evaluate(witness)['cost'],
        paths=[dict(max_edits=p['max_edits'], cost=p['final']['cost'], steps=[
            {k: s[k] for k in ('days', 'before', 'after', 'operator')} for s in p['steps']]) for p in paths],
        best_splice=report['best_splice']['cost'], shifts=report['independent_segment_shifts'],
        traced_checks=[dict(parent_cost=r['parent_cost'], best_cost=r['best']['cost'],
            changes=r['changes'], candidates=r['candidates']) for r in traced_checks],
        unique_evaluations=len(cache), seconds=report['seconds']), indent=2))


if __name__ == '__main__':
    p = ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--witness', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    run(args.root, args.baseline, args.witness, args.output_dir)
