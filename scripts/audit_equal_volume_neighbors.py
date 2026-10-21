"""Exact bounded equal-mileage neighborhoods plus saved search-budget audit.

No optimizer tuning or publication. Seven-day blocks identify the three repeated
patterns in the frozen candidate; they are not planner quotas or calendar weeks.
"""
from argparse import ArgumentParser
from datetime import date, datetime, time, timedelta
from itertools import combinations, product
import json
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

from run_analysis.continuous_training_dose import ContinuousDoseReference
from run_analysis.adaptive.decision_snapshot import restore, evaluate_snapshot
from run_analysis.adaptive.forecast_handoff import read_scope
from run_analysis.adaptive.new_model_daily_harness import Dose, planned_accounted
from run_analysis.adaptive.continuation_boundary_trial import continuation_boundary_cost
from scripts.research_manifest import freeze_sources


def budget_summary(search):
    ref = search['refinement']
    result = dict(evaluations=search['evaluations'], local_complete=search['local_neighborhood_complete'],
        joint_sweeps=len(ref['final_joint']), global_denials=ref['global_budget_denials'])
    result['final_evaluations_by_phase'] = {phase: sum(s['evaluations'] for s in ref['final_joint'] if s['phase'] == phase)
                                          for phase in ('coordinate', 'joint')}
    result['passes'] = {}
    for name in ('pre_calendar_distance', 'calendar', 'post_calendar_distance'):
        step = ref[name]
        result['passes'][name] = None if step is None else dict(
            budget=step['budget'], evaluations=step.get('evaluations', step.get('evaluated_calendars')),
            seconds=step.get('seconds'),
            lanes=len(step['lanes']), statuses={status: sum(lane['status'] == status for lane in step['lanes'])
                for status in sorted({lane['status'] for lane in step['lanes']})})
    result['note'] = 'Pass-local evaluation counts include global cache hits; final sweep counts are new global evaluations.'
    return result


def run(root, previous_report, out):
    out.mkdir(exist_ok=False)
    manifest = freeze_sources(out / 'sources.zip', config={}, entrypoint=__name__,
        inputs=(root / 'decision-0006.json', root / 'snapshot-0006.json', previous_report))
    saved = json.loads(previous_report.read_text())
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
        cache[program] = row
        return row
    baseline = tuple(map(tuple, saved['search']['program']))
    began = perf_counter()
    base = evaluate(baseline)
    assert abs(base['cost'] - saved['search']['cost']) < 1e-12
    easy = tuple(k for k in keys if catalog[k].role == 'easy')
    groups = []
    for block in range(3):
        start = state.context.decision + timedelta(days=7 * block)
        end = start + timedelta(days=7)
        selected = tuple((d, k) for d, k in baseline if start <= date.fromisoformat(d) < end and catalog[k].role == 'easy')
        unchanged = tuple(p for p in baseline if p not in selected)
        allowed = tuple((start + timedelta(days=i)).isoformat() for i in range(7)
                        if (start + timedelta(days=i)).isoformat() not in dict(unchanged))
        target = sum(round(catalog[k].miles * 10) for _, k in selected)
        cases = {}
        for count in (1, 2, 3):
            allocations = [p for p in product(easy, repeat=count) if sum(round(catalog[k].miles * 10) for k in p) == target]
            for days in combinations(allowed, count):
                for allocation in allocations:
                    alternative = tuple(zip(days, allocation))
                    priced = evaluate((*unchanged, *alternative))
                    assert abs(priced['full_miles'] - base['full_miles']) < 1e-9
                    assert abs(priced['hard_miles'] - base['hard_miles']) < 1e-9
                    cases[priced['program']] = dict(priced, easy_runs=count, block_allocation=alternative)
        eligible = [r for r in cases.values() if r['eligible']]
        groups.append(dict(start=start, end=end, miles=target / 10, candidates=len(cases),
            eligible=len(eligible), original=selected,
            best_by_run_count={str(n): min((r for r in eligible if r['easy_runs'] == n), key=lambda r: r['cost'], default=None)
                               for n in (1, 2, 3)}, best=min(eligible, key=lambda r: r['cost'])))
    # Cheap, explicitly coordinated follow-up: combine the individually best
    # blocks and evaluate again. Never assume their improvements add linearly.
    chosen = dict(baseline)
    for group in groups:
        for d, _ in group['original']:
            chosen.pop(d)
        chosen.update(group['best']['block_allocation'])
    combined = evaluate(tuple(chosen.items()))
    report = dict(manifest=manifest, baseline=base, groups=groups, combined=combined,
        unique_evaluations=len(cache), seconds=perf_counter() - began,
        saved_budget=budget_summary(saved['search']),
        scope='Exhaustive 1/2/3-easy-run equal-volume allocation inside each of three fixed 7-day blocks; other sessions/tail frozen. Not global optimality or daily stability.')
    with (out / 'report.json').open('x') as stream:
        json.dump(report, stream, indent=2, default=lambda v: v.isoformat())
    print(json.dumps(dict(baseline=base['cost'], unique_evaluations=len(cache), seconds=report['seconds'],
        groups=[dict(start=str(g['start']), candidates=g['candidates'], eligible=g['eligible'],
                     best_by_run_count={n: dict(cost=v['cost'], allocation=v['block_allocation']) if v else None
                                        for n, v in g['best_by_run_count'].items()}) for g in groups],
        combined={k: combined[k] for k in ('cost', 'eligible', 'hard_miles', 'program')}), indent=2))


if __name__ == '__main__':
    p = ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--previous-report', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    run(args.root, args.previous_report, args.output_dir)
