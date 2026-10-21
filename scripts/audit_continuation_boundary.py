"""Frozen failure-pair continuation test; no search or publication."""
from argparse import ArgumentParser
from dataclasses import replace
from datetime import date, datetime, time
import json
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

from run_analysis.continuous_training_dose import ContinuousDoseReference
from run_analysis.adaptive.decision_snapshot import restore, serialize, evaluate_snapshot
from run_analysis.adaptive.forecast_handoff import read_scope
from run_analysis.adaptive.new_model_daily_harness import Dose, planned_accounted
from run_analysis.adaptive.continuation_boundary_trial import continuation_boundary_cost, reference_keys
from run_analysis.adaptive.continuous_search_adapter import search
from run_analysis.adaptive.stage_calendar_seeds import complete_stage_seeds
from scripts.research_manifest import freeze_sources


def run(root, out, *, search_calendar=False, previous_report=None, budget=10000):
    out.mkdir(exist_ok=False)
    manifest = freeze_sources(out / 'sources.zip', config={'budget': budget}, entrypoint=__name__,
        inputs=tuple(root / f'{kind}-{i:04d}.json' for i in (5, 6) for kind in ('decision', 'snapshot'))
            + ((previous_report,) if previous_report else ()))
    ref = ContinuousDoseReference('sandbox upload accounting')
    events, states = [], []
    for i in (5, 6):
        e = json.loads((root / f'decision-{i:04d}.json').read_text())
        events.append(e)
        states.append(restore((root / e['snapshot_file']).read_text(), expected_fingerprint=e['snapshot_fingerprint']))
    rows = []
    for index, (e, s) in enumerate(zip(events, states)):
        catalog = {d['key']: Dose(**dict(d, phases=tuple(map(tuple, d['phases'])),
            phase_scenarios=tuple(map(tuple, d['phase_scenarios'])), forecast_scope=read_scope(d['forecast_scope'])))
            for d in e['stage_runtime']['catalog']}
        physical = {}

        def evaluate(program, state=None):
            state = s if state is None else state
            for day, key in program:
                if (day, key) not in physical:
                    physical[day, key] = planned_accounted(f'candidate:{day}:{key}',
                        datetime.combine(date.fromisoformat(day), time(18), ZoneInfo(s.timezone)),
                        catalog[key], e['forecast_handoff_by_dose'][key], ref, available=datetime.fromisoformat(e['as_of']))
            return evaluate_snapshot(state, tuple(physical[p] for p in program), tuple(k for _, k in program), reference=ref)

        archived = evaluate(tuple(map(tuple, e['program'])))
        assert abs(archived['evaluation'].total_cost - e['cost']) < 1e-12
        assert archived['evaluation'].fingerprint == e['selected']['score_fingerprint']
        refs = {}
        for role,key in reference_keys(s.context.policy,catalog,e['forecast_handoff_by_dose']).items():
            dose = catalog[key]
            r = planned_accounted('reference:' + key, datetime.combine(s.context.decision, time(18), ZoneInfo(s.timezone)),
                dose, e['forecast_handoff_by_dose'][key], ref, available=s.as_of)
            refs[role] = (dose.miles, r.demand)
        integrated_state=replace(s,context=replace(s.context,continuation_boundary_references=tuple(
            (role,miles,demand) for role,(miles,demand) in refs.items())))
        restored_state=restore(serialize(integrated_state),expected_fingerprint=integrated_state.fingerprint)
        for easy_day in ('2026-10-02', '2026-10-03'):
            program = [(easy_day if d == '2026-10-03' else d, k) for d, k in events[-1]['program']]
            if index == 0:
                program.insert(0, ('2026-09-30', 'quality:ordinary'))
            evaluated = evaluate(tuple(program))
            for distributed in (False, True):
                began = perf_counter()
                trial = continuation_boundary_cost(s.context, evaluated['evaluation'], refs, distributed=distributed)
                elapsed = perf_counter() - began
                if distributed:
                    integrated=evaluate(tuple(program),integrated_state)
                    replay=evaluate(tuple(program),restored_state)
                    assert abs(integrated['evaluation'].total_cost-trial['cost'])<1e-12
                    assert replay==integrated
                    assert integrated['decision_eligible']==evaluated['decision_eligible']
                row = dict(state='before_upload' if index == 0 else 'after_upload', easy_day=easy_day,
                    distributed=distributed, eligible=evaluated['decision_eligible'], old_cost=evaluated['evaluation'].total_cost,
                    seconds=elapsed, integrated_and_restored_parity=distributed, **trial)
                rows.append(row)
                print(json.dumps({k: row[k] for k in ('state', 'easy_day', 'distributed', 'old_cost', 'cost', 'volume_cost', 'cross_demand_cost', 'seconds')}), flush=True)
        if index == 1:
            # Timing includes repeated real-state calls but no optimizer rerun.
            began = perf_counter()
            for _ in range(100):
                continuation_boundary_cost(s.context, evaluated['evaluation'], refs)
            average = (perf_counter() - began) / 100
    report = dict(manifest=manifest, comparisons=rows, mean_added_seconds=average,
        extrapolated_10000_calls_seconds=average * 10000,
        scope='Fixed physical pairs only; no winner search, daily validation or live activation')
    if previous_report:
        saved=json.loads(previous_report.read_text())
        measured=evaluate(tuple(map(tuple,saved['search']['program'])),integrated_state)
        assert abs(measured['evaluation'].total_cost-saved['search']['cost'])<1e-12
        assert measured['decision_eligible']
        report['saved_benchmark_parity']=dict(path=str(previous_report),cost=measured['evaluation'].total_cost,
            eligible=measured['decision_eligible'])
    if search_calendar:
        from datetime import timedelta
        context = s.context
        dates = tuple((context.decision + timedelta(days=i)).isoformat()
                      for i in range((context.end - context.decision).days))
        keys = tuple(e['forecast_handoff_by_dose'])
        seed = set()
        for target in (context.policy.ordinary_miles, context.policy.ordinary_miles / 2):
            seed.add(min((k for k in keys if catalog[k].role == 'easy'),
                         key=lambda k: (abs(catalog[k].miles - target), k)))
        for role in ('long', 'quality'):
            seed.add(max((k for k in keys if catalog[k].role == role), key=lambda k: (
                sum(sec for phase, sec in catalog[k].phases if phase == 'work')
                if role == 'quality' else catalog[k].miles, k)))

        def objective(program):
            measured = evaluate(program,integrated_state)
            return dict(cost=measured['evaluation'].total_cost, eligible=measured['decision_eligible'],
                construction_eligible=measured['decision_eligible'])

        previous = tuple(map(tuple, e['program']))
        if previous_report:
            saved = json.loads(previous_report.read_text())
            if saved['manifest']['path'] == manifest['path']:
                raise ValueError('A refinement must preserve its source report')
            previous = tuple(map(tuple, saved['search']['program']))
            if abs(objective(previous)['cost'] - saved['search']['cost']) > 1e-12:
                raise AssertionError('Warm-start score must reproduce under the same frozen state')
        found = search(dates=dates, keys=keys, previous=previous,
            dose_miles={k: catalog[k].miles for k in keys},
            evaluate=objective, budget=budget, hard_end=context.hard_end.isoformat(),
            roles={k: catalog[k].role for k in keys}, seed_keys=tuple(sorted(seed)),
            previous_end=e['end'], seed_programs=complete_stage_seeds(dates, catalog, context.policy,
                supporting_miles=context.easy_duration_policy.supporting_easy_miles_per_7))
        from run_analysis.adaptive.new_model_daily_harness import training_metrics
        report['search'] = found
        report['selected_accounting'] = evaluate(found['program'],integrated_state)['evaluation'].parts['continuation_boundary']
        report['hard21_training'] = training_metrics([physical[p] for p in found['program']], s.history,
            context.decision, (context.hard_end - context.decision).days, ZoneInfo(s.timezone))
        report['full_miles'] = sum(catalog[k].miles for _, k in found['program'])
        report['scope'] = 'Frozen pair and one bounded post-upload search; not daily stability or acceptance'
        print(json.dumps(dict(search={k: found[k] for k in ('cost', 'seconds', 'evaluations', 'local_neighborhood_complete', 'program')},
            hard21_miles=report['hard21_training']['miles'], full_miles=report['full_miles'])), flush=True)
    with (out / 'report.json').open('x') as stream:
        json.dump(report, stream, indent=2, default=lambda v: v.isoformat())
    print(json.dumps({k: report[k] for k in ('mean_added_seconds', 'extrapolated_10000_calls_seconds')}))


if __name__ == '__main__':
    p = ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--search', action='store_true')
    p.add_argument('--previous-report', type=Path)
    p.add_argument('--budget', type=int, default=10000)
    args = p.parse_args()
    run(args.root, args.output_dir, search_calendar=args.search, previous_report=args.previous_report, budget=args.budget)
