"""Frozen quality-dose attribution, not a new scheduler or adaptation model.

Reproduce the saved winner, then replace each visible quality prescription with
every admitted native alternative. Also reprice existing exact-volume quality /
easy neighbors. All remaining dates and the entire continuation tail stay fixed.
No model coefficients, source forecasts, live data, or search budget are changed.
"""
from argparse import ArgumentParser
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

from scripts.replay_adaptive_search import load_decision
from scripts.research_manifest import freeze_sources
from run_analysis.adaptive.decimal_dose_search import quality_easy_neighbors


def score_summary(result):
    e = result['evaluation']
    q = e.parts['quality_programme']
    return dict(eligible=result['decision_eligible'], total=e.total_cost,
        fatigue=e.weighted_demand_cost, delivery=e.delivery_cost,
        short=e.short_run_cost,
        other=e.total_cost-e.weighted_demand_cost-e.delivery_cost-e.short_run_cost,
        quality=q['loss'], shared=q['shared_opportunity_loss'],
        descriptor=q['descriptor_loss'],
        sustained=q['purposes']['sustained']['loss'], faster=q['purposes']['faster']['loss'])


def run(root, output, *, index):
    stamp = perf_counter()
    case = load_decision(root, index)  # Mandatory archived full-score/admission parity.
    output.mkdir(exist_ok=False)
    state, catalog = case['state'], case['catalog']
    original = tuple(sorted(map(tuple, case['event']['program'])))
    hard_end = state.context.hard_end.isoformat()
    hard = tuple(p for p in original if p[0] < hard_end)
    tail = tuple(p for p in original if p[0] >= hard_end)
    credits = {r.key: r for r in state.context.quality_programme.planned}
    usable = {k: catalog[k] for k in case['handoffs']}
    cache = {}

    def evaluate(program):
        program = tuple(sorted(program))
        if program not in cache:
            assert tuple(p for p in program if p[0] >= hard_end) == tail
            assert [d for d, _ in program] == [d for d, _ in original]
            cache[program] = score_summary(case['evaluate_details'](program))
        return cache[program]

    baseline = evaluate(original)
    metrics = tuple(k for k in baseline if k != 'eligible')

    def delta(score):
        return {k: score[k] - baseline[k] for k in metrics}

    def describe(key):
        d, r = catalog[key], credits[key]
        definition = d.workout_definition
        return dict(key=key, family=definition['family'], variant=definition.get('variant'),
            miles=d.miles, work_minutes=sum(s for p, s in d.phases if p == 'work') / 60,
            sustained=asdict(r.sustained), faster=asdict(r.faster),
            opportunity=asdict(r.opportunity),
            forecast_cardio=case['handoffs'][key]['cardio'])

    menu = {k: describe(k) for k, d in usable.items() if d.role == 'quality'}
    rows = []
    for day, old in hard:
        if catalog[old].role != 'quality':
            continue
        for key in sorted(menu):
            p = tuple((d, key if d == day else k) for d, k in original)
            score = evaluate(p)
            rows.append(dict(mode='same_date', day=day, before=old, after=key,
                mileage_delta=round(catalog[key].miles - catalog[old].miles, 1),
                score=score, delta=delta(score)))

    roles = {k: d.role for k, d in usable.items()}
    miles = {k: d.miles for k, d in usable.items()}
    positions = {d: i for i, d in enumerate(case['dates'])}
    for p in quality_easy_neighbors(original, roles, miles, index=positions):
        edits = [(d, old, new) for (d, old), (_, new) in zip(original, p) if old != new]
        quality_edits = [r for r in edits if catalog[r[1]].role == 'quality']
        if len(quality_edits) != 1 or quality_edits[0][0] >= hard_end:
            continue
        # Hold the tail fixed, including the compensating easy run.
        if any(d >= hard_end for d, _, _ in edits):
            continue
        assert sum(round(catalog[k].miles * 10) for _, k in p) == sum(
            round(catalog[k].miles * 10) for _, k in original)
        day, old, key = quality_edits[0]
        score = evaluate(p)
        rows.append(dict(mode='equal_volume', day=day, before=old, after=key,
            edits=edits, mileage_delta=0., score=score, delta=delta(score)))

    q = case['evaluate_details'](original)['evaluation'].parts['quality_programme']
    report = dict(source_decision=index, as_of=case['event']['as_of'],
        snapshot_fingerprint=state.fingerprint, archived_winner_reproduced=True,
        baseline=baseline, contract=asdict(state.context.quality_programme.contract),
        menu=menu, selected=[dict(day=d, **menu[k]) for d, k in hard if k in menu],
        timing={name: detail['timing'] for name, detail in q['purposes'].items()},
        comparisons=rows, evaluations=len(cache), seconds=perf_counter() - stamp,
        limitations=['Conditional HR/exposure forecast, not measured adaptation.',
            'Fixed-date counterfactuals; not an exhaustive calendar search or daily replay.',
            'Larger/smaller dose comparisons do not establish physiological equivalence.'],
        live_changed=False, model_changed=False)
    report['manifest'] = freeze_sources(output / 'sources.zip', config={}, entrypoint=__name__,
        inputs=(root / 'scenario.json', root / f'decision-{index:04d}.json',
            root / case['event']['snapshot_file']))
    (output / 'report.json').write_text(json.dumps(report, default=str, indent=2, allow_nan=False))
    print(json.dumps(dict(evaluations=len(cache), comparisons=len(rows),
        seconds=report['seconds'], baseline=baseline), indent=2))


if __name__ == '__main__':
    p = ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--index', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    run(a.root, a.output, index=a.index)
