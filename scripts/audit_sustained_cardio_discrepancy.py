"""Read-only attribution of a saved upload's cardio error and date preference.

No search, publication, database writes, model changes or forecast calibration.
The counterfactual replaces one observed cardio quantity while freezing every
other input. It is not a full expected-completion simulation.
"""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import date, datetime, time
from hashlib import sha256
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from run_analysis.continuous_training_dose import ContinuousDoseReference
from run_analysis.workload_accounting import Quantity
from run_analysis.adaptive.decision_snapshot import restore, evaluate_snapshot
from run_analysis.adaptive.execution_forecast_contract import layout
from run_analysis.adaptive.forecast_handoff import read_scope, scoped_forecast
from run_analysis.adaptive.new_model_daily_harness import Dose, planned_accounted


def audit(root):
    evidence = {}

    def read(name):
        raw = (root / name).read_text()
        evidence[name] = sha256(raw.encode()).hexdigest()
        return json.loads(raw)

    def decision(index):
        event = read(f'decision-{index:04d}.json')
        raw = (root / event['snapshot_file']).read_text()
        evidence[event['snapshot_file']] = sha256(raw.encode()).hexdigest()
        return event, restore(raw, expected_fingerprint=event['snapshot_fingerprint'])

    before, pre = decision(5)
    after, post = decision(6)
    execution = read('execution-0002.json')
    comparison = execution['comparison']
    catalog = {d['key']: Dose(**dict(d, phases=tuple(map(tuple, d['phases'])),
        phase_scenarios=tuple(map(tuple, d['phase_scenarios'])),
        forecast_scope=read_scope(d['forecast_scope'])))
        for d in after['stage_runtime']['catalog']}
    dose = catalog['quality:ordinary']
    reference = ContinuousDoseReference('sandbox upload accounting')
    forecast = scoped_forecast(layout(dose.phases), pre.phase_history,
        as_of=pre.as_of, accounting_key=reference.accounting_key,
        scope=dose.forecast_scope)
    target = comparison['target_load']
    if abs(forecast['predicted_load'] - target) > 1e-12:
        raise AssertionError('Pre-upload forecast does not reproduce the issued target')
    donor = next(h for h in pre.phase_history
                 if h.key == execution['provenance']['source_key'])
    actual = next(h for h in post.phase_history if h.key == comparison['actual_key'])
    if len(actual.phases) != len(dose.phases):
        raise AssertionError('Actual phase partition differs from the prescription')
    pools = {kind: [p for p in donor.phases if p.kind == kind]
             for kind, _ in dose.phases}
    used = Counter()
    phases = defaultdict(lambda: dict(forecast=0., ideal_stretched_donor=0., observed=0.))
    source_keys = defaultdict(set)
    for (kind, seconds), predicted, observed in zip(dose.phases, forecast['phases'], actual.phases):
        source = pools[kind][min(used[kind], len(pools[kind]) - 1)]
        used[kind] += 1
        if source.load is None or abs(source.known_seconds - source.seconds) > 1e-6:
            raise AssertionError('This exact donor decomposition requires complete source HR')
        if observed.kind != kind or abs(observed.seconds - seconds) > 1e-6:
            raise AssertionError('Actual phase duration differs')
        phases[kind]['forecast'] += predicted['load']
        phases[kind]['ideal_stretched_donor'] += source.load / source.seconds * seconds
        phases[kind]['observed'] += observed.load
        source_keys[kind].update(s['key'] for s in predicted['source_sessions'])
    totals = {k: sum(p[k] for p in phases.values()) for k in next(iter(phases.values()))}
    if abs(totals['observed'] - comparison['actual_cardio']['estimate']) > 1e-10:
        raise AssertionError('Observed phase accounting does not reproduce the upload')

    corrected = replace(post, history=tuple(replace(r,
        demand=(r.demand[0], Quantity.point(target, 'diagnostic exact-cardio intervention'), r.demand[2]))
        if r.key == comparison['actual_key'] else r for r in post.history))
    rows = []
    for label, state in (('actual', post), ('cardio_exact_other_inputs_frozen', corrected)):
        for easy_day in ('2026-10-02', '2026-10-03'):
            program = tuple((easy_day if day == '2026-10-03' else day, key)
                            for day, key in after['program'])
            planned = tuple(planned_accounted(f'candidate:{day}:{key}',
                datetime.combine(date.fromisoformat(day), time(18), ZoneInfo(state.timezone)),
                catalog[key], after['forecast_handoff_by_dose'][key], reference,
                available=state.as_of) for day, key in program)
            result = evaluate_snapshot(state, planned, tuple(k for _, k in program), reference=reference)
            score = result['evaluation']
            if label == 'actual' and easy_day == '2026-10-03' and abs(score.total_cost - after['cost']) > 1e-12:
                raise AssertionError('Selected archived score does not reproduce')
            rows.append(dict(scenario=label, easy_day=easy_day, cost=score.total_cost,
                eligible=result['decision_eligible'], delivery_loss=score.parts['delivery_loss'],
                demand_cost=score.parts['demand_cost']))
    return dict(evidence_sha256=evidence, script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='One saved donor/forecast mismatch and fixed-pair cardio-only counterfactual; not search or whole-model validation',
        donor_key=donor.key, donor_phase_counts={k: len(v) for k, v in pools.items()},
        forecast_source_keys_by_kind={k: sorted(v) for k, v in source_keys.items()},
        phase_loads=dict(phases), total_loads=totals,
        observed_minus_forecast=totals['observed'] - totals['forecast'],
        transformed_vs_ideal_residual=totals['observed'] - totals['ideal_stretched_donor'],
        date_comparisons=rows,
        limitations=['Recorded trace execution is not generated from forecast rates',
                    'Donor phase reuse/time stretching is an engineering fixture, not predicted physiology',
                    'Exact cardio intervention does not reset duration, timing, future forecasts or all completion inputs',
                    'No calibrated error tolerance or new model weight is inferred'])


if __name__ == '__main__':
    p = ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = audit(args.root)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))
