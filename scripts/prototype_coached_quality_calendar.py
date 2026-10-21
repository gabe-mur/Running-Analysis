"""One frozen coached-quality projection through the established whole-plan search.

No live access, predictor refit or daily-pass claim. Independent fatigue axes
remain the default; the optional shared-running mode is an uncalibrated cost
assumption, not measured recovery or a lossless accounting repair.
Prescribed ordinary quality structures/doses are assigned before each proposal's
physical identity and full evaluation, including its entire scored soft tail.
"""
from argparse import ArgumentParser
from dataclasses import asdict, replace
import json
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

from scripts.replay_adaptive_search import load_decision
from scripts.research_manifest import freeze_sources
from scripts.check_explicit_distribution_tail import incoming_demand_trace
from run_analysis.adaptive.quality_coaching_policy import (
    freeze_sequence, continuation_reference, evaluate_coached_program)
from run_analysis.adaptive.coached_pipeline import coached_evaluator
from run_analysis.adaptive.continuous_search_adapter import search
from run_analysis.adaptive.continuous_program_adapter import evaluate_continuous
from run_analysis.adaptive.decision_snapshot import evaluate_snapshot
from run_analysis.adaptive.stage_calendar_seeds import complete_stage_seeds
from run_analysis.adaptive.decimal_dose_search import calendar_representatives
from run_analysis.adaptive.new_model_daily_harness import training_metrics
from run_analysis.continuous_training_dose import ContinuousDoseReference


def run(root, output, *, index=1, evaluations=10000, first_purpose='sustained',
        demand_interaction='independent_v1'):
    case = load_decision(root, index)  # Mandatory unmodified archived-winner parity.
    output.mkdir(exist_ok=False)
    state, catalog, dates = (case[k] for k in ('state', 'catalog', 'dates'))
    available = {k: catalog[k] for k in case['handoffs']}
    sequence = freeze_sequence(available, stage_key=state.stage_key,
        issued_at=state.as_of.isoformat(), ordinary_miles=state.context.policy.ordinary_miles,
        quality_work_minutes=state.context.policy.quality_work_minutes,
        first_purpose=first_purpose)
    engine = coached_evaluator(case, sequence,demand_interaction=demand_interaction)
    detail, evaluate, cache, quality_reference, bind = (engine[k] for k in
        ('detail', 'evaluate', 'cache', 'quality_reference', 'bind'))
    allowed = {r.key for r in sequence.ordinary}
    supported = {k: d for k, d in available.items() if d.role != 'quality' or k in allowed}

    original = tuple(map(tuple, case['event']['program']))
    rebound = bind(original)
    seed_policy = replace(state.context.policy,
        weekly_miles=max(s.miles_per_7_days for s in state.context.volume_targets))
    seeds = complete_stage_seeds(dates, supported, seed_policy,
        supporting_miles=max(s.miles_per_7_days for s in state.context.support_targets))
    representatives = calendar_representatives(supported,
        ordinary=state.context.policy.ordinary_miles,
        support=state.context.easy_duration_policy.supporting_easy_miles_per_7)
    seed_keys = {sequence.assignment(0).key}
    for role in ('easy', 'long'):
        options = [k for k, d in supported.items() if d.role == role]
        seed_keys.add(max(options, key=lambda k: (supported[k].miles, k)))
    stamp = perf_counter()
    result = search(dates=dates, keys=tuple(supported), previous=rebound, evaluate=evaluate,
        budget=evaluations, hard_end=state.context.hard_end.isoformat(),
        roles={k: d.role for k, d in supported.items()}, seed_programs=seeds,
        seed_keys=tuple(sorted(seed_keys)), complete_first=True,
        dose_miles={k: d.miles for k, d in supported.items()},
        exploration_keys=representatives, fair_coordinates=True,
        recombination_dates=(state.context.taper_start.isoformat(),),
        final_winner_budget=1000, program_transform=bind)
    seconds = perf_counter() - stamp
    program = result['program']
    # Independent final full-score evaluation, not a cached selection claim.
    cache.pop(program, None)
    final = detail(program)
    if not final['decision_eligible'] or abs(final['evaluation'].total_cost - result['cost']) > 1e-12:
        raise ValueError('Final physical winner does not reproduce or pass the original admission checks')
    evaluation = final['evaluation']

    def describe(program):
        rows = []; rank = 0
        for day, key in program:
            dose = catalog[key]
            row = dict(day=day, role=dose.role, miles=dose.miles)
            if dose.role == 'quality':
                assignment = sequence.assignment(rank); rank += 1
                row.update(purpose=assignment.purpose, recipe=assignment.recipe,
                    family=assignment.family, variant=dose.workout_definition['variant'],
                    work_minutes=assignment.commanded_work_minutes,
                    workout=[s['instruction'] for s in dose.workout_definition['structure']])
            rows.append(row)
        return rows

    def summarize(p):
        r = detail(p); e = r['evaluation']
        return dict(total=e.total_cost, fatigue=e.weighted_demand_cost, delivery=e.delivery_cost,
            short=e.short_run_cost, eligible=r['decision_eligible'],
            quality_cadence_loss=e.parts['shortfall_by_purpose']['quality'])

    report = dict(source_decision=index, source_snapshot=state.fingerprint,
        demand_interaction=demand_interaction,
        original_winner_reproduced=True, original_cost_different_objective=case['archived_cost'],
        coaching_contract=asdict(sequence), coaching_contract_key=sequence.key,
        cycle_length=sequence.cycle_length,
        coaching_cycle=[asdict(sequence.assignment(i)) for i in range(sequence.cycle_length)],
        coaching_quality_continuation_reference=asdict_ref(quality_reference),
        rebound_original=summarize(rebound), selected=summarize(program), search=result,
        search_seconds=seconds, actual_evaluator_calls=len(cache) + 1,
        decision=state.context.decision.isoformat(), hard_end=state.context.hard_end.isoformat(),
        end=state.context.end.isoformat(), hard_schedule=describe(tuple(p for p in program
            if p[0] < state.context.hard_end.isoformat())), full_schedule=describe(program),
        key_entry=[r for r in incoming_demand_trace(evaluation, state.context) if r['role'] != 'easy'],
        training_metrics=training_metrics([case['physical'][p] for p in program], state.history,
            state.context.decision, (state.context.hard_end - state.context.decision).days, ZoneInfo(state.timezone)),
        constraints=final['constraints'], suitability=final['suitability'],
        progression_delivery=final.get('progression_delivery'),
        exposure_diagnostics={k: asdict(r) for k, r in (
            (r.key, r) for r in state.context.quality_programme.planned) if k in allowed},
        limitations=['Frozen ordinary general-fitness coaching prototype; not live, race or reduced-dose policy.',
            'No daily/upload stability validation or observed adaptation claim.',
            'First purpose is an explicit new-policy opening assumption, not historical optimality.',
            'Future quality ranks are rebound before scoring; insertion/deletion can change later assignments.',
            'No calendar locks, continuity coefficient or forced rest day.',
            ('Existing independent fatigue axes retained.' if demand_interaction=='independent_v1' else
             'Uncalibrated shared running/work cost assumption; possible intensity double counting; not live.')],
        live_changed=False, original_forecasts_changed=False)
    report['manifest'] = freeze_sources(output / 'sources.zip', config={}, entrypoint=__name__,
        inputs=(root / 'scenario.json', root / f'decision-{index:04d}.json', root / case['event']['snapshot_file']))
    (output / 'report.json').write_text(json.dumps(report, default=str, indent=2, allow_nan=False))
    print(json.dumps({k: report[k] for k in ('search_seconds', 'rebound_original', 'selected',
        'hard_schedule', 'training_metrics')}, default=str, indent=2), flush=True)
    return report


def asdict_ref(row):
    role, miles, demand = row
    return dict(role=role, miles=miles, demand=[asdict(q) for q in demand])


if __name__ == '__main__':
    p = ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--index', type=int, default=1)
    p.add_argument('--evaluations', type=int, default=10000)
    p.add_argument('--first-purpose', choices=('sustained', 'faster'), default='sustained')
    p.add_argument('--demand-interaction',choices=('independent_v1','running_work_shared_v1'),default='independent_v1')
    a = p.parse_args()
    run(a.root, a.output, index=a.index, evaluations=a.evaluations, first_purpose=a.first_purpose,
        demand_interaction=a.demand_interaction)
