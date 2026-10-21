"""Bounded saved-frame attribution; no search, publication or model changes."""
from argparse import ArgumentParser
from dataclasses import replace
from datetime import date, datetime, time, timedelta
import json
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

from scripts.replay_adaptive_search import load_decision
from scripts.research_manifest import freeze_sources
from run_analysis.config import load_config
from run_analysis.adaptive.continuous_search_adapter import fingerprint
from run_analysis.adaptive.decision_snapshot import evaluate_snapshot
from run_analysis.adaptive.new_model_daily_harness import planned_accounted
from run_analysis.adaptive.program_objective_trial import PURPOSES
from run_analysis.adaptive.stage_calendar_seeds import complete_stage_seeds
from run_analysis.continuous_training_dose import ContinuousDoseReference


def splice_program(prefix, suffix, cut):
    """One unprivileged candidate, not a prefix lock or continuity preference."""
    result = tuple(sorted([tuple(p) for p in prefix if p[0] < cut] +
                          [tuple(p) for p in suffix if p[0] >= cut]))
    if len({d for d, _ in result}) != len(result):
        raise ValueError('One session per date required')
    return result


def fixed_taper_guard(guard, elapsed_days):
    """Diagnostic only: offset the moving taper anchor, not fatigue parameters."""
    if type(elapsed_days) is not int or elapsed_days < 0 or guard - elapsed_days < 1:
        raise ValueError('Positive retained guard required for this bounded control')
    return guard - elapsed_days


def seed_library(case, limit):
    path = case['event']['stage_runtime']['target_history'][-1]
    return complete_stage_seeds(case['dates'],
        {k: case['catalog'][k] for k in case['handoffs']},
        replace(case['state'].context.policy, weekly_miles=path['volume_goal']),
        supporting_miles=path['support_goal'], limit=limit)


def score(case, program, context=None):
    state = case['state'] if context is None else replace(case['state'], context=context)
    reference = ContinuousDoseReference('sandbox upload accounting')
    physical = tuple(planned_accounted(f'candidate:{d}:{k}',
        datetime.combine(date.fromisoformat(d), time(18), ZoneInfo(state.timezone)),
        case['catalog'][k], case['handoffs'][k], reference, available=state.as_of)
        for d, k in program)
    result = evaluate_snapshot(state, physical, tuple(k for _, k in program), reference=reference)
    evaluation = result['evaluation']
    parts = evaluation.parts
    weights = dict(zip(PURPOSES, state.context.policy.purpose_weights))
    review = result.get('progression_delivery', {})
    return dict(eligible=result['decision_eligible'], cost=evaluation.total_cost,
        weighted_demand=evaluation.weighted_demand_cost, delivery=evaluation.delivery_cost,
        short=evaluation.short_run_cost,
        weighted_purpose_losses={k: v * weights[k] for k, v in parts['shortfall_by_purpose'].items()},
        linear_volume=parts['volume_benefit']['added_volume_cost'],
        expected_credit=review.get('expected_credited_miles'),
        required_credit=review.get('required_miles'), funded=review.get('funded'))


def search_trace(event):
    refinement = event['refinement']
    phases = {}
    for name in ('pre_calendar_distance', 'calendar', 'post_calendar_distance'):
        part = refinement[name]
        lanes = part['lanes']
        phases[name] = dict(budget=part['budget'], lanes=len(lanes),
            lanes_with_priced_neighbors=sum(any(s.get('neighbors', 0) for s in lane.get('sweeps', ()))
                or any(s.get('eligible', 0) for s in lane.get('neighborhoods', ())) for lane in lanes))
        if name == 'calendar':
            phases[name]['details'] = [dict(seed=lane['seed_fingerprint'], path=lane['path'],
                status=lane['status'], neighborhoods=lane['neighborhoods']) for lane in lanes]
    return dict(as_of=event['as_of'], cost=event['cost'], prior_cost=event['prior_cost'],
        phases=phases, source_evaluations=refinement['global_new_by_source'],
        joint=[{k: s[k] for k in ('cost', 'best_cost', 'phase', 'evaluations', 'scan_complete')}
               for s in refinement['final_joint']],
        local_complete=event['local_neighborhood_complete'], budget_exhausted=event['budget_exhausted'])


def audit(root, before_index=22, after_index=23):
    started = perf_counter()
    before = load_decision(root, before_index)
    after = load_decision(root, after_index)
    prior = tuple(map(tuple, before['event']['program']))
    selected = tuple(map(tuple, after['event']['program']))
    old_end = before['event']['end']
    shared = tuple(p for p in selected if p[0] < old_end)
    a, b = before['state'].context, after['state'].context
    elapsed = (b.decision - a.decision).days
    clock = replace(a, decision=b.decision, hard_end=b.hard_end,
        continuation_guard_days=fixed_taper_guard(a.continuation_guard_days, elapsed))
    assert clock.taper_start == a.taper_start and clock.end == a.end
    taper = replace(clock, continuation_guard_days=a.continuation_guard_days)
    comparisons = []
    # Use the ORIGINAL snapshot clock for the original control. The accounting
    # adapter derives decision from as_of; changing context.decision alone is
    # not an adequate clock ablation.
    for name, case, context in (('original', before, a),
            ('clock_only_fixed_scored_dates_and_taper', after, clock),
            ('shift_taper_keep_old_end', after, taper), ('full_rollover', after, b)):
        p, n = score(case, prior, context), score(case, shared, context)
        comparisons.append(dict(name=name, as_of=case['event']['as_of'],
            decision=str(context.decision), hard_end=str(context.hard_end), end=str(context.end),
            taper_start=str(context.taper_start), diagnostic_guard=context.continuation_guard_days,
            prior=p, new_calendar_shared_dates=n, improvement=p['cost'] - n['cost']))
    assert abs(comparisons[0]['prior']['cost'] - before['event']['cost']) < 1e-12
    assert abs(comparisons[1]['prior']['cost'] - comparisons[0]['prior']['cost']) < 1e-12
    assert abs(comparisons[1]['new_calendar_shared_dates']['cost'] -
               comparisons[0]['new_calendar_shared_dates']['cost']) < 1e-12
    selected_score = score(after, selected)
    assert abs(selected_score['cost'] - after['event']['cost']) < 1e-12

    splices = []
    for cut in ((b.decision + timedelta(days=7)).isoformat(), b.hard_end.isoformat(),
                a.taper_start.isoformat()):
        program = splice_program(prior, selected, cut)
        checked = score(after, program)
        assert tuple(p for p in program if p[0] < cut) == tuple(p for p in prior if p[0] < cut)
        splices.append(dict(cut=cut, prior_prefix_unchanged=True, program=program, score=checked,
            improvement_over_selected=selected_score['cost'] - checked['cost']))

    pools = [seed_library(c, 168) for c in (before, after)]
    full_before = seed_library(before, 10000)
    old_pool = set(pools[0])
    seed_rows = []
    for label, case, pool in zip(('before', 'after'), (before, after), pools):
        rows = []
        for index, program in enumerate(pool):
            value = case['evaluate'](program)
            rows.append(dict(index=index, fingerprint=fingerprint(program), **value,
                opening=[p for p in program if p[0] < '2026-10-20']))
        seed_rows.append(dict(frame=label, count=len(pool),
            eligible=sum(r['eligible'] for r in rows),
            ranked_eligible=sorted((r for r in rows if r['eligible']), key=lambda r: (r['cost'], r['fingerprint'])),
            rows=rows))
    transferred = []
    for index, program in enumerate(pools[1]):
        clipped = tuple(p for p in program if p[0] < old_end)
        value = before['evaluate'](clipped)
        transferred.append(dict(after_index=index, was_supplied_before=clipped in old_pool,
            positions_in_full_before=[i for i, p in enumerate(full_before) if p == clipped],
            fingerprint=fingerprint(clipped), program=clipped, **value))
    eligible = sorted((r for r in transferred if r['eligible']), key=lambda r: (r['cost'], r['fingerprint']))
    return dict(status='bounded_saved_frame_diagnosis', original_winners_reproduced=True,
        no_search_or_publication=True, no_production_changes=True, root=str(root),
        before_index=before_index, after_index=after_index,
        unchanged=dict(history_keys=before['event']['history_keys'] == after['event']['history_keys'],
            history_cardio=before['event']['history_cardio_bounds'] == after['event']['history_cardio_bounds'],
            forecasts=before['event']['forecast_load_by_dose'] == after['event']['forecast_load_by_dose']),
        score_controls=comparisons, selected=selected_score, splices=splices,
        seed_analysis=dict(cap=168, complete_before_library_size=len(full_before),
            pools=seed_rows, after_seeds_under_before_inputs=transferred,
            best_after_seed_under_before_inputs=eligible[0]),
        search_traces=[search_trace(c['event']) for c in (before, after)], seconds=perf_counter() - started,
        limitations=['Six hundred or fewer prescribed candidate reprices, not a new optimizer or global proof',
            'Clock/taper/end ablations are order-dependent diagnostics, not new recommended policies',
            'A cheaper stable splice is one candidate; future daily stability is not demonstrated',
            'Original transformed-HR and conditional-recovery acceptance limits remain'])


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=False)
    inputs = tuple(args.root / name for name in ('decision-0022.json', 'snapshot-0022.json',
        'decision-0023.json', 'snapshot-0023.json', 'scenario.json', 'resume-0001.summary.json',
        'resume-0001.sources.zip'))
    manifest = freeze_sources(args.output_dir / 'sources.zip', config=load_config(),
        entrypoint=__name__, inputs=inputs)
    result = audit(args.root)
    result['manifest'] = manifest
    with (args.output_dir / 'report.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(seconds=result['seconds'],
        seed=result['seed_analysis']['best_after_seed_under_before_inputs'],
        splices=[{k: v for k, v in r.items() if k != 'program'} for r in result['splices']],
        controls=result['score_controls'])))
