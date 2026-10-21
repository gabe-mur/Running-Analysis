"""Reprice fixed rollover witnesses; no search, uploads or model changes."""
from argparse import ArgumentParser
from dataclasses import replace
from datetime import date, datetime, time, timedelta
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from scripts.replay_adaptive_search import load_decision
from run_analysis.adaptive.decision_snapshot import evaluate_snapshot
from run_analysis.adaptive.new_model_daily_harness import planned_accounted
from run_analysis.continuous_training_dose import ContinuousDoseReference
from run_analysis.program_delivery import _target_total


def price(case, program, *, state=None, handoffs=None):
    state = state or case['state']
    handoffs = case['handoffs'] if handoffs is None else handoffs
    reference = ContinuousDoseReference('sandbox upload accounting')
    rows = tuple(planned_accounted(f'candidate:{day}:{key}',
        datetime.combine(date.fromisoformat(day), time(18), ZoneInfo(state.timezone)),
        case['catalog'][key], handoffs[key], reference, available=state.as_of)
        for day, key in sorted(program.items()))
    result = evaluate_snapshot(state, rows, tuple(key for _, key in sorted(program.items())),
        reference=reference)
    evaluation = result['evaluation']
    return dict(cost=float(evaluation.total_cost), eligible=result['decision_eligible'],
        planned_miles=sum(r.miles for r in rows),
        delivery_cost=float(evaluation.delivery_cost),
        weighted_demand_cost=float(evaluation.weighted_demand_cost),
        short_run_cost=float(evaluation.short_run_cost),
        frequency_cost=evaluation.parts['frequency_preference']['cost'],
        purpose_losses=evaluation.parts['shortfall_by_purpose'],
        race_event_cost=evaluation.parts.get('race_event', {}).get('cost', 0.),
        linear_volume_cost=evaluation.parts.get('volume_benefit', {}).get('added_volume_cost', 0.))


def run(root, *, before, after):
    a, b = (load_decision(root, index) for index in (before, after))
    if a['state'].history != b['state'].history or a['state'].phase_history != b['state'].phase_history:
        raise ValueError('This attribution requires unchanged physical and phase history')
    if any(k not in a['handoffs'] for _, k in b['event']['program']):
        raise ValueError('Later prescription has no supported original forecast')
    day = a['state'].context.stage_start
    while day < a['state'].context.end:
        for field in ('volume_targets', 'support_targets'):
            av, bv = (getattr(c['state'].context, field) for c in (a, b))
            if abs(_target_total(av, day, day + timedelta(days=1))
                    - _target_total(bv, day, day + timedelta(days=1))) > 1e-10:
                raise ValueError('Dated targets changed within the common interval')
        day += timedelta(days=1)
    old, new = (dict(case['event']['program']) for case in (a, b))
    restored = {**{d: k for d, k in old.items() if d < b['dates'][0]},
        **{d: k for d, k in new.items() if d in a['dates']}}
    edits = b['event']['churn']['7']
    if any(e['date_changed'] or e['role_changed'] for e in edits):
        raise ValueError('Visible calendar edits require a separate attribution')
    visible = {e['day'] for e in edits}
    variants = dict(old=old, later_on_original=restored,
        later_keep_visible_doses={**restored, **{d: old[d] for d in visible}},
        old_only_visible_dose_changes={**old, **{d: new[d] for d in visible}})
    original = {name: price(a, p) for name, p in variants.items()}
    extension = {d: k for d, k in new.items() if d not in a['dates']}
    after_programs = {name: {**{d: k for d, k in p.items() if d in b['dates']}, **extension}
        for name, p in variants.items()}
    later = {name: price(b, p) for name, p in after_programs.items()}
    # Hold rate expectations AND the mean-continuation demands fixed, while
    # preserving the actual later dates, history, targets and admission contract.
    frozen_rates = replace(b['state'], context=replace(b['state'].context,
        continuation_boundary_references=a['state'].context.continuation_boundary_references))
    frozen = {name: price(b, p, state=frozen_rates, handoffs=a['handoffs'])
        for name, p in after_programs.items()}
    # Isolate evaluation-end extension under the ORIGINAL opening and physical
    # ledger. The extra dated target is needed for valid coverage, not new work.
    # evaluate_accounted recomputes decision from as_of, so do not pretend that
    # replacing context.decision alone would isolate rollover.
    end_only = replace(a['state'], context=replace(a['state'].context,
        end=b['state'].context.end, volume_targets=b['state'].context.volume_targets,
        support_targets=b['state'].context.support_targets))
    extended = {name: price(a, p, state=end_only) for name, p in variants.items()}
    shift = (b['state'].context.taper_start - a['state'].context.taper_start).days
    # This is a scoring ablation, NOT a proposed guard-policy change. With the
    # original opening frozen, changing guard length isolates its absolute
    # full-weight boundary and distributed continuation handoffs.
    taper_only = replace(a['state'], context=replace(a['state'].context,
        continuation_guard_days=a['state'].context.continuation_guard_days + shift))
    if taper_only.context.taper_start != b['state'].context.taper_start:
        raise ValueError('Guard ablation did not isolate the intended taper boundary')
    shifted = {name: price(a, p, state=taper_only) for name, p in variants.items()}
    result = dict(scope='Fixed candidate attribution, not a search, repair or global certificate',
        before=before, after=after, before_as_of=a['event']['as_of'], after_as_of=b['event']['as_of'],
        before_fingerprint=a['state'].fingerprint, after_fingerprint=b['state'].fingerprint,
        physical_history_identical=True, phase_history_identical=True,
        forecasts_identical=a['handoffs'] == b['handoffs'], visible_edits=edits,
        common_dated_targets_identical=True,
        all_common_edits=[dict(day=d, before=old.get(d), after=new.get(d))
            for d in sorted(set(old) | set(new)) if d in a['dates'] and d in b['dates']
            and old.get(d) != new.get(d)],
        newly_exposed_sessions=extension, original_objective=original,
        later_objective_with_common_tail_extension=later,
        later_objective_with_frozen_forecasts=frozen,
        original_objective_with_end_extension_only=extended,
        original_objective_with_taper_boundary_shift_only=shifted,
        original_final_pair_coverage=a['event']['refinement']['final_winner_doses'],
        cautions=['Old-visible-dose controls may change total volume; miles and admission are reported',
            'Same prescribed distance is not identical measured HR response',
            'Forecast bounds are scenarios, not confidence intervals',
            'Guard-length ablation isolates moving score coverage; it is not a policy recommendation',
            'No live server, score coefficients or search policy changed'])
    path = root / f'distance-churn-{before:04d}-{after:04d}-coverage.json'
    with path.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(path=str(path), before=before, after=after,
        original_costs={k: v['cost'] for k, v in original.items()},
        later_costs={k: v['cost'] for k, v in later.items()},
        frozen_forecast_costs={k: v['cost'] for k, v in frozen.items()},
        extended_end_costs={k: v['cost'] for k, v in extended.items()},
        shifted_taper_costs={k: v['cost'] for k, v in shifted.items()}), indent=2))
    return result


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--before', type=int, required=True)
    parser.add_argument('--after', type=int, required=True)
    args = parser.parse_args()
    run(args.root, before=args.before, after=args.after)
