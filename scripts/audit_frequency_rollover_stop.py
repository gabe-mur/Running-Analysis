"""Bounded repricing of the first v2 frequency rollover stop; no search or writes to live state."""
from argparse import ArgumentParser
from dataclasses import replace
from datetime import date, datetime, time
import json
from pathlib import Path
from zoneinfo import ZoneInfo
from time import perf_counter
from dataclasses import fields
from itertools import combinations

from scripts.replay_adaptive_search import load_decision
from scripts.research_manifest import freeze_sources
from run_analysis.config import load_config
from run_analysis.adaptive.decision_snapshot import evaluate_snapshot
from run_analysis.adaptive.new_model_daily_harness import planned_accounted
from run_analysis.continuous_training_dose import ContinuousDoseReference


def audit(root, output, *, investigate=False):
    started = perf_counter()
    before, after = (load_decision(root, i) for i in (1, 2))
    assert before['state'].history == after['state'].history
    assert before['state'].responses == after['state'].responses
    old = tuple(map(tuple, before['event']['program']))
    new = tuple(map(tuple, after['event']['program']))
    assert not any(d < after['dates'][0] for d, _ in old)
    reference = ContinuousDoseReference('sandbox upload accounting')

    cache = {}
    def price(case, program):
        program = tuple(sorted(program))
        identity = (case['state'].fingerprint, program, case['event']['index'])
        if identity in cache:
            return cache[identity]
        state = case['state']
        physical = tuple(planned_accounted(f'candidate:{d}:{k}',
            datetime.combine(date.fromisoformat(d), time(18), ZoneInfo(state.timezone)),
            case['catalog'][k], case['handoffs'][k], reference, available=state.as_of)
            for d, k in program)
        result = evaluate_snapshot(state, physical, tuple(k for _, k in program), reference=reference)
        raw = evaluate_snapshot(replace(state, frequency_contract=None), physical,
            tuple(k for _, k in program), reference=reference)
        e = result['evaluation']
        value = dict(cost=e.total_cost, eligible=result['decision_eligible'],
            raw_eligible=raw['decision_eligible'], raw_cost=raw['evaluation'].total_cost,
            fatigue=e.weighted_demand_cost, delivery=e.delivery_cost, short=e.short_run_cost,
            frequency=result['frequency_range'], preference=result['frequency_preference'],
            funding=result.get('progression_delivery'), suitability=result['suitability']['status'])
        cache[identity] = value
        return value

    rows = []
    for name, case, program in (
        ('original_winner_original_state', before, old),
        ('original_winner_rollover_state_no_new_tail', after, old),
        ('rollover_winner_rollover_state', after, new),
        ('rollover_winner_original_state_NEW_END_CLIPPED', before,
            tuple(p for p in new if p[0] < before['event']['end'])),
    ):
        rows.append(dict(name=name, program=program, score=price(case, program)))
    # Whole-horizon witnesses retaining the original visible seven-day prescription.
    # The tail is explicitly the new winner's, not an assertion of unchanged accounting.
    for cut in ('2026-10-07', after['event']['hard_end']):
        program = tuple(p for p in old if p[0] < cut) + tuple(p for p in new if p[0] >= cut)
        rows.append(dict(name='old_prefix_new_tail_'+cut, program=program, score=price(after, program)))
    assert abs(rows[0]['score']['cost'] - before['archived_cost']) < 1e-12
    assert abs(rows[2]['score']['cost'] - after['archived_cost']) < 1e-12
    detail = None
    if investigate:
        # No search: only explicit single quality moves and single tail removals.
        # Same dose, no RNG and no automatic choice of favorable frames/weights.
        controls = []
        old_quality = next((d, k) for d, k in old if before['catalog'][k].role == 'quality')
        new_quality = next((d, k) for d, k in new if after['catalog'][k].role == 'quality')
        assert old_quality[1] == new_quality[1]
        def move(program, source, destination):
            assert destination not in dict(program)
            return tuple(sorted((destination if d == source else d, k) for d, k in program))
        old_moved = move(old, old_quality[0], new_quality[0])
        new_restored = move(new, new_quality[0], old_quality[0])
        common_new = tuple(p for p in new if p[0] < before['event']['end'])
        common_restored = tuple(p for p in new_restored if p[0] < before['event']['end'])
        for name, case, program in (
            ('old_first_quality_moved_original_frame', before, old_moved),
            ('old_first_quality_moved_rollover_frame', after, old_moved),
            ('new_first_quality_restored_rollover_frame', after, new_restored),
            ('common_new_original_frame', before, common_new),
            ('common_new_rollover_frame', after, common_new),
            ('common_new_quality_restored_original_frame', before, common_restored),
            ('common_new_quality_restored_rollover_frame', after, common_restored),
        ):
            controls.append(dict(name=name, program=program, score=price(case, program)))
        drops = []
        for removed in old:
            if removed[0] < before['event']['hard_end'] or before['catalog'][removed[1]].role != 'easy':
                continue
            program = tuple(p for p in old if p != removed)
            # Keep the SAME finite ledger at both origins, no appended sessions.
            drops.append(dict(removed=removed, program=program,
                original=price(before, program), rollover=price(after, program)))
        weights_before = dict(before['state'].frequency_exposure.weights)
        weights_after = dict(after['state'].frequency_exposure.weights)
        weights = [dict(day=d, key=k, before=weights_before[date.fromisoformat(d)],
            after=weights_after[date.fromisoformat(d)],
            delta=weights_after[date.fromisoformat(d)]-weights_before[date.fromisoformat(d)])
            for d, k in old if weights_after[date.fromisoformat(d)] != weights_before[date.fromisoformat(d)]]
        detail = dict(controls=controls, single_tail_easy_removals=drops, moving_weights=weights,
            changed_snapshot_fields=[f.name for f in fields(before['state'])
                if getattr(before['state'], f.name) != getattr(after['state'], f.name)],
            forecast_loads_identical=before['event']['forecast_load_by_dose'] == after['event']['forecast_load_by_dose'],
            prior_revalidation=after['event']['prior_revalidation'],
            continuity=after['event']['continuity'],
            scope='One-at-a-time physical counterfactuals and exact shared-ledger two-origin comparisons; not optimum proof. Single removals are not volume-matched.')
        # Equal-mileage counterexamples using only adjacent tail easies. Bound
        # the panel before evaluating; no descent/repeated optimizer or retuning.
        tail_easy = [p for p in old if p[0] >= before['event']['hard_end']
            and before['catalog'][p[1]].role == 'easy']
        merge_rows = []
        append_day = before['event']['end']
        assert append_day in after['dates'] and append_day not in dict(old)
        long_key = next(k for _, k in old if before['catalog'][k].role == 'long')
        controls.append(dict(name='old_ledger_plus_new_end_long_rollover_frame',
            program=tuple(sorted((*old, (append_day, long_key)))),
            score=price(after, tuple(sorted((*old, (append_day, long_key)))))))
        for first, second in combinations(tail_easy, 2):
            if (date.fromisoformat(second[0])-date.fromisoformat(first[0])).days > 2:
                continue
            miles = sum(before['catalog'][p[1]].miles for p in (first, second))
            options = [k for k in before['handoffs'] if before['catalog'][k].role == 'easy'
                and abs(before['catalog'][k].miles-miles) < 1e-9 and k in after['handoffs']]
            if not options:
                continue
            for day in (first[0], second[0]):
                program = tuple(sorted([p for p in old if p not in (first, second)] + [(day, min(options))]))
                extended = tuple(sorted((*program, (append_day, long_key))))
                merge_rows.append(dict(merged=(first, second), kept_day=day,
                    program=program, original=price(before, program),
                    rollover=price(after, program),
                    extended_program=extended, extended_rollover=price(after, extended)))
        detail['equal_volume_tail_merges'] = merge_rows
        detail['context_changed_fields'] = [f.name for f in fields(before['state'].context)
            if getattr(before['state'].context, f.name) != getattr(after['state'].context, f.name)]
    output.mkdir(exist_ok=False)
    manifest = freeze_sources(output/'sources.zip', config=load_config(), entrypoint=__name__,
        inputs=(root/'decision-0001.json', root/'decision-0002.json',
            root/before['event']['snapshot_file'], root/after['event']['snapshot_file']))
    report = dict(status='bounded_rollover_attribution', manifest=manifest, rows=rows,
        investigation=detail, reprices=len(cache), seconds=perf_counter()-started,
        identical_physical_history_and_responses=True, live_unchanged=True,
        scope='Finite saved-frame reprices; clipped calendars, tail removals, extensions and prefix splices explicitly labelled; no new search, model change or daily pass')
    with (output/'report.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps([dict(name=r['name'], **{k:r['score'][k] for k in
        ('cost', 'eligible', 'raw_eligible', 'raw_cost', 'fatigue')}) for r in rows]))


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--investigate', action='store_true')
    args = parser.parse_args()
    audit(args.root, args.output, investigate=args.investigate)
