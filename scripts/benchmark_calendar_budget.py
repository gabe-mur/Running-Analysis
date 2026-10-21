"""Matched-input fixed-tail budget frontier; no planner-policy changes.

Replay saved inputs, reset conditional caches for every variant, and change only
the additional calendar/tail evaluation budget. Timings cover selection, not the
saved beam/proposal work. Cross-budget prescription differences are NOT daily
churn: actual sequential validation is a separate experiment.
"""
from argparse import ArgumentParser
from contextlib import nullcontext
from datetime import date, timedelta
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

from scripts.audit_continuation_upload import reconstruct, unpack_soft, evaluate_physical
from scripts.audit_regrowth_gates import check_candidate
from scripts.continuation_replan import select_program, signature, fingerprint
from scripts.expected_completion import expected_completion_policy
from scripts.full_score_refinement import full_score_refinement_policy
from scripts.suffix_regrowth_prototype import regrowth_finalist_policy


def prescription_difference(before, after, *, start, horizon):
    end = start + timedelta(days=horizon)
    def mapping(rows):
        return {r['date']: r['prescription'] for r in rows
                if start <= date.fromisoformat(r['date']) < end and r['prescription']
                and r['prescription']['workout_type'] != 'rest'}
    a, b = mapping(before), mapping(after)
    distances = [dict(date=d, before=a[d]['distance_target_miles'], after=b[d]['distance_target_miles'])
        for d in sorted(a.keys() & b.keys()) if a[d]['workout_type'] == b[d]['workout_type']
        and a[d]['distance_target_miles'] != b[d]['distance_target_miles']]
    return dict(date_changes=sorted(a.keys() ^ b.keys()),
        type_changes=[d for d in sorted(a.keys() & b.keys()) if a[d]['workout_type'] != b[d]['workout_type']],
        distance_changes=distances)


def verify_comparisons(inputs, snapshot, evidence, days, comparisons, index):
    verified = []
    for label, report in comparisons.items():
        saved = next((r for r in report['decisions'] if r['index'] == index), None)
        if saved is None:
            continue
        if saved['before'] != signature(days):
            raise ValueError('Comparison was produced from a different published input')
        hard = unpack_soft(saved['after'])
        full = hard + unpack_soft(saved['evidence']['soft_sessions'])
        if fingerprint(full) != saved['evidence']['selected_program_fingerprint']:
            raise ValueError('Saved comparison full-program fingerprint mismatch')
        with expected_completion_policy(inputs['config']):
            actual = evaluate_physical(inputs, hard, evidence, full=full)['continuation']
        if abs(actual['combined_cost'] - saved['after_cost']) > 1e-8:
            raise ValueError('Saved comparison full-program score mismatch')
        guards = actual['fixed_sequence_checks']
        if not guards['checks_pass']:
            raise ValueError('Saved comparison failed downstream guards')
        gates = check_candidate(snapshot, inputs, hard)
        if gates['failures'] or gates['whole_hard_peak_failures'] or gates['strict_cadence_findings']:
            raise ValueError('Saved comparison failed hard-plan gates')
        verified.append(dict(label=label, cost=actual['combined_cost'],
            hard_fingerprint=fingerprint(hard), full_fingerprint=fingerprint(full),
            guards_pass=True, gates=gates))
    return verified


def investigate(report, emit, *, budgets, indices, comparisons=None):
    if not budgets or any(type(n) is not int or n < 0 for n in budgets) or len(set(budgets)) != len(budgets):
        raise ValueError('Require distinct nonnegative integer budgets')
    if not indices or any(i < 0 or i >= len(report['replans']) for i in indices):
        raise ValueError('Require valid saved decision indices')
    results, verification = [], []
    def observe(index, inputs, days, evidence):
        if index not in indices:
            return
        snapshot = report['replans'][index]
        saved = snapshot['planner_diagnostics']['continuation_selection']
        base = report['synthetic_sandbox']['calendar_neighbor_prototype'][index]
        record = dict(base, evaluated=base['evaluated'] + saved.get('suffix_regrowth', {}).get('rows', []))
        checked = verify_comparisons(inputs, snapshot, evidence, days, comparisons or {}, index)
        verification.append(dict(index=index, comparisons=checked))
        # Check publication identity AND record the common immutable inputs.
        frozen = dict(config=inputs['config'], seed=signature(days),
            prior_dates=sorted(inputs['prior_run_offsets']),
            states=[s.model_dump(mode='json') for s in inputs['daily_states']], record=record)
        input_hash = sha256(json.dumps(frozen, sort_keys=True, default=str).encode()).hexdigest()
        baseline = None
        for budget in sorted(budgets):
            with expected_completion_policy(inputs['config']) as model, regrowth_finalist_policy(), \
                    (full_score_refinement_policy(budget=budget, starts=2, fixed_tail=True) if budget else nullcontext()):
                started = perf_counter()
                result = select_program(days, None, inputs, record, limit=6, continuation_days=28)
                seconds = perf_counter() - started
                gates = check_candidate(snapshot, inputs, result['planning_days'])
            after = signature(result['planning_days'])
            if baseline is None:
                baseline = after
            rows = [r for r in result['evidence']['finalists'] if 'combined_cost' in r]
            pre_refresh = [r for r in rows if r['source'] != 'full_score_neighbor' or 'fixed_tail_parent' in r]
            item = dict(index=index, generated_at=snapshot['generated_at'], budget=budget,
                frozen_input_hash=input_hash, seconds=seconds,
                timing_scope='selection only; saved proposals reused; fresh conditional cache per variant',
                selected_cost=result['winner'].total_cost, independent_cost=result['independent'].total_cost,
                stable_cost=result['stable'].total_cost if result['stable'] is not None else None,
                stable_selected=result['stable_selected'], tolerance=result['tolerance'],
                pre_tail_refresh_best=min(r['combined_cost'] for r in pre_refresh),
                after=after, gates=gates, evidence=result['evidence'],
                expected_state_rebuilds=model.rebuilds,
                difference_from_lowest_budget={str(n):prescription_difference(baseline, after,
                    start=inputs['daily_states'][0].as_of.date(), horizon=n) for n in (4,7)})
            results.append(item)
            emit(item)
            print(f"Budget panel {index} / {budget}: {seconds:.2f}s, selected {item['selected_cost']:.6f}, "
                  f"independent {item['independent_cost']:.6f}", flush=True)
            if gates['failures'] or gates['whole_hard_peak_failures'] or gates['strict_cadence_findings']:
                raise ValueError('Budget variant failed a plan check; evidence saved, stop')
    reconstruct(report, observe)
    return dict(scope=__doc__, budgets=sorted(budgets), indices=indices,
                comparisons=verification, decisions=results)


def cross_replays(baseline, alternative):
    """Price both saved physical programs under each decision's own prior inputs."""
    if (baseline['synthetic_sandbox']['config'] != alternative['synthetic_sandbox']['config']
            or baseline['realized_execution'] != alternative['realized_execution']
            or len(baseline['replans']) != len(alternative['replans'])):
        raise ValueError('Cross-replay panel requires matched config and completed execution')
    reports = {'baseline': baseline, 'alternative': alternative}
    rows = []
    for frame, report in reports.items():
        def observe(index, inputs, days, evidence):
            snapshot = report['replans'][index]
            entries = []
            for label, source in reports.items():
                other = source['replans'][index]
                if other['generated_at'] != snapshot['generated_at']:
                    raise ValueError('Cross-replay decisions are not aligned')
                saved = other['planner_diagnostics']['continuation_selection']
                # Hard prescriptions are saved in the published schedule's
                # fingerprint but not as complete objects in snapshots. Use
                # the source reconstruction collected in the first pass below.
                hard = hard_programs[label][index]
                full = hard + unpack_soft(saved['soft_sessions'])
                if fingerprint(full) != saved['selected_program_fingerprint']:
                    raise ValueError('Cross-replay program fingerprint mismatch')
                with expected_completion_policy(inputs['config']):
                    scored = evaluate_physical(inputs, hard, evidence, full=full)['continuation']
                if label == frame and abs(scored['combined_cost'] - saved['search_cost']) > 1e-8:
                    raise ValueError('Own-input score did not reproduce')
                entries.append(dict(program=label, cost=scored['combined_cost'],
                    guards_pass=scored['fixed_sequence_checks']['checks_pass'],
                    gates=check_candidate(snapshot,inputs,hard)))
            rows.append(dict(frame=frame,index=index,generated_at=snapshot['generated_at'],programs=entries))
        # Collect exact hard objects under each source's original context first;
        # never silently rematerialize an alternate calendar under this frame.
        if frame == 'baseline':
            hard_programs = {}
            for label, source in reports.items():
                hard_programs[label] = {}
                reconstruct(source, lambda index,inputs,days,evidence: hard_programs[label].update({index:days}))
        reconstruct(report, observe)
    return dict(scope='Saved physical programs scored under both prior-plan input frames; no search.',rows=rows)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--budgets', type=int, nargs='+', default=[0,40,80,160])
    parser.add_argument('--indices', type=int, nargs='+', default=[0,8,9])
    parser.add_argument('--expensive-comparison', type=Path)
    parser.add_argument('--fast-comparison', type=Path)
    parser.add_argument('--cross-variant', type=Path,
                        help='Instead of searching, cross-score this completed replay against --report')
    args = parser.parse_args()
    if args.output_json.exists():
        parser.error('Require a new output path')
    if args.cross_variant:
        result = cross_replays(json.loads(args.report.read_text()), json.loads(args.cross_variant.read_text()))
        with args.output_json.open('x') as stream:
            json.dump(result,stream,indent=2,default=str)
        return
    comparisons = {label:json.loads(path.read_text()) for label,path in (
        ('expensive',args.expensive_comparison), ('fast',args.fast_comparison)) if path is not None}
    with args.output_json.with_suffix('.jsonl').open('x') as journal:
        def emit(item):
            journal.write(json.dumps(item, default=str)+'\n')
            journal.flush()
        result = investigate(json.loads(args.report.read_text()), emit,
            budgets=args.budgets, indices=args.indices, comparisons=comparisons)
    with args.output_json.open('x') as stream:
        json.dump(result,stream,indent=2,default=str)


if __name__ == '__main__':
    main()
