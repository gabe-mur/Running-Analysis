"""Run integrated full-objective refinement on saved failing rollover inputs.

This is a frozen-input search comparison, not a sequential stability replay.
"""
from argparse import ArgumentParser
import json
from pathlib import Path
from time import perf_counter

from scripts.audit_continuation_upload import reconstruct
from scripts.audit_regrowth_gates import check_candidate
from scripts.continuation_replan import select_program, signature
from scripts.full_score_refinement import full_score_refinement_policy
from scripts.suffix_regrowth_prototype import regrowth_finalist_policy


def investigate(report, emit, *, budget=240, fixed_tail=False, indices=None):
    results = []
    def observe(index, inputs, days, evidence):
        if (index not in indices if indices is not None else index < len(report['replans']) - 2):
            return
        snapshot = report['replans'][index]
        saved = snapshot['planner_diagnostics']['continuation_selection']
        record = report['synthetic_sandbox']['calendar_neighbor_prototype'][index]
        record = dict(record, evaluated=record['evaluated']+saved.get('suffix_regrowth', {}).get('rows', []))
        # Start with the previously published full-score winner. Include the
        # same lane-preserving proposal shortlist; no previous map scores used.
        started = perf_counter()
        with full_score_refinement_policy(budget=budget, starts=2 if fixed_tail else 3,
                                          fixed_tail=fixed_tail), regrowth_finalist_policy():
            result = select_program(days, None, inputs, record, limit=6, continuation_days=28)
        item = dict(index=index, generated_at=snapshot['generated_at'],
                    before_cost=saved['search_cost'], after_cost=result['winner'].total_cost,
                    before=signature(days), after=signature(result['planning_days']),
                    gates=check_candidate(snapshot, inputs, result['planning_days']),
                    evidence=result['evidence'], seconds=perf_counter()-started)
        results.append(item)
        emit(item)
        print('Refined', index, item['before_cost'], item['after_cost'],
              'gates', item['gates']['failures'], flush=True)
    reconstruct(report, observe)
    return dict(scope=__doc__, decisions=results)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--budget', type=int)
    parser.add_argument('--fixed-tail', action='store_true')
    parser.add_argument('--indices', type=int, nargs='+')
    args = parser.parse_args()
    if args.budget is None:
        args.budget = 80 if args.fixed_tail else 240
    if args.output_json.exists():
        parser.error('Require a new output path')
    with args.output_json.with_suffix('.jsonl').open('x') as journal:
        def emit(item):
            journal.write(json.dumps(item, default=str)+'\n')
            journal.flush()
        result = investigate(json.loads(args.report.read_text()), emit, budget=args.budget,
                             fixed_tail=args.fixed_tail, indices=args.indices)
    with args.output_json.open('x') as stream:
        json.dump(result, stream, indent=2, default=str)


if __name__ == '__main__':
    main()
