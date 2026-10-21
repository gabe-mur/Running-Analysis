"""Compare complete saved search traces, ignoring only measured runtimes.

Does not sort candidate arrays or round scores. Any candidate-order, score,
prescription, decision-time or search-budget difference remains visible.
"""
from argparse import ArgumentParser
import json
from pathlib import Path

from scripts.audit_backward_controls import digest, plain


TIMINGS = frozenset({'seconds', 'planning_seconds', 'planner_timing_seconds',
    'acceptance_fact_seconds', 'acceptance_seconds', 'base_seconds', 'coaching_seconds',
    'insertion_seconds', 'neighbor_seconds', 'projection_seconds'})


def normalized(value):
    if isinstance(value, dict):
        return {k:normalized(v) for k,v in value.items() if k not in TIMINGS}
    if isinstance(value, list):
        return [normalized(v) for v in value]
    return value


def compare(reports, *, decisions=3):
    if len(reports) < 2 or decisions < 1:
        raise ValueError('Require at least two reports and a positive decision count')
    panels = {}
    for label, report in reports.items():
        if report['stopped_early'] or report['failures'] or len(report['replans']) < decisions:
            raise ValueError('Require completed passing decisions')
        sandbox = report['synthetic_sandbox']
        panel = dict(config=sandbox['config'], opening_state=sandbox['opening_state'],
            mode=report['mode'], execution_plan=sandbox['execution_plan'],
            refinement=sandbox['full_score_refinement'],
            decisions=report['replans'][:decisions],
            calendar_traces=sandbox['calendar_neighbor_prototype'][:decisions])
        panels[label] = normalized(panel)
    first = next(iter(panels.values()))
    rows = []
    for label, panel in panels.items():
        rows.append(dict(label=label, exact_match=panel==first, normalized_hash=digest(panel),
            decisions=[dict(index=i,
                frame_and_trace_match=panel['decisions'][i]==first['decisions'][i],
                calendar_trace_match=panel['calendar_traces'][i]==first['calendar_traces'][i],
                candidate_count=len(d['planner_diagnostics']['continuation_selection']['finalists']),
                full_fingerprint=d['planner_diagnostics']['continuation_selection']['selected_program_fingerprint'],
                score=d['planner_diagnostics']['continuation_selection']['search_cost'])
                for i,d in enumerate(panel['decisions'])]))
    return dict(all_match=all(r['exact_match'] for r in rows), comparisons=rows,
        ignored_fields=sorted(TIMINGS), scope=__doc__)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--reports', nargs='+', type=Path, required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        parser.error('Require a new output path')
    result = compare({str(p):json.loads(p.read_text()) for p in args.reports})
    encoded=json.dumps(plain(result),indent=2,allow_nan=False)
    with args.output_json.open('x') as stream:
        stream.write(encoded)
    print(encoded)
    if not result['all_match']:
        raise SystemExit('Search repeatability mismatch; investigate before changing planner')


if __name__ == '__main__':
    main()
