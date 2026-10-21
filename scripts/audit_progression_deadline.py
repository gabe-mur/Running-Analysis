"""Reprice a saved easy-run move and volume-deferral controls; never search/write.

The archived winner must reproduce first. Only the prospective funding gate is
removed; target history, physiology, continuation, doses and weights stay fixed.
"""
from argparse import ArgumentParser
from dataclasses import replace
import json
from pathlib import Path

from scripts.replay_adaptive_search import load_decision


def audit(root, index, *, easy_day, alternative_days):
    case = load_decision(root, index)
    program = tuple(tuple(p) for p in case['event']['program'])
    matches = tuple(k for d, k in program if d == easy_day and case['catalog'][k].role == 'easy')
    if len(matches) != 1:
        raise ValueError('Select exactly one saved easy run')
    key, = matches
    state = replace(case['state'], prospective_volume_review=None)
    candidates = [('saved', program)]
    for day in alternative_days:
        if day not in case['dates'] or any(d == day and d != easy_day for d, _ in program):
            raise ValueError('Alternative must be an unoccupied date in the identical scoring horizon')
        candidates.append((day, tuple(sorted((day if d == easy_day else d, k) for d, k in program))))
    candidates.extend((('without_easy_runs', tuple((d, k) for d, k in program
        if case['catalog'][k].role != 'easy')), ('without_future_running', ())))
    rows = []
    for name, candidate in candidates:
        result = case['evaluate_details'](candidate, state_override=state)
        e = result['evaluation']
        rows.append(dict(candidate=name, eligible=result['decision_eligible'], cost=float(e.total_cost),
            cost_change_percent=100 * (float(e.total_cost) / case['archived_cost'] - 1),
            weighted_demand=float(e.weighted_demand_cost), delivery=float(e.delivery_cost),
            volume_loss=e.parts.get('shortfall_by_purpose', {}).get('volume')))
    baseline = case['evaluate_details'](program)['evaluation']
    unblocked = case['evaluate_details'](program, state_override=state)['evaluation']
    # Snapshot-bound identity changes deliberately; physical and numerical
    # values must not. Frequency refinement also retains a pre-refinement ID.
    parts = dict(baseline.parts)
    if 'pre_frequency_fingerprint' in parts:
        parts['pre_frequency_fingerprint'] = unblocked.parts['pre_frequency_fingerprint']
    assert replace(baseline, fingerprint=unblocked.fingerprint, parts=parts) == unblocked
    return dict(snapshot=case['state'].fingerprint, decision=case['event']['index'],
        unchanged_score_for_identical_program=True, progression_forecast_gate_removed=True,
        actual_delivery_growth_checks_unchanged=True, searches=0, publication=False, candidates=rows)


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('index', type=int, help='Archived decision filename number, not event index')
    p.add_argument('--easy-day', required=True)
    p.add_argument('--alternative-day', action='append', required=True)
    args = p.parse_args()
    print(json.dumps(audit(args.root, args.index, easy_day=args.easy_day,
        alternative_days=args.alternative_day), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
