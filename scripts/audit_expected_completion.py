"""Same-time conditional-vs-realized state audit on the saved compliant upload."""
from argparse import ArgumentParser
from datetime import timedelta
import json
from pathlib import Path
from time import perf_counter

from scripts.audit_continuation_upload import reconstruct, raw_state_lookup, hard_score
from scripts.expected_completion import ExpectedCompletion
from run_analysis import weekly_schedule as planner, adherence_projection as projection
from run_analysis.selection_objective import EXPECTED_COMPLETION_STATE, shared_objective_policy
from run_analysis.prescription_distance import canonical_distance_policy
from scripts.finalized_scoring_trial import finalized_scoring_policy


def differences(left, right, prefix=''):
    result = {}
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(left.keys() | right.keys()):
            result.update(differences(left.get(key), right.get(key), f'{prefix}.{key}' if prefix else key))
    elif left != right:
        result[prefix] = dict(expected=left, realized=right)
    return result


def investigate(report):
    captured = []
    def observe(index, inputs, days, evidence):
        if index >= len(report['replans']) - 2:
            captured.append((inputs, days, evidence))
    reconstruct(report, observe)
    before, after = captured
    at = after[0]['daily_states'][0].as_of
    consumed = [d.recommendation for d in before[1] if d.recommendation and d.recommendation.planned_for < at]
    expected = ExpectedCompletion(before[2]['template'], before[2]['runs'], before[0]['config'])
    realized = ExpectedCompletion(after[2]['template'], after[2]['runs'], after[0]['config'])
    raw = raw_state_lookup(before[0], before[2])(at)
    legacy = planner._project_state(raw, consumed, before[0]['config'])
    actual = after[0]['daily_states'][0]
    conditional = expected.at(at, consumed)
    result = dict(at=at.isoformat(), legacy_state_differences=differences(
        legacy.model_dump(mode='json'), actual.model_dump(mode='json')),
        conditional_state_differences=differences(conditional.model_dump(mode='json'), actual.model_dump(mode='json')),
        target=dict(expected=expected.target_at(at, consumed), realized=after[0]['target_distance_range']),
        future_queries=[])
    for offset in (0, 2, 7, 20, 28):
        future = at + timedelta(days=offset)
        x, y = expected.at(future, consumed), realized.at(future)
        result['future_queries'].append(dict(at=future.isoformat(),
            differences=differences(x.model_dump(mode='json'), y.model_dump(mode='json')),
            expected_target=expected.target_at(future, consumed), realized_target=realized.target_at(future)))
    assert not result['conditional_state_differences'], result['conditional_state_differences']
    assert result['target']['expected'] == result['target']['realized']
    assert all(not row['differences'] and row['expected_target'] == row['realized_target'] for row in result['future_queries'])
    # At the identical next decision, build every raw timed input independently
    # from the two ledgers, then price the identical saved hard candidate.
    scored = []
    conditional_history = expected.ledger(at, consumed)
    for ledger in (conditional_history, after[2]['runs']):
        model = ExpectedCompletion(after[2]['template'], ledger, after[0]['config'])
        capacity = model.at(at).recent_load.capacity_reference_miles
        coaching = after[0]['config'].get('coaching', {})
        def rebuild(raw):
            return projection._state_at(after[2]['template'], ledger, raw.as_of, capacity,
                easy_baseline_half_life_days=float(coaching.get('capacity_retention_half_life_days', 84)),
                fatigue_half_life_days=float(coaching.get('continuous_fatigue_half_life_days', 7)))
        inputs = dict(after[0], daily_states=[rebuild(s) for s in after[0]['daily_states']],
            daily_state_options=[[rebuild(s) for s in choices] for choices in after[0]['daily_state_options']])
        assert inputs['daily_states'] == after[0]['daily_states']
        assert inputs['daily_state_options'] == after[0]['daily_state_options']
        token = EXPECTED_COMPLETION_STATE.set(model)
        try:
            with canonical_distance_policy(), shared_objective_policy(), finalized_scoring_policy():
                offsets = [i for i,d in enumerate(after[1]) if d.recommendation]
                _, score, _ = hard_score(inputs, offsets, days=after[1])
                scored.append(score.model_dump(mode='json'))
        finally:
            EXPECTED_COMPLETION_STATE.reset(token)
    result['same_future_candidate'] = dict(expected=scored[0], realized=scored[1],
        differences=differences(*scored))
    assert not result['same_future_candidate']['differences']
    return result


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        parser.error('Choose a new output path')
    started = perf_counter()
    result = investigate(json.loads(args.report.read_text()))
    result['seconds'] = perf_counter() - started
    with args.output_json.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
