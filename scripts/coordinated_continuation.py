"""Offline forward policy-improvement sweep, not an expanded calendar beam.

Compare each next-role alternative together with its regenerated downstream
continuation using ONE supplied complete-program score. Commit earlier choices
while sweeping forward; retain the incumbent on ties. The existing heuristic
only completes the suffix, never overrules the objective's winning sequence.
"""
from datetime import datetime
from math import isfinite

from run_analysis.prescription_distance import UnrepresentableDistanceTarget


def coordinate(make_continuation, evaluate, *, causal_opportunities=False):
    incumbent = make_continuation({})
    scored = evaluate(incumbent)
    initial = scored['combined_cost']
    if not isfinite(initial):
        raise ValueError('Coordinated continuation needs a finite initial score')
    overrides, decisions = {}, []
    evaluations = 1
    opportunities = [x['at'] for x in incumbent[2]]
    index = 0
    while index < len(incumbent[2]):
        at = incumbent[2][index]['at']
        prefix = [x['at'] for x in incumbent[2]][:index+1]
        chosen_role = incumbent[2][index]['requested_role']
        before = scored['combined_cost']
        alternatives = [{'role': chosen_role, 'cost': before, 'incumbent': True}]
        for role in ('easy', 'quality', 'long', 'rest'):
            if role == chosen_role:
                continue
            trial_overrides = {**overrides, datetime.fromisoformat(at).date(): role}
            try:
                trial = make_continuation(trial_overrides)
                candidate_times = [x['at'] for x in trial[2]]
                if (candidate_times[:index+1] != prefix if causal_opportunities else candidate_times != opportunities):
                    raise ValueError('Role comparison changed the opportunity calendar')
                candidate = evaluate(trial)
                evaluations += 1
            except UnrepresentableDistanceTarget as error:
                alternatives.append({'role': role, 'rejected': str(error)})
                continue
            cost = candidate['combined_cost']
            alternatives.append({'role': role, 'cost': cost})
            # Only numerical equality protection, not a new switching weight.
            if isfinite(cost) and cost < scored['combined_cost'] - 1e-9:
                incumbent, scored = trial, candidate
        selected = incumbent[2][index]['requested_role']
        overrides[datetime.fromisoformat(at).date()] = selected
        decisions.append({'at': at, 'before_cost': before,
            'selected_role': selected, 'after_cost': scored['combined_cost'],
            'alternatives': alternatives})
        index += 1
    return incumbent, scored, {'initial_cost': initial, 'final_cost': scored['combined_cost'],
        'evaluations': evaluations, 'opportunities': len(decisions),
        'decisions': decisions, 'causal_opportunities': causal_opportunities,
        'scope': ('One forward sweep; expected completions may change only future opportunities.'
                  if causal_opportunities else 'One forward sweep on fixed opportunities; no global optimum claim.')}
