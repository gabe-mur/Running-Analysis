"""Release guards; no expensive search and no application database mutation."""
from dataclasses import asdict, replace
from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest

from run_analysis.adaptive.new_model_daily_harness import HarnessPolicy
from run_analysis.adaptive.frequency_range_trial import (FrequencyMidpointPreference,
    FrequencyContract, FrequencyRangePolicy, FrequencyStageTarget)
from scripts.release_calendar_mass_planner import apply_build


def harness():
    policy = HarnessPolicy(evaluations=10000, soft_horizon_taper=True,
        delivery_contract='sustained_v1', continuation_boundary=True,
        continuation_guard_days=7, purpose_continuation=True,
        purpose_continuation_phase='known_full_phase_v1',
        complete_first_search=True, final_winner_evaluations=1000,
        coordinated_final_evaluations=64, continuation_distribution='c1_v1',
        general_progression='easy_first_coordinated_v2', calendar_merge_evaluations=32,
        progression_admission='observed_rolling_v2', progression_credit='actual_distance_v2')
    frequency = FrequencyContract(adopted_at=datetime.now(timezone.utc),
        policy=FrequencyRangePolicy(),
        targets=(FrequencyStageTarget.freeze('seed', date(2026, 10, 20), 20.),),
        preference=FrequencyMidpointPreference(.04, 54, .02))
    return SimpleNamespace(policy=policy, frequency_contract=frequency,
        stage=SimpleNamespace(policy=SimpleNamespace(volume_path='gradual', growth_focus='easy_volume')),
        history=('real history',), target_history=('original targets',),
        general_progression_state=None, race_goal=None)


def test_release_changes_only_the_three_accepted_contracts():
    h = harness()
    old = asdict(h.policy)
    stage, history, targets = h.stage, h.history, h.target_history
    frequency = h.frequency_contract
    apply_build(h, datetime.now(timezone.utc))
    changed = {k for k, v in asdict(h.policy).items() if old[k] != v}
    assert changed == {'general_progression', 'volume_shortfall_policy'}
    assert h.policy.general_progression == 'continuous_general_v7'
    assert h.policy.volume_shortfall_policy == 'forgone_issued_v2'
    assert h.frequency_contract == replace(frequency,
        preference=replace(frequency.preference, edge_fraction=.03))
    assert (h.stage, h.history, h.target_history) == (stage, history, targets)
    assert h.general_progression_state is None  # initialized by real publication, not invented here


@pytest.mark.parametrize('kind', ['race', 'already_adopted', 'missing_frequency', 'unaccepted_search', 'wrong_credit'])
def test_release_refuses_unsupported_or_repeated_migration(kind):
    h = harness()
    if kind == 'race': h.race_goal = object()
    elif kind == 'already_adopted': h.policy = replace(h.policy, general_progression='continuous_general_v7')
    elif kind == 'missing_frequency': h.frequency_contract = None
    elif kind == 'unaccepted_search': h.policy = replace(h.policy, coordinated_final_evaluations=32)
    elif kind == 'wrong_credit': h.policy = replace(h.policy, progression_credit='issued_nominal_v1')
    policy, frequency = h.policy, h.frequency_contract
    with pytest.raises(ValueError):
        apply_build(h, datetime.now(timezone.utc))
    assert h.policy == policy and h.frequency_contract == frequency
