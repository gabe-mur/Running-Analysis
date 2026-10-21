from datetime import timedelta
from types import SimpleNamespace
import json

import pytest

from scripts.audit_backward_controls import plain, digest, validate_control, capture, prior_frame
from scripts.continuation_replan import fingerprint
from test_typed_continuation import fixture


def control():
    _, make = fixture()
    full, states = make(49, {0: ('easy', 3.3), 2: ('easy', 3.5)})
    days = full[:21]
    snapshot = dict(generated_at=states[0].as_of.isoformat(), planner_diagnostics={
        'continuation_selection': dict(selected_hard_fingerprint=fingerprint(days),
            selected_program_fingerprint=fingerprint(full), search_cost=1.25)})
    return snapshot, dict(daily_states=states[:21]), days, full


def test_control_checks_exact_hard_full_identity_and_score():
    snapshot, inputs, days, full = control()
    validate_control(snapshot, inputs, days, full, [], 1.25)
    changed = full[:-1] + [full[-1].model_copy(update={'day_role': 'changed'})]
    with pytest.raises(ValueError, match='identity'):
        validate_control(snapshot, inputs, days, changed, [], 1.25)


@pytest.mark.parametrize('cost', [1.3, float('nan'), float('inf')])
def test_control_rejects_wrong_or_nonfinite_score(cost):
    snapshot, inputs, days, full = control()
    with pytest.raises(ValueError, match='score'):
        validate_control(snapshot, inputs, days, full, [], cost)


def test_post_upload_origin_cannot_allow_observations_after_generation_time():
    snapshot, inputs, days, full = control()
    origin = inputs['daily_states'][0].as_of
    snapshot['generated_at'] = (origin-timedelta(hours=12)).isoformat()
    history = [SimpleNamespace(start_time=origin-timedelta(hours=6))]
    with pytest.raises(ValueError, match='future work'):
        validate_control(snapshot, inputs, days, full, history, 1.25)


def test_control_rejects_date_holes_even_with_matching_fingerprint():
    snapshot, inputs, days, full = control()
    changed = full[:-1] + [full[-1].model_copy(update={'date': full[-1].date+timedelta(days=1)})]
    snapshot['planner_diagnostics']['continuation_selection']['selected_program_fingerprint'] = fingerprint(changed)
    with pytest.raises(ValueError, match='contiguous'):
        validate_control(snapshot, inputs, days, changed, [], 1.25)


def test_serialization_preserves_rejected_candidate_infinity_explicitly():
    assert json.loads(json.dumps(plain({'cost': float('inf')}), allow_nan=False)) == {
        'cost': {'non_finite': 'inf'}}
    assert digest({'b': 2, 'a': 1}) == digest({'a': 1, 'b': 2})


def test_invalid_indices_fail_before_reconstruction():
    with pytest.raises(ValueError, match='indices'):
        capture({'replans': [{}]}, indices=(1,))


def test_prior_hash_ignores_wrapper_clock_but_retains_planning_clock():
    _, _, days, _ = control()
    day = days[0]
    rec = day.recommendation
    other = rec.model_copy(update={'generated_at': rec.generated_at+timedelta(seconds=1)})
    assert digest(prior_frame({day.date: rec})) == digest(prior_frame({day.date: other}))
    other = other.model_copy(update={'planned_for': rec.planned_for+timedelta(hours=1)})
    assert digest(prior_frame({day.date: rec})) != digest(prior_frame({day.date: other}))


def test_control_rejects_saved_publication_mismatch():
    snapshot, inputs, days, full = control()
    snapshot['planner_diagnostics']['published_candidate'] = {'total_cost': 2.}
    with pytest.raises(ValueError, match='publication'):
        validate_control(snapshot, inputs, days, full, [], 1.25)
