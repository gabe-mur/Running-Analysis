from copy import deepcopy

import pytest

from scripts.audit_continuation_upload import source_row, unpack_soft, repeat_role_sweep
from scripts.continuation_replan import signature
from scripts.coordinated_continuation import coordinate
from scripts.typed_continuation_prototype import rollout
from run_analysis import weekly_schedule as planner
from test_typed_continuation import fixture, CONFIG


@pytest.mark.parametrize('execution', ['latest','opening'])
def test_reconstruction_honors_the_saved_execution_contract(monkeypatch,execution):
    from scripts import audit_continuation_upload as audit
    calls=[]
    monkeypatch.setattr(audit.projection,'simulate_adherence',lambda *args,**kwargs:calls.append(kwargs))
    report=dict(realized_execution=dict(start_at='2026-09-22T00:00:00+00:00'),
        synthetic_sandbox=dict(config=CONFIG,calendar_neighbor_prototype=[],execution_plan=execution),replans=[])
    audit.reconstruct(report,lambda *args:pytest.fail('No saved decisions'))
    assert (calls[0]['execution_schedule_selector'] is None)==(execution=='latest')


def test_saved_soft_prescriptions_round_trip_without_changing_physical_fields():
    _, program = fixture()
    days, _ = program(3, {1: ('long', 8.)})
    saved = signature(days)
    assert signature(unpack_soft(saved)) == saved


def test_source_row_requires_same_calendar_and_same_legacy_cost():
    wrong = dict(offsets=[0, 2], total_cost=1., roles={0: 'easy'})
    right = dict(offsets=[0, 2], total_cost=2., roles={0: 'long'})
    selected = dict(offsets=[0, 2], legacy_cost=2.)
    assert source_row(dict(evaluated=[wrong, right]), selected) is right
    with pytest.raises(StopIteration):
        source_row(dict(evaluated=[wrong]), selected)


def test_opportunity_override_changes_only_proposal_grid_not_input_states():
    state_at, program = fixture()
    days, states = program(21, {20: ('easy', 4.)})
    saved_states = deepcopy(states)
    targets = [(17., 18.)] * 28
    common = dict(state_at=state_at, continuation_days=7)
    request = planner.RecommendationRequest(health_status='normal')
    default = rollout(days, states, request, CONFIG, targets, **common)
    mean = states[0].recent_load.trailing_28d.distance_miles / states[0].recent_load.trailing_28d.activity_count
    matched = rollout(days, states, request, CONFIG, targets,
                      opportunity_mean_session=mean, **common)
    shifted = rollout(days, states, request, CONFIG, targets,
                      opportunity_mean_session=mean * 1.1, **common)
    assert signature(default[0]) == signature(matched[0])
    assert default[1:] == matched[1:]
    assert [x['at'] for x in default[2]] != [x['at'] for x in shifted[2]]
    assert shifted[1] == default[1]
    assert states == saved_states
    assert shifted[0][:21] == days


@pytest.mark.parametrize('value', [0., -1., float('nan'), float('inf')])
def test_invalid_opportunity_override_rejected(value):
    state_at, program = fixture()
    days, states = program(21, {20: ('easy', 4.)})
    with pytest.raises(ValueError, match='finite and positive'):
        rollout(days, states, planner.RecommendationRequest(health_status='normal'), CONFIG,
            [(17., 18.)] * 28, state_at=state_at, continuation_days=7,
            opportunity_mean_session=value)


def test_diagnostic_sweep_hook_restores_on_error():
    from scripts import coordinated_continuation as module
    assert module.coordinate is coordinate
    with pytest.raises(RuntimeError):
        with repeat_role_sweep(3):
            assert module.coordinate is not coordinate
            raise RuntimeError('fixture')
    assert module.coordinate is coordinate
