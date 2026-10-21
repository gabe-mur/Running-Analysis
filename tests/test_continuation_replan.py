from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from run_analysis import weekly_schedule as planner
from run_analysis.selection_objective import (COMPLETED_SELECTION_HISTORY, FINALIZED_PROGRAM_SELECTOR,
                                              build_with_selection_history)
from scripts.aerobic_distribution import continuation_support
from scripts.continuation_replan import (combined_breakdown, continuation_replan_policy, fingerprint,
                                        hard_funding_evidence, select_program, shortlist, stable_prefix)
from scripts.replay_synthetic_easy_dose import synthetic_history
from test_typed_continuation import fixture


def row(offsets, cost, roles=None):
    return dict(offsets=offsets, total_cost=cost, roles=roles or {}, allocation_trace=[str(offsets), str(roles)])


def test_shortlist_keeps_lanes_then_distinct_run_counts_within_budget():
    independent = row([1, 4, 7], 1.)
    stable = row([0, 4, 7], 3., {0: 'easy'})
    rejected = {**row([2, 4], .1), 'acceptance': {'rejected': True}}
    extra = [row([1, 5, 7], 1.1), row([1, 4], 4.), row([1, 4, 7, 10], 5.)]
    record = dict(unconstrained=independent, stable=stable,
                  evaluated=[independent, stable, rejected, *extra])
    chosen = shortlist(record, limit=5)
    assert chosen[:2] == [independent, stable]
    assert len(chosen) == 4  # one budget slot belongs to the published hard seed
    assert {len(r['offsets']) for r in chosen} == {2, 3, 4}
    assert rejected not in chosen


def test_same_calendar_different_role_lanes_are_not_deduplicated():
    independent = row([0, 4], 1., {0: 'long'})
    stable = row([0, 4], 2., {0: 'easy'})
    record = dict(unconstrained=independent, stable=stable, evaluated=[independent, stable])
    assert shortlist(record, limit=3) == [independent, stable]
    with pytest.raises(ValueError, match='Missing finalized'):
        shortlist({**record, 'evaluated': [independent]}, limit=3)


@pytest.mark.parametrize('budget', [0, 2, 13])
def test_finalist_budget_is_explicit_and_bounded(budget):
    with pytest.raises(ValueError):
        shortlist({}, limit=budget)


def test_context_passes_only_supplied_realized_history_and_restores_after_failure():
    _, program = fixture()
    _, states = program(1, {})
    template, history = synthetic_history(states[0].as_of)
    completed = replace(history[-1], projected=True)
    ledger = [*history[:-1], completed]
    token = FINALIZED_PROGRAM_SELECTOR.set(lambda **kw: None)
    try:
        def build(value):
            data = COMPLETED_SELECTION_HISTORY.get()
            assert data['source'] == 'realized_simulation'
            assert data['runs'] == tuple(ledger)
            assert data['runs'][-1].projected  # Never relabel modeled HR as observed.
            assert value == 42
            raise RuntimeError('expected')
        with pytest.raises(RuntimeError, match='expected'):
            build_with_selection_history(build, template, ledger, 42)
        assert COMPLETED_SELECTION_HISTORY.get() is None
    finally:
        FINALIZED_PROGRAM_SELECTOR.reset(token)


def test_disabled_history_handoff_preserves_existing_call_without_reading_history():
    class NoRead:
        def __iter__(self):
            raise AssertionError('disabled path must not copy evidence')
    assert build_with_selection_history(lambda value: value, None, NoRead(), 42) == 42


def test_explicit_realized_history_can_supply_support_without_weakening_default():
    _, program = fixture()
    days, states = program(21, {})
    _, history = synthetic_history(states[0].as_of)
    modeled = [replace(r, projected=True) for r in history]
    with pytest.raises(ValueError, match='explicit realized'):
        continuation_support(days, states, modeled, policy='bounded')
    actual = continuation_support(days, states, modeled, policy='bounded', history_source='realized_simulation')
    assert actual == continuation_support(days, states, history, policy='bounded')
    with pytest.raises(ValueError, match='precede'):
        continuation_support(days, states, [replace(modeled[0], start_time=states[0].as_of + timedelta(days=1))],
                             policy='bounded', history_source='realized_simulation')


def test_support_ledger_is_required_not_synthesized_from_load_averages():
    _, program = fixture()
    days, states = program(21, {})
    with pytest.raises(ValueError, match='explicit completed-history'):
        select_program(days, None, {'daily_states': states}, {}, limit=6, continuation_days=28)


def test_program_fingerprint_includes_range_structure_and_target():
    _, program = fixture()
    days, _ = program(21, {2: ('easy', 4.)})
    def changed(**changes):
        return [d.model_copy(update={'recommendation': d.recommendation.model_copy(update=changes)})
                if d.recommendation else d for d in days]
    assert fingerprint(days) != fingerprint(changed(distance_range_miles=(3.5, 4.5)))
    assert fingerprint(days) != fingerprint(changed(distance_target_miles=3.9))
    assert fingerprint(days) == fingerprint(changed(generated_at=days[2].recommendation.generated_at + timedelta(seconds=2)))


def test_stable_lane_depends_on_physical_dates_and_roles_not_candidate_label():
    _, program = fixture()
    days, _ = program(21, {0: ('easy', 4.), 3: ('long', 8.)})
    inputs = dict(prior_run_offsets={0, 3}, prior_roles_by_offset={0: 'easy', 3: 'long'},
                  forced_rest_offsets=set(), completed_run_offsets=set())
    assert stable_prefix(days, inputs)
    changed = [*days]
    changed[3] = days[3].model_copy(update={'recommendation': days[3].recommendation.model_copy(update={'workout_type': planner.WorkoutType.EASY})})
    assert not stable_prefix(changed, inputs)


def test_combined_components_include_switching_once_and_reject_mismatch():
    hard = planner.PlannerScoreBreakdown(total_cost=0., coaching_cost=0., target_violation=0.,
        support_violation=0., shape_violation=0., finalized_recovery_cost=0., distance_switching_cost=2.)
    full = dict(combined_cost=32., combined_coaching_cost=1., combined_target_violation=0.,
        combined_support_violation=.5, combined_shape_violation=1., trajectory_recovery_cost=3.,
        trajectory_recovery_components={})
    assert combined_breakdown(hard, full, 3.75).total_cost == 32.
    with pytest.raises(ValueError, match='components'):
        combined_breakdown(hard, {**full, 'combined_cost': 30.}, 3.75)


def test_finalized_hard_funding_check_matches_the_publication_boundary_allowance():
    _, program = fixture()
    sparse, states = program(
        21,
        {offset: ('easy', 4.0) for offset in range(0, 20, 2)},
    )
    inputs = dict(
        daily_states=states,
        completed_miles_by_offset={},
        expected_target_projector=lambda recommendations, length: tuple(
            (17.0, 18.0) for _ in range(length)
        ),
    )

    evidence = hard_funding_evidence(sparse, inputs)

    assert evidence['planned_miles'] == 40.0
    assert evidence['target'] == (51.0, 54.0)
    assert evidence['violation_miles'] > planner.HARD_FUNDING_ROUNDING_TOLERANCE_MILES


def test_selector_hook_is_restored_after_failure():
    original = FINALIZED_PROGRAM_SELECTOR.get()
    with pytest.raises(RuntimeError):
        with continuation_replan_policy(enabled=True, calendar_records=[]) as records:
            assert FINALIZED_PROGRAM_SELECTOR.get() is not None
            raise RuntimeError('expected')
    assert FINALIZED_PROGRAM_SELECTOR.get() is original


def test_daily_and_upload_builds_receive_fresh_execution_ledger(monkeypatch):
    from run_analysis import adherence_projection as projection
    from test_adherence_projection import _single_then_empty_schedule
    from test_recommendation import CONFIG, _state
    # The fixture schedules tomorrow at 07:00, then returns no more runs. Keep
    # that session inside the first 24-hour simulation slice; wall-clock starts
    # before 07:00 otherwise refresh it away before execution.
    template = _state(as_of=datetime(2026, 9, 22, 18, tzinfo=timezone.utc),
                      typical_easy_run_miles=4.)
    fake = _single_then_empty_schedule()
    captured = []
    def build(daily_states, *args, **kwargs):
        evidence = COMPLETED_SELECTION_HISTORY.get()
        captured.append((daily_states[0].as_of, evidence['runs']))
        return fake(daily_states, *args, **kwargs)
    monkeypatch.setattr(projection, 'build_weekly_schedule', build)
    with continuation_replan_policy(enabled=True, calendar_records=[]):
        weeks = projection.simulate_adherence(template, [], CONFIG, template.as_of,
                                              weeks=1, simulation_days=2)
    assert [len(runs) for _, runs in captured] == [0, 1, 1]
    assert all(run.start_time < at for at, runs in captured for run in runs)
    assert captured[1][1][0].projected
    assert [r.trigger for r in weeks[0].replan_snapshots] == ['scheduled_refresh', 'post_upload', 'scheduled_refresh']
    assert COMPLETED_SELECTION_HISTORY.get() is None


def test_publication_gate_checks_actual_ranges_even_when_scores_and_targets_match():
    from scripts.replay_synthetic_easy_dose import shared_objective_failure
    from run_analysis.adherence_projection import ProjectionPlanSession
    _, program = fixture()
    days, states = program(21, {2: ('easy', 4.)})
    r = days[2].recommendation
    session = ProjectionPlanSession(r.planned_for, r.workout_type, 4., 3.7, 4.2)
    physical = dict(planned_for=r.planned_for.isoformat(), workout_type=r.workout_type.value,
                    midpoint_miles=4., distance_low_miles=3.7, distance_high_miles=4.2)
    score = dict(total_cost=10., allocation_trace=['same'])
    details = dict(hard_sessions=[physical], selected_hard_fingerprint='same',
                   published_hard_fingerprint='same', search_cost=10., publication_cost=10.)
    snapshot = SimpleNamespace(generated_at=states[0].as_of, planned_sessions=(session,),
        planner_diagnostics=dict(winning_candidate=score, published_candidate=score, continuation_selection=details))
    assert shared_objective_failure(snapshot) is None
    snapshot.planned_sessions = (replace(session, distance_high_miles=4.5),)
    assert shared_objective_failure(snapshot).code == 'search_publication_mismatch'
