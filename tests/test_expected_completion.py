from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from run_analysis import adherence_projection as projection, weekly_schedule as planner
from run_analysis.selection_objective import COMPLETED_SELECTION_HISTORY, EXPECTED_COMPLETION_STATE
from run_analysis.web.schemas import WorkoutStep, QualitySessionType
from scripts.expected_completion import (ExpectedCompletion, expected_completion_policy,
    expected_completion_build_policy, expected_run)
from scripts.replay_synthetic_easy_dose import synthetic_history
from scripts.typed_continuation_prototype import rollout, runs
from scripts.continuation_replan import signature
from test_recommendation import CONFIG


def setup(kind='long', miles=8.):
    start = datetime(2026, 9, 22, tzinfo=timezone.utc)
    template, history = synthetic_history(start)
    item = planner.RecommendationResponse(generated_at=start, fitness_state_as_of=start,
        planned_for=start + timedelta(hours=7), workout_type=kind,
        planning_role='ordinary_easy' if kind == 'easy' else kind,
        title='Expected completion fixture', distance_target_miles=miles,
        distance_range_miles=(miles-.2, miles+.2), confidence='moderate', readiness='ready')
    if kind == 'intervals':
        item = item.model_copy(update={'quality_session_type': QualitySessionType.SHORT_INTERVALS,
            'structure': [WorkoutStep(instruction='Controlled reps', phase='work', repetitions=6,
                work_duration_minutes=1., target_zones=['Z4'])]})
    return template, history, item


@pytest.mark.parametrize('recoveries', [(1., 5.), (5., 1.)])
@pytest.mark.parametrize('mutate', [False, True])
def test_recovery_duration_cache_matches_fresh_state_in_either_order(recoveries, mutate):
    template, history, item = setup('intervals', 4.5)
    item = item.model_copy(update={'structure': [WorkoutStep(
        instruction='Controlled reps', phase='work', repetitions=6,
        work_duration_minutes=5., recovery_duration_minutes=recoveries[0],
        target_zones=['Z4'])]})
    at = item.planned_for + timedelta(days=1)
    model = ExpectedCompletion(template, history, CONFIG)
    first = model.ledger(at, [item])
    model.at(at, [item])
    if mutate:
        # API models are mutable; object identity cannot certify an unchanged dose.
        item.structure[0].recovery_duration_minutes = recoveries[1]
        changed = item
    else:
        changed = item.model_copy(update={'structure': [item.structure[0].model_copy(
            update={'recovery_duration_minutes': recoveries[1]})]})
    fresh = ExpectedCompletion(template, history, CONFIG)
    expected = fresh.ledger(at, [changed])
    assert expected[-1].moving_minutes != first[-1].moving_minutes
    assert model.ledger(at, [changed]) == expected
    assert model.at(at, [changed]) == fresh.at(at, [changed])
    assert model.ledger(at, [changed.model_copy(deep=True)]) is model.ledger(at, [changed])


@pytest.mark.parametrize('kind,miles', [('easy', 4.), ('long', 8.), ('intervals', 4.5)])
def test_conditional_state_matches_independent_exact_execution(kind, miles):
    template, history, item = setup(kind, miles)
    actual = projection.ProjectionRun(start_time=item.planned_for, distance_miles=miles,
        moving_minutes=miles*11, workout_type=item.workout_type,
        difficulty=projection._projected_difficulty(item, miles, 11), projected=True,
        planning_role=projection._completed_planning_role(item, item.workout_type),
        prescribed_workout_type=item.workout_type, completed_prescribed_workout=True,
        prescribed_target_miles=miles, prescribed_low_miles=miles-.2, prescribed_high_miles=miles+.2,
        prescription_title=item.title, quality_session_type=item.quality_session_type)
    observed = [*history, actual]
    model = ExpectedCompletion(template, history, CONFIG)
    for day in (1, 2, 7, 20, 28):
        at = template.as_of + timedelta(days=day)
        _, target, evidence = planner.derive_weekly_target(projection.planning_activities_for(observed), at, CONFIG)
        rebuilt = projection._state_at(template, observed, at, evidence.capacity_reference_miles)
        assert model.at(at, [item]) == rebuilt
        assert model.target_at(at, [item]) == target
    assert tuple(history) == model.completed
    assert len(history) == 36


def test_no_future_evidence_or_duplicate_completion():
    template, history, item = setup()
    model = ExpectedCompletion(template, history, CONFIG)
    assert model.at(item.planned_for, [item]) == model.at(item.planned_for)
    later = item.planned_for + timedelta(days=1)
    with pytest.raises(ValueError, match='duplicates'):
        model.at(later, [item, item])
    realized = ExpectedCompletion(template, [*history, expected_run(item, 11)], CONFIG)
    with pytest.raises(ValueError, match='duplicates'):
        realized.at(later, [item])


def test_above_range_work_does_not_disappear_into_expected_completion():
    template, history, item = setup('easy', 4.)
    expected = expected_run(item, 11)
    excess = replace(expected, distance_miles=6., moving_minutes=66.,
        difficulty=projection._projected_difficulty(item, 6., 11))
    at = template.as_of + timedelta(days=1)
    forecast = ExpectedCompletion(template, history, CONFIG).at(at, [item])
    actual = ExpectedCompletion(template, [*history, excess], CONFIG).at(at)
    assert actual.recent_load.trailing_7d.distance_miles == forecast.recent_load.trailing_7d.distance_miles + 2
    assert actual.recovery_residual_load != forecast.recovery_residual_load


def test_scope_requires_explicit_ledger_and_restores_after_error():
    with pytest.raises(ValueError, match='explicit realized'):
        with expected_completion_policy(CONFIG):
            pass
    template, history, _ = setup()
    token = COMPLETED_SELECTION_HISTORY.set(dict(template=template, runs=history))
    try:
        with pytest.raises(RuntimeError):
            with expected_completion_policy(CONFIG) as model:
                assert EXPECTED_COMPLETION_STATE.get() is model
                raise RuntimeError('fixture')
        assert EXPECTED_COMPLETION_STATE.get() is None
    finally:
        COMPLETED_SELECTION_HISTORY.reset(token)


def test_full_conditional_rollout_preserves_promoted_prefix_and_compute_cache():
    template, history, item = setup('easy', 4.)
    model = ExpectedCompletion(template, history, CONFIG)
    origin = template.as_of
    states = [model.at(origin + timedelta(days=i)) for i in range(21)]
    days = [planner.WeeklyScheduleDay(date=s.as_of.date(), recommendation=None,
             day_role='rest_day', rationale='fixture') for s in states]
    last = item.model_copy(update={'planned_for': origin + timedelta(days=20, hours=7)})
    days[-1] = days[-1].model_copy(update={'recommendation': last, 'day_role': 'easy_run'})
    token = EXPECTED_COMPLETION_STATE.set(model)
    try:
        args = dict(state_at=model.at, continuation_days=14)
        request = planner.RecommendationRequest(health_status='normal')
        full, full_states, _ = rollout(days, states, request, CONFIG, [(17., 18.)]*35, **args)
        first = next(i for i in range(21, 35) if full[i].recommendation)
        cut = first + 1
        promoted, _, _ = rollout(full[:cut], full_states[:cut], request, CONFIG, [(17., 18.)]*35,
            state_at=model.at, continuation_days=35-cut)
        assert signature(full) == signature(promoted)
        assert runs(full)
        initial = model.rebuilds
        model.at(origin + timedelta(days=1), [item])
        model.at(origin + timedelta(days=1), [item])
        assert model.rebuilds <= initial + 1
    finally:
        EXPECTED_COMPLETION_STATE.reset(token)


def test_same_future_candidate_has_same_score_after_expected_or_actual_completion():
    template, history, item = setup()
    origin = template.as_of + timedelta(days=1)
    before = ExpectedCompletion(template, history, CONFIG)
    conditional_history = before.ledger(origin, [item])
    actual_history = [*history, projection.ProjectionRun(
        start_time=item.planned_for, distance_miles=8., moving_minutes=88.,
        workout_type=item.workout_type, difficulty=projection._projected_difficulty(item, 8., 11),
        projected=True, planning_role='long', prescribed_workout_type=item.workout_type,
        completed_prescribed_workout=True, prescribed_target_miles=8.,
        prescribed_low_miles=7.8, prescribed_high_miles=8.2, prescription_title=item.title)]
    costs = []
    for ledger in (conditional_history, actual_history):
        model = ExpectedCompletion(template, ledger, CONFIG)
        states = [model.at(origin + timedelta(days=i)) for i in range(7)]
        days, options = [], []
        for i, state in enumerate(states):
            at = state.as_of + timedelta(hours=7)
            recommendation = item.model_copy(update={
                'planned_for': at, 'workout_type': planner.WorkoutType.EASY,
                'planning_role': 'ordinary_easy', 'distance_target_miles': 4.,
                'distance_range_miles': (3.8, 4.2)}) if i in (1, 3, 5) else None
            days.append(planner.WeeklyScheduleDay(date=state.as_of.date(), recommendation=recommendation,
                day_role='easy_run' if recommendation else 'rest_day', rationale='fixed candidate'))
            options.append([model.at(at)])
        token = EXPECTED_COMPLETION_STATE.set(model)
        try:
            target = model.target_at(origin)
            coaching, _ = planner._finalized_program_coaching_cost(days, states, options,
                planner.RecommendationRequest(health_status='normal'), CONFIG, target)
            recovery = planner._finalized_program_recovery_cost(days, states, CONFIG, target,
                session_states=[x[0] for x in options])
            costs.append((coaching, recovery))
        finally:
            EXPECTED_COMPLETION_STATE.reset(token)
    assert costs[0] == costs[1]


def test_training_identical_prescription_copies_share_cached_state():
    template, history, item = setup()
    model = ExpectedCompletion(template, history, CONFIG)
    at = template.as_of + timedelta(days=1)
    first = model.at(at, [item])
    rebuilt = model.rebuilds
    second = model.at(at, [item.model_copy(deep=True)])
    assert first is second
    assert model.rebuilds == rebuilt


def test_target_and_cadence_queries_match_full_states_without_building_them():
    template, history, item = setup()
    reference = ExpectedCompletion(template, history, CONFIG)
    model = ExpectedCompletion(template, history, CONFIG)
    for delta in (0, 1, 7, 28, 40):
        at = item.planned_for + timedelta(days=delta)
        window = reference.at(at + timedelta(microseconds=1), [item]).recent_load.trailing_28d
        target = reference.target_at(at, [item])
        if window.activity_count:
            assert model.mean_session_after(at, [item]) == window.distance_miles / window.activity_count
        else:
            with pytest.raises(ValueError, match='requires running history'):
                model.mean_session_after(at, [item])
        assert model.target_at(at, [item]) == target
    assert model.rebuilds == 0
    assert model._states == {}


def test_ledger_reuses_only_unchanged_causal_prefix(monkeypatch):
    import scripts.expected_completion as module
    template, history, first = setup('easy', 4.)
    second = first.model_copy(update={'planned_for': first.planned_for + timedelta(days=2)})
    end = second.planned_for + timedelta(days=1)
    calls = []
    original = module.expected_run
    def counted(item, pace):
        calls.append(item)
        return original(item, pace)
    monkeypatch.setattr(module, 'expected_run', counted)
    model = ExpectedCompletion(template, history, CONFIG)
    model.ledger(second.planned_for, [first])
    assert len(calls) == 1
    result = model.ledger(end, [second, first])
    assert len(calls) == 2  # earlier conditional execution was reused
    assert result is model.ledger(end, [first.model_copy(), second.model_copy()])
    changed = first.model_copy(update={'distance_target_miles': 4.1})
    revised = model.ledger(end, [changed, second])
    assert len(calls) == 4  # changing a prefix invalidates its entire suffix
    fresh = ExpectedCompletion(template, history, CONFIG).ledger(end, [changed, second])
    assert revised == fresh
    assert revised != result


@pytest.mark.parametrize('days', [7, 14, 28])
def test_direct_projection_window_matches_training_load_contract(days):
    from run_analysis.training_load import TrainingSession, rolling_load
    template, history, item = setup()
    edge = template.as_of - timedelta(days=days)
    # Both exact boundaries, a future run, and missing/zero zone evidence.
    records = [*history, *[replace(history[-1], start_time=at,
        difficulty=history[-1].difficulty.model_copy(update={'zone_load': zone}))
        for at, zone in ((edge, 0.), (edge+timedelta(microseconds=1), None),
                         (template.as_of, 0.), (template.as_of+timedelta(microseconds=1), 50.))]]
    sessions = [TrainingSession(activity_id=i, start_time=r.start_time,
        distance_miles=r.distance_miles, moving_minutes=r.moving_minutes,
        zone_load=r.difficulty.zone_load, hard_minutes=r.difficulty.zone_breakdown.hard_minutes)
        for i,r in enumerate(records)]
    expected = rolling_load(sessions, template.as_of, days)
    actual = projection._load_window(records, template.as_of, days)
    assert all(getattr(actual, field) == getattr(expected, field) for field in (
        'distance_miles','moving_minutes','zone_load','hard_minutes','activity_count','zone_load_activity_count'))


@pytest.mark.parametrize('keywords', [False, True])
def test_build_adapter_supports_wrapped_signatures_and_leaves_actual_ledger_untouched(monkeypatch, keywords):
    template, history, item = setup()
    def wrapped(*args, **kwargs):
        model = EXPECTED_COMPLETION_STATE.get()
        assert model is not None
        model.at(item.planned_for + timedelta(days=1), [item])
        assert COMPLETED_SELECTION_HISTORY.get()['runs'] == tuple(history)
        return 42
    monkeypatch.setattr(projection, 'build_weekly_schedule', wrapped)
    token = COMPLETED_SELECTION_HISTORY.set(dict(template=template, runs=tuple(history)))
    try:
        with expected_completion_build_policy() as records:
            if keywords:
                result = projection.build_weekly_schedule(daily_states=[template], request=None, config=CONFIG)
            else:
                result = projection.build_weekly_schedule([template], None, CONFIG)
            assert result == 42
            assert records[0]['history_count'] == len(history)
        assert projection.build_weekly_schedule is wrapped
        assert EXPECTED_COMPLETION_STATE.get() is None
    finally:
        COMPLETED_SELECTION_HISTORY.reset(token)


def test_conditional_project_preserves_exogenous_weather_and_health():
    template, history, item = setup()
    model = ExpectedCompletion(template, history, CONFIG)
    at = item.planned_for + timedelta(days=1)
    raw = model.at(at).model_copy(update={'current_health_status': 'little_tired',
        'recent_illness_or_recovery': True, 'normal_runs_since_health_event': 2})
    result = model.project(raw, [item])
    assert result.current_health_status == 'little_tired'
    assert result.recent_illness_or_recovery
    assert result.normal_runs_since_health_event == 2


def test_expected_duration_uses_the_same_daily_pace_assumption_as_execution():
    template, history, item = setup()
    # A slow sample is in the daily refresh window but rolls out by 07:00.
    old = replace(history[-1], start_time=item.planned_for-timedelta(days=28),
                  distance_miles=4., moving_minutes=60.)
    history = [r for r in history if r.start_time != old.start_time] + [old]
    window = projection._load_window(history, template.as_of, 28)
    pace = min(15., max(7., window.moving_minutes / window.distance_miles))
    model = ExpectedCompletion(template, history, CONFIG)
    ledger = model.ledger(template.as_of+timedelta(days=1), [item])
    assert ledger[-1].moving_minutes == pytest.approx(item.planned_distance_miles * pace)
