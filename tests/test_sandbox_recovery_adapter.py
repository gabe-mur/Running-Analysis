from dataclasses import replace
from datetime import timedelta
import json

import pytest

from run_analysis.recovery_assessment import AssessmentPolicy, sensitivity_panel
from run_analysis.recovery_response import ResponseExpectation
from scripts.expected_completion import ExpectedCompletion
from scripts.sandbox_recovery_adapter import capture, history_entry, review_completion, session_key, projection_conflicts
from scripts.audit_sandbox_recovery_adapter import phase_assumptions
from test_expected_completion import setup
from test_recommendation import CONFIG


def fixture():
    template, history, first = setup('easy', 4.)
    second = first.model_copy(deep=True, update={'planned_for': first.planned_for+timedelta(days=2)})
    days = [type('Day', (), {'recommendation': r})() for r in (first, second)]
    model = ExpectedCompletion(template, history, CONFIG)
    snapshot = capture(model, days, issued_at=template.as_of, assumptions=phase_assumptions())
    return snapshot, model, days, first, second


def test_snapshot_deeply_freezes_prescription_and_assumptions():
    snapshot, _, _, first, second = fixture()
    digest = snapshot.digest
    first.title = 'later edit'
    first.distance_target_miles = 8.
    clone = snapshot.recommendation(session_key(first.planned_for))
    clone.title = 'another edit'
    entry = snapshot.entries()[-1]
    entry.workload.structure['bouts'].append({'future modification': True})
    assert snapshot.digest == digest
    assert snapshot.recommendation(session_key(first.planned_for)).planned_distance_miles == 4.
    assert snapshot.option(session_key(second.planned_for)).workload.structure['bouts'] == []


def test_native_running_minutes_reused_without_label_based_hr_filling():
    snapshot, model, _, first, second = fixture()
    native = model.ledger(second.planned_for, [first])[-1]
    expected = next(e for e in snapshot.entries() if e.key == session_key(first.planned_for))
    assert expected.workload.running_minutes.low == native.moving_minutes
    actual = history_entry(native)
    assert actual.workload.cardio_points.low == 0
    assert actual.workload.cardio_points.high == native.moving_minutes*5
    assert actual.work_minutes.low == 0
    assert actual.work_minutes.high == native.moving_minutes


def test_prefix_excludes_next_dose_and_all_future_work():
    snapshot, _, _, first, second = fixture()
    prefix = snapshot.prefix(session_key(second.planned_for))
    assert prefix[-1].key == session_key(first.planned_for)
    assert all(e.start < second.planned_for for e in prefix)
    assert snapshot.option(session_key(second.planned_for)).workload.source == 'projected'


def test_exact_once_completion_and_full_assessment_parity():
    snapshot, _, _, first, second = fixture()
    key = session_key(second.planned_for)
    expected = snapshot.prefix(key)[-1]
    completed = replace(expected, workload=replace(expected.workload, source='observed'))
    actual = snapshot.after_completion(completed)
    assert len(actual) == len(snapshot.entries())
    policy = AssessmentPolicy('arithmetic', 130, 300, 24, 24)
    option = snapshot.option(key)
    before = sensitivity_panel(snapshot.prefix(key), option.workload, second.planned_for, [policy])
    after = sensitivity_panel(tuple(e for e in actual if e.start < second.planned_for),
                              option.workload, second.planned_for, [policy])
    assert before == after
    assert sum(e.key == completed.key for e in actual) == 1
    with pytest.raises(ValueError, match='exactly once'):
        snapshot.after_completion(replace(completed, key='unmatched'))


def test_snapshot_cannot_capture_future_observations_or_past_prescription():
    snapshot, model, days, first, _ = fixture()
    with pytest.raises(ValueError, match='unconsumed'):
        capture(model, days, issued_at=first.planned_for+timedelta(seconds=1), assumptions=phase_assumptions())
    with pytest.raises(ValueError, match='unavailable'):
        capture(model, days, issued_at=model.completed[-1].start_time, assumptions=phase_assumptions())


def test_completion_review_uses_frozen_baseline_and_requires_eligibility():
    snapshot, _, _, _, second = fixture()
    key = session_key(second.planned_for)
    expected = snapshot.prefix(key)[-1]
    completion = replace(expected, workload=replace(expected.workload, source='observed'))
    args = dict(alternatives=[], eligible_keys=set(),
        policies=[AssessmentPolicy('arithmetic', 1000, 5000, 24, 24)],
        minimum_target_miles=4., adjustment_source='No smaller eligible variant', readiness='normal')
    result = review_completion(snapshot, completion, key, **args)
    assert result['action'] == 'retain_issued'
    assert result['target_miles'] == 4.
    assert not result['planner_influence']
    with pytest.raises(ValueError, match='full-program eligibility'):
        review_completion(snapshot, completion, key,
            **dict(args, alternatives=[snapshot.option(key, option_key='menu_only')]))


def test_intervening_forecast_cannot_be_treated_as_completed():
    snapshot, model, days, first, second = fixture()
    third = second.model_copy(update={'planned_for': second.planned_for+timedelta(days=2)})
    days.append(type('Day', (), {'recommendation': third})())
    snapshot = capture(model, days, issued_at=model.template.as_of, assumptions=phase_assumptions())
    completion = next(e for e in snapshot.entries() if e.key == session_key(first.planned_for))
    completion = replace(completion, workload=replace(completion.workload, source='observed'))
    with pytest.raises(ValueError, match='intervening'):
        review_completion(snapshot, completion, session_key(third.planned_for),
            alternatives=[], eligible_keys=set(), policies=[],
            minimum_target_miles=4, adjustment_source='fixture')


def test_missing_hr_is_not_confirmed_perfect_hr_compliance():
    snapshot, model, _, first, second = fixture()
    actual = history_entry(model.ledger(second.planned_for, [first])[-1])
    expected = snapshot.prefix(session_key(second.planned_for))[-1]
    from run_analysis.workload_accounting import compare_execution
    assert compare_execution(expected.workload, actual.workload)['cardio_points'] == 'overlapping_uncertainty'
    assert 'phase' in json.loads(snapshot.payload)['assumptions']['base']['basis']


def test_shared_duration_contract_accommodates_structure_without_adding_distance():
    from run_analysis.web.schemas import WorkoutStep
    snapshot, model, days, first, _ = fixture()
    first.distance_target_miles = 3.7
    first.distance_range_miles = (3.5, 4.)
    first.structure = [WorkoutStep(instruction='Warm-up', duration_minutes=12),
        WorkoutStep(instruction='3 x 5', phase='work', repetitions=3,
                    work_duration_minutes=5, recovery_duration_minutes=2),
        WorkoutStep(instruction='Cool-down', duration_minutes=10)]
    conflicts = projection_conflicts(model, days, assumptions=phase_assumptions())
    assert conflicts == []
    snapshot = capture(model, days, issued_at=model.template.as_of, assumptions=phase_assumptions())
    expected = next(e for e in snapshot.entries() if e.key == session_key(first.planned_for))
    assert expected.workload.running_minutes.low == 41
    assert first.planned_distance_miles == 3.7


def test_expected_rpe_is_saved_at_issue_not_supplied_after_upload():
    from run_analysis.workload_accounting import Quantity
    _, model, days, first, second = fixture()
    key = session_key(first.planned_for)
    expectations = {key: ResponseExpectation('Issued fixture RPE', Quantity(2, 4))}
    snapshot = capture(model, days, issued_at=model.template.as_of,
                       assumptions=phase_assumptions(), expected_responses=expectations)
    expectations[key] = ResponseExpectation('Later revision', Quantity(7, 9))
    assert snapshot.expected_response(key).effort.high == 4
    completed = next(e for e in snapshot.entries() if e.key == key)
    completed = replace(completed, workload=replace(completed.workload, source='observed'))
    result = review_completion(snapshot, completed, session_key(second.planned_for),
        alternatives=[], eligible_keys=set(), policies=[AssessmentPolicy('fixture', 1000, 5000, 24, 24)],
        minimum_target_miles=4, adjustment_source='fixture', effort=8, readiness='unknown')
    assert result['action'] == 'request_readiness_check'
    assert result['response']['expectation_source'] == 'Issued fixture RPE'
