import json
import sqlite3

import pytest

from run_analysis import weekly_schedule as planner
from run_analysis.db import initialize
from run_analysis.prescription_distance import (
    CANONICAL_DISTANCE_TARGETS, canonical_distance_policy, rounded_distance_target,
    UnrepresentableDistanceTarget, reject_unrepresentable_program,
)
from run_analysis.prescription_matching import archive_weekly_prescriptions
from run_analysis.recommendation import scale_quality_session
from run_analysis.web.schemas import RecommendationResponse, WorkoutStep
from scripts.opportunity_search_repair import opportunity_search_policy
from test_finalized_scoring_trial import program
from test_prescription_matching import _schedule
from test_recommendation import CONFIG


@pytest.mark.parametrize('value,expected', [(3.25,3.3),(3.75,3.8),(3.95,4.0),(8.45,8.5),(3.74,3.7)])
def test_rounding_commits_a_tenth_before_scoring(value, expected):
    assert rounded_distance_target(value) == expected


def test_limits_override_nearest_rounding_and_infeasible_band_fails():
    assert rounded_distance_target(3.75, maximum=3.75) == 3.7
    assert rounded_distance_target(3.71, minimum=3.75, maximum=4) == 3.8
    with pytest.raises(ValueError):
        rounded_distance_target(3.75, minimum=3.75, maximum=3.75)


def test_legacy_loads_unchanged_but_new_target_is_not_range_midpoint():
    _, days = program()
    legacy = days[0].recommendation
    assert legacy.planned_distance_miles == 3.95
    new = RecommendationResponse.model_validate({**legacy.model_dump(), 'distance_target_miles':4.0})
    assert planner._midpoint(new) == 4
    assert new.distance_range_miles == (3.7,4.2)
    assert RecommendationResponse.model_validate_json(new.model_dump_json()).planned_distance_miles == 4
    assert planner.expected_compliance_activity(new, pace_min_mile=11).distance_miles == 4
    assert planner.expected_compliance_activity(new, pace_min_mile=11).moving_minutes == 44
    for value in [3.75, float('nan'), 4.3]:
        with pytest.raises(ValueError):
            RecommendationResponse.model_validate({**legacy.model_dump(), 'distance_target_miles':value})


@pytest.mark.parametrize('actual,expected', [(3.7,4.), (3.9,4.), (4.2,4.), (3.6,3.9), (4.3,4.1)])
def test_compliance_preserves_whole_range_with_explicit_target(actual, expected):
    assert planner.adherence_normalized_distance(actual,(3.7,4.2),4.) == pytest.approx(expected)
    # Missing explicit target remains backwards compatible, not retroactively rounded.
    assert planner.adherence_normalized_distance(4.,(3.7,4.2)) == 3.95


def test_every_allocation_scored_uses_displayable_target_and_keeps_inputs_immutable():
    states, days = program()
    scored = []
    def score(candidate):
        for day in candidate:
            r = day.recommendation
            if r:
                RecommendationResponse.model_validate(r.model_dump())
                assert r.distance_target_miles == round(r.distance_target_miles,1)
                assert planner._midpoint(r) == r.distance_target_miles
        scored.append(candidate)
        return -sum(planner._midpoint(d.recommendation) for d in candidate if d.recommendation)
    with canonical_distance_policy(), opportunity_search_policy():
        selected = planner._allocate_visible_distance_ranges(days,states,(8,9),CONFIG,
            allocation_selector=score,allocation_candidate_limit=1)
    assert scored and selected in scored
    assert all(d.recommendation.distance_target_miles is None for d in days if d.recommendation)
    assert not CANONICAL_DISTANCE_TARGETS.get()


def test_archive_retains_target_and_adherence_bounds_without_schema_migration():
    _, days = program()
    rec = days[0].recommendation.model_copy(update={'distance_target_miles':4.})
    with sqlite3.connect(':memory:') as db:
        db.row_factory = sqlite3.Row
        initialize(db)
        archive_weekly_prescriptions(db,_schedule(rec.generated_at,rec))
        low,high,payload = db.execute('SELECT distance_low_miles,distance_high_miles,recommendation_json FROM planned_workout_history').fetchone()
        assert (low,high) == (3.7,4.2)
        assert json.loads(payload)['distance_target_miles'] == 4.


def test_offline_switch_restores_after_failure():
    with canonical_distance_policy():
        with pytest.raises(RuntimeError), canonical_distance_policy(False):
            assert not CANONICAL_DISTANCE_TARGETS.get()
            raise RuntimeError('stop')
        assert CANONICAL_DISTANCE_TARGETS.get()
    assert not CANONICAL_DISTANCE_TARGETS.get()


def test_infeasible_target_rejects_candidate_not_bounds_or_unrelated_errors():
    @reject_unrepresentable_program
    def score(*, diagnostic_details, broken=False):
        if broken:
            raise RuntimeError('unrelated bug')
        return rounded_distance_target(2.75, minimum=2.75, maximum=2.75)
    details = {}
    assert score(diagnostic_details=details)[0] == float('inf')
    assert 'unrepresentable_distance_target' in details
    with pytest.raises(RuntimeError, match='unrelated bug'):
        score(diagnostic_details={}, broken=True)


def test_reconciliation_preserves_explicit_target_instead_of_reaveraging_range(monkeypatch):
    states, days = program()
    result = days[0].recommendation
    monkeypatch.setattr(planner, '_elapsed_workout_role', lambda *a, **k: 'easy')
    monkeypatch.setattr(planner, '_select_budgeted_timed_recommendation',
                        lambda *a, **k: (states[0], result))
    sessions = planner._materialize_candidate_sessions((0,), states, [[s] for s in states],
        planner.RecommendationRequest(health_status='normal'), CONFIG,
        projected_distance_ranges={0:(3.7,4.2)}, projected_distance_targets={0:4.})
    assert sessions[0].recommendation.planned_distance_miles == 4.
    assert sessions[0].recommendation.distance_range_miles == (3.7,4.2)


def test_quality_scaling_uses_both_committed_targets_not_range_midpoints():
    _, days = program()
    result = days[0].recommendation.model_copy(update={
        'workout_type': 'tempo_threshold', 'quality_session_type': 'threshold',
        'distance_range_miles': (3.2, 3.8), 'distance_target_miles': 3.6,
        'structure': [WorkoutStep(instruction='Warm up'),
                      WorkoutStep(instruction='Run 18 minutes', duration_minutes=18)],
    })
    # 3.6 / 4.0 reaches the existing 90% no-scaling boundary; averaging
    # either adherence range would incorrectly shrink the work segment.
    scaled = scale_quality_session(result, (3.7, 4.4), (3.2, 3.8),
                                   original_target_miles=4.0)
    assert scaled is result


def test_fixed_recovery_dose_commits_a_tenth_without_enlarging_it():
    states, days = program()
    days = [days[0].model_copy(update={'recommendation':
        days[0].recommendation.model_copy(update={
            'workout_type': 'recovery', 'distance_range_miles': (2.5, 3.0),
        })})]
    with canonical_distance_policy():
        result = planner._allocate_visible_distance_ranges(days, states[:1], (2.5, 3.), CONFIG)
    assert result[0].recommendation.distance_target_miles == 2.7
    assert result[0].recommendation.distance_range_miles == (2.5, 3.)
    assert days[0].recommendation.distance_target_miles is None
