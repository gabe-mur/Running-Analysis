from dataclasses import replace
import pytest

from run_analysis.prescription_load import projected_running_minutes, structured_duration_bounds
from run_analysis.web.schemas import WorkoutStep
from run_analysis import adherence_projection as projection
from run_analysis.workload_profile import prescribed_profile
from scripts.expected_completion import expected_run, ExpectedCompletion
from scripts.sandbox_recovery_adapter import projected_entry
from scripts.audit_sandbox_recovery_adapter import phase_assumptions
from test_expected_completion import setup
from test_recommendation import CONFIG


def prescription():
    template, history, rec = setup('intervals', 3.7)
    rec.structure = [WorkoutStep(instruction='Warmup', duration_minutes=12),
        WorkoutStep(instruction='Work', phase='work', repetitions=3, work_duration_minutes=5,
                    recovery_duration_minutes=2),
        WorkoutStep(instruction='Cooldown', duration_minutes=10)]
    return template, history, rec


def test_forecast_difficulty_and_synthetic_execution_share_41_minutes():
    _, _, rec = prescription()
    expected = expected_run(rec, 11)
    actual = projection._observed_human_difficulty(rec, 3.7, 11, rec.workout_type, too_intense=False)
    assert structured_duration_bounds(rec) == (41, 41)
    assert expected.moving_minutes == expected.difficulty.moving_minutes == actual.moving_minutes == 41
    assert rec.planned_distance_miles == expected.distance_miles == 3.7
    assert prescribed_profile(rec)['running_exposure']['minimum_duration_range_seconds'] == (2460, 2460)
    projected_entry(rec, expected, phase_assumptions())
    with pytest.raises(ValueError, match='accommodate'):
        projected_entry(rec, replace(expected, moving_minutes=40.7), phase_assumptions())


def test_expected_and_completed_states_still_agree_after_duration_floor():
    from datetime import timedelta
    template, history, rec = prescription()
    expected = ExpectedCompletion(template, history, CONFIG)
    completed = ExpectedCompletion(template, [*history, expected_run(rec, 11)], CONFIG)
    for days in (1, 2, 7, 21):
        at = rec.planned_for+timedelta(days=days)
        assert expected.at(at, [rec]) == completed.at(at)


def test_ranges_blocks_and_between_rep_recovery_not_double_counted():
    _, _, rec = prescription()
    rec.structure[1] = WorkoutStep(instruction='Work', phase='work', repetitions=3,
        work_duration_range_minutes=(4, 5), recovery_duration_minutes=2, duration_minutes=18)
    assert structured_duration_bounds(rec) == (40, 41)
    assert projected_running_minutes(rec, 3.7, 11) == 41
    rec.structure[1].duration_minutes = 25
    assert structured_duration_bounds(rec) == (47, 47)


def test_easy_and_long_distance_duration_unchanged_and_abandoned_structure_not_forced():
    for kind, miles in [('easy', 4), ('long', 9)]:
        _, _, rec = setup(kind, miles)
        assert projected_running_minutes(rec, miles, 11) == miles*11
    _, _, rec = prescription()
    assert projected_running_minutes(rec, 3.7, 11, honor_structure=False) == pytest.approx(40.7)
    actual = projection._observed_human_difficulty(rec, 3.7, 11, projection.WorkoutType.EASY, too_intense=False)
    assert actual.moving_minutes == pytest.approx(40.7)


@pytest.mark.parametrize('miles,pace', [(0, 11), (3, float('nan')), (-1, 10)])
def test_invalid_synthetic_inputs_rejected(miles, pace):
    _, _, rec = prescription()
    with pytest.raises(ValueError):
        projected_running_minutes(rec, miles, pace)


def test_daily_simulator_upload_uses_same_structured_duration(monkeypatch):
    from datetime import datetime, timezone
    from test_adherence_projection import _single_then_empty_schedule, _state
    _, _, prescribed = prescription()
    template = _state(as_of=datetime(2026,9,2,12,tzinfo=timezone.utc),typical_easy_run_miles=4.)
    history = []
    original_schedule = _single_then_empty_schedule()
    def schedule(*args, **kwargs):
        result = original_schedule(*args, **kwargs)
        if result.days:
            day = result.days[0]
            old = day.recommendation
            new = prescribed.model_copy(update={'planned_for':old.planned_for,
                'fitness_state_as_of':old.fitness_state_as_of,'generated_at':old.generated_at})
            day = day.model_copy(update={'recommendation':new,'day_role':'quality_run'})
            result = result.model_copy(update={'days':[day],'planning_days':[day],
                'projected_distance_range_miles':new.distance_range_miles})
        return result
    monkeypatch.setattr(projection,'build_weekly_schedule',schedule)
    completions=[]
    weeks=projection.simulate_adherence(template,history,CONFIG,template.as_of,
        weeks=1,simulation_days=2,execution_observer=lambda run,latest:completions.append(run))
    assert len(completions)==1
    assert completions[0].moving_minutes==completions[0].difficulty.moving_minutes==41
    upload=next(s for s in weeks[0].replan_snapshots if s.trigger=='post_upload')
    assert (upload.generated_at-upload.source_activity_at).total_seconds()==41*60
    assert upload.recovery_surprise_units==pytest.approx(0,abs=1e-4)
