from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from run_analysis.delivered_training import PurposeEvidence, dose_from_profile
from run_analysis.program_constraints import DateRestriction
from run_analysis.web.schemas import RecommendationResponse, WorkoutStep
from run_analysis.workload_accounting import Quantity
from run_analysis.workload_profile import WorkBout, bout_summary
from scripts.program_evidence_adapter import (
    QualityReference, evaluate_accounted, forecast_run, observed_run, quality_credit,
    reconcile_completion, reference_key,
)
from scripts.program_objective_trial import DemandScenario, TrialPolicy
from scripts.shared_program_evaluator import ProgramContext
from scripts.timely_purpose_trial import PurposeTimingPolicy

AT = datetime(2026, 9, 26, tzinfo=timezone.utc)
REF = QualityReference('threshold', 'work_seconds', Quantity.point(900, 'stage'),
                       Quantity(290, 310, basis='stage band'), 'declared test stage, not old prescription')


def point(value):
    return Quantity.point(value, 'fixture')


def recommendation(offset=0, kind='intervals', miles=4.5):
    return RecommendationResponse(generated_at=AT, fitness_state_as_of=AT,
        planned_for=AT+timedelta(days=offset, hours=7), workout_type=kind, title='test',
        distance_target_miles=miles, distance_range_miles=(miles-.2, miles+.2),
        confidence='moderate', readiness='ready', structure=([WorkoutStep(instruction='3x5',
            phase='work', repetitions=3, work_duration_minutes=5, recovery_duration_minutes=2)]
            if kind == 'intervals' else []))


def predicted(offset=0, kind='intervals', miles=4.5):
    return forecast_run(recommendation(offset, kind, miles), key=f'forecast:{offset}',
        issued_at=AT, reference=REF, purpose='threshold', duration=point(miles*10),
        base_rate=point(2), work_rate=point(2), recovery_rate=point(2))


def measured(offset=0, kind='intervals', miles=4.5, minutes=None, bouts=None,
             verified=True, available=None, key=None, reference=REF):
    start = AT+timedelta(days=offset, hours=7)
    minutes = minutes if minutes is not None else miles*10
    end = start+timedelta(minutes=minutes)
    key = key or f'actual:{offset}'
    if bouts is None:
        bouts = (WorkBout(300, 120), WorkBout(300, 120), WorkBout(300, None)) if kind == 'intervals' else ()
    profile = dict(source='observed',
        running_exposure=dict(distance_meters=miles*1609.344, moving_seconds=minutes*60),
        cardiovascular=dict(zone_seconds={'z2': minutes*60}),
        intensity_structure=dict(session_workout_type=kind, **bout_summary(bouts)), limitations=[])
    dose = dose_from_profile(profile, key=key, start=start, end=end, available_at=available or end)
    role = 'quality' if kind == 'intervals' else 'long' if kind == 'long' else 'easy'
    evidence = PurposeEvidence(key, 'threshold', available or end, 'explicit fixture purpose') if verified else None
    return observed_run(profile, dose, role=role, reference=reference,
                        as_of=available or end, purpose_evidence=evidence)


def ctx():
    return ProgramContext((), AT.date(), AT.date(), (AT+timedelta(days=21)).date(),
        (AT+timedelta(days=35)).date(), (44., 88., 15.), TrialPolicy(), DemandScenario(), .01,
        PurposeTimingPolicy(1.))


def fixtures():
    return (measured(-6, 'long', 8.4), measured(-3)), (
        predicted(), predicted(3, 'easy', 4.4), predicted(7, 'long', 8.4), predicted(23))


def evaluate(history, plan, *, as_of=AT, fraction=.5, **kwargs):
    return evaluate_accounted(ctx(), history, plan, as_of=as_of, timezone='UTC', reference=REF,
        purpose_fraction=fraction, history_complete_since=(AT-timedelta(days=90)).date(), **kwargs)


def test_actual_and_forecast_units_dose_and_quality_agree_for_identical_execution():
    actual, expected = measured(), predicted()
    assert actual.miles == expected.miles
    assert [(q.low, q.high) for q in actual.demand] == [(q.low, q.high) for q in expected.demand]
    assert actual.quality.low == expected.quality.low == 1
    assert actual.accounting_key == expected.accounting_key


def test_same_day_upload_removes_forecast_once_and_keeps_relative_scores():
    history, plan = fixtures()
    alt = (*plan[:1], replace(plan[1], start=plan[1].start+timedelta(days=1),
                            end=plan[1].end+timedelta(days=1)), *plan[2:])
    actual = measured()
    changes = []
    for candidate in (plan, alt):
        before = evaluate(history, candidate)
        h, p = reconcile_completion(history, candidate, actual, replaces_key=plan[0].key,
                                    as_of=AT+timedelta(hours=9))
        after = evaluate(h, p, as_of=AT+timedelta(hours=9))
        assert after['decision'] == (AT+timedelta(days=1)).date()
        assert after['constraints']['status'] == 'passes_checked_constraints_conditionally'
        assert sum(r.key == actual.key for r in after['evaluation'].conditional_ledger) == 1
        assert plan[0].key not in {r.key for r in after['evaluation'].conditional_ledger}
        changes.append(before['evaluation'].total_cost-after['evaluation'].total_cost)
    assert changes[0] == pytest.approx(changes[1], abs=1e-12)


@pytest.mark.parametrize('miles,minutes,bouts', [
    (4.3, 43., (WorkBout(300, 120), WorkBout(300, None))),
    (4.7, 50., (WorkBout(300, 120),)*3+(WorkBout(300, None),)),
])
def test_within_distance_band_actual_work_is_not_rewritten_to_target(miles, minutes, bouts):
    history, plan = fixtures()
    actual = measured(miles=miles, minutes=minutes, bouts=bouts)
    h, p = reconcile_completion(history, plan, actual, replaces_key=plan[0].key,
                                as_of=AT+timedelta(hours=9))
    done = evaluate(h, p, as_of=AT+timedelta(hours=9))['evaluation'].conditional_ledger[-len(p)-1]
    assert done.miles == miles
    assert done.demand[0].low == minutes
    assert done.demand[2].low == sum(b.work_seconds for b in bouts)/60
    assert done.quality_credit == min(len(bouts)/3, 1.)


def test_missing_or_different_purpose_is_not_full_credit_or_zero_load():
    unknown = measured(verified=False)
    assert (unknown.quality.low, unknown.quality.high) == (0., 1.)
    different = quality_credit(unknown.bouts, REF, purpose='speed_skill', structure_basis='explicit_segments')
    assert different.high == 0
    assert unknown.demand[2].low == 15
    assert unknown.demand[1].low > 0


def test_missing_structure_remains_unknown_not_zero_quality():
    row = measured(bouts=(), verified=False)
    assert (row.quality.low, row.quality.high) == (0., 1.)
    assert row.demand[2].low == 0 and row.demand[2].high == 45


def test_novel_three_by_500_can_be_forecast_without_prior_matching_sessions():
    ref = QualityReference('high_aerobic', 'work_meters', point(1500), point(500), 'declared new stage')
    rec = recommendation().model_copy(update={'structure': [WorkoutStep(instruction='3x500', phase='work', repetitions=3)]})
    result = forecast_run(rec, key='novel', issued_at=AT, reference=ref, purpose='high_aerobic',
        duration=point(45), base_rate=point(2), work_rate=point(4),
        explicit_bouts=(WorkBout(150, 120, 500), WorkBout(150, 120, 500), WorkBout(150, None, 500)))
    assert result.quality.low == result.quality.high == 1
    assert result.demand[2].low == 7.5


def test_no_history_uses_explicit_unknown_opening_scenarios_not_fake_completion():
    plan = (predicted(), predicted(7, 'long', 8.4), predicted(23))
    low, high = evaluate((), plan, fraction=0), evaluate((), plan, fraction=1)
    assert low['opening_coverage_bounds']['quality'].low == 0
    assert high['opening_coverage_bounds']['quality'].high == 1
    assert low['evaluation'].total_cost > high['evaluation'].total_cost
    assert low['constraints']['status'] == 'constraint_evidence_incomplete'
    assert not low['selection_ready']


@pytest.mark.parametrize('kind', ['reserved_rest', 'health_review', 'unsafe_weather'])
def test_constraints_gate_even_a_finite_favorable_score_and_cover_continuation(kind):
    h, p = fixtures()
    rest = DateRestriction('rest', p[-1].start.date(), p[-1].start.date()+timedelta(days=1),
                           AT, kind, 'explicit restriction')
    result = evaluate(h, p, restrictions=(rest,))
    assert result['evaluation'].total_cost > 0
    assert not result['constraint_eligible']
    assert result['constraints']['status'] == 'constraint_conflict'
    assert result['constraints']['recovery_feasibility'] == 'not_established'


def test_same_day_scope_does_not_make_tomorrow_evidence_available():
    h, p = fixtures()
    h, p = reconcile_completion(h, p, measured(), replaces_key=p[0].key, as_of=AT+timedelta(hours=9))
    future = DateRestriction('future', p[0].start.date(), p[0].start.date()+timedelta(days=1),
                             AT+timedelta(days=1), 'reserved_rest', 'not yet known')
    result = evaluate(h, p, as_of=AT+timedelta(hours=9), restrictions=(future,))
    assert result['constraints']['ignored_unavailable_restriction_keys'] == ['future']


def test_afternoon_refresh_and_next_morning_do_not_move_accounting_boundary_again():
    h, p = fixtures()
    h, p = reconcile_completion(h, p, measured(), replaces_key=p[0].key, as_of=AT+timedelta(hours=9))
    results = [evaluate(h, p, as_of=at) for at in
               (AT+timedelta(hours=9), AT+timedelta(hours=20), AT+timedelta(days=1))]
    assert len({r['evaluation'].total_cost for r in results}) == 1
    assert len({r['evaluation'].fingerprint for r in results}) == 1


@pytest.mark.parametrize('fraction', [0., .5, 1.])
def test_uncertain_credit_scenario_precedes_max_history_envelope(fraction):
    h, p = fixtures()
    uncertain = Quantity(.2, .9, basis='same uncertain purpose before and after')
    p = (replace(p[0], quality=uncertain), *p[1:])
    actual = replace(measured(), quality=uncertain)
    before = evaluate(h, p, fraction=fraction)['evaluation'].parts['timing']['quality']['daily'][1:]
    h, p = reconcile_completion(h, p, actual, replaces_key=p[0].key, as_of=AT+timedelta(hours=9))
    after = evaluate(h, p, fraction=fraction, as_of=AT+timedelta(hours=9))['evaluation'].parts['timing']['quality']['daily']
    assert [r['loss'] for r in before] == pytest.approx([r['loss'] for r in after], abs=1e-14)


def test_unknown_initial_history_is_anchored_across_rest_day_refresh():
    p = (predicted(3), predicted(7, 'long', 8.4), predicted(23))
    before = evaluate((), p)['evaluation'].parts['timing']['quality']['daily'][1:]
    after = evaluate((), p, as_of=AT+timedelta(days=1))['evaluation'].parts['timing']['quality']['daily']
    assert [r['loss'] for r in before] == pytest.approx([r['loss'] for r in after], abs=1e-14)


def test_completed_day_forecast_must_be_removed_and_duplicates_rejected():
    h, p = fixtures()
    with pytest.raises(ValueError, match='Remove completed-day'):
        evaluate((*h, measured()), p, as_of=AT+timedelta(hours=9))
    with pytest.raises(ValueError, match='already accounted'):
        reconcile_completion((*h, measured()), p, measured(), replaces_key=p[0].key, as_of=AT+timedelta(hours=9))


def test_completion_cannot_use_post_run_issue_changed_units_or_aliased_actual():
    h,p = fixtures()
    actual = measured()
    late = replace(p[0],start=p[0].start+timedelta(hours=2),end=p[0].end+timedelta(hours=2),
                   available_at=actual.start+timedelta(minutes=1))
    with pytest.raises(ValueError,match='pre-run'):
        reconcile_completion(h,(late,*p[1:]),actual,replaces_key=late.key,as_of=AT+timedelta(hours=9))
    with pytest.raises(ValueError,match='Changed accounting'):
        reconcile_completion(h,p,replace(actual,accounting_key='changed'),replaces_key=p[0].key,as_of=AT+timedelta(hours=9))
    with pytest.raises(ValueError,match='Overlapping actual'):
        reconcile_completion((*h,replace(actual,key='alias')),p,actual,replaces_key=p[0].key,as_of=AT+timedelta(hours=9))


def test_future_observation_and_mixed_units_or_reference_rejected():
    h, p = fixtures()
    with pytest.raises(ValueError, match='available completed'):
        evaluate((*h, measured()), p)
    with pytest.raises(ValueError, match='accounting units'):
        evaluate(h, (replace(p[0], accounting_key='hrr_trimp'), *p[1:]))
    with pytest.raises(ValueError, match='stage reference'):
        evaluate(h, (replace(p[0], quality_reference_key='other'), *p[1:]))


def test_distance_guard_rejects_spike_without_normalizing_actual_history():
    h, p = fixtures()
    spike = predicted(0, 'long', 12.)
    result = evaluate(h, (spike, *p[1:]))
    assert not result['constraint_eligible']
    assert result['constraints']['sessions'][0]['distance_guard'] == 'exceeds_guard'


def test_overnight_event_is_explicitly_unsupported_not_silently_misdated():
    h, p = fixtures()
    bad = replace(p[0], start=AT+timedelta(hours=23), end=AT+timedelta(days=1, hours=1))
    with pytest.raises(ValueError, match='Overnight'):
        evaluate(h, (bad, *p[1:]))


def test_forecast_structured_work_cannot_be_squeezed_into_too_short_total_time():
    with pytest.raises(ValueError, match='accommodate'):
        forecast_run(recommendation(), key='bad', issued_at=AT, reference=REF, purpose='threshold',
            duration=point(10), base_rate=point(2), work_rate=point(4))


def test_late_upload_is_not_included_before_it_is_available():
    actual = measured(available=AT+timedelta(days=1, hours=10))
    h, p = fixtures()
    with pytest.raises(ValueError, match='Available actual'):
        reconcile_completion(h, p, actual, replaces_key=p[0].key, as_of=AT+timedelta(hours=20))
    h, p = reconcile_completion(h, p, actual, replaces_key=p[0].key,
                                as_of=AT+timedelta(days=1, hours=10))
    result = evaluate(h, p, as_of=AT+timedelta(days=1, hours=10))
    assert result['decision'] == (AT+timedelta(days=1)).date()


@pytest.mark.parametrize('factor', [1., .95, 1.05])
def test_eight_day_chained_accounting_refresh_and_upload_with_imperfect_distance(factor):
    # Exercise the connected accounting path, not a scheduler/search simulation.
    h, p = fixtures()
    last_upload = None
    completions = 0
    for offset in range(8):
        today = AT+timedelta(days=offset)
        morning = evaluate(h, p, as_of=today)
        if last_upload is not None:
            assert morning['evaluation'].total_cost == pytest.approx(last_upload, abs=1e-12)
            last_upload = None
        evening = evaluate(h, p, as_of=today+timedelta(hours=19))
        assert evening['evaluation'].total_cost == morning['evaluation'].total_cost
        scheduled = next((r for r in p if r.start.date() == today.date()), None)
        if scheduled is not None:
            kind = 'intervals' if scheduled.role == 'quality' else scheduled.role
            actual = measured(offset, kind, scheduled.miles*factor)
            # Variable upload/run hour does not manufacture a date-level penalty.
            delay = timedelta(hours=0 if offset % 2 else 11)
            actual = replace(actual, start=actual.start+delay, end=actual.end+delay,
                             available_at=today+timedelta(hours=20))
            h, p = reconcile_completion(h, p, actual, replaces_key=scheduled.key,
                                        as_of=actual.available_at)
            after = evaluate(h, p, as_of=actual.available_at)
            assert after['constraint_eligible']
            assert after['evaluation'].conditional_ledger[len(h)-1].miles == actual.miles
            last_upload = after['evaluation'].total_cost
            completions += 1
    assert completions == 3
