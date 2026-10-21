from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from run_analysis.continuous_training_dose import (
    ContinuousDoseReference, continuous_dose, general_quality_credit, supported_work_minutes,
)
from run_analysis.signal_structure import SignalSpan
from run_analysis.training_signals import integrate_cardio_trace
from run_analysis.workload_accounting import Quantity
from run_analysis.workload_profile import WorkBout
from scripts.continuous_program_adapter import with_continuous_trace, evaluate_continuous
from scripts.program_evidence_adapter import AccountedRun, reconcile_completion
from scripts.program_objective_trial import TrialPolicy, DemandScenario
from scripts.shared_program_evaluator import ProgramContext
from scripts.timely_purpose_trial import PurposeTimingPolicy
from scripts.program_objective_trial import EasyDurationPolicy
from run_analysis.session_decision import (CoachingContract, DoseLimits, PrescriptionBinding,
    ProgramDecisionInput, CurrentContext, DatedResponse)
from run_analysis.recovery_response import ResponseExpectation

AT = datetime(2026, 9, 26, tzinfo=timezone.utc)
REF = ContinuousDoseReference('explicit test scenario')


def point(n):
    return Quantity.point(n, 'fixture')


def run(day=0, source='planned', role='quality', hrr=.8, work=12., key=None):
    start = AT+timedelta(days=day, hours=7)
    end = start+timedelta(minutes=40)
    raw = AccountedRun(key or f'{source}:{day}', start, end,
        AT if source == 'planned' else end, 4., role,
        (point(40), point(999), point(work)), point(0), source,
        tuple(WorkBout(work*10, 60) for _ in range(6)) if work else (),
        work_evidence_basis='prescribed' if source == 'planned' else 'explicit_segments')
    return convert(raw, (SignalSpan(0, 2400, hrr, hrr),))[0]


def convert(raw, spans, **kwargs):
    return with_continuous_trace(raw, spans, reference=REF, trace_key=raw.key,
        trace_source=raw.source, elapsed_seconds=2400, **kwargs)


def context():
    return ProgramContext((), AT.date(), AT.date(), (AT+timedelta(days=21)).date(),
        (AT+timedelta(days=35)).date(), (40., 40., 15.), TrialPolicy(), DemandScenario(),
        .01, PurposeTimingPolicy(1.), demand_accounting_key=REF.accounting_key)


def evaluate(history, future, as_of=AT, fraction=.5, ctx=None):
    return evaluate_continuous(ctx or context(), history, future, reference=REF,
        as_of=as_of, timezone='UTC', purpose_fraction=fraction)


def decision_inputs(rows, *, current=None, responses=()):
    contracts = {
        'easy': CoachingContract(AT, 'frozen test stage', 'easy', 'easy_intent',
            DoseLimits(Quantity(3., 4.5, basis='stage')), DoseLimits(point(2.5)),
            DoseLimits(Quantity(4.6, 4.8, basis='trial')), trial_target=4.6),
        'long': CoachingContract(AT, 'frozen test stage', 'long', 'easy_intent',
            DoseLimits(point(4.)), DoseLimits(point(3.))),
        'quality': CoachingContract(AT, 'frozen test stage', 'threshold', 'threshold_intent',
            DoseLimits(point(4.), Quantity(600, 900, basis='stage'), Quantity(60, 300, basis='stage'),
                       Quantity(.1, 1.5, basis='stage')),
            DoseLimits(point(4.), point(360), Quantity(60, 300, basis='stage'), Quantity(.1, 1.5, basis='stage'))),
    }
    return ProgramDecisionInput('test-stage', AT, 'declared fixture not athlete validation', tuple(
        PrescriptionBinding(r.key, contracts[r.role].purpose, contracts[r.role].intensity_key, contracts[r.role])
        for r in rows), current, tuple(responses))


def decide_program(rows, *, current=None, responses=(), history=None, inputs=None, as_of=AT):
    history = (run(-2, source='observed', role='long', work=0),) if history is None else history
    return evaluate_continuous(replace(context(), easy_duration_policy=EasyDurationPolicy(10,25,.01,'R1 test',7.)),
        history, rows, reference=REF, as_of=as_of, timezone='UTC', purpose_fraction=.5,
        decision_input=inputs or decision_inputs(rows, current=current, responses=responses))


def test_connected_decision_needs_policy_inputs_not_just_a_finite_score():
    result = evaluate((run(-2, source='observed', role='long', work=0),), (run(role='easy', work=0),))
    assert result['constraint_eligible']
    assert result['suitability']['status'] == 'not_assessed'
    assert not result['decision_eligible']
    assert not result['selection_ready']


def test_current_mild_fatigue_retains_easy_but_requires_quality_alternative():
    tired = CurrentContext(AT.date(), AT, 'little_tired', 'current report')
    easy = decide_program((run(role='easy', work=0),), current=tired)
    full = decide_program((run(),), current=tired)
    reduced = decide_program((run(work=6),), current=tired)
    assert easy['decision_eligible']
    assert easy['suitability']['action'] == 'continue_provisionally'
    assert not full['decision_eligible']
    assert reduced['decision_eligible']
    assert reduced['suitability']['action'] == 'propose_reduced_trial'
    tomorrow = decide_program((run(1),), current=tired)
    assert tomorrow['decision_eligible']
    assert tomorrow['suitability']['future_context_recheck_required']


def test_health_context_changes_decision_not_physical_load_and_cannot_be_waited_away():
    rows = (run(role='easy', work=0),)
    plain = decide_program(rows)
    concern = CurrentContext((AT-timedelta(days=1)).date(), AT-timedelta(hours=1), 'health_concern', 'report')
    blocked = decide_program(rows, current=concern)
    assert plain['evaluation'].total_cost == blocked['evaluation'].total_cost
    assert plain['evaluation'].fingerprint != blocked['evaluation'].fingerprint
    assert not blocked['decision_eligible']
    assert blocked['suitability']['action'] == 'review_current_context'
    assert not decide_program((run(1, role='easy', work=0),), current=concern)['decision_eligible']


def test_dated_adverse_response_needs_fresh_context_not_duplicate_cardio_load():
    previous = run(-2, source='observed', role='long', work=0)
    report = DatedResponse(previous.key, previous.start-timedelta(hours=1), previous.end,
        ResponseExpectation('pre-run effort band', effort=Quantity(2,4,basis='declared')), 'actual report', effort=8.)
    rows = (run(role='easy', work=0),)
    missing = decide_program(rows, history=(previous,), responses=(report,))
    assert missing['suitability']['action'] == 'request_current_context'
    clear = decide_program(rows, history=(previous,), responses=(report,),
        current=CurrentContext(AT.date(), AT, 'usual', 'fresh report'))
    assert clear['decision_eligible']
    import json
    json.dumps(clear['suitability'], allow_nan=False)
    assert clear['evaluation'].total_cost == missing['evaluation'].total_cost
    with pytest.raises(ValueError, match='pre-run expectation'):
        decide_program(rows, history=(previous,), responses=(replace(report,expected_at=previous.end),))


def test_unavailable_response_and_context_do_not_change_evaluation_identity():
    rows = (run(role='easy',work=0),)
    baseline = decide_program(rows)
    future = CurrentContext((AT+timedelta(days=1)).date(), AT+timedelta(days=1), 'health_concern', 'future report')
    report = DatedResponse('unknown_future_activity', AT, AT+timedelta(days=1),
        ResponseExpectation('future',effort=point(2.)), 'future report', effort=8.)
    ignored = decide_program(rows,current=future,responses=(report,))
    assert ignored['evaluation'].fingerprint == baseline['evaluation'].fingerprint
    assert ignored['suitability'] == baseline['suitability']


def test_small_actual_excess_does_not_trigger_old_review_cliff():
    rows = (run(role='easy', work=0),)
    previous = run(-2, source='observed', role='long', work=0)
    ordinary = decide_program(rows, history=(previous,))
    actual = replace(previous, miles=4.1, demand=tuple(q.scale(1.025) for q in previous.demand))
    excess = decide_program(rows, history=(actual,))
    assert ordinary['decision_eligible'] and excess['decision_eligible']
    assert excess['evaluation'].weighted_demand_cost > ordinary['evaluation'].weighted_demand_cost
    assert not excess['suitability']['physiological_recovery_established']


def test_full_sequence_contract_checks_include_tail_and_missing_recovery_structure():
    rows = (run(role='easy', work=0), run(22))
    inputs = decision_inputs(rows)
    assert decide_program(rows, inputs=inputs)['decision_eligible']
    missing = replace(inputs, prescriptions=inputs.prescriptions[:1])
    result = decide_program(rows, inputs=missing)
    assert not result['decision_eligible']
    assert result['suitability']['session_checks'][rows[1].key]['problems'] == ['prescription_contract_missing']
    quality = run()
    quality = replace(quality, bouts=tuple(replace(b,recovery_after_seconds=None) for b in quality.bouts))
    assert not decide_program((quality,))['decision_eligible']


def test_duration_floor_is_enforced_without_suppressing_actual_short_run_credit():
    short = replace(run(role='easy',work=0), demand=(point(5),point(5),point(0)),end=AT+timedelta(hours=7,minutes=5))
    result = decide_program((short,))
    assert not result['decision_eligible']
    assert 'planned_easy_below_minimum' in result['suitability']['reasons']
    unknown = replace(short,demand=(Quantity(5,15,basis='forecast'),point(5),point(0)),
                      end=AT+timedelta(hours=7,minutes=15))
    assert 'planned_easy_minimum_uncertain' in decide_program((unknown,))['suitability']['reasons']
    assert decide_program(())['suitability']['action'] == 'rest_candidate'


def test_missing_or_conflicting_prescription_intent_does_not_gain_clearance():
    rows = (run(),)
    inputs = decision_inputs(rows)
    binding = replace(inputs.prescriptions[0], intensity_key='unsupported_intent')
    result = decide_program(rows, inputs=replace(inputs,prescriptions=(binding,)))
    assert not result['decision_eligible']
    assert 'prescription_purpose_or_intensity_mismatch' in result['suitability']['session_checks'][rows[0].key]['problems']


def test_quality_trial_uses_committed_work_not_uncertainty_lower_edge():
    original = run(work=18.)
    row = replace(original, bouts=tuple(replace(b,duration_range_seconds=(b.work_seconds*.9,b.work_seconds*1.1))
                                        for b in original.bouts),
        demand=(original.demand[0],original.demand[1],Quantity(16.2,19.8,basis='forecast range')))
    inputs = decision_inputs((row,))
    binding = inputs.prescriptions[0]
    contract = replace(binding.contract, trial=DoseLimits(point(4.),Quantity(900,1200,basis='trial'),
        Quantity(120,400,basis='trial'),Quantity(.1,1.5,basis='trial')),trial_target=1080.)
    inputs = replace(inputs,prescriptions=(replace(binding,contract=contract),))
    result = decide_program((row,),inputs=inputs)
    assert result['decision_eligible']
    assert result['suitability']['action'] == 'propose_bounded_trial'
    inputs = replace(inputs,prescriptions=(replace(binding,contract=replace(contract,trial_target=1188.)),))
    assert not decide_program((row,),inputs=inputs)['decision_eligible']


def test_same_mean_different_trace_not_same_load():
    steady = continuous_dose([SignalSpan(0, 1200, .7, .7)], 1200, REF)
    varied = continuous_dose([SignalSpan(0, 600, .5, .5), SignalSpan(600, 1200, .9, .9)], 1200, REF)
    assert varied['mean_hr_shortcut_load'] == pytest.approx(steady['cardio'].low)
    assert varied['cardio'].low > steady['cardio'].low


def test_no_zone_boundary_jump_and_hr_monotonicity():
    values = [continuous_dose([SignalSpan(0, 60, r, r)], 60, REF)
              for r in (.599999, .60, .600001, .799999, .800001)]
    assert [v['cardio'].low for v in values] == sorted(v['cardio'].low for v in values)
    assert values[2]['high_effort_minutes'].low < .00001
    assert values[4]['cardio'].low-values[3]['cardio'].low < .00002


def test_crossing_integration_is_invariant_to_sample_splitting():
    whole = continuous_dose([SignalSpan(0, 1200, .4, .8)], 1200, REF)
    split = continuous_dose([SignalSpan(0, 600, .4, .6), SignalSpan(600, 1200, .6, .8)], 1200, REF)
    for key in ('cardio', 'high_effort_minutes'):
        assert whole[key].low == pytest.approx(split[key].low, rel=1e-8)


def test_short_work_survives_hr_lag_without_exact_template():
    short = run(hrr=.6, work=10.)
    assert short.quality.low == short.quality.high == pytest.approx(10/15)
    assert len(short.bouts) == 6
    assert short.demand[1].low == pytest.approx(40)  # Not old 999 points.


def test_novel_3x500_has_no_repeat_history_or_minimum_rep_duration_gate():
    original = replace(run(hrr=.6, work=6), bouts=tuple(WorkBout(120, 90, 500) for _ in range(3)))
    converted, _ = convert(original, (SignalSpan(0, 2400, .6, .6),))
    assert converted.quality.low == pytest.approx(.4)
    assert sum(b.distance_meters for b in converted.bouts) == 1500


def test_supported_continuous_quality_does_not_need_repetitions_or_template():
    work = supported_work_minutes((WorkBout(720, None),), point(12),
        evidence_basis='recorded_continuous_lap', source='observed', elapsed_seconds=2400)
    credit = general_quality_credit(work, role='quality', reference=REF)
    assert credit.low == credit.high == .8


def test_no_hr_plus_work_double_credit():
    # HR intensity changes load, but cannot add a second copy of work coverage.
    lower, higher = run(hrr=.6, work=10), run(hrr=.9, work=10)
    assert lower.quality.low == higher.quality.low == pytest.approx(2/3)
    assert higher.demand[1].low > lower.demand[1].low


def test_separately_converted_actual_and_forecast_agree():
    actual, forecast = run(source='observed'), run(source='planned')
    assert actual.demand == forecast.demand
    assert actual.quality == forecast.quality
    assert actual.accounting_key == forecast.accounting_key


def test_equal_integral_does_not_erase_sustained_structure():
    sustained = continuous_dose([SignalSpan(0, 600, .9, .9), SignalSpan(600, 1200, .6, .6)], 1200, REF)
    intermittent = continuous_dose([SignalSpan(i*100, (i+1)*100, r, r)
        for i in range(12) for r in ([.9] if i % 2 == 0 else [.6])], 1200, REF)
    assert sustained['cardio'].low == pytest.approx(intermittent['cardio'].low)
    assert sustained['sustained_exposure'] != intermittent['sustained_exposure']


def test_hot_easy_not_reclassified_but_load_retained():
    easy, hard_hr = run(role='easy', hrr=.6, work=0), run(role='easy', hrr=.9, work=0)
    assert hard_hr.quality.low == hard_hr.quality.high == 0
    assert hard_hr.demand[1].low > easy.demand[1].low


def test_unknown_context_is_not_fabricated_positive_evidence():
    credit = general_quality_credit(point(12), role='unknown', reference=REF)
    assert credit.low == 0 and credit.high == .8


def test_missing_hr_stays_bounded_not_zero_or_fully_observed():
    dose = continuous_dose([SignalSpan(0, 600, .85, .85)], 1200, REF)
    assert dose['coverage'] == .5
    assert dose['cardio'].low < dose['cardio'].high
    assert dose['high_effort_minutes'].low == pytest.approx(10)
    assert dose['high_effort_minutes'].high > 10


def test_exclude_unused_density_without_changing_integral():
    spans = [SignalSpan(0, 120, .6, .9)]
    previous, lean = integrate_cardio_trace(spans, 120), integrate_cardio_trace(spans, 120, compute_density=False)
    assert previous['total_load'] == lean['total_load']
    assert lean['density_peak_candidate'] is None and lean['response_load_candidate'] is None


def test_supplied_downstream_scenario_reaches_accounted_evaluator_without_target_drift():
    from scripts.program_objective_trial import TrialSession
    history = (run(-3, 'observed', 'long', .6, 0),)
    future = (run(1, role='easy', hrr=.6, work=0), run(34, role='easy', hrr=.6, work=0))
    base = evaluate(history, future)
    after = TrialSession('background_long', (AT+timedelta(days=36)).date(), 8.4,
                         'long', (point(84), point(84), point(0)), 0., 'planned')
    ctx = replace(context(), downstream=(after,), downstream_end=(AT+timedelta(days=42)).date(),
                  downstream_basis='explicit test continuation, not an issued prescription')
    result = evaluate(history, future, ctx=ctx)
    before_eval, after_eval = base['evaluation'], result['evaluation']
    assert before_eval.parts['targets'] == after_eval.parts['targets']
    assert before_eval.parts['delivery'] == after_eval.parts['delivery']
    assert before_eval.delivery_cost == after_eval.delivery_cost
    assert after_eval.weighted_demand_cost > before_eval.weighted_demand_cost
    assert after_eval.downstream_accounting['keys'] == ('background_long',)
    assert after_eval.conditional_ledger.count(after) == 1
    assert not result['selection_ready']


@pytest.mark.parametrize('fraction', [0., .5, 1.])
@pytest.mark.parametrize('completed_role', ['quality', 'unknown'])
@pytest.mark.parametrize('stage_work', [None,3.,12.,30.])
def test_connected_expected_upload_rollover_relative_score_parity(fraction, completed_role,stage_work):
    history = (run(-6, 'observed', 'long', .6, 0), run(-3, 'observed'))
    expected = run(role=completed_role)
    actual = replace(expected, key='actual', source='observed', available_at=expected.end,
                     work_evidence_basis='explicit_segments')
    costs = []
    ctx=replace(context(),policy=replace(context().policy,quality_work_minutes=stage_work))
    for offset in (3, 4):
        remaining = (run(offset), run(7, role='long', hrr=.6, work=0), run(23))
        before = evaluate(history, (expected, *remaining), fraction=fraction,ctx=ctx)
        h, p = reconcile_completion(history, (expected, *remaining), actual, replaces_key=expected.key, as_of=actual.end)
        after = evaluate(h, p, actual.end, fraction,ctx=ctx)
        rolled = evaluate(h, p, AT+timedelta(days=1), fraction,ctx=ctx)
        assert after['evaluation'].total_cost == pytest.approx(rolled['evaluation'].total_cost, abs=1e-12)
        costs.append((before['evaluation'].total_cost, after['evaluation'].total_cost))
    assert costs[0][0]-costs[1][0] == pytest.approx(costs[0][1]-costs[1][1], abs=1e-12)


def test_reject_mixed_units_or_old_normalization():
    with pytest.raises(ValueError, match='units'):
        evaluate((), (replace(run(), accounting_key='edwards'),))
    with pytest.raises(ValueError, match='Normalization'):
        evaluate((), (run(),), ctx=replace(context(), demand_accounting_key=''))


def test_reject_wrong_trace_or_outside_reference():
    row = run()
    with pytest.raises(ValueError, match='identity'):
        with_continuous_trace(row, (), reference=REF, trace_key='wrong', trace_source='observed', elapsed_seconds=2400)
    with pytest.raises(ValueError, match='outside reference'):
        convert(row, (), outside_reference_seconds=1)
    with pytest.raises(ValueError, match='elapsed'):
        with_continuous_trace(row, (), reference=REF, trace_key=row.key, trace_source=row.source, elapsed_seconds=2399)


@pytest.mark.parametrize('quality_hrr', [.8, .85, .9])
@pytest.mark.parametrize('quality_minutes', [10., 15., 20.])
def test_reference_sensitivity_never_rejects_short_quality(quality_hrr, quality_minutes):
    ref = replace(REF, quality_hrr=quality_hrr, quality_minutes=quality_minutes)
    credit = general_quality_credit(point(10), role='quality', reference=ref)
    assert credit.low == pytest.approx(min(1, 10/quality_minutes))


@pytest.mark.parametrize('hrr', [.6, .7, .85, .95])
def test_no_work_quality_label_cannot_manufacture_credit(hrr):
    assert run(hrr=hrr, work=0).quality.low == run(hrr=hrr, work=0).quality.high == 0


@pytest.mark.parametrize('padding', [10, 20, 60])
def test_padding_increases_load_without_quality_credit(padding):
    baseline = run(hrr=.7, work=2.)
    padded = replace(baseline, end=baseline.end+timedelta(minutes=padding),
                     demand=(point(40+padding), baseline.demand[1], baseline.demand[2]))
    adjusted, _ = with_continuous_trace(padded,
        (SignalSpan(0, 2400, .7, .7), SignalSpan(2400, (40+padding)*60, .7, .7)),
        reference=REF, trace_key=padded.key, trace_source=padded.source,
        elapsed_seconds=(40+padding)*60)
    assert adjusted.quality.low == baseline.quality.low == pytest.approx(2/15)
    assert adjusted.demand[1].low > baseline.demand[1].low


def test_missing_work_is_unknown_not_observed_zero_or_hr_filled():
    raw = replace(run(work=0), work_evidence_basis='unknown', demand=(point(40), point(1), Quantity(0, 40)))
    adjusted, report = convert(raw, (SignalSpan(0, 2400, .9, .9),))
    assert adjusted.quality.low == 0 and adjusted.quality.high == 1
    assert report['quality_evidence_status'] == 'unresolved_extent'
    assert report['quality_work_minutes'] is None


def test_provenance_and_segment_amount_must_support_work():
    raw = run(work=10)
    with pytest.raises(ValueError, match='provenance'):
        convert(replace(raw, source='observed', available_at=raw.end), ())
    with pytest.raises(ValueError, match='must agree'):
        convert(replace(raw, demand=(point(40), point(1), point(20))), ())
    with pytest.raises(ValueError, match='provenance'):
        convert(replace(raw, work_evidence_basis='unknown'), ())


def test_missing_hr_does_not_erase_supported_short_work():
    row, report = convert(run(work=6), ())
    assert row.quality.low == row.quality.high == pytest.approx(.4)
    assert report['coverage'] == 0
    assert row.demand[1].low == 0 < row.demand[1].high


def test_padding_has_no_quality_delivery_advantage_in_connected_score():
    baseline = run(work=2)
    padded = replace(baseline, end=baseline.end+timedelta(minutes=20),
                     demand=(point(60), point(1), baseline.demand[2]))
    padded, _ = with_continuous_trace(padded, (SignalSpan(0, 3600, .8, .8),),
        reference=REF, trace_key=padded.key, trace_source=padded.source, elapsed_seconds=3600)
    old, new = evaluate((), (baseline,)), evaluate((), (padded,))
    assert new['evaluation'].delivery_cost == old['evaluation'].delivery_cost
    assert new['evaluation'].total_cost > old['evaluation'].total_cost


def test_evaluator_surfaces_missing_work_and_rejects_stale_credit():
    row = replace(run(work=0), work_evidence_basis='unknown',
                  demand=(point(40), point(1), Quantity(0, 40)))
    row, _ = convert(row, (SignalSpan(0, 2400, .9, .9),))
    assert evaluate((), (row,))['unresolved_quality_work'] == [row.key]
    with pytest.raises(ValueError, match='Recompute'):
        evaluate((), (replace(run(work=2), quality=point(1)),))


def test_forecast_work_duration_bounds_remain_bounds():
    row = replace(run(work=10), bouts=(WorkBout(600, None, duration_range_seconds=(480, 720)),),
                  demand=(point(40), point(1), Quantity(8, 12)))
    row, _ = convert(row, ())
    assert row.quality.low == pytest.approx(8/15)
    assert row.quality.high == pytest.approx(12/15)
