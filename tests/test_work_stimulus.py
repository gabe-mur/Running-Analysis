from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from run_analysis.continuous_training_dose import ContinuousDoseReference
from run_analysis.signal_structure import SignalSpan
from run_analysis.work_stimulus import (
    StimulusTarget, WorkSignals, assess_work_stimulus, _shared_reference_bounds, _integrated_suitability,
)
from run_analysis.workload_accounting import Quantity
from run_analysis.workload_profile import WorkBout
from scripts.continuous_program_adapter import with_continuous_trace
from scripts.program_evidence_adapter import AccountedRun, reconcile_completion
from scripts.program_objective_trial import TrialPolicy, DemandScenario
from scripts.shared_program_evaluator import ProgramContext
from scripts.stimulus_program_adapter import evaluate_stimulus_program
from scripts.timely_purpose_trial import PurposeTimingPolicy

AT = datetime(2026, 9, 26, tzinfo=timezone.utc)
REF = ContinuousDoseReference('test-only accounting')
HR = StimulusTarget('elevated cardiovascular work', 'hrr', (.7, .8, .9, 1.), 15, 'illustrative test scenario')
FAST = StimulusTarget('fast running work', 'speed_over_easy', (1.05, 1.25, 1.6, 1.8), 15, 'illustrative test scenario')


def point(n):
    return Quantity.point(n, 'fixture')


def evidence(hrr=.85, speed=3., windows=((600., 1500.),), reference=2., source='planned', day=0):
    start = AT+timedelta(days=day, hours=7)
    end = start+timedelta(minutes=40)
    return WorkSignals(f'{source}:{day}', source, start, end, AT if source == 'planned' else end,
        windows, (SignalSpan(0, 2400, hrr, hrr),), (SignalSpan(0, 2400, speed, speed),),
        'prescribed' if source == 'planned' else 'explicit_segments',
        point(reference), AT if day >= 0 else start-timedelta(days=1), 'prior athlete easy-speed fixture')


def accounted(e):
    bouts = tuple(WorkBout(b-a, e.windows[n+1][0]-b if n+1 < len(e.windows) else None)
                  for n, (a, b) in enumerate(e.windows))
    work = sum(b.work_seconds for b in bouts)/60
    raw = AccountedRun(e.key, e.start, e.end, e.available_at, 4., 'quality',
        (point(40), point(0), point(work)), point(0), e.source, bouts, work_evidence_basis=e.basis)
    return with_continuous_trace(raw, e.hrr, reference=REF, trace_key=e.key,
        trace_source=e.source, elapsed_seconds=2400)[0]


def ctx():
    return ProgramContext((), AT.date(), AT.date(), (AT+timedelta(days=21)).date(),
        (AT+timedelta(days=35)).date(), (40., 40., 15.), TrialPolicy(), DemandScenario(), .01,
        PurposeTimingPolicy(1.), demand_accounting_key=REF.accounting_key)


def evaluate(rows, signals, target=HR, history=(), as_of=AT, fraction=.5):
    return evaluate_stimulus_program(ctx(), history, rows, signals, reference=REF, target=target,
        as_of=as_of, timezone='UTC', purpose_fraction=fraction)


def test_same_duration_cheaper_easy_effort_does_not_fulfill_hr_target():
    easy, useful = evidence(hrr=.65, speed=2), evidence()
    a, b = evaluate((accounted(easy),), {easy.key: easy}), evaluate((accounted(useful),), {useful.key: useful})
    assert a['quality_credit'][easy.key].high == 0
    assert b['quality_credit'][useful.key].low == 1
    assert a['generic_work_coverage'][easy.key].low == b['generic_work_coverage'][useful.key].low == 1
    assert b['evaluation'].total_cost < a['evaluation'].total_cost


def test_short_fast_work_survives_hr_lag_without_pretending_to_meet_hr_target():
    e = evidence(hrr=.65, windows=((600., 720.), (810., 930.), (1020., 1140.)))
    assert assess_work_stimulus(e, FAST)['coverage'].low == pytest.approx(.4)
    assert assess_work_stimulus(e, HR)['coverage'].high == 0
    assert evaluate((accounted(e),), {e.key: e}, FAST)['quality_credit'][e.key].low == pytest.approx(.4)


def test_target_specific_ranking_reverses_not_generic_cheapest_wins():
    cardiovascular = evidence(hrr=.85, speed=2.)
    fast = evidence(hrr=.65, speed=3.)
    costs = {}
    for target in (HR, FAST):
        costs[target.name] = [evaluate((accounted(e),), {e.key: e}, target)['evaluation'].total_cost
                              for e in (cardiovascular, fast)]
    assert costs[HR.name][0] < costs[HR.name][1]
    assert costs[FAST.name][1] < costs[FAST.name][0]


def test_faster_is_not_unlimited_benefit_and_each_minute_is_bounded():
    excessive = evidence(speed=4.)
    assert assess_work_stimulus(excessive, FAST)['coverage'].high == 0
    assert assess_work_stimulus(evidence(), FAST)['supported_minutes'].high == 15


def test_padding_and_recovery_hr_cannot_supply_work_exposure():
    e = evidence(hrr=.65)
    raised_outside = replace(e, hrr=(SignalSpan(0, 600, .85, .85), SignalSpan(600, 1500, .65, .65),
                                   SignalSpan(1500, 2400, .85, .85)))
    assert assess_work_stimulus(e, HR)['coverage'] == assess_work_stimulus(raised_outside, HR)['coverage']


def test_signal_integral_not_bout_average_and_sampling_invariant():
    e = evidence(windows=((0., 100.),))
    continuous = replace(e, hrr=(SignalSpan(0, 100, .6, 1.),))
    split = replace(e, hrr=(SignalSpan(0, 50, .6, .8), SignalSpan(50, 100, .8, 1.)))
    average = replace(e, hrr=(SignalSpan(0, 100, .8, .8),))
    a, b, c = [assess_work_stimulus(x, HR)['supported_minutes'].low for x in (continuous, split, average)]
    assert a == pytest.approx(b)
    assert a < c


def test_missing_hr_retains_fast_evidence_and_unknown_hr_bounds():
    e = replace(evidence(), hrr=())
    assert assess_work_stimulus(e, HR)['coverage'].low == 0
    assert assess_work_stimulus(e, HR)['coverage'].high == 1
    assert assess_work_stimulus(e, FAST)['coverage'].low == 1


def test_no_speed_baseline_not_a_repeat_workout_gate_or_fabricated_zero():
    e = replace(evidence(), easy_speed_mps=None, reference_at=None, reference_source='')
    assert assess_work_stimulus(e, FAST)['coverage'].low == 0
    assert assess_work_stimulus(e, FAST)['coverage'].high == 1
    assert assess_work_stimulus(e, HR)['coverage'].low == 1


def test_bounded_reference_includes_band_peak_not_just_endpoints():
    e = replace(evidence(speed=3.), easy_speed_mps=Quantity(1.5, 3.))
    q = assess_work_stimulus(e, FAST)['coverage']
    assert q.low == 0 and q.high == 1


def test_one_reference_across_bouts_not_sum_of_independent_best_cases():
    e = replace(evidence(windows=((0., 60.), (120., 180.))),
        easy_speed_mps=Quantity(1., 3.),
        speed_mps=(SignalSpan(0, 60, 2., 2.), SignalSpan(120, 180, 4., 4.)))
    r = assess_work_stimulus(e, FAST)
    assert sum(b['supported_minutes'].high for b in r['per_bout']) == pytest.approx(2.)
    assert r['supported_minutes'].high < 2.
    assert r['per_bout_bounds'] == 'marginal_not_additive'


def test_shared_reference_invariant_to_span_and_bout_splitting():
    reference = Quantity(1.5, 3.)
    full = _shared_reference_bounds([(120, 1.8, 4.2)], reference, FAST)
    split = _shared_reference_bounds([(60, 1.8, 3.), (60, 3., 4.2)], reference, FAST)
    assert full == pytest.approx(split, abs=1e-8)
    e = replace(evidence(windows=((0., 120.),)), easy_speed_mps=reference,
                speed_mps=(SignalSpan(0, 120, 1.8, 4.2),))
    split_bouts = replace(e, windows=((0., 60.), (60., 120.)))
    assert assess_work_stimulus(e, FAST)['supported_minutes'].low == pytest.approx(
        assess_work_stimulus(split_bouts, FAST)['supported_minutes'].low, abs=1e-8)
    assert assess_work_stimulus(e, FAST)['supported_minutes'].high == pytest.approx(
        assess_work_stimulus(split_bouts, FAST)['supported_minutes'].high, abs=1e-8)


def test_shared_reference_extrema_enclose_dense_checks_of_linear_speed_traces():
    import random
    generator = random.Random(812)
    for _ in range(20):
        pieces = [(generator.uniform(5, 120), generator.uniform(0, 5), generator.uniform(0, 5))
                  for _ in range(4)]
        ref = Quantity(1.2, 3.5)
        low, high = _shared_reference_bounds(pieces, ref, FAST)
        values = [sum(dt*_integrated_suitability(a/r, b/r, FAST) for dt, a, b in pieces)
                  for r in (ref.low+(ref.high-ref.low)*n/500 for n in range(501))]
        assert low <= min(values)+1e-7
        assert high >= max(values)-1e-7
        assert min(values)-low < .02
        assert high-max(values) < .02


def test_shared_reference_missing_speed_retains_unknown_bounds():
    e = replace(evidence(windows=((0., 120.),)), easy_speed_mps=Quantity(1.5, 3.),
                speed_mps=(SignalSpan(0, 60, 3., 3.),))
    r = assess_work_stimulus(e, FAST)
    assert r['unknown_seconds'] == 60
    assert r['supported_minutes'].high == pytest.approx(2.)


def test_near_constant_ramps_have_a_stable_numerical_enclosure():
    pieces = [(60, 3., 3.+1e-9)]
    low, high = _shared_reference_bounds(pieces, Quantity(2.5, 2.8), FAST)
    exact_low = 60*_integrated_suitability(3./2.8, (3.+1e-9)/2.8, FAST)
    exact_high = 60*_integrated_suitability(3./2.5, (3.+1e-9)/2.5, FAST)
    assert low <= exact_low+1e-10
    assert high >= exact_high-1e-10
    assert exact_low-low < 1e-6 and high-exact_high < 1e-6


def test_athlete_normalization_scales_not_person_specific_speed():
    slow, fast = evidence(speed=2., reference=1.5), evidence(speed=4., reference=3.)
    assert assess_work_stimulus(slow, FAST)['coverage'] == assess_work_stimulus(fast, FAST)['coverage']


def test_target_change_fingerprints_even_when_credit_ties():
    e = evidence()
    a, b = [evaluate((accounted(e),), {e.key: e}, t) for t in (HR, replace(HR, name='another explicit target'))]
    assert a['evaluation'].total_cost == b['evaluation'].total_cost
    assert a['evaluation'].fingerprint != b['evaluation'].fingerprint
    assert a['stimulus_fingerprint'] != b['stimulus_fingerprint']


def test_no_future_reference_wrong_identity_or_overlapping_work():
    e = evidence()
    with pytest.raises(ValueError, match='Freeze'):
        replace(e, reference_at=e.start+timedelta(seconds=1))
    with pytest.raises(ValueError, match='nonoverlapping'):
        replace(e, windows=((0, 200), (100, 300)))
    with pytest.raises(ValueError, match='identity'):
        evaluate((accounted(e),), {e.key: replace(e, key='other')})
    with pytest.raises(ValueError, match='supported work'):
        evaluate((accounted(e),), {e.key: replace(e, windows=((500, 1401),))})


def test_ranged_work_requires_resolved_signal_scenario_not_false_precision():
    e = evidence()
    row = accounted(e)
    row = replace(row, bouts=(WorkBout(900, None, duration_range_seconds=(840, 960)),),
                  demand=(row.demand[0], row.demand[1], Quantity(14, 16)),
                  quality=Quantity(14/15, 1, basis='supported_work_coverage_not_specific_stimulus'))
    with pytest.raises(ValueError, match='Resolve ranged work'):
        evaluate((row,), {e.key: e})


def test_missing_alignment_not_capped_by_generic_reference_or_counted_as_zero():
    e = evidence(windows=((600., 960.),))
    r = evaluate((accounted(e),), {}, replace(HR, minutes=5))
    q = r['quality_credit'][e.key]
    assert q.low == 0 and q.high == 1  # 6 minutes could fulfill a 5-minute target.


@pytest.mark.parametrize('zero,full', [(.68, .78), (.70, .80), (.72, .82)])
def test_declared_band_sensitivity_preserves_obvious_intensity_distinction(zero, full):
    target = replace(HR, band=(zero, full, .9, 1.))
    easy, useful = evidence(hrr=.65), evidence(hrr=.85)
    assert assess_work_stimulus(easy, target)['coverage'].high == 0
    assert assess_work_stimulus(useful, target)['coverage'].low == 1


@pytest.mark.parametrize('target', [HR, FAST])
@pytest.mark.parametrize('fraction', [0., .5, 1.])
def test_expected_completion_and_rollover_preserve_relative_future_scores(target, fraction):
    e = evidence()
    r = accounted(e)
    a = replace(e, key='actual', source='observed', available_at=e.end, basis='explicit_segments')
    actual = accounted(a)
    costs = []
    for day in (3, 4):
        f, tail = evidence(day=day), evidence(day=23)
        future = (accounted(f), accounted(tail))
        before = evaluate((r, *future), {s.key: s for s in (e, f, tail)}, target, fraction=fraction)
        h, p = reconcile_completion((), (r, *future), actual, replaces_key=r.key, as_of=a.end)
        signals = {s.key: s for s in (a, f, tail)}
        after = evaluate(p, signals, target, h, a.end, fraction)
        rollover = evaluate(p, signals, target, h, AT+timedelta(days=1), fraction)
        assert after['evaluation'].total_cost == pytest.approx(rollover['evaluation'].total_cost, abs=1e-12)
        costs.append((before['evaluation'].total_cost, after['evaluation'].total_cost))
    assert costs[0][0]-costs[1][0] == pytest.approx(costs[0][1]-costs[1][1], abs=1e-12)
