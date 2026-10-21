"""Fixed R3 behavioral controls, not a scheduler or physiological calibration.

Expectations are the R1 frozen action matrix. No fitting or candidate search.
Synthetic traces, contracts and stages are explicit; optional report absence
must not be mistaken for a measured recovery pass.
"""
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from run_analysis.continuous_training_dose import ContinuousDoseReference
from run_analysis.recovery_response import ResponseExpectation
from run_analysis.session_decision import (
    CoachingContract, CurrentContext, DatedResponse, DoseLimits,
    PrescriptionBinding, ProgramDecisionInput,
)
from run_analysis.signal_structure import SignalSpan
from run_analysis.workload_accounting import Quantity
from run_analysis.workload_profile import WorkBout
from scripts.continuous_program_adapter import evaluate_continuous, with_continuous_trace
from scripts.program_evidence_adapter import AccountedRun, reconcile_completion
from scripts.program_objective_trial import DemandScenario, EasyDurationPolicy, TrialPolicy
from scripts.shared_program_evaluator import ProgramContext
from scripts.timely_purpose_trial import PurposeTimingPolicy


AT = datetime(2026, 9, 27, tzinfo=timezone.utc)
REF = ContinuousDoseReference('R3 explicit synthetic HR scenario')


def q(value):
    return Quantity.point(value, 'R3 declared scenario')


def row(day, miles, role='easy', *, source='planned', key=None, pace=10.,
        hrr=None, bouts=None):
    start = AT + timedelta(days=day, hours=8)
    seconds = miles * pace * 60
    if bouts is None:
        bouts = tuple(WorkBout(300., 120.) for _ in range(3)) if role == 'quality' else ()
    raw = AccountedRun(key or f'{source}:{day}', start, start+timedelta(seconds=seconds),
        AT if source == 'planned' else start+timedelta(seconds=seconds), miles, role,
        (q(seconds/60), q(0), q(sum(b.work_seconds for b in bouts)/60)), q(0), source,
        tuple(bouts), work_evidence_basis='prescribed' if source == 'planned' else 'explicit_segments')
    hrr = hrr if hrr is not None else (.75 if role == 'quality' else .6)
    return with_continuous_trace(raw, (SignalSpan(0, seconds, hrr, hrr),),
        reference=REF, trace_key=raw.key, trace_source=source, elapsed_seconds=seconds)[0]


def contracts(scale=1.):
    # Deliberately declared experiment envelope, not history-inferred capacity.
    return {
        'easy': CoachingContract(AT, 'R3 supplied stage', 'easy', 'controlled_easy',
            DoseLimits(Quantity(1.*scale, 7.1*scale)), None,
            DoseLimits(Quantity(7.2*scale, 7.4*scale)), trial_target=7.2*scale),
        'long': CoachingContract(AT, 'R3 supplied stage', 'long', 'controlled_easy',
            DoseLimits(q(8.4*scale)), DoseLimits(q(7.6*scale))),
        'quality': CoachingContract(AT, 'R3 supplied stage', 'threshold', 'controlled_threshold',
            DoseLimits(q(4.5*scale), Quantity(600, 1200), Quantity(60, 900), Quantity(.1, 2)),
            DoseLimits(q(4.5*scale), q(360), Quantity(60, 300), Quantity(.1, 2))),
    }


def programme(easy=((2, 3.5), (6, 3.6)), *, cycles=range(5), source='planned', scale=1.):
    return tuple(row(7*cycle+day, miles*scale, role, source=source, pace=10./scale)
        for cycle in cycles for day, miles, role in
        ((0, 8.4, 'long'), (4, 4.5, 'quality'), *((d, m, 'easy') for d, m in easy)))


def context(*, memory=2., exchange=.01, scale=1.):
    return ProgramContext((), AT.date(), AT.date(), (AT+timedelta(days=21)).date(),
        (AT+timedelta(days=35)).date(), (35., 35., 15.),
        TrialPolicy(weekly_miles=20.*scale, ordinary_miles=3.5*scale,
                    long_miles=8.4*scale, ordinary_per_7=2.),
        DemandScenario(memory_days=memory), exchange, PurposeTimingPolicy(1.),
        demand_accounting_key=REF.accounting_key,
        easy_duration_policy=EasyDurationPolicy(10, 25, .01, 'R1 provisional', 7.*scale))


def evaluate(rows, *, history=None, ctx=None, current=None, responses=(), as_of=AT,
             scale=1., supplied=None):
    history = programme(cycles=range(-2, 0), source='observed', scale=scale) if history is None else history
    declared = contracts(scale) if supplied is None else supplied
    inputs = ProgramDecisionInput('R3 frozen stage', AT, 'synthetic behavioral controls',
        tuple(PrescriptionBinding(r.key, declared[r.role].purpose, declared[r.role].intensity_key,
                                  declared[r.role]) for r in rows), current, tuple(responses))
    return evaluate_continuous(ctx or context(scale=scale), history, rows,
        reference=REF, as_of=as_of, timezone='UTC', purpose_fraction=.5, decision_input=inputs)


def cost(result):
    return result['evaluation'].total_cost


@pytest.mark.parametrize('state', [None, 'usual', 'little_tired'])
def test_easy_day_after_long_remains_an_option_without_recovery_clearance(state):
    current = None if state is None else CurrentContext(AT.date(), AT, state, 'current report')
    result = evaluate((row(0, 3.5),), history=(row(-1, 8.4, 'long', source='observed'),), current=current)
    assert result['decision_eligible']
    assert result['suitability']['action'] == 'continue_provisionally'
    assert not result['suitability']['physiological_recovery_established']


def test_tired_demanding_session_compares_reduction_substitution_and_deferral():
    tired = CurrentContext(AT.date(), AT, 'little_tired', 'current report')
    candidates = {
        'full': (row(0, 8.4, 'long'),),
        'reduced': (row(0, 7.6, 'long'),),
        'easy': (row(0, 3.5),),
        'deferred': (row(1, 8.4, 'long'),),
        'omit': (),
    }
    results = {name: evaluate(rows, current=tired) for name, rows in candidates.items()}
    assert not results['full']['decision_eligible']
    assert all(results[name]['decision_eligible'] for name in ('reduced', 'easy', 'deferred', 'omit'))
    assert results['deferred']['suitability']['future_context_recheck_required']
    # Reduction/substitution/omission must retain lost delivery, not receive free credit.
    delivery = {name: result['evaluation'].parts['delivery'] for name, result in results.items()}
    assert delivery['full']['long'] > delivery['reduced']['long'] > delivery['easy']['long']
    assert delivery['full']['volume'] > delivery['reduced']['volume'] > delivery['easy']['volume'] > delivery['omit']['volume']


@pytest.mark.parametrize('kind', ['missing_rating', 'missing_expectation', 'favorable', 'adverse', 'conflicting'])
def test_response_unknown_is_not_adverse_and_favorable_does_not_manufacture_growth(kind):
    previous = row(-1, 8.4, 'long', source='observed')
    expectation = ResponseExpectation('preissued', effort=None if kind == 'missing_expectation' else Quantity(3, 5))
    report = DatedResponse(previous.key, previous.start-timedelta(hours=1), previous.end,
        expectation, 'synthetic report', effort=None if kind == 'missing_rating' else
        (2 if kind == 'favorable' else 8), conflict=kind == 'conflicting')
    rows = (row(0, 3.5),)
    baseline = evaluate(rows, history=(previous,))
    result = evaluate(rows, history=(previous,), responses=(report,))
    assert cost(result) == cost(baseline)
    assert result['decision_eligible'] == (kind not in {'adverse', 'conflicting'})
    assert not result['suitability']['responses'][previous.key]['automatic_progression']
    # Even a favorable rating cannot create a dose contract that was not issued.
    unsupported = evaluate((row(0, 7.5),), history=(previous,), responses=(report,))
    assert not unsupported['decision_eligible']


@pytest.mark.parametrize('state', ['health_concern', 'conflicting'])
def test_context_restriction_is_not_overruled_by_a_better_score(state):
    rows = programme()
    plain = evaluate(rows)
    blocked = evaluate(rows, current=CurrentContext(AT.date(), AT, state, 'health context'))
    assert cost(plain) == cost(blocked)
    assert not blocked['decision_eligible']
    assert blocked['suitability']['action'] == 'review_current_context'


def test_novel_short_and_sustained_work_preserve_structure_without_invented_multiplier():
    options = {
        'three_by_five': tuple(WorkBout(300, 120) for _ in range(3)),
        'six_by_two_point_five': tuple(WorkBout(150, 60) for _ in range(6)),
        'sustained_fifteen': (WorkBout(900, None),),
    }
    results = {name: evaluate((row(0, 4.5, 'quality', bouts=bouts),)) for name, bouts in options.items()}
    assert all(r['decision_eligible'] for r in results.values())
    assert len({r['evaluation'].fingerprint for r in results.values()}) == 3
    assert all(cost(r) == pytest.approx(cost(results['three_by_five'])) for r in results.values())
    excessive = evaluate((row(0, 4.5, 'quality', bouts=(WorkBout(1200, None),)),))
    assert not excessive['decision_eligible']  # Total supported, constituent bout unsupported.
    # Unfamiliar name/distance does not disqualify supported constituent work.
    novel = evaluate((row(0, 4.5, 'quality', bouts=tuple(WorkBout(200, 90, 500) for _ in range(3))),))
    assert novel['decision_eligible']


@pytest.mark.parametrize('memory', [1., 2., 4.])
@pytest.mark.parametrize('exchange', [.005, .01, .02])
def test_whole_program_preserves_volume_and_prices_compression(memory, exchange):
    ctx = context(memory=memory, exchange=exchange)
    programmes = {
        'distributed': programme(),
        'compressed': programme(((1, 3.5), (2, 3.6))),
        'underfunded': programme(()),
        'uneven': programme(((2, 2.5), (6, 4.6))),
        'fragmented': programme(((1, 1.), (2, 1.), (3, 1.), (6, 4.1))),
    }
    results = {name: evaluate(rows, ctx=ctx) for name, rows in programmes.items()}
    assert all(r['decision_eligible'] for r in results.values())
    assert cost(results['distributed']) < cost(results['compressed'])
    assert cost(results['distributed']) < cost(results['underfunded'])
    a, b = (results[name]['evaluation'] for name in ('distributed', 'uneven'))
    assert a.parts['easy_duration']['supporting_easy_volume'] == b.parts['easy_duration']['supporting_easy_volume']
    assert a.short_run_cost == b.short_run_cost == 0  # 2.5 is not banned or penalized at 25 min.
    assert results['fragmented']['evaluation'].short_run_cost > 0
    print('R3 panel', memory, exchange, {name: round(cost(r), 6) for name, r in results.items()})


@pytest.mark.parametrize('scale', [1., 2.])
def test_athlete_distance_and_pace_normalization_not_personal_mileage_copy(scale):
    # Twice the distance at twice the speed: same duration/HR workload, scaled
    # volume/long/support contracts. This is invariance, not physiological identity.
    original = evaluate(programme())
    other = evaluate(programme(scale=scale), scale=scale)
    assert original['decision_eligible'] and other['decision_eligible']
    assert cost(original) == pytest.approx(cost(other))
    assert other['evaluation'].parts['targets']['volume'] == scale*original['evaluation'].parts['targets']['volume']


def test_slower_or_higher_hr_same_distance_increases_cost_without_fake_clearance():
    base = evaluate((row(0, 3.5),))
    slower = evaluate((row(0, 3.5, pace=15),))
    higher_hr = evaluate((row(0, 3.5, hrr=.8),))
    assert all(r['decision_eligible'] for r in (base, slower, higher_hr))
    assert cost(slower) > cost(base)
    assert cost(higher_hr) > cost(base)


def test_expected_completion_and_rollover_preserve_pairwise_decision_comparison():
    history = programme(cycles=range(-2, 0), source='observed')
    expected = row(0, 8.4, 'long')
    actual = replace(expected, key='actual_long', source='observed', available_at=expected.end,
                     work_evidence_basis='explicit_segments')
    deltas = []
    for day in (1, 2):
        rest = tuple(r for r in programme() if r.key not in {'planned:0', 'planned:2'})
        remaining = (row(day, 3.5), *rest)
        before = evaluate((expected, *remaining), history=history)
        h, p = reconcile_completion(history, (expected, *remaining), actual,
                                     replaces_key=expected.key, as_of=actual.end)
        after = evaluate(p, history=h, as_of=actual.end)
        rollover = evaluate(p, history=h, as_of=AT+timedelta(days=1))
        assert all(r['decision_eligible'] for r in (before, after, rollover))
        assert cost(after) == pytest.approx(cost(rollover), abs=1e-12)
        deltas.append((cost(before), cost(after)))
    assert deltas[0][0]-deltas[1][0] == pytest.approx(deltas[0][1]-deltas[1][1], abs=1e-12)


@pytest.mark.parametrize('miles', [3.3, 3.5, 3.7, 3.8])
def test_band_edges_and_small_excess_retain_actual_load_without_forcing_review(miles):
    history = programme(cycles=range(-2, 0), source='observed')
    expected = row(0, 3.5)
    actual = row(0, miles, source='observed', key='actual_easy')
    future = programme(cycles=range(1, 5))
    h, p = reconcile_completion(history, (expected, *future), actual,
                                 replaces_key=expected.key, as_of=actual.end)
    result = evaluate(p, history=h, as_of=actual.end)
    assert result['decision_eligible']
    stored = next(r for r in result['evaluation'].conditional_ledger if r.key == actual.key)
    assert stored.miles == miles
    assert stored.demand[0].low == pytest.approx(miles*10)
    assert result['evaluation'].parts['targets'] == evaluate(p, history=history)['evaluation'].parts['targets']


def test_bounded_trial_is_not_ordinary_tolerance_or_stage_growth():
    rows = (row(0, 7.2),)
    result = evaluate(rows)
    assert result['decision_eligible']
    assert result['suitability']['action'] == 'propose_bounded_trial'
    tired = evaluate(rows, current=CurrentContext(AT.date(), AT, 'little_tired', 'report'))
    assert not tired['decision_eligible']
    assert result['evaluation'].parts['targets'] == tired['evaluation'].parts['targets']


def test_stacking_changes_the_tradeoff_not_a_streak_count_gate():
    future = (row(0, 3.5), *tuple(r for r in programme() if r.key != 'planned:0'))
    rest = tuple(r for r in future if r.key != 'planned:0')
    differences = []
    for dose in (3.5, 8.4):
        history = (row(-3, 8.4, 'long', source='observed'),
                   row(-2, dose, source='observed'), row(-1, dose, source='observed'))
        continued, omitted = evaluate(future, history=history), evaluate(rest, history=history)
        assert continued['decision_eligible'] and omitted['decision_eligible']
        differences.append(cost(continued)-cost(omitted))
    assert differences[1] > differences[0]
    assert differences[0] < 0 < differences[1]
    print('R3 stacked: continue-minus-omit', differences)


@pytest.mark.parametrize('memory', [1., 2., 4.])
def test_full_volume_sequence_compares_continue_reduce_defer_and_redistribute(memory):
    # Unlike the isolated run/rest contrast, keep a full 20-mile/7-day equivalent
    # sequence, including its long/quality delivery. These are five fixed choices.
    future = tuple(row(cycle*7+day, miles, role) for cycle in range(5)
        for day, miles, role in ((0, 3.5, 'easy'), (2, 4.5, 'quality'),
                                (4, 8.4, 'long'), (6, 3.6, 'easy')))
    fixed = future[1:]
    options = {'continue': future, 'reduce': (row(0, 2.5), *fixed), 'omit': fixed,
        'defer': (row(1, 3.5), *fixed),
        'redistribute': (row(0, 2.5), *(row(6, 4.6) if r.key == 'planned:6' else r for r in fixed))}
    contrasts = []
    for dose in (3.5, 8.4):
        history = (row(-3, 8.4, 'long', source='observed'),
                   row(-2, dose, source='observed'), row(-1, dose, source='observed'))
        results = {name: evaluate(rows, history=history, ctx=context(memory=memory))
                   for name, rows in options.items()}
        assert all(r['decision_eligible'] for r in results.values())
        parts = {name: r['evaluation'] for name, r in results.items()}
        assert parts['continue'].parts['delivery']['volume'] == 100.
        assert parts['reduce'].parts['delivery']['volume'] == 99.
        assert parts['omit'].parts['delivery']['volume'] == 96.5
        assert parts['continue'].parts['delivery']['volume'] == parts['redistribute'].parts['delivery']['volume']
        assert parts['continue'].delivery_cost == parts['redistribute'].delivery_cost
        assert parts['reduce'].delivery_cost > parts['continue'].delivery_cost
        assert parts['omit'].delivery_cost > parts['reduce'].delivery_cost
        assert cost(results['reduce']) < cost(results['omit'])
        changes = {name: cost(r)-cost(results['continue']) for name, r in results.items()}
        contrasts.append(changes)
        print('R3 full-volume alternatives', memory, dose,
              {name: round(value, 8) for name, value in changes.items()})
    # Same stage/forward work: extra actual lead-in makes a deferral more
    # attractive, without excluding either candidate or inventing a rest rule.
    assert contrasts[1]['defer'] < contrasts[0]['defer']
    assert contrasts[0]['defer'] > 0 > contrasts[1]['defer']


def test_missing_history_requests_evidence_instead_of_manufacturing_capacity():
    result = evaluate((row(0, 3.5),), history=())
    assert not result['decision_eligible']
    assert result['constraints']['status'] == 'constraint_evidence_incomplete'
    assert 'program_constraints_not_satisfied' in result['suitability']['reasons']
    assert not result['selection_ready']


def test_missing_or_below_floor_options_never_win_by_finite_score_alone():
    below = evaluate((row(0, 1., pace=8.),))
    unsupported = evaluate((row(0, 7.5),))
    assert not any(r['decision_eligible'] for r in (below, unsupported))
    assert 'planned_easy_below_minimum' in below['suitability']['reasons']
    assert 'outside_supported_dose_contract' in unsupported['suitability']['session_checks']['planned:0']['problems']
