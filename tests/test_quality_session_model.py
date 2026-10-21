from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from run_analysis.continuous_training_dose import ContinuousDoseReference
from run_analysis.signal_structure import SignalSpan
from run_analysis.work_stimulus import WorkSignals
from run_analysis.workload_accounting import Quantity
from run_analysis.workload_profile import WorkBout
from scripts.continuous_program_adapter import with_continuous_trace, evaluate_continuous
from scripts.program_evidence_adapter import AccountedRun
from scripts.program_objective_trial import TrialPolicy, DemandScenario
from scripts.shared_program_evaluator import ProgramContext
from scripts.timely_purpose_trial import PurposeTimingPolicy
from scripts.quality_session_model import (
    assess_quality_session, current_quality_pace, evaluate_quality_program, QualityHistoryPolicy,
)

AT = datetime(2026, 9, 26, tzinfo=timezone.utc)
REF = ContinuousDoseReference('fixture')


def record(key='one', day=-1, speed=3., source='observed', windows=((100., 220.), (300., 420.)),
           distance_known=True):
    start = AT+timedelta(days=day, hours=7)
    end = start+timedelta(seconds=1200)
    available = AT if source == 'planned' else end
    basis = 'prescribed' if source == 'planned' else 'explicit_segments'
    bouts = tuple(WorkBout(b-a, windows[i+1][0]-b if i+1 < len(windows) else None,
        (b-a)*speed if distance_known else None) for i, (a,b) in enumerate(windows))
    q = lambda x: Quantity.point(x, 'fixture')
    raw = AccountedRun(key, start, end, available, 2., 'quality',
        (q(20), q(0), q(sum(b-a for a,b in windows)/60)), q(0), source, bouts, work_evidence_basis=basis)
    signals = WorkSignals(key, source, start, end, available, windows,
        (SignalSpan(0., 1200., .8, .8),), (SignalSpan(0., 1200., speed, speed),), basis)
    row = with_continuous_trace(raw, signals.hrr, reference=REF, trace_key=key,
        trace_source=source, elapsed_seconds=1200)[0]
    return row, signals


def assessed(**kwargs):
    row, signals = record(**kwargs)
    return assess_quality_session(row, signals, reference=REF, as_of=max(AT, row.available_at))


def pace(history, seconds=137., at=AT, policy=QualityHistoryPolicy()):
    return current_quality_pace(history, as_of=at, bout_seconds=seconds,
        accounting_key=REF.accounting_key, policy=policy)


def test_recent_bout_means_not_old_peak_reference_and_novel_duration_allowed():
    current = assessed(speed=3.)
    old = assessed(key='old-fast', day=-60, speed=8.)
    result = pace([current, old])
    assert result.speed_mps.estimate == 3.
    assert [k for k, _, _ in result.session_estimates] == ['one']
    assert ('old-fast', 'outside_current_window') in result.excluded
    assert result.requested_bout_seconds == 137.
    assert result.fingerprint == pace([current]).fingerprint


def test_old_quality_only_does_not_silently_become_current_expectation():
    result = pace([assessed(day=-60)])
    assert result.speed_mps is None
    assert result.status == 'no_recent_work_pace_explicit_forecast_needed'


def test_historical_window_does_not_change_current_estimate():
    history = [assessed(), assessed(key='old', day=-60, speed=8.)]
    a = pace(history)
    b = pace(history, policy=QualityHistoryPolicy(historical_days=365))
    assert a.speed_mps == b.speed_mps
    assert a.session_estimates == b.session_estimates
    assert a.fingerprint == b.fingerprint


def test_freeze_no_future_late_planned_self_or_old_window_leakage():
    good = assessed()
    future = assessed(key='future', day=1, speed=8.)
    planned = assessed(key='plan', day=1, source='planned', speed=8.)
    late = replace(good, run=replace(good.run, key='late', available_at=AT+timedelta(days=1)))
    result = pace([good, future, planned, late])
    assert result.speed_mps == pace([good]).speed_mps
    assert result.fingerprint == pace([good]).fingerprint
    assert len(result.session_estimates) == 1
    assert pace([good], at=good.run.start).speed_mps is None
    fingerprint = result.fingerprint
    _ = pace([good, future], at=AT+timedelta(days=3))
    assert result.fingerprint == fingerprint


def test_repetitions_do_not_become_independent_session_votes():
    short = assessed(speed=3., windows=((100., 220.),))
    repeats = assessed(speed=3., windows=((100., 220.), (300., 420.), (500., 620.)))
    other = assessed(key='other', speed=4.)
    assert pace([short, other]).speed_mps == pace([repeats, other]).speed_mps


def test_interval_and_sustained_structure_retained_without_declaring_equivalence():
    intervals = assessed()
    sustained = assessed(windows=((100., 340.),))
    assert intervals.work_minutes.low == sustained.work_minutes.low
    assert intervals.run.quality == sustained.run.quality  # Generic recurrence only.
    assert len(intervals.bouts) == 2 and len(sustained.bouts) == 1
    assert intervals.bouts[0].recovery_after_seconds == 80.
    assert sustained.bouts[0].recovery_after_seconds is None


def test_planned_and_observed_equal_traces_produce_equal_physical_profiles():
    row, signals = record(day=1, source='planned')
    before = assess_quality_session(row, signals, reference=REF, as_of=AT)
    actual = replace(row, source='observed', available_at=row.end, work_evidence_basis='explicit_segments')
    observed = replace(signals, source='observed', available_at=row.end, basis='explicit_segments')
    after = assess_quality_session(actual, observed, reference=REF, as_of=row.end)
    assert before.bouts == after.bouts
    assert before.run.demand == after.run.demand
    assert before.run.quality == after.run.quality
    assert before.work_minutes.low == after.work_minutes.low


def test_missing_signals_keep_workload_and_do_not_become_zero_work():
    row, _ = record()
    result = assess_quality_session(row, None, reference=REF, as_of=AT)
    assert result.run.demand == row.demand
    assert result.work_minutes.low == 4.
    assert result.bouts == ()
    assert pace([result]).speed_mps is None


def test_bout_distance_retained_without_demanding_perfect_trace_edges():
    row, signals = record()
    result = assess_quality_session(row, replace(signals, speed_mps=()), reference=REF, as_of=AT)
    assert all(b.speed_mps == 3. and b.speed_known_seconds == 0 for b in result.bouts)
    assert all(b.speed_basis == 'supported_bout_distance_over_duration' for b in result.bouts)


def test_unknown_pace_not_zero_when_no_distance_or_complete_trace():
    row, signals = record(distance_known=False)
    result = assess_quality_session(row, replace(signals, speed_mps=()), reference=REF, as_of=AT)
    assert all(b.speed_mps is None for b in result.bouts)
    assert result.work_minutes.low == 4.


def test_identity_and_signal_availability_checked():
    row, signals = record()
    with pytest.raises(ValueError):
        assess_quality_session(row, replace(signals, key='wrong'), reference=REF, as_of=AT)
    with pytest.raises(ValueError):
        assess_quality_session(row, replace(signals, available_at=AT+timedelta(days=1)), reference=REF, as_of=AT)


def test_component_handoff_does_not_change_score_or_double_charge_demand():
    row, signal = record(day=1, source='planned')
    context = ProgramContext((), AT.date(), AT.date(), (AT+timedelta(days=21)).date(),
        (AT+timedelta(days=35)).date(), (40.,40.,15.), TrialPolicy(), DemandScenario(), .01,
        PurposeTimingPolicy(1.), demand_accounting_key=REF.accounting_key)
    kwargs = dict(reference=REF, as_of=AT, timezone='UTC', purpose_fraction=.5)
    baseline = evaluate_continuous(context, (), (row,), **kwargs)
    connected = evaluate_quality_program(context, (), (row,), {row.key:signal}, **kwargs)
    assert baseline['evaluation'] == connected['evaluation']
    assert baseline['constraints'] == connected['constraints']
    assert not connected['selection_ready']
    assert connected['quality_assessments'][row.key].run.demand == row.demand


def test_out_of_work_padding_changes_load_not_work_profile_or_reference():
    row, signal = record()
    original = assess_quality_session(row, signal, reference=REF, as_of=AT)
    padded_signal = replace(signal, end=signal.end+timedelta(minutes=20),
        available_at=signal.available_at+timedelta(minutes=20),
        hrr=(*signal.hrr, SignalSpan(1200.,2400.,.7,.7)))
    padded_row = replace(row, end=padded_signal.end, available_at=padded_signal.available_at,
        demand=(Quantity.point(40.,'fixture'), row.demand[1], row.demand[2]))
    padded_row = with_continuous_trace(padded_row, padded_signal.hrr, reference=REF,
        trace_key=row.key, trace_source=row.source, elapsed_seconds=2400)[0]
    padded = assess_quality_session(padded_row, padded_signal, reference=REF, as_of=AT)
    assert padded.bouts == original.bouts
    assert padded.run.quality == original.run.quality
    assert padded.run.demand[1].low > original.run.demand[1].low
    assert pace([padded]).speed_mps == pace([original]).speed_mps


def test_order_and_simultaneous_athlete_speed_rescaling():
    history = [assessed(key='a',speed=3.), assessed(key='b',speed=4.)]
    assert pace(history) == pace(reversed(history))
    doubled = [assessed(key='a',speed=6.), assessed(key='b',speed=8.)]
    assert pace(doubled).speed_mps.estimate == pytest.approx(2*pace(history).speed_mps.estimate)


@pytest.mark.parametrize('days', [0., -1., float('nan'), 85.])
def test_invalid_policy(days):
    with pytest.raises(ValueError):
        QualityHistoryPolicy(current_days=days)


def validation_fixture():
    from dataclasses import asdict
    import json
    from scripts.audit_work_stimulus import encode
    old, new = assessed(key='prior',day=-2,speed=3.), assessed(key='target',day=-1,speed=4.)
    reference = pace([old,new], seconds=120., at=new.run.start)
    cases = [dict(key=a.run.key, start=a.run.start, label='intervals', assessment=asdict(a),
        prior_references=[asdict(reference)] if a is new else []) for a in (old,new)]
    return json.loads(json.dumps(dict(version='fixture',cases=cases), default=encode))


def test_conditional_validation_equal_session_metrics_and_missing_denominator():
    from scripts.validate_quality_pace import validate
    result = validate(validation_fixture())
    assert result['total_cases'] == 2 and result['eligible'] == 1
    assert sum(result['excluded'].values()) == 1
    expected = 1609.344*(1/3-1/4)
    assert result['overall']['model_mae_seconds_mile'] == pytest.approx(expected)
    assert result['overall']['model_bias_seconds_mile'] == pytest.approx(expected)
    assert result['overall']['improvement_seconds_mile'] == 0
    assert not result['predeclared_directional_check_passed']


def test_validation_rejects_future_reference_source():
    from scripts.validate_quality_pace import validate
    data = validation_fixture()
    data['cases'][0]['assessment']['run']['end'] = data['cases'][1]['start']
    with pytest.raises(ValueError, match='Future/self/stale'):
        validate(data)


def test_validation_never_calls_empty_comparison_success():
    from scripts.validate_quality_pace import validate
    data = validation_fixture()
    for c in data['cases']:
        c['prior_references'] = []
    result = validate(data)
    assert result['eligible'] == 0
    assert result['overall']['model_mae_seconds_mile'] is None
    assert not result['predeclared_directional_check_passed']
