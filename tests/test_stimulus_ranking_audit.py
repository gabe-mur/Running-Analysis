from datetime import datetime, timedelta, timezone

import pytest

from run_analysis.signal_structure import SignalSpan
from run_analysis.work_stimulus import WorkSignals
from run_analysis.workload_accounting import Quantity
from run_analysis.workload_profile import WorkBout
from scripts import audit_stimulus_rankings as module
from scripts.continuous_program_adapter import with_continuous_trace
from scripts.program_evidence_adapter import AccountedRun


def test_summary_reports_reversals_regret_and_ties_without_probabilities():
    result = module.ranking_summary([
        {'scores': {'a': 1., 'b': 2.}}, {'scores': {'a': 3., 'b': 2.}},
        {'scores': {'a': 4., 'b': 4.}},
    ])
    assert result['possible_winners'] == ['a', 'b']
    assert result['observed_winner_sets'] == {'a': 1, 'b': 1, 'a | b': 1}
    assert result['maximum_regret_by_candidate'] == {'a': 1., 'b': 1.}
    assert not result['winner_unchanged_on_sampled_scenarios']
    assert result['minimum_top_two_gap'] == 0


def test_summary_invariant_to_common_offsets_and_candidate_order():
    original = [{'scores': {'a': 1., 'b': 2.}}, {'scores': {'a': 3., 'b': 5.}}]
    shifted = [{'scores': {k: v+100 for k, v in reversed(list(c['scores'].items()))}}
               for c in original]
    assert module.ranking_summary(original) == module.ranking_summary(shifted)


@pytest.mark.parametrize('cases', [[], [{'scores': {'a': 1}}],
    [{'scores': {'a': 1, 'b': 2}}, {'scores': {'a': 1, 'c': 2}}],
    [{'scores': {'a': float('nan'), 'b': 2}}],
    [{'scores': {'a': 1, 'b': float('inf')}}]])
def test_summary_rejects_incomparable_scores(cases):
    with pytest.raises(ValueError):
        module.ranking_summary(cases)


@pytest.mark.parametrize('tolerance', [-1., float('nan'), float('inf')])
def test_summary_rejects_invalid_tolerance(tolerance):
    with pytest.raises(ValueError):
        module.ranking_summary([{'scores': {'a': 1, 'b': 2}}], tolerance=tolerance)


AT = datetime(2026, 9, 26, tzinfo=timezone.utc)


def inventory(reference):
    rows, signals = [], {}
    for index, role in enumerate(('easy', 'long', 'quality', 'quality')):
        start = AT-timedelta(days=10-index)
        end = start+timedelta(minutes=40)
        key = str(index)
        quality = role == 'quality'
        q = lambda value: Quantity.point(value, 'fixture')
        hrr = (SignalSpan(0., 2400., .85 if quality else .6, .85 if quality else .6),)
        bouts = (WorkBout(900., None),) if quality else ()
        basis = 'explicit_segments' if quality else 'no_supported_work'
        raw = AccountedRun(key, start, end, end, 4.44, role,
            (q(40), q(0), q(15 if quality else 0)), q(0), 'observed', bouts,
            work_evidence_basis=basis)
        rows.append(with_continuous_trace(raw, hrr, reference=reference,
            trace_key=key, trace_source='observed', elapsed_seconds=2400)[0])
        if quality:
            signals[key] = WorkSignals(key, 'observed', start, end, end,
                ((600., 1500.),), hrr, (SignalSpan(0., 2400., 3.+index*.3, 3.+index*.3),),
                basis, Quantity(2., 3., 2.5), start-timedelta(days=1), 'test prior reference')
    return tuple(rows), signals, {'speed_mps': Quantity(2., 3., 2.5)}


def test_transplant_uses_one_reference_preserving_raw_response_and_history():
    reference = module.ContinuousDoseReference('fixture')
    rows, signals, _ = inventory(reference)
    frozen = dict(signals)
    for row in rows[-2:]:
        planned, signal = module.transplant(row, signals, key='future',
            start=AT+timedelta(days=3), issued_at=AT, speed=2.4)
        assert planned.source == signal.source == 'planned'
        assert planned.work_evidence_basis == signal.basis == 'prescribed'
        assert planned.miles == 4.4 and row.miles == 4.44
        assert planned.demand == row.demand
        assert signal.hrr == signals[row.key].hrr
        assert signal.speed_mps == signals[row.key].speed_mps
        assert signal.easy_speed_mps.low == signal.easy_speed_mps.high == 2.4
        assert signal.reference_at == signal.available_at == AT
    assert signals == frozen


def test_full_audit_repeatable_with_reversed_candidates_and_shared_hr_control(tmp_path, monkeypatch):
    database = tmp_path/'fixture.sqlite'
    database.write_bytes(b'readonly inventory is supplied by fixture')
    monkeypatch.setattr(module, 'load_inventory', lambda db, cfg, at, ref: inventory(ref))
    first = module.audit(database, {'timezone_default': 'UTC'}, AT, grid_points=3)
    def reversed_inventory(db, cfg, at, ref):
        rows, signals, prior = inventory(ref)
        return tuple(reversed(rows)), signals, prior
    monkeypatch.setattr(module, 'load_inventory', reversed_inventory)
    second = module.audit(database, {'timezone_default': 'UTC'}, AT, grid_points=3)
    assert first['database_unchanged'] and second['database_unchanged']
    assert first['panels'] == second['panels']
    hr, pace = first['panels']
    assert all(c['scores'] == hr['nominal']['scores'] for c in hr['primary_cases'])
    assert len({tuple(c['scores'].values()) for c in pace['primary_cases']}) > 1
    for panel in first['panels']:
        for case in panel['primary_cases']:
            for key, details in case['details'].items():
                assert case['scores'][key] == pytest.approx(
                    details['delivery_cost']+details['weighted_demand_cost'])
