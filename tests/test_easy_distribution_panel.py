from datetime import date, timedelta

import pytest

from scripts.audit_easy_distribution import exposures, panel


def test_all_controls_hold_total_volume_and_key_doses_and_tail_fixed():
    origin = date(2026,9,25)
    baseline = exposures(4, origin)
    for n in (2,4,8,12,16,20):
        rows = exposures(n,origin)
        evaluated = [e for e in rows if e.day < origin+timedelta(days=28)]
        assert sum(e.miles for e in evaluated) == pytest.approx(69.2)
        assert sum(e.role == 'easy' for e in evaluated) == n
        assert len({e.day for e in rows}) == len(rows)
        assert [e for e in rows if e.role != 'easy'] == [e for e in baseline if e.role != 'easy']
        assert [e for e in rows if e.day >= origin+timedelta(days=28)] == [e for e in baseline if e.day >= origin+timedelta(days=28)]


def test_integer_tenths_allocation_does_not_lose_mileage():
    rows = exposures(12,date(2026,9,25))
    easy = [e for e in rows if e.role == 'easy' and e.day < date(2026,10,23)]
    assert sorted(e.miles for e in easy) == [1.4]*4+[1.5]*8


def test_frozen_original_evidence_panel_does_not_turn_floor_pass_into_selection():
    result = panel()
    cases = result['cases']
    assert cases['20']['floors']['central']['passing_sessions'] == 0
    assert cases['16']['floors']['central']['passing_sessions'] == 16
    assert cases['16']['floors']['upper_quartile']['passing_sessions'] == 0
    for case in cases.values():
        assert case['selection'] is None
        assert case['easy_total_minutes_at_reference_pace'] == pytest.approx(cases['4']['easy_total_minutes_at_reference_pace'])
        assert len(case['receiving_session_comparisons']) == 12
        assert sum(case['tail_order_counts'].values()) == 4
    assert cases['4']['main_order_counts'] == {'equal': 8}
    assert cases['4']['tail_order_counts'] == {'equal': 4}
    assert not result['optimal_calendar_established'] and not result['recovery_cleared']
