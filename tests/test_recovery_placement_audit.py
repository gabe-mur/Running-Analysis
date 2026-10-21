from types import SimpleNamespace
from datetime import timedelta

import pytest

from scripts.audit_recovery_placement import move_session
from scripts.audit_recovery_full_score import cadence_charges, passes_checked_constraints
from test_typed_continuation import fixture


def test_move_preserves_dose_and_other_days_without_mutating_source():
    _, make = fixture()
    days, _ = make(21, {2: ('easy', 4.), 5: ('long', 8.)})
    before = [d.model_dump_json() for d in days]
    revised = move_session(days, days[2].date, days[3].date)
    assert revised[2].recommendation is None
    old, new = days[2].recommendation, revised[3].recommendation
    assert new.planned_for == old.planned_for + timedelta(days=1)
    assert new.model_dump(exclude={'planned_for', 'fitness_state_as_of'}) == old.model_dump(
        exclude={'planned_for', 'fitness_state_as_of'})
    assert revised[5] is days[5]
    assert before == [d.model_dump_json() for d in days]
    with pytest.raises(ValueError, match='empty date'):
        move_session(days, days[2].date, days[5].date)


def test_strict_cadence_advisory_is_not_an_active_failure():
    gates = dict(failures=[], whole_hard_peak_failures=[], strict_cadence_findings=['overdue'])
    acceptance = SimpleNamespace(rejected=False, reason='funded')
    assert passes_checked_constraints({'checks_pass': True}, gates, acceptance)
    assert not passes_checked_constraints({'checks_pass': False}, gates, acceptance)
    assert not passes_checked_constraints({'checks_pass': True},
        dict(gates, failures=['recovery']), acceptance)
    assert not passes_checked_constraints({'checks_pass': True}, gates,
        SimpleNamespace(rejected=False, reason='missing_context'))


def test_cadence_charge_includes_overdue_occurrence_before_reset():
    _, make = fixture()
    days, states = make(11, {9: ('long', 8.)})
    opening = states[0].model_copy(update={'days_since_long_run': 0., 'days_since_quality_run': None})
    charges = cadence_charges(days, opening, {})
    assert charges == [dict(date=str(days[9].date), role='long', age_days=9.,
                           reference_days=7., cost=20.)]
