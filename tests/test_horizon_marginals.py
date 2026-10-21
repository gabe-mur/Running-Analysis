from datetime import date
from types import SimpleNamespace

import pytest

from scripts.audit_horizon_marginals import additive_parts, explicit_fatigue_rows, subtract


def evaluation():
    return SimpleNamespace(total_cost=42., delivery_cost=6., short_run_cost=7., parts={
        'soft_horizon': {'demand': 9.},
        'continuation_boundary': {'volume_cost': 1., 'cross_demand_cost': 2.},
        'volume_benefit': {'added_volume_cost': 3.},
        'race_event': {'cost': 10.},
        'frequency_preference': {'cost': 8.}})


def test_exclusive_components_do_not_double_charge_volume_or_race():
    terms = additive_parts(evaluation())
    assert terms['purpose_delivery'] == 2.
    assert terms['race_fatigue'] == 10.
    assert sum(terms.values()) == 42.


def test_component_mismatch_fails_closed():
    ev = evaluation()
    ev.total_cost += .01
    with pytest.raises(ValueError, match='decomposition'):
        additive_parts(ev)


def test_dated_fatigue_respects_taper_and_common_span():
    context = SimpleNamespace(stage_start=date(2026, 1, 1), taper_start=date(2026, 1, 8),
        end=date(2026, 1, 15), exchange=2., scenario=SimpleNamespace(axis_weights=(1., 1., 1.)))
    # Seven days into the soft tail: row at Jan 11 receives 4/7 weight.
    from run_analysis.adaptive.program_objective_trial import AXES
    costs = dict.fromkeys(AXES, 3.)
    expected = 3.+12./7
    ev = SimpleNamespace(parts={'session_costs': [
        dict(day=date(2026, 1, 3), axis_cost=costs),
        dict(day=date(2026, 1, 11), axis_cost=costs)],
        'soft_horizon': {'demand': expected}})
    rows = explicit_fatigue_rows(context, ev)
    assert sum(rows.values()) == pytest.approx(expected)
    assert rows['2026-01-11'] == pytest.approx(12./7)


def test_delta_includes_missing_dates_and_retains_sign():
    assert subtract({'a': 1., 'b': 2.}, {'b': 1., 'c': 3.}) == {'a': -1., 'b': -1., 'c': 3.}
