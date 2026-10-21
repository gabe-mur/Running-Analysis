from datetime import timedelta

from scripts.audit_recency_demand import future_events, at_opportunity, scenario_rates, ORIGIN
from scripts.audit_program_delivery import regular
from run_analysis.recency_demand import compare_recency
from run_analysis.workload_accounting import Quantity


def rates():
    return {r: {c: {'bounds': Quantity(1, 20, basis='shared_scenario')}
                for c in ('running_minutes', 'cardio')} for r in ('easy', 'long', 'quality')}


def compare_at(a, b, purpose, cycle):
    x = [e for e in a if e.purpose == purpose][cycle]
    y = [e for e in b if e.purpose == purpose][cycle]
    return compare_recency(at_opportunity([], a, x), x, at_opportunity([], b, y), y)


def test_short_support_has_opposite_effects_at_quality_and_long():
    a = future_events(regular()+[(5, .9, 'easy', .9, None, None)], rates(), 'same_units')
    b = future_events(regular(easy=5.3), rates(), 'same_units')
    assert compare_at(a, b, 'threshold', 2)['recency_demand_order'] == 'left_no_greater'
    assert compare_at(a, b, 'long', 2)['recency_demand_order'] == 'right_no_greater'


def test_compression_is_not_globally_better_or_worse():
    a = future_events([(0, 8.8, 'long', 8.8, None, None), (1, 4.9, 'easy', 4.9, None, None),
        (2, 4.5, 'quality', 2., 'threshold', (300., 300., 300.))], rates(), 'same_units')
    b = future_events(regular(long=8.8, easy=4.9), rates(), 'same_units')
    assert compare_at(a, b, 'threshold', 2)['recency_demand_order'] == 'right_no_greater'
    assert compare_at(a, b, 'long', 2)['recency_demand_order'] == 'left_no_greater'


def test_future_horizon_cannot_affect_prior_opportunity():
    a = future_events(regular(), rates(), 'same_units', horizon_days=21)
    b = future_events(regular(), rates(), 'same_units', horizon_days=56)
    assert all(e.source == 'planned' and e.day >= ORIGIN for e in a+b)
    assert compare_at(a, b, 'threshold', 1)['recency_demand_order'] == 'equal'


def test_missing_role_history_is_unknown_not_a_session_exclusion():
    import datetime
    r = scenario_rates([], datetime.datetime(2026, 9, 1))
    a = future_events(regular(), r, 'same_units')
    assert len(a) == 12
    assert all(t.bounds is None for e in a for t in e.terms if t.channel == 'cardio')


def test_reference_filter_keeps_original_history_separate():
    import datetime
    cutoff = datetime.datetime(2026, 9, 1)
    def row(key, start, eligible, value):
        return dict(key=key, start=start, eligible=eligible, role='easy', miles=2.,
            quantities={c: Quantity.point(value, 'recorded') for c in ('running_minutes', 'cardio')})
    rows = [row('valid', cutoff, True, 20), row('old', cutoff-timedelta(days=1), True, 999),
            row('illness', cutoff, False, 999)]
    r = scenario_rates(rows, cutoff)['easy']['cardio']
    assert r['keys'] == ['valid']
    assert r['bounds'].low == r['bounds'].high == 10
    assert len(rows) == 3  # Anchor exclusions must not delete actual exposure.
