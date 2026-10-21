from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

import pytest

from run_analysis.program_constraints import DateRestriction, ReportedDistanceReference, assess_program_constraints
from run_analysis.recency_demand import CHANNELS, DemandEvent, DemandTerm
from run_analysis.workload_accounting import Quantity

DAY = date(2026, 9, 25)
AT = datetime(2026, 9, 25, 12, tzinfo=timezone.utc)


def test_reported_distance_allows_current_level_without_fabricated_history_or_extra_growth():
    report=ReportedDistanceReference('reported',AT,DAY,8.)
    r=audit([event('normal',0,8.)],history=[],history_complete_since=None,reported_distance=report)
    row=r['sessions'][0]
    assert row['distance_guard']=='within_guard'
    assert row['lower_reference_miles'] is None
    assert row['reference_keys']==[]
    assert row['reported_distance_reference']['basis']=='self_report_not_physical_history'
    assert not r['progression_accepted']
    assert audit([event('growth',0,8.1)],history=[],history_complete_since=None,
        reported_distance=report)['sessions'][0]['distance_guard']=='unresolved_reference'


def test_reported_distance_ages_out_and_does_not_override_real_restrictions():
    report=ReportedDistanceReference('reported',AT,DAY-timedelta(days=31),8.)
    assert audit([event('normal',0,8.)],history=[],history_complete_since=None,
        reported_distance=report)['sessions'][0]['distance_guard']=='unresolved_reference'
    r=audit([event('normal',0,8.)],history=[],reported_distance=replace(report,reference_day=DAY),
        restrictions=[restriction()])
    assert r['status']=='constraint_conflict'
    with pytest.raises(ValueError,match='Future reported'):
        audit([event('normal',0,8.)],reported_distance=replace(report,available_at=AT+timedelta(days=1)))


def event(key, offset, miles, source='planned', purpose='long'):
    return DemandEvent(key, DAY+timedelta(days=offset), tuple(DemandTerm(c, key+':'+c, 1.,
        (miles if isinstance(miles, Quantity) else Quantity.point(miles, 'fixture'))
        if c == 'miles' else None) for c in CHANNELS), 'same_units', purpose, None, source)


def audit(planned, history=None, **kwargs):
    opts = dict(as_of=AT, history_complete_since=DAY-timedelta(days=60), plan_complete=True)
    opts.update(kwargs)
    return assess_program_constraints(planned, [event('past', -2, 8.4, 'observed')] if history is None else history, **opts)


def restriction(kind='reserved_rest', **changes):
    return replace(DateRestriction('restriction', DAY, DAY+timedelta(days=1), AT, kind, 'explicit_test'), **changes)


def test_committed_ceiling_rounds_down():
    a = audit([event('today', 0, 9.2)])['sessions'][0]
    b = audit([event('today', 0, 9.3)])['sessions'][0]
    assert a['committed_ceiling_miles'] == 9.2
    assert a['distance_guard'] == 'within_guard'
    assert b['distance_guard'] == 'exceeds_guard'


def test_exact_ten_percent_is_permitted_by_guard_only():
    r = audit([event('today', 0, 11.)], [event('past', -1, 10., 'observed')])
    assert r['status'] == 'passes_checked_constraints_conditionally'
    assert r['recovery_feasibility'] == 'not_established'
    assert not r['progression_accepted'] and not r['running_authorized']


@pytest.mark.parametrize('role', ['long', 'easy', 'threshold'])
def test_label_does_not_hide_distance_spike(role):
    assert audit([event('today', 0, 9.3, purpose=role)])['sessions'][0]['distance_guard'] == 'exceeds_guard'


def test_incomplete_history_does_not_prove_violation():
    r = audit([event('today', 0, 9.3)], history_complete_since=None)
    assert r['sessions'][0]['distance_guard'] == 'unresolved_reference'


def test_known_lower_reference_still_supports_guard_with_incomplete_history():
    assert audit([event('today', 0, 9.2)], history_complete_since=None)['sessions'][0]['distance_guard'] == 'within_guard'


def test_uncertain_distance_preserved():
    history = [event('past', -1, Quantity(8., 9., basis='uncertain_distance'), 'observed')]
    assert audit([event('today', 0, 8.8)], history)['sessions'][0]['distance_guard'] == 'within_guard'
    assert audit([event('today', 0, 9.2)], history)['sessions'][0]['distance_guard'] == 'unresolved_reference'
    assert audit([event('today', 0, 10.)], history)['sessions'][0]['distance_guard'] == 'exceeds_guard'


def test_no_history_is_unknown_not_zero_capacity():
    assert audit([event('today', 0, 2.)], [])['sessions'][0]['distance_guard'] == 'unresolved_reference'


@pytest.mark.parametrize('kind', ['reserved_rest', 'health_review', 'unsafe_weather'])
def test_explicit_restriction_wins_over_distance_guard(kind):
    r = audit([event('today', 0, 3.)], restrictions=[restriction(kind)])['sessions'][0]
    assert r['status'] == 'constraint_conflict'
    assert r['distance_guard'] == 'within_guard'
    assert r['restrictions'][0]['kind'] == kind


def test_later_notice_cannot_change_earlier_decision():
    n = restriction(available_at=AT+timedelta(days=1))
    r = audit([event('today', 0, 3.)], restrictions=[n])
    assert r['status'] == 'passes_checked_constraints_conditionally'
    assert r['ignored_unavailable_restriction_keys'] == [n.key]


def test_blocked_projection_cannot_grant_headroom():
    r = audit([event('blocked', 0, 9.2), event('later', 2, 10.1)], restrictions=[restriction()])
    second = r['sessions'][1]
    assert second['committed_ceiling_miles'] == 9.2
    assert second['projected_reference_keys'] == []
    assert second['excluded_prefix_keys'] == ['blocked']


def test_failing_projection_cannot_grant_headroom():
    r = audit([event('too_far', 0, 10.), event('later', 2, 11.)])
    assert r['sessions'][0]['distance_guard'] == 'exceeds_guard'
    assert r['sessions'][1]['committed_ceiling_miles'] == 9.2


def test_future_projection_cannot_retroactively_grant_headroom():
    r = audit([event('future', 2, 11.), event('now', 0, 10.)])
    assert r['sessions'][0]['key'] == 'now'
    assert r['sessions'][0]['distance_guard'] == 'exceeds_guard'


def test_valid_projection_dependency_is_visible_not_a_growth_endorsement():
    r = audit([event('first', 0, 9.2), event('second', 2, 10.1), event('third', 4, 11.1)])
    assert r['status'] == 'passes_checked_constraints_conditionally'
    assert r['sessions'][1]['projected_reference_keys'] == ['first']
    assert not r['progression_accepted']  # Guard alone permits compound growth: must not approve it.


def test_expected_completion_replacement_preserves_next_ceiling():
    projected = audit([event('first', 0, 9.2), event('second', 2, 10.1)])['sessions'][1]
    observed = audit([event('second', 2, 10.1)],
        [event('past', -2, 8.4, 'observed'), event('first', 0, 9.2, 'observed')],
        as_of=AT+timedelta(days=1))['sessions'][0]
    assert projected['committed_ceiling_miles'] == observed['committed_ceiling_miles']
    assert projected['distance_guard'] == observed['distance_guard']
    assert observed['projected_reference_keys'] == []


@pytest.mark.parametrize('offset, expected', [(-30, 'within_guard'), (-31, 'unresolved_reference')])
def test_lookback_boundary_declared(offset, expected):
    assert audit([event('today', 0, 8.4)], [event('old', offset, 8.4, 'observed')])['sessions'][0]['distance_guard'] == expected


def test_horizon_tail_does_not_change_prefix():
    a = [event('first', 0, 8.4)]
    assert audit(a)['sessions'] == audit(a+[event('tail', 50, 8.4)])['sessions'][:1]


def test_duplicate_and_same_day_plans_rejected():
    with pytest.raises(ValueError, match='Duplicate'):
        audit([event('same', 0, 8.), event('same', 1, 8.)])
    with pytest.raises(ValueError, match='Same-day'):
        audit([event('a', 0, 8.), event('b', 0, 8.)])


def test_actual_same_day_order_not_invented():
    with pytest.raises(ValueError, match='same-day'):
        audit([event('next', 1, 8.)], [event('today', 0, 8., 'observed')])


def test_distance_commits_to_display_precision():
    with pytest.raises(ValueError, match='one-decimal'):
        audit([event('unrounded', 0, 9.24)])


def test_complete_past_can_establish_today_violation_without_future_coverage():
    r = audit([event('today', 0, 9.3)], plan_complete=False)
    assert r['sessions'][0]['distance_guard'] == 'exceeds_guard'


def test_later_unknown_future_days_prevent_definite_violation():
    r = audit([event('later', 3, 9.3)], plan_complete=False)
    assert r['sessions'][0]['distance_guard'] == 'unresolved_reference'
