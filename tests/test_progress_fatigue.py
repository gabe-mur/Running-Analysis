from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from math import exp
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from run_analysis.adaptive.program_objective_trial import (
    AXES, DemandScenario, TrialPolicy, TrialSession, components,
)
from run_analysis.progress_fatigue import fatigue_history, residual
from run_analysis.workload_accounting import Quantity


def row(id, day, doses=(40., 60., 10.)):
    return dict(activity_id=id, start_time=f'{day}T12:00:00+00:00',
        accounting_key='units', **{axis: asdict(Quantity.point(v, 'test')) for axis, v in zip(AXES, doses)})


def fixture():
    end = datetime(2026, 10, 10, 20, tzinfo=timezone.utc)
    rows = [row(i, (end-timedelta(days=i)).date()) for i in range(61)]
    sessions = [SimpleNamespace(activity_id=r['activity_id'], start_time=datetime.fromisoformat(r['start_time'])) for r in rows]
    points = {s.activity_id: SimpleNamespace(included_in_trend=True,
        standardized_pace_min_mile=10+s.activity_id/100, uncertainty_95_min_mile=.4) for s in sessions}
    details = {s.activity_id: {'health_tag': 'normal'} for s in sessions}
    ledger = dict(status='available', reason=None, rows=rows, snapshot_as_of=end.isoformat(),
        fatigue_model=dict(history_start=rows[-1]['start_time'], timezone='UTC',
            accounting_key='units', scales=(40., 60., 10.), scenario=asdict(DemandScenario())))
    return ledger, sessions, points, details, end


def test_residual_matches_scheduler_preceding_demand_exactly():
    scenario = DemandScenario(memory_days=2.3, bound_fraction=.7, axis_weights=(2., 1., .5))
    scales = (40., 60., 10.)
    history = tuple(TrialSession(str(i), date(2026, 10, d), 4., 'easy',
        tuple(Quantity(v*.8, v*1.2, v, 'test') for v in (40., 60., 10.)), source='observed') for i,d in enumerate((1, 3, 5)))
    decision = date(2026, 10, 8)
    planned = TrialSession('new', decision, 4., 'quality', tuple(Quantity.point(v, 'test') for v in scales))
    parts = components(history, (planned,), stage_start=date(2026, 10, 1), decision=decision,
        end=date(2026, 10, 9), scales=scales, policy=TrialPolicy(), scenario=scenario)
    rows = [dict(start_time=f'{r.day}T12:00:00+00:00', **{axis:asdict(q) for axis,q in zip(AXES,r.demand)}) for r in history]
    prior = residual(rows, decision, scales, scenario, ZoneInfo('UTC'), entering=True)
    for axis in AXES:
        assert parts['session_costs'][0]['axis_cost'][axis] == pytest.approx(.5+prior['by_axis'][axis], abs=1e-12)
    assert prior['value'] == pytest.approx(sum(prior['by_axis'][a]*w for a,w in zip(AXES, scenario.axis_weights))/3.5)


def test_daily_decay_excludes_own_date_before_run_and_future():
    rows = [row(1, '2026-10-08'), row(2, '2026-10-10'), row(3, '2026-10-11')]
    scenario = DemandScenario()
    before = residual(rows, date(2026, 10, 10), (40.,60.,10.), scenario, ZoneInfo('UTC'), entering=True)
    after = residual(rows, date(2026, 10, 10), (40.,60.,10.), scenario, ZoneInfo('UTC'))
    assert before['value'] == pytest.approx(exp(-1))
    assert after['value'] == pytest.approx(1+exp(-1))
    # Local dates, not elapsed hours or UTC dates.
    rows = [row(1, '2026-10-10') | {'start_time':'2026-10-10T02:00:00+00:00'}]
    assert residual(rows, date(2026,10,10), (40.,60.,10.), scenario,
        ZoneInfo('America/New_York'), entering=True)['value'] == pytest.approx(exp(-.5))


def test_display_window_does_not_reset_memory_reference_or_comparison():
    inputs = fixture()
    long = fatigue_history(*inputs, 28)
    short = fatigue_history(*inputs, 14)
    assert short['current'] == long['current']
    assert short['usual'] == long['usual']
    assert short['runs'] == long['runs']
    assert short['comparison'] == long['comparison']
    assert len(short['daily']) == 15 and len(long['daily']) == 29
    assert long['current']['value'] > 2


def test_missing_recent_accounting_is_not_displayed_as_recovery():
    ledger, sessions, points, details, end = fixture()
    ledger['rows'] = [r for r in ledger['rows'] if r['activity_id'] != 1]
    result = fatigue_history(ledger, sessions, points, details, end, 28)
    assert result['status'] == 'incomplete'
    assert result['current']['value'] is None
    assert result['usual'] is None
    assert all(r['activity_id'] != 0 for r in result['runs'])
    # Same-day work has not occurred in the entering-run daily estimate.
    ledger, sessions, points, details, end = fixture()
    ledger['rows'] = [r for r in ledger['rows'] if r['activity_id'] != 0]
    result = fatigue_history(ledger, sessions, points, details, end, 28)
    assert result['current']['value'] is None
    assert any(r['activity_id'] == 0 for r in result['runs'])


def test_bounds_preserved_and_short_history_not_zero_filled():
    ledger, sessions, points, details, end = fixture()
    ledger['rows'][0]['cardio'] = asdict(Quantity(30., 90., 50., 'incomplete HR'))
    result = fatigue_history(ledger, sessions, points, details, end, 28)
    assert result['current']['low'] < result['current']['value'] < result['current']['high']
    ledger['fatigue_model']['history_start'] = (end-timedelta(days=3)).isoformat()
    result = fatigue_history(ledger, sessions, points, details, end, 28)
    assert not any(r['complete'] for r in result['daily'])
    assert result['comparison'] is None and result['usual'] is None


def test_performance_comparison_excludes_unsupported_health_and_future_runs():
    ledger, sessions, points, details, end = fixture()
    points[1].included_in_trend = False
    details[2]['health_tag'] = 'sick_or_recovering'
    sessions.append(SimpleNamespace(activity_id=100, start_time=end+timedelta(days=1)))
    points[100] = points[0]
    details[100] = details[0]
    result = fatigue_history(ledger, sessions, points, details, end, 28)
    assert not {1,2,100}.intersection(r['activity_id'] for r in result['runs'])
    assert fatigue_history({}, [], {}, {}, end, 28)['status'] == 'unavailable'
