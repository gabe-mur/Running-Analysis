from datetime import timedelta

import pytest

from run_analysis import weekly_schedule as planner
from run_analysis.prescription_distance import canonical_distance_policy
from scripts.audit_volume_search import cadence_trace, funding_trace, retime, timing_stress
from scripts.replay_synthetic_easy_dose import synthetic_history
from test_typed_continuation import fixture, CONFIG


def test_retime_preserves_dose_dates_and_input_requires_available_option():
    lookup, make = fixture()
    days, states = make(2, {0: ('easy', 3.3)})
    at = states[0].as_of.replace(hour=19)
    options = [[lookup(at)], [states[1]]]
    original = [d.model_dump_json() for d in days]
    changed = retime(days, 0, at, options)
    assert changed[0].recommendation.planned_for == at
    assert changed[0].planned_at == at
    assert changed[0].recommendation.model_dump(exclude={'planned_for'}) == \
        days[0].recommendation.model_dump(exclude={'planned_for'})
    assert [d.model_dump_json() for d in days] == original
    with pytest.raises(ValueError, match='available'):
        retime(days, 0, at.replace(hour=17), options)
    with pytest.raises(ValueError, match='same physical date'):
        retime(days, 1, states[1].as_of, options)


def test_calendar_cadence_trace_is_not_a_time_of_day_penalty():
    lookup, make = fixture()
    days, states = make(10, {0: ('intervals', 4.5), 9: ('intervals', 4.5)})
    states[0] = states[0].model_copy(update={'days_since_quality_run': 0., 'days_since_long_run': None})
    inputs = dict(daily_states=states, config={'coaching': {'quality_recency_reference_days': 7}})
    trace = cadence_trace(days, inputs)
    assert trace == [dict(role='quality', date=days[9].date.isoformat(), age=9., cost=20.)]
    at = states[0].as_of.replace(hour=19)
    assert cadence_trace(retime(days, 0, at, [[lookup(at)]]), inputs) == trace


def test_existing_funding_threshold_is_measured_not_relaxed():
    _, make = fixture()
    days, states = make(21, {0: ('easy', 4.)})
    inputs = dict(daily_states=states, config=CONFIG,
        expected_target_projector=lambda planned, n: [(17., 18.)] * n)
    with canonical_distance_policy():
        result = funding_trace(inputs, days)
    assert result['target'] == (51., 54.)
    assert result['raw_violation'] == pytest.approx(51.-4.-result['allowance'])
    assert result['gated_violation'] == pytest.approx(result['raw_violation']+result['ordinary'])
    assert planner._gate_aligned_hard_funding_violation(.5, 3.5) == .5
    assert planner._gate_aligned_hard_funding_violation(.5001, 3.5) == pytest.approx(4.0001)


def test_timing_panel_covers_both_sessions_without_changing_prescriptions():
    lookup, make = fixture()
    days, states = make(4, {0: ('easy', 3.5), 1: ('intervals', 4.5)})
    template, history = synthetic_history(states[0].as_of)
    options = [[lookup(s.as_of+timedelta(hours=h)) for h in (7, 12, 19)] for s in states]
    inputs = dict(daily_states=states, daily_state_options=options, config=CONFIG,
        request=planner.RecommendationRequest(health_status='normal'))
    original = [d.model_dump_json() for d in days]
    rows = timing_stress(inputs, days, dict(template=template, runs=history))
    assert len(rows) == 9
    assert min(r['elapsed_hours'] for r in rows) == 12
    assert max(r['elapsed_hours'] for r in rows) == 36
    assert all(len(r['checks']) == 2 for r in rows)
    assert [d.model_dump_json() for d in days] == original
