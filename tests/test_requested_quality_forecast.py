from dataclasses import replace
from datetime import timedelta
from math import fsum

import pytest

from run_analysis.adaptive.quality_prescription_contract import Effort, compile_quality
from run_analysis.adaptive.requested_quality_forecast import (
    PaceProfile, SpeedPhase, command_scenario, effort_pace, forecast_physical, forecast_prescription,
)
from run_analysis.workload_accounting import Quantity
from test_quality_dose_policy import AT, menu, CONFIG
from run_analysis.signal_structure import SignalSpan
from scripts.compare_requested_quality_response import saved_effort, trace_metrics


CFG = dict(CONFIG, zones=dict(CONFIG['zones'], z1=(128, 140), z2=(141, 153), z5=(181, 194)))


def fit(slope=.1):
    return dict(issued_at=AT-timedelta(seconds=1), accounting_key='units', source_keys=('past',),
        dynamic=dict(intercept=.4, slope=slope, tau_seconds=45., speed_domain=(0., 6.),
            tau_at_search_boundary=False))


def physical(phases, model=None, **kwargs):
    return forecast_physical(phases, model or fit(), issued_at=AT, accounting_key='units',
        initial_reserve=.45, **kwargs)


def test_requested_intensity_changes_cost_without_a_family_coefficient():
    slow = physical((SpeedPhase('work', 600, 3., 3., 'declared'),))
    fast = physical((SpeedPhase('work', 600, 4., 4., 'declared'),))
    assert fast['predicted_load'] > slow['predicted_load']
    assert slow['source_keys'] == fast['source_keys'] == ('past',)


def test_recovery_carries_response_instead_of_resetting_or_zeroing_load():
    slow = SpeedPhase('recovery', 120, 2., 2., 'declared')
    high = physical((SpeedPhase('work', 300, 4., 4., 'declared'), slow))
    low = physical((SpeedPhase('work', 300, 2., 2., 'declared'), slow))
    assert high['phases'][1]['load'] > low['phases'][1]['load'] > 0
    assert high['diagnostics']['segment_initializations'] == 1


def test_same_exposure_is_identical_independent_of_workout_name():
    p = next(p for p in menu()['prescriptions'] if p.request.family == 'long_intervals')
    # One continuous explicit time block can be authored as either variant.
    r = replace(p.request, work=(p.request.work[0],), recoveries=())
    a = compile_quality(r, easy_pace=p.easy_pace, work_pace=p.work_pace)
    b = compile_quality(replace(r, family='threshold'), easy_pace=p.easy_pace, work_pace=p.work_pace)
    kwargs = dict(accounting_key='units', initial_reserve=.45)
    x, y = (forecast_prescription(p, fit(), **kwargs) for p in (a, b))
    assert x['nominal']['predicted_load'] == y['nominal']['predicted_load']
    assert x['nominal']['normalized_spans'] == y['nominal']['normalized_spans']
    assert x['prescription_key'] != y['prescription_key']


def test_all_six_families_and_units_call_the_same_interface():
    for p in menu()['prescriptions']:
        report = forecast_prescription(p, fit(), accounting_key='units', initial_reserve=.45)
        assert report['nominal']['status'] == 'predicted'
        assert report['all_scenarios_supported']
        assert not report['uncertainty_calibrated'] and not report['planner_influence']
        assert report['nominal']['predicted_load'] == pytest.approx(fsum(v['load'] for v in report['nominal']['phases']))


def test_primary_units_keep_pace_and_duration_coupled():
    for p in menu()['prescriptions']:
        nominal = command_scenario(p)
        assert [v.seconds for v in nominal] == pytest.approx([v.seconds.estimate for v in p.phases])
        assert [v.meters for v in nominal] == pytest.approx([v.meters.estimate for v in p.phases])
        slow, fast = (command_scenario(p, point=point) for point in ('lower_speed', 'higher_speed'))
        for original, a, b in zip(p.phases, slow, fast):
            if original.primary.unit != 'time':
                assert a.meters == pytest.approx(original.meters.estimate)
                assert b.meters == pytest.approx(original.meters.estimate)
                assert a.seconds > b.seconds
            elif original.primary.amount.low == original.primary.amount.high:
                assert a.seconds == b.seconds == original.seconds.estimate


def test_gradual_profile_is_not_a_flat_second_half_and_preserves_compiled_dose():
    p = next(p for p in menu()['prescriptions'] if p.request.family == 'progression')
    phases = command_scenario(p)
    assert phases[1].first_speed < phases[1].last_speed
    assert phases[1].meters == pytest.approx(p.phases[1].meters.estimate)
    flat = replace(phases[1], first_speed=2.8, last_speed=2.8)
    assert physical(phases)['normalized_spans'] != physical((phases[0], flat))['normalized_spans']


def test_short_and_sustained_equal_work_minutes_have_different_responses():
    hard = SpeedPhase('work', 60, 4., 4., 'explicit')
    easy = SpeedPhase('recovery', 90, 2., 2., 'explicit')
    short = physical((hard, easy)*6)
    long = physical((replace(hard, seconds=360), replace(easy, seconds=540)))
    assert short['predicted_load'] != pytest.approx(long['predicted_load'])
    assert max(s.last for s in short['normalized_spans']) < max(s.last for s in long['normalized_spans'])


def test_integration_refinement_and_athlete_speed_scaling():
    phases = (SpeedPhase('work', 600, 2., 4., 'declared'),)
    a, b = physical(phases), physical(phases, step_seconds=.25)
    assert a['predicted_load'] == pytest.approx(b['predicted_load'], rel=1e-5)
    scaled = (replace(phases[0], first_speed=4., last_speed=8.),)
    assert physical(scaled, fit(slope=.05))['predicted_load'] == pytest.approx(a['predicted_load'])


def test_future_incompatible_and_invalid_response_are_rejected_not_clipped():
    phases = (SpeedPhase('work', 300, 15., 15., 'unsupported'),)
    report = physical(phases)
    assert report['status'] == 'predicted_hr_outside_reference' and report['predicted_load'] is None
    assert report['normalized_spans'][-1].last > 1.
    with pytest.raises(ValueError, match='Causal compatible'):
        physical(phases, dict(fit(), issued_at=AT+timedelta(seconds=1)))
    with pytest.raises(ValueError, match='Causal compatible'):
        physical(phases, dict(fit(), accounting_key='other'))


def test_missing_response_is_unknown_not_zero_or_a_family_ban():
    p = menu()['prescriptions'][0]
    result = forecast_prescription(p, None, accounting_key='units', initial_reserve=.45)
    assert result['nominal']['predicted_load'] is None and result['scenario_envelope'] is None
    assert not result['all_scenarios_supported']


def test_effort_query_does_not_transfer_interval_pace_to_moderate_work():
    fast = effort_pace(Effort(('Z4 effort',), ('Z4 effort',), 'purposeful'), fit(), CFG, issued_at=AT)
    moderate = effort_pace(Effort(('Z2',), ('Z3',), 'gradual'), fit(), CFG, issued_at=AT)
    assert moderate['pace'].seconds_per_meter.estimate > fast['pace'].seconds_per_meter.estimate
    assert moderate['profile'].first_speed.estimate < moderate['profile'].last_speed.estimate
    assert not moderate['calibrated_effort_execution']


def test_unknown_words_and_nonpositive_inversion_do_not_invent_targets():
    unknown = effort_pace(Effort(('strong',), ('strong',), 'purposeful'), fit(), CFG, issued_at=AT)
    assert unknown['status'] == 'unresolved_requested_effort_numerics' and unknown['pace'] is None
    bad = dict(fit(), dynamic=dict(fit()['dynamic'], intercept=.8))
    result = effort_pace(Effort(('Z2',), ('Z3',), 'gradual'), bad, CFG, issued_at=AT)
    assert result['status'] == 'unsupported_effort_response_range' and result['pace'] is None
    with pytest.raises(ValueError, match='Future response'):
        effort_pace(Effort(('Z4',), ('Z4',), 'purposeful'), dict(fit(), issued_at=AT+timedelta(seconds=1)), CFG, issued_at=AT)


def test_explicit_effort_profile_is_frozen_into_forecast_identity():
    original = next(p for p in menu()['prescriptions'] if p.request.family == 'progression')
    q = effort_pace(original.phases[1].effort, fit(), CFG, issued_at=AT)
    p = compile_quality(original.request, easy_pace=original.easy_pace, work_pace=q['pace'])
    a = forecast_prescription(p, fit(), accounting_key='units', initial_reserve=.45, profiles={1:q['profile']})
    b = forecast_prescription(p, fit(), accounting_key='units', initial_reserve=.45)
    assert a['prescription_key'] == b['prescription_key'] and a['fingerprint'] != b['fingerprint']
    assert a['nominal']['predicted_load'] != pytest.approx(b['nominal']['predicted_load'])
    with pytest.raises(ValueError, match='frozen compiled'):
        command_scenario(p, profiles={1:replace(q['profile'], first_speed=Quantity.point(1., 'bad'))})
    with pytest.raises(ValueError, match='Causal explicit'):
        command_scenario(p, profiles={1:replace(q['profile'], issued_at=AT+timedelta(seconds=1))})


def test_curve_phase_diagnostics_match_known_hr_not_full_phase_average():
    predicted = (SignalSpan(0, 60, .6, .6), SignalSpan(60, 120, .9, .9))
    actual = (SignalSpan(0, 60, .6, .6),)
    metrics = trace_metrics(predicted, actual, (('work', 0, 120),), 130)
    assert metrics['whole']['seconds'] == metrics['phases']['work']['known_seconds'] == 60
    assert metrics['whole']['fraction_error'] == metrics['phases']['work']['fraction_error'] == 0
    assert trace_metrics(predicted, (), (('work', 0, 120),), 130)['whole'] is None


def test_saved_intent_is_not_inferred_from_role_or_observed_hr():
    assert saved_effort(dict(workout_type='tempo_threshold', structure=[]), 'work') is None
    payload = dict(structure=[dict(phase=None, target_zones=['Z1', 'Z2']),
        dict(phase='work', target_zones=['Z2', 'Z3'],
            instruction='Gradually increase, finishing controlled in Z3 without sprinting.')])
    effort = saved_effort(payload, 'work')
    assert effort.start_cues == ('Z2',) and effort.finish_cues == ('Z3',)
    assert effort.purpose == 'gradual'
    finish = saved_effort(payload, 'cooldown')
    assert finish.start_cues == finish.finish_cues == ('Z1', 'Z2')
    assert finish.purpose == 'easy'  # declared nonwork scenario, not invented finish intent
