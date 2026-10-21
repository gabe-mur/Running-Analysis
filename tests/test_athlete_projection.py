from dataclasses import replace
from datetime import datetime, timedelta, timezone
from copy import deepcopy

import pytest

from run_analysis.athlete_projection import (
    PaceSample, SessionEvidence, ForecastPhase, LoadSeed, accounting_signature,
    snapshot, exposure_summary, pace_reference, project_load_history,
    forecast_trace, issue_forecast, compare_completion,
)
from run_analysis.response_model import Policy, ewma
from run_analysis.training_signals import integrate_cardio_trace
from run_analysis.workload_accounting import Quantity

NOW = datetime(2026, 9, 25, tzinfo=timezone.utc)
KEY = accounting_signature(50, 200)


def point(value):
    return Quantity.point(value, 'explicit_test_assumption')


def session(key='prior', days=1, role='easy', speed=2., cardio=None, samples=None):
    end = NOW-timedelta(days=days)
    return SessionEvidence(key, end-timedelta(minutes=40), end, end, role, 2400*speed, 2400,
        cardio if cardio is not None else point(40), KEY,
        samples if samples is not None else (PaceSample('easy', 2400, 2400*speed, 'raw'),),
        ordinary_reference_eligible=role == 'easy')


def phase(name='easy', seconds=600, reserve=.6, speed=2.):
    return ForecastPhase(name, seconds, None, point(speed) if speed else None,
                         point(reserve), 'predeclared_assumption')


def issued(phases=None, state=None):
    return issue_forecast(state or snapshot([], NOW, KEY), phases or [phase()],
        initial_reserve=point(.6), initial_source='test', tau_seconds=(45.,))


def test_cutoff_uses_completion_and_upload_availability():
    prior = session()
    late = replace(session('late'), available_at=NOW+timedelta(days=1))
    future = session('future', days=-1)
    state = snapshot([future, late, prior], NOW, KEY)
    assert state.sessions == (prior,)
    assert dict(state.excluded) == {'late': 'not_available', 'future': 'not_completed'}


def test_future_changed_or_added_data_does_not_change_issued_values():
    prior = session()
    future = session('future', -1, speed=10)
    a = snapshot([prior], NOW, KEY)
    b = snapshot([prior, future], NOW, KEY)
    assert pace_reference(a, 'easy') == pace_reference(b, 'easy')
    assert exposure_summary(a) == exposure_summary(b)
    assert issued(state=a) == issued(state=b)


def test_frozen_snapshot_does_not_update_when_completion_arrives():
    history = [session()]
    state = snapshot(history, NOW, KEY)
    forecast = issued(state=state)
    before = deepcopy(forecast)
    history.append(session('later', -1))
    assert forecast == before
    assert len(state.sessions) == 1


def test_duplicate_and_overlapping_sessions_are_rejected():
    with pytest.raises(ValueError, match='Duplicate'):
        snapshot([session(), session()], NOW, KEY)
    with pytest.raises(ValueError, match='Overlapping'):
        snapshot([session(), session('second')], NOW, KEY)


def test_accounting_mismatch_cannot_enter_ledger_or_completion_comparison():
    other = replace(session(), accounting_key=accounting_signature(50, 210))
    state = snapshot([other], NOW, KEY)
    assert not state.sessions
    with pytest.raises(ValueError, match='mixed'):
        compare_completion(issued(), replace(other, start=NOW, end=NOW+timedelta(minutes=40),
                                            available_at=NOW+timedelta(minutes=40)))
    with pytest.raises(ValueError, match='policy'):
        issue_forecast(snapshot([], NOW, KEY), [phase()], initial_reserve=point(.6),
            initial_source='test', policy=replace(Policy(), intensity_exponent=1.67))


def test_ordinary_reference_excludes_support_and_long_doses():
    rows = [session('easy', 1), replace(session('support', 2, role='recovery'), moving_seconds=600),
            replace(session('long', 3, role='long'), moving_seconds=2400)]
    result = exposure_summary(snapshot(rows, NOW, KEY))
    assert result['ordinary_easy_minutes'] == 40
    assert result['long_count'] == 1
    assert result['run_count'] == 3


def test_sessions_not_repetitions_get_independent_pace_votes():
    slow = session('slow', 2, samples=(PaceSample('work', 120, 360, 'lap'),)*20)
    fast = session('fast', 1, samples=(PaceSample('work', 120, 600, 'lap'),))
    result = pace_reference(snapshot([slow, fast], NOW, KEY), 'work', seconds=120)
    assert result['source_count'] == 2
    assert result['speed_mps'].estimate == 4


def test_work_domain_is_not_extrapolated_from_easy_or_short_reps():
    s = replace(session(), pace_samples=(PaceSample('work', 100, 400, 'lap'),))
    state = snapshot([s], NOW, KEY)
    assert pace_reference(state, 'work', seconds=300)['speed_mps'] is None
    assert pace_reference(state, 'easy')['speed_mps'] is None
    assert pace_reference(state, 'work', meters=400)['status'] == 'single_session'


def test_cold_start_keeps_history_unknown_but_can_forecast_explicit_timed_work():
    state = snapshot([], NOW, KEY)
    assert pace_reference(state, 'easy')['status'] == 'unsupported'
    assert exposure_summary(state)['ordinary_easy_minutes'] is None
    assert project_load_history(state)['atl'] is None
    forecast = issued([phase(speed=None)], state)
    assert forecast['cardio_scenario_range'] == pytest.approx((10, 10))
    assert forecast['scenarios'][0]['distance_meters'] is None


def test_unknown_initial_state_is_not_zero_even_with_real_sessions():
    state = snapshot([session()], NOW, KEY)
    assert project_load_history(state, complete_recording_since=NOW-timedelta(days=5))['ctl'] is None


def test_load_history_rate_convention_and_missing_bounds():
    prior = session(cardio=Quantity(30, 50))
    state = snapshot([prior], NOW, KEY)
    seed = LoadSeed(prior.start, point(10), point(20), 'explicit_known_seed', KEY)
    result = project_load_history(state, seed=seed, complete_recording_since=seed.at)
    duration = 40/1440
    assert result['atl'] == pytest.approx([ewma(ewma(10, x/duration, duration, 7), 0, 1, 7) for x in (30, 50)])
    assert result['ctl'] == pytest.approx([ewma(ewma(20, x/duration, duration, 42), 0, 1, 42) for x in (30, 50)])
    assert not result['injury_risk_interpretation']


def test_unavailable_completed_session_is_not_assumed_rest():
    late = replace(session(), available_at=NOW+timedelta(days=1))
    state = snapshot([late], NOW, KEY)
    seed = LoadSeed(NOW-timedelta(days=10), point(0), point(0), 'declared_inactivity', KEY)
    assert project_load_history(state, seed=seed, complete_recording_since=seed.at)['atl'] is None


def test_clock_rollover_has_no_special_load_jump():
    prior = session()
    seed = LoadSeed(prior.start, point(10), point(20), 'test', KEY)
    left = project_load_history(snapshot([prior], NOW-timedelta(seconds=1), KEY), seed=seed,
                               complete_recording_since=seed.at)
    right = project_load_history(snapshot([prior], NOW+timedelta(seconds=1), KEY), seed=seed,
                                complete_recording_since=seed.at)
    assert right['atl'][0] == pytest.approx(ewma(left['atl'][0], 0, 2/86400, 7))


def test_distance_forecast_scales_with_athlete_speed_without_hr_discount():
    fast = ForecastPhase('easy', None, 1609.344*5, point(1609.344/300), point(.6), '5min_mile')
    slow = replace(fast, speed_mps=point(1609.344/1200), assumption_source='20min_mile')
    a, b = issued([fast]), issued([slow])
    assert a['duration_scenario_range'] == pytest.approx((1500, 1500))
    assert b['duration_scenario_range'] == pytest.approx((6000, 6000))
    assert b['cardio_scenario_range'][0] == pytest.approx(a['cardio_scenario_range'][0]*4)


def test_short_work_hr_lags_and_recovery_carries_previous_hr():
    spans, rows, _ = forecast_trace([phase(seconds=30, reserve=.9), phase('recovery', 30, .5)],
                                    initial_reserve=.6, tau_seconds=45)
    assert .6 < spans[29].last < .9
    assert spans[30].first == spans[29].last
    assert .5 < spans[-1].last < spans[29].last


def test_phase_splitting_and_smaller_steps_preserve_forecast():
    a, _, duration = forecast_trace([phase(seconds=100, reserve=.9)], initial_reserve=.5, tau_seconds=45)
    b, _, _ = forecast_trace([phase(seconds=37, reserve=.9), phase(seconds=63, reserve=.9)],
                             initial_reserve=.5, tau_seconds=45)
    fine, _, _ = forecast_trace([phase(seconds=100, reserve=.9)], initial_reserve=.5, tau_seconds=45, step_seconds=.1)
    load = integrate_cardio_trace(a, duration)['total_load']
    assert integrate_cardio_trace(b, duration)['total_load'] == pytest.approx(load)
    assert integrate_cardio_trace(fine, duration)['total_load'] == pytest.approx(load, rel=1e-5)


def test_perfect_forecast_completion_uses_same_accounting():
    state = snapshot([session()], NOW, KEY)
    forecast = issued([phase()], state)
    spans, _, duration = forecast_trace(forecast['phases'], initial_reserve=.6, tau_seconds=45)
    actual_load = integrate_cardio_trace(spans, duration)['total_load']
    end = NOW+timedelta(seconds=duration)
    done = SessionEvidence('done', NOW, end, end, 'easy', duration*2, duration, point(actual_load), KEY)
    assert compare_completion(forecast, done)['cardio_status'] == 'within_scenarios'
    after = snapshot([*state.sessions, done], end, KEY)
    assert len(after.sessions) == 2
    assert forecast['history_keys'] == ('prior',)


def test_higher_effort_completion_is_discrepancy_not_date_edit_permission():
    f = issued()
    end = NOW+timedelta(minutes=10)
    actual = SessionEvidence('done', NOW, end, end, 'easy', 1200, 600, point(50), KEY)
    result = compare_completion(f, actual)
    assert result['cardio_status'] == 'above_scenarios'
    assert not result['adaptation_authorized']


def test_completion_in_history_is_rejected():
    s = session()
    with pytest.raises(ValueError, match='held out'):
        compare_completion(issued(state=snapshot([s], NOW, KEY)), s)


def test_distance_completion_preserves_total_for_every_speed_scenario():
    p = phase(seconds=60)
    end = ForecastPhase('finish_easy', None, 1000, Quantity(1, 3), point(.6), 'test', True)
    f = issued([p, end])
    assert all(s['distance_meters'] == pytest.approx(1000) for s in f['scenarios'])
    assert f['duration_scenario_range'] == pytest.approx((60+880/3, 940))


def test_impossible_structure_does_not_silently_increase_total_distance():
    end = ForecastPhase('finish_easy', None, 100, point(2), point(.6), 'test', True)
    with pytest.raises(ValueError, match='exhaust'):
        issued([phase(seconds=60), end])


def test_scenario_panel_includes_fast_high_hr_and_slow_high_hr():
    p = ForecastPhase('easy', None, 1000, Quantity(1, 3), Quantity(.5, .8), 'test')
    f = issued([p])
    assert {(s['speed_side'], s['hr_side']) for s in f['scenarios']} == {
        ('low', 'low'), ('high', 'high'), ('low', 'high'), ('high', 'low')}


@pytest.mark.parametrize('changes', [dict(seconds=None, meters=None), dict(seconds=20, meters=100),
                                    dict(seconds=-1), dict(equilibrium_reserve=point(1.1))])
def test_invalid_phase_contract(changes):
    with pytest.raises(ValueError):
        replace(phase(), **changes)


def test_pace_reference_matches_observed_relative_hr_not_fixed_bpm():
    s = session(samples=(PaceSample('easy', 120, 240, 'raw', .6),
                         PaceSample('easy', 120, 360, 'raw', .8)))
    ref = pace_reference(snapshot([s], NOW, KEY), 'easy', reserve_band=Quantity(.55, .65))
    assert ref['speed_mps'].estimate == 2


def test_original_adapter_keeps_good_windows_when_other_windows_have_missing_hr(tmp_path):
    from run_analysis.projection_evidence import evidence_from_activity
    from run_analysis.tcx import parse_tcx
    from test_tcx import make_tcx
    from test_workload_profile import CONFIG
    activity = parse_tcx(make_tcx(tmp_path/'run.tcx')).activities[0]
    template = activity.trackpoints[0]
    start = activity.start_time_utc
    activity.trackpoints = [replace(template, timestamp_utc=start+timedelta(seconds=s),
        distance_m=s*2, heart_rate_bpm=145, point_index=s//10) for s in range(0, 2401, 10)]
    activity.total_elapsed_time_s = activity.lap_recorded_time_s = 2400
    activity.total_distance_m = 4800
    kwargs = dict(key='test', available_at=start+timedelta(seconds=2400), workout_type='easy')
    before = evidence_from_activity(activity, CONFIG, **kwargs)
    activity.trackpoints[70].heart_rate_bpm = None
    after = evidence_from_activity(activity, CONFIG, **kwargs)
    assert 1 < len(after.pace_samples) < len(before.pace_samples)
    assert after.cardio.low < after.cardio.high
    assert all(s.meters/s.seconds == pytest.approx(2) for s in after.pace_samples)


def test_long_pause_suppresses_only_affected_and_post_pause_windows():
    from run_analysis.projection_evidence import steady_pace_samples
    from run_analysis.training_signals import HeartRateReference
    from test_signal_structure import interval
    from test_workload_profile import CONFIG
    rows = [interval(i, seconds=10, speed=2, hr=145, moving=0 if 1200 <= i < 1260 else 10)
            for i in range(0, 2400, 10)]
    samples = steady_pace_samples(rows, rows[0].start.timestamp_utc, CONFIG,
                                  HeartRateReference(50, 200, 'test'))
    assert len(samples) >= 10
    assert len(samples) < 17  # 6-minute warmup exclusion; stop/recovery windows excluded too.


def test_offline_steady_window_coordinates_preserve_default_samples_and_validity():
    from run_analysis.projection_evidence import steady_pace_samples
    from run_analysis.training_signals import HeartRateReference
    from test_signal_structure import interval
    from test_workload_profile import CONFIG
    rows = [interval(i, seconds=10, speed=2, hr=145, moving=0 if 1200 <= i < 1260 else 10)
            for i in range(0,2400,10)]
    origin,hr=rows[0].start.timestamp_utc,HeartRateReference(50,200,'test')
    samples=steady_pace_samples(rows,origin,CONFIG,hr)
    windows=steady_pace_samples(rows,origin,CONFIG,hr,with_windows=True)
    assert [p for _,_,p in windows]==samples
    assert all(b-a==p.seconds for a,b,p in windows)
    assert all(not (a<1260 and b>1200) for a,b,_ in windows)


def test_load_seed_must_match_history_units():
    seed = LoadSeed(NOW-timedelta(days=10), point(0), point(0), 'test', 'Edwards')
    with pytest.raises(ValueError, match='same accounting'):
        project_load_history(snapshot([session()], NOW, KEY), seed=seed, complete_recording_since=seed.at)


def test_health_flag_excludes_capacity_not_completed_load():
    sick = replace(session(), capacity_eligible=False)
    state = snapshot([sick], NOW, KEY)
    assert pace_reference(state, 'easy')['speed_mps'] is None
    assert exposure_summary(state)['recorded_cardio_low'] == 40


def test_easy_label_alone_does_not_redefine_ordinary_dose():
    support = replace(session(), moving_seconds=600, ordinary_reference_eligible=False)
    assert exposure_summary(snapshot([support], NOW, KEY))['ordinary_easy_minutes'] is None
