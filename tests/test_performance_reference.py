from copy import deepcopy
from dataclasses import replace
from datetime import timedelta

import pytest

from run_analysis.athlete_projection import (
    PaceSample, ForecastPhase, snapshot, compare_completion,
)
from run_analysis.performance_reference import (
    freeze_aerobic_reference, aerobic_speed, phase_pace_reference, compare_aerobic_completion,
)
from run_analysis.workload_accounting import Quantity
from test_athlete_projection import NOW, KEY, session, point, issued


def steady(key='prior', days=1, speed=2., reserve=.6):
    return session(key, days, speed=speed,
        samples=(PaceSample('easy', 120, speed*120, 'steady', reserve),))


def completion(speed=2., reserve=.6, **changes):
    return replace(steady('done', -1, speed, reserve), **changes)


def resolve(state, phase='work', reserve=.85, **dose):
    return phase_pace_reference(state, phase, equilibrium_reserve=point(reserve), **dose)


def test_phase_easy_reference_is_band_conditioned_not_midpoint_conditioned():
    # A band containing only high-end observations cannot establish pace at
    # its midpoint. The separate forecast midpoint is an assumption, not a
    # measured HR associated with this pace reference.
    state = snapshot([steady(reserve=.69, speed=2.5)], NOW, KEY)
    band = Quantity(.60, .70, .65, 'declared_effort_band')
    reference = phase_pace_reference(state, 'easy', equilibrium_reserve=band)
    assert reference['speed_mps'].estimate == pytest.approx(2.5)
    assert reference['status'] == 'phase_reference'
    other_center = phase_pace_reference(state, 'easy', equilibrium_reserve=replace(band, estimate=.62))
    assert other_center == reference


def test_existing_work_pace_does_not_establish_equilibrium_hr_at_zone_midpoint():
    state = snapshot([session(samples=(PaceSample('work', 100, 400, 'recorded_lap'),))], NOW, KEY)
    first = phase_pace_reference(state, 'work', seconds=100,
        equilibrium_reserve=Quantity(.75, .90, .80, 'declared_work_band'))
    second = phase_pace_reference(state, 'work', seconds=100,
        equilibrium_reserve=Quantity(.75, .90, .88, 'declared_work_band'))
    assert first == second
    assert first['speed_mps'].estimate == pytest.approx(4)


@pytest.mark.parametrize('dose', [dict(seconds=300), dict(meters=500), dict(seconds=301)])
def test_novel_work_uses_one_short_rep_session_without_a_matching_gate(dose):
    prior = session(samples=(PaceSample('work', 100, 400, 'recorded_lap'),))
    ref = resolve(snapshot([prior], NOW, KEY), **dose)
    assert ref['source_count'] == 1
    assert ref['speed_mps'].estimate == pytest.approx(4)
    assert not ref['prediction_error_calibrated']
    assert ref['status'] == 'phase_transfer'


def test_similarity_is_continuous_across_old_factor_two_boundary():
    a = session('short', 2, samples=(PaceSample('work', 100, 400, 'lap'),))
    b = session('long', 1, samples=(PaceSample('work', 300, 900, 'lap'),))
    state = snapshot([a, b], NOW, KEY)
    left = resolve(state, seconds=199.999)['speed_mps'].estimate
    right = resolve(state, seconds=200.001)['speed_mps'].estimate
    assert left == pytest.approx(right, rel=1e-5)
    assert resolve(state, seconds=100)['speed_mps'].estimate > resolve(state, seconds=300)['speed_mps'].estimate


def test_repeated_reps_do_not_multiply_session_weight():
    a = session('a', 2, samples=(PaceSample('work', 100, 400, 'lap'),))
    b = session('b', 1, samples=(PaceSample('work', 300, 900, 'lap'),))
    first = resolve(snapshot([a, b], NOW, KEY), seconds=300)
    many = resolve(snapshot([replace(a, pace_samples=a.pace_samples*20), b], NOW, KEY), seconds=300)
    assert first['speed_mps'].estimate == pytest.approx(many['speed_mps'].estimate)
    assert many['source_count'] == 2


def test_first_ever_interval_uses_provisional_athlete_reference():
    state = snapshot([steady()], NOW, KEY)
    for phase in ('work', 'recovery'):
        ref = resolve(state, phase, meters=500)
        assert ref['speed_mps'].estimate == pytest.approx(2*.85/.6)
        assert ref['status'] == 'provisional_aerobic_reference'
        assert ref['transferred_source_count'] == 1
        assert not ref['prediction_error_calibrated']


@pytest.mark.parametrize('meters', [400, 500])
def test_novel_distance_session_has_finite_load_and_preserves_total(meters):
    state = snapshot([steady()], NOW, KEY)
    work = resolve(state, meters=meters)['speed_mps']
    phases = [ForecastPhase('warmup', 600, None, point(2), point(.6), 'test')]
    for index in range(3):
        phases.append(ForecastPhase('work', None, meters, work, point(.85), 'test'))
        if index < 2:
            phases.append(ForecastPhase('recovery', 90, None, point(2), point(.5), 'test'))
    phases.append(ForecastPhase('cooldown', None, 5000, point(2), point(.6), 'test', True))
    f = issued(phases, state)
    assert f['central']['cardio_load'] > 0
    assert all(s['distance_meters'] == pytest.approx(5000) for s in f['scenarios'])


def test_three_by_five_forecasts_without_quality_history():
    state = snapshot([steady()], NOW, KEY)
    pace = resolve(state, seconds=300)['speed_mps']
    f = issued([ForecastPhase('work', 300, None, pace, point(.85), 'test')]*3, state)
    assert f['central']['elapsed_seconds'] == pytest.approx(900)
    assert f['central']['cardio_load'] > 0


def test_no_history_needs_no_pace_for_timed_load_and_accepts_supplied_distance_pace():
    state = snapshot([], NOW, KEY)
    assert resolve(state, seconds=300)['speed_mps'] is None
    f = issued([ForecastPhase('work', 300, None, None, point(.85), 'test')]*3, state)
    assert f['central']['cardio_load'] > 0
    ref = resolve(state, meters=500, supplied_pace=point(3))
    assert ref['status'] == 'supplied_pace'
    assert ref['speed_mps'].estimate == 3


def test_athlete_scaling_not_runner_specific_constants():
    fast = resolve(snapshot([steady(speed=4)], NOW, KEY), meters=500)
    slow = resolve(snapshot([steady(speed=1)], NOW, KEY), meters=500)
    assert fast['speed_mps'].estimate == pytest.approx(4*slow['speed_mps'].estimate)


@pytest.mark.parametrize('reserve', [.1, 1.])
def test_outside_aerobic_transfer_domain_does_not_block_timed_load(reserve):
    state = snapshot([steady()], NOW, KEY)
    ref = resolve(state, reserve=reserve, seconds=60)
    assert ref['speed_mps'] is None
    assert ref['status'] == 'supplied_pace_needed_outside_transfer_domain'
    f = issued([ForecastPhase('work', 60, None, None, point(reserve), 'test')], state)
    assert f['central']['cardio_load'] > 0
    assert resolve(state, reserve=reserve, supplied_pace=point(4))['speed_mps'].estimate == 4


def test_reference_keeps_single_session_and_within_run_noise_distinct():
    a = steady('a', 2, speed=2)
    b = steady('b', 1, speed=4)
    ref = freeze_aerobic_reference(snapshot([a, b], NOW, KEY))
    repeated = freeze_aerobic_reference(snapshot([replace(a, pace_samples=a.pace_samples*20), b], NOW, KEY))
    assert ref == repeated
    assert aerobic_speed(ref, .6)['source_count'] == 2
    assert aerobic_speed(ref, .6)['speed_mps'].estimate == pytest.approx(8**.5)


@pytest.mark.parametrize('speed,reserve,expected', [
    (2.2, .6, .1), (2., .5, .2), (2., .6, 0.), (1.8, .54, 0.), (1.8, .6, -.1),
])
def test_performance_distinguishes_improvement_from_merely_running_easier(speed, reserve, expected):
    ref = freeze_aerobic_reference(snapshot([steady()], NOW, KEY))
    result = compare_aerobic_completion(ref, completion(speed, reserve))
    assert result['speed_change_fraction'] == pytest.approx(expected, abs=1e-12)
    assert not result['fitness_gain_established']
    assert not result['adaptation_authorized']
    assert not result['context_adjusted']


def test_performance_interpolates_supported_curve_instead_of_forcing_proportional_hr():
    s = session(samples=(PaceSample('easy', 120, 240, 'steady', .5),
                         PaceSample('easy', 120, 480, 'steady', .7)))
    ref = freeze_aerobic_reference(snapshot([s], NOW, KEY))
    result = compare_aerobic_completion(ref, completion(3, .6))
    assert result['speed_change_fraction'] == pytest.approx(0)
    assert result['windows'][0]['transferred_source_count'] == 0


def test_lower_cardio_load_alone_cannot_become_performance_gain():
    state = snapshot([steady()], NOW, KEY)
    f = issued(state=state)
    done = completion(cardio=point(1), pace_samples=())
    result = compare_completion(f, done)
    assert result['cardio_status'] == 'below_scenarios'
    assert result['aerobic_performance']['speed_change_fraction'] is None


def test_positive_evidence_survives_broad_load_scenario_range():
    state = snapshot([steady()], NOW, KEY)
    f = issued([ForecastPhase('easy', 2400, None, point(2), Quantity(.3, .9), 'test')], state)
    result = compare_completion(f, completion(2.2, .6, cardio=point(40)))
    assert result['cardio_status'] == 'within_scenarios'
    assert result['aerobic_performance']['speed_change_fraction'] == pytest.approx(.1)


def test_new_evidence_updates_next_reference_but_not_its_own_forecast():
    prior = steady()
    state = snapshot([prior], NOW, KEY)
    f = issued(state=state)
    original = deepcopy(f)
    done = completion(2.2, .6)
    first = compare_completion(f, done)
    after = snapshot([prior, done], done.available_at, KEY)
    newer = freeze_aerobic_reference(after)
    assert 2 < aerobic_speed(newer, .6)['speed_mps'].estimate < 2.2
    assert compare_completion(f, done) == first
    assert f == original
    assert len(newer.sessions) == 2


def test_future_late_upload_and_health_excluded_data_cannot_change_reference():
    prior = steady()
    future = steady('future', -1, speed=10)
    late = replace(steady('late', 2, speed=10), available_at=NOW+timedelta(days=2))
    sick = replace(steady('sick', 3, speed=10), capacity_eligible=False)
    original = snapshot([prior], NOW, KEY)
    other = snapshot([prior, future, late, sick], NOW, KEY)
    assert freeze_aerobic_reference(original) == freeze_aerobic_reference(other)
    assert resolve(original) == resolve(other)


def test_flagged_completion_is_not_capacity_evidence():
    ref = freeze_aerobic_reference(snapshot([steady()], NOW, KEY))
    result = compare_aerobic_completion(ref, completion(3, .6, capacity_eligible=False))
    assert result['status'] == 'excluded_capacity_evidence'
    assert result['speed_change_fraction'] is None


def test_work_rep_hr_lag_does_not_masquerade_as_aerobic_efficiency():
    ref = freeze_aerobic_reference(snapshot([steady()], NOW, KEY))
    done = completion(pace_samples=(PaceSample('work', 60, 400, 'lap', .6),))
    assert compare_aerobic_completion(ref, done)['speed_change_fraction'] is None


def test_accounting_and_chronology_are_checked():
    ref = freeze_aerobic_reference(snapshot([steady()], NOW, KEY))
    with pytest.raises(ValueError, match='Mixed'):
        compare_aerobic_completion(ref, completion(accounting_key='other'))
    with pytest.raises(ValueError, match='held out'):
        compare_aerobic_completion(ref, steady())


@pytest.mark.parametrize('kwargs', [dict(seconds=-1), dict(meters=0), dict(lookback_days=0)])
def test_invalid_dose_is_not_silently_forecast(kwargs):
    with pytest.raises(ValueError):
        resolve(snapshot([steady()], NOW, KEY), **kwargs)
