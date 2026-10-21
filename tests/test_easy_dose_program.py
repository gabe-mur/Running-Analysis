"""Offline integration contracts; not approval of experimental dose policy."""

from dataclasses import replace
from datetime import timedelta
from math import ceil

import pytest

from scripts import audit_dose_assumptions as fixture
from scripts import audit_easy_dose_program as experiment
from scripts import replay_easy_dose_program as replay
from scripts.easy_dose_prototype import Observation, build_profile


def profile():
    return build_profile([Observation("ordinary", fixture.ORIGIN - timedelta(days=2), 3.7)],
                         fixture.ORIGIN, half_life_days=84)


def trajectory(mode="history_trajectory"):
    return experiment.dose_trajectory(profile(), [fixture.ORIGIN.date() + timedelta(days=i) for i in range(21)],
                                      [(16 + i / 10, 18 + i / 10) for i in range(21)], 16,
                                      mode=mode, fixed_fraction=.08)


def test_target_trajectory_is_explicit_not_candidate_generated_growth():
    p = profile()
    result = trajectory()
    assert result.upper_miles[0] == pytest.approx(3.7 * 17 / 16)
    assert result.upper_miles[-1] == pytest.approx(3.7 * 19 / 16)
    assert p.ordinary_planning_miles == 3.7
    assert len(set(trajectory("history_fixed").upper_miles)) == 1


def test_flat_program_target_does_not_compound_with_elapsed_days():
    t = trajectory()
    result = experiment.dose_trajectory(profile(), t.dates, [(16, 16)] * 21, 16,
                                        mode="history_trajectory", fixed_fraction=.08)
    assert result.upper_miles == (3.7,) * 21


def test_same_absolute_date_target_survives_horizon_shift():
    t = trajectory()
    shifted = experiment.dose_trajectory(profile(), t.dates[1:], t.target_ranges[1:], 16,
                                         mode="history_trajectory", fixed_fraction=.08)
    assert shifted.upper_miles == t.upper_miles[1:]


@pytest.mark.parametrize("raises", [False, True])
def test_adapter_restores_production_helper_on_exit(raises):
    original = fixture.planner._easy_allocation_bounds
    original_options = fixture.planner._easy_allocation_options
    try:
        with experiment.history_allocator(profile(), trajectory(), 3.25):
            assert fixture.planner._easy_allocation_bounds is not original
            if raises:
                raise RuntimeError("intentional")
    except RuntimeError:
        assert raises
    assert fixture.planner._easy_allocation_bounds is original
    assert fixture.planner._easy_allocation_options is original_options


def test_capture_restores_allocator_after_exception():
    original = fixture.planner._allocate_visible_distance_ranges
    with pytest.raises(RuntimeError):
        with experiment.capture_allocations():
            raise RuntimeError("intentional")
    assert fixture.planner._allocate_visible_distance_ranges is original


def test_ordinary_recovery_does_not_resize_intent_band_or_change_weights():
    p = profile()
    observed = []
    with experiment.history_allocator(p, trajectory(), 3.25):
        for residual in (.105, .12, .177, .236):
            state = fixture.synthetic_state(3.7, residual)
            result = fixture.recommendation(state)
            bounds = fixture.planner._easy_allocation_bounds(state, result, fixture.aerobic_maximum(state), 3,
                                                            preserve_projected_recovery_caps=True)
            observed.append(bounds)
    assert len(set(observed)) == 1
    assert observed[0][3] == 1.75


@pytest.mark.parametrize("evidence", ["recovery", "load", "rpe", "weather"])
def test_material_cautions_keep_exact_existing_policy(evidence):
    state = fixture.synthetic_state(3.7, .236)
    if evidence == "recovery":
        state = state.model_copy(update={"recovery_residual_load": .8})
    elif evidence == "load":
        state = state.model_copy(update={"recent_load": state.recent_load.model_copy(update={"acute_distance_to_capacity_ratio": 1.5})})
    elif evidence == "rpe":
        state = state.model_copy(update={"last_run": state.last_run.model_copy(update={"perceived_exertion": 8})})
    else:
        state = state.model_copy(update={"planned_weather": fixture.PlannedWeather(
            forecast_time=state.as_of, apparent_temperature_f=98, temperature_f=95, dewpoint_f=74)})
    result = fixture.recommendation(state)
    original = fixture.planner._easy_allocation_bounds
    args = (state, result, fixture.aerobic_maximum(state), 2)
    expected = original(*args, preserve_projected_recovery_caps=True)
    with experiment.history_allocator(profile(), trajectory(), 3.25) as counts:
        actual = fixture.planner._easy_allocation_bounds(*args, preserve_projected_recovery_caps=True)
    assert actual == expected
    assert counts["material_caution_fallbacks"] == 1


def test_presentation_rounding_does_not_round_away_independent_maximum():
    state = fixture.synthetic_state(3.7, .12)
    result = fixture.recommendation(state)
    t = replace(trajectory(), upper_miles=(3.999,) * 21)
    with experiment.history_allocator(profile(), t, 3.25):
        normal = fixture.planner._easy_allocation_bounds(state, result, 6, 3, preserve_projected_recovery_caps=True)
        restricted = fixture.planner._easy_allocation_bounds(state, result, 3.9, 3, preserve_projected_recovery_caps=True)
    assert normal[1] == 4
    assert restricted[1] <= 3.9


def test_pool_includes_added_opportunities_throughout_horizon():
    seed = (1, 3, 5, 10, 12, 14, 17, 19)
    allowed = set(range(21)) - {2}
    pool = experiment.opportunity_pool(seed, allowed, {5}, max_additions=3)
    assert seed in pool
    assert tuple(sorted((*seed, 0, 7, 16))) in pool
    assert all(set(x).issubset(allowed) and 5 in x and len(x) <= 11 for x in pool)
    assert all(len(x) <= 8 for x in experiment.opportunity_pool(seed, allowed, {5}, max_additions=0))


@pytest.mark.parametrize("capacity", [0, -1, float("nan")])
def test_no_invented_capacity_reference(capacity):
    t = trajectory()
    with pytest.raises(ValueError):
        experiment.dose_trajectory(profile(), t.dates, t.target_ranges, capacity,
                                    mode="history_trajectory", fixed_fraction=.08)


@pytest.mark.parametrize("residual", [0, .10, .12, .236, .55, .8])
@pytest.mark.parametrize("preserve", [False, True])
def test_extracted_default_matches_previous_formula(residual, preserve):
    state = fixture.synthetic_state(3.7, residual)
    result = fixture.recommendation(state)
    low, high = result.distance_range_miles
    easy = fixture.planner.typical_easy_distance(state)
    maximum = max(high, fixture.aerobic_maximum(state))
    minimum = easy[0] + min(.25, (easy[1] - easy[0]) / 2)
    preferred = min(maximum, ceil(sum(easy) / 2 * 2 - 1e-9) / 2)
    shortened = high < easy[1] - 1e-9
    if shortened and fixture.planner._easy_has_material_dose_limit(state, result):
        maximum = (low + high) / 2
        minimum, preferred = min(minimum, maximum), min(preferred, maximum)
    elif preserve:
        pressure = fixture.planner._easy_volume_recovery_pressure(result)
        if pressure > 0 or shortened:
            maximum = fixture.planner._easy_recovery_allocation_ceiling(state, result, fixture.aerobic_maximum(state)) if pressure > 0 else (low + high) / 2
            minimum, preferred = min(minimum, maximum), min(preferred, maximum)
    assert fixture.planner._easy_allocation_bounds(state, result, fixture.aerobic_maximum(state), 3,
                                                  preserve_projected_recovery_caps=preserve) == (minimum, maximum, preferred, 1.75)


def test_replay_rebuilds_reference_and_restores_both_entry_points(monkeypatch):
    from types import SimpleNamespace
    state = fixture.synthetic_state(3.7, .12)
    original_projection = replay.projection.build_weekly_schedule

    def fake_build(daily_states, request, config, target_evidence=None, expected_target_projector=None):
        s = daily_states[0]
        return fixture.planner._easy_allocation_bounds(s, fixture.recommendation(s), 8, 3,
                                                       preserve_projected_recovery_caps=True)

    monkeypatch.setattr(replay.planner, "build_weekly_schedule", fake_build)
    evidence = SimpleNamespace(capacity_reference_miles=16)
    with replay.replay_policy("history_trajectory") as records:
        for reference in (3.7, 4.):
            replay.projection.build_weekly_schedule(
                [state.model_copy(update={"typical_easy_run_miles": reference})], fixture.REQUEST, fixture.CONFIG,
                target_evidence=evidence, expected_target_projector=lambda candidate, n: ((17., 18.),) * n,
            )
    assert [x["ordinary_reference"] for x in records] == [3.7, 4.]
    assert replay.planner.build_weekly_schedule is fake_build
    assert replay.projection.build_weekly_schedule is original_projection


def test_replay_restores_hooks_when_build_fails(monkeypatch):
    original_projection = replay.projection.build_weekly_schedule
    original_bounds = fixture.planner._easy_allocation_bounds

    def failing(daily_states, request, config, target_evidence=None, expected_target_projector=None):
        raise RuntimeError("intentional")

    monkeypatch.setattr(replay.planner, "build_weekly_schedule", failing)
    with pytest.raises(RuntimeError):
        with replay.replay_policy("history_trajectory"):
            replay.projection.build_weekly_schedule([fixture.synthetic_state(3.7, .12)], fixture.REQUEST, fixture.CONFIG)
    assert replay.projection.build_weekly_schedule is original_projection
    assert fixture.planner._easy_allocation_bounds is original_bounds


def test_replay_refuses_existing_database(tmp_path, monkeypatch):
    import sys
    database = tmp_path / "preserved.sqlite"
    database.write_text("existing data")
    monkeypatch.setattr(sys, "argv", ["replay", "--snapshot", str(database), "--database", str(database),
                                     "--project-root", str(tmp_path), "--output-json", str(tmp_path / "new.json")])
    with pytest.raises(SystemExit):
        replay.main()
    assert database.read_text() == "existing data"


def saved_options(saved_range=(5., 5.5), *, residual=.12, maximum=6.125,
                  prior_type=fixture.WorkoutType.EASY, shifted=False):
    state = fixture.synthetic_state(3.7, residual)
    result = fixture.recommendation(state)
    prior = result.model_copy(update={"distance_range_miles": saved_range, "workout_type": prior_type})
    day = state.as_of.date()
    original_options = fixture.planner._easy_allocation_options
    with experiment.history_allocator(profile(), trajectory(), 3.25,
                                       prior_prescriptions={day + timedelta(days=int(shifted)): prior}) as counts:
        bounds = fixture.planner._easy_allocation_bounds(state, result, maximum, 3,
                                                        preserve_projected_recovery_caps=True)
        options, ranges = fixture.planner._easy_allocation_options(state, result, bounds[0], bounds[1], maximum,
                                                                   plan_date=day)
    assert fixture.planner._easy_allocation_options is original_options
    return bounds, options, ranges, counts


def test_saved_dose_is_one_option_not_a_wider_band():
    bounds, options, ranges, counts = saved_options()
    assert bounds[:2] == (3.25, 4.)
    assert options == [3.25, 3.5, 3.75, 4., 5.25]
    assert ranges == {5.25: (5., 5.5)}
    assert counts["prior_candidates_available"] == 1


def test_saved_off_grid_center_and_original_execution_range_are_retained():
    _, options, ranges, _ = saved_options((4.7, 5.2))
    assert 4.95 in options and ranges[4.95] == (4.7, 5.2)
    assert 4.25 not in options and 4.5 not in options and 4.75 not in options


@pytest.mark.parametrize("kwargs,reason", [
    ({"residual": .8}, "material_caution"),
    ({"maximum": 4.5}, "independent_aerobic_maximum"),
    ({"prior_type": fixture.WorkoutType.LONG}, "different_workout_type"),
    ({"prior_type": fixture.WorkoutType.INTERVALS}, "different_workout_type"),
])
def test_saved_dose_has_no_restriction_or_workout_type_exemption(kwargs, reason):
    _, options, ranges, counts = saved_options(**kwargs)
    assert 5.25 not in options and not ranges
    assert counts["prior_rejections"] == {reason: 1}


def test_saved_date_does_not_follow_a_moved_run():
    _, options, ranges, counts = saved_options(shifted=True)
    assert options == [3.25, 3.5, 3.75, 4.]
    assert not ranges and counts["prior_candidates_available"] == 0


def test_saved_small_dose_is_isolated_without_lowering_generated_band():
    bounds, options, ranges, _ = saved_options((2.25, 2.75))
    assert bounds[0] == 3.25
    assert options == [2.5, 3.25, 3.5, 3.75, 4.]
    assert ranges == {2.5: (2.25, 2.75)}


def test_saved_dose_does_not_change_future_intents_or_profile():
    p, t = profile(), trajectory()
    before = (p, t)
    saved_options()
    assert (profile(), trajectory()) == before


def test_prior_mapping_excludes_consumed_forced_old_and_different_workout_dates():
    from types import SimpleNamespace
    first = fixture.ORIGIN.date()
    state = fixture.synthetic_state(3.7, .12)
    result = fixture.recommendation(state)
    days = [SimpleNamespace(date=first + timedelta(days=i), recommendation=result) for i in (-1, 0, 1, 2, 3)]
    days[-1].recommendation = result.model_copy(update={"workout_type": fixture.WorkoutType.LONG})
    prior = SimpleNamespace(planning_days=days, days=[])
    mapped = experiment.eligible_prior_prescriptions(prior, trajectory().dates, excluded_offsets={0, 1})
    assert list(mapped) == [first + timedelta(days=2)]


@pytest.mark.parametrize("bounds", [(3.25, 4.), (3.71, 3.74), (3.95, 4.25)])
def test_default_option_interface_preserves_grid_exactly(bounds):
    state = fixture.synthetic_state(3.7, .12)
    options, ranges = fixture.planner._easy_allocation_options(state, fixture.recommendation(state), *bounds, 6,
                                                              plan_date=state.as_of.date())
    assert options == fixture.planner._distance_options(*bounds)
    assert ranges == {}


def test_saved_range_is_priced_before_selection_and_does_not_force_selection(monkeypatch):
    state = fixture.synthetic_state(3.7, .12)
    result = fixture.recommendation(state)
    saved = (4.7, 5.2)
    prior = result.model_copy(update={"distance_range_miles": saved})
    day = fixture.WeeklyScheduleDay(date=state.as_of.date(), planned_at=state.as_of,
                                    recommendation=result, day_role="easy_run", rationale="fixture")
    priced = []
    original_load = fixture.planner._recommendation_load_units

    def load(recommendation, *args, **kwargs):
        priced.append(recommendation.distance_range_miles)
        return original_load(recommendation, *args, **kwargs)

    monkeypatch.setattr(fixture.planner, "_recommendation_load_units", load)
    with experiment.history_allocator(profile(), trajectory(), 3.25,
                                       prior_prescriptions={day.date: prior}):
        low = fixture.planner._allocate_visible_distance_ranges(
            [day], [state], (3.5, 4.), fixture.CONFIG, preserve_projected_recovery_caps=True,
        )
        high = fixture.planner._allocate_visible_distance_ranges(
            [day], [state], saved, fixture.CONFIG, preserve_projected_recovery_caps=True,
            _prune_dominated_allocations=False, allocation_candidate_limit=20,
            allocation_selector=lambda days: abs(fixture.planner._midpoint(days[0].recommendation) - 4.95),
        )
    assert low[0].recommendation.distance_range_miles != saved
    assert high[0].recommendation.distance_range_miles == saved
    assert saved in priced
