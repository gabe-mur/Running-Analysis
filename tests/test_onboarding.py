from __future__ import annotations

import sqlite3

import pytest

from run_analysis.onboarding import (
    MINIMUM_SUPPORTING_RUNS,
    ZONE_NAMES,
    ZoneMethod,
    derive_zones,
    estimate_max_hr,
    recommend_comparison_hr,
)


def test_tanaka_is_used_rather_than_220_minus_age() -> None:
    """220 - age would give 190 at thirty and 180 at forty; Tanaka is flatter."""
    assert estimate_max_hr(30) == 187
    assert estimate_max_hr(40) == 180
    assert estimate_max_hr(60) == 166


@pytest.mark.parametrize(
    "method,kwargs",
    [
        (ZoneMethod.HEART_RATE_RESERVE, {"maximum_hr": 194, "resting_hr": 49}),
        (ZoneMethod.PERCENT_MAX, {"maximum_hr": 194}),
    ],
)
def test_computed_zones_are_ordered_contiguous_and_reach_max(method, kwargs) -> None:
    zones = derive_zones(method, **kwargs)
    assert list(zones) == list(ZONE_NAMES)
    bounds = [zones[name] for name in ZONE_NAMES]
    for lower, upper in bounds:
        assert lower <= upper
    for (_, previous_upper), (next_lower, _) in zip(bounds, bounds[1:]):
        assert next_lower == previous_upper + 1
    assert bounds[-1][1] == kwargs["maximum_hr"]


def test_reserve_zones_account_for_a_low_resting_pulse() -> None:
    """The whole point of Karvonen: two athletes with the same max HR but
    different resting pulses do not share an easy zone."""
    athletic = derive_zones(ZoneMethod.HEART_RATE_RESERVE, maximum_hr=190, resting_hr=45)
    sedentary = derive_zones(ZoneMethod.HEART_RATE_RESERVE, maximum_hr=190, resting_hr=70)
    assert athletic["z2"][0] < sedentary["z2"][0]


def test_percent_max_and_reserve_disagree_so_the_method_must_be_recorded() -> None:
    reserve = derive_zones(ZoneMethod.HEART_RATE_RESERVE, maximum_hr=194, resting_hr=49)
    percent = derive_zones(ZoneMethod.PERCENT_MAX, maximum_hr=194)
    assert reserve["z2"] != percent["z2"]


def test_reserve_zones_require_a_resting_heart_rate() -> None:
    with pytest.raises(ValueError):
        derive_zones(ZoneMethod.HEART_RATE_RESERVE, maximum_hr=194)


def test_device_zones_are_passed_through_untouched() -> None:
    supplied = {"z1": (128, 140), "z2": (141, 153), "z3": (154, 166), "z4": (167, 180), "z5": (181, 194)}
    assert derive_zones(ZoneMethod.DEVICE, maximum_hr=194, boundaries=supplied) == supplied


def test_device_zones_must_be_complete() -> None:
    with pytest.raises(ValueError):
        derive_zones(ZoneMethod.DEVICE, maximum_hr=194, boundaries={"z1": (128, 140)})


def _segments(connection: sqlite3.Connection, rows: list[tuple[int, int]]) -> None:
    connection.execute(
        "CREATE TABLE segments (activity_id INTEGER, average_hr_bpm REAL, is_pathological INTEGER DEFAULT 0)"
    )
    connection.executemany(
        "INSERT INTO segments(activity_id, average_hr_bpm) VALUES (?, ?)", rows
    )


ZONES = {"z1": (128, 140), "z2": (141, 153), "z3": (154, 166), "z4": (167, 180), "z5": (181, 194)}


def test_the_comparison_heart_rate_follows_the_evidence_not_the_midpoint() -> None:
    connection = sqlite3.connect(":memory:")
    # 150 bpm is reached on many separate days; 145 on only a few.
    rows = [(run, 150) for run in range(30)] + [(run, 145) for run in range(3)]
    _segments(connection, rows)
    result = recommend_comparison_hr(connection, ZONES)
    assert result.recommended_bpm == 150
    assert result.run_count == 30
    assert "150 bpm" in result.rationale


def test_breadth_across_runs_beats_depth_within_one_run() -> None:
    """Forty segments from three long runs describe those runs, not the athlete."""
    connection = sqlite3.connect(":memory:")
    rows = [(1, 147) for _ in range(200)] + [(run, 149) for run in range(20)]
    _segments(connection, rows)
    assert recommend_comparison_hr(connection, ZONES).recommended_bpm == 149


def test_a_thin_history_falls_back_to_the_middle_of_z2_and_says_so() -> None:
    connection = sqlite3.connect(":memory:")
    _segments(connection, [(run, 150) for run in range(MINIMUM_SUPPORTING_RUNS - 1)])
    result = recommend_comparison_hr(connection, ZONES)
    assert result.recommended_bpm == 147
    assert "placeholder" in result.rationale


def test_no_easy_running_at_all_recommends_nothing(monkeypatch) -> None:
    connection = sqlite3.connect(":memory:")
    _segments(connection, [(1, 170)])
    result = recommend_comparison_hr(connection, ZONES)
    assert result.recommended_bpm is None
    assert "no recorded segments" in result.rationale.lower()


def test_only_heart_rates_inside_z2_are_considered() -> None:
    """A comparison point outside the easy zone would anchor the whole app to
    an effort the athlete does not spend most of their time at."""
    connection = sqlite3.connect(":memory:")
    rows = [(run, 160) for run in range(50)] + [(run, 149) for run in range(10)]
    _segments(connection, rows)
    result = recommend_comparison_hr(connection, ZONES)
    assert result.recommended_bpm == 149
    assert all(ZONES["z2"][0] <= c.heart_rate_bpm <= ZONES["z2"][1] for c in result.candidates)


def test_pathological_segments_do_not_vote() -> None:
    connection = sqlite3.connect(":memory:")
    _segments(connection, [(run, 150) for run in range(20)])
    connection.execute("UPDATE segments SET is_pathological = 1")
    connection.executemany(
        "INSERT INTO segments(activity_id, average_hr_bpm) VALUES (?, ?)",
        [(run, 146) for run in range(12)],
    )
    assert recommend_comparison_hr(connection, ZONES).recommended_bpm == 146


def _setup_config(**overrides) -> dict:
    config = {
        "max_hr": 194,
        "resting_hr": 49,
        "target_hr": 145,
        "zones": {"z1": [128, 140], "z2": [141, 153], "z3": [154, 166], "z4": [167, 180], "z5": [181, 194]},
        "profile": {"birth_date": "1995-11-28", "sex": "male", "weight_lb": 140.0},
        "coaching": {"training_goal": "general_fitness"},
        "weather": {"historical_enabled": False, "forecast_enabled": False, "privacy_jitter_radius_km": 2.0},
    }
    config.update(overrides)
    return config


def _activities(connection: sqlite3.Connection, count: int) -> None:
    connection.execute("CREATE TABLE activities (id INTEGER PRIMARY KEY, sport TEXT)")
    connection.executemany(
        "INSERT INTO activities(sport) VALUES ('Running')", [() for _ in range(count)]
    )


def test_a_setting_that_merely_has_a_value_is_not_confirmed() -> None:
    """Everything here ships with a plausible default. Treating a filled field
    as an answer would mark setup complete before a single question was asked."""
    from run_analysis.onboarding import SetupStep, setup_state

    connection = sqlite3.connect(":memory:")
    _activities(connection, 12)
    state = setup_state(connection, _setup_config())
    assert not state.complete
    incomplete = {step.step for step in state.steps if not step.complete}
    assert SetupStep.HEART_RATE in incomplete
    assert SetupStep.ZONES in incomplete


def test_confirming_every_step_completes_setup() -> None:
    from run_analysis.onboarding import SetupStep, setup_state

    connection = sqlite3.connect(":memory:")
    _activities(connection, 12)
    config = _setup_config(
        setup={"confirmed_steps": [s.value for s in SetupStep], "zone_method": "device"}
    )
    assert setup_state(connection, config).complete


def test_no_runs_is_reported_as_blocking() -> None:
    """Nothing downstream can be judged without data, so this step is not
    merely incomplete -- it is the one that stops everything else."""
    from run_analysis.onboarding import SetupStep, setup_state

    connection = sqlite3.connect(":memory:")
    _activities(connection, 0)
    state = setup_state(connection, _setup_config())
    runs = next(step for step in state.steps if step.step == SetupStep.RUNS)
    assert not runs.complete and runs.blocking
    assert state.next_step.step == SetupStep.RUNS


def test_a_comparison_heart_rate_outside_z2_is_called_out() -> None:
    from run_analysis.onboarding import SetupStep, setup_state

    connection = sqlite3.connect(":memory:")
    _activities(connection, 12)
    state = setup_state(connection, _setup_config(target_hr=170))
    step = next(item for item in state.steps if item.step == SetupStep.COMPARISON_HR)
    assert "outside your Z2" in step.detail


def test_an_estimated_max_says_it_is_the_largest_error_source() -> None:
    from run_analysis.onboarding import SetupStep, setup_state

    connection = sqlite3.connect(":memory:")
    _activities(connection, 12)
    measured = _setup_config()
    measured["profile"] = {**measured["profile"], "max_hr_source": "measured"}
    estimated = setup_state(connection, _setup_config())
    assert "largest single source of error" in next(
        s for s in estimated.steps if s.step == SetupStep.HEART_RATE
    ).detail
    assert "largest single source of error" not in next(
        s for s in setup_state(connection, measured).steps if s.step == SetupStep.HEART_RATE
    ).detail
