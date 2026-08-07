from __future__ import annotations

import pytest


def test_reference_position_policy_is_declared_in_configuration() -> None:
    # This guards the product-level comparison policy: duration can affect load
    # without giving an eight-mile run more late-fatigue fitness windows than a
    # two-mile run.
    from pathlib import Path
    import yaml

    config = yaml.safe_load((Path(__file__).parents[1] / "config.example.yaml").read_text())
    assert config["reference_conditions"]["within_run_minutes"] == 20
    assert config["model"]["run_effect_max_effective_segments"] == 4


def test_long_run_late_windows_do_not_receive_extra_fitness_weight() -> None:
    from run_analysis.objective_modeling import select_comparable_run_windows

    rows = [{"moving_minutes_into_run": minute} for minute in (5, 10, 15, 20, 25, 30, 35, 40, 45)]
    selected = select_comparable_run_windows(rows, list(range(len(rows))), 20, 4)
    assert [rows[index]["moving_minutes_into_run"] for index in selected] == [20, 15, 25, 10]
    assert all(rows[index]["moving_minutes_into_run"] <= 27.5 for index in selected)


def _steady_config() -> dict:
    return {
        "target_hr": 145,
        "reference_conditions": {"within_run_minutes": 20},
        "model": {
            "primary_hr_weight_scale_bpm": 10,
            "primary_time_weight_scale_minutes": 15,
            "primary_hr_change_weight_scale_bpm": 8,
            "primary_speed_change_weight_scale_fraction": 0.20,
            "steady_benchmark_reference_minutes": 20,
            "steady_benchmark_maximum_reference_offset_minutes": 2.5,
            "steady_benchmark_minimum_minutes": 10,
            "steady_benchmark_maximum_minutes": 35,
            "steady_benchmark_minimum_hr_bpm": 140,
            "steady_benchmark_maximum_hr_bpm": 150,
            "steady_benchmark_maximum_hr_change_bpm": 6,
            "steady_benchmark_maximum_hr_range_bpm": 15,
            "steady_benchmark_maximum_speed_change_fraction": 0.15,
            "steady_benchmark_maximum_stopped_seconds": 5,
        },
    }


def _steady_row(minute: float, *, hr: float = 145, hr_change: float = 1, speed_change: float = 0.02) -> dict:
    return {
        "moving_minutes_into_run": minute,
        "average_hr_bpm": hr,
        "heart_rate_change_bpm": hr_change,
        "heart_rate_range_bpm": 8,
        "speed_change_fraction": speed_change,
        "stopped_time_s": 0,
    }


def test_steady_benchmark_rejects_transition_and_uses_fixed_time() -> None:
    from run_analysis.objective_modeling import select_steady_aerobic_window

    rows = [
        _steady_row(19.5, speed_change=0.30),
        _steady_row(21.0, hr=144),
        _steady_row(17.0),
    ]
    assert select_steady_aerobic_window(rows, [0, 1, 2], _steady_config()) == [1]


def test_steady_benchmark_does_not_substitute_distant_window() -> None:
    from run_analysis.objective_modeling import select_steady_aerobic_window

    rows = [_steady_row(17.0), _steady_row(23.0)]
    assert select_steady_aerobic_window(rows, [0, 1], _steady_config()) == []


def test_fixed_time_fallback_retains_nearest_reliable_window_as_estimate() -> None:
    from run_analysis.objective_modeling import select_fixed_time_benchmark_fallback

    rows = [
        _steady_row(17.0, hr=137, speed_change=0.18),
        _steady_row(20.5, hr=153, speed_change=0.18),
        _steady_row(24.0, hr=145, speed_change=0.02),
    ]
    assert select_fixed_time_benchmark_fallback(rows, [0, 1, 2], _steady_config()) == [1]


def test_shared_hr_time_model_preserves_run_specific_offsets() -> None:
    from run_analysis.objective_modeling import _within_run_hr_time_slopes

    rows = []
    speeds = []
    for activity_id, offset in ((1, 2.3), (2, 2.6), (3, 2.9), (4, 3.2)):
        for minute, hr in ((10, 138), (15, 142), (20, 146), (25, 150), (30, 148)):
            rows.append(
                {
                    "activity_id": activity_id,
                    "moving_minutes_into_run": minute,
                    "average_hr_bpm": hr,
                }
            )
            speeds.append(offset + 0.004 * hr - 0.006 * minute)
    hr_slope, time_slope, diagnostics = _within_run_hr_time_slopes(rows, speeds)
    assert hr_slope == pytest.approx(0.004)
    assert time_slope == pytest.approx(-0.006)
    assert diagnostics["contributing_runs"] == 4


def test_reference_time_support_prefers_interpolation_and_rejects_heroic_extrapolation() -> None:
    from run_analysis.objective_modeling import _reference_time_support

    config = _steady_config()
    config["model"].update(
        {
            "primary_window_seconds": 120,
            "maximum_reference_extrapolation_minutes": 5,
            "reference_extrapolation_uncertainty_scale_minutes": 1,
        }
    )
    supported = _reference_time_support([10, 15, 20, 25], 20, config)
    limited = _reference_time_support([8, 13, 16, 18], 20, config)
    rejected = _reference_time_support([5, 8, 11, 13], 20, config)
    assert supported == (True, 1.0, 0.0, "interpolation")
    assert limited[0] is True and limited[1] > 1
    assert rejected[0] is False
