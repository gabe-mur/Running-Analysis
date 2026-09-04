from __future__ import annotations

from run_analysis.processing import _eligibility


def test_prescription_only_workout_type_is_a_valid_partial_override() -> None:
    config = {
        "model": {
            "minimum_run_minutes": 10,
            "minimum_gps_coverage": 0.8,
            "minimum_hr_coverage": 0.8,
            "maximum_stop_fraction": 0.35,
        },
        "activity_classification": {
            "high_confidence_walk_pace_min_mile": 18,
            "high_confidence_walk_cadence_max_spm": 110,
            "high_confidence_bike_pace_min_mile": 5.5,
        },
    }

    eligible, reasons = _eligibility(
        {"total_distance_m": 0},
        [],
        [],
        {"moving_time_s": 0, "stopped_time_s": 0, "analysis_distance_m": 0},
        config,
        {"workout_type": "easy"},
    )

    assert eligible is False
    assert "manual_exclusion" not in reasons
    assert "workout_type_easy" not in reasons
