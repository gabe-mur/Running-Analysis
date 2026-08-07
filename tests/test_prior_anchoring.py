from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import math

import pytest
import yaml

from run_analysis.objective_modeling import _estimate_heat_posterior


def config() -> dict:
    return yaml.safe_load((Path(__file__).parents[1] / "config.example.yaml").read_text())


def test_personal_matched_evidence_updates_but_does_not_replace_heat_prior() -> None:
    settings = config()
    start = datetime(2026, 5, 1, tzinfo=timezone.utc)
    personal_effect = 0.004
    rows = []
    for index in range(12):
        exposure = 0.0 if index % 2 == 0 else 10.0
        rows.append(
            {
                "activity_id": index + 1,
                "start_time_utc": (start + timedelta(days=index * 3)).isoformat(),
                "heat_exposure_c": exposure,
                "pace_before_heat": 10.0 * math.exp(personal_effect * exposure),
            }
        )
    posterior = _estimate_heat_posterior(rows, settings)
    assert posterior["personal_likelihood_mean_fraction_per_c"] == pytest.approx(
        personal_effect
    )
    assert 0 < posterior["personal_data_weight"] < 1
    assert posterior["prior_mean_fraction_per_c"] < posterior["posterior_mean_fraction_per_c"]
    assert posterior["posterior_mean_fraction_per_c"] < personal_effect


def test_no_matched_evidence_leaves_prior_unchanged() -> None:
    settings = config()
    rows = [
        {
            "activity_id": 1,
            "start_time_utc": "2026-05-01T00:00:00+00:00",
            "heat_exposure_c": 1.0,
            "pace_before_heat": 10.0,
        },
        {
            "activity_id": 2,
            "start_time_utc": "2026-12-01T00:00:00+00:00",
            "heat_exposure_c": 10.0,
            "pace_before_heat": 10.5,
        },
    ]
    posterior = _estimate_heat_posterior(rows, settings)
    assert posterior["matched_pair_count"] == 0
    assert posterior["personal_data_weight"] == 0
    assert posterior["posterior_mean_fraction_per_c"] == posterior[
        "prior_mean_fraction_per_c"
    ]
    assert posterior["confidence"] == "low"
