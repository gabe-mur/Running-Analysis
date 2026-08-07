from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from run_analysis.workload import WorkloadInput, compute_workloads


def test_workloads_use_prior_activities_only() -> None:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    records = [
        WorkloadInput(1, start, 3, 30, False),
        WorkloadInput(2, start + timedelta(days=3), 4, 40, True),
        WorkloadInput(3, start + timedelta(days=8), 5, 50, False),
    ]
    features = compute_workloads(records)
    assert features[0].previous_7d_miles == 0
    assert features[1].previous_7d_miles == 3
    # At day 8, day 0 is outside the inclusive prior seven-day window. The
    # current activity itself and all future activities are excluded.
    assert features[2].previous_7d_miles == 4
    assert features[2].previous_28d_miles == 7
    assert features[2].days_since_previous_run == pytest.approx(5)
    assert features[2].days_since_previous_hard_run == pytest.approx(5)
