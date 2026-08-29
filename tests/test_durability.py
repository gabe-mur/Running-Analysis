from datetime import datetime, timedelta, timezone

import pytest

from run_analysis.durability import retained_long_run_capacity


def test_long_run_capacity_has_no_thirty_day_cliff() -> None:
    as_of = datetime(2026, 8, 29, tzinfo=timezone.utc)
    run = (as_of - timedelta(days=31), 8.0)

    assert retained_long_run_capacity([run], as_of) == pytest.approx(8.0)


def test_old_peak_fades_smoothly_but_a_new_completion_refreshes_capacity() -> None:
    as_of = datetime(2026, 8, 29, tzinfo=timezone.utc)
    old_peak = (as_of - timedelta(days=162), 8.0)
    maintained = (as_of - timedelta(days=7), 7.0)

    old_only = retained_long_run_capacity([old_peak], as_of)
    refreshed = retained_long_run_capacity([old_peak, maintained], as_of)

    assert old_only == pytest.approx(6.062866266)
    assert refreshed == pytest.approx(7.0)
