from datetime import datetime, timedelta, timezone

import pytest

from run_analysis.durability import (
    retained_long_run_capacity,
    supported_long_run_capacity,
)


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


def test_single_overdistance_run_does_not_ratchet_safe_long_anchor() -> None:
    as_of = datetime(2026, 9, 2, tzinfo=timezone.utc)
    runs = [
        (as_of - timedelta(days=14), 7.0, 7.0),
        (as_of - timedelta(days=1), 10.0, 7.5),
    ]

    assert supported_long_run_capacity(runs, as_of) == pytest.approx(7.5)


def test_repeated_overdistance_runs_can_establish_long_capacity() -> None:
    as_of = datetime(2026, 9, 2, tzinfo=timezone.utc)
    runs = [
        (as_of - timedelta(days=14), 9.0, 7.0),
        (as_of - timedelta(days=1), 10.0, 7.5),
    ]

    assert supported_long_run_capacity(runs, as_of) == pytest.approx(9.0)


def test_unprescribed_history_remains_full_long_capacity_evidence() -> None:
    as_of = datetime(2026, 9, 2, tzinfo=timezone.utc)

    assert supported_long_run_capacity(
        [(as_of - timedelta(days=1), 8.4, None)], as_of
    ) == pytest.approx(8.4)
