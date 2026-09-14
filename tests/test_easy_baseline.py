from datetime import datetime, timedelta, timezone

from run_analysis.easy_baseline import (
    ordinary_easy_sample_distance,
    recency_weighted_easy_distance,
)


def test_deliberately_short_and_quality_work_do_not_redefine_easy_baseline() -> None:
    assert ordinary_easy_sample_distance(2.0, "support_easy", (2.0, 2.5)) is None
    assert ordinary_easy_sample_distance(3.0, "quality", (3.0, 3.5)) is None


def test_medium_long_completion_preserves_aerobic_session_scale() -> None:
    assert ordinary_easy_sample_distance(4.4, "medium_long", (4.0, 4.5)) == 4.4


def test_recent_medium_long_completions_prevent_a_circular_baseline_collapse() -> None:
    as_of = datetime(2026, 9, 13, tzinfo=timezone.utc)
    samples = [
        (as_of - timedelta(days=25), 2.0),
        (
            as_of - timedelta(days=10),
            ordinary_easy_sample_distance(4.75, "medium_long", (4.5, 5.0)),
        ),
        (
            as_of - timedelta(days=2),
            ordinary_easy_sample_distance(4.39, "medium_long", (4.0, 4.5)),
        ),
    ]

    assert recency_weighted_easy_distance(
        [(occurred_at, distance) for occurred_at, distance in samples if distance],
        as_of,
        half_life_days=28,
    ) == 4.39


def test_matched_ordinary_run_clamps_execution_noise_to_prescription() -> None:
    assert ordinary_easy_sample_distance(2.5, "ordinary_easy", (3.5, 4.0)) == 3.5
    assert ordinary_easy_sample_distance(5.0, "ordinary_easy", (3.5, 4.0)) == 4.0
    assert ordinary_easy_sample_distance(3.8, "ordinary_easy", (3.5, 4.0)) == 3.8


def test_unmatched_easy_run_remains_direct_baseline_evidence() -> None:
    assert ordinary_easy_sample_distance(4.6, None, None) == 4.6
