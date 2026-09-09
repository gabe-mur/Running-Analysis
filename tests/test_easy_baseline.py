from run_analysis.easy_baseline import ordinary_easy_sample_distance


def test_only_ordinary_easy_work_can_redefine_the_easy_baseline() -> None:
    assert ordinary_easy_sample_distance(2.0, "support_easy", (2.0, 2.5)) is None
    assert ordinary_easy_sample_distance(5.5, "medium_long", (5.0, 6.0)) is None
    assert ordinary_easy_sample_distance(3.0, "quality", (3.0, 3.5)) is None


def test_matched_ordinary_run_clamps_execution_noise_to_prescription() -> None:
    assert ordinary_easy_sample_distance(2.5, "ordinary_easy", (3.5, 4.0)) == 3.5
    assert ordinary_easy_sample_distance(5.0, "ordinary_easy", (3.5, 4.0)) == 4.0
    assert ordinary_easy_sample_distance(3.8, "ordinary_easy", (3.5, 4.0)) == 3.8


def test_unmatched_easy_run_remains_direct_baseline_evidence() -> None:
    assert ordinary_easy_sample_distance(4.6, None, None) == 4.6
