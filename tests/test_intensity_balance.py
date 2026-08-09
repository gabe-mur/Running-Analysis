from __future__ import annotations

from run_analysis.intensity_balance import IntensityBalance, assess_intensity_balance


def _assess(easy: float, moderate: float, hard: float, *, known: float = 900.0, missing: float = 0.0):
    return assess_intensity_balance(
        easy_percent=easy,
        moderate_percent=moderate,
        hard_percent=hard,
        known_minutes=known,
        missing_minutes=missing,
    )


def test_grey_zone_outranks_a_respectable_easy_share() -> None:
    """77% easy looks fine until you notice where the other 23% went."""
    verdict = _assess(77.0, 21.0, 2.0)
    assert verdict.balance == IntensityBalance.GREY_ZONE
    assert "21%" in verdict.detail
    assert "conversational" in verdict.detail


def test_a_polarized_split_is_approved() -> None:
    verdict = _assess(82.0, 8.0, 10.0)
    assert verdict.balance == IntensityBalance.BALANCED


def test_running_everything_too_hard_reads_as_too_hard_not_grey_zone() -> None:
    verdict = _assess(60.0, 12.0, 28.0)
    assert verdict.balance == IntensityBalance.TOO_HARD
    assert "60%" in verdict.detail


def test_an_all_easy_block_is_described_not_scolded() -> None:
    """A base or recovery block is a legitimate choice, so it is not a fault."""
    verdict = _assess(95.0, 4.0, 1.0)
    assert verdict.balance == IntensityBalance.NEARLY_ALL_EASY
    assert "recovery or base block" in verdict.detail


def test_a_correct_easy_share_with_no_quality_still_names_the_gap() -> None:
    verdict = _assess(88.0, 10.0, 2.0)
    assert verdict.balance == IntensityBalance.NO_HARD_STIMULUS
    assert "quality session" in verdict.detail


def test_too_little_data_makes_no_claim() -> None:
    assert _assess(80.0, 10.0, 10.0, known=60.0).balance == IntensityBalance.INSUFFICIENT_DATA


def test_poor_heart_rate_coverage_makes_no_claim() -> None:
    """A split over a third of the block is a sample, not a distribution."""
    verdict = _assess(80.0, 10.0, 10.0, known=900.0, missing=600.0)
    assert verdict.balance == IntensityBalance.INSUFFICIENT_DATA
    assert "40%" in verdict.detail


def test_every_verdict_has_a_headline_and_an_actionable_sentence() -> None:
    for split in ((77, 21, 2), (82, 8, 10), (60, 12, 28), (95, 4, 1), (88, 10, 2)):
        verdict = _assess(*split)
        assert verdict.headline and not verdict.headline.endswith(".")
        assert verdict.detail.endswith(".") and len(verdict.detail) > 60
