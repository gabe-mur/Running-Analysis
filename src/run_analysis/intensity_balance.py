"""Turn an easy/moderate/hard time split into a verdict an athlete can act on.

The raw split is three percentages, and three percentages are not advice. What
an athlete wants to know is whether the distribution is the shape that produces
adaptation, and if not, which way it is wrong -- because the two failure modes
have opposite fixes. Accumulating grey-zone time means slowing the easy runs
down; running everything easy means adding a hard session. Reporting only the
numbers leaves that inference to the reader.

Thresholds follow the polarized/pyramidal consensus (Seiler; Esteve-Lanao et
al.): roughly 80% of time genuinely easy, most of the remainder genuinely hard,
and little in between. They are conventions, not physiology, so they are named
here rather than buried in comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

#: The conventional easy share. Below this, easy running is being run too hard.
EASY_TARGET_PERCENT = 80.0

#: Enough below the target to be a pattern rather than a rounding artefact.
EASY_TOLERANCE_PERCENT = 5.0

#: Moderate time above this is the dominant problem, whatever else is true.
#: This is the "grey zone": above conversational, below a threshold stimulus.
GREY_ZONE_PERCENT = 15.0

#: Below this share of hard time there is little driving adaptation upward.
MINIMUM_HARD_PERCENT = 4.0

#: An easy share this high, with no hard work, is maintenance rather than
#: training. It is not a fault -- it is the right shape for a recovery block --
#: so it is reported neutrally.
NEARLY_ALL_EASY_PERCENT = 92.0

#: Too little recorded time to characterise a distribution at all.
MINIMUM_MINUTES = 120.0

#: Above this share of unrecorded time, the split describes the runs that
#: happened to have a heart-rate signal rather than the training block.
MAXIMUM_MISSING_FRACTION = 0.25


class IntensityBalance(StrEnum):
    """Which shape the effort distribution is, not how much of each zone."""

    BALANCED = "balanced"
    GREY_ZONE = "grey_zone"
    TOO_HARD = "too_hard"
    NO_HARD_STIMULUS = "no_hard_stimulus"
    NEARLY_ALL_EASY = "nearly_all_easy"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass(frozen=True, slots=True)
class BalanceVerdict:
    balance: IntensityBalance
    headline: str
    detail: str


def _percent(value: float | None) -> float:
    return float(value) if value is not None else 0.0


def assess_intensity_balance(
    *,
    easy_percent: float | None,
    moderate_percent: float | None,
    hard_percent: float | None,
    known_minutes: float,
    missing_minutes: float,
) -> BalanceVerdict:
    """Describe the shape of an effort distribution in one sentence.

    Rules are evaluated in a fixed order and the first match wins, so the
    verdict names the largest problem rather than every deviation at once.
    """

    total = known_minutes + missing_minutes
    if known_minutes < MINIMUM_MINUTES:
        return BalanceVerdict(
            IntensityBalance.INSUFFICIENT_DATA,
            "Not enough to judge",
            "There is under two hours of heart-rate data in this period, which is too "
            "little to say anything about how your effort is distributed.",
        )
    if total and missing_minutes / total > MAXIMUM_MISSING_FRACTION:
        share = round(missing_minutes / total * 100)
        return BalanceVerdict(
            IntensityBalance.INSUFFICIENT_DATA,
            "Not enough to judge",
            f"{share}% of your running time has no heart-rate data, so this split "
            "describes the runs that happened to record it rather than your training.",
        )

    easy = _percent(easy_percent)
    moderate = _percent(moderate_percent)
    hard = _percent(hard_percent)

    # Grey-zone accumulation outranks everything else: it is the failure mode
    # that feels productive while producing neither recovery nor adaptation.
    if moderate >= GREY_ZONE_PERCENT:
        return BalanceVerdict(
            IntensityBalance.GREY_ZONE,
            "Too much grey zone",
            f"{moderate:.0f}% of your time is moderate -- too hard to recover from, too "
            "easy to drive adaptation. Slowing your easy runs until they are genuinely "
            "conversational is the single biggest change available here.",
        )
    if easy < EASY_TARGET_PERCENT - EASY_TOLERANCE_PERCENT:
        return BalanceVerdict(
            IntensityBalance.TOO_HARD,
            "Running too hard overall",
            f"Only {easy:.0f}% of your time is easy, against the {EASY_TARGET_PERCENT:.0f}% "
            "that usually supports consistent training. Most of that gap should come back "
            "as slower easy running, not fewer runs.",
        )
    if easy >= NEARLY_ALL_EASY_PERCENT and hard < MINIMUM_HARD_PERCENT:
        return BalanceVerdict(
            IntensityBalance.NEARLY_ALL_EASY,
            "Almost entirely easy",
            f"{easy:.0f}% of your time is easy and almost none is hard. That is the right "
            "shape for a recovery or base block, and the wrong one if you are trying to "
            "get faster.",
        )
    if hard < MINIMUM_HARD_PERCENT:
        return BalanceVerdict(
            IntensityBalance.NO_HARD_STIMULUS,
            "Easy base, little stimulus",
            f"Your easy share is where it should be, but only {hard:.0f}% of your time is "
            "hard. One genuine quality session a week is what turns this base into speed.",
        )
    return BalanceVerdict(
        IntensityBalance.BALANCED,
        "Well distributed",
        f"{easy:.0f}% easy with {hard:.0f}% hard and little in between -- the polarized "
        "shape that lets hard sessions stay hard and easy days stay restorative.",
    )
