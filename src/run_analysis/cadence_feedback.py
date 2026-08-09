"""Cadence feedback that is personal and contextual, never an internet ideal.

"Get your cadence to 180" is advice, not analysis. Optimal cadence varies with
height, leg length, speed, terrain, and training history, and a runner told to
chase a population number will usually just shorten their stride at the same
speed. Nothing in this module compares the athlete to anyone else.

Two things make the feedback useful instead:

**Speed decomposes exactly.** Running speed is the product of turnover and
stride length::

    speed = cadence x stride length

so taking logarithms splits any speed change into two additive shares::

    ln(speed_2 / speed_1) = ln(cadence_2 / cadence_1) + ln(stride_2 / stride_1)

That is an identity, not a model, which is what lets the app say a pace change
came "mostly through longer stride rather than higher turnover" and mean it
arithmetically. Stride length here is a proxy — speed divided by cadence —
because no ground-contact sensor is present; it is a true average stride only
insofar as the cadence reading is accurate.

**The reference is the athlete's own history at the same speed.** Cadence rises
with speed, so comparing a slow long run against a fast tempo says nothing. The
comparison band is built from this athlete's past quarter-mile segments run at
a similar pace, and reports a median and spread. "Unusual" therefore means
unusual *for this runner at this pace*.

All cadence values are total steps per minute via the single conversion in
:mod:`run_analysis.cadence`.
"""

from __future__ import annotations

from math import log
from statistics import median
import sqlite3

from .movement import MovementInterval
from .segmentation import METERS_PER_MILE
from .web.schemas import CadenceAnalysis, CadenceComparison

#: Half-to-half cadence differences smaller than this are sampling noise.
MEANINGFUL_CADENCE_CHANGE_SPM = 2.0

#: Pace changes smaller than this are not worth attributing.
MEANINGFUL_PACE_CHANGE_SECONDS = 5.0

#: A share of the speed change at or above this dominates the explanation.
DOMINANT_SHARE = 0.65

#: Historical segments within this fractional pace distance form the comparison
#: band. Wide enough to find neighbours in a modest history, tight enough that
#: the cadence/speed relationship is roughly flat across it.
COMPARISON_PACE_TOLERANCE = 0.08

#: Below this many comparable segments the personal band is not trustworthy.
MINIMUM_COMPARISON_SEGMENTS = 12

#: Cadence must cover at least this fraction of moving time to be reported.
MINIMUM_COVERAGE_FRACTION = 0.5

#: Multiples of the personal spread that count as unusual.
UNUSUAL_DEVIATIONS = 2.0


def _weighted_stats(pairs: list[tuple[float, float]]) -> tuple[float | None, float]:
    """Time-weighted mean and total weight for ``(value, seconds)`` pairs."""
    total = sum(weight for _value, weight in pairs)
    if total <= 0:
        return None, 0.0
    return sum(value * weight for value, weight in pairs) / total, total


def _interval_cadence(interval: MovementInterval) -> float | None:
    values = [
        value
        for value in (interval.start.cadence_spm, interval.end.cadence_spm)
        if value is not None
    ]
    return sum(values) / len(values) if values else None


def _stride_length_m(speed_mps: float | None, cadence_spm: float | None) -> float | None:
    """Metres per step. A proxy: no ground-contact sensor is available."""
    if not speed_mps or not cadence_spm:
        return None
    return speed_mps / (cadence_spm / 60.0)


def _half_summary(intervals: list[MovementInterval]) -> dict[str, float | None]:
    cadence_pairs: list[tuple[float, float]] = []
    distance = moving = 0.0
    for interval in intervals:
        if interval.moving_time_s <= 0:
            continue
        distance += interval.distance_m
        moving += interval.moving_time_s
        cadence = _interval_cadence(interval)
        if cadence:
            cadence_pairs.append((cadence, interval.moving_time_s))
    cadence, covered = _weighted_stats(cadence_pairs)
    speed = distance / moving if moving > 0 else None
    return {
        "cadence_spm": cadence,
        "speed_mps": speed,
        "pace_min_mile": (moving / 60.0) / (distance / METERS_PER_MILE) if distance > 0 else None,
        "stride_length_m": _stride_length_m(speed, cadence),
        "moving_seconds": moving,
        "cadence_seconds": covered,
    }


def _split_halves(intervals: list[MovementInterval]) -> tuple[list[MovementInterval], list[MovementInterval]]:
    """Split at the moving-time midpoint, not the interval-count midpoint."""
    usable = [item for item in intervals if item.moving_time_s > 0]
    total = sum(item.moving_time_s for item in usable)
    first: list[MovementInterval] = []
    second: list[MovementInterval] = []
    elapsed = 0.0
    for interval in usable:
        midpoint = elapsed + interval.moving_time_s / 2.0
        (first if midpoint <= total / 2.0 else second).append(interval)
        elapsed += interval.moving_time_s
    return first, second


def personal_cadence_band(
    connection: sqlite3.Connection, activity_id: int, pace_min_mile: float
) -> CadenceComparison:
    """This athlete's own cadence at a similar pace, excluding this run."""

    low = pace_min_mile * (1 - COMPARISON_PACE_TOLERANCE)
    high = pace_min_mile * (1 + COMPARISON_PACE_TOLERANCE)
    rows = connection.execute(
        """
        SELECT s.average_cadence_spm AS cadence
        FROM segments s
        WHERE s.activity_id != ?
          AND s.is_pathological = 0
          AND s.average_cadence_spm IS NOT NULL
          AND s.moving_pace_min_mile BETWEEN ? AND ?
        """,
        (activity_id, low, high),
    ).fetchall()
    values = [float(row["cadence"]) for row in rows]
    if len(values) < MINIMUM_COMPARISON_SEGMENTS:
        return CadenceComparison(
            available=False,
            comparable_segment_count=len(values),
            pace_band_low_min_mile=low,
            pace_band_high_min_mile=high,
            detail=(
                f"Only {len(values)} past quarter-mile segments were run near this pace, "
                f"so there is no personal cadence range to compare against yet."
            ),
        )
    centre = median(values)
    spread = median(abs(value - centre) for value in values) * 1.4826
    return CadenceComparison(
        available=True,
        personal_median_spm=centre,
        personal_spread_spm=spread,
        comparable_segment_count=len(values),
        pace_band_low_min_mile=low,
        pace_band_high_min_mile=high,
        detail=(
            f"Your usual cadence near this pace is {centre:.0f} spm "
            f"(typical range {centre - spread:.0f}-{centre + spread:.0f}), "
            f"from {len(values)} past segments."
        ),
    )


def _observations(
    overall: dict, first: dict, second: dict, comparison: CadenceComparison
) -> list[str]:
    """Plain-language findings, each tied to an arithmetic fact."""

    notes: list[str] = []
    cadence_1, cadence_2 = first["cadence_spm"], second["cadence_spm"]
    speed_1, speed_2 = first["speed_mps"], second["speed_mps"]

    if cadence_1 and cadence_2 and speed_1 and speed_2:
        cadence_change = cadence_2 - cadence_1
        pace_change_seconds = (second["pace_min_mile"] - first["pace_min_mile"]) * 60.0
        # Exact decomposition: ln(speed ratio) = ln(cadence ratio) + ln(stride ratio)
        speed_log = log(speed_2 / speed_1)
        cadence_log = log(cadence_2 / cadence_1)
        stride_log = speed_log - cadence_log
        if abs(pace_change_seconds) >= MEANINGFUL_PACE_CHANGE_SECONDS and abs(speed_log) > 1e-9:
            direction = "quicker" if pace_change_seconds < 0 else "slower"
            cadence_share = cadence_log / speed_log
            stride_share = stride_log / speed_log
            opening = f"Your second half was {abs(pace_change_seconds):.0f} sec/mi {direction}"
            # When one component moves against the speed change its share goes
            # negative and the other exceeds 100%. That is arithmetically right
            # but unreadable as a percentage, so it is described instead.
            if cadence_share < 0:
                notes.append(
                    f"{opening}, entirely through stride length; your turnover moved the "
                    "other way and partly offset it."
                )
            elif stride_share < 0:
                notes.append(
                    f"{opening}, entirely through turnover; your stride length moved the "
                    "other way and partly offset it."
                )
            elif stride_share >= DOMINANT_SHARE:
                notes.append(
                    f"{opening}, mostly through stride length rather than turnover "
                    f"({stride_share * 100:.0f}% stride, {cadence_share * 100:.0f}% cadence)."
                )
            elif cadence_share >= DOMINANT_SHARE:
                notes.append(
                    f"{opening}, mostly through turnover rather than stride length "
                    f"({cadence_share * 100:.0f}% cadence, {stride_share * 100:.0f}% stride)."
                )
            else:
                notes.append(
                    f"{opening}, through a roughly even mix of turnover and stride length "
                    f"({cadence_share * 100:.0f}% cadence, {stride_share * 100:.0f}% stride)."
                )
        if abs(cadence_change) >= MEANINGFUL_CADENCE_CHANGE_SPM:
            verb = "rose" if cadence_change > 0 else "fell"
            clause = f"Cadence {verb} {abs(cadence_change):.0f} spm between halves"
            if cadence_change < 0 and pace_change_seconds > MEANINGFUL_PACE_CHANGE_SECONDS:
                clause += f" while pace slowed {pace_change_seconds:.0f} sec/mi"
            elif cadence_change < 0 and abs(pace_change_seconds) < MEANINGFUL_PACE_CHANGE_SECONDS:
                clause += " while pace held, so your stride lengthened to compensate"
            notes.append(clause + ".")

    average = overall["cadence_spm"]
    if comparison.available and average and comparison.personal_spread_spm:
        deviation = (average - comparison.personal_median_spm) / max(
            comparison.personal_spread_spm, 1e-9
        )
        if abs(deviation) >= UNUSUAL_DEVIATIONS:
            side = "higher" if deviation > 0 else "lower"
            notes.append(
                f"At {average:.0f} spm this run's cadence was unusually {side} for you at this "
                f"pace, about {abs(deviation):.1f} times your usual variation from "
                f"{comparison.personal_median_spm:.0f} spm."
            )
        else:
            notes.append(
                f"At {average:.0f} spm this run's cadence was typical for you at this pace."
            )
    return notes


def build_cadence_analysis(
    connection: sqlite3.Connection, activity_id: int, intervals: list[MovementInterval]
) -> CadenceAnalysis:
    """Summarize cadence for one run against the athlete's own history."""

    overall = _half_summary(intervals)
    coverage = (
        overall["cadence_seconds"] / overall["moving_seconds"]
        if overall["moving_seconds"]
        else 0.0
    )
    if not overall["cadence_spm"] or coverage < MINIMUM_COVERAGE_FRACTION:
        return CadenceAnalysis(
            available=False,
            coverage_fraction=coverage,
            limitations=[
                "This activity does not have enough recorded cadence to analyze."
                if coverage < MINIMUM_COVERAGE_FRACTION
                else "No cadence was recorded for this activity."
            ],
        )

    first_intervals, second_intervals = _split_halves(intervals)
    first = _half_summary(first_intervals)
    second = _half_summary(second_intervals)
    comparison = (
        personal_cadence_band(connection, activity_id, overall["pace_min_mile"])
        if overall["pace_min_mile"]
        else CadenceComparison(available=False, detail="This run has no usable pace.")
    )
    cadence_values = [
        value
        for value in (_interval_cadence(item) for item in intervals if item.moving_time_s > 0)
        if value is not None
    ]
    return CadenceAnalysis(
        available=True,
        average_spm=overall["cadence_spm"],
        median_spm=median(cadence_values) if cadence_values else None,
        first_half_spm=first["cadence_spm"],
        second_half_spm=second["cadence_spm"],
        change_spm=(
            second["cadence_spm"] - first["cadence_spm"]
            if first["cadence_spm"] and second["cadence_spm"]
            else None
        ),
        average_stride_length_m=overall["stride_length_m"],
        first_half_stride_length_m=first["stride_length_m"],
        second_half_stride_length_m=second["stride_length_m"],
        coverage_fraction=coverage,
        comparison=comparison,
        observations=_observations(overall, first, second, comparison),
        limitations=[
            "Stride length is speed divided by cadence, not a measured ground-contact "
            "distance, so it inherits any error in either.",
            "No target cadence is prescribed. Optimal turnover depends on your height, "
            "leg length, speed, and terrain, and copying a population number usually "
            "just shortens the stride at the same speed.",
        ],
    )
