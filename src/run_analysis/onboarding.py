"""First-run setup: heart-rate zones, comparison heart rate, and what is missing.

Every number this app produces is anchored to two athlete-specific settings:
the heart-rate zones that decide what counts as easy, and the comparison heart
rate that every "pace at X bpm" figure refers to. Both ship with defaults, and
defaults that are silently wrong are worse than no answer -- the app will keep
producing confident figures about the wrong athlete.

So the job here is not to collect settings. It is to get those two right with
the least possible guessing, prefer measured values over formulas, say which
was used, and derive the comparison heart rate from where the athlete's own
history actually has evidence rather than from a round number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import StrEnum
import json
import sqlite3

ZONE_NAMES: tuple[str, ...] = ("z1", "z2", "z3", "z4", "z5")


class ZoneMethod(StrEnum):
    """How the five zone boundaries were arrived at."""

    #: Copied from the watch. Preferred when it exists: it is what the athlete
    #: already sees mid-run, and disagreeing with it just creates two truths.
    DEVICE = "device"
    #: Karvonen. Uses both max and resting HR, so it adapts to a low resting
    #: pulse instead of pretending everyone's easy pace starts at the same %.
    HEART_RATE_RESERVE = "heart_rate_reserve"
    #: Percent of max HR. Cruder, but the only option without a resting HR.
    PERCENT_MAX = "percent_max"
    #: Entered by hand, from a lab test or a coach.
    CUSTOM = "custom"


class MaxHrSource(StrEnum):
    MEASURED = "measured"
    ESTIMATED = "estimated"


#: Zone boundaries as fractions, upper edge of each zone. Both schemes use the
#: conventional five-zone 50/60/70/80/90/100 split; what differs is the
#: quantity the percentage is taken of.
_ZONE_EDGES: tuple[float, ...] = (0.50, 0.60, 0.70, 0.80, 0.90, 1.00)


def estimate_max_hr(age_years: float) -> int:
    """Tanaka, Monahan & Seals (2001): 208 - 0.7 x age.

    Preferred over 220 - age, which overestimates in the young and
    underestimates past about forty. Both carry a standard deviation of
    roughly 7 bpm across individuals, which is why a measured value always
    wins and why :mod:`vo2_estimation` widens its interval when this is used.
    """

    return int(round(208.0 - 0.7 * float(age_years)))


def _edges_to_zones(values: list[int]) -> dict[str, tuple[int, int]]:
    """Turn six ascending boundaries into five non-overlapping zones."""

    zones: dict[str, tuple[int, int]] = {}
    for index, name in enumerate(ZONE_NAMES):
        lower = values[index] if index == 0 else values[index] + 1
        zones[name] = (lower, values[index + 1])
    return zones


def derive_zones(
    method: ZoneMethod,
    *,
    maximum_hr: int,
    resting_hr: int | None = None,
    boundaries: dict[str, tuple[int, int]] | None = None,
) -> dict[str, tuple[int, int]]:
    """Produce five zones by the named method.

    ``DEVICE`` and ``CUSTOM`` are pass-throughs: the athlete supplies the
    numbers and this only checks their shape. The two computed methods are
    conventions, not measurements, which is why the method is stored alongside
    the result instead of being forgotten once the numbers exist.
    """

    if method in (ZoneMethod.DEVICE, ZoneMethod.CUSTOM):
        if not boundaries or set(boundaries) != set(ZONE_NAMES):
            raise ValueError("All five zones must be supplied for this method")
        return {name: (int(low), int(high)) for name, (low, high) in boundaries.items()}

    if method == ZoneMethod.HEART_RATE_RESERVE:
        if resting_hr is None:
            raise ValueError("Heart-rate reserve zones need a resting heart rate")
        if resting_hr >= maximum_hr:
            raise ValueError("Resting heart rate must be below maximum heart rate")
        reserve = maximum_hr - resting_hr
        edges = [int(round(resting_hr + fraction * reserve)) for fraction in _ZONE_EDGES]
    else:
        edges = [int(round(fraction * maximum_hr)) for fraction in _ZONE_EDGES]

    # Rounding can collapse two edges onto the same bpm at low reserves; nudge
    # them apart so no zone ends up empty or inverted.
    for index in range(1, len(edges)):
        if edges[index] <= edges[index - 1]:
            edges[index] = edges[index - 1] + 1
    edges[-1] = max(edges[-1], maximum_hr)
    return _edges_to_zones(edges)


@dataclass(frozen=True, slots=True)
class ComparisonHrCandidate:
    heart_rate_bpm: int
    run_count: int
    segment_count: int


@dataclass(frozen=True, slots=True)
class ComparisonHrRecommendation:
    """A comparison heart rate chosen by evidence, plus why."""

    recommended_bpm: int | None
    run_count: int
    segment_count: int
    zone_lower_bpm: int
    zone_upper_bpm: int
    rationale: str
    candidates: list[ComparisonHrCandidate] = field(default_factory=list)


#: Below this many distinct runs at a heart rate, the standardized pace there
#: rests on too few observations to be a stable comparison point.
MINIMUM_SUPPORTING_RUNS = 8


def comparison_hr_support(
    connection: sqlite3.Connection, zone_lower: int, zone_upper: int
) -> list[ComparisonHrCandidate]:
    """Count how much real evidence exists at each heart rate inside a zone.

    Breadth is counted in distinct runs rather than segments on purpose: forty
    segments from three long runs describe those three runs, while forty runs
    describe the athlete.
    """

    rows = connection.execute(
        """
        SELECT CAST(ROUND(average_hr_bpm) AS INTEGER) AS bpm,
               COUNT(DISTINCT activity_id) AS runs,
               COUNT(*) AS segments
        FROM segments
        WHERE average_hr_bpm IS NOT NULL
          AND is_pathological = 0
          AND average_hr_bpm BETWEEN ? AND ?
        GROUP BY bpm
        ORDER BY bpm
        """,
        (zone_lower, zone_upper),
    ).fetchall()
    return [ComparisonHrCandidate(int(r[0]), int(r[1]), int(r[2])) for r in rows]


def recommend_comparison_hr(
    connection: sqlite3.Connection, zones: dict[str, tuple[int, int]]
) -> ComparisonHrRecommendation:
    """Pick the heart rate inside Z2 where this athlete has the most evidence.

    The comparison heart rate is where every pace figure is evaluated, so any
    run whose heart rate sat far from it must be extrapolated, with uncertainty
    that grows the further it reaches. Choosing the best-supported point inside
    the easy zone therefore maximises how many runs are directly comparable,
    which is a very different thing from choosing a round number.
    """

    lower, upper = (int(value) for value in zones["z2"])
    candidates = comparison_hr_support(connection, lower, upper)
    if not candidates:
        return ComparisonHrRecommendation(
            recommended_bpm=None,
            run_count=0,
            segment_count=0,
            zone_lower_bpm=lower,
            zone_upper_bpm=upper,
            rationale=(
                "There are no recorded segments inside Z2 yet, so there is no evidence "
                "to choose a comparison heart rate from. Upload some easy runs first."
            ),
            candidates=[],
        )

    # Runs first, segments only to break ties: a heart rate reached on many
    # separate days is a better anchor than one held for a long time once.
    best = max(candidates, key=lambda item: (item.run_count, item.segment_count))
    midpoint = (lower + upper) // 2
    if best.run_count < MINIMUM_SUPPORTING_RUNS:
        return ComparisonHrRecommendation(
            recommended_bpm=midpoint,
            run_count=best.run_count,
            segment_count=best.segment_count,
            zone_lower_bpm=lower,
            zone_upper_bpm=upper,
            rationale=(
                f"No heart rate in Z2 has more than {best.run_count} runs behind it yet, "
                f"which is too thin to choose between them. The middle of Z2 "
                f"({midpoint} bpm) is a reasonable placeholder until more easy runs "
                "accumulate."
            ),
            candidates=candidates,
        )
    return ComparisonHrRecommendation(
        recommended_bpm=best.heart_rate_bpm,
        run_count=best.run_count,
        segment_count=best.segment_count,
        zone_lower_bpm=lower,
        zone_upper_bpm=upper,
        rationale=(
            f"{best.heart_rate_bpm} bpm is the best-supported point in your Z2 "
            f"({lower}–{upper} bpm): {best.run_count} separate runs spent time there, "
            f"across {best.segment_count} segments. Anchoring the comparison here means "
            "fewer runs have to be extrapolated to be compared."
        ),
        candidates=candidates,
    )


def automatic_comparison_hr(connection, zones, *, current=None, as_of=None):
    """Recent comparable aerobic evidence, with a provisional Z2 midpoint.

    This is an analysis coordinate, not a workout target or a fitness signal.
    Re-evaluate all displayed history at one coordinate after it changes. The
    90-day window limits old training's influence; distinct runs limit a long
    run's influence. Prefer the current coordinate when support is tied.
    """
    as_of = as_of or datetime.now(timezone.utc)
    lower, upper = map(int, zones['z2'])
    cutoff = as_of - timedelta(days=90)
    candidates = []
    if connection is not None:
        from .run_feedback import _infer_workout_type, _long_run_threshold_from_connection

        long_threshold = _long_run_threshold_from_connection(connection)
        rows = connection.execute('''
            SELECT CAST(ROUND(s.average_hr_bpm) AS INTEGER) AS bpm,
                   a.id, COUNT(*) AS segments, o.workout_type,
                   m.detected_workout_type, a.total_distance_m,
                   m.calculated_moving_time_s, m.device_timer_time_s,
                   m.hr_zone_seconds_json
            FROM segments s JOIN activities a ON a.id=s.activity_id
            JOIN activity_metrics m ON m.activity_id=a.id
            LEFT JOIN run_overrides o ON o.activity_id=a.activity_id
            WHERE a.sport='Running'
              AND julianday(a.start_time_utc) BETWEEN julianday(?) AND julianday(?)
              AND s.average_hr_bpm BETWEEN ? AND ? AND s.is_pathological=0
              AND m.model_eligible=1 AND COALESCE(o.illness,0)=0
              AND COALESCE(o.health_tag,'normal')='normal'
              AND COALESCE(o.include_in_model,1)!=0
            GROUP BY a.id,bpm ORDER BY bpm,a.id
        ''', (cutoff.isoformat(), as_of.isoformat(), lower, upper)).fetchall()
        # detected_workout_type records structured sessions, not every ordinary
        # easy/long run. Resolve roles exactly as Run Analysis/Progress do;
        # absence of a structured-workout label is not absence of evidence.
        roles = {}
        support = {}
        for bpm, activity_id, segments, explicit, detected, distance, moving, timer, zone_json in rows:
            if activity_id not in roles:
                roles[activity_id] = _infer_workout_type(
                    explicit, detected, float(distance or 0) / 1609.344,
                    float(moving or timer or 0) / 60,
                    json.loads(zone_json or '{}'), long_threshold,
                ).value
            if roles[activity_id] not in {'easy', 'long', 'recovery'}:
                continue
            run_ids, segment_count = support.get(int(bpm), (set(), 0))
            run_ids.add(activity_id)
            support[int(bpm)] = run_ids, segment_count + int(segments)
        candidates = [ComparisonHrCandidate(bpm, len(run_ids), segments)
            for bpm, (run_ids, segments) in sorted(support.items())]
    midpoint = (lower + upper) // 2
    best = max(candidates, key=lambda item: (item.run_count,
        item.heart_rate_bpm == current, item.segment_count,
        -abs(item.heart_rate_bpm - midpoint)), default=None)
    supported = best is not None and best.run_count >= MINIMUM_SUPPORTING_RUNS
    chosen = best.heart_rate_bpm if supported else midpoint
    return ComparisonHrRecommendation(chosen, best.run_count if best else 0,
        best.segment_count if best else 0, lower, upper,
        'Chosen from your recent easy running.' if supported else
        'Using the middle of Z2 until enough comparable easy runs are available.', candidates)


class SetupStep(StrEnum):
    """The decisions that must be real before the numbers mean anything."""

    RUNS = "runs"
    HEART_RATE = "heart_rate"
    ZONES = "zones"
    COMPARISON_HR = "comparison_hr"  # Legacy saved confirmations; no user setup step.
    PROFILE = "profile"
    GOAL = "goal"
    PROGRESSION = "progression"
    WEATHER = "weather"


@dataclass(frozen=True, slots=True)
class SetupStepState:
    step: SetupStep
    title: str
    complete: bool
    #: What the app is doing right now, whether or not the step is done. A
    #: default in use is not the same as a question unanswered, and saying
    #: which is which is the difference between a settings page and guidance.
    detail: str
    blocking: bool = False


@dataclass(frozen=True, slots=True)
class SetupState:
    complete: bool
    steps: list[SetupStepState]

    @property
    def next_step(self) -> SetupStepState | None:
        return next((step for step in self.steps if not step.complete), None)


def _confirmed(config: dict) -> set[str]:
    raw = (config.get("setup") or {}).get("confirmed_steps") or []
    return {str(value) for value in raw}


def setup_state(connection, config: dict) -> SetupState:
    """Report what is set up and what is still running on a default.

    Confirmation is tracked explicitly rather than inferred from a value being
    present, because every one of these settings ships with a plausible
    default. Inferring completeness from a non-empty field would mark the whole
    setup done before the athlete had answered a single question.
    """

    confirmed = _confirmed(config)
    runs = connection.execute(
        "SELECT COUNT(*) FROM activities WHERE sport = 'Running'"
    ).fetchone()[0]
    from .training_background import reported_background, calibration_activity
    background = reported_background(config)
    calibrated = (calibration_activity(connection,config,
        reported_at=datetime.fromisoformat(config['training_background_reported_at']))
        if background and config.get('training_background_reported_at') else None)
    profile = config.get("profile") or {}
    zones = config.get("zones") or {}
    zone_method = (config.get("setup") or {}).get("zone_method")
    coaching = config.get("coaching") or {}
    weather = config.get("weather") or {}

    maximum_hr = config.get("max_hr")
    resting_hr = config.get("resting_hr")
    max_hr_source = str(profile.get("max_hr_source") or "estimated")
    zone_2 = zones.get("z2")

    steps = [
        SetupStepState(
            SetupStep.RUNS,
            "Record your first easy run" if background and not calibrated else "Add your runs",
            complete=bool(calibrated) if background else runs > 0,
            detail=(
                'Starting training is self-reported. Record one normal easy run with GPS and heart rate to calibrate the full plan.'
                if background and not calibrated else
                f"{runs} running activities imported."
                if runs
                else (
                    "Import recent FIT, TCX or GPX activity files, or a Garmin ZIP, or use Sync. No history? "
                    "Finish personal details and heart-rate setup before a baseline run."
                )
            ),
            blocking=True,
        ),
        SetupStepState(
            SetupStep.HEART_RATE,
            "Maximum and resting heart rate",
            complete=SetupStep.HEART_RATE in confirmed,
            detail=(
                f"Max {maximum_hr} bpm ({max_hr_source}), resting {resting_hr} bpm."
                if maximum_hr and resting_hr
                else "Not set."
            )
            + (
                ""
                if max_hr_source == "measured"
                else " An estimated max is the largest single source of error in your VO₂ figure."
            ),
        ),
        SetupStepState(
            SetupStep.ZONES,
            "Heart-rate zones",
            complete=SetupStep.ZONES in confirmed,
            detail=(
                f"Z2 is {int(zone_2[0])}–{int(zone_2[1])} bpm"
                + (f", set from {ZONE_METHOD_LABELS.get(zone_method, zone_method)}." if zone_method else ", using the shipped default.")
                if zone_2
                else "Not set."
            ),
        ),
        SetupStepState(
            SetupStep.PROFILE,
            "About you",
            complete=bool(profile.get("birth_date") and profile.get("sex") and profile.get("weight_lb") and profile.get("height_in")),
            detail=(
                "Used only for the VO₂ estimate, which needs weight, sex and age."
                if profile
                else "Without this the VO₂ estimate cannot be calculated."
            ),
        ),
        SetupStepState(
            SetupStep.GOAL,
            "What you are training for",
            complete=SetupStep.GOAL in confirmed,
            detail=(
                f"Currently {str(config.get('race_preparation', {}).get('distance', 'general_fitness')).replace('_', ' ')}."
                + (' Race preparation is a saved draft, not an active calendar.'
                   if config.get('race_preparation', {}).get('distance', 'general_fitness') != 'general_fitness' else '')
            ),
        ),
        SetupStepState(
            SetupStep.WEATHER,
            "Weather and privacy",
            complete=SetupStep.WEATHER in confirmed,
            detail=(
                "Historical weather "
                + ("on" if weather.get("historical_enabled") else "off")
                + ", forecasts "
                + ("on" if weather.get("forecast_enabled") else "off")
                + f". Locations are rounded and jittered by {weather.get('privacy_jitter_radius_km', 0)} km before any request."
            ),
        ),
    ]
    from .adaptive.live_bridge import enabled as adaptive_enabled
    if adaptive_enabled(config):
        steps.insert(-1, SetupStepState(
            SetupStep.PROGRESSION,
            "Running progression",
            complete=SetupStep.PROGRESSION in confirmed,
            detail="Long-run goal guides progression; run frequency guides how training is spread across days.",
        ))
    # History gates analysis/planning, not personal calibration. New athletes
    # must have their profile and zones before a first run is suggested.
    order = [SetupStep.PROFILE, SetupStep.HEART_RATE, SetupStep.ZONES,
             SetupStep.COMPARISON_HR, SetupStep.GOAL, SetupStep.PROGRESSION,
             SetupStep.WEATHER, SetupStep.RUNS]
    steps.sort(key=lambda step: order.index(step.step))
    return SetupState(complete=all(step.complete for step in steps), steps=steps)


ZONE_METHOD_LABELS = {
    ZoneMethod.DEVICE: "your watch",
    ZoneMethod.HEART_RATE_RESERVE: "heart-rate reserve",
    ZoneMethod.PERCENT_MAX: "percent of max",
    ZoneMethod.CUSTOM: "custom values",
}
