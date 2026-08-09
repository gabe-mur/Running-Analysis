"""FIT parsing into the same records the TCX reader produces.

Everything downstream works from :class:`~.models.Activity`, so supporting FIT
is a parser, not a second pipeline. What makes it delicate is that FIT stores
several quantities in units nobody would guess, and getting one wrong produces
plausible numbers rather than an error:

* ``cadence`` is strides per minute for **one leg**, like Garmin's TCX
  extension. Not doubling it halves every value and turns runs into walks.
* ``position_lat``/``position_long`` are semicircles, not degrees.
* ``altitude`` is stored as ``(metres + 500) * 5``; ``enhanced_altitude`` is
  plain metres.
* timestamps count from 1989-12-31, not the Unix epoch.
* ``sport`` is an enum, and the rest of this application matches the literal
  string ``"Running"``.

Each of those is handled once, here, with the conversion named.

FIT also records something TCX cannot: explicit timer start/stop events. Those
are collected as real pause intervals rather than left to be inferred from
speed, which is strictly better evidence than the TCX path can offer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
import gzip
import io

from .activity_assembly import finish_activity
from .cadence import FIT_ONE_SIDED_CADENCE_SOURCE
from .models import Activity, Lap, ParsedTCX, Trackpoint

#: FIT timestamps are seconds since 1989-12-31T00:00:00Z. Some files carry
#: already-converted datetimes, which is why this is applied conditionally.
FIT_EPOCH = datetime(1989, 12, 31, tzinfo=timezone.utc)

#: Semicircles to degrees: the field is a signed 32-bit fraction of a turn.
SEMICIRCLE_TO_DEGREES = 180.0 / (2**31)

#: ``sport`` enum values this application treats as running. The rest of the
#: codebase matches the literal string, so the mapping has to be exact.
_SPORT_NAMES = {
    "running": "Running",
    "cycling": "Biking",
    "walking": "Walking",
    "hiking": "Hiking",
    "swimming": "Swimming",
}
_SPORT_ENUM = {1: "Running", 2: "Biking", 11: "Walking", 17: "Hiking", 5: "Swimming"}


@dataclass(slots=True)
class TimerEvent:
    """A recorded timer transition, used to derive genuine stopped intervals."""

    timestamp_utc: datetime
    stopped: bool


@dataclass(slots=True)
class FitPauses:
    """Pause intervals a FIT file states outright."""

    intervals: list[tuple[datetime, datetime]] = field(default_factory=list)

    @property
    def stopped_seconds(self) -> float:
        return sum((end - start).total_seconds() for start, end in self.intervals)


def _as_datetime(value) -> datetime | None:
    """Normalise a FIT timestamp to an aware UTC datetime."""

    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    return FIT_EPOCH + timedelta(seconds=seconds)


def _degrees(semicircles) -> float | None:
    if semicircles is None:
        return None
    try:
        return float(semicircles) * SEMICIRCLE_TO_DEGREES
    except (TypeError, ValueError):
        return None


def _sport_name(raw) -> str:
    if raw is None:
        return "Unknown"
    if isinstance(raw, str):
        return _SPORT_NAMES.get(raw.strip().casefold(), raw.strip().title() or "Unknown")
    try:
        return _SPORT_ENUM.get(int(raw), "Unknown")
    except (TypeError, ValueError):
        return "Unknown"


def _number(value) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result


def _altitude_m(values: dict) -> float | None:
    """Prefer ``enhanced_altitude``; fall back to the scaled legacy field.

    fitdecode applies the FIT profile's own scale and offset, so both arrive in
    metres. The raw-integer form is only handled for files whose definition
    messages omit the profile, where a value in the thousands is unmistakably
    unscaled.
    """

    for key in ("enhanced_altitude", "altitude"):
        value = _number(values.get(key))
        if value is None:
            continue
        if value > 10000:  # metres this large are not a place anyone runs
            return value / 5.0 - 500.0
        return value
    return None


def _cadence_spm_parts(values: dict) -> int | None:
    """One-sided strides per minute, including the fractional half-stride.

    Returned unchanged and unmultiplied: doubling belongs to
    :mod:`~.cadence`, which every reader goes through.
    """

    cadence = _number(values.get("cadence"))
    if cadence is None:
        return None
    fractional = _number(values.get("fractional_cadence")) or 0.0
    return int(round(cadence + fractional))


def _open(path: Path):
    """FIT files exported from Strava are sometimes gzipped."""

    if path.suffix.casefold() == ".gz" or path.name.casefold().endswith(".fit.gz"):
        return io.BytesIO(gzip.decompress(path.read_bytes()))
    return io.BytesIO(path.read_bytes())


def parse_fit(path: str | Path, default_timezone: str = "America/New_York") -> ParsedTCX:
    """Read a FIT file into the shared activity records.

    Returns the same container the TCX reader does so the importer does not
    have to care which format a run arrived in.
    """

    import fitdecode  # imported lazily so TCX-only installs need no FIT support

    source_path = Path(path)
    warnings: list[str] = []
    laps: list[Lap] = []
    trackpoints: list[Trackpoint] = []
    events: list[TimerEvent] = []
    session: dict = {}
    file_id: dict = {}
    device: dict = {}

    with fitdecode.FitReader(_open(source_path)) as reader:
        for frame in reader:
            if frame.frame_type != fitdecode.FIT_FRAME_DATA:
                continue
            values = {f.name: f.value for f in frame.fields}
            if frame.name == "record":
                trackpoints.append(_record_to_trackpoint(values, len(laps), len(trackpoints)))
            elif frame.name == "lap":
                laps.append(_lap(values, len(laps)))
            elif frame.name == "session":
                session = values
            elif frame.name == "file_id":
                file_id = values
            elif frame.name == "device_info" and values.get("manufacturer") and not device:
                device = values
            elif frame.name == "event" and values.get("event") == "timer":
                stamp = _as_datetime(values.get("timestamp"))
                event_type = str(values.get("event_type") or "")
                if stamp and event_type in {"start", "stop", "stop_all", "stop_disable", "stop_disable_all"}:
                    events.append(TimerEvent(stamp, stopped=event_type != "start"))

    if not trackpoints:
        warnings.append("fit_no_record_messages")
    recorded = _apply_timer_pauses(trackpoints, timer_pauses(events))
    if recorded:
        warnings.append("stopped_time_from_recorded_timer_events")

    activity = Activity(
        activity_id=_activity_id(file_id, session, trackpoints),
        sport=_sport_name(session.get("sport") if session else None),
        notes=None,  # FIT has no free-text equivalent of the TCX Notes element
        creator=_creator(device, file_id),
        namespaces=[],  # an XML concept with no FIT counterpart
        laps=laps or _synthetic_lap(session, trackpoints),
        trackpoints=trackpoints,
        parse_warnings=warnings,
    )
    finish_activity(activity, default_timezone)
    return ParsedTCX([activity], [], sorted(set(warnings)), "binary/fit")


def _record_to_trackpoint(values: dict, lap_index: int, point_index: int) -> Trackpoint:
    flags: list[str] = []
    latitude = _degrees(values.get("position_lat"))
    longitude = _degrees(values.get("position_long"))
    in_range = (
        latitude is not None
        and longitude is not None
        and -90.0 <= latitude <= 90.0
        and -180.0 <= longitude <= 180.0
    )
    if latitude is None and longitude is None:
        gps_valid = False
    elif (latitude is None) != (longitude is None):
        gps_valid = False
        flags.append("incomplete_position")
    elif not in_range:
        gps_valid = False
        flags.append("out_of_range_position")
    else:
        gps_valid = True

    heart_rate = _number(values.get("heart_rate"))
    speed = _number(values.get("enhanced_speed"))
    if speed is None:
        speed = _number(values.get("speed"))
    return Trackpoint(
        lap_index=lap_index,
        track_index=0,
        point_index=point_index,
        timestamp_utc=_as_datetime(values.get("timestamp")),
        latitude=latitude if gps_valid else None,
        longitude=longitude if gps_valid else None,
        gps_valid=gps_valid,
        altitude_m=_altitude_m(values),
        distance_m=_number(values.get("distance")),
        heart_rate_bpm=int(heart_rate) if heart_rate is not None else None,
        cadence=_cadence_spm_parts(values),
        run_cadence=_cadence_spm_parts(values),
        cadence_source=FIT_ONE_SIDED_CADENCE_SOURCE if values.get("cadence") is not None else None,
        speed_mps=speed,
        parse_flags=sorted(set(flags)),
    )


def _lap(values: dict, index: int) -> Lap:
    average_hr = _number(values.get("avg_heart_rate"))
    maximum_hr = _number(values.get("max_heart_rate"))
    calories = _number(values.get("total_calories"))
    maximum_speed = _number(values.get("enhanced_max_speed"))
    if maximum_speed is None:
        maximum_speed = _number(values.get("max_speed"))
    # total_timer_time excludes paused time, which is what the TCX
    # TotalTimeSeconds element also reports.
    total_time = _number(values.get("total_timer_time"))
    if total_time is None:
        total_time = _number(values.get("total_elapsed_time"))
    return Lap(
        lap_index=index,
        start_time_utc=_as_datetime(values.get("start_time")),
        total_time_s=total_time,
        distance_m=_number(values.get("total_distance")),
        calories=int(calories) if calories is not None else None,
        average_hr_bpm=int(average_hr) if average_hr is not None else None,
        maximum_hr_bpm=int(maximum_hr) if maximum_hr is not None else None,
        maximum_speed_mps=maximum_speed,
        intensity=str(values.get("intensity")) if values.get("intensity") is not None else None,
        trigger_method=str(values.get("lap_trigger")) if values.get("lap_trigger") is not None else None,
    )


def _synthetic_lap(session: dict, trackpoints: list[Trackpoint]) -> list[Lap]:
    """Some files carry a session but no lap messages.

    The summary fields the assembly step needs -- distance, recorded time,
    heart rates -- all live on the session too, so one lap is synthesised from
    it rather than leaving the activity with no summary at all.
    """

    if not session and not trackpoints:
        return []
    if not session:
        return []
    return [_lap({**session, "start_time": session.get("start_time")}, 0)]


def _activity_id(file_id: dict, session: dict, trackpoints: list[Trackpoint]) -> str | None:
    for candidate in (file_id.get("time_created"), session.get("start_time")):
        stamp = _as_datetime(candidate)
        if stamp:
            return stamp.isoformat().replace("+00:00", "Z")
    stamps = [point.timestamp_utc for point in trackpoints if point.timestamp_utc]
    return min(stamps).isoformat().replace("+00:00", "Z") if stamps else None


def _creator(device: dict, file_id: dict) -> str | None:
    for key in ("garmin_product", "product_name", "product", "manufacturer"):
        value = device.get(key) or file_id.get(key)
        if value:
            return str(value).replace("_", " ").strip() or None
    return None


def timer_pauses(events: list[TimerEvent]) -> FitPauses:
    """Turn timer transitions into closed stopped intervals.

    A stop with no matching start is dropped rather than assumed to run to the
    end of the file: an unterminated pause is ambiguous, and inventing a
    duration for it would silently inflate stopped time.
    """

    pauses = FitPauses()
    stopped_at: datetime | None = None
    for event in sorted(events, key=lambda item: item.timestamp_utc):
        if event.stopped and stopped_at is None:
            stopped_at = event.timestamp_utc
        elif not event.stopped and stopped_at is not None:
            if event.timestamp_utc > stopped_at:
                pauses.intervals.append((stopped_at, event.timestamp_utc))
            stopped_at = None
    return pauses


def _apply_timer_pauses(trackpoints: list[Trackpoint], pauses: FitPauses) -> int:
    """Attach each recorded pause to the point it follows.

    During a pause a watch normally stops writing records altogether, so the
    pause shows up as a gap between two consecutive points rather than as
    points of its own. Marking the point before the gap is therefore the whole
    signal, and it lets the movement classifier treat that interval as a known
    stop instead of arguing with a stale endpoint speed.
    """

    if not pauses.intervals or not trackpoints:
        return 0
    ordered = [point for point in trackpoints if point.timestamp_utc is not None]
    applied = 0
    for start, end in pauses.intervals:
        preceding = [point for point in ordered if point.timestamp_utc <= start]
        if not preceding:
            continue
        anchor = preceding[-1]
        anchor.pause_after_s = (anchor.pause_after_s or 0.0) + (end - start).total_seconds()
        applied += 1
    return applied
