"""Typed records emitted by the TCX parser."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from .cadence import cadence_spm


@dataclass(slots=True)
class Trackpoint:
    lap_index: int
    track_index: int
    point_index: int
    timestamp_utc: datetime | None
    latitude: float | None
    longitude: float | None
    gps_valid: bool
    altitude_m: float | None
    distance_m: float | None
    heart_rate_bpm: int | None
    cadence: int | None
    run_cadence: int | None
    cadence_source: str | None
    speed_mps: float | None
    #: Seconds of pause the device itself recorded immediately after this
    #: point. Only FIT states this; TCX has no equivalent, so it stays None
    #: there and stopped time continues to be inferred from movement.
    pause_after_s: float | None = None
    parse_flags: list[str] = field(default_factory=list)

    @property
    def cadence_spm(self) -> float | None:
        """Total steps per minute. Raw ``cadence``/``cadence_source`` are kept
        as recorded so the conversion stays auditable."""
        return cadence_spm(self.cadence, self.cadence_source)


@dataclass(slots=True)
class Lap:
    lap_index: int
    start_time_utc: datetime | None
    total_time_s: float | None
    distance_m: float | None
    calories: int | None
    average_hr_bpm: int | None
    maximum_hr_bpm: int | None
    maximum_speed_mps: float | None
    intensity: str | None
    trigger_method: str | None


@dataclass(slots=True)
class Activity:
    activity_id: str | None
    sport: str
    notes: str | None
    creator: str | None
    namespaces: list[str]
    laps: list[Lap]
    trackpoints: list[Trackpoint]
    start_time_utc: datetime | None = None
    start_time_local: datetime | None = None
    timezone_name: str | None = None
    timezone_source: str | None = None
    total_elapsed_time_s: float | None = None
    lap_recorded_time_s: float | None = None
    total_distance_m: float | None = None
    calories: int | None = None
    average_hr_bpm: float | None = None
    maximum_hr_bpm: int | None = None
    gps_quality: str = "gps_missing"
    hr_quality: str = "hr_missing"
    elevation_quality: str = "elevation_missing"
    cadence_quality: str = "cadence_missing"
    distance_source: str = "unknown"
    parse_warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ParsedTCX:
    activities: list[Activity]
    namespaces: list[str]
    warnings: list[str]
    encoding: str

