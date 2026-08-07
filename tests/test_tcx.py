from __future__ import annotations

from pathlib import Path

import pytest

from run_analysis.tcx import _quality, parse_tcx


TCX_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<TrainingCenterDatabase xmlns="http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"
 xmlns:ae="http://www.garmin.com/xmlschemas/ActivityExtension/v2">
 <Activities><Activity Sport="Running">
  <Id>{activity_id}</Id><Notes>{notes}</Notes>
  <Lap StartTime="{start}">
   <TotalTimeSeconds>20</TotalTimeSeconds><DistanceMeters>40</DistanceMeters>
   <Calories>4</Calories><AverageHeartRateBpm><Value>145</Value></AverageHeartRateBpm>
   <MaximumHeartRateBpm><Value>150</Value></MaximumHeartRateBpm>
   <Track><Trackpoint><Time>{start}</Time>{position}<AltitudeMeters>5</AltitudeMeters>
    <DistanceMeters>0</DistanceMeters>{hr}<Cadence>77</Cadence>
    <Extensions><ae:TPX><ae:Speed>2.0</ae:Speed></ae:TPX></Extensions>
   </Trackpoint><Trackpoint><Time>{end}</Time>{position}<AltitudeMeters>6</AltitudeMeters>
    <DistanceMeters>40</DistanceMeters>{hr}
    <Extensions><ae:TPX><ae:Speed>2.1</ae:Speed><ae:RunCadence>79</ae:RunCadence></ae:TPX></Extensions>
   </Trackpoint></Track>
  </Lap>
 </Activity></Activities>
</TrainingCenterDatabase>"""


def make_tcx(
    path: Path,
    *,
    start: str = "2024-07-01T12:00:00Z",
    end: str = "2024-07-01T12:00:20Z",
    activity_id: str = "2024-07-01T12:00:00Z",
    gps: bool = True,
    hr: bool = True,
) -> Path:
    position = (
        "<Position><LatitudeDegrees>40.7</LatitudeDegrees>"
        "<LongitudeDegrees>-73.95</LongitudeDegrees></Position>"
        if gps
        else ""
    )
    heart_rate = "<HeartRateBpm><Value>145</Value></HeartRateBpm>" if hr else ""
    path.write_text(
        TCX_TEMPLATE.format(
            activity_id=activity_id,
            start=start,
            end=end,
            notes="test run",
            position=position,
            hr=heart_rate,
        ),
        encoding="utf-8",
    )
    return path


def test_namespace_extensions_and_fields(tmp_path: Path) -> None:
    parsed = parse_tcx(make_tcx(tmp_path / "run.tcx"))
    activity = parsed.activities[0]
    assert activity.activity_id == "2024-07-01T12:00:00Z"
    assert activity.total_distance_m == 40
    assert activity.lap_recorded_time_s == 20
    assert activity.total_elapsed_time_s == 20
    assert activity.average_hr_bpm == 145
    assert activity.maximum_hr_bpm == 150
    assert activity.gps_quality == "gps_complete"
    assert activity.hr_quality == "hr_complete"
    assert activity.trackpoints[0].cadence == 77
    assert activity.trackpoints[0].cadence_source == "cadence"
    assert activity.trackpoints[1].cadence == 79
    assert activity.trackpoints[1].cadence_source == "run_cadence_extension"
    assert activity.trackpoints[1].speed_mps == pytest.approx(2.1)


def test_missing_gps_and_hr_are_flagged_not_invented(tmp_path: Path) -> None:
    activity = parse_tcx(make_tcx(tmp_path / "missing.tcx", gps=False, hr=False)).activities[0]
    assert activity.gps_quality == "gps_missing"
    assert activity.hr_quality == "hr_missing"
    assert all(point.latitude is None and point.longitude is None for point in activity.trackpoints)
    assert all(point.heart_rate_bpm is None for point in activity.trackpoints)
    assert activity.distance_source == "device"


def test_sensor_quality_tolerates_sparse_missing_trackpoints() -> None:
    assert _quality("gps", 98, 100) == "gps_complete"
    assert _quality("gps", 94, 100) == "gps_partial"
    assert _quality("gps", 0, 100) == "gps_missing"


def test_zero_zero_position_is_invalid_but_preserved(tmp_path: Path) -> None:
    path = make_tcx(tmp_path / "zero.tcx")
    text = path.read_text().replace("40.7", "0").replace("-73.95", "0")
    path.write_text(text)
    activity = parse_tcx(path).activities[0]
    assert activity.gps_quality == "gps_missing"
    assert activity.trackpoints[0].latitude == 0
    assert activity.trackpoints[0].longitude == 0
    assert not activity.trackpoints[0].gps_valid
    assert "zero_zero_position" in activity.trackpoints[0].parse_flags


def test_local_timezone_observes_dst(tmp_path: Path) -> None:
    summer = parse_tcx(make_tcx(tmp_path / "summer.tcx")).activities[0]
    winter = parse_tcx(
        make_tcx(
            tmp_path / "winter.tcx",
            start="2024-01-01T12:00:00Z",
            end="2024-01-01T12:00:20Z",
            activity_id="2024-01-01T12:00:00Z",
        )
    ).activities[0]
    assert summer.start_time_local.utcoffset().total_seconds() == -4 * 3600
    assert winter.start_time_local.utcoffset().total_seconds() == -5 * 3600


def test_cp1252_notes_fallback_is_explicit(tmp_path: Path) -> None:
    path = make_tcx(tmp_path / "legacy.tcx")
    raw = path.read_bytes().replace(b"test run", b"runner\x92s notes")
    path.write_bytes(raw)
    parsed = parse_tcx(path)
    assert parsed.encoding == "cp1252_fallback"
    assert "declared_utf8_repaired_as_cp1252" in parsed.warnings
    assert parsed.activities[0].notes == "runner’s notes"

