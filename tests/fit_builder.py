"""A minimal FIT writer, used only to build fixtures for the FIT reader tests.

fitdecode reads but does not write, and a FIT test that mocks the decoder would
verify nothing about the unit conversions -- which is the entire risk in this
format. So the fixtures are real binary FIT files, encoded here with raw field
values exactly as a device would store them: semicircles, the 1989 epoch,
one-sided cadence, and the (metres + 500) * 5 altitude scaling.
"""

from __future__ import annotations

from datetime import datetime, timezone
import struct

FIT_EPOCH = datetime(1989, 12, 31, tzinfo=timezone.utc)
DEGREES_TO_SEMICIRCLES = (2**31) / 180.0

ENUM, UINT8, UINT16, UINT32, SINT32 = 0x00, 0x02, 0x84, 0x86, 0x85
_SIZES = {ENUM: 1, UINT8: 1, UINT16: 2, UINT32: 4, SINT32: 4}
_FORMATS = {ENUM: "B", UINT8: "B", UINT16: "H", UINT32: "I", SINT32: "i"}

_CRC_TABLE = (
    0x0000, 0xCC01, 0xD801, 0x1400, 0xF001, 0x3C00, 0x2800, 0xE401,
    0xA001, 0x6C00, 0x7800, 0xB401, 0x5000, 0x9C01, 0x8801, 0x4400,
)


def fit_crc(data: bytes, crc: int = 0) -> int:
    for byte in data:
        for nibble in (byte & 0xF, (byte >> 4) & 0xF):
            table = _CRC_TABLE[crc & 0xF]
            crc = (crc >> 4) & 0x0FFF
            crc = crc ^ table ^ _CRC_TABLE[nibble]
    return crc & 0xFFFF


def fit_timestamp(moment: datetime) -> int:
    return int((moment - FIT_EPOCH).total_seconds())


def semicircles(degrees: float) -> int:
    return int(round(degrees * DEGREES_TO_SEMICIRCLES))


class FitWriter:
    """Builds a FIT byte stream one message at a time."""

    def __init__(self) -> None:
        self._data = bytearray()
        self._definitions: dict[int, list[tuple[int, int]]] = {}
        self._local_for_global: dict[int, int] = {}

    def _local_type(self, global_message: int, fields: list[tuple[int, int]]) -> int:
        signature = (global_message, tuple(fields))
        existing = self._local_for_global.get(global_message)
        if existing is not None and self._definitions[existing] == fields:
            return existing
        local = len(self._local_for_global) % 16
        self._local_for_global[global_message] = local
        self._definitions[local] = fields
        header = 0x40 | local
        body = struct.pack("<BBHB", 0, 0, global_message, len(fields))
        for number, base_type in fields:
            body += struct.pack("<BBB", number, _SIZES[base_type], base_type)
        self._data += bytes([header]) + body
        del signature
        return local

    def message(self, global_message: int, values: dict[int, tuple[int, int]]) -> None:
        """Append one data message.

        ``values`` maps field number to ``(base_type, raw_value)`` -- raw, so
        the test states the on-disk encoding and the reader has to do the
        conversion rather than being handed the answer.
        """

        fields = [(number, base_type) for number, (base_type, _) in sorted(values.items())]
        local = self._local_type(global_message, fields)
        payload = bytes([local])
        for number, base_type in fields:
            payload += struct.pack("<" + _FORMATS[base_type], values[number][1])
        self._data += payload

    def build(self) -> bytes:
        header = struct.pack("<BBHI4s", 12, 0x10, 2140, len(self._data), b".FIT")
        body = header + bytes(self._data)
        return body + struct.pack("<H", fit_crc(body))


# Global message numbers used by the fixtures.
FILE_ID, SESSION, LAP, RECORD, EVENT = 0, 18, 19, 20, 21


def build_run(
    path,
    *,
    start: datetime,
    points: int = 12,
    latitude: float = 40.71,
    longitude: float = -73.99,
    altitude_m: float = 30.0,
    heart_rate: int = 150,
    one_sided_cadence: int = 82,
    sport: int = 1,
    pauses: list[tuple[int, int]] | None = None,
    include_lap: bool = True,
    seconds_per_point: int = 10,
    metres_per_point: float = 30.0,
) -> bytes:
    """Write a small but structurally complete running FIT file."""

    writer = FitWriter()
    writer.message(
        FILE_ID,
        {
            0: (ENUM, 4),  # activity
            1: (UINT16, 1),  # Garmin
            2: (UINT16, 3121),
            4: (UINT32, fit_timestamp(start)),
        },
    )
    for index in range(points):
        moment = start.fromtimestamp(start.timestamp() + index * seconds_per_point, tz=timezone.utc)
        writer.message(
            RECORD,
            {
                253: (UINT32, fit_timestamp(moment)),
                0: (SINT32, semicircles(latitude + index * 0.0001)),
                1: (SINT32, semicircles(longitude)),
                2: (UINT16, int(round((altitude_m + 500.0) * 5))),
                3: (UINT8, heart_rate),
                4: (UINT8, one_sided_cadence),
                5: (UINT32, int(round(index * metres_per_point * 100))),
                6: (UINT16, 3000),
            },
        )
    for stop_offset, start_offset in pauses or []:
        for offset, event_type in ((stop_offset, 4), (start_offset, 0)):  # stop_all / start
            moment = start.fromtimestamp(start.timestamp() + offset, tz=timezone.utc)
            writer.message(
                EVENT,
                {253: (UINT32, fit_timestamp(moment)), 0: (ENUM, 0), 1: (ENUM, event_type)},
            )
    duration = (points - 1) * seconds_per_point
    if include_lap:
        writer.message(
            LAP,
            {
                253: (UINT32, fit_timestamp(start)),
                2: (UINT32, fit_timestamp(start)),
                7: (UINT32, duration * 1000),
                8: (UINT32, duration * 1000),
                9: (UINT32, int(round((points - 1) * metres_per_point * 100))),
                11: (UINT16, 120),
                15: (UINT8, heart_rate),
                16: (UINT8, heart_rate + 12),
            },
        )
    writer.message(
        SESSION,
        {
            253: (UINT32, fit_timestamp(start)),
            2: (UINT32, fit_timestamp(start)),
            5: (ENUM, sport),
            7: (UINT32, duration * 1000),
            8: (UINT32, duration * 1000),
            9: (UINT32, int(round((points - 1) * metres_per_point * 100))),
            11: (UINT16, 120),
            16: (UINT8, heart_rate),
            17: (UINT8, heart_rate + 12),
        },
    )
    payload = writer.build()
    path.write_bytes(payload)
    return payload
