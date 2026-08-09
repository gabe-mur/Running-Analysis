"""ISO-8601 parsing shared by the text-based parsers.

Kept apart from any one format because the naive-timestamp rule below is a
data-provenance decision, not a parsing detail, and both readers must apply it
the same way.
"""

from __future__ import annotations

from datetime import datetime, timezone


def parse_datetime(value: str | None, warnings: list[str], field_name: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        warnings.append(f"malformed_{field_name}")
        return None
    if parsed.tzinfo is None:
        # TCX track and lap timestamps are UTC. Some Smashrun Activity/Id values
        # omit the suffix while retaining the UTC clock time.
        warnings.append(f"naive_{field_name}_treated_as_utc")
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
