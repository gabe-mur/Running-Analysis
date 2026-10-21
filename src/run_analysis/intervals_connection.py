"""Credentials and read-only original-file access; no imports or planner side effects."""

from __future__ import annotations

import base64
from datetime import date, datetime, timedelta, timezone
import gzip
import io
import json
import os
from pathlib import Path
import stat
import tempfile
import re
import zlib
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener


MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_FIT_BYTES = 50 * 1024 * 1024
# Intervals explicitly documents Python-urllib being blocked by its edge
# protection and recommends a browser-style User-Agent for API requests.
# https://forum.intervals.icu/t/api-access-to-intervals-icu/609
API_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/134.0.0.0 Safari/537.36"
)


class ConnectionProblem(Exception):
    """Only messages safe to return to the browser belong here."""


def _credential_path(root: Path) -> Path:
    path = root / "data" / "credentials" / "intervals.json"
    # Do not follow a redirected credential directory or secret file.
    if any(item.is_symlink() for item in (root / "data", path.parent, path)):
        raise ConnectionProblem("The credential path must not be a symbolic link.")
    return path


def _key(root: Path) -> str | None:
    try:
        path = _credential_path(root)
        try:
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        except FileNotFoundError:
            return None
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            metadata = os.fstat(handle.fileno())
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 4096:
                raise ConnectionProblem("The local credential file is invalid.")
            if os.name == "posix" and (metadata.st_mode & 0o077 or metadata.st_uid != os.getuid()):
                raise ConnectionProblem("The credential file must be owned by you and readable only by you (chmod 600).")
            payload = json.load(handle)
        if not isinstance(payload, dict) or not isinstance(payload.get("api_key"), str) or not payload["api_key"]:
            raise ConnectionProblem("The local credential file is invalid. Save a new API key.")
        return payload["api_key"]
    except ConnectionProblem:
        raise
    except Exception:
        raise ConnectionProblem("Could not read the local credential file. Check its permissions or save a new key.") from None


def connection_status(root: Path) -> dict:
    try:
        configured = bool(_key(root))
        return {"available": True, "configured": configured,
                "storage": "Local credential file", "automatic_import": False}
    except ConnectionProblem as exc:
        return {"available": False, "configured": False, "storage": "Local credential file",
                "automatic_import": False, "detail": str(exc)}


def save_key(root: Path, value: str) -> dict:
    value = value.strip()
    if not 8 <= len(value) <= 512 or not value.isascii() or any(c.isspace() or not c.isprintable() for c in value):
        raise ConnectionProblem("Enter a valid Intervals.icu API key (8–512 characters, without spaces).")
    try:
        path = _credential_path(root)
        path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        if os.name == "posix":
            path.parent.chmod(0o700)
        temporary = None
        try:
            # mkstemp creates mode 0600 before writing any credential bytes.
            descriptor, temporary = tempfile.mkstemp(prefix=".intervals-", dir=path.parent)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump({"api_key": value}, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)
        if _key(root) != value:
            raise ConnectionProblem("The API key could not be saved in the local credential file.")
    except ConnectionProblem:
        raise
    except Exception:
        raise ConnectionProblem("Could not save the local credential file. Check directory permissions.") from None
    return connection_status(root)


def remove_key(root: Path) -> dict:
    try:
        _credential_path(root).unlink(missing_ok=True)
    except ConnectionProblem:
        raise
    except Exception:
        raise ConnectionProblem("Could not remove the local credential file.") from None
    return connection_status(root)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward Authorization to a redirect destination.
        return None


def download_original_fit(root: Path, activity_id: str) -> bytes:
    """Read the original recording, never Intervals' regenerated ``fit-file``.

    Return normalized FIT bytes so gzip transport metadata cannot change the
    content hash. Does not persist/import data or trigger the planner.
    """
    if not isinstance(activity_id, str) or not re.fullmatch(r"i?[0-9]+", activity_id):
        raise ConnectionProblem("Invalid Intervals.icu activity ID.")
    key = _key(root)
    if not key:
        raise ConnectionProblem("Save your Intervals.icu API key first.")
    token = base64.b64encode(f"API_KEY:{key}".encode()).decode()
    request = Request(f"https://intervals.icu/api/v1/activity/{activity_id}/file",
        headers={"Authorization": f"Basic {token}", "Accept": "application/octet-stream",
                 "User-Agent": API_USER_AGENT})
    try:
        with build_opener(_NoRedirect()).open(request, timeout=20) as response:
            content = response.read(MAX_FIT_BYTES + 1)
        if len(content) > MAX_FIT_BYTES:
            raise ConnectionProblem("The Intervals.icu original file exceeds 50 MB.")
        if content.startswith(b"\x1f\x8b"):
            with gzip.GzipFile(fileobj=io.BytesIO(content)) as compressed:
                content = compressed.read(MAX_FIT_BYTES + 1)
            if len(content) > MAX_FIT_BYTES:
                raise ConnectionProblem("The expanded Intervals.icu original file exceeds 50 MB.")
    except HTTPError as exc:
        code = exc.code
        exc.close()
        message = {
            401: "Intervals.icu rejected authentication. Check your saved API key.",
            403: "Intervals.icu denied access to this original file (HTTP 403).",
            404: "Intervals.icu has no downloadable original file for this activity.",
            429: "Intervals.icu is rate limiting requests. Try again later.",
        }.get(code, "Intervals.icu could not download the original file. Try again later.")
        raise ConnectionProblem(message) from None
    except (gzip.BadGzipFile, EOFError, zlib.error):
        raise ConnectionProblem("Intervals.icu returned an invalid compressed original file.") from None
    except (URLError, TimeoutError, OSError):
        raise ConnectionProblem("Could not download the Intervals.icu original file securely.") from None
    if (len(content) < 14 or content[0] not in (12, 14) or content[8:12] != b".FIT"):
        raise ConnectionProblem("The Intervals.icu original recording is not a FIT file.")
    declared_length = content[0] + int.from_bytes(content[4:8], "little") + 2
    if len(content) != declared_length:
        raise ConnectionProblem("The Intervals.icu original FIT file is truncated or has an invalid length.")
    return content


def list_recent_activities(root: Path) -> list[dict]:
    """Read a 30-day overlap, including late-arriving Garmin activities."""
    key = _key(root)
    if not key:
        raise ConnectionProblem("Save your Intervals.icu API key first.")
    now = datetime.now(timezone.utc)
    query = urlencode({"oldest": (now - timedelta(days=30)).date().isoformat(),
                       "newest": (now + timedelta(days=1)).date().isoformat()})
    token = base64.b64encode(f"API_KEY:{key}".encode()).decode()
    request = Request(f"https://intervals.icu/api/v1/athlete/0/activities?{query}",
                      headers={"Authorization": f"Basic {token}", "Accept": "application/json",
                               "User-Agent": API_USER_AGENT})
    try:
        with build_opener(_NoRedirect()).open(request, timeout=15) as response:
            content = response.read(MAX_RESPONSE_BYTES + 1)
        if len(content) > MAX_RESPONSE_BYTES:
            raise ConnectionProblem("Intervals.icu returned too much data for this connection check.")
        activities = json.loads(content)
        if not isinstance(activities, list) or any(not isinstance(item, dict) for item in activities):
            raise ValueError("Unexpected response shape")
    except HTTPError as exc:
        # Never expose remote response bodies, exception URLs or auth headers.
        if exc.code == 401:
            message = "Intervals.icu rejected authentication. Check your saved API key."
        elif exc.code == 403:
            message = "Intervals.icu blocked the request or denied activity access (HTTP 403). This does not necessarily mean your API key is invalid."
        elif exc.code == 429:
            message = "Intervals.icu is rate limiting requests. Try again later."
        else:
            message = "Intervals.icu could not complete the connection check. Try again later."
        exc.close()
        raise ConnectionProblem(message) from None
    except (URLError, TimeoutError, OSError):
        raise ConnectionProblem("Could not reach Intervals.icu securely. Check your connection and try again.") from None
    except (ValueError, UnicodeError):
        raise ConnectionProblem("Intervals.icu returned an unexpected response. Try again later.") from None
    return activities


def check_connection(root: Path) -> dict:
    activities = list_recent_activities(root)
    return {"connected": True, "recent_activity_count": len(activities),
            "checked_at": datetime.now(timezone.utc).isoformat(), "automatic_import": False}


def list_wellness(root: Path, *, days=45) -> list[dict]:
    """Read daily wellness separately from activities; never change settings."""
    if type(days) is not int or not 1 <= days <= 366:
        raise ValueError('Wellness lookback must be 1–366 days')
    key=_key(root)
    if not key:
        raise ConnectionProblem('Save your Intervals.icu API key first.')
    now=datetime.now(timezone.utc).date()
    query=urlencode(dict(oldest=(now-timedelta(days=days-1)).isoformat(),newest=now.isoformat()))
    token=base64.b64encode(f'API_KEY:{key}'.encode()).decode()
    request=Request(f'https://intervals.icu/api/v1/athlete/0/wellness?{query}',
        headers={'Authorization':f'Basic {token}','Accept':'application/json','User-Agent':API_USER_AGENT})
    try:
        with build_opener(_NoRedirect()).open(request,timeout=15) as response:
            raw=response.read(MAX_RESPONSE_BYTES+1)
        if len(raw)>MAX_RESPONSE_BYTES:
            raise ValueError('Oversized response')
        rows=json.loads(raw)
        if not isinstance(rows,list) or any(not isinstance(r,dict) or not isinstance(r.get('id'),str) for r in rows):
            raise ValueError('Invalid wellness rows')
        dates=[date.fromisoformat(r['id']) for r in rows]
        if len(set(dates))!=len(dates) or any(not now-timedelta(days=days-1)<=at<=now for at in dates):
            raise ValueError('Duplicate or out-of-window dates')
    except HTTPError as exc:
        code=exc.code;exc.close()
        raise ConnectionProblem(f'Could not read Intervals.icu wellness data (HTTP {code}). Activity sync can continue.') from None
    except (URLError,TimeoutError,OSError):
        raise ConnectionProblem('Could not reach Intervals.icu wellness data securely. Activity sync can continue.') from None
    except (ValueError,UnicodeError):
        raise ConnectionProblem('Intervals.icu returned invalid or oversized wellness data. Activity sync can continue.') from None
    return rows
