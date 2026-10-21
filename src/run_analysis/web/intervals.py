"""Local settings API. Credentials are never serialized in responses."""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from .. import intervals_connection as service
from ..intervals_sync import sync_runs
from .upload_service import UploadBusyError


def _local_request(request: Request, *, mutation: bool = False) -> None:
    origin = request.headers.get("origin")
    if ((origin is not None and origin != str(request.base_url).rstrip("/"))
            or request.headers.get("sec-fetch-site") == "cross-site"):
        raise HTTPException(403, "This connection can only be managed from the local app.")
    if mutation and request.headers.get("x-run-analysis-local") != "1":
        raise HTTPException(403, "Use the local app connection controls.")


def _problem(exc: service.ConnectionProblem) -> HTTPException:
    return HTTPException(400, str(exc))


def connection_router(root: Path, config_path: Path | None = None) -> APIRouter:
    router = APIRouter(prefix="/api/connections/intervals", tags=["connections"])

    @router.get("")
    def status(request: Request):
        _local_request(request)
        return service.connection_status(root)

    @router.put("")
    async def save(request: Request):
        _local_request(request, mutation=True)
        if request.headers.get("content-type", "").split(";", 1)[0].strip() != "application/json":
            raise HTTPException(415, "Send the connection settings as JSON.")
        content = bytearray()
        async for chunk in request.stream():
            content.extend(chunk)
            if len(content) > 4096:
                raise HTTPException(413, "Connection settings are too large.")
        # Parse explicitly so Pydantic validation errors cannot echo a secret
        # through their 'input' field (including malformed/non-string values).
        try:
            payload = json.loads(content)
        except (ValueError, UnicodeError):
            raise HTTPException(422, "Invalid connection settings.") from None
        if not isinstance(payload, dict) or set(payload) != {"api_key"} or not isinstance(payload["api_key"], str):
            raise HTTPException(422, "Provide only an API key as text.")
        try:
            return await run_in_threadpool(service.save_key, root, payload["api_key"])
        except service.ConnectionProblem as exc:
            raise _problem(exc) from None

    @router.delete("")
    def remove(request: Request):
        _local_request(request, mutation=True)
        try:
            return service.remove_key(root)
        except service.ConnectionProblem as exc:
            raise _problem(exc) from None

    @router.post("/check")
    def check(request: Request):
        _local_request(request, mutation=True)
        try:
            return service.check_connection(root)
        except service.ConnectionProblem as exc:
            raise _problem(exc) from None

    @router.post('/sync')
    def sync(request: Request):
        _local_request(request, mutation=True)
        try:
            return sync_runs(root, config_path or root / 'config.yaml')
        except UploadBusyError as exc:
            raise HTTPException(409, str(exc)) from None
        except service.ConnectionProblem as exc:
            raise _problem(exc) from None
        except (OSError, ValueError):
            raise HTTPException(500, 'Sync could not complete. Check the local configuration and try again.') from None

    return router
