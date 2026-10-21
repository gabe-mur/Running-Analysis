"""User-triggered original-FIT sync through the existing upload pipeline."""
from contextlib import closing
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import tempfile

from . import intervals_connection as remote
from .config import load_config, resolve_project_path
from .db import connect, initialize
from .privacy import private_directory, private_file
from .web.upload_service import (
    UploadPayload, complete_source_for_digest, run_upload_pipeline, upload_operation,
)

MAX_SYNC_FILES = 25
MAX_SYNC_BYTES = 100 * 1024 * 1024


def _cache_path(root):
    path = root / 'data' / 'intervals-sync.json'
    if path.is_symlink():
        raise remote.ConnectionProblem('The local sync history must not be a symbolic link.')
    return path


def _read_cache(root):
    path = _cache_path(root)
    if not path.exists():
        return {}
    try:
        if path.stat().st_size > 2 * 1024 * 1024:
            return {}
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            return {}
        return {key: value for key, value in data.items()
                if re.fullmatch(r'i?[0-9]+', key) and isinstance(value, str)
                and re.fullmatch(r'[0-9a-f]{64}', value)}
    except (OSError, ValueError):
        # Lost history only costs re-downloads; the content guard still applies.
        return {}


def _save_cache(root, data):
    path = _cache_path(root)
    private_directory(path.parent)
    descriptor, temporary = tempfile.mkstemp(prefix='.intervals-sync-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w') as handle:
            json.dump(data, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        private_file(path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sync_runs(root: Path, config_path: Path) -> dict:
    with upload_operation(root):
        return _sync_runs(root, config_path)


def _sync_runs(root, config_path):
    activities = remote.list_recent_activities(root)
    runs = {}
    for activity in activities:
        if activity.get('type') not in ('Run', 'TrailRun', 'VirtualRun'):
            continue
        identifier = activity.get('id')
        if not isinstance(identifier, str) or not re.fullmatch(r'i?[0-9]+', identifier):
            raise remote.ConnectionProblem('Intervals.icu returned an invalid activity ID.')
        runs[identifier] = activity
    report = dict(checked_runs=len(runs), new_runs=0, already_synced=0,
                  downloaded_files=0, remaining_runs=0, errors=[], upload=None)
    # Optional daily context is isolated from activity success and never
    # triggers model fitting, HR-setting changes or a calendar refresh.
    from .wellness_signals import refresh_wellness
    try:
        report['wellness']=refresh_wellness(root)
    except (remote.ConnectionProblem,OSError,ValueError):
        report['wellness']=dict(status='unavailable',detail='Daily wellness data could not refresh; saved data was retained.')
    if not runs:
        return report
    config = load_config(config_path)
    database = resolve_project_path(root, config['paths']['database'])
    cache = _read_cache(root)
    original_cache = dict(cache)
    pending = []
    with closing(connect(database)) as connection:
        initialize(connection)
        before_count = connection.execute('SELECT count(*) FROM activities').fetchone()[0]
        for identifier, activity in sorted(runs.items(), key=lambda pair: str(pair[1].get('start_date_local', ''))):
            digest = cache.get(identifier)
            if digest and complete_source_for_digest(connection, digest, fit=True) is not None:
                report['already_synced'] += 1
            else:
                pending.append((identifier, activity))
    payloads = []
    downloaded = []
    byte_count = 0
    for index, (identifier, activity) in enumerate(pending):
        # Never synthesize a FIT from TCX/GPX: that would lose Garmin evidence.
        file_type = activity.get('file_type')
        if file_type and str(file_type).casefold() not in ('fit', 'fit.gz'):
            report['errors'].append('A running activity has no original FIT; upload its TCX manually.')
            continue
        if len(payloads) >= MAX_SYNC_FILES:
            report['remaining_runs'] = len(pending) - index
            break
        try:
            content = remote.download_original_fit(root, identifier)
        except remote.ConnectionProblem as exc:
            report['errors'].append(str(exc))
            continue
        if byte_count + len(content) > MAX_SYNC_BYTES:
            report['remaining_runs'] = len(pending) - index
            break
        byte_count += len(content)
        payloads.append(UploadPayload(f'intervals-{identifier}.fit', content))
        downloaded.append((identifier, sha256(content).hexdigest()))
    report['downloaded_files'] = len(payloads)
    if payloads:
        # Batch once so multiple new files request only one model/planner update.
        result = run_upload_pipeline(root, config_path, payloads, refresh_plan=False)
        report['upload'] = result.model_dump(mode='json')
        for (identifier, digest), item in zip(downloaded, result.files, strict=True):
            if item.status != 'failed' and item.activity_ids:
                cache[identifier] = digest
        with closing(connect(database)) as connection:
            report['new_runs'] = connection.execute('SELECT count(*) FROM activities').fetchone()[0] - before_count
    if cache != original_cache:
        # Keep the bounded recent overlap, rather than an unbounded history file.
        cache = {key: value for key, value in cache.items() if key in runs}
        _save_cache(root, cache)
    return report
