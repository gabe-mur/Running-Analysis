"""Bounded cache for invariant database reads, never decisions or readiness.

File-backed autocommit or explicitly verified query-only snapshots: arbitrary
transactions must not borrow a different reader's result.
Any database/WAL replacement or write invalidates entries. Check again after
building so a concurrent write cannot publish a result under the wrong version.
Callers receive copies, not shared mutable models. No TTL or rounded clock is
used; time-dependent calculations remain outside this cache.
"""
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from pathlib import Path
from threading import RLock


_ENTRIES = OrderedDict()
_LOCK = RLock()
_LIMIT = 128
_SNAPSHOTS = ContextVar('verified_read_snapshots',default={})


def _file_version(connection):
    filename = connection.execute("PRAGMA database_list").fetchone()[2]
    if not filename:
        return None
    path = Path(filename).resolve()
    def stamp(file):
        try:
            stat = file.stat()
            # chmod/permission checks on connect change ctime but not evidence.
            return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        except FileNotFoundError:
            return None
    database = stamp(path)
    return (str(path), database, stamp(Path(f"{path}-wal"))) if database else None


def _version(connection):
    verified=_SNAPSHOTS.get().get(id(connection))
    if connection.in_transaction and verified is None:
        return None
    current=_file_version(connection)
    return current if verified is None or current==verified else None


@contextmanager
def invariant_snapshot(connection):
    """Pin a read snapshot, borrowing cache entries only for its verified version.

    Ordinary caller transactions still never borrow the cache. If a commit races
    snapshot creation or later reading, fall back to that snapshot's own reads.
    """
    if connection.in_transaction:
        raise ValueError('A fresh read connection is required')
    version=_version(connection)
    query_only=connection.execute('PRAGMA query_only').fetchone()[0]
    token=None
    try:
        connection.execute('PRAGMA query_only=ON')
        connection.execute('BEGIN')
        connection.execute('SELECT count(*) FROM sqlite_master').fetchone()  # Pin SQLite's snapshot.
        pinned=_file_version(connection)
        if version is not None and version==pinned:
            token=_SNAPSHOTS.set({**_SNAPSHOTS.get(),id(connection):version})
        yield connection
    finally:
        if token is not None:
            _SNAPSHOTS.reset(token)
        connection.rollback()
        connection.execute(f'PRAGMA query_only={int(query_only)}')


def invariant_read(connection, key, build, *, cacheable=lambda result: True):
    version = _version(connection)
    if version is None:
        return build()
    cache_key = (version, key)
    with _LOCK:
        if cache_key in _ENTRIES:
            result = deepcopy(_ENTRIES[cache_key])
            _ENTRIES.move_to_end(cache_key)
            if _version(connection) == version:
                return result
    result = build()
    if cacheable(result) and _version(connection) == version:
        with _LOCK:
            _ENTRIES[cache_key] = deepcopy(result)
            _ENTRIES.move_to_end(cache_key)
            while len(_ENTRIES) > _LIMIT:
                _ENTRIES.popitem(last=False)
    return result
