"""Disposable local HR-load summaries. Never planner state or raw run evidence."""
from contextlib import closing
from functools import lru_cache
from hashlib import sha256
import json
import os
from pathlib import Path
import sqlite3
import sys


@lru_cache(maxsize=1)
def implementation_key():
    root=Path(__file__).parent
    files=('derived_load_cache.py','progress_training_load.py','processing.py',
        'movement.py','geo.py','training_signals.py','response_model.py',
        'continuous_training_dose.py','workload_accounting.py','signal_structure.py')
    try:
        return sha256(b''.join((root/name).read_bytes() for name in files)+
            repr(sys.version_info[:3]).encode()).hexdigest()
    except OSError:
        return None  # Installed without inspectable sources: compute normally.


def trace_key(key):
    implementation=implementation_key()
    if implementation is None:
        return None
    return sha256(json.dumps((implementation,key),sort_keys=True,
        separators=(',',':'),allow_nan=False).encode()).hexdigest()


def cache_path(database):
    return Path(database).parent/'.hr-load-cache.sqlite'


def read_traces(database):
    """Failure/corruption is a miss; compute normally rather than fail a page."""
    try:
        with closing(sqlite3.connect(cache_path(database).resolve().as_uri()+'?mode=ro',uri=True)) as db:
            rows=db.execute('SELECT key,payload,digest FROM traces').fetchall()
        return {key:json.loads(payload) for key,payload,digest in rows
            if sha256(payload.encode()).hexdigest()==digest}
    except (OSError,sqlite3.Error,ValueError,TypeError):
        return {}


def store_traces(database,values):
    if not values:
        return
    try:
        path=cache_path(database)
        try:
            os.close(os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600))
        except FileExistsError:
            pass
        with closing(sqlite3.connect(path,timeout=.1)) as db:
            with db:
                db.execute('CREATE TABLE IF NOT EXISTS traces(key TEXT PRIMARY KEY,payload TEXT NOT NULL,digest TEXT NOT NULL)')
                for key,value in values.items():
                    # Only the exact fields used by recorded_cardio, not HR samples.
                    payload=json.dumps({name:value[name] for name in
                        ('elapsed_seconds','covered_seconds','load_bounds_within_reference')},
                        sort_keys=True,separators=(',',':'),allow_nan=False)
                    db.execute('INSERT OR REPLACE INTO traces VALUES (?,?,?)',
                        (key,payload,sha256(payload.encode()).hexdigest()))
                db.execute('DELETE FROM traces WHERE rowid NOT IN (SELECT rowid FROM traces ORDER BY rowid DESC LIMIT 4096)')
    except (OSError,sqlite3.Error,ValueError,KeyError,TypeError):
        pass  # Optional acceleration must never prevent analysis.
