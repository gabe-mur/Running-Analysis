"""Passive, source-labelled daily context; never a readiness score or HR setting.

Fetched on explicit sync, read locally by diagnostics. No networking from page
reads, no assumptions about missing nights, and no blend with exercise DFA.
"""
from datetime import date, datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import tempfile

from . import intervals_connection as remote
from .privacy import private_directory, private_file

FIELDS = {'hrv':('hrv_ms',0,1000),'restingHR':('resting_hr_bpm',0,250),
    'sleepSecs':('sleep_seconds',-1,86400),'sleepScore':('sleep_score',-1,100),
    'vo2max':('reported_vo2max_ml_kg_min',0,200)}


def _path(root):
    root=Path(root)
    target=root/'data'/'intervals-wellness.json'
    if (root/'data').is_symlink() or target.is_symlink():
        raise remote.ConnectionProblem('The local wellness cache must not be a symbolic link.')
    return target


def clean_rows(rows):
    result=[]
    for row in sorted(rows,key=lambda r:r['id']):
        at=date.fromisoformat(row['id']).isoformat()
        measured={}
        for key,(name,lo,hi) in FIELDS.items():
            value=row.get(key)
            if isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) and lo<value<=hi:
                measured[name]=value
        if measured:
            result.append(dict(date=at,**measured))
    return result


def read_wellness(root):
    target=_path(root)
    try:
        if not target.exists() or target.stat().st_size>remote.MAX_RESPONSE_BYTES:
            return None
        if os.name=='posix' and (target.stat().st_mode&0o077 or target.stat().st_uid!=os.getuid()):
            return None
        value=json.loads(target.read_text())
        if value.get('version')!=1 or not isinstance(value.get('series'),list):
            return None
        dates=[]
        for row in value['series']:
            if not isinstance(row,dict):return None
            dates.append(date.fromisoformat(row['date']))
            if any(not isinstance(v,(int,float)) or isinstance(v,bool) or not math.isfinite(v)
                   for k,v in row.items() if k!='date'):
                return None
        if len(set(dates))!=len(dates):return None
        return value
    except (OSError,ValueError,TypeError,AttributeError,KeyError):
        return None


def refresh_wellness(root, *, days=45):
    rows=remote.list_wellness(root,days=days)
    now=datetime.now(timezone.utc)
    target=_path(root)
    previous=read_wellness(root) or {}
    start=(now.date()-timedelta(days=days-1)).isoformat()
    cutoff=(now.date()-timedelta(days=365)).isoformat()
    merged={r['date']:r for r in previous.get('series',[]) if cutoff<=r['date']<start}
    merged.update({r['date']:r for r in clean_rows(rows)})
    result=dict(version=1,source='Intervals.icu wellness API',upstream_device='not specified by this endpoint',
        retrieved_at=now.isoformat(),series=[merged[k] for k in sorted(merged)],
        interpretation='Daily reported values, separate from activity RR/DFA; no personal HRV baseline or readiness inference.',
        planner_influence=False,fitness_influence=False)
    private_directory(target.parent)
    descriptor,temporary=tempfile.mkstemp(prefix='.wellness-',dir=target.parent)
    try:
        with os.fdopen(descriptor,'w') as stream:
            json.dump(result,stream,allow_nan=False);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,target);private_file(target)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)
    return dict(status='saved',source=result['source'],days_with_data=len(result['series']),
        coverage={name:sum(r.get(name) is not None for r in result['series']) for name,_,_ in FIELDS.values()},
        planner_influence=False,fitness_influence=False)
