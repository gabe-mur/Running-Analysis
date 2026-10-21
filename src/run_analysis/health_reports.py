"""Actual user-report timestamps, separate from automatic schedule refreshes."""
from datetime import datetime,timezone
from uuid import uuid4

from .athlete_projection import aware

STATUSES={'normal','little_tired','sick_or_recovering','pain_or_injury_concern'}


def record_health_report(connection,status,*,reported_at=None,source,report_key=None):
    at=aware(reported_at or datetime.now(timezone.utc))
    if status not in STATUSES or not source:
        raise ValueError('Explicit user health status and report source required')
    key=report_key or str(uuid4())
    connection.execute('''CREATE TABLE IF NOT EXISTS user_health_reports(
        report_key TEXT PRIMARY KEY,reported_at TEXT NOT NULL,status TEXT NOT NULL,source TEXT NOT NULL)''')
    row=(key,at.isoformat(),status,source)
    old=connection.execute('SELECT report_key,reported_at,status,source FROM user_health_reports WHERE report_key=?',(key,)).fetchone()
    if old is not None and tuple(old)!=row:
        raise ValueError('Immutable report retry changed')
    connection.execute('INSERT OR IGNORE INTO user_health_reports VALUES (?,?,?,?)',row)
    connection.commit()
    return key


def read_health_reports(connection,*,as_of):
    aware(as_of)
    if not connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='user_health_reports'").fetchone():
        return ()
    rows=[dict(report_key=r[0],reported_at=aware(datetime.fromisoformat(r[1])),status=r[2],source=r[3])
          for r in connection.execute('SELECT report_key,reported_at,status,source FROM user_health_reports')]
    return tuple(sorted((r for r in rows if r['reported_at']<=as_of),key=lambda r:(r['reported_at'],r['report_key'])))
