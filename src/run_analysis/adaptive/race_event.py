"""Explicit event outcome, separate from a race objective or training role.

A calendar date is not completion evidence. This journal never fabricates an
activity and never rewrites an issued decision. Corrections are new revisions.
"""
from dataclasses import dataclass
from datetime import date, datetime, timezone
import json
from hashlib import sha256


@dataclass(frozen=True)
class RaceEventResolution:
    key: str
    race_key: str
    status: str
    available_at: datetime
    physical_key: str | None
    parent_key: str
    effective_on: date

    def __post_init__(self):
        if self.available_at.tzinfo is None or not self.race_key or not self.parent_key or type(self.effective_on) is not date:
            raise ValueError('Dated, revision-linked race outcome required')
        object.__setattr__(self, 'available_at', self.available_at.astimezone(timezone.utc))
        if self.status not in {'recorded', 'did_not_start'} or (self.status == 'recorded') != bool(self.physical_key):
            raise ValueError('Select an uploaded race activity or did not start')
        if self.key != self.identity(self.payload):
            raise ValueError('Race outcome identity changed')

    @property
    def payload(self):
        return dict(version='race_event_v1', race_key=self.race_key, status=self.status,
            available_at=self.available_at.isoformat(), physical_key=self.physical_key,
            parent_key=self.parent_key,effective_on=self.effective_on.isoformat())

    @staticmethod
    def identity(payload):
        return sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    @classmethod
    def create(cls, *, race_key, status, available_at, physical_key=None, parent_key,effective_on):
        value = dict(version='race_event_v1', race_key=race_key, status=status,
            available_at=available_at.astimezone(timezone.utc).isoformat(),
            physical_key=physical_key, parent_key=parent_key,effective_on=effective_on.isoformat())
        return cls(cls.identity(value), race_key, status, available_at, physical_key, parent_key,effective_on)


def latest_resolution(db, race_key, *, as_of=None):
    if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='adaptive_race_events'").fetchone():
        return None
    latest = None
    parent = None
    for key, content in db.execute('SELECT key,payload FROM adaptive_race_events ORDER BY rowid'):
        value = json.loads(content)
        if value.pop('version') != 'race_event_v1':
            raise ValueError('Unsupported race outcome journal')
        row = RaceEventResolution(key, **dict(value, available_at=datetime.fromisoformat(value['available_at']),
            effective_on=date.fromisoformat(value['effective_on'])))
        if row.race_key != race_key:
            continue
        if row.parent_key != (parent.key if parent else race_key) or (
                parent and row.available_at <= parent.available_at):
            raise ValueError('Race outcome journal chain changed')
        parent = row
        if as_of is None or row.available_at <= as_of:
            latest = row
    return latest


def record_resolution(db, *, revision, history, timezone_name, status, physical_key,
                      expected_revision, as_of):
    from zoneinfo import ZoneInfo
    if revision is None or revision.goal is None:
        raise ValueError('No applied race to reconcile')
    if as_of.astimezone(ZoneInfo(timezone_name)).date() < revision.race_date:
        raise ValueError('The race date has not arrived')
    previous = latest_resolution(db, revision.key)
    parent = previous.key if previous else revision.key
    if expected_revision != parent:
        raise ValueError('Race outcome changed; reload the form')
    if as_of <= revision.available_at or previous and as_of <= previous.available_at:
        raise ValueError('Race outcome must follow its goal and prior report')
    if status == 'recorded':
        activity = next((r for r in history if r.key == physical_key), None)
        if (activity is None or activity.source != 'observed' or activity.available_at > as_of
                or activity.end > as_of or activity.start.astimezone(ZoneInfo(timezone_name)).date() != revision.race_date):
            raise ValueError('Choose an available original activity from the race date')
    row = RaceEventResolution.create(race_key=revision.key, status=status, physical_key=physical_key,
        available_at=as_of, parent_key=parent,effective_on=as_of.astimezone(ZoneInfo(timezone_name)).date())
    if previous and (previous.status, previous.physical_key) == (status, physical_key):
        return previous
    db.execute('CREATE TABLE IF NOT EXISTS adaptive_race_events(key TEXT PRIMARY KEY,payload TEXT NOT NULL)')
    with db:
        db.execute('BEGIN IMMEDIATE')
        if latest_resolution(db, revision.key) != previous:
            raise ValueError('Concurrent race outcome change')
        db.execute('INSERT INTO adaptive_race_events VALUES (?,?)',
            (row.key, json.dumps(row.payload, sort_keys=True, separators=(',', ':'))))
    return row


def known_resolution(harness, revision, as_of):
    if revision is None or revision.goal is None:
        return None
    return latest_resolution(harness.sandbox.db, revision.key, as_of=as_of)


def resolution_history(harness, as_of):
    revisions={h.revision.key:h.revision for h in getattr(harness,'race_goal_history',()) if h.applied_at<=as_of}
    current=getattr(harness,'race_goal',None)
    if current is not None:
        revisions[current.key]=current
    rows=(latest_resolution(harness.sandbox.db,key,as_of=as_of) for key in sorted(revisions))
    return tuple(row for row in rows if row is not None)


def physical_keys(owner):
    rows=getattr(owner,'race_event_history',())
    current=getattr(owner,'race_event',None)
    return frozenset(r.physical_key for r in (*rows,*((current,) if current else ())) if r.physical_key)


def resolution_for(owner,revision):
    if revision is None:return None
    current=getattr(owner,'race_event',None)
    if current and current.race_key==revision.key:return current
    return next((r for r in getattr(owner,'race_event_history',()) if r.race_key==revision.key),None)


def no_race_recovery(revision, resolution, day):
    return bool(resolution and resolution.race_key == revision.key
        and resolution.status == 'did_not_start' and day > revision.race_date and day >= resolution.effective_on)


def preparation_active(owner, day):
    """A reconciled non-start has no race build/return phase to progress."""
    revision=getattr(owner,'race_goal',None)
    return bool(revision and revision.goal and revision.phase(day)!='complete'
        and not no_race_recovery(revision,resolution_for(owner,revision),day))


def training_fraction(revision, resolution, day):
    return 1. if no_race_recovery(revision, resolution, day) else revision.training_fraction(day)


def purpose_fraction(revision, resolution, day, purpose):
    return 1. if no_race_recovery(revision, resolution, day) else revision.purpose_fraction(day, purpose)


def dated_purpose_fraction(owner,day,purpose):
    """Use the same applied-date history as the volume target overlay."""
    applications=getattr(owner,'race_goal_history',())
    revision=(next((r.revision for r in reversed(applications) if r.effective_on<=day),None)
        if applications else getattr(owner,'race_goal',None))
    return purpose_fraction(revision,resolution_for(owner,revision),day,purpose) if revision and revision.goal else 1.
