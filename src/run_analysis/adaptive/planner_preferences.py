"""Dated athlete preferences, separate from physical load and accepted dose.

The runtime journal owns requests; an issued snapshot owns the applied revision.
Saving is not a stage reset or a claim that a new calendar has been issued.
"""
from dataclasses import dataclass, replace
from contextlib import nullcontext
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
import json
from math import isfinite


CENTERS = {'fewer': .25, 'balanced': .5, 'more': .75}
VERSION = 'adaptive_preferences_v1'


def payload(*, available_at, goal_long_miles, frequency_mode, parent_key,
            long_run_weekday=None, long_run_not_before=None):
    value = dict(version=VERSION, available_at=available_at.astimezone(timezone.utc).isoformat(),
        goal_long_miles=float(goal_long_miles), frequency_mode=frequency_mode,
        parent_key=parent_key, source='user:settings')
    # Automatic preserves the identity and meaning of existing revisions.
    if long_run_weekday is not None:
        if type(long_run_weekday) is not int or not 0 <= long_run_weekday <= 6 or type(long_run_not_before) is not date:
            raise ValueError('Valid weekday and frozen matching transition date required')
        value.update(long_run_weekday=long_run_weekday,
            long_run_not_before=long_run_not_before.isoformat())
    return value


def identity(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class PlannerPreferencesRevision:
    key: str
    available_at: datetime
    goal_long_miles: float
    frequency_mode: str
    parent_key: str
    long_run_weekday: int | None = None  # Monday=0; None leaves search unchanged.
    long_run_not_before: date | None = None  # Frozen transition, never a moving horizon.

    def __post_init__(self):
        if (not isinstance(self.available_at, datetime) or self.available_at.tzinfo is None
                or self.available_at.utcoffset() is None or isinstance(self.goal_long_miles, bool)
                or not isfinite(self.goal_long_miles) or not 0 < self.goal_long_miles <= 100
                or self.frequency_mode not in CENTERS or not self.parent_key):
            raise ValueError('Valid dated goal and frequency preference required')
        object.__setattr__(self, 'available_at', self.available_at.astimezone(timezone.utc))
        object.__setattr__(self, 'goal_long_miles', float(self.goal_long_miles))
        if (self.long_run_weekday is not None and (type(self.long_run_weekday) is not int
                or not 0 <= self.long_run_weekday <= 6
                or type(self.long_run_not_before) is not date
                or self.long_run_not_before.weekday() != self.long_run_weekday)
                or self.long_run_weekday is None and self.long_run_not_before is not None):
            raise ValueError('Valid weekday and frozen matching transition date required')
        if self.key != identity(self.payload):
            raise ValueError('Planner preference identity changed')

    @property
    def payload(self):
        return payload(available_at=self.available_at, goal_long_miles=self.goal_long_miles,
            frequency_mode=self.frequency_mode, parent_key=self.parent_key,
            long_run_weekday=self.long_run_weekday,long_run_not_before=self.long_run_not_before)

    @classmethod
    def create(cls, *, available_at, goal_long_miles, frequency_mode, parent_key,
               long_run_weekday=None, long_run_not_before=None):
        value = payload(available_at=available_at, goal_long_miles=goal_long_miles,
            frequency_mode=frequency_mode, parent_key=parent_key,
            long_run_weekday=long_run_weekday,long_run_not_before=long_run_not_before)
        return cls(identity(value), available_at, goal_long_miles, frequency_mode, parent_key,
            long_run_weekday,long_run_not_before)


def latest_request(db, *, as_of=None):
    if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='adaptive_preference_requests'").fetchone():
        return None
    rows = db.execute('SELECT key,payload FROM adaptive_preference_requests ORDER BY rowid DESC')
    for key, content in rows:
        value = json.loads(content)
        if value.pop('version') != VERSION or value.pop('source') != 'user:settings':
            raise ValueError('Unsupported planner preference revision')
        revision = PlannerPreferencesRevision(key, **dict(value,
            available_at=datetime.fromisoformat(value['available_at']),
            **({'long_run_not_before':date.fromisoformat(value['long_run_not_before'])}
                if 'long_run_not_before' in value else {})))
        if as_of is None or revision.available_at <= as_of:
            return revision
    return None


def queue_request(db, *, state, stage, patch, as_of, prior_program=(), manage_transaction=True):
    """Append a user request atomically; do not issue or edit a training stage."""
    if as_of <= state.as_of:
        raise ValueError('Preference availability must follow the current issued decision')
    if not manage_transaction and not db.in_transaction:
        raise ValueError('Caller-owned preference transaction required')
    previous = latest_request(db)
    applied = state.planner_preferences
    current = previous or applied
    goal = current.goal_long_miles if current else stage['goal_long_miles']
    mode = current.frequency_mode if current else 'balanced'
    weekday = current.long_run_weekday if current else None
    requested_day = patch.long_run_weekday if 'long_run_weekday' in patch.model_fields_set else weekday
    not_before = current.long_run_not_before if current else None
    parent = current.key if current else 'adopted:'+state.fingerprint
    if (goal, mode, weekday) == (patch.long_run_goal_miles, patch.run_frequency, requested_day):
        return None  # Complete unchanged form / retry is not a new policy event.
    if patch.expected_revision != parent:
        raise ValueError('Planner preferences changed since this form loaded. Reload Settings before saving.')
    if patch.long_run_goal_miles < stage['long_miles']:
        raise ValueError('The destination cannot be below the current long-run target. Reducing the active training dose needs a separate stage review, not a destination edit.')
    if previous and as_of <= previous.available_at:
        raise ValueError('Preference requests must advance their availability time')
    if requested_day != weekday:
        not_before = None
        if requested_day is not None:
            from zoneinfo import ZoneInfo
            today = max(state.context.decision,as_of.astimezone(ZoneInfo(state.timezone)).date())
            contracts = dict(state.contracts)
            next_longs = [date.fromisoformat(day) for day,key in prior_program
                if date.fromisoformat(day) >= today and key in contracts and contracts[key].purpose == 'long']
            if not next_longs:
                raise ValueError('No issued future long run to anchor this change. Refresh the plan before choosing a weekday.')
            anchor = min(next_longs)
            not_before = anchor+timedelta(days=(requested_day-anchor.weekday())%7)
    revision = PlannerPreferencesRevision.create(available_at=as_of,
        goal_long_miles=patch.long_run_goal_miles, frequency_mode=patch.run_frequency,
        parent_key=parent,long_run_weekday=requested_day,long_run_not_before=not_before)
    db.execute('CREATE TABLE IF NOT EXISTS adaptive_preference_requests(key TEXT PRIMARY KEY,payload TEXT NOT NULL)')
    with db if manage_transaction else nullcontext():
        if manage_transaction:
            db.execute('BEGIN IMMEDIATE')
        newest = latest_request(db)
        if newest != previous:
            raise ValueError('Concurrent preference edit; reload Settings')
        db.execute('INSERT INTO adaptive_preference_requests VALUES (?,?)',
            (revision.key, json.dumps(revision.payload, sort_keys=True, separators=(',', ':'))))
    return revision


def apply_request(harness, *, as_of):
    revision = latest_request(harness.sandbox.db, as_of=as_of)
    applied = getattr(harness, 'planner_preferences', None)
    if revision is None or revision == applied:
        return False
    if applied and revision.available_at < applied.available_at:
        raise ValueError('Cannot revert an applied planner preference')
    if revision.goal_long_miles < harness.stage.long_miles:
        raise ValueError('Requested destination is below the now-current long target; revise preferences')
    harness.planner_preferences = revision
    preference = replace(harness.frequency_contract.preference,
        center_fraction=CENTERS[revision.frequency_mode])
    harness.frequency_contract = replace(harness.frequency_contract, preference=preference)
    return True


def progression_stage(stage, revision):
    """Review future growth against the requested destination without a rebootstrap."""
    return replace(stage, goal_long_miles=revision.goal_long_miles) if revision else stage


def long_day_allowed(revision, day):
    return (revision is None or revision.long_run_weekday is None or
        day.weekday() == revision.long_run_weekday and day >= revision.long_run_not_before)


def long_day_admission(state, planned):
    """User scheduling constraint only. Never a recovery claim or a fatigue weight."""
    revision = state.planner_preferences
    if revision is None or revision.long_run_weekday is None:
        return None
    from zoneinfo import ZoneInfo
    zone = ZoneInfo(state.timezone)
    violations = tuple(r.key for r in planned if r.role == 'long'
        and not long_day_allowed(revision,r.start.astimezone(zone).date()))
    return dict(eligible=not violations,violations=violations,
        weekday=revision.long_run_weekday,not_before=revision.long_run_not_before.isoformat(),
        source='user:settings',recovery_override=False)
