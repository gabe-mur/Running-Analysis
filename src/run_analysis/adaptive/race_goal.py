"""Durable, explicitly requested race objectives; never an inferred race clearance.

The request journal owns edits. Only a successfully published decision owns the
applied revision. Dated taper accounting is independent of the search horizon.
"""
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
import json
from zoneinfo import ZoneInfo

from run_analysis.race_preparation import RacePreparationGoal, destination
from run_analysis.program_delivery import TargetSegment

VERSION = 'dated_race_objective_v1'


@dataclass(frozen=True)
class RaceGoalRevision:
    key: str
    available_at: datetime
    distance: str
    race_date: date | None
    preparation_mode: str
    parent_key: str
    general_long_goal_miles: float | None = None

    def __post_init__(self):
        if self.available_at.tzinfo is None or not self.parent_key:
            raise ValueError('Aware availability and parent revision required')
        object.__setattr__(self, 'available_at', self.available_at.astimezone(timezone.utc))
        if self.distance == 'general_fitness':
            if self.race_date is not None or self.preparation_mode != 'balanced':
                raise ValueError('General fitness has no race date or preparation mode')
        else:
            RacePreparationGoal(self.distance, self.race_date, self.preparation_mode)
        if self.general_long_goal_miles is not None:
            from math import isfinite
            if not isfinite(self.general_long_goal_miles) or self.general_long_goal_miles <= 0:
                raise ValueError('Positive retained general-fitness destination required')
        if self.key != self.identity(self.payload):
            raise ValueError('Race objective identity changed')

    @staticmethod
    def identity(value):
        return sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    @property
    def payload(self):
        value = dict(version=VERSION, source='user:settings',
            available_at=self.available_at.isoformat(), distance=self.distance,
            race_date=self.race_date.isoformat() if self.race_date else None,
            preparation_mode=self.preparation_mode, parent_key=self.parent_key)
        if self.general_long_goal_miles is not None:
            value['general_long_goal_miles']=self.general_long_goal_miles
        return value

    @classmethod
    def create(cls, *, available_at, distance, race_date, preparation_mode, parent_key, general_long_goal_miles=None):
        value = dict(version=VERSION, source='user:settings',
            available_at=available_at.astimezone(timezone.utc).isoformat(), distance=distance,
            race_date=race_date.isoformat() if race_date else None,
            preparation_mode=preparation_mode, parent_key=parent_key)
        if general_long_goal_miles is not None:
            value['general_long_goal_miles']=general_long_goal_miles
        return cls(cls.identity(value), available_at, distance, race_date, preparation_mode, parent_key,general_long_goal_miles)

    @property
    def goal(self):
        return (None if self.distance == 'general_fitness' else
                RacePreparationGoal(self.distance, self.race_date, self.preparation_mode))

    @property
    def taper_start(self):
        return self.race_date - timedelta(days=self.goal.profile.taper_days) if self.goal else None

    def phase(self, day):
        if not self.goal:
            return 'general_fitness'
        return 'build' if day < self.taper_start else 'taper' if day < self.race_date else (
            'race' if day == self.race_date else
            'post_race' if day <= self.race_date+timedelta(days=self.return_days) else 'complete')

    @property
    def return_days(self):
        # A reduced-dose review period, not a hard rest or recovery deadline.
        return {'5k': 3, '10k': 5, 'half_marathon': 10, 'marathon': 21}[self.distance]

    def training_fraction(self, day):
        # AS140: a gradual reduction to 50%, not a fitted physiological boundary.
        # The race itself is separate physical work, never training-volume credit.
        if self.phase(day) == 'post_race':
            return .5 + .5 * (day-self.race_date).days / self.return_days
        if self.phase(day) != 'taper':
            return 0. if self.phase(day) == 'race' else 1.
        elapsed = (day-self.taper_start).days + 1
        return 1. - .5 * elapsed / self.goal.profile.taper_days

    def purpose_fraction(self, day, purpose):
        if not self.goal or day < self.taper_start or self.phase(day) == 'complete':
            return 1.
        # Stop demanding full long/quality maintenance through the event. Actual
        # post-event load/readiness governs return, not an automatic catch-up debt.
        return 0.


def latest_request(db, *, as_of=None):
    if not db.execute("SELECT 1 FROM sqlite_master WHERE name='adaptive_race_requests' AND type='table'").fetchone():
        return None
    for key, content in db.execute('SELECT key,payload FROM adaptive_race_requests ORDER BY rowid DESC'):
        value = json.loads(content)
        if value.pop('version') != VERSION or value.pop('source') != 'user:settings':
            raise ValueError('Unsupported race request')
        revision = RaceGoalRevision(key, **dict(value,
            available_at=datetime.fromisoformat(value['available_at']),
            race_date=date.fromisoformat(value['race_date']) if value['race_date'] else None))
        if as_of is None or revision.available_at <= as_of:
            return revision
    return None


def queue_request(db, *, state, draft, expected_revision, as_of, general_long_goal_miles=None,manage_transaction=True,
                  general_long_goal_override=None):
    from contextlib import nullcontext
    if not manage_transaction and not db.in_transaction:
        raise ValueError('Caller-owned race transaction required')
    previous = latest_request(db)
    current = previous or state.race_goal
    parent = current.key if current else 'adopted:'+state.fingerprint
    fields = (draft.distance, draft.race_date, draft.preparation_mode)
    if current and fields == (current.distance, current.race_date, current.preparation_mode):
        return None
    if not current and fields == ('general_fitness', None, 'balanced'):
        return None
    if expected_revision != parent:
        raise ValueError('Race goal changed since this form loaded. Reload Settings.')
    applied=state.race_goal
    if applied and applied.goal and applied.race_date < as_of.astimezone(ZoneInfo(state.timezone)).date():
        from .race_event import latest_resolution
        if latest_resolution(db,applied.key,as_of=as_of) is None:
            raise ValueError('Confirm the past race outcome before replacing or clearing its goal')
    if as_of <= state.as_of or previous and as_of <= previous.available_at:
        raise ValueError('Race request must follow the current decision and previous request')
    revision = RaceGoalRevision.create(available_at=as_of, distance=draft.distance,
        race_date=draft.race_date, preparation_mode=draft.preparation_mode, parent_key=parent,
        general_long_goal_miles=(general_long_goal_override if general_long_goal_override is not None else
            state.planner_preferences.goal_long_miles if state.planner_preferences else
            current.general_long_goal_miles if current else general_long_goal_miles))
    db.execute('CREATE TABLE IF NOT EXISTS adaptive_race_requests(key TEXT PRIMARY KEY,payload TEXT NOT NULL)')
    with db if manage_transaction else nullcontext():
        if manage_transaction:
            db.execute('BEGIN IMMEDIATE')
        if latest_request(db) != previous:
            raise ValueError('Concurrent race edit; reload Settings')
        db.execute('INSERT INTO adaptive_race_requests VALUES (?,?)',
            (revision.key, json.dumps(revision.payload, sort_keys=True, separators=(',', ':'))))
    return revision


def requested_revision(harness, as_of):
    sandbox=getattr(harness,'sandbox',None)
    return (latest_request(sandbox.db,as_of=as_of) if sandbox else None) or getattr(harness,'race_goal',None)


@dataclass(frozen=True)
class RaceGoalApplication:
    revision: RaceGoalRevision
    applied_at: datetime
    effective_on: date

    def __post_init__(self):
        if self.applied_at.tzinfo is None or self.applied_at < self.revision.available_at:
            raise ValueError('Applied race goal must follow its request')


def application_history(harness, as_of):
    """Prospective publication view; persisted only with the winning decision."""
    history = getattr(harness, 'race_goal_history', ())
    revision = getattr(harness, 'race_goal', None)
    if revision is not None and (not history or history[-1].revision != revision):
        history += (RaceGoalApplication(revision, as_of, harness.decision(as_of)),)
    return history


def overlay_applications(segments, history, *, timezone, resolution=None,resolutions=()):
    """Edits/cancellation cannot reprice an earlier applied taper."""
    from zoneinfo import ZoneInfo
    zone = ZoneInfo(timezone)
    applications = tuple((h.effective_on, h.revision) for h in history)
    outcomes={r.race_key:r for r in resolutions}
    if resolution:outcomes[resolution.race_key]=resolution
    result = []
    for segment in segments:
        day = segment.start
        while day < segment.end:
            revision = next((revision for start, revision in reversed(applications) if start <= day), None)
            from .race_event import training_fraction
            # An outcome becomes effective prospectively, never rewrites a
            # previously issued taper or pre-report recovery target.
            outcome=outcomes.get(revision.key) if revision else None
            known = outcome if outcome and day >= outcome.available_at.astimezone(zone).date() else None
            factor = training_fraction(revision,known,day) if revision is not None and revision.goal else 1.
            rate = max(.1, segment.miles_per_7_days * factor)
            if result and result[-1].end == day and result[-1].miles_per_7_days == rate:
                old = result.pop()
                result.append(TargetSegment(old.start, day+timedelta(days=1), rate))
            else:
                result.append(TargetSegment(day, day+timedelta(days=1), rate))
            day += timedelta(days=1)
    return tuple(result)


def overlay_targets(segments, revision, *, activation_day):
    """Split at absolute dates; preserve every historical target byte/value.

    Do not retrospectively apply a goal saved midway through its taper.
    """
    if revision is None or revision.goal is None:
        return segments
    result = []
    for segment in segments:
        day = segment.start
        while day < segment.end:
            factor = revision.training_fraction(day) if day >= activation_day else 1.
            rate = max(.1, segment.miles_per_7_days * factor)
            if result and result[-1].end == day and result[-1].miles_per_7_days == rate:
                old = result.pop()
                result.append(TargetSegment(old.start, day+timedelta(days=1), rate))
            else:
                result.append(TargetSegment(day, day+timedelta(days=1), rate))
            day += timedelta(days=1)
    return tuple(result)


def preparation_status(harness, day, *, as_of):
    revision = getattr(harness, 'race_goal', None)
    if revision is None or revision.goal is None:
        return None
    target = destination(harness.stage, revision.goal)
    from .race_event import physical_keys, resolution_for, no_race_recovery
    race_keys=physical_keys(harness)
    outcome=resolution_for(harness,revision)
    if outcome is not None and outcome.available_at>as_of:
        outcome=None
    phase=('did_not_start' if no_race_recovery(revision,outcome,day)
        else revision.phase(day))
    prior = max((r.miles for r in harness.history if r.source=='observed' and r.available_at<=as_of
        and day-timedelta(days=30)<=r.start.astimezone(harness.zone).date()<day
        and getattr(harness,'capacity_eligibility',{}).get(r.key,True)
        and r.key not in race_keys),default=None)
    from run_analysis.program_constraints import _ceiling
    ceiling = _ceiling(prior) if prior is not None else None
    return dict(revision=revision.key, distance=revision.distance,
        race_date=revision.race_date.isoformat(), phase=phase,
        outcome=outcome.status if outcome else 'unknown',
        target_volume_per_7=target.volume_per_7, target_long_miles=target.long_miles,
        accepted_volume_per_7=harness.stage.weekly_equivalent_miles,
        accepted_long_miles=harness.stage.long_miles,
        longest_prior_30d_miles=prior, ordinary_training_session_limit_miles=ceiling,
        event_exceeds_training_session_limit=ceiling is None or revision.goal.profile.distance_miles>ceiling,
        preparation_gap=(harness.stage.weekly_equivalent_miles < target.volume_per_7
                         or harness.stage.long_miles < target.long_miles),
        race_participation_cleared=False, training_guard_unchanged=True)
