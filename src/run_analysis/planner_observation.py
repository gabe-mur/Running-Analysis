"""Ephemeral, request-scoped planner telemetry. Never input to the optimizer.

The search budget supplies a real allowance denominator, not a time estimate.
UI estimates reserve 8% for preparation and the final 10% for refinement/save.
These are display allocations only, never optimizer weights or eligibility.
Observers are absent for normal offline tests/replays. Nothing is persisted.
"""
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from threading import Lock
from time import monotonic


_active = ContextVar('planner_observer', default=None)
_registry = ContextVar('planner_progress_registry', default=None)
PHASES = {'waiting', 'history', 'weather', 'search', 'refine', 'publish'}


@dataclass
class _Work:
    started: float
    updated: float
    phase: str = 'waiting'
    state: str = 'running'
    model_checks: int = 0
    phases: tuple = ('waiting',)
    budget_used: int = 0
    budget_total: int | None = None
    estimated_percent: float = 0.
    following: '_Work | None' = None
    inherited_progress: bool = False
    waiting_key: str | None = None
    owns_job: bool = False


class PlannerProgress:
    def __init__(self, *, clock=monotonic, retention_seconds=1800, capacity=128):
        self.clock, self.retention, self.capacity = clock, retention_seconds, capacity
        self._work = OrderedDict()
        self._jobs = OrderedDict()
        self._lock = Lock()

    @contextmanager
    def provide(self):
        """Make upload/sync-owned planner work observable, even without a header."""
        token = _registry.set(self)
        try:
            yield
        finally:
            _registry.reset(token)

    @contextmanager
    def observe(self, request_id):
        now = self.clock()
        with self._lock:
            self._prune(now)
            work = _Work(now, now)
            self._work[request_id] = work
            while len(self._work) > self.capacity:
                self._work.popitem(last=False)
        token = _active.set((self, work))
        registry_token = _registry.set(self)
        try:
            yield
        except BaseException:
            self.finish(work, 'failed')
            raise
        finally:
            _active.reset(token)
            _registry.reset(registry_token)

    def _prune(self, now):
        for key, work in tuple(self._work.items()):
            if now - work.updated > self.retention:
                del self._work[key]
        for key, work in tuple(self._jobs.items()):
            if now - work.updated > self.retention:
                del self._jobs[key]

    def finish(self, work, state):
        with self._lock:
            work.state, work.updated = state, self.clock()
            if state == 'complete':
                work.estimated_percent = 100.

    def snapshot(self, request_id):
        now = self.clock()
        with self._lock:
            self._prune(now)
            work = self._work.get(request_id)
            if work is None:
                return {'state': 'unknown'}
            if work.state == 'running' and work.waiting_key and work.following is None:
                owner = self._jobs.get(work.waiting_key)
                if owner is not None and owner is not work and owner.state == 'running':
                    work.following = owner
            end = now if work.state == 'running' else work.updated
            # A refreshing browser may be waiting on an upload-owned rebuild.
            # Follow that actual work, but never finish the waiting HTTP request
            # early or inherit another request's failure as its own result.
            source = work.following if work.state == 'running' and work.following else work
            percent = source.estimated_percent if source is work else min(97., source.estimated_percent)
            return dict(state=work.state, phase=source.phase, model_checks=source.model_checks,
                phases=list(source.phases), elapsed_seconds=max(0., end-work.started),
                budget_used=source.budget_used, budget_total=source.budget_total,
                estimated_percent=percent, following_active_job=source is not work and not work.owns_job)


def follow_job(key):
    """Called before the existing live planner lock; never acquires that lock."""
    observer = _active.get()
    if observer is None:
        return
    registry, work = observer
    with registry._lock:
        work.waiting_key = str(key)
        owner = registry._jobs.get(str(key))
        if owner is not None and owner is not work and owner.state == 'running':
            work.following = owner


@contextmanager
def planner_job(key):
    """Observe the actual lock owner, including jobs started by upload/sync.

    Enter only after the existing thread lock. This changes neither ownership,
    serialization nor search inputs. Offline runs still have no observer.
    """
    registry = _registry.get()
    if registry is None:
        yield
        return
    observer = _active.get()
    request_work = observer[1] if observer is not None else None
    # The actual computation outlives a disconnected browser request. Give it
    # its own record so cancelling/refreshing that request cannot mark an
    # ongoing search failed and make the next browser lose its progress.
    work = _Work(registry.clock(), registry.clock())
    with registry._lock:
        if request_work is not None and request_work.following is not None:
            source = request_work.following
            work.model_checks, work.budget_used, work.budget_total = source.model_checks, source.budget_used, source.budget_total
            work.estimated_percent = min(97., source.estimated_percent)
            work.phases = tuple(dict.fromkeys((*source.phases, *request_work.phases)))
            work.inherited_progress = True
        if request_work is not None:
            request_work.waiting_key = None
            request_work.following = work
            request_work.owns_job = True
        registry._jobs[str(key)] = work
        registry._jobs.move_to_end(str(key))
        while len(registry._jobs) > registry.capacity:
            registry._jobs.popitem(last=False)
    token = _active.set((registry, work))
    try:
        yield
    except BaseException:
        registry.finish(work, 'failed')
        raise
    else:
        registry.finish(work, 'complete')
    finally:
        if request_work is not None:
            with registry._lock:
                request_work.phase, request_work.phases = work.phase, work.phases
                request_work.model_checks = work.model_checks
                request_work.budget_used, request_work.budget_total = work.budget_used, work.budget_total
                request_work.estimated_percent = min(97., work.estimated_percent)
                request_work.following = None
                request_work.owns_job = False
        _active.reset(token)


def phase(name):
    observer = _active.get()
    if observer is None or name not in PHASES:
        return
    registry, work = observer
    with registry._lock:
        if name == 'search' and work.inherited_progress:
            # New evidence may require a second real search after the shared
            # job. Its counter must not masquerade as the completed first one.
            work.estimated_percent, work.model_checks = 8., 0
            work.budget_used, work.budget_total = 0, None
            work.phases = ('waiting', 'history', 'weather')
            work.inherited_progress = False
        work.phase, work.updated = name, registry.clock()
        if name not in work.phases:
            work.phases = (*work.phases, name)
        floor = dict(waiting=0, history=3, weather=6, search=8, refine=92, publish=97)[name]
        work.estimated_percent = max(work.estimated_percent, floor)


def model_check():
    observer = _active.get()
    if observer is None:
        return
    registry, work = observer
    with registry._lock:
        if work.phase in {'search', 'refine'}:
            work.model_checks += 1
            work.updated = registry.clock()


def budget_progress(used, total):
    observer = _active.get()
    if observer is None:
        return
    registry, work = observer
    with registry._lock:
        work.budget_used, work.budget_total = used, total
        work.updated = registry.clock()
        if total > 0:
            work.estimated_percent = max(work.estimated_percent, 8 + 82*min(1,used/total))


def finished(success=True):
    observer = _active.get()
    if observer is not None:
        registry, work = observer
        registry.finish(work, 'complete' if success else 'failed')
