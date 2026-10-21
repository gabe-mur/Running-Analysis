"""Offline conditional execution state, using the simulator's upload transition.

No forecast work is written to observed history or counted as measured response.
Each query rebuilds an immutable actual-plus-conditional ledger at one exact
timestamp; expected completion and identical execution must agree there.
"""

from contextlib import contextmanager
from datetime import timedelta

from run_analysis import adherence_projection as projection, weekly_schedule as planner
from run_analysis.selection_objective import COMPLETED_SELECTION_HISTORY, EXPECTED_COMPLETION_STATE
from run_analysis.training_load import TrainingSession, distance_capacity
from run_analysis.prescription_load import projected_running_minutes


def expected_run(item, pace):
    if item.planned_for is None or item.distance_range_miles is None or item.workout_type == planner.WorkoutType.REST:
        raise ValueError('Expected execution requires a timed running prescription')
    miles = item.planned_distance_miles
    return projection.ProjectionRun(
        start_time=item.planned_for, distance_miles=miles, moving_minutes=projected_running_minutes(item, miles, pace),
        workout_type=item.workout_type, difficulty=projection._projected_difficulty(item, miles, pace),
        projected=True, planning_role=projection._completed_planning_role(item, item.workout_type),
        prescribed_workout_type=item.workout_type, completed_prescribed_workout=True,
        prescribed_target_miles=item.distance_target_miles,
        prescribed_low_miles=item.distance_range_miles[0], prescribed_high_miles=item.distance_range_miles[1],
        prescription_title=item.title,
        quality_session_type=item.quality_session_type if item.workout_type in projection.QUALITY_TYPES else None)


class ExpectedCompletion:
    def __init__(self, template, completed, config):
        self.template = template
        self.completed = tuple(sorted(completed, key=lambda r: r.start_time))
        self.config = config
        self._states = {}
        self._ledgers = {}
        self._targets = {}
        self._capacities = {}
        self._activities = {}
        self.calls = 0
        self.rebuilds = 0

    def _pace_reference_at(self, item):
        return item.planned_for.replace(hour=0, minute=0, second=0, microsecond=0)

    def ledger(self, at, planned):
        causal = tuple(sorted((r for r in planned if r.planned_for is not None and r.planned_for < at
                       and r.workout_type != planner.WorkoutType.REST), key=lambda r: r.planned_for))
        def physical(item):
            # Include the complete structured dose (especially between-rep
            # recovery), not a handpicked subset of its fields. Re-read mutable
            # API models instead of caching their signature by object identity.
            # Identical copies still share the expensive ledger/state results.
            return (item.planned_for, item.workout_type, item.planning_role,
                item.distance_target_miles, item.distance_range_miles, item.quality_session_type,
                item.title, tuple(s.model_dump_json() for s in item.structure))
        key = tuple(physical(r) for r in causal)
        cached = self._ledgers.get(key)
        if cached is not None:
            return cached[1]
        actual_times = {r.start_time for r in self.completed}
        seen = set()
        for item in causal:
            if item.planned_for in actual_times or item.planned_for in seen:
                raise ValueError('Expected execution duplicates completed or planned work')
            seen.add(item.planned_for)
        # Neighboring calendars and role sweeps share long physical prefixes.
        # Extend the longest cached ledger instead of recomputing each earlier
        # expected completion's pace. Never reuse a suffix after a changed run.
        prefix_length = 0
        history = list(self.completed)
        for stop in range(len(causal) - 1, 0, -1):
            prefix = self._ledgers.get(key[:stop])
            if prefix is not None:
                prefix_length = stop
                history = list(prefix[1])
                break
        for index, item in enumerate(causal[prefix_length:], prefix_length):
            previous = [r for r in history if r.start_time < item.planned_for]
            # Daily execution assumes the pace available at that day's plan
            # refresh, not a different sample that rolls out before run time.
            pace_at = self._pace_reference_at(item)
            window = projection._load_window(previous, pace_at, 28)
            pace = min(15., max(7., window.moving_minutes / window.distance_miles if window.distance_miles else 11.))
            history.append(expected_run(item, pace))
            history.sort(key=lambda r: r.start_time)
            self._ledgers[key[:index+1]] = (causal[:index+1], tuple(history))
        if causal:
            return self._ledgers[key][1]
        history.sort(key=lambda r: r.start_time)
        result = tuple(history)
        self._ledgers[key] = (causal, result)
        return result

    def at(self, at, planned=()):
        history = self.ledger(at, planned)
        key = (at, id(history))
        self.calls += 1
        if key not in self._states:
            # Capacity is derived at the same start-of-day boundary as an
            # uploaded run's next daily planner, not frozen at proposal time.
            boundary = at.replace(hour=0, minute=0, second=0, microsecond=0)
            target_key = (boundary, id(history))
            coaching = self.config.get('coaching', {})
            if id(history) not in self._activities:
                self._activities[id(history)] = projection.planning_activities_for(history)
            if target_key not in self._capacities:
                activities = self._activities[id(history)]
                self._capacities[target_key] = distance_capacity([
                    TrainingSession(activity_id=i, start_time=a.start_time,
                        distance_miles=planner._target_activity_miles(a),
                        moving_minutes=float(a.moving_minutes or 0.), zone_load=None, hard_minutes=0.)
                    for i, a in enumerate(activities) if planner._target_activity_miles(a) > 0], boundary,
                    retention_grace_days=int(coaching.get('capacity_retention_grace_days', 28)),
                    retention_half_life_days=float(coaching.get('capacity_retention_half_life_days', 84))).reference_miles
            state = projection._state_at(self.template, history, at, self._capacities[target_key],
                easy_baseline_half_life_days=float(coaching.get('capacity_retention_half_life_days', 84)),
                fatigue_half_life_days=float(coaching.get('continuous_fatigue_half_life_days', 7)))
            self._states[key] = state
            self.rebuilds += 1
        return self._states[key]

    def project(self, raw, planned):
        state = self.at(raw.as_of, planned)
        # Exogenous conditions do not become forecast training adaptations.
        return state.model_copy(update={name: getattr(raw, name) for name in (
            'planned_weather', 'weather_exposure_baseline', 'current_health_status',
            'recent_illness_or_recovery', 'normal_runs_since_health_event')})

    def target_at(self, at, planned=()):
        history = self.ledger(at, planned)
        # Target derivation only consumes activities. Building recovery, pace,
        # fatigue and capacity state here is redundant (derive_weekly_target
        # computes its own capacity evidence).
        if id(history) not in self._activities:
            self._activities[id(history)] = projection.planning_activities_for(history)
        boundary = at.replace(hour=0, minute=0, second=0, microsecond=0)
        key = (boundary, id(history))
        if key not in self._targets:
            self._targets[key] = planner.derive_weekly_target(self._activities[id(history)], boundary, self.config)
        return self._targets[key][1]

    def mean_session_after(self, cursor, planned):
        at = cursor + timedelta(microseconds=1)
        window = projection._load_window(self.ledger(at, planned), at, 28)
        if not window.activity_count or window.distance_miles <= 0:
            raise ValueError('Conditional opportunity cadence requires running history')
        return window.distance_miles / window.activity_count


@contextmanager
def expected_completion_policy(config, *, enabled=True):
    evidence = COMPLETED_SELECTION_HISTORY.get()
    if enabled and evidence is None:
        raise ValueError('Expected completion needs an explicit realized history ledger')
    model = ExpectedCompletion(evidence['template'], evidence['runs'], config) if enabled else None
    token = EXPECTED_COMPLETION_STATE.set(model)
    try:
        yield model
    finally:
        EXPECTED_COMPLETION_STATE.reset(token)


@contextmanager
def expected_completion_build_policy(enabled=True):
    """Scope one conditional ledger to each simulator build, including search."""
    original = projection.build_weekly_schedule
    records = []
    def build(*args, **kwargs):
        # Offline wrappers may expose only *args/**kwargs; the public builder
        # contract is (daily_states, request, config, ...).
        config = kwargs['config'] if 'config' in kwargs else args[2]
        with expected_completion_policy(config) as model:
            states = kwargs['daily_states'] if 'daily_states' in kwargs else args[0]
            if any(run.start_time >= states[0].as_of for run in model.completed):
                raise ValueError('Realized build history includes unavailable future work')
            try:
                return original(*args, **kwargs)
            finally:
                records.append(dict(calls=model.calls, rebuilds=model.rebuilds,
                    history_count=len(model.completed), conditional_ledgers=len(model._ledgers)))
    try:
        if enabled:
            projection.build_weekly_schedule = build
        yield records
    finally:
        projection.build_weekly_schedule = original
