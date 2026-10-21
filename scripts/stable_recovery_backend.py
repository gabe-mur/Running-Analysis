"""Offline recovery replacement inside the established scheduler, not a search.

Preserves the native expected-duration transition. Recorded HR rates supply a
declared synthetic sidecar, never fabricated measured HR in FitnessState.
Envelopes are a sensitivity hypothesis, not calibrated athlete parameters.
"""
from contextlib import contextmanager
from dataclasses import replace
from datetime import timedelta

from run_analysis import weekly_schedule as planner
from run_analysis.selection_objective import RECOVERY_BACKEND, EXPECTED_COMPLETION_STATE
from run_analysis.recovery_assessment import AssessmentPolicy, SessionEntry
from run_analysis.workload_accounting import Quantity, Workload
from run_analysis.workload_profile import prescribed_profile
from run_analysis.prescription_load import prescribed_zone_minutes
from run_analysis.web.schemas import ReadinessFlag
from scripts.hr_backed_sandbox import choose_donor


POLICY = AssessmentPolicy('matched_plus_work_h36', 160, 320, 36, 36, 24, 36)


class StableRecoveryBackend:
    def __init__(self, donors, historical_runs, *, policy=POLICY):
        self.donors = donors
        self.policy = policy
        self.completed = {}
        self.issued = {}
        self.prefixes = {}
        self.entries = {}
        self.checks = {}
        self.actual_residuals = {}
        self.metrics = dict(checks=0, cache_hits=0, guards=0, program_scores=0,
                            unresolved_work_prescriptions=0)
        for run in historical_runs:
            rec = planner.RecommendationResponse(
                generated_at=run.start_time, fitness_state_as_of=run.start_time,
                planned_for=run.start_time, workout_type=run.workout_type,
                distance_range_miles=(run.distance_miles, run.distance_miles),
                title='Paired synthetic historical fixture', confidence='moderate', readiness='ready',
                structure=[planner.WorkoutStep(instruction='Seed 6 x 1 minute', phase='work',
                    repetitions=6, work_duration_minutes=1, target_zones=['Z4'])]
                    if run.workout_type in planner.QUALITY_WORKOUT_TYPES else [])
            self.completed[run.start_time] = self.make_entry(rec, run.moving_minutes, source='observed')

    @staticmethod
    def physical(rec):
        return (rec.planned_for, rec.workout_type, rec.planned_distance_miles,
                tuple((s.phase, s.repetitions, s.work_duration_minutes,
                       s.work_duration_range_minutes, s.duration_minutes,
                       tuple(s.target_zones)) for s in rec.structure))

    def make_entry(self, rec, minutes, *, source='projected'):
        shape = prescribed_profile(rec)['intensity_structure']
        # Donor identity is selected at a fixed representative distance, not
        # cherry-picked again when the optimizer changes a target by 0.1 mile.
        role = ('quality' if rec.workout_type in planner.QUALITY_WORKOUT_TYPES else
                'long' if rec.workout_type == planner.WorkoutType.LONG else 'easy')
        donor = self.donors[role]
        missing = 'work_duration_missing' in prescribed_profile(rec)['limitations']
        if missing:
            # Preserve uncertainty instead of treating a prose-only finish as
            # confirmed zero work or silently changing the issued workout.
            work_low, work_high = 0., minutes
            self.metrics['unresolved_work_prescriptions'] += 1
        else:
            bounds = [b['duration_range_seconds'] or (b['work_seconds'], b['work_seconds'])
                      for b in shape['bouts']]
            work_low = sum(b[0] for b in bounds)/60
            work_high = sum(b[1] for b in bounds)/60
        if work_high > minutes+1e-8:
            raise ValueError('Native expected duration cannot hold prescribed work')
        if role == 'quality':
            if donor.work_rate is None or donor.other_rate is None:
                raise ValueError('Quality requires a recorded HR phase donor')
            endpoints = [(w*donor.work_rate.low+(minutes-w)*donor.other_rate.low,
                          w*donor.work_rate.high+(minutes-w)*donor.other_rate.high)
                         for w in (work_low, work_high)]
            low, high = min(x[0] for x in endpoints), max(x[1] for x in endpoints)
        else:
            low, high = minutes*donor.total_rate.low, minutes*donor.total_rate.high
        workload = Workload(source, Quantity.point(rec.planned_distance_miles*1609.344, 'synthetic distance'),
            Quantity.point(minutes, 'native expected/completed duration'),
            Quantity(low, high, basis='fixed original HR phase rates rescaled to synthetic duration'),
            {'donor_key': donor.key, 'synthetic': True}, ('Not a measured physiological response',))
        return SessionEntry('paired:'+rec.planned_for.isoformat(), rec.planned_for,
            rec.planned_for+timedelta(minutes=minutes), workload,
            Quantity(work_low, work_high, basis='numeric work or explicitly unresolved duration'))

    def reset_build(self):
        self.prefixes.clear()
        self.entries.clear()
        self.checks.clear()
        self.actual_residuals.clear()

    def register_state(self, state, planned):
        self.prefixes[id(state)] = (state, tuple(planned))

    def prefix(self, state):
        saved = self.prefixes.get(id(state))
        return saved[1] if saved is not None and saved[0] is state else ()

    def entry(self, rec, prefix):
        model = EXPECTED_COMPLETION_STATE.get()
        if model is None:
            raise ValueError('Recovery replacement requires native expected-completion accounting')
        key = (self.physical(rec), tuple(self.physical(r) for r in prefix))
        if key not in self.entries:
            ledger = model.ledger(rec.planned_for+timedelta(microseconds=1), [*prefix, rec])
            run = next(r for r in ledger if r.start_time == rec.planned_for)
            self.entries[key] = self.make_entry(rec, run.moving_minutes)
        return self.entries[key]

    def check(self, rec, prefix=()):
        if rec.workout_type == planner.WorkoutType.REST:
            return dict(compatible=True, excess=0., demands={})
        prefix = tuple(r for r in prefix if r.workout_type != planner.WorkoutType.REST
                       and r.planned_for < rec.planned_for)
        key = (self.physical(rec), tuple(self.physical(r) for r in prefix))
        self.metrics['checks'] += 1
        if key in self.checks:
            self.metrics['cache_hits'] += 1
            return self.checks[key]
        proposed = self.entry(rec, prefix)
        earlier = [self.entry(r, prefix[:i]) for i, r in enumerate(prefix)]
        model = EXPECTED_COMPLETION_STATE.get()
        actual = [self.completed[r.start_time] for r in model.completed]
        if len({e.key for e in (*actual, *earlier, proposed)}) != len(actual)+len(earlier)+1:
            raise ValueError('Duplicate completed/expected recovery entry')
        p = self.policy
        specs = [('running_minutes', p.running_minutes_envelope, p.running_half_life_hours),
                 ('cardio_points', p.cardio_points_envelope, p.cardio_half_life_hours)]
        if p.work_minutes_envelope is not None:
            specs.append(('work_minutes', p.work_minutes_envelope, p.work_half_life_hours))
        def dose(e, name):
            return (e.work_minutes if name == 'work_minutes' else getattr(e.workload, name)).high
        paths = []
        # Ordinary execution timing variation, anchored to absolute dates.
        for phase in (-1, 0, 1):
            def shift(e): return timedelta(hours=phase*(6 if e.start.date().toordinal()%2 else -6))
            at = proposed.start+shift(proposed)
            demands = {}
            for name, budget, half in specs:
                actual_key = (at, name, half)
                if actual_key not in self.actual_residuals:
                    self.actual_residuals[actual_key] = (float('inf') if any(e.end > at for e in actual)
                        else sum(dose(e,name)*2**(-((at-e.end).total_seconds()/3600)/half) for e in actual))
                residual = self.actual_residuals[actual_key]
                for e in earlier:
                    end = e.end+shift(e)
                    if end > at:
                        # Not physiologically cleared by clamping negative time.
                        residual = float('inf')
                        break
                    residual += dose(e, name)*2**(-((at-end).total_seconds()/3600)/half)
                demands[name] = residual+dose(proposed, name)
            excess = max(max(0., demands[n]/b-1) for n,b,_ in specs)
            paths.append(dict(phase=phase, excess=excess, demands=demands))
        worst = max(paths, key=lambda r:r['excess'])
        answer = dict(compatible=worst['excess'] <= 1e-9, excess=worst['excess'], paths=paths)
        self.checks[key] = answer
        return answer

    def check_state(self, state, rec):
        return self.check(rec, self.prefix(state))

    def guard(self, state, rec):
        if rec.workout_type == planner.WorkoutType.REST:
            return rec  # Health/extreme-weather veto is never erased.
        if self.check_state(state, rec)['compatible']:
            return rec
        self.metrics['guards'] += 1
        return rec.model_copy(update=dict(workout_type=planner.WorkoutType.REST,
            readiness=ReadinessFlag.NOT_READY, distance_range_miles=None,
            distance_target_miles=None, structure=[], title='No compatible dose at this slot',
            readiness_reason='offline_new_recovery_incompatible'))

    def session_cost(self, state, rec):
        return planner.PROGRAM_FIT_UNIT*self.check_state(state, rec)['excess']

    def program_cost(self, days, states, diagnostic):
        self.metrics['program_scores'] += 1
        prefix = list(self.prefix(states[0])) if states else []
        cost = 0.
        for day in days:
            rec = day.recommendation
            if rec is None or rec.workout_type == planner.WorkoutType.REST:
                continue
            cost += planner.PROGRAM_FIT_UNIT*self.check(rec, prefix)['excess']
            prefix.append(rec)
        # This finite routing cost guides proposals only. The unchanged
        # continuation pipeline calls the replaced sequence gate before final
        # selection, so training value cannot buy incompatible recovery.
        if diagnostic is not None:
            diagnostic.update(new_demand_excess=cost)
        return cost

    def complete(self, run, rec):
        expected = self.issued.get(run.start_time)
        if expected is None:
            raise ValueError('Execution has no frozen issued recovery expectation')
        if abs(run.moving_minutes-expected.workload.running_minutes.high) > 1e-8:
            raise ValueError('Perfect completion differs from native expected duration')
        self.completed[run.start_time] = replace(expected, workload=replace(expected.workload, source='observed'))


@contextmanager
def recovery_backend_policy(backend):
    token = RECOVERY_BACKEND.set(backend)
    try:
        yield backend
    finally:
        RECOVERY_BACKEND.reset(token)
