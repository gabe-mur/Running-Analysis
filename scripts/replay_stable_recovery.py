"""Paired fail-fast recovery experiment using the ACTUAL stable search stack.

No alternate calendar search, settings write, live activation or source DB write.
Stop on any visible-week date/role change or >0.3-mile target change. Smaller
edits remain explicit diagnostics, matching the acknowledged stable benchmark.
"""
from argparse import ArgumentParser
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import json
import yaml

from run_analysis import adherence_projection as projection
from run_analysis.config import load_config
from run_analysis.selection_objective import FINALIZED_PROGRAM_SELECTOR
from run_analysis.projection_gate import ProjectionGate, ProjectionGateConfig, ProjectionGateTriggered
from scripts.replay_synthetic_easy_dose import synthetic_history, shared_objective_failure
from scripts.replay_easy_dose_program import replay_policy
from scripts.opportunity_search_repair import opportunity_search_policy
from run_analysis.prescription_distance import canonical_distance_policy
from run_analysis.selection_objective import shared_objective_policy
from scripts.finalized_scoring_trial import finalized_scoring_policy
from scripts.calendar_neighbor_prototype import calendar_neighbor_policy, NoAcceptableCalendarError
from scripts.continuation_replan import continuation_replan_policy
from scripts.expected_completion import expected_completion_build_policy
from scripts.volume_path_prototype import volume_path_policy
from scripts.boundary_funding_prototype import boundary_funding_policy
from scripts.full_score_refinement import full_score_refinement_policy
from scripts.finalized_distance_search import distance_refinement_policy
from scripts.suffix_regrowth_prototype import regrowth_replan_policy
from scripts.adaptation_audit import adaptation_transitions
from scripts.audit_backward_controls import plain
from scripts.hr_backed_sandbox import load_bank, choose_donor
from scripts.stable_recovery_backend import StableRecoveryBackend, recovery_backend_policy
from scripts.typed_continuation_prototype import runs
from run_analysis.web.schemas import RecommendationResponse, WorkoutStep


@contextmanager
def stable_stack():
    """Exact configuration of post-calendar-distance-21day-perfect.json."""
    with ExitStack() as stack:
        stack.enter_context(distance_refinement_policy(enabled=True, budget=128,
            visible_days=21, post_calendar_budget=128))
        stack.enter_context(full_score_refinement_policy(budget=80, starts=2, fixed_tail=True))
        stack.enter_context(shared_objective_policy(True))
        stack.enter_context(canonical_distance_policy(True))
        stack.enter_context(opportunity_search_policy(True))
        stack.enter_context(finalized_scoring_policy(True))
        calendars = stack.enter_context(calendar_neighbor_policy(enabled=True, audit_opening=False,
            max_passes=1, enforce_acceptance=True))
        records = stack.enter_context(continuation_replan_policy(enabled=True,
            calendar_records=calendars, finalist_budget=6))
        stack.enter_context(replay_policy('history_trajectory'))
        stack.enter_context(expected_completion_build_policy(True))
        stack.enter_context(volume_path_policy(True))
        stack.enter_context(boundary_funding_policy())
        stack.enter_context(regrowth_replan_policy())
        yield records


class StopExperiment(RuntimeError):
    pass


def donors(bank, start):
    selected = {}
    for role, kind, miles in [('easy','easy',3.75), ('long','long',7.5), ('quality','intervals',4.5)]:
        rec = RecommendationResponse(generated_at=start, fitness_state_as_of=start,
            planned_for=start, workout_type=kind,
            distance_range_miles=(miles,miles), title='Fixed donor selection',
            confidence='moderate', readiness='ready', structure=[WorkoutStep(
                instruction='6 x 1 minute', phase='work', repetitions=6, work_duration_minutes=1,
                target_zones=['Z4'])] if role=='quality' else [])
        selected[role] = choose_donor(bank, rec)
    return selected


def run(mode, days, path, *, journal_path=None):
    if path.exists():
        raise ValueError('Refusing to overwrite an experiment')
    start = datetime(2026,9,22,tzinfo=timezone.utc)
    template, history = synthetic_history(start)
    config = yaml.safe_load(Path('config.example.yaml').read_text())
    bank = load_bank(Path('data/run_analysis.sqlite'), load_config())
    backend = StableRecoveryBackend(donors(bank, start), history)
    gate = ProjectionGate(ProjectionGateConfig(enforce_stability=True,
        audit_key_session_cadence=True, enforce_aerobic_exposure_cadence=True), fail_fast=True)
    snapshots, executions, builds = [], [], []
    stopped = None
    started = perf_counter()
    latest = {}

    def observe(s):
        snapshots.append(s)
        print(f'{mode} {s.generated_at.isoformat()} {s.trigger} {s.planning_seconds:.2f}s', flush=True)
        if journal_path:
            with journal_path.open('a') as stream:
                stream.write(json.dumps(plain(s))+'\n')
        gate.observe(s)
        mismatch = shared_objective_failure(s)
        if mismatch:
            raise StopExperiment(mismatch.detail)
        if len(snapshots)>1:
            change = adaptation_transitions(snapshots[-2:])[0]
            visible = change['windows']['7']
            if visible['date_changes'] or visible['type_changes']:
                raise StopExperiment('Visible-week date/role instability: '+json.dumps(plain(change)))
            # Distances reported below regardless of the legacy gate tolerance.
            from run_analysis.projection_gate import _future_distance_changes
            material = _future_distance_changes(snapshots[-2], s, horizon_days=7, tolerance_miles=.30000001)
            if material:
                raise StopExperiment('Visible-week distance instability: '+json.dumps(plain(material)))

    def executed(actual, rec):
        backend.complete(actual, rec)
        executions.append(dict(at=actual.start_time, miles=actual.distance_miles,
            minutes=actual.moving_minutes, role=actual.workout_type.value))

    try:
        with stable_stack() as records, recovery_backend_policy(backend if mode=='new' else None):
            original_build = projection.build_weekly_schedule
            original_select = FINALIZED_PROGRAM_SELECTOR.get()
            def selected(**kwargs):
                result = original_select(**kwargs)
                # Freeze the actually selected prescriptions while the native
                # expected-completion ledger is still in invocation scope.
                prefix = []
                published = []
                for rec in runs(result['planning_days']):
                    entry = backend.entry(rec, prefix)
                    backend.issued[rec.planned_for] = entry
                    published.append(dict(at=rec.planned_for, miles=rec.planned_distance_miles,
                        role=rec.workout_type.value, recovery=backend.check(rec,prefix)))
                    prefix.append(rec)
                if mode=='new' and any(not r['recovery']['compatible'] for r in published):
                    raise StopExperiment('Published hard plan violates replacement recovery')
                evidence = result['evidence']
                latest.clear()
                latest.update(dict(published=published, score=evidence['search_cost'],
                    prior=evidence['prior_program'], candidates=[{
                        k:v for k,v in r.items() if k in ('source','offsets','combined_cost','rejected',
                            'stable_prefix','hard_fingerprint','tail_fingerprint')}
                        for r in evidence['finalists']], metrics=dict(backend.metrics)))
                return result
            def build(*args, **kwargs):
                backend.reset_build()
                result = original_build(*args, **kwargs)
                builds.append(dict(latest))
                return result
            token = FINALIZED_PROGRAM_SELECTOR.set(selected)
            projection.build_weekly_schedule = build
            try:
                projection.simulate_adherence(template, history, config, start,
                    simulation_days=days, weeks=max(1,(days+6)//7), replan_interval_days=1,
                    initial_schedule=None, emit_post_upload_replans=True,
                    replan_observer=observe, execution_observer=executed)
            finally:
                projection.build_weekly_schedule = original_build
                FINALIZED_PROGRAM_SELECTOR.reset(token)
    except (StopExperiment, ProjectionGateTriggered, NoAcceptableCalendarError) as error:
        stopped = str(error)
        print('STOP: '+stopped, flush=True)
    except BaseException as error:
        stopped = 'Experiment interrupted or integration error: '+repr(error)
        raise
    finally:
        report = dict(mode=mode, requested_days=days, stopped=stopped,
            completed_decisions=len(snapshots), seconds=perf_counter()-started,
            recovery_policy=asdict(backend.policy), donor_ids={k:v.key for k,v in backend.donors.items()},
            sidecar_scope='Synthetic HR rates on native expected durations, not observed athlete forecasts',
            legacy_continuous_corridor_gate_retained=True,
            transitions=adaptation_transitions(snapshots), executions=executions,
            builds=builds, failures=gate.failures, metrics=backend.metrics)
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x') as stream:
            json.dump(plain(report), stream, indent=2)
    return report


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['legacy','new'], required=True)
    parser.add_argument('--days', type=int, default=7)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--journal', type=Path)
    args = parser.parse_args()
    if args.journal:
        args.journal.parent.mkdir(parents=True, exist_ok=True)
        with args.journal.open('x'): pass
    result = run(args.mode,args.days,args.output,journal_path=args.journal)
    if result['stopped']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
