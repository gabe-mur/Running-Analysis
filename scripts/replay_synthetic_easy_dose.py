"""Forward-only continuity sandbox: synthetic history, no database or old plan.

Uses the real daily/upload simulator and existing gates. Stock example settings
are loaded directly, without personal local overlays. Weather and measured
physiological response are absent/unchanging, not simulated observations.
"""

from argparse import ArgumentParser
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from math import isfinite

import yaml

from scripts.replay_easy_dose_program import replay_policy
from scripts.opportunity_search_repair import opportunity_search_policy
from run_analysis.prescription_distance import canonical_distance_policy
from run_analysis.selection_objective import shared_objective_policy
from scripts.issued_plan_execution import OpeningPlanExecution
from scripts.finalized_scoring_trial import finalized_scoring_policy
from scripts.calendar_neighbor_prototype import calendar_neighbor_policy, NoAcceptableCalendarError
from scripts.continuation_replan import continuation_replan_policy
from scripts.adaptation_audit import adaptation_transitions
from scripts.expected_completion import expected_completion_build_policy
from scripts.volume_path_prototype import volume_path_policy
from scripts.boundary_funding_prototype import boundary_funding_policy
from scripts.full_score_refinement import full_score_refinement_policy
from scripts.finalized_distance_search import distance_refinement_policy
from scripts.audit_delivery_contract import audit as delivery_contract_audit
from run_analysis.projection_gate import _json_ready
from run_analysis import adherence_projection as projection
from run_analysis.projection_gate import (
    ProjectionGate, ProjectionGateConfig, ProjectionGateTriggered, ProjectionGateFailure, build_projection_report,
)
from run_analysis.web.schemas import (
    FitnessState, LoadContext, LoadWindow, RecommendationResponse, WorkoutStep, WorkoutType,
)


def synthetic_history(start):
    """Twelve weeks of explicit fixture assumptions, not athlete measurements.

    Three sessions per seven-day seed cycle: 3.75 easy, 4.5 quality, 7.5 long.
    This seeds fitness evidence only; it is not a future scheduling rule.
    """
    runs = []
    for offset in range(-84, 0):
        role = {1: (WorkoutType.EASY, 3.75), 3: (WorkoutType.INTERVALS, 4.5),
                5: (WorkoutType.LONG, 7.5)}.get(offset % 7)
        if role is None:
            continue
        workout, miles = role
        when = (start + timedelta(days=offset)).replace(hour=7)
        recommendation = RecommendationResponse(
            generated_at=when, fitness_state_as_of=when, planned_for=when,
            workout_type=workout, title="Synthetic historical evidence",
            distance_range_miles=(miles, miles), confidence="moderate", readiness="ready",
            structure=[WorkoutStep(instruction="Synthetic quality dose", phase="work",
                repetitions=6, work_duration_minutes=1.0, target_zones=["Z4"])]
                if workout == WorkoutType.INTERVALS else [],
        )
        difficulty = projection._projected_difficulty(recommendation, miles, 11.0)
        runs.append(projection.ProjectionRun(
            start_time=when, distance_miles=miles, moving_minutes=miles * 11,
            workout_type=workout, difficulty=difficulty, projected=False,
        ))
    empty = {f"trailing_{days}d": LoadWindow(days=days, distance_miles=0,
             moving_minutes=0, activity_count=0) for days in (7, 14, 28)}
    template = FitnessState(as_of=start, window_days=28, fitness_trend="stable",
        trend_confidence="moderate", recent_load=LoadContext(**empty, confidence="moderate"),
        recent_performance_response="within_recent_range")
    # 15.75 is the explicit seed-cycle mileage, not a hand-injected future target.
    state = projection._state_at(template, runs, start, capacity_reference=15.75)
    return state, runs


def shared_objective_failure(snapshot):
    """Do not call a replay clean when its winning score describes another plan."""
    diagnostics = snapshot.planner_diagnostics or {}
    search = diagnostics.get('winning_candidate')
    published = diagnostics.get('published_candidate')
    continuation = diagnostics.get('continuation_selection')
    if continuation is not None:
        actual = [dict(planned_for=s.planned_for.isoformat(), workout_type=s.workout_type.value,
            midpoint_miles=s.midpoint_miles, distance_low_miles=s.distance_low_miles,
            distance_high_miles=s.distance_high_miles) for s in snapshot.planned_sessions]
        if (actual != continuation.get('hard_sessions')
                or continuation.get('selected_hard_fingerprint') != continuation.get('published_hard_fingerprint')
                or abs(continuation['search_cost'] - continuation['publication_cost']) > 1e-8):
            return ProjectionGateFailure('search_publication_mismatch', snapshot.generated_at,
                'Materialized continuation selection does not match the actual published hard prescriptions.',
                {'expected': continuation.get('hard_sessions'), 'actual': actual})
    if (search is not None and published is not None
            and abs(search['total_cost'] - published['total_cost']) <= 1e-8
            and search['allocation_trace'] == published['allocation_trace']):
        return None
    return ProjectionGateFailure('search_publication_mismatch', snapshot.generated_at,
        'Shared-objective search and publication disagree on the scored prescription.',
        {'search': search, 'published': published})


def main(*, funding_prototype=False, regrowth_prototype=False):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--scenario", choices=["perfect", *(x.value for x in projection.DeterministicAdherenceScenario)],
                        default="perfect")
    parser.add_argument("--prototype", choices=["current", "history_trajectory"], default="history_trajectory")
    parser.add_argument("--execution-plan", choices=["latest", "opening"], default="latest")
    parser.add_argument("--calendar-neighbors", action="store_true", help="Offline bounded full-horizon calendar polish")
    parser.add_argument("--candidate-acceptance", action="store_true", help="Filter both prototype lanes with the shared finalized-program gap check")
    parser.add_argument("--finalized-scoring", action="store_true", help="Offline consistent allocation/final scoring trial")
    parser.add_argument("--opportunity-repair", action="store_true", help="Bounded search timing and opening opportunity repair")
    parser.add_argument("--canonical-targets", action="store_true", help="Offline explicit tenth-mile distance targets")
    parser.add_argument("--shared-objective", action="store_true", help="Offline identical distance switching costs in search and publication")
    parser.add_argument('--continuation-selection', action='store_true', help='Offline full continuation objective on bounded finalized candidates')
    parser.add_argument('--continuation-finalists', type=int, default=6)
    parser.add_argument('--full-score-refinement', action='store_true')
    parser.add_argument('--fixed-tail-search', action='store_true',
                        help='Offline alternating calendar/tail search, sharing exact scored tails')
    parser.add_argument('--refinement-budget', type=int)
    parser.add_argument('--post-calendar-distance-budget', type=int, default=0,
                        help='Distance descent for the best full-score calendar neighbor')
    parser.add_argument('--distance-refinement', action='store_true')
    parser.add_argument('--distance-refinement-days', type=int, default=21)
    parser.add_argument('--distance-refinement-budget', type=int, default=128)
    parser.add_argument('--above-range-easy-fraction', type=float, default=.5,
        help='Extra distance above the prescribed upper bound as a fraction of ordinary easy size; minimum extra 0.1 mile')
    parser.add_argument("--calendar-neighbor-passes", type=int, default=1, help="Maximum offline neighbor passes; stop early if winners do not change")
    parser.add_argument("--audit-opening-neighbors", action="store_true", help="Report but do not select better neighbors of the polished opening plan")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.refinement_budget is None:
        args.refinement_budget = 80 if args.fixed_tail_search else 240
    if args.output_json.exists() or args.days < 1:
        parser.error("Require a new output path and positive duration")
    if args.execution_plan == "opening" and (args.days > 7 or args.scenario not in {"perfect", "in_range_low", "in_range_high"}):
        parser.error("Opening-contract screen supports at most 7 days and compliant execution scenarios")
    if args.audit_opening_neighbors and not args.calendar_neighbors:
        parser.error("Opening-neighbor audit requires --calendar-neighbors")
    if args.candidate_acceptance and not args.calendar_neighbors:
        parser.error("Candidate acceptance trial requires --calendar-neighbors")
    if args.opportunity_repair and not (args.calendar_neighbors and args.candidate_acceptance and args.finalized_scoring):
        parser.error("Opportunity repair requires finalized scoring, calendar neighbors and candidate acceptance")
    if args.shared_objective and not (args.canonical_targets and args.finalized_scoring):
        parser.error("Shared objective requires canonical targets and finalized scoring")
    if args.continuation_selection and not (args.shared_objective and args.opportunity_repair
                                          and args.prototype == 'history_trajectory'):
        parser.error('Continuation selection requires shared objective, opportunity repair and history trajectory')
    if funding_prototype and not args.continuation_selection:
        parser.error('Funding experiment requires the complete continuation workflow')
    if args.fixed_tail_search and not args.full_score_refinement:
        parser.error('Fixed-tail search requires --full-score-refinement')
    if args.distance_refinement and not (args.continuation_selection and args.canonical_targets and args.finalized_scoring and args.shared_objective):
        parser.error('Distance refinement requires canonical, finalized, shared continuation scoring')
    if not 1 <= args.distance_refinement_days <= 21 or args.distance_refinement_budget < 1:
        parser.error('Distance refinement requires 1–21 days and a positive evaluation budget')
    if args.full_score_refinement and not (funding_prototype and args.continuation_selection):
        parser.error('Full-score refinement requires full-path funding and continuation selection')
    if args.refinement_budget < 1:
        parser.error('Refinement budget must be positive')
    if args.post_calendar_distance_budget < 0:
        parser.error('Post-calendar distance budget must be nonnegative')
    if args.post_calendar_distance_budget and not args.fixed_tail_search:
        parser.error('Post-calendar distance refinement requires fixed-tail search')
    if regrowth_prototype and not (funding_prototype and args.continuation_selection):
        parser.error('Regrowth experiment requires full-path funding and continuation selection')
    if not 3 <= args.continuation_finalists <= 12:
        parser.error('Continuation finalist budget must be 3–12')
    if not isfinite(args.above_range_easy_fraction) or args.above_range_easy_fraction < 0:
        parser.error('Above-range distance fraction must be finite and nonnegative')
    if args.calendar_neighbor_passes < 1 or (args.calendar_neighbor_passes != 1 and not args.calendar_neighbors):
        parser.error("Positive neighbor pass budget requires --calendar-neighbors")
    start = datetime(2026, 9, 22, tzinfo=timezone.utc)
    decision_journal = None
    if args.full_score_refinement:
        decision_journal = args.output_json.with_suffix('.decisions.jsonl')
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with decision_journal.open('x'):
            pass
    config_path = Path(__file__).resolve().parents[1] / "config.example.yaml"
    config = yaml.safe_load(config_path.read_text())
    state, runs = synthetic_history(start)
    gate = ProjectionGate(ProjectionGateConfig(
        enforce_stability=True, audit_key_session_cadence=True,
        enforce_aerobic_exposure_cadence=args.scenario in {"perfect", "in_range_low", "in_range_high"},
        require_capacity_progression=args.scenario == "perfect" and args.days >= 90,
    ), fail_fast=True)
    weeks, stopped = [], False
    executions = []

    def observe(snapshot):
        print(f"{snapshot.generated_at.isoformat()} {snapshot.trigger} {snapshot.planning_seconds:.2f}s", flush=True)
        if decision_journal is not None:
            # Preserve each completed decision during expensive offline searches,
            # including the failing snapshot before the gate raises.
            with decision_journal.open('a') as stream:
                stream.write(json.dumps(_json_ready(snapshot))+'\n')
        gate.observe(snapshot)
        if args.shared_objective:
            failure = shared_objective_failure(snapshot)
            if failure is not None:
                gate.failures.append(failure)
                raise ProjectionGateTriggered(failure)

    contract = OpeningPlanExecution() if args.execution_plan == "opening" else None

    def observe_execution(run, latest):
        # Retain completed work even when a later fail-fast replan interrupts
        # simulate_adherence before it can return its reporting slices.
        executions.append({
            "occurred_at": run.start_time.isoformat(),
            "workout_type": run.workout_type.value,
            "distance_miles": run.distance_miles,
            "prescribed_low_miles": run.prescribed_low_miles,
            "prescribed_high_miles": run.prescribed_high_miles,
            "prescribed_target_miles": run.prescribed_target_miles,
            "adherence_note": run.adherence_note,
        })
        if contract:
            contract.observe(run, latest)

    from scripts.suffix_regrowth_prototype import regrowth_replan_policy
    try:
        with distance_refinement_policy(enabled=args.distance_refinement,budget=args.distance_refinement_budget,visible_days=args.distance_refinement_days,post_calendar_budget=args.post_calendar_distance_budget), (full_score_refinement_policy(budget=args.refinement_budget, starts=2 if args.fixed_tail_search else 3, fixed_tail=args.fixed_tail_search) if args.full_score_refinement else nullcontext()), shared_objective_policy(args.shared_objective), canonical_distance_policy(args.canonical_targets), opportunity_search_policy(args.opportunity_repair), finalized_scoring_policy(args.finalized_scoring) as scoring_measurements, calendar_neighbor_policy(enabled=args.calendar_neighbors, audit_opening=args.audit_opening_neighbors,
                                      max_passes=args.calendar_neighbor_passes,
                                      enforce_acceptance=args.candidate_acceptance) as calendar_builds, continuation_replan_policy(
                                      enabled=args.continuation_selection, calendar_records=calendar_builds,
                                      finalist_budget=args.continuation_finalists) as continuation_builds, replay_policy(args.prototype) as builds, expected_completion_build_policy(args.continuation_selection) as expected_builds, volume_path_policy(args.continuation_selection) as volume_metrics, (boundary_funding_policy() if funding_prototype else nullcontext(dict(evaluations=0,seconds=0.))) as funding_metrics, (regrowth_replan_policy() if regrowth_prototype else nullcontext([])) as regrowth_builds:
            try:
                weeks = projection.simulate_adherence(
                    state, runs, config, start, simulation_days=args.days,
                    weeks=max(1, (args.days + 6) // 7), replan_interval_days=1,
                    initial_schedule=None, emit_post_upload_replans=True, replan_observer=observe,
                    execution_schedule_selector=contract.select if contract else None,
                    execution_observer=observe_execution,
                    deterministic_profile=None if args.scenario == "perfect" else
                        projection.DeterministicAdherenceProfile(scenario=projection.DeterministicAdherenceScenario(args.scenario),
                            above_range_easy_fraction=args.above_range_easy_fraction),
                )
                gate.finalize(weeks)
            except ProjectionGateTriggered as error:
                stopped = True
                print(f"Stopped: {error.failure.code}: {error.failure.detail}", flush=True)
            except NoAcceptableCalendarError as error:
                stopped = True
                gate.failures.append(ProjectionGateFailure("no_acceptable_calendar", error.as_of, str(error), {}))
                print(f"Stopped: {error}", flush=True)
    finally:
        if contract:
            contract.close()
    report = build_projection_report(mode=args.scenario, seed=0, replan_interval_days=1,
                                    weeks=weeks, gate=gate, stopped_early=stopped)
    report['adaptation_audit'] = _json_ready(adaptation_transitions(gate.snapshots))
    if not stopped:
        reported_miles = sum(week.assumed_completed_miles for week in weeks)
        if abs(reported_miles - sum(x["distance_miles"] for x in executions)) > 1e-8:
            raise ValueError("Execution ledger does not match simulator completed mileage")
    report["realized_execution"] = {
        "start_at": start.isoformat(),
        "observed_until": (max([start]
                               + [x.generated_at for x in gate.snapshots]
                               + [datetime.fromisoformat(x["occurred_at"]) for x in executions])
                           if stopped else start + timedelta(days=args.days)).isoformat(),
        "complete": not stopped,
        "sessions": executions,
        "historical_sessions": [{"occurred_at": run.start_time.isoformat(),
                                 "workout_type": run.workout_type.value,
                                 "distance_miles": run.distance_miles} for run in runs],
        "limitation": "Execution observer covers prescribed deterministic scenarios; successful totals are checked against simulator output. Interrupted days are not complete observation days.",
    }
    report["synthetic_sandbox"] = {
        "scope": __doc__, "prototype": args.prototype, "initial_schedule": None,
        "history_count": len(runs), "history_weekly_miles": 15.75,
        "history_easy_miles": 3.75, "history_long_miles": 7.5,
        "history_pace_min_mile": 11, "history_quality_hard_minutes": 6,
        "requested_days": args.days, "config": config, "builds": builds,
        "opening_state": state.model_dump(mode="json"),
        "execution_plan": args.execution_plan,
        "opportunity_search_repair": args.opportunity_repair,
        "canonical_distance_targets": args.canonical_targets,
        "shared_selection_objective": args.shared_objective,
        'continuation_selection': {'enabled': args.continuation_selection, 'builds': continuation_builds},
        'full_score_refinement': {'enabled': args.full_score_refinement, 'budget': args.refinement_budget,
                                  'version': 'fixed_tail_calendar_descent_v1' if args.fixed_tail_search else 'full_score_calendar_descent_v1'},
        'distance_refinement': {'enabled': args.distance_refinement, 'budget': args.distance_refinement_budget, 'visible_days': args.distance_refinement_days,
                                'post_calendar_budget': args.post_calendar_distance_budget},
        'volume_path': {'enabled': args.continuation_selection, 'version': 'additive_date_path_v1', **volume_metrics},
        'funding_accounting': {'enabled': funding_prototype, 'version': 'physical_path_funding_v1' if funding_prototype else 'legacy_hard_quota',
            'scope': 'Full continuation only; original hard proposals and release gates unchanged.', **funding_metrics},
        'suffix_regrowth': {'enabled': regrowth_prototype, 'version': 'bounded_suffix_v1',
            'builds': regrowth_builds, 'scope': 'Offline current-input proposals, two reserved existing slots; unchanged gates.'},
        'expected_completion': {'enabled': args.continuation_selection,
                                'state_transition_version': 'daily_pace_v1',
                                'policy': 'required_for_continuation_selection', 'builds': expected_builds},
        'above_range_easy_fraction': args.above_range_easy_fraction,
        "finalized_scoring_trial": {"enabled": args.finalized_scoring, **scoring_measurements},
        "calendar_neighbor_prototype": calendar_builds if args.calendar_neighbors else None,
        "issued_contract": {"decisions": contract.decisions, "executions": contract.executions,
                            "limitation": "Simulation history uses the known issued contract; production matcher runs separately in memory."} if contract else None,
        "limitation": "One synthetic athlete; no upload parsing, personal data, physiology validation or migration test.",
    }
    report['delivery_contract_audit']=delivery_contract_audit(report)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(f"Wrote {args.output_json}", flush=True)
    if stopped or gate.failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
