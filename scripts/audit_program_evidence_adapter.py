"""Original-file bridge audit plus trace-backed identity transitions, not a scheduler."""
from argparse import ArgumentParser
from dataclasses import asdict, replace
from datetime import datetime, timedelta, time
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
from statistics import median
from time import perf_counter
from zoneinfo import ZoneInfo

from run_analysis.config import load_config
from run_analysis.delivered_training import dose_from_profile, continuous_phase_from_activity
from run_analysis.web.schemas import RecommendationResponse, WorkoutStep
from run_analysis.workload_accounting import Quantity
from scripts.audit_workload_contract import load_sessions
from scripts.audit_workload_expectation import app_session_types
from scripts.program_evidence_adapter import (
    QualityReference, evaluate_accounted, observed_run, forecast_run, reconcile_completion,
)
from scripts.program_objective_trial import TrialPolicy, DemandScenario
from scripts.shared_program_evaluator import ProgramContext
from scripts.timely_purpose_trial import PurposeTimingPolicy


def json_ready(value):
    if isinstance(value, datetime) or hasattr(value, 'isoformat'):
        return value.isoformat()
    raise TypeError(type(value).__name__)


from run_analysis.adaptive.evidence_history import load_history


def future_plan(origin, issued_at, history, reference, *, quality_shift=0):
    """A declared test calendar, not searched or recommended; no held-out response."""
    rows = []
    for offset in range(35):
        role = {1: 'long', 3+quality_shift: 'quality', 5: 'easy'}.get(offset % 7)
        if role is None:
            continue
        miles = {'long': 8.4, 'quality': 4.5, 'easy': 4.4}[role]
        donors = [r for r in history if r.role == role]
        if not donors:
            raise ValueError('Audit needs declared alternatives for missing role-rate evidence')
        duration = Quantity(min(r.demand[0].low/r.miles for r in donors)*miles,
            max(r.demand[0].high/r.miles for r in donors)*miles, basis='prior_role_duration_extrema_not_calibrated')
        rate = Quantity(min(r.demand[1].low/r.demand[0].high for r in donors),
            min(5., max(r.demand[1].high/max(r.demand[0].low, .001) for r in donors)),
            basis='prior_role_hr_rate_extrema_not_phase_model')
        rec = RecommendationResponse(generated_at=issued_at, fitness_state_as_of=issued_at,
            planned_for=origin+timedelta(days=offset, hours=7), workout_type='intervals' if role == 'quality' else role,
            title='Explicit bridge test dose', distance_target_miles=miles,
            distance_range_miles=(miles-.2, miles+.2), confidence='low', readiness='ready',
            structure=([WorkoutStep(instruction='3x5', phase='work', repetitions=3,
                work_duration_minutes=5, recovery_duration_minutes=2)] if role == 'quality' else []))
        rows.append(forecast_run(rec, key=f'candidate:{offset}', issued_at=issued_at,
            reference=reference, purpose='threshold', duration=duration,
            base_rate=rate, work_rate=rate, recovery_rate=rate))
    return tuple(rows)


def audit(database, config, as_of):
    started = perf_counter()
    digest = sha256(database.read_bytes()).hexdigest()
    zone = ZoneInfo(config['timezone_default'])
    reference = QualityReference('threshold', 'work_seconds', Quantity.point(900, 'test_3x5_total'),
        Quantity(290, 310, basis='declared_bout_band'), 'comparison_stage_not_historical_issued_contract')
    history, unresolved = load_history(database, config, as_of, reference)
    if unresolved:
        raise ValueError(f'Original-file bridge has unresolved sessions: {unresolved}')
    origin = datetime.combine(as_of.astimezone(zone).date(), time(), zone)
    easy = [r for r in history if r.role == 'easy' and r.start >= as_of-timedelta(days=28)]
    scales = tuple(4.4*median((r.demand[i].low+r.demand[i].high)/2/r.miles for r in easy)
                   for i in (0, 1))+(15.,)
    ctx = ProgramContext((), origin.date(), origin.date(), (origin+timedelta(days=21)).date(),
        (origin+timedelta(days=35)).date(), scales, TrialPolicy(), DemandScenario(), .01, PurposeTimingPolicy(1.))
    plan = future_plan(origin, as_of, history, reference)
    cases = []
    for demand_fraction, purpose_fraction in product((0., .5, 1.), repeat=2):
        context = replace(ctx, scenario=DemandScenario(bound_fraction=demand_fraction))
        result = evaluate_accounted(context, history, plan, as_of=as_of,
            timezone=config['timezone_default'], reference=reference, purpose_fraction=purpose_fraction)
        cases.append(dict(demand_fraction=demand_fraction, purpose_fraction=purpose_fraction,
            score=result['evaluation'].total_cost, constraints=result['constraints'],
            opening_coverage_bounds={k: asdict(v) for k, v in result['opening_coverage_bounds'].items()},
            quality_timing_loss=result['evaluation'].parts['timing']['quality']['loss'],
            constraint_eligible=result['constraint_eligible'], selection_ready=result['selection_ready']))

    # Hold the last trace out of earlier history. Copy its workload bounds into
    # an expected event ONLY to test accounting, not forecast accuracy. Commit
    # a canonical forecast distance; preserve unrounded actual GPS distance.
    actual = max(history, key=lambda r: r.start)
    past = tuple(r for r in history if r.key != actual.key)
    prior_asof = datetime.combine(actual.start.astimezone(zone).date(), time(), zone)
    expected = replace(actual, key='identity_forecast', source='planned', available_at=prior_asof,
                       miles=round(actual.miles, 1))
    next_origin = prior_asof+timedelta(days=1)
    plans = [future_plan(next_origin, prior_asof, past, reference, quality_shift=shift) for shift in (0, 1)]
    initial_ctx = replace(ctx, stage_start=prior_asof.date(), decision=prior_asof.date(),
        hard_end=(prior_asof+timedelta(days=21)).date(), end=(next_origin+timedelta(days=35)).date())
    transitions = []
    for demand_fraction, purpose_fraction in product((0., .5, 1.), repeat=2):
        context = replace(initial_ctx, scenario=DemandScenario(bound_fraction=demand_fraction))
        costs = []
        for remaining in plans:
            before = evaluate_accounted(context, past, (expected, *remaining), as_of=prior_asof,
                timezone=config['timezone_default'], reference=reference, purpose_fraction=purpose_fraction)
            after_history, after_plan = reconcile_completion(past, (expected, *remaining), actual,
                replaces_key=expected.key, as_of=actual.available_at)
            after = evaluate_accounted(context, after_history, after_plan, as_of=actual.available_at,
                timezone=config['timezone_default'], reference=reference, purpose_fraction=purpose_fraction)
            rollover = evaluate_accounted(context, after_history, after_plan, as_of=next_origin,
                timezone=config['timezone_default'], reference=reference, purpose_fraction=purpose_fraction)
            if abs(after['evaluation'].total_cost-rollover['evaluation'].total_cost) > 1e-10:
                raise AssertionError('Upload/rollover changed identical remaining evidence')
            costs.append((before['evaluation'].total_cost, after['evaluation'].total_cost))
        difference = (costs[0][0]-costs[1][0])-(costs[0][1]-costs[1][1])
        if abs(difference) > 1e-10:
            raise AssertionError(f'Identical completion changed relative future score: {difference}')
        transitions.append(dict(demand_fraction=demand_fraction, purpose_fraction=purpose_fraction,
            relative_score_difference_error=difference, unchanged_upload_to_rollover=True,
            forecast_distance=expected.miles, actual_distance=actual.miles,
            transition_kind='within_band_distance_matched_workload_bounds'))
    if sha256(database.read_bytes()).hexdigest() != digest:
        raise ValueError('Source database changed during audit')
    return dict(status='connected_offline_accounting_passed_not_scheduler_validation',
        as_of=as_of, history_count=len(history), history_roles={role: sum(r.role == role for r in history)
            for role in ('easy', 'long', 'quality', 'unknown')},
        quality_sessions=[dict(key=r.key, day=r.start.astimezone(zone).date(),
            bouts=len(r.bouts), work_minutes=asdict(r.demand[2]), credit=asdict(r.quality))
            for r in history if r.role == 'quality'],
        reference=asdict(reference), normalization=scales, cases=cases, transitions=transitions,
        database_unchanged=True, elapsed_seconds=perf_counter()-started, planner_influence=False,
        limitations=['Historical purpose not independently confirmed; credit remains bounded',
            'Illustrative future calendar; no search or chained planner stability validation',
            'Trace-backed transition uses canonical forecast distance and raw actual distance; not predictive accuracy',
            'Known constraints checked, recovery feasibility not established'])


if __name__ == '__main__':
    p = ArgumentParser(description=__doc__)
    p.add_argument('--database', type=Path, default=Path('data/run_analysis.sqlite'))
    p.add_argument('--config', type=Path, default=Path('config.yaml'))
    p.add_argument('--as-of', type=datetime.fromisoformat, default=datetime.fromisoformat('2026-09-26T04:00:00+00:00'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Choose a new output path')
    result = audit(args.database, load_config(args.config), args.as_of)
    with args.output.open('x') as stream:
        json.dump(result, stream, default=json_ready, allow_nan=False, indent=2)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'], history_roles=result['history_roles'],
        scenarios=len(result['cases']), transitions=len(result['transitions']),
        constraints=sorted({r['constraints']['status'] for r in result['cases']}),
        seconds=result['elapsed_seconds']), indent=2))
