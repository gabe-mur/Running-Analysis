"""Declared decision scenarios with original-file history, not actual readiness advice."""
from collections import Counter
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from time import perf_counter
import json

from scripts.audit_program_delivery import ORIGIN, ledger
from scripts.audit_recency_demand import audit as load_audit, future_events, json_ready
from run_analysis.config import load_config
from run_analysis.development_report import ProgramOption
from run_analysis.program_constraints import DateRestriction
from run_analysis.recency_demand import CHANNELS, DemandEvent, DemandTerm
from run_analysis.session_decision import CoachingContract, CurrentContext, DoseLimits, SessionAlternative, decide_session
from run_analysis.workload_accounting import Quantity


def run_rest_crossover_probe():
    """Small score diagnostic, NOT an athlete replay or a recovery calibration.

    Distinguish permanently omitting today's training from delivering it on a
    different date. The opening history is an abstract dose stress test; its
    large multiples must not be presented as plausible sessions or safe limits.
    Other future sessions are fixed, so this is not a complete calendar search.
    """
    from datetime import date
    from scripts.program_objective_trial import (
        TrialSession, TrialPolicy, DemandScenario, components, score,
    )

    origin = date(2026, 9, 27)
    policy = TrialPolicy()
    scenarios = tuple((memory, DemandScenario(memory_days=memory)) for memory in (1., 2., 4.))
    scales = (44., 44., 15.)
    q = lambda value: Quantity.point(value, 'synthetic R1 probe; not observed physiology')

    def session(key, day, miles, role, minutes, cardio, work=0., source='planned'):
        return TrialSession(key, origin+timedelta(days=day), miles, role,
            tuple(q(value) for value in (minutes, cardio, work)), min(1., work/15), source)

    motif = ((0, 4.4, 'easy', 44., 44., 0.),
             (2, 8.4, 'long', 84., 84., 0.),
             (5, 4.5, 'quality', 45., 65., 15.))
    future = tuple(session(f'{week}:{role}', week*7+day, miles, role, minutes, cardio, work)
        for week in range(5) for day, miles, role, minutes, cardio, work in motif)
    fixed = tuple(row for row in future if row.day != origin)
    options = {'omit': fixed}
    for day in (0, 1, 3, 4, 6):
        for miles in (1., 2.2, 3.5, 4.4):
            options[f'day+{day}/{miles}'] = (
                session('candidate', day, miles, 'easy', miles*10, miles*10), *fixed)

    results = []
    for memory, scenario in scenarios:
        for multiple in (0., .5, 1., 2., 4., 8., 16., 32., 64.):
            history = (() if multiple == 0 else
                (session('stress_probe', -1, 4.4, 'easy', 44*multiple, 44*multiple,
                         source='observed'),))
            parts = {key: components(history, programme, stage_start=origin, decision=origin,
                end=origin+timedelta(days=35), scales=scales, policy=policy, scenario=scenario)
                for key, programme in options.items()}
            for exchange in (.005, .01, .02):
                prices = {key: score(value, exchange) for key, value in parts.items()}
                today = min((key for key in prices if key.startswith('day+0/')), key=prices.get)
                other = min((key for key in prices if not key.startswith('day+0/')), key=prices.get)
                removal = min(('omit', today), key=prices.get)
                results.append(dict(memory_days=memory, exchange=exchange,
                    preceding_dose_multiple=multiple,
                    omission_only_winner=removal, whole_panel_winner=min(prices, key=prices.get),
                    best_today=today, best_no_today=other,
                    no_today_minus_today=prices[other]-prices[today],
                    omission_minus_today=prices['omit']-prices[today],
                    # Same physical dose/benefit, different date: exchange weight
                    # cannot reverse this pair for any positive exchange.
                    full_tomorrow_minus_today=prices['day+1/4.4']-prices['day+0/4.4'],
                    full_tomorrow_minus_today_delivery=(parts['day+1/4.4']['delivery_loss']
                                                      - parts['day+0/4.4']['delivery_loss'])))
    return dict(protocol='R1_fixed_stage_run_rest_diagnostic_v1', cases=results,
        options=len(options), normalized_scales=scales, physiology_validated=False,
        policy_accepted=False, limitations=[
            'Synthetic five-cycle fixture and abstract preceding-demand multipliers, not athlete advice',
            'Existing experimental aggregate score; no calibrated fatigue cutoff',
            'Remaining calendar fixed; movement panel is not full search or sequence suitability',
            'Current response, stage acceptance and existing hard guards are not assessed here',
            'The chosen score winner is a diagnostic, not a required training outcome'])


def training_logic_panel(*, role_rates=None, easy_duration_policy=None):
    """Frozen R1 coaching counterexamples plus unscored physical diagnostics.

    Uses the actual shared evaluator with timely purpose, not a new optimizer.
    See the predeclared protocol in recovery_scheduler_reconciliation.md.
    """
    from datetime import date
    from scripts.program_objective_trial import TrialSession, TrialPolicy, DemandScenario
    from scripts.shared_program_evaluator import ProgramContext, ProgramCandidate, evaluate_program
    from scripts.timely_purpose_trial import PurposeTimingPolicy

    origin = date(2026, 9, 27)
    # Optional original-file rates replace illustrative minutes/cardio together.
    # They are scenario inputs, not inferred tolerance or a forecast calibration.
    if role_rates is None:
        role_rates = {role: (10., 65/4.5 if role == 'quality' else 10.)
                      for role in ('easy', 'long', 'quality')}
    from math import isfinite
    if (set(role_rates) != {'easy', 'long', 'quality'} or
            any(len(values) != 2 or any(not isfinite(v) or v <= 0 for v in values)
                for values in role_rates.values())):
        raise ValueError('Positive duration/cardio per-mile scenario rates for each role required')
    q = lambda value: Quantity.point(value, 'R1 illustrative programme; not observed athlete data')
    def session(key, offset, miles, role, source='planned'):
        return TrialSession(key, origin+timedelta(days=offset), miles, role,
            (q(miles*role_rates[role][0]), q(miles*role_rates[role][1]),
             q(15. if role == 'quality' else 0.)),
            1. if role == 'quality' else 0., source)

    def programme(easy_days=(2, 6), cycles=range(5), source='planned'):
        return tuple(session(f'{source}:{cycle}:{key}', cycle*7+day, miles, role, source)
            for cycle in cycles for key, day, miles, role in (
                ('long', 0, 8.4, 'long'), ('early', easy_days[0], 3.5, 'easy'),
                ('quality', 4, 4.5, 'quality'), ('late', easy_days[1], 3.6, 'easy')))

    history = programme(cycles=range(-2, 0), source='observed')
    choices = dict(distributed=programme(), compressed=programme((1, 2)),
                   before_quality=programme((3, 6)))
    choices['underfunded'] = tuple(r for r in choices['distributed'] if r.role != 'easy')
    # Same total easy volume and unchanged long/quality dates. These are fixed
    # counterexamples, not another scheduler or a search over training frequency.
    key_work = tuple(r for r in choices['distributed'] if r.role != 'easy')
    for name, doses in (
        ('near_floor_fragments', ((1, 1.), (2, 1.), (3, 1.), (6, 4.1))),
        ('above_band_split', ((1, 2.3), (3, 2.4), (6, 2.4))),
        ('concentrated_easy', ((2, 7.1),)),
        ('below_floor', ((1, .5), (2, 1.), (3, 1.), (6, 4.6))),
    ):
        choices[name] = (*key_work, *(session(f'{name}:{cycle}:{day}', cycle*7+day, miles, 'easy')
                                     for cycle in range(5) for day, miles in doses))
    policy = TrialPolicy(weekly_miles=20., ordinary_miles=3.5, ordinary_per_7=2.)

    def physical(rows, *, end=35):
        all_rows = (*history, *rows)
        local = lambda r: (r.day-origin).days
        def peak(span, axis):
            return max((sum(r.demand[axis].low for r in all_rows
                            if day-span < local(r) <= day)
                        for day in range(end)), default=0.)
        days = sorted(local(r) for r in all_rows)
        gaps = [b-a for a,b in zip(days, days[1:]) if b >= 0]
        lead = [dict(key=r.key, day=local(r), preceding_two_day_minutes=sum(
            p.demand[0].low for p in all_rows if 0 < (r.day-p.day).days <= 2),
            preceding_one_day_minutes=sum(p.demand[0].low for p in all_rows
                                          if (r.day-p.day).days == 1))
            for r in rows if r.role == 'quality']
        return dict(miles=sum(r.miles for r in rows), minutes=sum(r.demand[0].low for r in rows),
            runs=len(rows), peak_3day_minutes=peak(3, 0), peak_7day_work_minutes=peak(7, 2),
            longest_bracketed_gap_days=max(gaps, default=None), quality_lead_ins=lead,
            max_quality_lead_in_minutes=max((r['preceding_two_day_minutes'] for r in lead), default=0.),
            max_quality_prior_day_minutes=max((r['preceding_one_day_minutes'] for r in lead), default=0.))

    scales = (3.5*role_rates['easy'][0], 3.5*role_rates['easy'][1], 15.)
    def evaluate(rows, memory, exchange, *, end=35, downstream=(), downstream_end=None,
                 history_override=None):
        context = ProgramContext(history if history_override is None else history_override,
            origin, origin, origin+timedelta(days=21),
            origin+timedelta(days=end), scales, policy,
            DemandScenario(memory_days=memory), exchange, PurposeTimingPolicy(1.),
            downstream=downstream, downstream_end=(origin+timedelta(days=downstream_end)
                                                   if downstream_end is not None else None),
            downstream_basis='explicit fixed diagnostic background' if downstream_end is not None else '')
        context = replace(context, easy_duration_policy=easy_duration_policy)
        candidate = ProgramCandidate(tuple(r for r in rows if r.day < context.hard_end),
            tuple(r for r in rows if r.day >= context.hard_end), 'fixed R1 comparison continuation')
        result = evaluate_program(context, candidate)
        return dict(cost=result.total_cost, delivery_cost=result.delivery_cost,
            demand_cost=result.parts['demand_cost'], fingerprint=result.fingerprint,
            shortfall_by_purpose=result.parts['shortfall_by_purpose'],
            targets=result.parts['targets'], delivery=result.parts['delivery'],
            downstream_accounting=result.downstream_accounting,
            easy_duration=result.parts.get('easy_duration'),
            selection_ready=result.selection_ready)

    metrics = {key: physical(rows) for key, rows in choices.items()}
    cases = []
    for memory in (1., 2., 4.):
        for exchange in (.005, .01, .02):
            scores = {key: evaluate(rows, memory, exchange) for key, rows in choices.items()}
            cases.append(dict(memory_days=memory, exchange=exchange, scores=scores,
                maintained_over_underfunded=scores['distributed']['cost'] < scores['underfunded']['cost'],
                spread_over_compressed=scores['distributed']['cost'] < scores['compressed']['cost'],
                before_quality_preferred=scores['before_quality']['cost'] < scores['distributed']['cost']))

    # Identical prefix and per-purpose totals. No future work is omitted from a
    # claimed comparison: report explicitly that day 36 is outside the short scope.
    prefix = tuple(r for r in choices['distributed'] if (r.day-origin).days < 28)
    boundary = {key: (*prefix, session('final_easy', day, 3.5, 'easy'))
                for key, day in (('earlier', 30), ('edge', 34))}
    following = session('following_long', 36, 8.4, 'long')
    boundary_cases = []
    for memory in (1., 2., 4.):
        short = {key: evaluate(rows, memory, .01) for key, rows in boundary.items()}
        legacy_extended = {key: evaluate((*rows, following), memory, .01, end=42)
                           for key, rows in boundary.items()}
        scope_only = {key: evaluate(rows, memory, .01, downstream_end=42)
                      for key, rows in boundary.items()}
        extended = {key: evaluate(rows, memory, .01, downstream=(following,), downstream_end=42)
                    for key, rows in boundary.items()}
        more_coverage = {key: evaluate(rows, memory, .01, downstream=(following,), downstream_end=49)
                         for key, rows in boundary.items()}
        winner = lambda values: min(values, key=lambda key: values[key]['cost'])
        boundary_cases.append(dict(memory_days=memory, short=short, extended=extended,
            legacy_extended=legacy_extended, scope_only=scope_only, more_coverage=more_coverage,
            short_winner=winner(short), extended_winner=winner(extended),
            rank_reversal=winner(short) != winner(extended)))
    # Predeclared run/rest alternatives. No selected prescription or physiological
    # ceiling: compare marginal loss and demand, with the same onward work priced.
    opening = tuple(session(f'marginal:{cycle}:{role}:{day}', cycle*7+day, miles, role)
        for cycle in range(5) for day, miles, role in (
            (0, 3.5, 'easy'), (2, 4.5, 'quality'), (4, 8.4, 'long'), (6, 3.6, 'easy')))
    fixed = opening[1:]
    options = dict(continue_today=opening,
        reduce_today=(replace(opening[0], miles=2.5, demand=tuple(
            Quantity.point(v.low*2.5/3.5, 'proportional scenario reduction')
            for v in opening[0].demand)), *fixed),
        omit_today=fixed,
        defer_one_day=(replace(opening[0], day=origin+timedelta(days=1)), *fixed))
    options['redistribute'] = tuple(
        session(row.key, (row.day-origin).days, 4.6, 'easy')
        if row.day == origin+timedelta(days=6) else row
        for row in options['reduce_today'])
    # The preceding long is a counterfactual placement, not the athlete's history.
    # All other lead-in work is held fixed. An added easy at -2 deliberately tests
    # stacking without inventing a 32x/64x run or assuming a streak is prohibited.
    lead_in = tuple(row for row in history if row.day < origin-timedelta(days=1))
    background = (session('background_long', 36, 8.4, 'long'),)
    marginal_cases = []
    for preceding_miles, stacked in ((8.4, False), (8.5, False), (8.6, False),
                                    (9.2, False), (8.4, True)):
        past = (*lead_in, session('preceding_long', -1, preceding_miles, 'long', 'observed'))
        if stacked:
            past += (session('extra_easy', -2, 3.5, 'easy', 'observed'),)
        for memory in (1., 2., 4.):
            for exchange in (.005, .01, .02):
                scores = {key: evaluate(rows, memory, exchange, downstream=background,
                    downstream_end=42, history_override=past) for key, rows in options.items()}
                base = scores['continue_today']
                contrasts = {}
                for key, row in scores.items():
                    loss = row['delivery_cost']-base['delivery_cost']
                    saved = base['demand_cost']-row['demand_cost']
                    short_cost = ((row['easy_duration'] or {}).get('cost', 0.)
                                  -(base['easy_duration'] or {}).get('cost', 0.))
                    contrasts[key] = dict(delivery_loss_change=loss, demand_saved=saved,
                        short_run_cost_change=short_cost,
                        shortfall_change_by_purpose={purpose: value-base['shortfall_by_purpose'][purpose]
                            for purpose, value in row['shortfall_by_purpose'].items()},
                        score_change=row['cost']-base['cost'],
                        crossover_exchange=(loss+short_cost)/saved if saved > 0 else None)
                marginal_cases.append(dict(preceding_miles=preceding_miles, stacked=stacked,
                    memory_days=memory, exchange=exchange,
                    winner=min(scores, key=lambda key: scores[key]['cost']),
                    contrasts=contrasts))
    return dict(protocol='R1_training_logic_panel_v3_marginal_actions', metrics=metrics, cases=cases,
        boundary_cases=boundary_cases,
        marginal_cases=marginal_cases, scenario_rates=role_rates, normalized_scales=scales,
        marginal_options={key: dict(miles=sum(r.miles for r in rows), runs=len(rows),
            first_dates=[str(r.day) for r in rows[:4]]) for key, rows in options.items()},
        predeclared_checks_pass=all(r['maintained_over_underfunded'] and r['spread_over_compressed'] for r in cases),
        physiology_validated=False, policy_accepted=False, limitations=[
            'Fixed illustrative prescriptions and assumed HR costs; no new athlete observations',
            'Concentration measures are physical diagnostics, not measured fatigue or injury cutoffs',
            'No full search, daily/upload replay, readiness integration or stage acceptance',
            'Passing preference controls is necessary but insufficient for defensible programme selection'])


def short_easy_curve_experiment(role_rates, *, preserve_easy_volume=False):
    """Nine predeclared curves on fixed calendars, no scheduling or weight fitting."""
    from scripts.program_objective_trial import EasyDurationPolicy
    results = []
    for free in (20., 25., 30.):
        for peak in (.005, .01, .02):
            # Carry the panel's existing 3.5 * 2 reference volume, not a new
            # preferred size or mandatory count. Never refit it to a candidate.
            policy = EasyDurationPolicy(10., free, peak, 'R1 bounded curve experiment; not calibrated',
                                        supporting_easy_miles_per_7=7. if preserve_easy_volume else None)
            report = training_logic_panel(role_rates=role_rates, easy_duration_policy=policy)
            row = next(r for r in report['cases'] if r['memory_days'] == 2 and r['exchange'] == .01)
            scores = row['scores']
            eligible = {key: value for key, value in scores.items()
                if not value['easy_duration']['below_minimum_keys']
                and not value['easy_duration']['uncertain_minimum_keys']}
            marginal = next(r for r in report['marginal_cases'] if r['memory_days'] == 2
                and r['exchange'] == .01 and r['preceding_miles'] == 8.4 and not r['stacked'])
            results.append(dict(penalty_free_minutes=free, maximum_penalty=peak,
                fixed_panel_winner=min(eligible, key=lambda key: eligible[key]['cost']),
                score_differences={key: value['cost']-scores['distributed']['cost']
                                   for key, value in scores.items()},
                penalty_by_candidate={key: value['easy_duration']['cost'] for key, value in scores.items()},
                excluded_keys={key: value['easy_duration']['below_minimum_keys']
                               for key, value in scores.items() if value['easy_duration']['below_minimum_keys']},
                redistribution_vs_original=marginal['contrasts']['redistribute'],
                failed_prior_controls=[dict(memory_days=r['memory_days'], exchange=r['exchange'],
                    maintained_over_underfunded=r['maintained_over_underfunded'],
                    spread_over_compressed=r['spread_over_compressed'],
                    underfunded_minus_distributed=r['scores']['underfunded']['cost']-r['scores']['distributed']['cost'])
                    for r in report['cases'] if not (r['maintained_over_underfunded'] and r['spread_over_compressed'])],
                previous_controls_pass_all_scenarios=report['predeclared_checks_pass']))
    return dict(protocol='R1_near_minimum_squared_curve_v1', cases=results,
        supporting_easy_miles_per_7=7. if preserve_easy_volume else None,
        demand_memory_days=2., exchange=.01, minimum_minutes=10.,
        default_changed=False, physiological_validation=False, scheduler_run=False)


def original_signal_training_panel(database, config, as_of):
    """Original HR data supplies dose scales, never a recovered/not-recovered label.

    The fixed calendars remain counterfactual. This is not a chronological replay:
    the bridge assumes completed files were available and uses current role labels.
    """
    from hashlib import sha256
    from statistics import median
    from run_analysis.continuous_training_dose import ContinuousDoseReference
    from run_analysis.training_signals import HeartRateReference, observed_cardio_trace
    from run_analysis.movement import classify_movement
    from scripts.audit_program_evidence_adapter import load_history
    from scripts.program_evidence_adapter import QualityReference
    from scripts.continuous_program_adapter import with_continuous_trace

    started = perf_counter()
    digest = sha256(database.read_bytes()).hexdigest()
    reference = ContinuousDoseReference('R1 original-file scale scenario, not outcome fitting')
    hr = HeartRateReference(config['resting_hr'], config['max_hr'], 'configured athlete reference')
    old = QualityReference('threshold', 'work_seconds', Quantity.point(900, 'bridge only'),
        Quantity(290, 310, basis='bridge only'), 'old credit discarded by continuous adapter')

    def convert(row, activity):
        intervals = classify_movement(activity.trackpoints, config['moving_time']).intervals
        trace = observed_cardio_trace(activity, intervals, hr, reference.policy,
                                     include_spans=True, compute_density=False)
        return with_continuous_trace(row, trace['normalized_spans'], reference=reference,
            trace_key=row.key, trace_source='observed', elapsed_seconds=trace['elapsed_seconds'],
            outside_reference_seconds=trace['exclusions']['outside_reference_seconds'])[0]

    history, unresolved = load_history(database, config, as_of, old, run_transform=convert,
                                      since=as_of-timedelta(days=28))
    if unresolved:
        raise ValueError(f'Unresolved original-file evidence: {unresolved}')
    rates, sources = {}, {}
    for role in ('easy', 'long', 'quality'):
        rows = [r for r in history if r.role == role]
        if not rows:
            raise ValueError(f'Missing {role} rate evidence; no silent fallback')
        rates[role] = tuple(median((r.demand[i].low+r.demand[i].high)/2/r.miles
                                   for r in rows) for i in (0, 1))
        sources[role] = [r.key for r in rows]
    panel = training_logic_panel(role_rates=rates)
    if sha256(database.read_bytes()).hexdigest() != digest:
        raise ValueError('Original database changed during read-only panel')
    return dict(panel, original_evidence=dict(as_of=as_of, sessions=len(history),
        source_keys=sources, accounting_key=reference.accounting_key,
        rate_basis='median of per-mile midpoints of recorded bounds; scenario, not prediction intervals'),
        database_unchanged=True, elapsed_seconds=perf_counter()-started,
        limitations=[
            'Observed HR-derived rates; counterfactual calendars, not the athlete actual lead-in or recommendation',
            'Fixed 15-minute future quality work is a diagnostic assumption, not copied from every observed workout',
            'Midpoint/median rates discard predictive variability; not calibrated uncertainty or tolerance',
            *panel['limitations'][1:]])


def audit(database, config, as_of):
    started = perf_counter()
    source = load_audit(database, config, as_of)
    history = tuple(DemandEvent('actual:'+r['key'], r['day'], tuple(
        DemandTerm(c, f"actual:{r['key']}:{c}", 1., r['quantities'][c]) for c in CHANNELS),
        r['accounting_key'], r['role'], r['structure'], 'observed') for r in source['sessions'])
    motif = [(0, 4.4, 'easy', 4.4, None, None), (3, 8.4, 'long', 8.4, None, None),
             (6, 4.5, 'quality', 2., 'threshold', (300., 300., 300.))]
    exposures = tuple(e for e in ledger(motif) if e.source == 'planned' and e.day < ORIGIN+timedelta(days=28))
    demands = tuple(future_events(motif, source['rates'], history[0].accounting_key))
    target_key = demands[0].key
    def option(key, miles=4.4, offset=0):
        e = replace(exposures[0], miles=miles, easy_miles=miles, day=ORIGIN+timedelta(days=offset))
        d = replace(demands[0], day=e.day, terms=tuple(
            replace(t, coefficient=t.coefficient*miles/4.4) if t.channel != 'work_minutes' else t
            for t in demands[0].terms))
        return SessionAlternative(key, ProgramOption(key, (e, *exposures[1:]), (d, *demands[1:])), target_key, 'controlled_easy')
    q = lambda lo, hi: Quantity(lo, hi, basis='predeclared_sandbox_bounds_not_validated_training_limits')
    contract = CoachingContract(as_of-timedelta(days=1), 'sandbox_decision_contract_not_actual_issued_plan',
        'easy', 'controlled_easy', DoseLimits(q(4., 4.5)), DoseLimits(q(3., 3.5)), DoseLimits(q(4.6, 4.8)), trial_target=4.8)
    issued = option('issued')
    alternatives = [option('reduced_3_0', 3.), option('reduced_3_5', 3.5), option('cosmetic_4_3', 4.3),
                    option('trial_4_8', 4.8), option('tomorrow', offset=1)]
    context = lambda state: CurrentContext(ORIGIN, as_of, state, 'hypothetical_current_report_not_athlete_report')
    n = DateRestriction('reserved', ORIGIN, ORIGIN+timedelta(days=1), as_of, 'reserved_rest', 'hypothetical_reserved_date')
    extra = DemandEvent('hypothetical_extra_easy', ORIGIN-timedelta(days=1), tuple(
        DemandTerm(c, 'extra_fixture:'+c, 1., Quantity.point(value, 'counterfactual_not_recorded'))
        for c, value in [('miles', .5), ('running_minutes', 6.), ('cardio', 8.), ('work_minutes', 0.)]),
        history[0].accounting_key, 'easy', None, 'observed')
    scenarios = dict(ordinary_missing_reports={}, current_little_tired=dict(context=context('little_tired')),
        health_concern=dict(context=context('health_concern')),
        reserved_date=dict(restrictions=[n]), requested_dose_trial=dict(trial_requested=True),
        unexpectedly_hard_previous=dict(unexpected_previous_response=True),
        normal_current_after_hard_previous=dict(unexpected_previous_response=True, context=context('usual')),
        conflicting_context=dict(context=context('conflicting')),
        counterfactual_extra_exposure=dict(history=(*history, extra)))
    results = {}
    for name, changes in scenarios.items():
        args = dict(as_of=as_of, history=history, expected_history=history, timezone=config['timezone_default'])
        args.update(changes)
        r = decide_session(contract, issued, alternatives, **args)
        results[name] = {k: v for k, v in r.items() if k not in {'option_checks', 'downstream_demand_comparisons'}}
        results[name]['option_checks'] = {k: dict(problems=v['problems'], fits=v.get('fits'),
            constraints=v.get('constraints', {}).get('status'), lead_in=v.get('lead_in_channel_orders'))
            for k, v in r['option_checks'].items()}
        results[name]['downstream_order_counts'] = dict(Counter(x['recency_demand_order'] for x in r.get('downstream_demand_comparisons', [])))
    return dict(as_of=as_of, original_sessions=len(history), cases=results, elapsed_seconds=perf_counter()-started,
        planner_influence=False, physiological_validation=False,
        limitations=['Future programs and dose limits are predeclared test assumptions, not actual prescriptions',
            'All current-context/restriction reports are hypothetical, not extracted athlete reports',
            'Historical files unchanged; extra-exposure case is explicitly counterfactual',
            'No optimal reduction percentage, original program suitability or scheduler stability established'])


if __name__ == '__main__':
    from argparse import ArgumentParser
    p = ArgumentParser(description=__doc__)
    p.add_argument('--database', type=Path, required=True)
    p.add_argument('--config', type=Path, default=Path('config.yaml'))
    p.add_argument('--as-of', type=datetime.fromisoformat, required=True)
    args = p.parse_args()
    print(json.dumps(audit(args.database, load_config(args.config), args.as_of), default=json_ready, allow_nan=False, indent=2))
