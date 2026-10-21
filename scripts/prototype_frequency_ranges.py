"""Offline frequency envelope on existing controls and one real frozen search.

Reuse the 5/20/60 runner panel and established continuous calendar/dose search.
Periodic variants are labelled component controls, NOT daily replay or novice
admission. Frozen search has original readiness/HR forecasts and funding gates.
No online database mutation, publication or live deployment. Score unchanged at
the default zero midpoint weight; nonzero is explicit AS120 coaching utility.
"""
from argparse import ArgumentParser
from dataclasses import asdict, replace
from datetime import date, datetime, time, timedelta
import json
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

from scripts.check_easy_distribution_profiles import Profile, patterns
from scripts.check_explicit_distribution_tail import (
    fixture, periodic_demand, periodic_recovery_audit, incoming_demand_trace)
from scripts.replay_adaptive_search import load_decision
from scripts.research_manifest import freeze_sources
from run_analysis.adaptive.frequency_range_trial import (
    FrequencyRangePolicy, FrequencyStageTarget, assess_frequency, ranged_evaluator,
    FrequencyMidpointPreference, midpoint_evaluator)
from run_analysis.adaptive.continuous_search_adapter import search, NoEligibleProgram
from run_analysis.adaptive.stage_calendar_seeds import complete_stage_seeds
from run_analysis.adaptive.decimal_dose_search import calendar_representatives
from run_analysis.adaptive.new_model_daily_harness import planned_accounted, churn
from run_analysis.adaptive.decision_snapshot import evaluate_snapshot
from run_analysis.continuous_training_dose import ContinuousDoseReference
from run_analysis.config import load_config
from run_analysis.workload_accounting import Quantity


def variants(panel_path, policy, *, weights=(0., .01, .02, .05)):
    panel = json.loads(panel_path.read_text())
    case = load_decision(Path('output/memory-continuation-daily-2-20260929'), 6)
    context = case['state'].context
    hs, catalog = case['handoffs'], case['catalog']
    key = min((k for k in hs if catalog[k].role == 'easy'), key=lambda k: abs(catalog[k].miles-4.4))
    cardio = hs[key]['cardio']['estimate']/hs[key]['running_minutes']['estimate']
    output = []
    for row in panel['rows']:
        p = Profile(**row['profile'])
        if p.long:
            c, demand, cycle, _, _ = fixture(case, p, row['scales'], single_volume=True)
        else:
            c = replace(context, scales=tuple(row['scales']))
            def demand(role, miles):
                minutes = miles*p.pace
                return tuple(Quantity.point(v, 'synthetic easy-only control')
                             for v in (minutes, minutes*cardio, 0.))
            def cycle(pattern):
                return tuple((day, 'easy', miles) for day, miles in pattern)
        target = FrequencyStageTarget.freeze(p.name, c.decision, p.volume, policy=policy)
        candidates = []
        for pattern in patterns(p):
            entries = cycle(pattern)
            assert abs(sum(m for _, _, m in entries)-p.volume) < 1e-8
            days = tuple(c.decision+timedelta(days=week*7+d)
                         for week in range(4) for d, _, _ in entries)
            admission = assess_frequency((target,), (), days, decision=c.decision,
                coverage_end=c.decision+timedelta(days=28), policy=policy)
            score = periodic_demand(entries, demand, c)
            candidates.append(dict(pattern=pattern, entries=entries, count=len(entries),
                total_miles=p.volume, eligible=admission['eligible'],
                cost=score['fatigue']+score['short'], frequency_range=admission, **score))
        plain = min(candidates, key=lambda r: (r['cost'], r['entries']))
        allowed = [r for r in candidates if r['eligible']]
        chosen = min(allowed, key=lambda r: (r['cost'], r['entries'])) if allowed else None
        def summary(item):
            if item is None:
                return None
            return dict(item, recovery=periodic_recovery_audit(item['entries'], demand, c),
                easy_minutes=[round(m*p.pace, 1) for _, role, m in item['entries'] if role == 'easy'])
        sweep = []
        if chosen:
            for fraction in weights:
                preference = FrequencyMidpointPreference(chosen['fatigue'], 28, fraction)
                winner = min(allowed, key=lambda r: (
                    r['cost']+preference.assess(r['frequency_range'])['cost'], r['entries']))
                penalty = preference.assess(winner['frequency_range'])
                sweep.append(dict(edge_fraction=fraction, selected=summary(winner),
                    preference=penalty, combined_cost=winner['cost']+penalty['cost'],
                    fatigue_change_vs_range_percent=100*(winner['fatigue']/chosen['fatigue']-1)))
        output.append(dict(profile=asdict(p), target=asdict(target), candidates=len(candidates),
            permitted_candidates=len(allowed), unchanged_volume_comparison=True,
            available_periodic_counts=sorted({r['count'] for r in allowed}),
            baseline=summary(plain), selected=summary(chosen),
            fatigue_change_percent=(100*(chosen['fatigue']/plain['fatigue']-1) if chosen else None),
            midpoint_sweep=sweep,
            scope='Equal-volume bounded periodic COMPONENT control, not a full planner or novice admission'))
    return output


def frozen_search(root, index, policy, budget, edge_fraction=0.):
    case = load_decision(root, index)  # Original winner MUST reprice before any migration.
    state, catalog, handoffs, dates = (case[k] for k in ('state', 'catalog', 'handoffs', 'dates'))
    context = state.context
    stage = case['event']['stage_runtime']['stage']
    if stage['policy'].get('balance') is not None:
        raise ValueError('Use accepted v20, not the rejected AS118 inverse-curve progression experiment')
    target = FrequencyStageTarget.freeze(stage['key'], context.decision,
        stage['weekly_equivalent_miles'], policy=policy)
    zone = ZoneInfo(state.timezone)
    observed = tuple(r.start.astimezone(zone).date() for r in state.history)
    reference = ContinuousDoseReference('sandbox upload accounting')
    physical, cache = {}, {}
    def evaluate(program):
        program = tuple(sorted(program))
        if program not in cache:
            for day, key in program:
                if (day, key) not in physical:
                    physical[day, key] = planned_accounted(f'candidate:{day}:{key}',
                        datetime.combine(date.fromisoformat(day), time(18), zone),
                        catalog[key], handoffs[key], reference, available=state.as_of)
            r = evaluate_snapshot(state, tuple(physical[p] for p in program),
                tuple(k for _, k in program), reference=reference)
            cache[program] = dict(cost=r['evaluation'].total_cost, eligible=r['decision_eligible'],
                construction_eligible=r.get('construction_eligible', r['decision_eligible']),
                fatigue=r['evaluation'].weighted_demand_cost,
                delivery=r['evaluation'].delivery_cost, short=r['evaluation'].short_run_cost,
                progression_delivery=r.get('progression_delivery'))
        return cache[program]
    wrapped = ranged_evaluator(evaluate, (target,), observed, decision=context.decision,
        coverage_end=context.end, policy=policy)
    prior = tuple(tuple(p) for p in case['event']['program'])
    original = wrapped(prior)
    assert abs(original['cost']-case['archived_cost']) < 1e-12
    preference = FrequencyMidpointPreference(original['fatigue'],
        (context.end-context.decision).days, edge_fraction)
    wrapped = midpoint_evaluator(wrapped, preference)
    baseline = wrapped(prior)
    supported = {k: catalog[k] for k in handoffs}
    runtime = case['event']['stage_runtime']['target_history'][-1]
    seed_policy = context.policy
    support = context.easy_duration_policy.supporting_easy_miles_per_7
    if runtime['ramp_days']:
        seed_policy = replace(seed_policy, weekly_miles=runtime['volume_goal'])
        support = runtime['support_goal']
    seeds = complete_stage_seeds(dates, supported, seed_policy, supporting_miles=support)
    roles = {k: d.role for k, d in supported.items()}
    seed_keys = set()
    easy = [k for k in supported if roles[k] == 'easy']
    for target_miles in (context.policy.ordinary_miles, context.policy.ordinary_miles/2):
        if easy:
            seed_keys.add(min(easy, key=lambda k: (abs(supported[k].miles-target_miles), k)))
    for role in ('long', 'quality'):
        keys = [k for k in supported if roles[k] == role]
        if keys:
            seed_keys.add(max(keys, key=lambda k: (
                sum(s for kind, s in supported[k].phases if kind == 'work')
                if role == 'quality' else supported[k].miles, k)))
    started = perf_counter()
    try:
        result = search(dates=dates, keys=tuple(supported), previous=prior,
            evaluate=wrapped, budget=budget, hard_end=context.hard_end.isoformat(), roles=roles,
            seed_programs=seeds, seed_keys=tuple(sorted(seed_keys)),
            seed_width=6, starts=3, stability_tolerance=None,
            dose_miles={k: d.miles for k, d in supported.items()},
            exploration_keys=calendar_representatives(supported,
                ordinary=context.policy.ordinary_miles, support=support),
            recombination_dates=(context.taper_start.isoformat(),))
    except NoEligibleProgram as error:
        return dict(status='no_eligible_program_found', error=str(error), target=asdict(target),
                    baseline=baseline, seconds=perf_counter()-started, evaluations=len(cache))
    fresh = wrapped(result['program'])
    assert fresh['eligible'] and abs(fresh['cost']-result['cost']) < 1e-12
    hard = [(d, catalog[k].role, catalog[k].miles) for d, k in result['program']
            if d < context.hard_end.isoformat()]
    def trace(program):
        r = evaluate_snapshot(state, tuple(physical[p] for p in program),
            tuple(k for _, k in program), reference=reference)
        return incoming_demand_trace(r['evaluation'], context)
    return dict(status='frozen_search_complete', result=result, target=asdict(target),
        baseline=baseline, fresh_reprice=fresh, seconds=perf_counter()-started,
        preference=asdict(preference), baseline_session_trace=trace(prior),
        selected_session_trace=trace(tuple(tuple(p) for p in result['program'])),
        snapshot_fingerprint=state.fingerprint, hard_schedule=hard,
        hard_runs=len(hard), hard_miles=sum(m for _, _, m in hard),
        original_hard_miles=sum(catalog[k].miles for d, k in prior if d < context.hard_end.isoformat()),
        churn_vs_pre_policy_plan=churn(prior, result['program'], context.decision, catalog),
        churn_scope='Policy-adoption difference, NOT compliant daily churn or stability acceptance',
        scope='Original evidence/search/admission; explicit midpoint utility only; no publication')


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('output/v20-perfect-daily-7-20261002'))
    parser.add_argument('--index', type=int, default=1)
    parser.add_argument('--panel', type=Path, default=Path('output/easy-profile-volume-curve-20260929/report.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budget', type=int, default=10000)
    parser.add_argument('--midpoint-edge-fraction', type=float, default=0.,
        help='Explicit coaching utility at a range edge relative to frozen fatigue; default zero')
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    event = json.loads((args.root/f'decision-{args.index:04d}.json').read_text())
    variant_root = Path('output/memory-continuation-daily-2-20260929')
    variant_event = json.loads((variant_root/'decision-0006.json').read_text())
    manifest = freeze_sources(args.output/'sources.zip', config=load_config(), entrypoint=__name__,
        inputs=(args.panel, args.root/f'decision-{args.index:04d}.json',
                args.root/event['snapshot_file'], args.root/'scenario.json',
                variant_root/'decision-0006.json', variant_root/variant_event['snapshot_file'],
                variant_root/'scenario.json'))
    policy = FrequencyRangePolicy()
    panel = variants(args.panel, policy)
    with (args.output/'variants.json').open('x') as stream:
        json.dump(dict(policy=asdict(policy), profiles=panel, manifest=manifest), stream,
                  indent=2, default=str, allow_nan=False)
    print(json.dumps(dict(variants=[dict(name=r['profile']['name'],
        before=r['baseline']['count'], after=r['selected']['count'] if r['selected'] else None,
        fatigue_change_percent=r['fatigue_change_percent']) for r in panel])), flush=True)
    result = frozen_search(args.root, args.index, policy, args.budget, args.midpoint_edge_fraction)
    with (args.output/'frozen-search.json').open('x') as stream:
        json.dump(dict(result, policy=asdict(policy), manifest=manifest), stream,
                  indent=2, default=str, allow_nan=False)
    print(json.dumps(dict(status=result['status'], seconds=result['seconds'],
        runs=result.get('hard_runs'), miles=result.get('hard_miles'),
        baseline_miles=result.get('original_hard_miles'),
        frequency=result.get('fresh_reprice', {}).get('frequency_range'))), flush=True)


if __name__ == '__main__':
    main()
