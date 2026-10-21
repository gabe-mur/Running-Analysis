"""Bounded synthetic equal-volume controls; no search, publication or clearance.

Moderate/high profiles use the guarded shared evaluator. The all-easy low-volume
profile uses a separately labelled steady-periodic demand/short-cost diagnostic:
the current full contract cannot disable long/quality purposes.
"""
from argparse import ArgumentParser
from dataclasses import asdict, dataclass, replace
from datetime import timedelta
from itertools import combinations
import json
from math import exp, expm1, isfinite
from pathlib import Path
from time import perf_counter

from scripts.replay_adaptive_search import load_decision
from scripts.research_manifest import freeze_sources
from run_analysis.config import load_config
from run_analysis.adaptive.program_objective_trial import TrialSession
from run_analysis.adaptive.shared_program_evaluator import ProgramCandidate, evaluate_program
from run_analysis.adaptive.volume_benefit_trial import linear_volume_loss,volume_benefit_cost
from run_analysis.workload_accounting import Quantity


@dataclass(frozen=True)
class Profile:
    name: str
    volume: float
    pace: float
    long: float
    quality: float
    work: float
    history_easy_count: int

    @property
    def support(self):
        return round(self.volume-self.long-self.quality, 1)


def allocations(units, count, pace):
    """Exact tenths: balanced plus one shorter/longer run; NOT exhaustive grid."""
    base, extra = divmod(units, count)
    choices = {tuple(base+(i in bigger) for i in range(count))
               for bigger in combinations(range(count), extra)}
    if count > 1:
        anchors = {round(15/pace*10), round(25/pace*10), round(1.5*units/count)}
        for anchor in anchors:
            rest, remainder = divmod(units-anchor, count-1)
            for position in range(count):
                others = [i for i in range(count) if i != position]
                for bigger in combinations(others, remainder):
                    choices.add(tuple(anchor if i == position else rest+(i in bigger)
                                      for i in range(count)))
    return tuple(sorted(c for c in choices if min(c)*pace/10 >= 10-1e-9))


def patterns(profile):
    free = (1, 2, 3, 5, 6) if profile.long else tuple(range(7))
    for count in range(1, len(free)+1):
        for days in combinations(free, count):
            for doses in allocations(round(profile.support*10), count, profile.pace):
                yield tuple(zip(days, (n/10 for n in doses)))


def steady_cost(pattern, *, demand, scales, scenario, exchange, short_policy):
    """Exact infinite periodic exponential interactions; no terminal boundary.

    Volume/support deficits vanish for an established exact 7-day pattern with
    28-day volume/7-day smoothing. Only used for easy-only low-volume control.
    This is NOT the complete shared score or admission model.
    """
    vectors = [(day, tuple(q.estimate/s for q, s in zip(demand('easy', miles), scales)))
               for day, miles in pattern]
    axes = [0., 0., 0.]
    memory = scenario.memory_days
    for day, q in vectors:
        for axis in range(3):
            prior = sum(other[axis]*exp(-(((day-d)%7 or 7)-scenario.timing_compression_days)/memory)
                        /(-expm1(-7/memory)) for d, other in vectors)
            axes[axis] += .5*q[axis]**2+q[axis]*prior
    fatigue = exchange*sum(w*v for w, v in zip(scenario.axis_weights, axes))/sum(scenario.axis_weights)
    short = sum(short_policy.penalty(demand('easy', miles)[0].estimate) for _, miles in pattern)
    return dict(cost=fatigue+short, fatigue=fatigue, short=short, delivery=0.)


def run(root, output, *, single_volume=False, volume_linear=0.):
    if volume_linear and not single_volume:
        raise ValueError('Linear volume comparison requires the single-volume objective')
    if not isfinite(volume_linear) or volume_linear<0:
        raise ValueError('Finite nonnegative volume coefficient required')
    output.mkdir(exist_ok=False)
    case = load_decision(root, 6)  # Mandatory exact reprice before controls.
    ctx = case['state'].context
    assert ctx.continuation_guard_days == 7 and (ctx.end-ctx.decision).days == 54
    hs, catalog = case['handoffs'], case['catalog']
    easy_key = min((k for k in hs if catalog[k].role == 'easy'),
                   key=lambda k: abs(catalog[k].miles-4.4))
    easy_miles = catalog[easy_key].miles
    base_pace = hs[easy_key]['running_minutes']['estimate']/easy_miles
    cardio_per_minute = hs[easy_key]['cardio']['estimate']/hs[easy_key]['running_minutes']['estimate']
    quality_key = 'quality:ordinary'
    base_work = sum(s for kind, s in catalog[quality_key].phases if kind == 'work')/60
    quality_cardio_ratio = (hs[quality_key]['cardio']['estimate']/hs[quality_key]['running_minutes']['estimate'])/cardio_per_minute
    profiles = (
        Profile('low_5_at_12', 5, 12, 0, 0, 0, 3),
        Profile('low_5_at_18', 5, 18, 0, 0, 0, 3),
        Profile('moderate_20', 20, base_pace, 8.4, 3.9, base_work, 3),
        Profile('moderate_20_history_two', 20, base_pace, 8.4, 3.9, base_work, 2),
        Profile('high_60_at_8', 60, 8, 14, 7, 20, 4),
        Profile('high_60_at_8_history_five', 60, 8, 14, 7, 20, 5),
        Profile('high_60_at_11', 60, 11, 14, 7, 20, 4),
    )
    manifest = freeze_sources(output/'sources.zip', config=load_config(), entrypoint=__name__,
        inputs=(root/'decision-0006.json', root/case['event']['snapshot_file'], root/'scenario.json'))
    stamp = perf_counter(); rows = []; calls = 0
    for p in profiles:
        # Explicit synthetic established-capacity assumption, not athlete fitting.
        factor = p.volume*p.pace/(20*base_pace)
        scales = (ctx.scales[0]*factor, ctx.scales[1]*factor,
                  ctx.scales[2]*(p.work/base_work if p.work else 1.))
        def demand(role, miles):
            minutes = miles*p.pace
            cardio = minutes*cardio_per_minute*(quality_cardio_ratio if role == 'quality' else 1.)
            return tuple(Quantity.point(v, 'synthetic point rate; not observed athlete data')
                         for v in (minutes, cardio, p.work if role == 'quality' else 0.))
        def session(offset, role, miles, source):
            return TrialSession(f'{source}:{offset}', ctx.decision+timedelta(days=offset),
                                miles, role, demand(role, miles), float(role == 'quality'), source)
        def cycle(easies):
            return tuple(sorted((*( (d, 'easy', m) for d, m in easies),
                                 *(( (0, 'quality', p.quality), (4, 'long', p.long)) if p.long else ()))))
        def repeat(entries, start, end, source, phase=0):
            return tuple(session(day, role, miles, source) for day in range(start, end)
                         for offset, role, miles in entries if (day-phase)%7 == offset)
        history_days = {2: (2, 6), 3: (1, 3, 6), 4: (1, 2, 3, 6), 5: (1, 2, 3, 5, 6)}[p.history_easy_count]
        hist_doses = allocations(round(p.support*10), p.history_easy_count, p.pace)
        balanced = min(hist_doses, key=lambda d: max(d)-min(d))
        history_cycle = cycle(tuple(zip(history_days, (n/10 for n in balanced))))
        if p.long:
            policy = replace(ctx.policy, weekly_miles=p.volume, ordinary_miles=p.support/p.history_easy_count,
                             long_miles=p.long, quality_work_minutes=p.work,
                             purpose_weights=(1.,0.,1.,1.) if single_volume else ctx.policy.purpose_weights)
            c = replace(ctx, history=repeat(history_cycle, -56, 0, 'observed'),
                        stage_start=ctx.decision, scales=scales, policy=policy, volume_targets=(), support_targets=(),
                        easy_duration_policy=replace(ctx.easy_duration_policy, supporting_easy_miles_per_7=p.support),
                        opening_coverage=(), continuation_boundary_references=tuple(
                            (role, miles, demand(role, miles)) for role, miles in (
                                ('easy', policy.ordinary_miles), ('long', p.long), ('quality', p.quality))))
        def evaluate(pattern, phase=0, matched_history=False):
            nonlocal calls
            calls += 1
            if not p.long:
                return steady_cost(pattern, demand=demand, scales=scales, scenario=ctx.scenario,
                                   exchange=ctx.exchange, short_policy=ctx.easy_duration_policy)
            active = replace(c, history=repeat(cycle(pattern) if matched_history else history_cycle,
                                               -56, 0, 'observed', phase=phase))
            program = repeat(cycle(pattern), 0, 54, 'planned', phase=phase)
            candidate = ProgramCandidate(tuple(r for r in program if r.day < c.hard_end),
                tuple(r for r in program if r.day >= c.hard_end), 'synthetic recurring distribution control')
            e = evaluate_program(active, candidate)
            assert abs(sum(r.miles for r in candidate.hard)-3*p.volume) < 1e-8
            assert abs(e.total_cost-e.weighted_demand_cost-e.short_run_cost-e.delivery_cost) < 1e-10
            linear=linear_volume_loss(active,e) if single_volume else 0.
            curve=volume_benefit_cost(active,e,linear_coefficient=volume_linear,linear_loss=linear) if single_volume else None
            return dict(cost=curve['cost'] if curve else e.total_cost,
                        baseline_single_volume_cost=e.total_cost if single_volume else None,
                        linear_volume_loss=linear if single_volume else None,
                        added_volume_cost=curve['added_volume_cost'] if curve else 0.,
                        fatigue=e.weighted_demand_cost, short=e.short_run_cost,
                        delivery=curve['delivery_cost'] if curve else e.delivery_cost,
                        purpose_losses=e.parts['shortfall_by_purpose'],
                        pre_handoff_volume_losses={k: e.parts['soft_horizon']['losses'][k]
                                                   for k in ('volume', 'ordinary_easy')})
        results = []
        for pattern in patterns(p):
            value = evaluate(pattern)
            results.append(dict(easy_count=len(pattern), total_runs=len(pattern)+(2 if p.long else 0),
                pattern=pattern, minutes=[round(m*p.pace, 2) for _, m in pattern],
                longest_easy_minutes=max(m*p.pace for _, m in pattern),
                nominal_easy_exceeds_long=bool(p.long and max(m for _,m in pattern)>p.long+1e-9), **value))
        finalists = [min((r for r in results if r['easy_count'] == n), key=lambda r:r['cost'])
                      for n in sorted({r['easy_count'] for r in results})]
        # Reprice finalists across all calendar phases; this is robustness of the
        # selected patterns, NOT exhaustive reoptimization at each phase.
        for row in finalists:
            row['phase_costs'] = [evaluate(row['pattern'], phase)['cost'] for phase in range(7)]
            row['matched_history_score'] = evaluate(row['pattern'], matched_history=True)
        rows.append(dict(profile=asdict(p), support=p.support, scales=scales, candidates=len(results),
            contract='guarded_shared_score' if p.long else 'easy_only_steady_component_control',
            finalists=finalists, winner=min(finalists, key=lambda r:r['cost']),
            single_volume_winner=min(results,key=lambda r:r['baseline_single_volume_cost']) if single_volume and p.long else None,
            without_short_penalty=min(results, key=lambda r:r['cost']-r['short']),
            phase_winning_counts=[min(finalists, key=lambda r:r['phase_costs'][i])['easy_count'] for i in range(7)]))
        print(json.dumps(dict(profile=p.name, candidates=len(results), winner=rows[-1]['winner']['pattern'],
                              phase_counts=rows[-1]['phase_winning_counts'])), flush=True)
    report = dict(manifest=manifest, seconds=perf_counter()-stamp, evaluations=calls, rows=rows,
        assumptions=dict(single_volume=single_volume,volume_linear=volume_linear,
            minimum_minutes=10, penalty_free_minutes=25, no_easy_upper_cap=True,
            synthetic_history_days=56, quality_offset=0, long_offset=4,
            normalization='duration/cardio scale with volume*time; quality work scale with declared work',
            allocations='balanced plus one 15-minute/25-minute/1.5-mean anchor; exact tenths'),
        limitations=['Not clinical, admission, progression, full search or churn validation',
            'Synthetic demand and normalization; not a forecast for other actual athletes',
            'No doubles, variable key doses, or exhaustive dose compositions',
            'No readiness or distance eligibility; mathematical winner may be inappropriate',
            'Low-volume no-key profile unsupported by full contract; component control only',
            'Matched-history finalist control changes history per candidate; stationary diagnostic, not a decision comparison',
            'Phase control reprices finalists, not full phase reoptimization'])
    (output/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(evaluations=calls, seconds=report['seconds'])))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('output/memory-continuation-daily-2-20260929'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--single-volume',action='store_true')
    parser.add_argument('--volume-linear',type=float,default=0.)
    args = parser.parse_args()
    run(args.root, args.output,single_volume=args.single_volume,volume_linear=args.volume_linear)
