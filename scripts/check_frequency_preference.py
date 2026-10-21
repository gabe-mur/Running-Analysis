"""Reuse bounded 5/20/60-mile controls. No online search or publication."""
from argparse import ArgumentParser
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

from scripts.check_easy_distribution_profiles import Profile, patterns
from scripts.check_explicit_distribution_tail import (
    explicit_context, fixture, periodic_demand, periodic_recovery_audit,
)
from scripts.replay_adaptive_search import load_decision
from scripts.research_manifest import freeze_sources
from run_analysis.config import load_config
from run_analysis.adaptive.frequency_preference_trial import ANCHORS, FrequencyPreference
from run_analysis.adaptive.shared_program_evaluator import ProgramCandidate, evaluate_program
from run_analysis.adaptive.volume_benefit_trial import volume_benefit_cost
from run_analysis.workload_accounting import Quantity

WEIGHTS = (0., .005, .01, .02, .05)


def without_short_preference(context):
    """Keep minimum-dose and support-volume policy; remove only its cost."""
    return replace(context, easy_duration_policy=replace(
        context.easy_duration_policy, maximum_penalty=0.))


def run(root, panel_path, output):
    case = load_decision(root, 6)
    panel = json.loads(panel_path.read_text())
    assert panel['assumptions']['single_volume']
    output.mkdir(exist_ok=False)
    manifest = freeze_sources(output/'sources.zip', config=load_config(), entrypoint=__name__,
        inputs=(panel_path, root/'decision-0006.json', root/case['event']['snapshot_file'], root/'scenario.json'))
    start = perf_counter(); reports = []; calls = 0; sampled_count = 0
    hs, catalog = case['handoffs'], case['catalog']
    easy = min((k for k in hs if catalog[k].role == 'easy'), key=lambda k:abs(catalog[k].miles-4.4))
    cardio_rate = hs[easy]['cardio']['estimate']/hs[easy]['running_minutes']['estimate']
    for row in panel['rows']:
        p = Profile(**row['profile'])
        if p.long:
            c, demand, cycle, repeat, history = fixture(case, p, row['scales'], single_volume=True)
        else:
            c = replace(case['state'].context, scales=tuple(row['scales']))
            def demand(role, miles):
                return tuple(Quantity.point(v, 'synthetic easy-only control') for v in (
                    miles*p.pace, miles*p.pace*cardio_rate, 0.))
            def cycle(pattern):
                return tuple((day, 'easy', miles) for day, miles in pattern)
        sampled = [(pattern, periodic_demand(cycle(pattern), demand, c)) for pattern in patterns(p)]
        sampled_count += len(sampled)
        baseline = min(sampled, key=lambda r:(r[1]['fatigue']+r[1]['short'], r[0]))
        no_short = min(sampled, key=lambda r:(r[1]['fatigue'], r[0]))
        reference = no_short[1]['fatigue']
        def summarize(item, weight=0., old=False):
            pattern, score = item
            policy = FrequencyPreference(p.volume, reference, weight)
            count = len(cycle(pattern))
            prior = policy.cost(count)
            audit = periodic_recovery_audit(cycle(pattern), demand, c)
            return dict(pattern=pattern, runs=count, easy_minutes=[round(m*p.pace, 1) for _, m in pattern],
                fatigue=score['fatigue'], legacy_short=score['short'], frequency_cost=prior,
                cost=score['fatigue']+(score['short'] if old else prior),
                fatigue_increase_percent=100*(score['fatigue']/reference-1),
                key_sessions=audit['key_sessions'])
        winners = []
        for w in WEIGHTS:
            policy = FrequencyPreference(p.volume, reference, w)
            chosen = min(sampled, key=lambda r:(r[1]['fatigue']+policy.cost(len(cycle(r[0]))), r[0]))
            winners.append(dict(weight=w, **summarize(chosen, w)))
        result = dict(profile=p.name, target_volume=p.volume, center=FrequencyPreference(p.volume, reference, 0).center,
            scope=row['contract'], candidates=len(sampled), legacy=summarize(baseline, old=True),
            fatigue_only=summarize(no_short), weights=winners)
        if p.long:
            plain = without_short_preference(c)
            selected = sorted({tuple(tuple(x) for x in r['pattern']) for r in winners} | {baseline[0]})
            def evaluate(pattern, days, *, legacy=False):
                nonlocal calls
                calls += 1
                base = c if legacy else plain
                ctx = base if days == 54 else explicit_context(base, days)
                physical = repeat(cycle(pattern), 0, days, 'planned')
                e = evaluate_program(ctx, ProgramCandidate(tuple(s for s in physical if s.day < ctx.hard_end),
                    tuple(s for s in physical if s.day >= ctx.hard_end), 'fixed recurring diagnostic only'))
                if not legacy:
                    assert e.short_run_cost == 0.
                price = volume_benefit_cost(ctx, e, linear_coefficient=panel['assumptions']['volume_linear'])
                return dict(cost=price['cost'], fatigue=e.weighted_demand_cost, delivery=price['delivery_cost'],
                    short=e.short_run_cost,
                    losses=e.parts['shortfall_by_purpose'], hard_miles=sum(s.miles for s in physical if s.day < ctx.hard_end))
            finite = {}
            legacy_controls = {}
            for days in (54, 112):
                prices = {pattern:evaluate(pattern, days) for pattern in selected}
                checks = []
                for w in WEIGHTS:
                    policy = FrequencyPreference(p.volume, reference, w)
                    def total(pattern):
                        return prices[pattern]['cost']+policy.cost(len(cycle(pattern)))
                    chosen = min(selected, key=lambda pattern:(total(pattern), pattern))
                    smallest = min(chosen, key=lambda x:(x[1], x[0]))
                    controls = []
                    for kind, altered in (
                        ('omit_smallest_easy', tuple(x for x in chosen if x != smallest)),
                        ('cut_0.1_mile', tuple((d, round(m-.1, 1) if d == smallest[0] else m) for d, m in chosen)),
                    ):
                        adverse = evaluate(altered, days)
                        count = len(cycle(altered))
                        controls.append(dict(kind=kind, score=adverse,
                            lost_miles_per_7=round(sum(m for _, m in chosen)-sum(m for _, m in altered), 1),
                            total_cost_delta=adverse['cost']+policy.cost(count)-total(chosen),
                            frequency_cost_delta=policy.cost(count)-policy.cost(len(cycle(chosen)))))
                    checks.append(dict(weight=w, runs=len(cycle(chosen)), pattern=chosen,
                        score=prices[chosen], total_cost=total(chosen), underfunding_controls=controls))
                finite[str(days)] = checks
                original_pattern = baseline[0]
                smallest = min(original_pattern, key=lambda x:(x[1], x[0]))
                cut = tuple((d, round(m-.1, 1) if d == smallest[0] else m) for d, m in original_pattern)
                original = evaluate(original_pattern, days, legacy=True)
                adverse = evaluate(cut, days, legacy=True)
                legacy_controls[str(days)] = dict(funded=original, cut=adverse,
                    total_cost_delta=adverse['cost']-original['cost'])
            result['finite_repricing'] = finite
            result['legacy_small_cut_controls'] = legacy_controls
        reports.append(result)
        print(json.dumps(dict(profile=p.name, center=result['center'], legacy=result['legacy']['runs'],
            fatigue_only=result['fatigue_only']['runs'], weights=[(r['weight'], r['runs'], r['fatigue_increase_percent']) for r in winners])), flush=True)
    report = dict(manifest=manifest, seconds=perf_counter()-start, candidates=sampled_count, full_evaluations=calls,
        assumptions=dict(anchors=ANCHORS, weights=WEIGHTS, capped_squared_deviation=4,
            reference='frozen fatigue-only stationary winner at full target volume',
            old_short_preference='removed; original dose minimum and supporting-volume accounting retained',
            frequency='distinct running days in declared seven-day cycle; no doubles'),
        scope='Bounded sampled distribution/weight probe. Fixed quality/long dates. Not true calendar search, daily replay, novice admission, or deployment.',
        limitations=['Synthetic demand rates and normalization; not measured readiness or injury safety',
            'Full evaluator checks only union of stationary winners against common history',
            'Finite prior uses declared recurring rate, not partial endpoint counts; no online horizon contract yet',
            'Low-volume profiles are easy-only component controls; mandatory key purposes remain unresolved'],
        profiles=reports)
    (output/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(seconds=report['seconds'], full_evaluations=calls, candidates=sampled_count)), flush=True)
    return report


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('output/memory-continuation-daily-2-20260929'))
    parser.add_argument('--panel', type=Path, default=Path('output/easy-profile-volume-curve-20260929/report.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.panel, args.output)
