"""Read-only placement alternatives around coached key sessions.

Reproduce the full selected score before moving either of its first two key
sessions (or all keys with --all-keys). Preserve ordinary dose/variety, all other
dates and scored tail. Single-date controls do not certify coordinated optima.
"""
from argparse import ArgumentParser
from datetime import date, timedelta
import json
from pathlib import Path

from scripts.replay_adaptive_search import load_decision
from scripts.prototype_coached_quality_calendar import coached_evaluator
from scripts.check_explicit_distribution_tail import incoming_demand_trace
from run_analysis.adaptive.quality_coaching_policy import CoachedRecipe, QualityCoachingSequence


def run(root, projection, *, all_keys=False):
    report = json.loads((projection / 'report.json').read_text())
    case = load_decision(root, report['source_decision'])
    if case['state'].fingerprint != report['source_snapshot']:
        raise ValueError('Projection input snapshot changed')
    policy = dict(report['coaching_contract'])
    policy['ordinary'] = tuple(CoachedRecipe(**r) for r in policy['ordinary'])
    sequence = QualityCoachingSequence(**policy)
    if sequence.key != report['coaching_contract_key']:
        raise ValueError('Projection coaching contract changed')
    engine = coached_evaluator(case, sequence,
        demand_interaction=report.get('demand_interaction','independent_v1'))
    program = tuple(map(tuple, report['search']['program']))
    def detail(p):
        p = engine['bind'](p)
        result = engine['detail'](p); e = result['evaluation']
        return dict(eligible=result['decision_eligible'], total=e.total_cost,
            fatigue=e.weighted_demand_cost, delivery=e.delivery_cost,
            quality=e.parts['shortfall_by_purpose']['quality'],
            volume=e.parts['shortfall_by_purpose']['volume'],
            key_entry=[r for r in incoming_demand_trace(e, case['state'].context) if r['role'] != 'easy'])
    baseline = detail(program)
    if not baseline['eligible'] or abs(baseline['total'] - report['selected']['total']) > 1e-12:
        raise ValueError('Coached projection no longer reproduces')
    rows = []
    keys = [(d, k) for d, k in program if case['catalog'][k].role in {'long', 'quality'}]
    if not all_keys:keys=keys[:2]
    for old, key in keys:
        for shift in (-3, -2, -1, 1, 2, 3):
            day = (date.fromisoformat(old) + timedelta(days=shift)).isoformat()
            if day not in case['dates'] or day in dict(program):
                continue
            candidate = tuple(sorted((day if d == old else d, k) for d, k in program))
            scored = detail(candidate)
            rows.append(dict(from_day=old, to_day=day, role=case['catalog'][key].role,
                result=scored, delta={k: scored[k] - baseline[k] for k in
                    ('total', 'fatigue', 'delivery', 'quality', 'volume')}))
    result = dict(selected_score_reproduced=True, baseline=baseline, alternatives=rows,
        key_scope='all hard and soft key sessions' if all_keys else 'first two key sessions',
        scope=__doc__, live_changed=False, additional_calendar_search=False)
    filename='all-key-placement-audit.json' if all_keys else 'placement-audit.json'
    (projection / filename).write_text(json.dumps(result, default=str, indent=2, allow_nan=False))
    print(json.dumps(dict(baseline={k:v for k,v in baseline.items() if k!='key_entry'},
        alternatives=[{k:v for k,v in r.items() if k!='result'} | dict(eligible=r['result']['eligible'])
            for r in rows]), indent=2))


if __name__ == '__main__':
    p = ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path); p.add_argument('projection', type=Path)
    p.add_argument('--all-keys',action='store_true')
    a = p.parse_args(); run(a.root, a.projection,all_keys=a.all_keys)
