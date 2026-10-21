"""Frozen sequence-policy comparison, independent of the new evening FIT."""
from argparse import ArgumentParser
from collections import Counter
from dataclasses import replace
from datetime import timedelta
from itertools import product
from pathlib import Path
from time import perf_counter
import json

from scripts.audit_easy_distribution import exposures, program_option
from scripts.audit_easy_placement import ORIGIN, records
from scripts.audit_easy_tolerance import inputs, policy_for, stacked_records, describe
from scripts.audit_recency_demand import json_ready
from scripts.sequence_dose_trial import compare
from run_analysis.session_decision import CurrentContext


def panel(directory=Path('output')):
    started = perf_counter()
    base,rates,history,refs = inputs(directory)
    policy = policy_for(base,refs)
    main_end, end = ORIGIN+timedelta(days=28), ORIGIN+timedelta(days=42)
    reference = program_option('reference',exposures(8,ORIGIN),history,rates)
    kwargs = dict(as_of=base['as_of'],start=ORIGIN,main_end=main_end,end=end)
    def evaluate(key,rows,**extra):
        candidate = program_option(key,rows,history,rates)
        return dict(compare(reference,candidate,history,policy,**kwargs,**extra),
                    calendar=[dict(key=e.key,day=e.day,miles=e.miles,role=e.role) for e in rows],
                    description=describe(rows))
    original = exposures(8,ORIGIN)
    witness_days = (0,4,7,12,14,19,21,26)
    reports = dict(unchanged=evaluate('unchanged',original),
        witness=evaluate('witness',records(8,witness_days)),
        stacked=evaluate('stacked',stacked_records()),
        three_easy=evaluate('three_easy',records(8,(0,1,2,7,12,14,19,21))))
    reports['health_concern'] = evaluate('health',records(8,witness_days),
        context=CurrentContext(ORIGIN,base['as_of'],'health_concern','synthetic_control'))
    reports['unexpected_response'] = evaluate('response',records(8,witness_days),unexpected_previous_response=True)
    # Underfunded candidates do not win by reducing load; key work stays identical.
    reports['underfunded'] = evaluate('underfunded',tuple(e for e in original if e.key != 'easy:5'))
    changed = tuple(replace(e,work_bout_seconds=(600.,)*3) if e.key=='quality:6' else e for e in original)
    reports['doubled_quality_work'] = evaluate('doubled_quality_work',changed)
    tail = tuple(replace(e,miles=2.2,easy_miles=2.2) if e.key=='easy:28' else e for e in original)
    reports['changed_tail'] = evaluate('changed_tail',tail)
    improvements = []
    for moves in product((0,1),repeat=4):
        if not any(moves):
            continue
        days = tuple(d for c in range(4) for d in (7*c,7*c+5-moves[c]))
        result = evaluate(str(days),records(8,days))
        improvements.append(dict(days=days,decision=result['decision'],
            concentration=result['concentration'],key_order_counts=result['key_order_counts'],
            changed_easy_dates=result['changed_easy_dates']))
    distributions = {str(n):evaluate(str(n),exposures(n,ORIGIN)) for n in (2,4,8,12,16,20)}
    return dict(as_of=base['as_of'],protocol='sequence_dose_trial_v1',original_history_count=len(history),
        supplied_evening_fit_used_in_noon_snapshot=False,
        reports=reports,improvements=improvements,distributions=distributions,
        improvement_decisions=dict(Counter(r['decision'] for r in improvements)),
        elapsed_seconds=perf_counter()-started,planner_influence=False,program_approved=False,
        limitations=['Phase priority is a coaching choice, not fitted recovery physiology',
            'A spacing trial can increase total concentration; this is explicitly reported',
            'No frequency optimum, uncertainty-calibrated benefit or complete suitability model',
            'Same-date evaluation tests do not replace daily/upload scheduler replay'])


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    report = panel()
    encoded = json.dumps(report,default=json_ready,allow_nan=False,indent=2)
    if args.output:
        args.output.write_text(encoded+'\n')
        print(json.dumps(dict(output=str(args.output),elapsed_seconds=report['elapsed_seconds'],
            decisions={k:v['decision'] for k,v in report['reports'].items()},
            improvement_decisions=report['improvement_decisions']),indent=2))
    else:
        print(encoded)
