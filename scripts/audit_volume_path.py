"""Bounded saved-program check of additive full-path volume assessment."""
from argparse import ArgumentParser
from datetime import timedelta
import json
from pathlib import Path
from time import perf_counter

from scripts.audit_continuation_upload import reconstruct, unpack_soft, raw_state_lookup
from scripts.audit_conditional_upload import score, window, path_trace
from scripts.volume_path_prototype import volume_path, replace_volume_cost, volume_path_policy
from scripts.typed_continuation_prototype import runs


def investigate(report):
    captures=[]
    reconstruct(report, lambda i,inputs,days,evidence: captures.append((inputs,days,evidence))
                if i >= len(report['replans'])-2 else None, historical_run_time_pace=True)
    origins=[x[0]['daily_states'][0].as_of for x in captures]
    sequences=[days+unpack_soft(s['planner_diagnostics']['continuation_selection']['soft_sessions'])
               for (_,days,_),s in zip(captures,report['replans'][-2:])]
    sequences[1]=window(sequences[0],origins[0].date(),origins[1].date())+sequences[1]
    end=origins[0].date()+timedelta(days=21)
    close=end+timedelta(days=28)
    rows=[]
    for i,(inputs,days,evidence) in enumerate(captures):
        lookup=raw_state_lookup(inputs,evidence)
        for source,sequence in zip(('before','after'),sequences):
            original=score(inputs,evidence,sequence,end,close,historical_pace=True)
            hard=window(sequence,origins[i].date(),end)
            full=window(sequence,origins[i].date(),close)
            states=[lookup(origins[i]+timedelta(days=d)) for d in range(len(full))]
            targets=inputs['expected_target_projector'](runs(hard),len(full))
            path=volume_path(full,states,targets,inputs['config'],trace=True)
            old_path=path_trace(hard,inputs,length=7)['worst']
            revised=replace_volume_cost(original['components'],old_path,path,path['ordinary_miles'])
            with volume_path_policy() as metrics:
                integrated=score(inputs,evidence,sequence,end,close,historical_pace=True)
            assert abs(integrated['components']['total']-revised['total'])<1e-8
            for field in ('recovery','shape','support','coaching','switching'):
                assert integrated['components'][field]==original['components'][field]
            rows.append(dict(origin=('before','after')[i],program=source,
                original=original['components'],revised=revised,volume=path,
                integrated=integrated['components'],metrics=metrics))
    # Shared-date prices, not just aggregate totals, must survive execution.
    for source in ('before','after'):
        a,b=[r for r in rows if r['program']==source]
        prior={r['date']:r for r in a['volume']['days']}
        for day in b['volume']['days']:
            expected=prior[day['date']]
            assert abs(day['rate']-expected['rate'])<1e-10
            # The saved first target upper endpoint changes by .1; neither
            # physical program exceeds that range, so the cost is unaffected.
            assert abs(day['squared_violation']-expected['squared_violation'])<1e-10
    return dict(scope=__doc__,probes=rows,shared_date_costs_match=True)


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--output-json',type=Path,required=True)
    args=parser.parse_args()
    if args.output_json.exists():
        parser.error('Require a new output path')
    started=perf_counter()
    result=investigate(json.loads(args.report.read_text()))
    result['seconds']=perf_counter()-started
    with args.output_json.open('x') as stream:
        json.dump(result,stream,indent=2)
        stream.write('\n')
    for row in result['probes']:
        print(row['origin'],row['program'],row['original']['total'],row['revised'],
              'volume milliseconds',1000*row['volume']['seconds'])
    print('Seconds',result['seconds'])


if __name__=='__main__':
    main()
