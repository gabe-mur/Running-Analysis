"""Read-only fixed-program/coverage probes for an additive-volume replan failure."""
from argparse import ArgumentParser
from datetime import timedelta
import json
from pathlib import Path
from time import perf_counter

from scripts.audit_continuation_upload import reconstruct, unpack_soft, raw_state_lookup
from scripts.audit_conditional_upload import coverage, score, window
from scripts.typed_continuation_prototype import runs
from scripts.volume_path_prototype import volume_path, volume_path_policy


def investigate(report, regenerate=False):
    if report['synthetic_sandbox'].get('volume_path',{}).get('version')!='additive_date_path_v1':
        raise ValueError('Require a saved additive-volume replay')
    if report['synthetic_sandbox']['expected_completion'].get('state_transition_version')!='daily_pace_v1':
        raise ValueError('Require the recorded daily pace transition')
    captures=[]
    reconstruct(report,lambda i,inputs,days,evidence:captures.append((inputs,days,evidence))
                if i>=len(report['replans'])-2 else None)
    origins=[x[0]['daily_states'][0].as_of for x in captures]
    physical=[days+unpack_soft(s['planner_diagnostics']['continuation_selection']['soft_sessions'])
              for (_,days,_),s in zip(captures,report['replans'][-2:])]
    physical[1]=window(physical[0],origins[0].date(),origins[1].date())+physical[1]
    hard_end=origins[0].date()+timedelta(days=21)
    full_end=hard_end+timedelta(days=28)
    probes=[]
    with volume_path_policy():
        for i,(inputs,days,evidence) in enumerate(captures):
            own_end=origins[i].date()+timedelta(days=21)
            result=score(inputs,evidence,physical[i],own_end,own_end+timedelta(days=28),historical_pace=False)
            expected=report['replans'][-2+i]['planner_diagnostics']['winning_candidate']['total_cost']
            assert abs(result['components']['total']-expected)<1e-8
            probes.append(dict(origin=('before','after')[i],program=('before','after')[i],mode='native',**result))
            for source,sequence in zip(('before','after'),physical):
                for rebuilt in ([False,True] if regenerate else [False]):
                    result=score(inputs,evidence,sequence,hard_end,full_end,
                        historical_pace=False,regenerate=rebuilt)
                    probes.append(dict(origin=('before','after')[i],program=source,
                        mode='regenerated_tail' if rebuilt else 'fixed_program',**result))
                    print(probes[-1]['origin'],source,probes[-1]['mode'],result['components'],flush=True)
    path_traces=[]
    for i,(inputs,_,evidence) in enumerate(captures):
        lookup=raw_state_lookup(inputs,evidence)
        for source,sequence in zip(('before','after'),physical):
            full=window(sequence,origins[i].date(),full_end)
            hard=window(sequence,origins[i].date(),hard_end)
            states=[lookup(origins[i]+timedelta(days=j)) for j in range(len(full))]
            targets=inputs['expected_target_projector'](runs(hard),len(full))
            path_traces.append(dict(origin=('before','after')[i],program=source,
                **volume_path(full,states,targets,inputs['config'],trace=True)))
    return dict(scope=__doc__,probes=probes,coverage=coverage(report,physical),volume_traces=path_traces)


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--output-json',type=Path,required=True)
    parser.add_argument('--regenerate-tails',action='store_true')
    args=parser.parse_args()
    if args.output_json.exists():
        parser.error('Choose a new output path')
    started=perf_counter()
    result=investigate(json.loads(args.report.read_text()),args.regenerate_tails)
    result['seconds']=perf_counter()-started
    with args.output_json.open('x') as stream:
        json.dump(result,stream,indent=2)
        stream.write('\n')
    print('Seconds',result['seconds'])


if __name__=='__main__':
    main()
