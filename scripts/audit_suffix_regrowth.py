"""Saved opening-decision test of bounded regrowth; no later decision inputs."""
from argparse import ArgumentParser
from datetime import timedelta
import json
from pathlib import Path
from time import perf_counter

from scripts.audit_conditional_upload import score, components
from scripts.audit_continuation_upload import hard_score, reconstruct, unpack_soft, evaluate_physical
from scripts.audit_calendar_materialization import prescriptions
from scripts.audit_volume_search import timing_stress
from scripts.continuation_replan import select_program, fingerprint
from scripts.suffix_regrowth_prototype import propose, diverse_seeds, regrowth_finalist_policy


def investigate(report, *, probe_discarded=False, reserve_slots=False):
    if not report['synthetic_sandbox'].get('funding_accounting',{}).get('enabled'):
        raise ValueError('Require full-path funding trial')
    output={}
    # Deliberately discard subsequent decisions before reconstruction. Neither
    # proposals nor final selection can read the post-upload calendar/history.
    opening=dict(report,replans=report['replans'][:1])
    def observe(index,inputs,days,evidence):
        record=opening['synthetic_sandbox']['calendar_neighbor_prototype'][0]
        full=days+unpack_soft(opening['replans'][0]['planner_diagnostics']['continuation_selection']['soft_sessions'])
        end=days[0].date+timedelta(days=21)
        native=score(inputs,evidence,full,end,end+timedelta(days=28),historical_pace=False)
        expected=opening['replans'][0]['planner_diagnostics']['winning_candidate']['total_cost']
        assert abs(native['components']['total']-expected)<1e-8
        output['native_control']=native['components']
        rows,programs,metrics=propose(inputs,record,evidence)
        output['proposal_metrics']=metrics
        output['new_rows']=rows
        augmented=dict(record,evaluated=record['evaluated']+rows)
        # Reproduce the actual publication seed, not the previous full winner.
        publication,_,_=hard_score(inputs,record['winner']['offsets'])
        started=perf_counter()
        from contextlib import nullcontext
        with regrowth_finalist_policy() if reserve_slots else nullcontext():
            result=select_program(publication,None,inputs,augmented,limit=6,continuation_days=28)
        output['reserved_regrowth_slots']=2 if reserve_slots else 0
        output['selection_seconds']=perf_counter()-started
        output['selection']=result['evidence']
        output['winner']=result['winner'].model_dump(mode='json')
        output['prescriptions']=prescriptions(result['planning_days'])
        output['timing_stress']=timing_stress(inputs,result['planning_days'],evidence)
        output['original_timing_stress']=timing_stress(inputs,days,evidence)
        if probe_discarded:
            scored={r.get('hard_fingerprint') for r in result['evidence']['finalists']}
            rejected=[]
            for row in rows:
                key=(tuple(row['offsets']),tuple(sorted(row['roles'].items())))
                if fingerprint(programs[key]) not in scored:
                    rejected.append(row)
            output['discarded_probes']=[]
            for row in diverse_seeds(dict(evaluated=rejected),limit=3):
                key=(tuple(row['offsets']),tuple(sorted(row['roles'].items())))
                probe=evaluate_physical(inputs,programs[key],evidence)
                output['discarded_probes'].append(dict(offsets=row['offsets'],roles=row['roles'],
                    components=components(probe),**probe))
    reconstruct(opening,observe)
    return output


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--output-json',type=Path,required=True)
    parser.add_argument('--probe-discarded',action='store_true',help='Diagnostic only: up to 3 extra full scores')
    parser.add_argument('--reserve-slots',action='store_true',help='Offline two-slot diversity experiment')
    args=parser.parse_args()
    if args.output_json.exists():
        parser.error('Choose a new output path')
    started=perf_counter()
    result=investigate(json.loads(args.report.read_text()),probe_discarded=args.probe_discarded,
                       reserve_slots=args.reserve_slots)
    result['seconds']=perf_counter()-started
    with args.output_json.open('x') as stream:
        json.dump(result,stream,indent=2)
        stream.write('\n')
    print('Before',result['native_control']['total'],'after',result['winner']['total_cost'],
          'proposal seconds',result['proposal_metrics']['seconds'],'total',result['seconds'])


if __name__=='__main__':
    main()
