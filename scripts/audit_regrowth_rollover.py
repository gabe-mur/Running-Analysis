"""Fixed-physical-program audit of the saved final no-evidence rollover.

Keep shared absolute endpoints when comparing origins. Separately regenerate
tails at the new native horizon; do not confuse that with an identical program.
"""
from argparse import ArgumentParser
from datetime import timedelta
import json
from pathlib import Path
from time import perf_counter

from scripts.audit_conditional_upload import score, window, scoring, components
from scripts.audit_continuation_upload import reconstruct, unpack_soft, source_row, hard_score, evaluate_physical
from scripts.audit_regrowth_gates import check_candidate
from scripts.audit_volume_search import cadence_trace
from scripts.boundary_funding_prototype import boundary_funding_policy
from scripts.volume_path_prototype import volume_path_policy
from scripts.calendar_neighbor_prototype import calendar_neighbors


def investigate(report):
    if not report['synthetic_sandbox'].get('suffix_regrowth',{}).get('enabled'):
        raise ValueError('Require saved suffix-regrowth trial')
    captured=[]
    stable=[]
    calendar_probes=[]
    def observe(index,inputs,days,evidence):
        if index>=len(report['replans'])-2:
            captured.append((inputs,days,evidence))
        if index==len(report['replans'])-2:
            from datetime import datetime
            origin=days[0].date
            offsets=tuple((datetime.fromisoformat(r['planned_for']).date()-origin).days
                for r in report['replans'][-1]['planned_sessions']
                if 0<=(datetime.fromisoformat(r['planned_for']).date()-origin).days<len(days))
            original_offsets=tuple(i for i,d in enumerate(days) if d.recommendation)
            allowed=set(range(len(days)))-set(inputs['forced_rest_offsets'])-set(inputs['completed_run_offsets'])
            candidate,price,_=hard_score(inputs,offsets)
            calendar_probes.append(dict(origin=0,name='revised_dates_rebuilt_before',days=candidate,
                hard=price.model_dump(mode='json'),one_edit_neighbor=offsets in calendar_neighbors(original_offsets,allowed)))
        if index==len(report['replans'])-1:
            saved=report['replans'][index]['planner_diagnostics']['continuation_selection']
            finalist=min((r for r in saved['finalists'] if r.get('stable_prefix')),
                         key=lambda r:r['combined_cost'])
            record=report['synthetic_sandbox']['calendar_neighbor_prototype'][index]
            row=source_row(dict(record,evaluated=record['evaluated']+saved['suffix_regrowth']['rows']),finalist)
            candidate,_,_=hard_score(inputs,row['offsets'],roles={int(k):v for k,v in row.get('roles',{}).items()})
            stable.append((candidate,finalist))
            prior_days=captured[0][1]+unpack_soft(report['replans'][-2]['planner_diagnostics']['continuation_selection']['soft_sessions'])
            offsets=tuple((d.date-days[0].date).days for d in prior_days if d.recommendation
                          and 0<=(d.date-days[0].date).days<len(days))
            candidate,price,_=hard_score(inputs,offsets)
            calendar_probes.append(dict(origin=1,name='prior_dates_rebuilt_after',days=candidate,
                                        hard=price.model_dump(mode='json')))
    reconstruct(report,observe)
    if len(captured)!=2:
        raise ValueError('Need two saved decisions')
    snapshots=report['replans'][-2:]
    physical=[days+unpack_soft(snap['planner_diagnostics']['continuation_selection']['soft_sessions'])
        for (_,days,_),snap in zip(captured,snapshots)]
    starts=[x[1][0].date for x in captured]
    # This panel is specifically a no-execution rollover, not an upload audit.
    if captured[0][2]['runs']!=captured[1][2]['runs']:
        raise ValueError('History changed across this transition')
    if any(d.recommendation for d in window(physical[0],starts[0],starts[1])):
        raise ValueError('Cannot silently remove uncompleted prescribed work')
    physical[1]=window(physical[0],starts[0],starts[1])+physical[1]
    output=dict(native_controls=[],fixed_common_window=[],new_horizon=[],coverage=[])
    end=starts[0]+timedelta(days=21)
    close=end+timedelta(days=28)
    with volume_path_policy(),boundary_funding_policy():
        for i,(inputs,days,evidence) in enumerate(captured):
            native_end=starts[i]+timedelta(days=21)
            native=score(inputs,evidence,physical[i],native_end,native_end+timedelta(days=28),historical_pace=False)
            expected=snapshots[i]['planner_diagnostics']['winning_candidate']['total_cost']
            assert abs(native['components']['total']-expected)<1e-8
            output['native_controls'].append(native['components'])
            for label,sequence in zip(('prior','revised'),physical):
                probe=score(inputs,evidence,sequence,end,close,historical_pace=False)
                output['fixed_common_window'].append(dict(origin=i,program=label,**probe))
                print('Fixed',i,label,probe['components'],flush=True)
        inputs,days,evidence=captured[1]
        # Use the shared absolute hard end: promoting the previous soft day's
        # fractional timestamp would invent a new hard timing option.
        end=starts[0]+timedelta(days=21)
        for label,sequence in zip(('prior','revised'),physical):
            probe=score(inputs,evidence,sequence,end,end+timedelta(days=28),
                        historical_pace=False,regenerate=True)
            hard=window(sequence,starts[1],end)
            with scoring(inputs,evidence,False):
                gates=check_candidate(snapshots[1],inputs,hard)
                cadence=cadence_trace(hard,inputs)
            output['new_horizon'].append(dict(program=label,hard_days=len(hard),
                scope='Common absolute hard end, regenerated 28-day tail; not native 21-day gate clearance',
                gates=gates,cadence=cadence,**probe))
            print('Common hard end regenerated',label,probe['components'],flush=True)
        with scoring(inputs,evidence,False):
            candidate,finalist=stable[0]
            result=evaluate_physical(inputs,candidate,evidence)
            assert abs(result['continuation']['combined_cost']-finalist['combined_cost'])<1e-8
            output['current_stable_finalist']=dict(components=components(result),
                hard_cadence=cadence_trace(candidate,inputs),**result)
        output['calendar_probes']=[]
        for item in calendar_probes:
            raw,_,history=captured[item['origin']]
            with scoring(raw,history,False):
                result=evaluate_physical(raw,item['days'],history)
                gates=check_candidate(snapshots[item['origin']],raw,item['days'])
            output['calendar_probes'].append(dict({k:v for k,v in item.items() if k!='days'},
                components=components(result),gates=gates,continuation=result['continuation']))
            print('Rebuilt calendar',item['name'],components(result),flush=True)
        for i,(inputs,days,evidence) in enumerate(captured):
            record=report['synthetic_sandbox']['calendar_neighbor_prototype'][-2+i]
            continuation=snapshots[i]['planner_diagnostics']['continuation_selection']
            evaluated=record['evaluated']+continuation.get('suffix_regrowth',{}).get('rows',[])
            for label,sequence in zip(('prior','revised'),physical):
                offsets=[(d.date-starts[i]).days for d in sequence if d.recommendation
                         and 0<=(d.date-starts[i]).days<21]
                output['coverage'].append(dict(origin=i,program=label,offsets=offsets,
                    evaluated=[r for r in evaluated if r['offsets']==offsets],
                    finalists=[r for r in continuation['finalists'] if r['offsets']==offsets]))
    return output


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--output-json',type=Path,required=True)
    args=parser.parse_args()
    if args.output_json.exists():
        parser.error('Choose a new output path')
    started=perf_counter()
    result=investigate(json.loads(args.report.read_text()))
    result['seconds']=perf_counter()-started
    with args.output_json.open('x') as stream:
        json.dump(result,stream,indent=2,default=str)
        stream.write('\n')
    print('Seconds',result['seconds'])


if __name__=='__main__':
    main()
