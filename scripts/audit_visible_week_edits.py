"""Cross-frame fixed-program audit of visible-week dose edits, without search."""
from argparse import ArgumentParser
from datetime import date, datetime, timedelta
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

from scripts.audit_backward_controls import plain, validate_control
from scripts.audit_backward_same_origin import coverage
from scripts.audit_continuation_upload import reconstruct, unpack_soft, evaluate_physical
from scripts.audit_distance_continuity import components
from scripts.audit_horizon_rollover import absolute_slice, probe
from scripts.continuation_replan import fingerprint
from scripts.expected_completion import expected_completion_policy


def replace_prescribed_dose(program, donor, dates):
    """Copy dose plus its semantic label, never times or unrelated prescriptions."""
    source={d.date:d for d in donor}
    result=list(program)
    for i,day in enumerate(program):
        if day.date not in dates:
            continue
        other=source[day.date]
        if day.recommendation.workout_type.value!='easy' or other.recommendation.workout_type.value!='easy':
            raise ValueError('Require easy prescriptions on both sides')
        fields=('distance_target_miles','distance_range_miles','planning_role','title')
        rec=day.recommendation.model_copy(update={k:getattr(other.recommendation,k) for k in fields})
        result[i]=day.model_copy(update=dict(recommendation=rec,day_role=other.day_role))
    if not dates <= {d.date for d in program}:
        raise ValueError('Missing dose date')
    return result


def splice_tail(prefix, donor, boundary):
    """Join exact prescriptions on a shared date grid; feasibility is separate."""
    if [d.date for d in prefix]!=[d.date for d in donor]:
        raise ValueError('Tail splice requires identical complete date grids')
    start,end=prefix[0].date,prefix[-1].date+timedelta(days=1)
    if not start<boundary<end:
        raise ValueError('Tail boundary must be inside the program')
    return absolute_slice(prefix,start,boundary)+absolute_slice(donor,boundary,end)


def investigate(report, branch, *, indices=(2,3,4), target=date(2026,10,1), partner=date(2026,10,2), cross_tails=False):
    started=perf_counter()
    if len(indices)!=3 or tuple(sorted(set(indices)))!=tuple(indices):
        raise ValueError('Require three distinct chronological decision indices')
    if target==partner:
        raise ValueError('Require distinct target and partner dates')
    first,middle,last=indices
    available={r['index']:r for r in branch['decisions']}
    if not set(indices)<=available.keys():
        raise ValueError('Missing verified decision frame')
    frames={i:available[i] for i in indices}
    originals={i:unpack_soft(r['full_program']) for i,r in frames.items()}
    for i,days in originals.items():
        if fingerprint(days)!=frames[i]['full_fingerprint']:
            raise ValueError('Saved full-program mismatch')
    start=originals[first][0].date; end=originals[last][-1].date+timedelta(days=1)
    programs={}
    for i,days in originals.items():
        # Known earlier PRESCRIPTIONS, not later observed response, restore an
        # elapsed prefix. Terminal extension is an explicit later-plan choice.
        prefix=absolute_slice(originals[first],start,days[0].date)
        suffix=absolute_slice(originals[last],days[-1].date+timedelta(days=1),end)
        programs[i]=prefix+days+suffix
    common_hard=date.fromisoformat(frames[first]['hard_end_exclusive'])
    common_full=date.fromisoformat(frames[first]['full_end_exclusive'])
    rows=[]; ledger=None
    def observe(index,inputs,days,evidence):
        nonlocal ledger
        if index not in frames:
            return
        snapshot=report['replans'][index]; own=originals[index]
        with expected_completion_policy(inputs['config']) as model:
            native=evaluate_physical(inputs,days,evidence,full=own)
            validate_control(snapshot,inputs,days,own,evidence['runs'],sum(components(native).values()))
            if index==middle:
                at=datetime.fromisoformat(frames[last]['origin'])
                expected=plain(model.ledger(at,[d.recommendation for d in days if d.recommendation]))
                actual=frames[last]['frame']['completed_history']
                fields=('start_time','distance_miles','moving_minutes','workout_type','difficulty',
                    'planning_role','prescribed_target_miles','prescribed_low_miles','prescribed_high_miles',
                    'prescribed_workout_type','completed_prescribed_workout','prescription_title','quality_session_type')
                project=lambda xs:[{k:r[k] for k in fields} for r in xs]
                ledger=dict(training_fields=list(fields),equal=project(expected)==project(actual),
                    expected=project(expected),actual=project(actual))
        native_hard=date.fromisoformat(frames[index]['hard_end_exclusive'])
        native_full=date.fromisoformat(frames[index]['full_end_exclusive'])
        variants=[(f'program_{j}',programs[j]) for j in frames]
        # Local changes isolate the key dose and its consecutive-day interaction.
        for donor in frames:
            variants.append((f'own_target_from_{donor}',replace_prescribed_dose(programs[index],programs[donor],{target})))
            variants.append((f'own_pair_from_{donor}',replace_prescribed_dose(programs[index],programs[donor],{target,partner})))
        modes=[('common_endpoints',common_hard,common_full)]
        if native_hard!=common_hard:
            modes.append(('move_hard_cut_only',native_hard,common_full))
        modes.append(('native_endpoints',native_hard,native_full))
        for mode,h_end,f_end in modes:
            mode_variants=list(variants)
            if cross_tails and mode=='native_endpoints':
                mode_variants.extend((f'hard_{a}_tail_{b}',splice_tail(programs[a],programs[b],h_end))
                    for a in frames for b in frames if a!=b)
            for label,program in mode_variants:
                result=probe(snapshot,inputs,evidence,program,h_end,f_end)
                result.update(frame=index,mode=mode,variant=label,
                    doses=[dict(date=d.date.isoformat(),target=d.recommendation.planned_distance_miles,
                        role=d.recommendation.planning_role) for d in program if d.date in (target,partner)])
                if mode=='native_endpoints' and (label.startswith('program_') or label.startswith('hard_')):
                    origin=inputs['daily_states'][0].as_of.date()
                    result['coverage']=coverage(report,index,absolute_slice(program,origin,h_end),
                        absolute_slice(program,origin,f_end))
                rows.append(result)
            print(f'Visible-week frame {index}/{mode}: '+', '.join(
                f"{r['variant']}={r.get('cost','rejected')}" for r in rows[-len(mode_variants):]),flush=True)
    reconstruct(dict(report,replans=report['replans'][:last+1]),observe)
    return dict(scope=__doc__,indices=list(indices),target=target,partner=partner,cross_tails=cross_tails,
        rows=rows,expected_upload_ledger=ledger,seconds=perf_counter()-started,
        boundary_contract='Earlier published prefix restored only before consumption; no observed outcome injected. Unknown terminal days use explicit latest-plan actions for native comparisons, never default rest.')


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument('--controls',type=Path,required=True)
    p.add_argument('--output-json',type=Path,required=True)
    p.add_argument('--indices',type=int,nargs=3,default=(2,3,4))
    p.add_argument('--target',type=date.fromisoformat,default=date(2026,10,1))
    p.add_argument('--partner',type=date.fromisoformat,default=date(2026,10,2))
    p.add_argument('--cross-tails',action='store_true')
    a=p.parse_args()
    if a.output_json.exists():
        p.error('Require new output path')
    controls=json.loads(a.controls.read_text())
    if len(controls['branches'])!=1:
        p.error('Require single verified branch')
    branch=next(iter(controls['branches'].values()))
    source=Path(branch['source']).read_bytes()
    if sha256(source).hexdigest()!=branch['source_sha256']:
        raise ValueError('Source changed since verification')
    result=investigate(json.loads(source),branch,indices=a.indices,target=a.target,partner=a.partner,cross_tails=a.cross_tails)
    result.update(source=branch['source'],source_sha256=branch['source_sha256'],
        controls=str(a.controls),controls_sha256=sha256(a.controls.read_bytes()).hexdigest())
    with a.output_json.open('x') as out:
        json.dump(plain(result),out,indent=2,allow_nan=False)


if __name__=='__main__':
    main()
