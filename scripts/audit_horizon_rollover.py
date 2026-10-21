"""BA3: fixed absolute-date controls for a rest-day rollover, without search.

Keep original ledgers and compare concrete programs; separate dropping elapsed
rest, moving the hard/soft cut, and extending the full accounting endpoint.
"""
from argparse import ArgumentParser
from dataclasses import asdict
from datetime import date, timedelta
from hashlib import sha256
import json
from pathlib import Path

from run_analysis import weekly_schedule as planner
from run_analysis.aerobic_acceptance import assess_aerobic_exposure
from scripts.audit_backward_controls import plain, validate_control
from scripts.audit_backward_same_origin import coverage
from scripts.audit_continuation_upload import reconstruct, unpack_soft, evaluate_physical, raw_state_lookup, hard_score
from scripts.audit_distance_continuity import components, replace_easy_dose
from scripts.audit_regrowth_gates import check_candidate
from scripts.continuation_checks import check_sequence
from scripts.expected_completion import expected_completion_policy
from scripts.continuation_replan import fingerprint, stable_prefix


def absolute_slice(program, start, end):
    selected = [d for d in program if start <= d.date < end]
    expected = [start+timedelta(days=i) for i in range((end-start).days)]
    if [d.date for d in selected] != expected or len({d.date for d in program}) != len(program):
        raise ValueError('Missing or duplicate physical dates; never pad as rest')
    return selected


def align_rest_rollover(before, after):
    if after[0].date != before[0].date+timedelta(days=1):
        raise ValueError('Require a one-day rollover')
    rec = before[0].recommendation
    if rec is not None and rec.workout_type != planner.WorkoutType.REST:
        raise ValueError('Elapsed run requires separate expected-completion accounting')
    # This is a documented elapsed rest day, not an invented future rest day.
    return before, [before[0], *after]


def exact_timed_inputs(inputs, hard, state_at):
    """Supply raw historical states for fixed prescriptions, never retime them.

    A former soft session may not use a native hard-menu time. These additions
    permit physical accounting, not a claim of native candidate eligibility.
    """
    options=[list(row) for row in inputs['daily_state_options'][:len(hard)]]
    additions=[]
    for i,day in enumerate(hard):
        rec=day.recommendation
        if rec is None:
            continue
        at=rec.planned_for or day.planned_at
        if at==inputs['daily_states'][i].as_of or any(s.as_of==at for s in options[i]):
            continue
        raw=state_at(at)
        if raw.as_of!=at:
            raise ValueError('Exact timed state mismatch')
        options[i].append(raw)
        additions.append(at.isoformat())
    return dict(inputs,daily_states=inputs['daily_states'][:len(hard)],
        daily_state_options=options),additions


def probe(snapshot, inputs, evidence, program, hard_end, full_end):
    start=inputs['daily_states'][0].as_of.date()
    hard=absolute_slice(program,start,hard_end)
    full=absolute_slice(program,start,full_end)
    adjusted,added_times=exact_timed_inputs(inputs,hard,raw_state_lookup(inputs,evidence))
    if len(adjusted['daily_states'])!=len(hard):
        raise ValueError('Missing native hard states')
    with expected_completion_policy(inputs['config']):
        checks=check_sequence(full,adjusted,raw_state_lookup(adjusted,evidence),())
        gates=check_candidate(snapshot,adjusted,hard)
        offsets=[i for i,d in enumerate(hard) if d.recommendation and d.recommendation.workout_type!=planner.WorkoutType.REST]
        _,_,details=hard_score(adjusted,offsets,days=hard)
        acceptance=assess_aerobic_exposure(**details['aerobic_acceptance_facts'],
            planning_mode=inputs['target_evidence'].planning_mode.value)
        feasible=(checks['checks_pass'] and not acceptance.rejected and acceptance.reason!='missing_context'
            and not any(gates[k] for k in ('failures','whole_hard_peak_failures','strict_cadence_findings'))
            and not(set(offsets)&(set(inputs['forced_rest_offsets'])|set(inputs['completed_run_offsets']))))
        row=dict(start=start.isoformat(),hard_end_exclusive=hard_end.isoformat(),full_end_exclusive=full_end.isoformat(),
            hard_days=len(hard),soft_days=len(full)-len(hard),feasible=feasible,checks=checks,gates=gates,
            acceptance=asdict(acceptance),stable_prefix=stable_prefix(hard,adjusted),
            added_raw_state_times=added_times,native_timing_covered=not added_times,
            hard_fingerprint=fingerprint(hard),full_fingerprint=fingerprint(full))
        if feasible:
            scored=evaluate_physical(adjusted,hard,evidence,full=full,length=len(full)-len(hard))
            parts=components(scored)
            row.update(cost=sum(parts.values()),components=parts,
                recovery_components=scored['continuation']['trajectory_recovery_components'],
                volume_path=scored['continuation']['volume_path'],
                funding=scored['continuation']['funding_accounting'],
                policy_rate=scored['continuation']['policy_rate'],soft_target=scored['continuation']['soft_target'])
    return row


def investigate(report, controls, *, before=2, after=3):
    frames={r['index']:r for r in controls['decisions']}
    a,b=frames[before],frames[after]
    if a['frame']['completed_history']!=b['frame']['completed_history']:
        raise ValueError('Rest-rollover comparison requires identical completed history')
    earlier=unpack_soft(a['full_program']); later=unpack_soft(b['full_program'])
    for program,saved in ((earlier,a),(later,b)):
        if fingerprint(program)!=saved['full_fingerprint']:
            raise ValueError('Saved physical fingerprint mismatch')
    earlier,later=align_rest_rollover(earlier,later)
    hard_end=date.fromisoformat(a['hard_end_exclusive'])
    full_end=date.fromisoformat(a['full_end_exclusive'])
    later_hard_end=date.fromisoformat(b['hard_end_exclusive'])
    later_full_end=date.fromisoformat(b['full_end_exclusive'])
    # Explicit experimental extension, copied from the later planned program.
    # It is a proposed future action/rest, never an observed outcome or default.
    extension=absolute_slice(later,full_end,later_full_end)
    variants={'earlier':earlier+extension,'later':later}
    target_date=date.fromisoformat(b['origin'][:10])
    rows=[]
    def observe(index,inputs,days,evidence):
        if index not in (before,after):
            return
        snapshot=report['replans'][index]
        own=earlier if index==before else later[1:]
        with expected_completion_policy(inputs['config']):
            score=evaluate_physical(inputs,days,evidence,full=own)
            validate_control(snapshot,inputs,days,own,evidence['runs'],sum(components(score).values()))
        modes=[('common_absolute_endpoints',hard_end,full_end)]
        if index==after:
            modes.extend([('move_hard_cut_only',later_hard_end,full_end),
                ('native_horizon_explicit_extension',later_hard_end,later_full_end)])
        for mode,h_end,f_end in modes:
            for label,program in variants.items():
                for dose in (None,3.3,3.5):
                    candidate=(program if dose is None else replace_easy_dose(program,target_date,dose,
                        (3.,3.5) if dose==3.3 else (3.2,3.8)))
                    result=probe(snapshot,inputs,evidence,candidate,h_end,f_end)
                    result.update(frame=index,mode=mode,program=label,dose_override=dose)
                    if mode=='native_horizon_explicit_extension':
                        hard=absolute_slice(candidate,inputs['daily_states'][0].as_of.date(),h_end)
                        full=absolute_slice(candidate,inputs['daily_states'][0].as_of.date(),f_end)
                        result['coverage']=coverage(report,index,hard,full)
                    rows.append(result)
            print(f"Horizon frame {index} / {mode}: " + ', '.join(
                f"{r['program']} dose={r['dose_override']}: {r.get('cost','infeasible')}" for r in rows[-6:]),flush=True)
    reconstruct(dict(report,replans=report['replans'][:after+1]),observe)
    return dict(scope=__doc__,rows=rows,extension_source='Later saved plan, explicit matched terminal-day experiment',
        extension=[dict(date=d.date.isoformat(),type=d.recommendation.workout_type.value if d.recommendation else 'rest') for d in extension])


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--controls',type=Path,required=True)
    parser.add_argument('--output-json',type=Path,required=True)
    args=parser.parse_args()
    if args.output_json.exists():
        parser.error('Require a new output path')
    controls=json.loads(args.controls.read_text())
    if len(controls['branches'])!=1:
        parser.error('Require one replay branch')
    branch=next(iter(controls['branches'].values()))
    source=Path(branch['source']).read_bytes()
    if sha256(source).hexdigest()!=branch['source_sha256']:
        raise ValueError('Source changed since verification')
    result=investigate(json.loads(source),branch)
    result.update(source=branch['source'],source_sha256=branch['source_sha256'],
        controls=str(args.controls),controls_sha256=sha256(args.controls.read_bytes()).hexdigest())
    encoded=json.dumps(plain(result),indent=2,allow_nan=False)
    with args.output_json.open('x') as stream:
        stream.write(encoded)


if __name__=='__main__':
    main()
