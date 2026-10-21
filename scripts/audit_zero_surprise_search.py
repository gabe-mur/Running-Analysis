"""Trace rollover incumbent loss and one later winner through allocation stages.

Diagnostic only: replay saved frames, score concrete programs, and widen the
allocator only for one witnessed calendar. Never publish a counterfactual.
"""
from argparse import ArgumentParser
from contextlib import contextmanager
from datetime import date, timedelta
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
from time import perf_counter

from run_analysis import weekly_schedule as planner
from scripts.audit_backward_controls import plain, validate_control
from scripts.audit_backward_same_origin import coverage
from scripts.audit_continuation_upload import reconstruct, unpack_soft, hard_score, evaluate_physical, raw_state_lookup
from scripts.audit_distance_continuity import components
from scripts.audit_horizon_rollover import absolute_slice, probe
from scripts.audit_visible_week_edits import replace_prescribed_dose, splice_tail
from scripts.continuation_replan import fingerprint
from scripts.expected_completion import expected_completion_policy
from scripts.finalized_distance_search import neighbors


def translate_saved_program(previous, current, *, origin, hard_end, full_end):
    """Consume elapsed dates and use explicit current terminal actions."""
    if previous[0].date >= origin or current[0].date != origin:
        raise ValueError('Require an actually advanced boundary')
    retained=absolute_slice(previous,origin,min(full_end,previous[-1].date+timedelta(days=1)))
    extension=absolute_slice(current,retained[-1].date+timedelta(days=1),full_end)
    translated=retained+extension
    hard=absolute_slice(translated,origin,hard_end)
    full=absolute_slice(translated,origin,full_end)
    return hard,full,extension


def copy_doses(program, donor, workout_types):
    """Copy dose semantics on matching sessions, retaining dates and times."""
    source={d.date:d for d in donor}
    result=list(program); changed=[]
    for i,day in enumerate(program):
        other=source.get(day.date)
        rec=day.recommendation
        if rec is None or other is None or other.recommendation is None:
            continue
        if rec.workout_type not in workout_types:
            continue
        if other.recommendation.workout_type != rec.workout_type:
            raise ValueError('Dose donor changes workout type')
        fields=('distance_target_miles','distance_range_miles','planning_role','title')
        revised=rec.model_copy(update={k:getattr(other.recommendation,k) for k in fields})
        result[i]=day.model_copy(update={'recommendation':revised,'day_role':other.day_role})
        if (rec.distance_target_miles,rec.distance_range_miles)!=(revised.distance_target_miles,revised.distance_range_miles):
            changed.append(day.date.isoformat())
    return result,changed


def copy_times(program, donor, dates):
    """Promote a saved prescription onto an already native candidate time."""
    source={d.date:d for d in donor}; result=list(program)
    for i,day in enumerate(program):
        if day.date not in dates:
            continue
        other=source.get(day.date)
        if not day.recommendation or not other or not other.recommendation:
            raise ValueError('Time promotion requires two running prescriptions')
        if day.recommendation.workout_type!=other.recommendation.workout_type:
            raise ValueError('Time promotion cannot change workout type')
        at=other.recommendation.planned_for
        rec=day.recommendation.model_copy(update={'planned_for':at,'fitness_state_as_of':at})
        result[i]=day.model_copy(update={'planned_at':at,'recommendation':rec})
    if not dates<={d.date for d in program}:
        raise ValueError('Missing time-promotion date')
    return result


def sessions(days):
    return [dict(date=d.date.isoformat(),type=d.recommendation.workout_type.value,
        target=d.recommendation.planned_distance_miles,role=d.recommendation.planning_role)
        for d in days if d.recommendation and d.recommendation.workout_type!=planner.WorkoutType.REST]


def compact_probe(row):
    """Retain causal evidence without duplicating large daily accounting traces."""
    result={k:row.get(k) for k in ('cost','components','feasible','native_timing_covered',
        'added_raw_state_times','stable_prefix','hard_fingerprint','full_fingerprint','policy_rate','soft_target')}
    result['checks']=dict(checks_pass=row['checks']['checks_pass'],failed_offsets=row['checks']['failed_offsets'])
    result['gates']={k:row['gates'][k] for k in ('failures','whole_hard_peak_failures','strict_cadence_findings')}
    if 'coverage' in row:
        c=row['coverage']
        result['coverage']=dict(offsets=c['offsets'],proposal_calendar_rows=len(c['proposal_calendar_rows']),
            finalist_calendar_rows=len(c['finalist_calendar_rows']),exact_hard_rows=len(c['exact_hard_rows']),
            exact_hard_tail_rows=len(c['exact_hard_tail_rows']),exact_pair_scored=c['exact_pair_scored'],
            limitation=c['limitation'])
    return result


def capture_allocator(run):
    """Compare bounded proxy-retention settings on one exact calendar."""
    original=planner._allocate_visible_distance_ranges
    calls=[]; depth=0
    def allocate(*args,**kwargs):
        nonlocal depth
        if depth or kwargs.get('allocation_selector') is None:
            return original(*args,**kwargs)
        depth+=1
        try:
            base_days,allocation_states,config=args[0],args[1],args[3]
            planned=[]; long_inputs=[]
            for i,(day,raw) in enumerate(zip(base_days,allocation_states,strict=True)):
                rec=day.recommendation
                if rec is None or rec.workout_type==planner.WorkoutType.REST:
                    continue
                projected=planner._project_state(raw,planned,config)
                if rec.workout_type==planner.WorkoutType.LONG:
                    long_inputs.append(dict(date=day.date.isoformat(),recommendation_target=rec.planned_distance_miles,
                        raw_longest_30d=raw.longest_run_30d_miles,
                        raw_retained_capacity=raw.retained_long_run_capacity_miles,
                        raw_progression_reference=planner.single_session_progression_reference_miles(raw),
                        conditional_longest_30d=projected.longest_run_30d_miles,
                        conditional_retained_capacity=projected.retained_long_run_capacity_miles,
                        conditional_progression_reference=planner.single_session_progression_reference_miles(projected)))
                planned.append(rec)
            variants=[('default',{}),('wider_finalists',{'allocation_candidate_limit':16}),
                ('wider_dp',{'allocation_candidate_limit':16,'assignments_per_total':8})]
            results=[]
            for label,overrides in variants:
                seen=[]; selector=kwargs['allocation_selector']
                def observed(candidate, *, _selector=selector, _seen=seen):
                    score=_selector(candidate)
                    _seen.append(dict(score=score,fingerprint=fingerprint(candidate),sessions=sessions(candidate)))
                    return score
                stamp=perf_counter()
                selected=original(*args,**{**kwargs,**overrides,'allocation_selector':observed})
                results.append(dict(label=label,seconds=perf_counter()-stamp,
                    candidates=seen,selected_fingerprint=fingerprint(selected),selected_sessions=sessions(selected)))
            calls.append(dict(long_inputs=long_inputs,variants=results))
            return original(*args,**kwargs)
        finally:
            depth-=1
    planner._allocate_visible_distance_ranges=allocate
    try:
        value=run()
    finally:
        planner._allocate_visible_distance_ranges=original
    return value,calls


@contextmanager
def conditional_allocation_states(config):
    """Diagnostic counterfactual: feed allocation the same planned prefix."""
    original=planner._allocation_states_at_planned_times
    def projected(days,daily_states,daily_state_options):
        raw_states=original(days,daily_states,daily_state_options)
        result=[]; planned=[]
        for day,raw in zip(days,raw_states,strict=True):
            result.append(planner._project_state(raw,planned,config))
            rec=day.recommendation
            if rec is not None and rec.workout_type!=planner.WorkoutType.REST:
                planned.append(rec)
        return result
    planner._allocation_states_at_planned_times=projected
    try:
        yield
    finally:
        planner._allocation_states_at_planned_times=original


def investigate(report, controls):
    verified={r['index']:r for r in controls['decisions']}
    originals={i:unpack_soft(verified[i]['full_program']) for i in (5,6,7,9,10)}
    for i,program in originals.items():
        if fingerprint(program)!=verified[i]['full_fingerprint']:
            raise ValueError('Saved verified program mismatch')
    result={'scope':__doc__,'terminal':{},'rollover':{},'long_upload':{}}
    def observe(index,inputs,days,evidence):
        if index not in (6,7,9):
            return
        snapshot=report['replans'][index]
        own=originals[index]
        with expected_completion_policy(inputs['config']):
            scored=evaluate_physical(inputs,days,evidence,full=own)
            validate_control(snapshot,inputs,days,own,evidence['runs'],sum(components(scored).values()))
        origin=inputs['daily_states'][0].as_of.date()
        hard_end=date.fromisoformat(verified[index]['hard_end_exclusive'])
        full_end=date.fromisoformat(verified[index]['full_end_exclusive'])
        if index==6:
            _,translated,_=translate_saved_program(originals[5],own,
                origin=origin,hard_end=hard_end,full_end=full_end)
            rows=[]
            promoted=copy_times(translated,own,{date(2026,10,16)})
            for timing,base in (('saved_time',translated),('native_promoted_time',promoted)):
                for suffix_days in range(9):
                    candidate=(base if suffix_days==0 else splice_tail(
                        base,own,full_end-timedelta(days=suffix_days)))
                    row=probe(snapshot,inputs,evidence,candidate,hard_end,full_end)
                    summary=compact_probe(row)
                    summary.update(timing=timing,suffix_days=suffix_days,terminal_sessions=sessions(
                        absolute_slice(candidate,full_end-timedelta(days=max(1,suffix_days)),full_end)))
                    rows.append(summary)
            result['terminal']=dict(previous=5,current=6,rows=rows)
            return
        if index==7:
            hard,full,extension=translate_saved_program(originals[6],own,
                origin=origin,hard_end=hard_end,full_end=full_end)
            checked=probe(snapshot,inputs,evidence,full,hard_end,full_end)
            checked['coverage']=coverage(report,index,hard,full)
            result['rollover']=dict(previous=6,current=7,consumed=[d.date.isoformat()
                for d in originals[6] if d.date<origin],extension=[dict(date=d.date.isoformat(),
                    type=d.recommendation.workout_type.value if d.recommendation else 'rest') for d in extension],
                selected_cost=report['replans'][index]['planner_diagnostics']['continuation_selection']['search_cost'],
                translated=compact_probe(checked))
            return

        # Backward witness: retain the current-day prescribed long, then use
        # the later program. It is candidate evidence, never observed history.
        # Frame 10 starts later. Build the backward date grid explicitly from
        # the frame-9 opening action; do not treat that prescription as an
        # observed completion inside the earlier frame.
        event_full=absolute_slice(own,origin,originals[10][0].date)+absolute_slice(
            originals[10],originals[10][0].date,full_end)
        event_hard=absolute_slice(event_full,origin,hard_end)
        offsets=[i for i,d in enumerate(event_hard) if d.recommendation
            and d.recommendation.workout_type!=planner.WorkoutType.REST]
        roles={i:('long' if d.recommendation.workout_type==planner.WorkoutType.LONG else
            'quality' if d.recommendation.workout_type in planner.QUALITY_WORKOUT_TYPES else 'easy')
            for i,d in enumerate(event_hard) if d.recommendation}
        (native_hard,_,_),allocator_calls=capture_allocator(lambda: hard_score(inputs,offsets,roles=roles))
        with conditional_allocation_states(inputs['config']):
            conditional_hard,_,_=hard_score(inputs,offsets,roles=roles)
        if [d.date for d in native_hard] != [d.date for d in event_hard]:
            raise ValueError('Native materialization changed the requested date grid')
        current_tail=absolute_slice(own,hard_end,full_end)
        event_tail=absolute_slice(event_full,hard_end,full_end)
        variants=[]
        def add(label,hard,tail):
            program=hard+tail
            row=probe(snapshot,inputs,evidence,program,hard_end,full_end)
            row.update(label=label,hard_sessions=sessions(hard),tail_fingerprint=fingerprint(tail),
                coverage=coverage(report,index,hard,program))
            summary=compact_probe(row)
            summary.update(label=label,hard_sessions=sessions(hard),tail_fingerprint=fingerprint(tail))
            variants.append(summary)
        add('published',absolute_slice(own,origin,hard_end),current_tail)
        add('native_event_calendar_current_tail',native_hard,current_tail)
        add('native_event_calendar_event_tail',native_hard,event_tail)
        add('conditional_state_calendar_current_tail',conditional_hard,current_tail)
        add('conditional_state_calendar_event_tail',conditional_hard,event_tail)
        changed_easy=[d.date for d,e in zip(conditional_hard,event_hard,strict=True)
            if d.recommendation and e.recommendation
            and d.recommendation.workout_type==planner.WorkoutType.EASY
            and (d.recommendation.distance_target_miles,d.recommendation.distance_range_miles)
                !=(e.recommendation.distance_target_miles,e.recommendation.distance_range_miles)]
        for count in range(1,len(changed_easy)+1):
            for chosen in combinations(changed_easy,count):
                hard=replace_prescribed_dose(conditional_hard,event_hard,set(chosen))
                for tail_label,tail in (('current_tail',current_tail),('event_tail',event_tail)):
                    add('conditional_easy_'+','.join(d.isoformat() for d in chosen)+'_'+tail_label,hard,tail)
                    variants[-1]['changed_dates']=[d.isoformat() for d in chosen]
        path=[]; cursor=conditional_hard
        for chosen in (date(2026,10,1),date(2026,10,9),date(2026,10,2)):
            intended=replace_prescribed_dose(cursor,event_hard,{chosen})
            available=[]; matched=False
            for candidate in neighbors(cursor,inputs,raw_state_lookup(inputs,evidence),visible_days=21):
                a=next(d.recommendation for d in candidate if d.date==chosen)
                available.append(dict(target=a.planned_distance_miles,range=a.distance_range_miles,
                    role=a.planning_role))
                if fingerprint(candidate)==fingerprint(intended):
                    matched=True
            path.append(dict(date=chosen.isoformat(),desired_target=next(d.recommendation.planned_distance_miles
                for d in intended if d.date==chosen),available=available,exact_neighbor=matched))
            cursor=intended
        add('eventual_hard_current_tail',event_hard,current_tail)
        add('eventual_hard_event_tail',event_hard,event_tail)
        for label,types in (('easy', {planner.WorkoutType.EASY}),('long',{planner.WorkoutType.LONG}),
                            ('easy_and_long',{planner.WorkoutType.EASY,planner.WorkoutType.LONG})):
            hard,changed=copy_doses(native_hard,event_hard,types)
            for tail_label,tail in (('current_tail',current_tail),('event_tail',event_tail)):
                add(f'native_plus_{label}_{tail_label}',hard,tail)
                variants[-1]['changed_dates']=changed
        result['long_upload']=dict(before=9,after=10,event_calendar_offsets=offsets,
            allocator_calls=allocator_calls,conditional_state_sessions=sessions(conditional_hard),variants=variants)
        result['long_upload']['coordinate_path']=path
    reconstruct(dict(report,replans=report['replans'][:10]),observe)
    return result


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument('--controls',type=Path,required=True)
    p.add_argument('--output-json',type=Path,required=True)
    a=p.parse_args()
    if a.output_json.exists():
        p.error('Require a new output path')
    controls=json.loads(a.controls.read_text())
    if len(controls['branches'])!=1:
        p.error('Require a single verified branch')
    branch=next(iter(controls['branches'].values()))
    source=Path(branch['source']); content=source.read_bytes()
    if sha256(content).hexdigest()!=branch['source_sha256']:
        raise ValueError('Source changed since verification')
    result=investigate(json.loads(content),branch)
    result.update(source=str(source),source_sha256=branch['source_sha256'],controls=str(a.controls),
        controls_sha256=sha256(a.controls.read_bytes()).hexdigest())
    with a.output_json.open('x') as out:
        json.dump(plain(result),out,indent=2,allow_nan=False)


if __name__=='__main__':
    main()
