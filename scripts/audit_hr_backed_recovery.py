"""Daily/upload component replay using transformed original HR, not calendar search."""
from argparse import ArgumentParser
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter
import json

from run_analysis.config import load_config
from run_analysis.recovery_assessment import sensitivity_panel
from run_analysis.workload_accounting import compare_execution
from run_analysis.workload_profile import prescribed_profile
from run_analysis.web.schemas import RecommendationResponse, WorkoutStep
from scripts.audit_backward_controls import digest, plain
from scripts.audit_recovery_placement import selected_program
from scripts.audit_sandbox_recovery_adapter import policies
from scripts.hr_backed_sandbox import load_bank, choose_donor, transformed_entry, issued_envelope
from scripts.sandbox_recovery_adapter import freeze_entries, session_key
from scripts.typed_continuation_prototype import runs


def seed(bank, start):
    result = []
    for i, day in enumerate(range(-32, 0, 2)):
        role = ('long','easy','intervals','easy')[i%4]
        miles = {'long':7.5, 'easy':3.8, 'intervals':4.5}[role]
        at = start+timedelta(days=day,hours=7)
        rec = RecommendationResponse(generated_at=start-timedelta(days=33), fitness_state_as_of=at,
            planned_for=at,workout_type=role,title='HR-backed synthetic seed',
            distance_target_miles=miles,distance_range_miles=(miles,miles),confidence='low',readiness='ready',
            structure=[WorkoutStep(instruction='8 controlled reps',phase='work',repetitions=8,
                work_duration_minutes=1,recovery_duration_minutes=1)] if role=='intervals' else [])
        result.append(transformed_entry(choose_donor(bank,rec,variant=i%3),rec))
    return result


def shifted_program(journal):
    with journal.open() as stream:
        snapshot = json.loads(next(stream))
    _, full = selected_program(snapshot,None)
    # All original donors predate this counterfactual timeline. No original run
    # is retroactively declared compliant with a real historical prescription.
    start = datetime(2027,1,1,tzinfo=timezone.utc)
    delta = start.date()-full[0].date
    program = []
    for day in full:
        rec = day.recommendation
        if rec is not None:
            at = rec.planned_for+delta
            rec = rec.model_copy(deep=True,update={'planned_for':at,'fitness_state_as_of':at,'generated_at':start})
            if 'work_duration_missing' in prescribed_profile(rec)['limitations']:
                # The saved progression is prose-only. Make this synthetic
                # test intent explicit BEFORE choosing donors, using the old
                # simulator's quarter-session approximation, not observed HR.
                # This is not a production prescription or inferred real work.
                minutes=round(rec.planned_distance_miles*11*.25,1)
                rec=rec.model_copy(update={'title':rec.title+' [numeric sandbox finish]',
                    'structure':[s.model_copy(update={'duration_minutes':minutes})
                                 if s.phase=='work' and s.duration_minutes is None
                                 and s.work_duration_minutes is None and s.work_duration_range_minutes is None
                                 else s for s in rec.structure]})
        program.append(day.model_copy(update={'date':day.date+delta,'recommendation':rec}))
    return start, program


def replay(bank, program, start, families, *, scenario, days):
    started = perf_counter()
    prescriptions = runs(program)
    donors = {session_key(r.planned_for):choose_donor(bank,r) for r in prescriptions}
    history = seed(bank,start)
    cache, daily, uploads = {}, [], []
    def assess(prefix, rec, entry):
        key = digest((prefix,rec.planned_for,entry.workload,entry.work_minutes))
        if key not in cache:
            result = {}
            for family, group in families.items():
                rows = []
                for policy in group:
                    panel = sensitivity_panel(prefix,entry.workload,rec.planned_for,(policy,),
                                              proposed_work_minutes=entry.work_minutes)
                    rows.append(dict(policy=policy.name,status=panel['status'],
                        bounded_input_uncertainty=any(r['status']=='uncertain' for r in panel['scenarios']),
                        timing_disagreement=len({r['status'] for r in panel['scenarios']})>1))
                result[family] = rows
            cache[key] = result
        return cache[key]
    def compact(result):
        return {family:dict(status_counts=dict(Counter(r['status'] for r in rows)),
            bounded_input_uncertainty=sum(r['bounded_input_uncertainty'] for r in rows),
            timing_disagreement=sum(r['timing_disagreement'] for r in rows)) for family,rows in result.items()}

    previous_daily = None
    for day in range(days):
        at = start+timedelta(days=day)
        future = [r for r in prescriptions if r.planned_for >= at]
        entries = [(transformed_entry(donors[session_key(r.planned_for)],r,source='projected')
                    if scenario=='exact' else issued_envelope(donors[session_key(r.planned_for)],r)) for r in future]
        snapshot = freeze_entries(future,[*history,*entries],issued_at=at,
            assumptions=dict(source='Precommitted recorded-HR donors; synthetic transformations',scenario=scenario))
        current = assess(history,future[0],entries[0])
        signature = digest((history,entries[0]))
        if previous_daily and previous_daily[0] == signature and previous_daily[1] != current:
            raise AssertionError('Unchanged daily inputs changed recovery assessment')
        previous_daily = (signature,current)
        daily.append(dict(at=at,next_run=future[0].planned_for,assessments=compact(current)))
        today = [r for r in future if r.planned_for.date()==at.date()]
        for rec in today:
            key = session_key(rec.planned_for)
            expected = next(e for e in entries if e.key==key)
            next_rec = next(r for r in future if r.planned_for>rec.planned_for)
            next_entry = next(e for e in entries if e.key==session_key(next_rec.planned_for))
            donor = donors[key]
            index = len(uploads)
            if scenario=='different_donor':
                donor=choose_donor(bank,rec,exclude=(donor.key,))
            low,high=rec.distance_range_miles
            target=rec.planned_distance_miles
            miles=target if scenario=='exact' else target+(.8*(high-target) if index%2 else -.8*(target-low))
            if scenario=='above_range' and index==2:
                miles=high*1.2
            shifted=rec.planned_for+(timedelta(hours=(-6,6)[index%2]) if scenario=='time_shift' else timedelta())
            completion=transformed_entry(donor,rec,miles=miles,at=shifted)
            if scenario=='exact' and (expected.start != completion.start or expected.end != completion.end
                    or expected.work_minutes != completion.work_minutes
                    or replace(expected.workload,source='observed') != completion.workload):
                raise AssertionError('Exact control did not preserve physical HR workload')
            expected_prefix=snapshot.prefix(next_entry.key)
            updated=snapshot.after_completion(completion)
            actual_prefix=tuple(e for e in updated if e.start<next_rec.planned_for)
            before=assess(expected_prefix,next_rec,next_entry)
            after=assess(actual_prefix,next_rec,next_entry)
            transitions={family:dict(Counter(f'{a["status"]} -> {b["status"]}'
                         for a,b in zip(before[family],after[family]))) for family in families}
            if scenario=='exact' and before!=after:
                raise AssertionError('Identical HR-backed completion changed assessment')
            execution=compare_execution(expected.workload,completion.workload)
            uploads.append(dict(at=completion.end,expected_donor=donors[key].key,actual_donor=donor.key,
                issued_snapshot_digest=snapshot.digest,kind=rec.workout_type.value,
                target=target,actual_miles=miles,distance_in_range=low<=miles<=high,
                numeric_work_matches=expected.work_minutes==completion.work_minutes,
                expected_workload=asdict(expected.workload),actual_workload=asdict(completion.workload),
                execution=execution,offset_hours=(shifted-rec.planned_for).total_seconds()/3600,
                before=compact(before),after=compact(after),policy_transitions=transitions))
            history.append(completion)
        print(f'{scenario}: day {day+1}/{days}, uploads {len(uploads)}',flush=True) if (day+1)%7==0 else None
    return dict(scenario=scenario,days=days,daily=daily,uploads=uploads,seconds=perf_counter()-started,
        distance_compliant=sum(u['distance_in_range'] for u in uploads),
        numeric_work_compliant=sum(u['numeric_work_matches'] for u in uploads),
        execution_outcomes={name:dict(Counter(u['execution'][name] for u in uploads))
                            for name in ('distance_meters','running_minutes','cardio_points')},
        distinct_assessments=len(cache),planner_influence=False)


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--journal',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--days',type=int,default=21,choices=range(1,22))
    parser.add_argument('--scenarios',nargs='+',choices=['exact','in_range','time_shift','different_donor','above_range'],
                        default=['exact','in_range','time_shift','different_donor','above_range'])
    args=parser.parse_args()
    if args.output.exists(): parser.error('Choose a new output path')
    started=perf_counter()
    config=load_config(Path('config.yaml'))
    bank=load_bank(Path(config['paths']['database']),config)
    bank_before=digest(bank)
    families=policies()
    start,program=shifted_program(args.journal)
    result=dict(scope=__doc__,source_bank=[asdict(d) for d in bank],source_bank_hash=bank_before,
        policy_counts={name:len(group) for name,group in families.items()},
        scenario_prescription_overrides=[r.model_dump(mode='json') for r in runs(program)
                                         if '[numeric sandbox finish]' in r.title],
        assumptions=['Synthetic transformations of original HR-phase integrals, not new physiological observations',
                     'No fitter, readiness label or outcome used to select donors',
                     'Fixed saved calendar/distances; prose-only progression gets explicit numeric test work before donor choice',
                     'Daily assessments, not daily calendar optimization',
                     'No automatic new recovery default or replacement live penalty'],
        scenarios=[replay(bank,program,start,families,scenario=s,days=args.days) for s in args.scenarios])
    if digest(bank)!=bank_before: raise AssertionError('Original donor bank mutated')
    result['seconds']=perf_counter()-started
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f: json.dump(plain(result),f,indent=2,allow_nan=False)
    print(json.dumps(dict(seconds=result['seconds'],runs=[dict(scenario=s['scenario'],
        uploads=len(s['uploads']),compliant=s['distance_compliant'],seconds=s['seconds']) for s in result['scenarios']]),indent=2))


if __name__=='__main__': main()
