"""Bounded offline scheduler driven by new recovery, volume and key exposure.

Not a deployment or replacement for production health/availability validation.
Fixed declared doses/targets isolate integration from capacity-learning policy.
Daily searches move dates, exchange occupied dates, and add/remove easy support.
No legacy readiness, recovery score, role selector or sequence veto is called.
"""
from argparse import ArgumentParser
from collections import Counter
from dataclasses import dataclass, replace, asdict
from datetime import datetime, timedelta, timezone
from math import log
from pathlib import Path
from time import perf_counter
import json

from run_analysis.config import load_config
from run_analysis.recovery_assessment import AssessmentPolicy, SessionEntry
from run_analysis import weekly_schedule as planner
from run_analysis.web.schemas import RecommendationResponse, WorkoutStep
from scripts.audit_backward_controls import plain, digest
from scripts.audit_hr_backed_recovery import shifted_program
from scripts.hr_backed_sandbox import load_bank, choose_donor, transformed_entry
from scripts.typed_continuation_prototype import runs


@dataclass(frozen=True)
class Run:
    role: str
    entry: SessionEntry

    @property
    def day(self): return self.entry.start.date()

    @property
    def miles(self): return self.entry.workload.distance_meters.low/1609.344

    def on(self, day, *, key=None):
        offset=day-self.day
        return replace(self,entry=replace(self.entry,key=key or self.entry.key,
            start=self.entry.start+offset,end=self.entry.end+offset))


def signature(program):
    # Source/identity/provenance do not change the prescribed physical program.
    return tuple((r.day.isoformat(),r.role,round(r.miles,8),r.entry.workload.running_minutes.high,
                  r.entry.workload.cardio_points.high,r.entry.work_minutes.high) for r in program)


def recovery_path(history, program, policy, phase=0):
    """O(n) exact exponential accumulation; tested against the reference assessor.

    Timing phase is tied to absolute dates, not remaining session index: uploading
    a run cannot flip all later early/late assumptions. Actual history stays actual.
    """
    keys=[r.entry.key for r in (*history,*program)]
    if len(set(keys))!=len(keys): raise ValueError('Duplicate session identity')
    specs=[('running_minutes',policy.running_minutes_envelope,policy.running_half_life_hours),
           ('cardio_points',policy.cardio_points_envelope,policy.cardio_half_life_hours)]
    if policy.work_minutes_envelope is not None:
        specs.append(('work_minutes',policy.work_minutes_envelope,policy.work_half_life_hours))
    def dose(run,name):
        q=run.entry.work_minutes if name=='work_minutes' else getattr(run.entry.workload,name)
        if q is None: raise ValueError('Unbounded workload cannot authorize a trial prescription')
        return q.high
    residual=None; previous_end=None; maximum=total=0.; rows=[]
    for run in program:
        offset=phase*(6 if run.day.toordinal()%2 else -6)
        at=run.entry.start+timedelta(hours=offset)
        duration=run.entry.end-run.entry.start
        if residual is None:
            if any(r.entry.end>at for r in history): raise ValueError('Unfinished history')
            residual=[sum(dose(r,n)*2**(-((at-r.entry.end).total_seconds()/3600)/h)
                          for r in history) for n,_,h in specs]
        else:
            if previous_end>at: raise ValueError('Overlapping candidate sessions')
            residual=[v*2**(-((at-previous_end).total_seconds()/3600)/h)
                      for v,(_,_,h) in zip(residual,specs)]
        demands=[v+dose(run,n) for v,(n,_,_) in zip(residual,specs)]
        excess=max(max(0.,v/b-1) for v,(_,b,_) in zip(demands,specs))
        maximum=max(maximum,excess); total+=excess
        rows.append(dict(date=str(run.day),role=run.role,excess=excess,
                         demands={n:v for (n,_,_),v in zip(specs,demands)}))
        residual=[v*2**(-(duration.total_seconds()/3600)/h)+dose(run,n)
                  for v,(n,_,h) in zip(residual,specs)]
        previous_end=at+duration
    return dict(maximum_excess=maximum,total_excess=total,sessions=rows)


def volume_and_cadence(history, program, origin, end, targets, ordinary):
    """Existing daily volume integral and overdue-key formula, one owner each.

    Evaluate the same absolute endpoint across the short replay. Completed
    in-trial mileage remains in the integral once; past daily cost is a constant
    shared by all candidates. Long/quality cadence includes the historical seam.
    """
    norm=log(2); half=7.
    before=[r for r in history if r.entry.start<origin]
    current=[r for r in history if r.entry.start>=origin]+list(program)
    if len({r.day for r in current})!=len(current): raise ValueError('Duplicate date in trial ledger')
    by_day={r.day:r for r in current}
    rate=sum(r.miles*norm*2**(-((origin-r.entry.start).total_seconds()/86400)/half) for r in before)
    previous={role:max((r.day for r in before if key_role(r.role)==role),default=None)
              for role in ('long','quality')}
    under=over=cadence=0.
    for i in range((end-origin).days):
        day=origin.date()+timedelta(days=i); run=by_day.get(day)
        hour=((run.entry.start-origin.replace(year=day.year,month=day.month,day=day.day)).total_seconds()/3600
              if run else None)
        rate=planner._advance_continuous_mileage_rate(rate,run.miles if run else 0.,
            daily_decay=2**(-1/half),session_normalization=norm,impulse_hour=hour)
        low,high=targets[min(i,len(targets)-1)]
        tolerance=max(.1,ordinary*norm)
        under+=max(0.,low-tolerance-rate)**2/half
        over+=max(0.,rate-high-tolerance)**2/half
        for role,last in previous.items():
            if last is not None:
                cadence+=planner._key_session_cadence_overdue_cost((day-last).days,7.)
        if run and key_role(run.role) in previous: previous[key_role(run.role)]=day
    return dict(volume=planner.TARGET_FIT_UNIT*(under+over)/ordinary**2,
                underload=under,overload=over,cadence=cadence)


def key_role(role): return 'quality' if role in {'intervals','tempo_threshold','race'} else role


def neighbors(program, today, easy_options, *, blocked=frozenset(), edit_days=21,reserved_keys=frozenset()):
    """Bounded physical-dose neighborhood; tail immutable beyond rolling day 21."""
    last=today+timedelta(days=edit_days-1)
    occupied={r.day for r in program}
    used_keys={r.entry.key for r in program}|set(reserved_keys)
    def normalized(rows): return tuple(sorted(rows,key=lambda r:r.entry.start))
    for i,r in enumerate(program):
        if not today<=r.day<=last: continue
        for delta in (-1,1):
            day=r.day+timedelta(days=delta)
            if today<=day<=last and day not in occupied and day not in blocked:
                yield normalized((*program[:i],r.on(day),*program[i+1:]))
        if r.role=='easy':
            yield program[:i]+program[i+1:]
            for option in easy_options:
                if abs(option.miles-r.miles)>1e-8:
                    yield normalized((*program[:i],option.on(r.day,key=r.entry.key),*program[i+1:]))
        # Exchange nearby occupied dates: a long/quality should not need an
        # empty destination just to explore different relative ordering.
        for j in range(i+1,len(program)):
            other=program[j]
            if other.day>last or (other.day-r.day).days>2: break
            if other.role!=r.role:
                rows=list(program); rows[i]=r.on(other.day); rows[j]=other.on(r.day)
                yield normalized(rows)
    for day in (today+timedelta(days=i) for i in range(edit_days)):
        if day in occupied or day in blocked: continue
        serial=0
        while (key:=f'added-easy/{day}/{serial}') in used_keys: serial+=1
        for option in easy_options:
            yield normalized((*program,option.on(day,key=key)))


def optimize(history, opening, today, easy_options, origin, end, targets, ordinary, policy,
             *, budget=2400, blocked=frozenset(), recovery_enabled=True, edit_days=21):
    started=perf_counter(); cache={}
    if budget<1 or edit_days not in (21,28): raise ValueError('Positive budget and declared edit window required')
    if any(r.day in blocked for r in opening): raise ValueError('Opening candidate conflicts with reserved rest')
    def evaluate(program):
        key=signature(program)
        if key in cache: return cache[key]
        if len(cache)>=budget: return None
        result=assess(history,program,origin,end,targets,ordinary,policy,recovery_enabled=recovery_enabled)
        cache[key]=result
        return result
    current=tuple(sorted(opening,key=lambda r:r.entry.start)); current_score=evaluate(current)
    path=[]; complete=False
    while True:
        best=current; best_score=current_score; complete=True
        for candidate in neighbors(current,today,easy_options,blocked=blocked,edit_days=edit_days,
                                   reserved_keys=frozenset(r.entry.key for r in history)):
            value=evaluate(candidate)
            if value is None: complete=False; break
            # Exact ties retain the incumbent; no continuity fee or frozen dates.
            a,b=value['rank'],best_score['rank']
            improves=a[0]<b[0]-1e-9 or (abs(a[0]-b[0])<=1e-9 and a[1]<b[1]-1e-9)
            if improves: best,best_score=candidate,value
        if signature(best)==signature(current): break
        current,current_score=best,best_score
        path.append(current_score)
        if not complete: break
    return current,dict(score=current_score,evaluations=len(cache),iterations=len(path),
        neighborhood_complete=complete,compatible=current_score['recovery_excess']<=1e-8,
        seconds=perf_counter()-started,path=path)


def assess(history,program,origin,end,targets,ordinary,policy,*,recovery_enabled=True):
    costs=volume_and_cadence(history,program,origin,end,targets,ordinary)
    paths=[recovery_path(history,program,policy,phase) for phase in (0,-1,1)] if recovery_enabled else []
    excess=max((p['total_excess'] for p in paths),default=0.)
    # Do not buy a contradicted prescription with additional volume points.
    # If no compatible candidate is found, return failure, not a rest order.
    return dict(recovery_excess=excess,**costs,objective=costs['volume']+costs['cadence'],
                rank=(0. if excess<=1e-8 else excess,costs['volume']+costs['cadence']))


def backward_compare(previous,history,later,origin,end,targets,ordinary,policy,edit_days=21):
    old_history,old_program,old_today=previous
    old_keys={r.entry.key for r in old_history}
    # Earlier counterfactuals must use earlier expected completion, not inject
    # the later observed above-range run into the earlier evidence frame.
    old_by_key={r.entry.key:r for r in old_program}
    completed=[r for r in history if r.entry.key not in old_keys]
    if any(r.entry.key not in old_by_key for r in completed):
        raise ValueError('Unscheduled work needs a distinct backward-audit contract')
    prefix=[old_by_key[r.entry.key] for r in completed]
    alternative=tuple(sorted((*prefix,*later),key=lambda r:r.entry.start))
    before=assess(old_history,old_program,origin,end,targets,ordinary,policy)
    counterfactual=assess(old_history,alternative,origin,end,targets,ordinary,policy)
    changes=churn(old_program,alternative,old_today)
    old_days={r.day:(r.role,round(r.miles,1)) for r in old_program}
    new_days={r.day:(r.role,round(r.miles,1)) for r in alternative}
    outside=[str(d) for d in old_days.keys()|new_days.keys()
             if old_days.get(d)!=new_days.get(d) and d>=old_today+timedelta(days=edit_days)]
    done={r.entry.key for r in history}
    current_incumbent=tuple(r for r in old_program if r.entry.key not in done)
    return dict(prior_incumbent=before,later_program_under_prior_inputs=counterfactual,
                changed_dates_outside_prior_edit_window=sorted(outside),prior_visible_changes=changes,
                incumbent_under_current_inputs=assess(history,current_incumbent,origin,end,targets,ordinary,policy),
                later_under_current_inputs=assess(history,later,origin,end,targets,ordinary,policy))


def metrics(program, today, days=21):
    rows=[r for r in program if today<=r.day<today+timedelta(days=days)]
    dates=[r.day for r in rows]; occupied=set(dates)
    streak=longest=0
    for i in range(days):
        streak=streak+1 if today+timedelta(days=i) in occupied else 0
        longest=max(streak,longest)
    def gaps(role):
        ds=[r.day for r in rows if key_role(r.role)==role]
        return [(b-a).days for a,b in zip(ds,ds[1:])]
    return dict(runs=len(rows),miles=round(sum(r.miles for r in rows),2),
        roles=dict(Counter(key_role(r.role) for r in rows)),
        longest_start_gap_days=max(((b-a).days for a,b in zip(dates,dates[1:])),default=0),
        longest_running_streak=longest,long_gaps=gaps('long'),quality_gaps=gaps('quality'),
        sessions=[dict(date=str(r.day),role=r.role,miles=round(r.miles,1)) for r in rows])


def churn(before, after, today):
    def visible(program):
        return {str(r.day):(r.role,round(r.miles,1)) for r in program if today<=r.day<today+timedelta(days=7)}
    a,b=visible(before),visible(after)
    return [dict(date=d,before=a.get(d),after=b.get(d)) for d in sorted(a.keys()|b.keys()) if a.get(d)!=b.get(d)]


def fixture(bank,journal):
    origin,days=shifted_program(journal)
    program=[]
    first_easy=next(rec for rec in runs(days) if rec.workout_type.value=='easy')
    easy_donor=choose_donor(bank,first_easy)
    for rec in runs(days):
        rec=rec.model_copy(update={'planned_for':rec.planned_for.replace(hour=12,minute=0,second=0)})
        donor=easy_donor if rec.workout_type.value=='easy' else choose_donor(bank,rec)
        program.append(Run(rec.workout_type.value,transformed_entry(donor,rec,source='projected')))
    # Common absolute closeout day 63, not a fresh far-edge truncation each day.
    # Repeat the final fourteen physical days solely as a fixed boundary.
    # Repeating offsets 21..34 after offset 48 would invent a 14-day quality gap.
    for r in tuple(program):
        if 35<=(r.day-origin.date()).days<49:
            program.append(r.on(r.day+timedelta(days=14),key=r.entry.key+'/closeout'))
    history=[]
    for day in range(-84,0):
        spec={1:('easy',3.75),3:('intervals',4.5),5:('long',7.5)}.get(day%7)
        if spec is None: continue
        role,miles=spec; at=origin+timedelta(days=day,hours=12)
        rec=RecommendationResponse(generated_at=at,fitness_state_as_of=at,planned_for=at,
            workout_type=role,title='Declared historical seed',distance_range_miles=(miles,miles),
            confidence='low',readiness='ready',structure=[WorkoutStep(instruction='6x1 fixture',phase='work',
                repetitions=6,work_duration_minutes=1)] if role=='intervals' else [])
        history.append(Run(role,transformed_entry(choose_donor(bank,rec),rec)))
    easy_options=tuple(next(r for r in program if r.role=='easy' and round(r.miles,1)==m) for m in (3.3,3.8))
    with journal.open() as f: snap=json.loads(next(f))
    return origin,tuple(history),tuple(program),easy_options,snap['target_trajectory'],snap['ordinary_easy_midpoint_miles']


def replay(initial,policy,*,days=7,budget=2400,scenario='exact',edit_days=21):
    origin,history,program,easy_options,targets,ordinary=initial
    history=list(history); end=origin+timedelta(days=63); reports=[]; uploads=0; previous=None
    for day in range(days):
        today=origin.date()+timedelta(days=day)
        before=program
        program,info=optimize(history,program,today,easy_options,origin,end,targets,ordinary,policy,budget=budget,edit_days=edit_days)
        reports.append(dict(trigger='daily',date=str(today),**info,metrics=metrics(program,today),
                            churn=churn(before,program,today)))
        if previous and reports[-1]['churn']:
            reports[-1]['backward_audit']=backward_compare(previous,history,program,origin,end,targets,ordinary,policy,edit_days)
        previous=(tuple(history),program,today)
        print(policy.name,scenario,today,'daily',info['evaluations'],round(info['seconds'],2),
              'compatible',info['compatible'],'complete',info['neighborhood_complete'],flush=True)
        if not info['compatible'] or not info['neighborhood_complete']:
            return dict(policy=asdict(policy),scenario=scenario,stopped='no compatible locally converged plan',decisions=reports)
        today_run=next((r for r in program if r.day==today),None)
        if today_run is None: continue
        expected=today_run
        actual=replace(expected,entry=replace(expected.entry,workload=replace(expected.entry.workload,source='observed')))
        if scenario=='above_range' and uploads==0:
            from scripts.audit_hr_recovery_programs import scaled_entry
            actual=replace(actual,entry=scaled_entry(actual.entry,1.25))
        remaining=tuple(r for r in program if r.entry.key!=expected.entry.key)
        # A full scorer must produce identical values for conditional exact
        # completion and the same observed completion, independently of search.
        expected_cost=volume_and_cadence([*history,expected],remaining,origin,end,targets,ordinary)
        actual_cost=volume_and_cadence([*history,actual],remaining,origin,end,targets,ordinary)
        expected_recovery=[recovery_path([*history,expected],remaining,policy,p) for p in (0,-1,1)]
        actual_recovery=[recovery_path([*history,actual],remaining,policy,p) for p in (0,-1,1)]
        if scenario=='exact' and (expected_cost!=actual_cost or expected_recovery!=actual_recovery):
            raise AssertionError('Exact completion changed conditional accounting')
        history.append(actual); uploads+=1
        before=remaining
        program,info=optimize(history,remaining,today+timedelta(days=1),easy_options,origin,end,
            targets,ordinary,policy,budget=budget,edit_days=edit_days)
        reports.append(dict(trigger='upload',date=str(today),**info,metrics=metrics(program,today+timedelta(days=1)),
            churn=churn(before,program,today+timedelta(days=1)),
            expected_miles=expected.miles,actual_miles=actual.miles,exact_accounting=expected_cost==actual_cost
            and expected_recovery==actual_recovery))
        if reports[-1]['churn']:
            reports[-1]['backward_audit']=backward_compare(previous,history,program,origin,end,targets,ordinary,policy,edit_days)
        previous=(tuple(history),program,today+timedelta(days=1))
        print(policy.name,scenario,today,'upload',info['evaluations'],round(info['seconds'],2),
              'churn',len(reports[-1]['churn']),flush=True)
        if not info['compatible'] or not info['neighborhood_complete']:
            return dict(policy=asdict(policy),scenario=scenario,stopped='upload found no compatible locally converged plan',decisions=reports)
    return dict(policy=asdict(policy),scenario=scenario,stopped=None,decisions=reports,uploads=uploads,
        realized=metrics(tuple(history),origin.date(),days),
        near_changes_after_initial=sum(len(r['churn']) for r in reports[1:]))


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--journal',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--days',type=int,default=7,choices=range(1,8))
    parser.add_argument('--budget',type=int,default=2400)
    parser.add_argument('--edit-days',type=int,choices=[21,28],default=21)
    parser.add_argument('--scenarios',nargs='+',choices=['exact','above_range'],default=['exact'])
    args=parser.parse_args()
    if args.output.exists(): parser.error('Choose a new output path')
    started=perf_counter(); config=load_config(Path('config.yaml'))
    bank=load_bank(Path(config['paths']['database']),config); before=digest(bank)
    initial=fixture(bank,args.journal)
    policies=(AssessmentPolicy('duration_hr_h36',160,320,36,36),
              AssessmentPolicy('matched_plus_work_h36',160,320,36,36,24,36),
              AssessmentPolicy('shorter_decay_plus_work_h24',130,280,24,24,24,24))
    results=[]
    for policy in policies:
        for scenario in args.scenarios:
            result=replay(initial,policy,days=args.days,budget=args.budget,scenario=scenario,edit_days=args.edit_days)
            results.append(result)
            # A failed lane is reported, not concealed by running many more days.
            if result['stopped']: break
    if digest(bank)!=before: raise AssertionError('Original donor bank changed')
    origin,history,opening,options,targets,ordinary=initial
    unguarded,ablation=optimize(history,opening,origin.date(),options,origin,origin+timedelta(days=63),
        targets,ordinary,policies[0],budget=args.budget,recovery_enabled=False,edit_days=args.edit_days)
    report=dict(scope=__doc__,results=results,seconds=perf_counter()-started,live=False,edit_days=args.edit_days,
        source_bank_digest=before,source_bank_count=len(bank),
        recovery_disabled_control=dict(search=ablation,metrics=metrics(unguarded,origin.date()),
            reassessment={p.name:assess(history,unguarded,origin,origin+timedelta(days=63),targets,ordinary,p)
                          for p in policies}),
        assumptions=['Declared policy hypotheses, not fitted or selected live defaults',
            'Recorded-HR transformed copies, not new physiological observations; original activities unchanged',
            'Normal health/no weather changes/no reserved dates in this fixture',
            f'Rolling {args.edit_days}-day editable window; hard prescription 21 days; fixed common absolute 63-day evaluation endpoint',
            'Last target corridor and closeout template held fixed, not a capacity-growth model',
            'Fixed long/quality tokens; easy support menu 3.3/3.8; no arbitrary streak penalty',
            'Recovery compatibility across three timing paths; target-volume and cadence rank compatible candidates',
            'Local neighborhood convergence only; no global optimality or production-planner stability claim'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f: json.dump(plain(report),f,indent=2,allow_nan=False)
    print('Total seconds',report['seconds'],flush=True)


if __name__=='__main__': main()
