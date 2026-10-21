"""Read-only, fixed-issued-course distance counterfactual; no planner search.

Only original published events are used as historical target knowledge. Actual
seed history stays actual. Hypothetical nominal completions are separate
Exposure values, never AccountedRuns, database uploads or new physical/HR data.
This does not replay the different decisions a truly perfect runner would get.
"""
from argparse import ArgumentParser
from dataclasses import dataclass
from datetime import datetime, timedelta, time
from pathlib import Path
from zoneinfo import ZoneInfo
import json
import sqlite3

from run_analysis.adaptive.historical_progression import Exposure, observations, support
from run_analysis.adaptive.program_evidence_adapter import read_accounted
from run_analysis.adaptive.smoothed_progression import growth
from run_analysis.stage_target_history import target_segments, target_miles_between
from scripts.trial_smoothed_progression import target
from scripts.research_manifest import freeze_sources


@dataclass(frozen=True)
class IssuedRun:
    key: str
    day: str
    fingerprint: str
    intent: dict
    role: str
    actual: Exposure | None


def issue_event(events, at):
    """Latest published knowledge at original issuance, never a later target."""
    known=[e for e in events if datetime.fromisoformat(e['as_of'])<=at]
    if not known:
        raise ValueError('Original target publication unavailable at issuance')
    return max(known,key=lambda e:datetime.fromisoformat(e['as_of']))


def nominal(run, mode):
    """Distance-only hypothetical with explicit clock/availability assumptions."""
    issued=datetime.fromisoformat(run.intent['issued_at'])
    if mode=='recorded_clock' and run.actual is not None:
        start,end,available=run.actual.start,run.actual.end,run.actual.available_at
        if start<issued:
            raise ValueError('Nominal distance was not known before the actual run')
    else:
        # A late reissue must never authorize a completion BEFORE issuance.
        start=max(issued,datetime.fromisoformat(run.intent['suggested_start']))
        duration=sum(p['seconds']['target'] for p in run.intent['phases'])
        if duration<=0:
            raise ValueError('Positive original nominal duration required')
        end=start+timedelta(seconds=duration)
        available=end  # Explicit counterfactual: immediate upload after finish.
    return Exposure(run.key,start,end,available,run.intent['distance_miles']['target'],run.role,
        'counterfactual_nominal_distance:'+mode)


def course(history, runs, as_of, mode):
    """Preserve unmodified seeds; do not double-count the matched seven runs."""
    if mode not in {'actual','recorded_clock','scheduled_clock'}:
        raise ValueError('Known actual or declared hypothetical course required')
    changed={r.key for r in runs if r.actual is not None}
    records=[r for r in history if r.key not in changed]
    for r in runs:
        value=r.actual if mode=='actual' else nominal(r,mode)
        if value is not None:
            records.append(value)
    records=tuple(sorted((r for r in records if r.available_at<=as_of and r.end<=as_of),
        key=lambda r:(r.start,r.key)))
    if len({r.key for r in records})!=len(records):
        raise ValueError('Counterfactual counted a physical identity twice')
    return records


def evidence_summary(records, targets, at, zone, levels):
    e=support(records,(),targets,as_of=at,zone=zone)
    return dict(actual_per_7=e['actual_per_7'],dated_target_per_7=e['target_per_7'],
        matched_delivery_ratio=e['delivery_ratio'],actual_count=len(records),
        growth_at_levels={str(level):growth(level,e) for level in levels})


def target_volume(records, at, start, end, zone):
    segments,_=target_segments(records,as_of=at,start=start.date(),end=end.date())
    return target_miles_between(segments,start,end,zone)


def load(source):
    scenario=json.loads((source/'scenario.json').read_text())
    all_events=[json.loads(p.read_text()) for p in sorted(source.glob('decision-*.json'))]
    events=[]
    for e in all_events:
        if e['reason']=='continuous_general_review' or e['reason'].startswith('offline_'):
            break
        events.append(e)
    if not events:
        raise ValueError('Original published course is required')
    published={x['fingerprint'] for e in events for x in e['active'].values()}
    db=sqlite3.connect((source/'sandbox.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
    db.row_factory=sqlite3.Row
    try:
        raw=[read_accounted(r) for r in scenario['history']]
        completed=[]
        issues={r['fingerprint']:json.loads(r['serialized'])['intent']
            for r in db.execute('SELECT fingerprint,serialized FROM sandbox_issues')
            if r['fingerprint'] in published}
        for row in db.execute('SELECT fingerprint,accounted FROM sandbox_completions ORDER BY available_at'):
            actual=read_accounted(json.loads(row['accounted']))
            if row['fingerprint'] not in issues:
                raise ValueError('A measured upload is not matched to original issuance')
            intent=issues[row['fingerprint']]
            completed.append((row['fingerprint'],intent,actual))
            raw.append(actual)
        zone=ZoneInfo(json.loads((source/'execution-sandbox.json').read_text())['timezone'])
        as_of=datetime.fromisoformat(json.loads((source.parent/'opening.json').read_text())['support']['as_of'])
        excluded={k for k,v in scenario['capacity_eligibility'].items() if not v}
        actuals=observations(raw,(),as_of,excluded_keys=excluded)
        by_key={r.key:r for r in actuals}
        runs=[IssuedRun(a.key,intent['session_key'].removeprefix('slot:'),f,intent,a.role,by_key[a.key])
            for f,intent,a in completed if a.key in by_key]
        # The original slot moved from Oct 17 to Oct 18. Use the contract known
        # BEFORE its proposed start, not the 19:43 reissue with an 18:00 start.
        choices=[(f,i) for f,i in issues.items() if i['session_key']=='slot:2026-10-18'
            and datetime.fromisoformat(i['issued_at'])<=datetime.fromisoformat(i['suggested_start'])]
        f,intent=max(choices,key=lambda pair:datetime.fromisoformat(pair[1]['issued_at']))
        runs.append(IssuedRun('hypothetical:omitted:2026-10-18','2026-10-18',f,intent,'easy',None))
        return scenario,events,actuals,tuple(runs),zone,as_of,issues
    finally:
        db.close()


def frozen_publication(event, issues, history, targets, zone):
    """Follow ONE original full publication, not pieces of different rerolls.

    No future actual uploads or later target amendments enter this course.
    Conditional targets are held frozen; this is not a lifecycle simulation.
    """
    issued=datetime.fromisoformat(event['as_of'])
    targets=tuple(t for t in targets if t.available_at<=issued)
    seed=tuple(r for r in history if r.available_at<=issued and r.end<=issued)
    runs=[]
    for day,value in event['active'].items():
        if not event['decision']<=day<event['hard_end']:
            continue
        intent=issues[value['fingerprint']]
        role='long' if value['dose'].startswith('long:') else 'quality' if value['dose'].startswith('quality:') else 'easy'
        runs.append(IssuedRun('counterfactual:publication:'+str(event['index'])+':'+day,
            day,value['fingerprint'],intent,role,None))
    nominal_total=sum(r.intent['distance_miles']['target'] for r in runs)
    if abs(nominal_total-event['volume'])>1e-8:
        raise ValueError('Full published programme and original issued distances disagree')
    start=datetime.combine(datetime.fromisoformat(event['decision']).date(),time.min,zone)
    end=datetime.combine(datetime.fromisoformat(event['hard_end']).date(),time.min,zone)
    checkpoints=[issued]+[start+timedelta(days=n) for n in range(1,(end-start).days+1)]
    trace=[]
    for at in checkpoints:
        level=targets[-1].rate(at.astimezone(zone).date(),'volume_per_7')
        e=evidence_summary(course(seed,runs,at,'scheduled_clock'),targets,at,zone,(level,))
        trace.append(dict(at=at.isoformat(),conditional_level=level,**e))
    return dict(index=event['index'],published_at=event['as_of'],start=start.isoformat(),end=end.isoformat(),
        issued_miles=nominal_total,dated_target_miles=target_volume(targets,issued,start,end,zone),
        known_actual_seed_count=len(seed),hypothetical_runs=len(runs),trace=trace,
        limitation='One frozen publication followed exactly; no later actuals, target revisions or counterfactual rerolls')


def audit(source):
    scenario,events,history,runs,zone,at,issues=load(source)
    rows=[]
    for r in runs:
        issued=datetime.fromisoformat(r.intent['issued_at'])
        e=issue_event(events,issued)
        rows.append(dict(day=r.day,physical_key=r.key,fingerprint=r.fingerprint,
            issued_at=issued.isoformat(),publication=e['index'],
            target_at_issue=e['progression_target'],nominal_miles=r.intent['distance_miles']['target'],
            actual_miles=r.actual.miles if r.actual else None,
            suggested_start=r.intent['suggested_start']))
    final=events[-1]
    targets=tuple(map(target,final['stage_runtime']['target_history']))
    baseline=final['stage_runtime']['stage']['weekly_equivalent_miles']
    current=targets[-1].rate(at.astimezone(zone).date(),'volume_per_7')
    levels=tuple(sorted({baseline,current}))
    courses={mode:course(history,runs,at,mode) for mode in ('actual','recorded_clock','scheduled_clock')}
    results={mode:evidence_summary(records,targets,at,zone,levels) for mode,records in courses.items()}
    # Daily accounting samples are display/diagnostic only, not a review gate.
    trace=[]
    for e in events:
        when=datetime.fromisoformat(e['as_of'])
        runtime=e.get('stage_runtime',scenario)
        ts=tuple(map(target,runtime['target_history']))
        level=ts[-1].rate(datetime.fromisoformat(e['decision']).date(),'volume_per_7')
        trace.append(dict(index=e['index'],at=e['as_of'],target_mode=e['progression_target'],
            courses={mode:evidence_summary(course(history,runs,when,mode),ts,when,zone,(level,))
                for mode in courses}))
    start=datetime(2026,10,5,tzinfo=zone);end=datetime(2026,10,19,tzinfo=zone)
    actual=sum(r.actual.miles for r in runs if r.actual)
    nominal_total=sum(r.intent['distance_miles']['target'] for r in runs)
    required=target_volume(targets,at,start,end,zone)
    publications=[frozen_publication(e,issues,history,
        tuple(map(target,e.get('stage_runtime',scenario)['target_history'])),zone)
        for e in events if e['index'] in {0,7,29}]
    return dict(status='fixed_original_course_accounting_only',as_of=at.isoformat(),
        original_last_index=final['index'],original_prescriptions=rows,
        provisional_lead_in=[dict(effective_on=t.effective_on.isoformat(),available_at=t.available_at.isoformat(),
            source=t.source,volume_per_7=t.volume_per_7) for t in targets
            if t.effective_on<t.available_at.astimezone(zone).date()],
        unchanged_seed_keys=[r.key for r in history if r.key not in {x.key for x in runs}],
        original_stage_baseline=baseline,current_forecast_level=current,
        descriptive_interval=dict(start=start.isoformat(),end=end.isoformat(),
            actual_miles=actual,exact_prescribed_miles=nominal_total,dated_target_miles=required),
        courses=results,trace=trace,frozen_publications=publications,
        target_migration=dict(original_baseline=baseline,first_continuous_volume=current,
            basis='continuous_general_progression.propose initializes from latest dated path.rate(day), not original stage baseline'),
        limitations=[
            'Seven matched uploads and the original omitted Oct 18 opportunity; older eligible seed history stays actual, not assumed perfect.',
            'Original schedules and target versions frozen; no counterfactual replanning or new publication.',
            'Recorded-clock changes only nominal distance at actual timestamps/availability; scheduled-clock uses original nominal start and duration with immediate upload.',
            'Distance/volume component only; no hypothetical HR, quality delivery, fatigue or health clearance.',
            'Calendar totals and trace sampling are diagnostics, not a new progression window.',
            'The pre-adoption 15-mile lead-in is an explicitly provisional reference, not a prescription issued to the runner in August.',
        ],live_changed=False,planner_searches=0)


def main(source,output):
    report=audit(source)
    output.mkdir(parents=True,exist_ok=True)
    report['manifest']=freeze_sources(output/'sources.zip',config={},entrypoint=__name__,
        inputs=(source/'sandbox.sqlite',source/'scenario.json',source/'execution-sandbox.json',source.parent/'opening.json')
            +tuple(source.glob('decision-*.json')))
    with (output/'report.json').open('x') as stream:
        json.dump(report,stream,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k in {
        'status','as_of','original_stage_baseline','current_forecast_level','descriptive_interval','courses','limitations'}}),flush=True)


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    main(args.source,args.output)
