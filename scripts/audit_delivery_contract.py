"""Independent delivered-work/reference-drift audit of real daily replay reports.

No optimizer scores, smoothing constants, future mileage credit or new pass/fail
training thresholds. 14/28-day windows are reporting spans, not calendar weeks
or scheduling quotas. Findings require interpretation of adherence and evidence.
"""
from argparse import ArgumentParser
from datetime import timedelta
import json
from math import isfinite
from pathlib import Path

from scripts.audit_realized_training import at, _bounds, _day


def audit(report):
    ledger=report['realized_execution']
    start,until=at(ledger['start_at']),at(ledger['observed_until'])
    if until<start:
        raise ValueError('Observation ends before it starts')
    actual=sorted(ledger['sessions'],key=lambda s:at(s['occurred_at']))
    if any(not start<=at(s['occurred_at'])<=until or not isfinite(s['distance_miles'])
           or s['distance_miles']<0 for s in actual):
        raise ValueError('Actual ledger contains out-of-observation or invalid work')
    snapshots=sorted(report['replans'],key=lambda s:at(s['generated_at']))
    if any(not start<=at(s['generated_at'])<=until for s in snapshots):
        raise ValueError('Decision lies outside the observed replay')
    first_day=_day(start)+(timedelta(days=1) if start!=_day(start) else timedelta())
    reference={}
    for snap in snapshots:
        generated=at(snap['generated_at'])
        decision=at(snap.get('decision_start_date') or generated.date().isoformat()).replace(tzinfo=generated.tzinfo)
        for offset,limits in enumerate(snap.get('target_trajectory') or ()):
            when=decision+timedelta(days=offset)
            if when<first_day or generated>when:
                # Do not rewrite a day's reference after observing that day.
                continue
            low,high=limits
            if not all(isfinite(x) and x>=0 for x in limits) or low>high:
                raise ValueError('Invalid target trajectory')
            key=when.date().isoformat()
            item=reference.setdefault(key,dict(first=limits,first_issued_at=snap['generated_at']))
            item.update(latest=limits,latest_issued_at=snap['generated_at'],
                latest_material_evidence=list(snap.get('material_evidence_reasons') or ()))
    daily=[]
    day=first_day
    while day+timedelta(days=1)<=until:
        end=day+timedelta(days=1)
        completed=[s for s in actual if day<=at(s['occurred_at'])<end]
        daily.append(dict(date=day.date().isoformat(),miles=sum(s['distance_miles'] for s in completed),
            runs=len(completed),reference=reference.get(day.date().isoformat())))
        day=end
    opening_capacity=snapshots[0].get('capacity_reference_miles') if snapshots else None
    history=[s for s in ledger['historical_sessions']
             if start-timedelta(days=28)<=at(s['occurred_at'])<start]
    history_rate=sum(s['distance_miles'] for s in history)/4
    reference_series=[dict(at=s['generated_at'],capacity_reference_miles=s.get('capacity_reference_miles'),
        target_low_miles=s.get('target_low_miles'),target_high_miles=s.get('target_high_miles'),
        material_evidence=list(s.get('material_evidence_reasons') or ())) for s in snapshots]
    windows=[]
    for length in (14,28):
        for index in range(length,len(daily)+1):
            days=daily[index-length:index]
            complete=all(d['reference'] is not None for d in days)
            delivered=sum(d['miles'] for d in days)
            original=(sum(d['reference']['first'][0] for d in days)/7 if complete else None)
            latest=(sum(d['reference']['latest'][0] for d in days)/7 if complete else None)
            evidence=[dict(date=d['date'],reasons=d['reference']['latest_material_evidence'])
                      for d in days if d['reference'] and d['reference']['latest_material_evidence']]
            windows.append(dict(days=length,from_date=days[0]['date'],through_date=days[-1]['date'],
                actual_miles=delivered,actual_runs=sum(d['runs'] for d in days),
                actual_weekly_equivalent=delivered*7/length,reference_complete=complete,
                first_issued_integrated_low=original,latest_pre_day_integrated_low=latest,
                reference_reduction_miles=original-latest if complete else None,
                shortfall_to_first_issued=max(0.,original-delivered) if complete else None,
                shortfall_to_latest=max(0.,latest-delivered) if complete else None,
                latest_target_masks_original_shortfall=(complete and latest<=delivered+1e-9<original),
                no_work_despite_positive_original_target=(complete and original>0 and delivered==0),
                opening_capacity_weekly_miles=opening_capacity,
                opening_history_weekly_miles=history_rate,material_evidence=evidence))
    def next_run(snap):
        return min((at(s['planned_for']) for s in snap['planned_sessions']
                    if at(s['planned_for'])>=_bounds(snap)),default=None)
    deferrals=[]
    streak=0
    for previous,current in zip(snapshots,snapshots[1:]):
        before,after=next_run(previous),next_run(current)
        executed=any(at(previous['generated_at'])<at(s['occurred_at'])<=at(current['generated_at']) for s in actual)
        if before and after and after.date()>before.date() and not executed:
            streak+=1
            deferrals.append(dict(at=current['generated_at'],previous_next=before.isoformat(),
                revised_next=after.isoformat(),consecutive_observed_deferrals=streak,
                material_evidence=list(current.get('material_evidence_reasons') or ()),
                attribution='unresolved: planner revision versus missed/changed execution'))
        elif executed or not before or not after or after.date()<before.date():
            streak=0
    # Clock changes alone are not run-date deferrals. Record coverage rather
    # than trusting a filename or a requested simulation duration.
    decision_dates={at(s['generated_at']).date().isoformat() for s in snapshots
                    if s.get('trigger','scheduled_refresh')=='scheduled_refresh'}
    missing=[d['date'] for d in daily if d['date'] not in decision_dates]
    findings=[]
    for row in windows:
        for key in ('latest_target_masks_original_shortfall','no_work_despite_positive_original_target'):
            if row[key]:
                findings.append(dict(code=key,from_date=row['from_date'],through_date=row['through_date'],
                    days=row['days'],material_evidence=row['material_evidence']))
    if deferrals:
        findings.append(dict(code='next_run_date_deferred_without_intervening_execution',
            count=len(deferrals),longest_observed_chain=max(r['consecutive_observed_deferrals'] for r in deferrals)))
    return dict(scope=__doc__,version='independent_delivery_v1',
        status=('insufficient_observation' if not windows else 'observational_findings'
                if findings else 'observed_without_listed_findings'),
        interpretation='Not release clearance; overlapping windows are not independent, and material evidence may justify adaptations.',
        complete=ledger['complete'],observed_until=ledger['observed_until'],
        complete_observation_days=len(daily),matured_14_day_windows=sum(w['days']==14 for w in windows),
        matured_28_day_windows=sum(w['days']==28 for w in windows),
        load_reference_coverage_complete=bool(windows) and all(w['reference_complete'] for w in windows),
        daily_decision_coverage_complete=not missing,missing_daily_decision_dates=missing,
        post_upload_decisions=sum(s.get('trigger')=='post_upload' for s in snapshots),
        reference_series=reference_series,
        total_completed_miles=sum(s['distance_miles'] for s in actual),
        daily=daily,windows=windows,deferrals=deferrals,findings=findings,
        original_gate_status=report.get('status'),original_gate_failures=report.get('failures',[]))


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--output-json',type=Path,required=True)
    args=parser.parse_args()
    if args.output_json.exists():
        parser.error('Choose a new output path')
    result=audit(json.loads(args.report.read_text()))
    with args.output_json.open('x') as stream:
        json.dump(result,stream,indent=2)
        stream.write('\n')
    print(json.dumps({k:result[k] for k in ('status','complete_observation_days',
        'matured_14_day_windows','matured_28_day_windows','findings')}))


if __name__=='__main__':
    main()
