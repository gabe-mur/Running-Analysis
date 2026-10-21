"""Read-only replay diagnostics: issued versus delivered, churn and key entry.

This does not search, alter model weights, or infer physiological clearance.
Every saved winner must reproduce before its normalized demand trace is reported.
"""
from argparse import ArgumentParser
from datetime import date, datetime, time, timedelta
import json
from pathlib import Path
from statistics import mean, median
from zoneinfo import ZoneInfo

from scripts.replay_adaptive_search import load_decision
from scripts.check_explicit_distribution_tail import incoming_demand_trace
from scripts.research_manifest import freeze_sources
from run_analysis.config import load_config
from run_analysis.adaptive.decision_snapshot import evaluate_snapshot
from run_analysis.adaptive.new_model_daily_harness import planned_accounted
from run_analysis.continuous_training_dose import ContinuousDoseReference


def edit_counts(events,window):
    edits=[dict(event=e['index'],as_of=e['as_of'],reason=e['reason'],**r)
        for e in events for r in e['churn'][str(window)]]
    return dict(edited_slots=len(edits),date_edits=sum(r['date_changed'] for r in edits),
        type_edits=sum(r['role_changed'] for r in edits),
        dose_edits=sum(r['distance_change'] is not None for r in edits),
        largest_absolute_dose_edit=max((abs(r['distance_change']) for r in edits
            if r['distance_change'] is not None),default=0.),edits=edits)


def quality_edit_counts(events,catalogs,window):
    """A quality→quality family swap is NOT merely unchanged run role.

    Reconstruct from archived native definitions; no guessed family from keys.
    Keep recipe/structure changes separate from family and mileage changes.
    """
    rows=[]
    unknown=0
    for e in events:
        catalog=catalogs[e['index']]
        for edit in e['churn'][str(window)]:
            a,b=(catalog.get(edit[k]) for k in ('before','after'))
            if not a or not b or a.role!='quality' or b.role!='quality':continue
            left,right=a.workout_definition or {},b.workout_definition or {}
            if not left.get('family') or not right.get('family'):
                unknown+=1;continue
            family=left['family']!=right['family']
            recipe=left.get('recipe_key')!=right.get('recipe_key')
            structure=left.get('structure')!=right.get('structure')
            if family or recipe or structure:
                rows.append(dict(event=e['index'],as_of=e['as_of'],day=edit['day'],
                    before_family=left['family'],after_family=right['family'],
                    before_recipe=left.get('recipe_key'),after_recipe=right.get('recipe_key'),
                    family_changed=family,recipe_changed=recipe,structure_changed=structure,
                    distance_change=edit['distance_change']))
    return dict(family_edits=sum(r['family_changed'] for r in rows),
        recipe_edits=sum(r['recipe_changed'] for r in rows),
        structure_edits=sum(r['structure_changed'] for r in rows),
        unknown_family_comparisons=unknown,edits=rows,
        scope='Archived quality family/recipe/steps; includes raw tail-to-visible revisions unless explicitly filtered')


def visible_churn_event(events,catalogs,window):
    """Compare current UI dates only to their previously visible prescriptions.

    The UI starts at event.decision (also after upload), not at the wall date.
    Keep raw first-entry edits separately. Remember visible rest dates too, and
    never forget a promise just because a later window omitted its date.
    """
    previous={}
    for event in events[:-1]:
        start=date.fromisoformat(event['decision']); program=dict(event['program'])
        for offset in range(window):
            day=(start+timedelta(days=offset)).isoformat()
            previous[day]=(program.get(day),event['index'])
    current=events[-1]; start=date.fromisoformat(current['decision'])
    program=dict(current['program']); edits=[]
    for offset in range(window):
        day=(start+timedelta(days=offset)).isoformat()
        if day not in previous:continue
        a,index=previous[day]; b=program.get(day)
        if a==b:continue
        old=catalogs[index][a] if a is not None else None
        new=catalogs[current['index']][b] if b is not None else None
        edits.append(dict(day=day,before=a,after=b,date_changed=(a is None)!=(b is None),
            role_changed=bool(old and new and old.role!=new.role),
            distance_change=new.miles-old.miles if old and new else None))
    return dict(current,churn={str(window):edits},
        entering_window_edits=[r for r in current['churn'][str(window)] if r['day'] not in previous])


def visible_edit_counts(events,catalogs,window):
    filtered=[visible_churn_event(events[:n+1],catalogs,window) for n in range(len(events))]
    return dict(prescription_edits=edit_counts(filtered,window),
        quality_edits=quality_edit_counts(filtered,catalogs,window),
        entering_window_edits=[dict(event=e['index'],**r) for e in filtered for r in e['entering_window_edits']],
        scope='Only changes after actual UI visibility; first-entry tail revisions retained separately')


def visible_dose_history(events,catalogs,window):
    """First-visible promise to last pre-execution version, not just adjacent edits.

    Exposure starts when a date enters the visible window. Ordinary removal
    after completion/expiry is not a date edit; far-tail revisions do not count.
    This reports accumulated movement and native identity, without changing
    the driver's adjacent-edit distance threshold.
    """
    histories={}
    for event in events:
        start=date.fromisoformat(event['decision']); program=dict(event['program'])
        for offset in range(window):
            day=(start+timedelta(days=offset)).isoformat(); key=program.get(day)
            dose=catalogs[event['index']].get(key)
            value=dict(role=dose.role if dose else None,miles=dose.miles if dose else None)
            definition=(dose.workout_definition or {}) if dose else {}
            value.update(family=definition.get('family'),recipe=definition.get('recipe_key'),
                structure=definition.get('structure'))
            rows=histories.setdefault(day,[])
            if not rows or any(rows[-1][k]!=value[k] for k in value):
                rows.append(dict(event=event['index'],**value))
    changed=[]
    date_edits=type_edits=family_edits=recipe_edits=structure_edits=0
    for day,rows in sorted(histories.items()):
        if len(rows)<2:continue
        date_edits+=sum((a['role'] is None)!=(b['role'] is None) for a,b in zip(rows,rows[1:]))
        type_edits+=sum(a['role'] is not None and b['role'] is not None and a['role']!=b['role']
            for a,b in zip(rows,rows[1:]))
        for a,b in zip(rows,rows[1:]):
            if a['role']==b['role']=='quality':
                family_edits+=a['family']!=b['family']
                recipe_edits+=a['recipe']!=b['recipe']
                structure_edits+=a['structure']!=b['structure']
        first,last=rows[0],rows[-1]
        values=[r['miles'] for r in rows if r['miles'] is not None]
        movements=[abs(b['miles']-a['miles']) for a,b in zip(rows,rows[1:])
            if a['role']==b['role'] and a['miles'] is not None and b['miles'] is not None]
        changed.append(dict(day=day,versions=rows,
            net_dose_edit=last['miles']-first['miles'] if first['role']==last['role']
                and first['miles'] is not None and last['miles'] is not None else None,
            dose_span=max(values)-min(values) if values else 0.,total_dose_movement=sum(movements)))
    return dict(scope='First visible prescription to last pre-execution version; dose adaptation is not automatically a bug',
        date_edits=date_edits,type_edits=type_edits,family_edits=family_edits,
        recipe_edits=recipe_edits,structure_edits=structure_edits,
        largest_net_dose_edit=max((abs(r['net_dose_edit']) for r in changed if r['net_dose_edit'] is not None),default=0.),
        largest_dose_span=max((r['dose_span'] for r in changed),default=0.),
        total_dose_movement=sum(r['total_dose_movement'] for r in changed),changed_dates=changed)


def target_integral(segments,start,end):
    return sum(max(0,(min(end,r.end)-max(start,r.start)).days)*r.miles_per_7_days/7 for r in segments)


def cardio_comparison(comparison):
    actual=comparison['actual_cardio']; predicted=comparison['target_load']
    return dict(forecast_cardio=predicted,actual_cardio=actual['estimate'],
        actual_cardio_bounds=[actual['low'],actual['high']],
        cardio_difference_percent=100*(actual['estimate']/predicted-1) if actual['estimate'] is not None else None,
        cardio_difference_percent_bounds=[100*(actual[k]/predicted-1) for k in ('low','high')])


def replay_report(root, report_file=None):
    if report_file is not None:
        if Path(report_file).name != report_file or not report_file.endswith('.json'):
            raise ValueError('Report must be a JSON filename inside the replay root')
        path=root/report_file
    else:
        reports=sorted(root.glob('resume-*.summary.json'))
        path=reports[-1] if reports else root/'lifecycle-summary.json'
    report=json.loads(path.read_text())
    # A saved checkpoint is a partial observation, not a fabricated final pass.
    if 'requested_days' not in report:
        if report.get('status')!='checkpoint_no_fast_fail' or 'continuation_days' not in report:
            raise ValueError('Report lacks the requested replay duration')
        report=dict(report,requested_days=report['continuation_days'])
    return path,report


def audit(root, report_file=None):
    report_path,report=replay_report(root,report_file)
    events=report['events']; results={}; decisions=[]
    reference=ContinuousDoseReference('sandbox upload accounting')
    for event in events:
        case=load_decision(root,event['index']+1)
        state,catalog=case['state'],case['catalog']
        program=tuple(tuple(p) for p in event['program'])
        physical=tuple(planned_accounted(f'candidate:{day}:{key}',
            datetime.combine(date.fromisoformat(day),time(18),ZoneInfo(state.timezone)),
            catalog[key],case['handoffs'][key],reference,available=state.as_of) for day,key in program)
        checked=evaluate_snapshot(state,physical,tuple(k for _,k in program),reference=reference)
        e=checked['evaluation']
        assert abs(e.total_cost-event['cost'])<1e-12 and checked['decision_eligible']
        trace=incoming_demand_trace(e,state.context)
        results[event['index']]=(case,trace)
        target=target_integral(state.context.volume_targets,state.context.decision,state.context.hard_end)
        decisions.append(dict(index=event['index'],as_of=event['as_of'],reason=event['reason'],
            projection=dict(decision=state.context.decision.isoformat(),
                hard_end=state.context.hard_end.isoformat(),end=state.context.end.isoformat(),
                hard_days=(state.context.hard_end-state.context.decision).days,
                selectable_tail_days=(state.context.end-state.context.hard_end).days),
            model_version=event['model_version'],score=e.total_cost,fatigue=e.weighted_demand_cost,
            delivery=e.delivery_cost,short=e.short_run_cost,miles21=event['volume'],
            dated_target_integral21=target,funding_ratio=event['volume']/target,
            search_seconds=event['seconds'],evaluations=event['evaluations'],
            budget_exhausted=event['budget_exhausted'],
            local_complete=event['local_neighborhood_complete'],
            eligibility=checked['decision_eligible'],suitability=checked['suitability']['status'],
            planned_training=event['planned_training'],key_entry=[r for r in trace if r['role']!='easy']))
    executed=[]
    for upload in report['uploads']:
        day=upload['day']; comparison=upload['comparison']
        owning=[e for e in events if day in e['active'] and
            e['active'][day]['fingerprint']==comparison['issued_fingerprint']]
        owner=owning[-1]
        case,trace=results[owner['index']]
        row=next(r for r in trace if r['day']==day)
        dose=case['catalog'][owner['active'][day]['dose']]
        definition=dose.workout_definition or {}
        executed.append(dict(day=day,role=dose.role,issued_miles=dose.miles,
            quality_family=definition.get('family'),quality_recipe=definition.get('recipe_key'),
            quality_variant=definition.get('variant'),
            prescribed_work_minutes=sum(s for k,s in dose.phases if k=='work')/60,
            actual_miles=comparison['actual_distance_miles'],
            distance_adherence=comparison['distance_adherence'],structure=comparison['structure'],
            actual_minutes=comparison['actual_elapsed_seconds']/60,**cardio_comparison(comparison),
            issuing_decision=owner['index'],entry=row,provenance=upload['provenance']))
    first=results[events[0]['index']][0]['state']; last=results[events[-1]['index']][0]['state']
    start=(date.fromisoformat(report['delivery_start']) if report.get('delivery_start') else
        datetime.fromisoformat(json.loads((root/'scenario.json').read_text())['origin']).astimezone(
        ZoneInfo(first.timezone)).date())
    end=start+timedelta(days=report['delivery_window_days'])
    target=target_integral(last.context.volume_targets,start,end)
    searched=[e['seconds'] for e in events[1:]] if report.get('initial_search_reused') else [e['seconds'] for e in events]
    rolling=report['delivered']['rolling_7']
    return dict(status=report['status'],report_file=report_path.name,
        completed_days=report['completed_days'],requested_days=report['requested_days'],
        stability_contract=report.get('stability_contract','archived raw-window gate; visible history recomputed read-only'),
        failures=report['failures'],model_versions=sorted({e['model_version'] for e in events}),
        decisions=len(events),new_searches=len(searched),
        search_seconds=dict(total=sum(searched),minimum=min(searched,default=None),
            maximum=max(searched,default=None),median=median(searched) if searched else None),
        churn={str(w):edit_counts(events,w) for w in (4,7)},
        quality_churn={str(w):quality_edit_counts(events,
            {i:case['catalog'] for i,(case,_) in results.items()},w) for w in (4,7)},
        previously_visible_churn={str(w):visible_edit_counts(events,
            {i:case['catalog'] for i,(case,_) in results.items()},w) for w in (4,7)},
        visible_prescription_history={str(w):visible_dose_history(events,
            {i:case['catalog'] for i,(case,_) in results.items()},w) for w in (4,7)},
        delivered=report['delivered'],delivered_target_integral=target,
        delivered_funding_ratio=report['delivered']['miles']/target,
        mean_daily_rolling_7_miles=mean(r['miles'] for r in rolling),
        mean_daily_rolling_7_runs=mean(r['runs'] for r in rolling),
        initial_stage=events[0].get('stage_runtime',{}).get('stage'),final_stage=report['final_stage'],
        stage_reviews=report['stage_reviews'],executed=executed,decision_diagnostics=decisions,
        driver_recovery=report.get('recovery'),
        acceptance_scope='Daily AND post-upload reoptimization, distance/structure adherence with transformed recorded HR; not identical forecast/actual physiology or full live-route validation',
        demand_interpretation='Normalized modeled duration/cardio/work carryover, not fatigue percentages or performance predictions',
        score_replay_verified=True,live_database_modified=False)


if __name__=='__main__':
    p=ArgumentParser(); p.add_argument('root',type=Path)
    p.add_argument('--label',help='Preserve separate audits of a resumed checkpoint, e.g. day7')
    p.add_argument('--report-file',help='Explicit saved checkpoint JSON filename inside the root; never changes its status')
    a=p.parse_args()
    if a.label is not None and (not a.label or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in a.label)):
        p.error('Audit label must contain lowercase letters, numbers, hyphens or underscores')
    stem='daily-audit'+('-'+a.label if a.label else '')
    if a.report_file is not None and a.label is None:
        p.error('Checkpoint audits need a separate --label to preserve the final audit')
    result=audit(a.root,a.report_file)
    result['audit_manifest']=freeze_sources(a.root/(stem+'.sources.zip' if a.label else 'audit.sources.zip'),config=load_config(),
        entrypoint=__name__,inputs=tuple(a.root.glob('decision-*.json'))+tuple(a.root.glob('snapshot-*.json'))+
        tuple(a.root.glob('*summary.json'))+tuple(a.root.glob('execution-*.json'))+
        tuple(a.root.glob('inherited-execution-*.json'))+tuple(a.root.glob('replay-inherited-completions.json'))+
        (a.root/result['report_file'],a.root/'scenario.json'))
    with (a.root/(stem+'.json')).open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ('status','completed_days','requested_days','decisions','new_searches',
        'search_seconds','delivered_target_integral','delivered_funding_ratio','mean_daily_rolling_7_miles')}))
