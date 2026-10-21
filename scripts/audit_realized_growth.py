"""Read-only signal probe on a frozen replay, not a new planner replay.

The no-growth reference is an EXPLICIT easy-first allocation approximation:
keep the opening's calendar/key recipes, remove its declared easy-volume ramp
from easy doses, and freeze the result. No later target or actual can reprice
it. This reference adapter needs allocation-level verification before release.
"""
from argparse import ArgumentParser
from dataclasses import replace
from datetime import datetime, timedelta
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from run_analysis.adaptive.adaptive_stage_lifecycle import review_entries
from run_analysis.adaptive.execution_upload_sandbox import ExecutionUploadSandbox
from run_analysis.adaptive.realized_growth import ReferenceDose, estimate, review_reference
from run_analysis.adaptive.prescription_progression import published_funding
from run_analysis.adaptive.smoothed_progression import advance, days_between
from run_analysis.adaptive.review_seed import read_entry
from scripts.research_manifest import freeze_sources
from scripts.trial_smoothed_progression import target
from run_analysis.workload_accounting import Quantity


def audit(root, output):
    output.mkdir(parents=True,exist_ok=True)
    source=root/'resume-0001.summary.json'
    summary=json.loads(source.read_text())
    events=[json.loads(p.read_text()) for p in sorted(root.glob('decision-*.json'))]
    opening=summary['events'][0]
    at_open=datetime.fromisoformat(opening['as_of'])
    scenario=json.loads((root/'scenario.json').read_text())
    zone=ZoneInfo(json.loads((root/'execution-sandbox.json').read_text())['timezone'])
    db=sqlite3.connect((root/'sandbox.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
    db.row_factory=sqlite3.Row
    sandbox=ExecutionUploadSandbox.__new__(ExecutionUploadSandbox); sandbox.db=db
    catalog={row['key']:SimpleNamespace(role=row['role']) for event in events
        for row in event.get('stage_runtime',scenario)['catalog']}
    path=target(opening['stage_runtime']['target_history'][-1])
    source_support=path.supporting_easy_per_7
    refs=[]; roles={}; totals={}; rows=[]
    try:
        # All reference values depend ONLY on this first immutable publication.
        for day,selected in opening['active'].items():
            intent=sandbox.record(selected['fingerprint']).payload['intent']
            role=catalog[selected['dose']].role
            scheduled=datetime.fromisoformat(intent['suggested_start'])
            base=intent['distance_miles']['target']
            if role=='easy':
                base*=source_support/path.rate(scheduled.astimezone(zone).date(),'supporting_easy_per_7')
            key='slot:'+day; roles[key]=role
            totals[role]=totals.get(role,0.)+base
            refs.append(ReferenceDose(key,scheduled,at_open,base,'opening-no-growth:'+str(opening['index']),
                'Frozen opening easy-first ramp removal; fixed key recipes; diagnostic approximation',
                basis='diagnostic_easy_ramp_removal_v1',component=role))
        total=sum(totals.values())
        refs=tuple(replace(r,component_share=totals[r.component]/total) for r in refs)
        reference_keys={r.opportunity for r in refs}
        for event in summary['events']:
            at=datetime.fromisoformat(event['as_of'])
            h=SimpleNamespace(sandbox=sandbox,zone=zone,catalog=catalog,
                events=[e for e in events if datetime.fromisoformat(e['as_of'])<=at],
                seed_review_entries=tuple(read_entry(r) for r in scenario.get('seed_review_entries',())))
            entries=tuple(e for e in review_entries(h,at,original_schedule=True) if e.key in reference_keys)
            signal=estimate(entries,refs,as_of=at,zone=zone,allow_diagnostic_reference=True)
            rows.append(dict(at=at.isoformat(),status=signal['status'],growth_per_7=signal['growth_per_7'],
                level_ratio=signal['level_ratio'],last_opportunity_at=signal.get('last_opportunity_at'),
                components=signal.get('components'),opportunities=len(signal['included']),unresolved=signal['unresolved']))
        final_entries=entries; final_at=at; final_signal=signal
        # Isolate the signal's contract using the SAME frozen irregular calendar.
        # These synthetic distances are NOT newly realized activity or physiology.
        controls={}
        for desired in (0.,.01,.02,.04):
            reference={r.opportunity:r for r in refs}; changed=[]
            for entry in final_entries:
                if not entry.completion: continue
                ref=reference[entry.key]
                elapsed=(entry.scheduled_at-at_open).total_seconds()/86400
                miles=ref.miles*(1+desired)**(elapsed/7)
                changed.append(replace(entry,completion=replace(entry.completion,
                    distance_meters=Quantity.point(miles*1609.344,'explicit synthetic distance-only control'))))
            result=estimate(changed,refs,as_of=final_at,zone=zone,allow_diagnostic_reference=True)
            assert abs(result['growth_per_7']-desired)<1e-9
            controls[str(desired)]={k:result[k] for k in ('status','growth_per_7','level_ratio')}
        # The REAL replay reference remains approximate and cannot authorize
        # adoption. Separate synthetic controls declare exact toy allocations
        # and generate their own distances; they do not relabel real evidence.
        original_level=opening['progression_target']['planning_volume_per_7']
        real_adoption=review_reference(None,volume=original_level,
            physical=summary['stage_reviews'][-1]['smoothed_support'],signal=final_signal,
            funding=published_funding(summary['events'][-1],as_of=final_at,zone=zone),
            baseline_volume=original_level,as_of=final_at,zone=zone,destination=100.)
        assert real_adoption['proposal']['rate_status']=='unverified_no_growth_reference'
        assert real_adoption['state']['volume']==original_level
        toy_refs=tuple(replace(r,basis='independent_no_growth_allocation_v1',
            source='EXACT synthetic generator allocation; not a verified real-allocator reference') for r in refs)
        ref_by_key={r.opportunity:r for r in toy_refs}
        last_clock=max(e.scheduled_at for e in final_entries)
        recognized_controls=[]
        for observed_level in (22.,25.):
            for desired in (0.,.02,.04,.06):
                changed=[]
                for entry in final_entries:
                    if not entry.completion: continue
                    elapsed=days_between(entry.scheduled_at,last_clock)
                    miles=ref_by_key[entry.key].miles*observed_level/20*(1+desired)**(elapsed/7)
                    changed.append(replace(entry,completion=replace(entry.completion,
                        distance_meters=Quantity.point(miles*1609.344,'synthetic reference-adoption control only'))))
                signal=estimate(changed,toy_refs,as_of=final_at,zone=zone)
                step=review_reference(None,volume=20.,physical=dict(actual_per_7=observed_level,
                    delivery_ratio=1.,mean_age_days=14),signal=signal,
                    funding=dict(status='known',funded_level=20.),baseline_volume=20.,
                    as_of=final_at,zone=zone,destination=100.)
                state=step['state']; proposal=step['proposal']
                projection=advance(state['volume'],proposal,elapsed_days=7,destination=100.)
                assert abs(state['volume']-observed_level)<1e-9
                assert abs(projection-observed_level*(1+min(desired,.04)))<1e-9
                repeat=review_reference(state,volume=state['volume'],physical=dict(actual_per_7=observed_level,
                    delivery_ratio=1.,mean_age_days=14),signal=signal,
                    funding=dict(status='known',funded_level=20.),baseline_volume=20.,
                    as_of=final_at,zone=zone,destination=100.)
                assert repeat['newly_recognized_increase']==0.
                recognized_controls.append(dict(observed_level=observed_level,original_target=20.,
                    original_funded_level=20.,measured_growth=desired,recognized_target=state['volume'],
                    planner_growth_per_7=state['growth_per_7'],conditional_seven_day_projection=projection,
                    repeat_recognition=repeat['newly_recognized_increase']))
    finally:
        db.close()
    manifest=freeze_sources(output/'sources.zip',config={},entrypoint=__name__,inputs=(source,
        root/'scenario.json',root/'sandbox.sqlite',root/'execution-sandbox.json',*sorted(root.glob('decision-*.json'))))
    result=dict(status='offline_signal_probe_only',rows=rows,controls=controls,
        synthetic_reference_adoption_controls=recognized_controls,
        real_reference_adoption=dict(status=real_adoption['proposal']['rate_status'],
            before=original_level,after=real_adoption['state']['volume']),
        reference_allocations=[dict(key=r.opportunity,miles=r.miles,component=r.component,share=r.component_share) for r in refs],
        reference_basis='Frozen opening schedule; easy distances divided by declared EASY-support ramp only; held keys NOT divided by growing total target',
        reference_limitation='Proportional easy allocation is a diagnostic approximation, not a verified no-growth rerun of the actual dose allocator. Different allocation shapes can affect the estimate.',
        manifest=manifest,planner_searches=0,live_changed=False,
        limitation='Original V4 synthetic future uploads. Not a newly searched calendar, new stability pass, historical initialization acceptance, or physiological validation.')
    with (output/'report.json').open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(last=rows[-1],controls=controls,
        synthetic_reference_adoption_controls=recognized_controls,
        real_reference_adoption=result['real_reference_adoption'],planner_searches=0,live_changed=False),indent=2))


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); audit(args.root,args.output)
