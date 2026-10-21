"""Reprice one coordinated selection and compare its expected realized review.

The future completion ledger is a declared nominal counterfactual, not measured
HR, uploaded training, stage acceptance or clearance. No new search is run.
"""
from argparse import ArgumentParser
from datetime import date,datetime,time,timedelta
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from scripts.search_prospective_progression import coordinated_case
from scripts.check_explicit_distribution_tail import incoming_demand_trace
from run_analysis.continuous_training_dose import ContinuousDoseReference
from run_analysis.adaptive.decision_snapshot import restore,evaluate_snapshot
from run_analysis.adaptive.new_model_daily_harness import planned_accounted
from run_analysis.delivered_training import DoseContract,DoseTarget,DeliveredDose
from run_analysis.staged_progression import ReviewEntry,review_recent_delivery,MILE
from run_analysis.workload_accounting import Quantity


def verify(root,index,output):
    saved=json.loads((output/'result.json').read_text())
    state=restore((output/'snapshot.json').read_text(),expected_fingerprint=saved['snapshot_fingerprint'])
    case=coordinated_case(root,index,review_hour=state.prospective_volume_review.end.astimezone(ZoneInfo(state.timezone)).hour)
    ref=ContinuousDoseReference('sandbox upload accounting')
    def evaluated(program):
        physical=tuple(planned_accounted(f'candidate:{d}:{k}',
            datetime.combine(date.fromisoformat(d),time(18),ZoneInfo(state.timezone)),
            case['catalog'][k],case['handoffs'][k],ref,available=state.as_of) for d,k in program)
        result=evaluate_snapshot(state,physical,tuple(k for _,k in program),reference=ref)
        return result,physical
    selected,physical=evaluated(saved['program'])
    assert selected['decision_eligible'] and abs(selected['evaluation'].total_cost-saved['cost'])<1e-12
    contract=state.prospective_volume_review
    entries=list(case['entries'])
    policy=json.loads((root/'scenario.json').read_text())['search_policy']
    margin=policy['distance_margin'];fraction=policy['phase_fraction_margin']
    for r,(_,key) in zip(physical,saved['program']):
        start=max(r.start,state.as_of);end=start+(r.end-r.start)
        if not (contract.start<=start<contract.end and end<=contract.end):continue
        dose=case['catalog'][key]
        work=[s for kind,s in dose.phases if kind=='work']
        targets=[DoseTarget('distance_meters',Quantity((dose.miles-margin)*MILE,
            (dose.miles+margin)*MILE,dose.miles*MILE,'expected issued band'))]
        if work:targets.append(DoseTarget('work_seconds',Quantity(sum(work)*(1-fraction),
            sum(work)*(1+fraction),sum(work),'expected issued work band')))
        intended=DoseContract(state.as_of,'high_aerobic' if work else 'long_endurance' if dose.role=='long' else 'easy_volume',
            tuple(targets),'nominal conditional execution',
            bout_seconds=Quantity(min(work)*(1-fraction),max(work)*(1+fraction)) if work else None)
        expected=DeliveredDose(r.key,start,end,end,Quantity.point(r.miles*MILE,'expected nominal distance'),
            Quantity.point((end-start).total_seconds(),'expected command duration'),r.bouts,
            'explicit_segments' if r.bouts else 'unknown',r.role)
        entries.append(ReviewEntry(r.key,start,intended,expected))
    recent=[r.miles for r in state.history if state.as_of-timedelta(days=30)<=r.start<state.as_of]
    realized=review_recent_delivery(case['stage'],entries,contract.end,ledger_complete=True,
        longest_prior_30d_miles=max(recent,default=None),required_miles=contract.required_miles)
    assert abs(realized['credited_miles']-selected['progression_delivery']['expected_credited_miles'])<1e-9
    assert realized['status']=='conditional_development_candidate'
    def pricing(result):
        e=result['evaluation']
        return dict(cost=e.total_cost,fatigue=e.weighted_demand_cost,delivery=e.delivery_cost,
            key_entry=[t for t in incoming_demand_trace(e,state.context) if t['role']!='easy'])
    archived=tuple((d,k) for d,k in case['event']['program'] if date.fromisoformat(d)>=state.context.decision)
    before,_=evaluated(archived)
    report=dict(selected=pricing(selected),baseline=pricing(before),expected_review=realized,
        volume_credit_matches=True,score_replayed=True,
        assumption='Nominal future distance/structure completion; no future adverse feedback assumed for this arithmetic check',
        limitation='Conditional review proposal, not actual completed training or accepted progression',
        live_changed=False)
    with (output/'verification.json').open('x') as stream:json.dump(report,stream,indent=2,default=str)
    print(json.dumps(dict(review_status=realized['status'],credited=realized['credited_miles'],
        required=realized['required_miles'],candidate_volume=realized['candidate'].weekly_equivalent_miles,
        score=saved['cost']),default=str))


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--root',type=Path,default=Path('output/current-v18-volume-curve-daily-2-20260930'))
    p.add_argument('--event',type=int,default=1);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();verify(args.root,args.event,args.output)
