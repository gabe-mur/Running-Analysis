"""Small physical counterfactual panel for the stopped repaired daily replay.

No search, publication, weight change, gate relaxation or completion recovery.
"""
from argparse import ArgumentParser
from dataclasses import replace
from datetime import date, datetime, time, timedelta
import json
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

from scripts.replay_adaptive_search import load_decision
from scripts.audit_progression_rollover_instability import (
    fixed_taper_guard, seed_library, score, splice_program, search_trace,
)
from scripts.check_rollover_search_repair import check_empty_rollover
from scripts.research_manifest import freeze_sources
from run_analysis.config import load_config
from run_analysis.adaptive.decision_snapshot import evaluate_snapshot
from run_analysis.adaptive.new_model_daily_harness import planned_accounted
from run_analysis.adaptive.soft_horizon_trial import day_weight
from run_analysis.continuous_training_dose import ContinuousDoseReference


def timing_tail(case,program,context):
    """Expose the same priced timing rows, not a second purpose model."""
    state=replace(case['state'],context=context)
    reference=ContinuousDoseReference('sandbox upload accounting')
    physical=tuple(planned_accounted(f'candidate:{d}:{k}',
        datetime.combine(date.fromisoformat(d),time(18),ZoneInfo(state.timezone)),
        case['catalog'][k],case['handoffs'][k],reference,available=state.as_of)
        for d,k in program)
    evaluation=evaluate_snapshot(state,physical,tuple(k for _,k in program),
                                 reference=reference)['evaluation']
    span=(context.end-context.stage_start).days
    return dict(cost=evaluation.total_cost,end=str(context.end),
        daily=[dict(day=str(row['day']),loss=row['loss'],
                    weight=day_weight(context,row['day']),
                    weighted_long_loss=row['loss']*day_weight(context,row['day'])
                        *context.policy.purpose_weights[2]/span)
               for row in evaluation.parts['timing']['long']['daily'][-12:]])


def audit(root,output,*,before_index=6,after_index=7):
    before=load_decision(root,before_index); after=load_decision(root,after_index)
    output.mkdir(exist_ok=False)
    manifest=freeze_sources(output/'sources.zip',config=load_config(),entrypoint=__name__,
        inputs=(root/f'decision-{before_index:04d}.json',root/f'decision-{after_index:04d}.json',
                root/before['event']['snapshot_file'],root/after['event']['snapshot_file'],
                root/'scenario.json',root/'lifecycle-summary.json'))
    started=perf_counter()
    prior=tuple(map(tuple,before['event']['program']))
    selected=tuple(map(tuple,after['event']['program']))
    unchanged=check_empty_rollover(before,after,prior)
    a,b=before['state'].context,after['state'].context
    shared=tuple(p for p in selected if p[0]<before['event']['end'])
    clock=replace(a,decision=b.decision,hard_end=b.hard_end,
                  continuation_guard_days=fixed_taper_guard(a.continuation_guard_days,1))
    taper=replace(clock,continuation_guard_days=a.continuation_guard_days)
    controls=[]
    calls=0
    def price(case,program,context=None):
        nonlocal calls
        calls+=1
        return score(case,program,context)
    for name,case,context in (('original',before,a),('clock_only_old_absolute_weights',after,clock),
                              ('shift_taper_keep_old_end',after,taper),('full_rollover',after,b)):
        old,new=price(case,prior,context),price(case,shared,context)
        controls.append(dict(name=name,prior=old,new_shared_dates=new,
            improvement=old['cost']-new['cost'],taper_start=str(context.taper_start),end=str(context.end)))
    assert abs(controls[0]['prior']['cost']-before['event']['cost'])<1e-12
    # Equality held for the earlier October12/13 frame, but must be measured,
    # not assumed for an earlier frame with an active progression ramp.
    clock_deltas={k:controls[1][k]['cost']-controls[0][k]['cost']
                  for k in ('prior','new_shared_dates')}
    assert abs(controls[0]['improvement']-controls[1]['improvement'])<1e-12
    winner=price(after,selected)
    assert abs(winner['cost']-after['event']['cost'])<1e-12
    cuts=sorted({(b.decision+timedelta(days=7)).isoformat(),b.hard_end.isoformat(),
                 a.taper_start.isoformat(),b.taper_start.isoformat()})
    splices=[]; dose_repairs=[]
    for cut in cuts:
        program=splice_program(prior,selected,cut)
        value=price(after,program)
        splices.append(dict(cut=cut,program=program,score=value))
        # If the seam loses volume funding, compare small, fully priced easy
        # top-ups before the ORIGINAL review deadline. Nothing is published or
        # accepted just because it repairs arithmetic; every guard is rerun.
        if not value['funded']:
            for day,key in program:
                dose=after['catalog'][key]
                if dose.role!='easy' or day>=after['state'].prospective_volume_review.end.date().isoformat():
                    continue
                for tick in (1,2,3):
                    options=[k for k in after['handoffs'] if after['catalog'][k].role=='easy'
                             and abs(after['catalog'][k].miles-(dose.miles+tick/10))<1e-8]
                    if not options:
                        continue
                    target=min(options)
                    revised=tuple((d,target if d==day else k) for d,k in program)
                    checked=price(after,revised)
                    dose_repairs.append(dict(cut=cut,day=day,added_miles=tick/10,
                        program=revised,score=checked))
    pools=[seed_library(case,1680) for case in (before,after)]
    low,high=after['dates'][0],before['event']['end']
    clip=lambda pool:tuple(tuple(p for p in plan if low<=p[0]<high) for plan in pool)
    eligible=[r for r in (*splices,*dose_repairs) if r['score']['eligible']]
    tail_controls={name:{label:timing_tail(case,program,context)
                        for label,program in (('prior',prior),('new_shared_dates',shared))}
                   for name,case,context in (('shift_taper_keep_old_end',after,taper),
                                              ('full_rollover',after,b))}
    calls+=4
    for control in controls[-2:]:
        for label in ('prior','new_shared_dates'):
            assert abs(tail_controls[control['name']][label]['cost']-
                       control[label]['cost'])<1e-12
    result=dict(status='bounded_stop_attribution',manifest=manifest,unchanged=unchanged,
        controls=controls,clock_only_cost_deltas=clock_deltas,
        long_timing_tail=tail_controls,
        selected=winner,splices=splices,dose_repairs=dose_repairs,
        best_stable_prefix_candidate=min(eligible,key=lambda r:r['score']['cost']) if eligible else None,
        complete_seed_shapes=dict(before=len(pools[0]),after=len(pools[1]),
                                  ordered_common_shapes_identical=clip(pools[0])==clip(pools[1])),
        traces=[search_trace(c['event']) for c in (before,after)],
        reprices=calls,seconds=perf_counter()-started,
        scope='Finite counterexamples and ordered horizon ablations, not global optimum/daily pass/physiological clearance; no live change')
    with (output/'report.json').open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ('controls','selected','complete_seed_shapes','reprices','seconds')}),flush=True)
    print(json.dumps(dict(best_stable_prefix_candidate=result['best_stable_prefix_candidate']),default=str),flush=True)
    return result


if __name__=='__main__':
    parser=ArgumentParser()
    parser.add_argument('root',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    audit(args.root,args.output_dir)
