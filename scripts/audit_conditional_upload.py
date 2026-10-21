"""Fixed-program diagnosis of a saved conditional-state upload failure.

No search, weight changes, database writes, or gate relaxation. Historical pace
is an explicit reconstruction control, not a scheduler option. Assert saved
fingerprints and native score identity before trusting counterfactual probes.
"""
from argparse import ArgumentParser
from contextlib import contextmanager
from datetime import timedelta
import json
from pathlib import Path
from time import perf_counter
from math import log

from scripts.audit_continuation_upload import (
    reconstruct, saved_run_time_pace, unpack_soft, evaluate_physical,
)
from scripts.audit_expected_completion import differences
from scripts.expected_completion import ExpectedCompletion
from scripts.finalized_scoring_trial import finalized_scoring_policy
from scripts.typed_continuation_prototype import runs
from run_analysis import adherence_projection as projection, weekly_schedule as planner
from run_analysis.prescription_distance import canonical_distance_policy
from run_analysis.selection_objective import EXPECTED_COMPLETION_STATE, shared_objective_policy


@contextmanager
def scoring(inputs, evidence, historical_pace):
    model = ExpectedCompletion(evidence['template'], evidence['runs'], inputs['config'])
    token = EXPECTED_COMPLETION_STATE.set(model)
    try:
        with saved_run_time_pace(historical_pace), canonical_distance_policy(), \
                shared_objective_policy(prior_schedule=inputs.get('prior_schedule')), finalized_scoring_policy():
            yield model
    finally:
        EXPECTED_COMPLETION_STATE.reset(token)


def window(sequence, start, end):
    selected = [d for d in sequence if start <= d.date < end]
    if [d.date for d in selected] != [start + timedelta(days=i) for i in range((end-start).days)]:
        raise ValueError('Fixed physical program does not cover the requested window exactly')
    return selected


def components(result):
    c, h = result['continuation'], result['hard']
    return dict(total=c['combined_cost'], target=c['combined_target_cost'],
        target_violation=c['combined_target_violation'],
        support=planner.PROGRAM_FIT_UNIT * planner.AEROBIC_SUPPORT_FIT_FRACTION * c['combined_support_violation'],
        shape=planner.PROGRAM_FIT_UNIT * c['combined_shape_violation'],
        recovery=c['trajectory_recovery_cost'], recovery_components=c['trajectory_recovery_components'],
        coaching=c['combined_coaching_cost'], switching=h['distance_switching_cost'],
        policy_rate=c['policy_rate'], soft_target=c['soft_target'],
        hard_target_violation=h['target_violation'],
        hard_miles=h['planned_midpoint_miles'], hard_integrated_target=h['integrated_target_low_miles'])


def path_trace(days, inputs, *, length, anchor=None):
    """Expand the existing formula for audit; assert exact scalar reproduction."""
    start=inputs['daily_states'][0].as_of
    targets=inputs['expected_target_projector'](runs(days),len(days))[:length]
    opening=inputs['daily_states'][0].recent_load.continuous_distance_miles
    half_life=max(.1,float(inputs['config'].get('coaching',{}).get('continuous_fatigue_half_life_days',7)))
    decay=.5**(1/half_life)
    normalization=log(2)*7/half_life
    ordinary=sum(planner.typical_easy_distance(inputs['daily_states'][0]))/2
    tolerance=ordinary*normalization
    miles=[planner._midpoint(d.recommendation) if d.recommendation else 0. for d in days[:length]]
    hours=planner._planned_impulse_hours(days)[:length]
    rate=opening
    anchor=anchor or dict(at=start,rate=opening,target=targets[0])
    low_offset=min(anchor['rate'],anchor['target'][0])-anchor['target'][0]
    high_offset=max(anchor['rate'],anchor['target'][1])-anchor['target'][1]
    rows=[]
    for i,(distance,target,hour) in enumerate(zip(miles,targets,hours)):
        rate=planner._advance_continuous_mileage_rate(rate,distance,daily_decay=decay,
            session_normalization=normalization,impulse_hour=hour,
            opening_hour=planner._clock_hour(start) if i==0 else 0.)
        end=start.replace(hour=0,minute=0,second=0,microsecond=0)+timedelta(days=i+1)
        elapsed=(end-anchor['at']).total_seconds()/86400
        lower=target[0]+low_offset*decay**elapsed
        upper=target[1]+high_offset*decay**elapsed
        rows.append(dict(date=days[i].date.isoformat(),end_at=end.isoformat(),miles=distance,
            rate=rate,lower=lower,upper=upper,tolerance=tolerance,
            deficit=max(0.,lower-tolerance-rate),excess=max(0.,rate-upper-tolerance)))
    worst=max(max(r['deficit'],r['excess']) for r in rows)
    if anchor['at']==start and anchor['rate']==opening and anchor['target']==targets[0]:
        actual=planner._continuous_mileage_path_violation(miles,planner._average_target_range(targets),
            session_tolerance_miles=tolerance,opening_weekly_rate=opening,half_life_days=half_life,
            target_weekly_ranges=targets,impulse_hours=hours,opening_hour=planner._clock_hour(start))
        assert abs(worst-actual)<1e-10
    return dict(worst=worst,opening_rate=opening,days=rows)


def score(inputs, evidence, sequence, hard_end, full_end, *, historical_pace, regenerate=False):
    start = inputs['daily_states'][0].as_of.date()
    hard = window(sequence, start, hard_end)
    full = window(sequence, start, full_end)
    adjusted = dict(inputs, daily_states=inputs['daily_states'][:len(hard)],
                    daily_state_options=inputs['daily_state_options'][:len(hard)])
    with scoring(adjusted, evidence, historical_pace):
        result = evaluate_physical(adjusted, hard, evidence,
            full=None if regenerate else full, length=len(full)-len(hard))
    totals=components(result)
    assert abs(totals['total']-sum(totals[k] for k in (
        'target','support','shape','recovery','coaching','switching')))<1e-8
    return dict(components=totals, **result)


def replace_path_cost(result, inputs, *, old_path, new_path):
    """Accounting counterfactual only: all other scalar components stay fixed."""
    ordinary=sum(planner.typical_easy_distance(inputs['daily_states'][0]))/2
    c=result['components']
    violation=c['target_violation']-old_path+new_path
    assert violation>=-1e-8
    return c['total']-c['target']+planner.TARGET_FIT_UNIT*(max(0.,violation)/ordinary)**2


def coverage(report, physical):
    rows = []
    for i, snapshot in enumerate(report['replans'][-2:]):
        from datetime import date
        origin = date.fromisoformat(snapshot['decision_start_date'])
        record = report['synthetic_sandbox']['calendar_neighbor_prototype'][-2+i]
        finalists = snapshot['planner_diagnostics']['continuation_selection']['finalists']
        for source, sequence in zip(('before', 'after'), physical):
            offsets = [(d.date-origin).days for d in sequence if d.recommendation
                       and 0 <= (d.date-origin).days < 21]
            near = [offset for offset in offsets if offset < planner.NEAR_TERM_STABILITY_DECISION_OFFSETS]
            rows.append(dict(origin=('before','after')[i], program=source, offsets=offsets,
                evaluated_exact=[r for r in record['evaluated'] if r['offsets']==offsets],
                shortlisted_exact=[r for r in finalists if r['offsets']==offsets], near_offsets=near,
                evaluated_near_count=sum([o for o in r['offsets'] if o < planner.NEAR_TERM_STABILITY_DECISION_OFFSETS]==near
                                         for r in record['evaluated']),
                shortlisted_near=[r for r in finalists if [o for o in r['offsets']
                    if o < planner.NEAR_TERM_STABILITY_DECISION_OFFSETS]==near]))
    return rows


def investigate(report, *, historical_pace=False, regenerate=False):
    captured = []
    def observe(index, inputs, days, evidence):
        if index >= len(report['replans'])-2:
            captured.append((inputs, days, evidence))
    reconstruct(report, observe, historical_run_time_pace=historical_pace)
    before, after = captured
    origins = [x[0]['daily_states'][0].as_of for x in captured]
    physical = [days + unpack_soft(snap['planner_diagnostics']['continuation_selection']['soft_sessions'])
                for (_, days, _), snap in zip(captured, report['replans'][-2:])]
    physical[1] = window(physical[0], origins[0].date(), origins[1].date()) + physical[1]
    end = origins[0].date()+timedelta(days=21)
    close = end+timedelta(days=28)
    rows = []
    # Native score controls must reproduce actual saved selection before any
    # horizon, target, history or pace substitutions are interpreted.
    for i, (inputs, days, evidence) in enumerate(captured):
        hard_end = origins[i].date()+timedelta(days=21)
        full_end = hard_end+timedelta(days=28)
        result = score(inputs, evidence, physical[i], hard_end, full_end, historical_pace=historical_pace)
        expected = report['replans'][-2+i]['planner_diagnostics']['winning_candidate']['total_cost']
        assert abs(result['components']['total']-expected)<1e-8, (i,result['components'],expected)
        rows.append(dict(origin=('before','after')[i], program=('before','after')[i],
                         mode='native_control', **result))
        print('Native score matched:', ('before', 'after')[i], expected, flush=True)
    for i, (inputs, days, evidence) in enumerate(captured):
        for source, sequence in zip(('before','after'),physical):
            for regenerated in ([False, True] if regenerate else [False]):
                result=score(inputs,evidence,sequence,end,close,historical_pace=historical_pace,regenerate=regenerated)
                rows.append(dict(origin=('before','after')[i],program=source,
                    mode='fixed_hard_regenerated_tail' if regenerated else 'fixed_absolute_program', **result))
                print('Probe:', ('before','after')[i], source, rows[-1]['mode'], result['components'], flush=True)
    # One common future decision, common hard end and common tail end. Compare
    # expected execution with independently realized execution, not two origins.
    boundary=origins[1]
    predicted=ExpectedCompletion(before[2]['template'],before[2]['runs'],before[0]['config'])
    realized=ExpectedCompletion(after[2]['template'],after[2]['runs'],after[0]['config'])
    consumed=[r for r in runs(before[1]) if r.planned_for<boundary]
    ledger=predicted.ledger(boundary,consumed)
    state_diff=differences(predicted.at(boundary,consumed).model_dump(mode='json'),
                           realized.at(boundary).model_dump(mode='json'))
    assert not state_diff, state_diff
    same_time=[]
    for label,history in (('predicted',ledger),('realized',after[2]['runs'])):
        evidence=dict(after[2],runs=history)
        model=ExpectedCompletion(evidence['template'],history,after[0]['config'])
        capacity=model.at(boundary).recent_load.capacity_reference_miles
        coaching=after[0]['config'].get('coaching',{})
        def rebuild(raw):
            return projection._state_at(evidence['template'],history,raw.as_of,capacity,
                easy_baseline_half_life_days=float(coaching.get('capacity_retention_half_life_days',84)),
                fatigue_half_life_days=float(coaching.get('continuous_fatigue_half_life_days',7)))
        inputs=dict(after[0], daily_states=[rebuild(s) for s in after[0]['daily_states']],
            daily_state_options=[[rebuild(s) for s in opts] for opts in after[0]['daily_state_options']])
        assert inputs['daily_states']==after[0]['daily_states']
        assert inputs['daily_state_options']==after[0]['daily_state_options']
        for source,sequence in zip(('before','after'),physical):
            result=score(inputs,evidence,sequence,end,close,historical_pace=historical_pace)
            same_time.append(dict(history=label,program=source,**result))
    for i in range(2):
        assert same_time[i]['components']==same_time[i+2]['components']
    # Freeze the old candidate-independent demand curve on identical absolute
    # dates, without letting a candidate lower its own funding target.
    shift=(boundary.date()-origins[0].date()).days
    prior_targets=before[0]['expected_target_projector'](runs(before[1]),49)
    fresh_targets=after[0]['expected_target_projector'](runs(after[1]),49-shift)
    def aligned_prior(planned,length):
        assert length+shift<=len(prior_targets)
        return prior_targets[shift:shift+length]
    aligned_inputs=dict(after[0],expected_target_projector=aligned_prior,
                        target_distance_range=prior_targets[shift])
    path_probes=[]
    for source,sequence in zip(('before','after'),physical):
        for i,(inputs,_,_) in enumerate(captured):
            hard=window(sequence,origins[i].date(),end)
            path_probes.append(dict(origin=('before','after')[i],program=source,mode='native_seven_days',
                                   **path_trace(hard,inputs,length=7)))
        hard=window(sequence,origins[0].date(),end)
        path_probes.append(dict(origin='before',program=source,mode='through_same_absolute_day',
                               **path_trace(hard,before[0],length=7+shift)))
        anchor=dict(at=origins[0],rate=before[0]['daily_states'][0].recent_load.continuous_distance_miles,
                    target=prior_targets[0])
        hard=window(sequence,origins[1].date(),end)
        path_probes.append(dict(origin='after',program=source,mode='prior_anchor_same_absolute_targets',
                               **path_trace(hard,aligned_inputs,length=7,anchor=anchor)))
        path_probes.append(dict(origin='after',program=source,mode='same_old_end_current_anchor',
                               **path_trace(hard,after[0],length=7-shift)))
    for source,sequence in zip(('before','after'),physical):
        result=score(aligned_inputs,after[2],sequence,end,close,historical_pace=historical_pace)
        rows.append(dict(origin='after',program=source,mode='fixed_program_prior_demand',**result))
        # Current pace policy is also checked on the same physical saved choices;
        # this is not a fresh search and cannot certify its selected calendar.
        if historical_pace:
            result=score(after[0],after[2],sequence,end,close,historical_pace=False)
            rows.append(dict(origin='after',program=source,mode='current_daily_pace_fixed_program',**result))
    counterfactuals=[]
    for p in path_probes:
        if p['mode']=='native_seven_days':
            continue
        baseline=next(x for x in rows if x['mode']=='fixed_absolute_program'
                      and x['origin']==p['origin'] and x['program']==p['program'])
        old_path=next(x['worst'] for x in path_probes if x['mode']=='native_seven_days'
                     and x['origin']==p['origin'] and x['program']==p['program'])
        counterfactuals.append(dict(origin=p['origin'],program=p['program'],mode=p['mode'],
            total=replace_path_cost(baseline,captured[0 if p['origin']=='before' else 1][0],
                                    old_path=old_path,new_path=p['worst'])))
    return dict(historical_run_time_pace=historical_pace, state_differences=state_diff,path_probes=path_probes,
        path_only_counterfactuals=counterfactuals,
        same_time=same_time, probes=rows, coverage=coverage(report,physical),
        target_changes=[dict(date=(boundary+timedelta(days=i)).date().isoformat(),before=a,after=b)
                        for i,(a,b) in enumerate(zip(prior_targets[shift:],fresh_targets)) if a!=b],
        scope=__doc__)


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--output-json',type=Path,required=True)
    parser.add_argument('--historical-run-time-pace',action='store_true')
    parser.add_argument('--regenerate-tails',action='store_true')
    args=parser.parse_args()
    if args.output_json.exists():
        parser.error('Choose a fresh output path')
    started=perf_counter()
    result=investigate(json.loads(args.report.read_text()),
        historical_pace=args.historical_run_time_pace,regenerate=args.regenerate_tails)
    result['seconds']=perf_counter()-started
    with args.output_json.open('x') as stream:
        json.dump(result,stream,indent=2)
        stream.write('\n')
    for row in result['probes']:
        print(row['origin'],row['program'],row['mode'],row['components'],flush=True)
    print('Wrote',args.output_json,'in',result['seconds'],'seconds')


if __name__=='__main__':
    main()
