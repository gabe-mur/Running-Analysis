"""Saved-case full-path funding experiment; no search, deployment or gate edits."""
from argparse import ArgumentParser
from datetime import timedelta
import json
from pathlib import Path
from time import perf_counter

from run_analysis import weekly_schedule as planner
from run_analysis.adherence_projection import ProjectionReplan, ProjectionPlanSession
from run_analysis.projection_gate import ProjectionGateConfig, evaluate_replan_regressions
from scripts.audit_conditional_upload import score, scoring, window
from scripts.audit_continuation_upload import reconstruct, unpack_soft
from scripts.audit_volume_search import funding_trace, timing_stress
from scripts.boundary_funding_prototype import boundary_funding_policy
from scripts.volume_path_prototype import volume_path_policy


def scale_day(day, factor):
    if day.recommendation is None:
        return day
    result = day.recommendation
    return day.model_copy(update={'recommendation': result.model_copy(update={
        'distance_target_miles': round(planner._midpoint(result)*factor,1),
        'distance_range_miles': tuple(round(x*factor,1) for x in result.distance_range_miles),
    })})


def adversaries(full):
    """Deliberately altered doses, not executable or guard-cleared prescriptions."""
    removed = sum(planner._midpoint(d.recommendation) for d in full[:7] if d.recommendation)
    closing = sum(planner._midpoint(d.recommendation) for d in full[-7:] if d.recommendation)
    if not closing:
        raise ValueError('Catch-up control needs a nonempty closing block')
    delayed = [d.model_copy(update={'recommendation':None,'planned_at':None,'day_role':'rest_day'})
               if i < 7 else scale_day(d,1+removed/closing) if i >= len(full)-7 else d
               for i,d in enumerate(full)]
    return dict(underload_40_percent=[scale_day(d,.6) for d in full],
        excess_40_percent=[scale_day(d,1.4) for d in full],
        first_week_deferred_to_last=delayed)


def legacy_funding_gate(inputs, days):
    """Invoke the unchanged real gate on the sliced funding inputs only.

    This is not a full replan-acceptance report: no new recovery/stability
    evidence is fabricated for the counterfactual hard length.
    """
    opening=inputs['daily_states'][0]
    recommendations=[d.recommendation for d in days if d.recommendation]
    boundary=planner._recent_boundary_session(opening)
    snapshot=ProjectionReplan(generated_at=opening.as_of,opening_load_ratio=None,
        target_low_miles=inputs['target_distance_range'][0],target_high_miles=inputs['target_distance_range'][1],
        planned_sessions=tuple(ProjectionPlanSession(planned_for=r.planned_for,workout_type=r.workout_type,
            midpoint_miles=planner._midpoint(r)) for r in recommendations),committed_sessions=(),
        ordinary_easy_midpoint_miles=sum(planner.typical_easy_distance(opening))/2,
        boundary_session_miles=(boundary or (0.,0.))[1],
        target_trajectory=tuple(inputs['expected_target_projector'](recommendations,len(days))))
    return [dict(code=f.code,detail=f.detail,evidence=f.evidence)
            for f in evaluate_replan_regressions(snapshot,ProjectionGateConfig())
            if f.code=='hard_horizon_underfunded']


def investigate(report):
    if (report['synthetic_sandbox'].get('volume_path',{}).get('version')!='additive_date_path_v1'
        or report['synthetic_sandbox']['expected_completion'].get('state_transition_version')!='daily_pace_v1'
        or len(report['replans'])!=2 or report['synthetic_sandbox'].get('funding_accounting',{}).get('enabled')):
        raise ValueError('Require saved additive/daily-pace first-upload failure')
    captured = []
    reconstruct(report,lambda i,inputs,days,evidence:captured.append((inputs,days,evidence)))
    physical = [days+unpack_soft(snapshot['planner_diagnostics']['continuation_selection']['soft_sessions'])
                for (_,days,_),snapshot in zip(captured,report['replans'])]
    output = dict(native_controls=[], boundary=[], prior_comparison=[], adverse=[], timing={})
    with volume_path_policy():
        for i,(inputs,days,evidence) in enumerate(captured):
            end=days[0].date+timedelta(days=21)
            native=score(inputs,evidence,physical[i],end,end+timedelta(days=28),historical_pace=False)
            expected=report['replans'][i]['planner_diagnostics']['winning_candidate']['total_cost']
            assert abs(native['components']['total']-expected)<1e-8
            output['native_controls'].append(native['components'])
        inputs,days,evidence=captured[1]
        close=days[0].date+timedelta(days=49)
        with scoring(inputs,evidence,False):
            original_timing=timing_stress(inputs,days,evidence)
        for count in (19,20,21):
            end=days[0].date+timedelta(days=count)
            before=score(inputs,evidence,physical[1],end,close,historical_pace=False)
            with boundary_funding_policy() as metrics:
                after=score(inputs,evidence,physical[1],end,close,historical_pace=False)
            for key in ('recovery','coaching','support','shape','switching','policy_rate','soft_target'):
                assert before['components'][key]==after['components'][key],key
            accounting=after['continuation']['funding_accounting']
            assert abs(before['components']['total']-after['components']['total']-accounting['removed_cost'])<1e-8
            with scoring(inputs,evidence,False):
                gate=funding_trace(inputs,days[:count])
                failures=legacy_funding_gate(inputs,days[:count])
            assert bool(failures)==(gate['raw_violation']>planner.HARD_FUNDING_ROUNDING_TOLERANCE_MILES+1e-9)
            output['boundary'].append(dict(hard_days=count,old=before,new=after,
                unchanged_legacy_funding_breach=gate['raw_violation']>planner.HARD_FUNDING_ROUNDING_TOLERANCE_MILES+1e-9,
                legacy_funding_trace=gate,unchanged_gate_failures=failures,metrics=metrics))
        assert max(r['new']['components']['total'] for r in output['boundary'])-min(
            r['new']['components']['total'] for r in output['boundary'])<1e-8
        with boundary_funding_policy(), scoring(inputs,evidence,False):
            revised_timing=timing_stress(inputs,days,evidence)
        assert original_timing==revised_timing
        output['timing']=dict(unchanged=True,passing=sum(r['checks_pass'] for r in revised_timing),
            combinations=len(revised_timing),rows=revised_timing)
        with boundary_funding_policy():
            for name,sequence in adversaries(physical[1]).items():
                probe=score(inputs,evidence,sequence,days[0].date+timedelta(days=21),close,historical_pace=False)
                output['adverse'].append(dict(name=name,**probe))
            earlier,early_days,early_evidence=captured[0]
            hard_end=early_days[0].date+timedelta(days=21)
            full_end=hard_end+timedelta(days=28)
            later_with_consumed=window(physical[0],early_days[0].date,days[0].date)+physical[1]
            for name,sequence in (('original',physical[0]),('later_calendar',later_with_consumed)):
                probe=score(earlier,early_evidence,sequence,hard_end,full_end,historical_pace=False)
                output['prior_comparison'].append(dict(name=name,**probe))
        base=output['boundary'][-1]['new']['continuation']['funding_accounting']
        for row in output['adverse']:
            new=row['continuation']['funding_accounting']
            assert new['target_cost']>base['target_cost'],row['name']
            if row['name']=='first_week_deferred_to_last':
                assert new['volume_cost']>base['volume_cost']
    return output


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--output-json',type=Path,required=True)
    args=parser.parse_args()
    if args.output_json.exists():
        parser.error('Choose a new output path')
    started=perf_counter()
    result=investigate(json.loads(args.report.read_text()))
    result['seconds']=perf_counter()-started
    with args.output_json.open('x') as stream:
        json.dump(result,stream,indent=2)
        stream.write('\n')
    for row in result['boundary']:
        print('Boundary',row['hard_days'],row['old']['components']['total'],row['new']['components']['total'],
            'legacy funding breach',row['unchanged_legacy_funding_breach'])
    for row in result['adverse']+result['prior_comparison']:
        print(row['name'],row['components']['total'],row['continuation']['funding_accounting'])
    print('Timing',result['timing']['passing'],'/',result['timing']['combinations'],'unchanged')
    print('Seconds',result['seconds'])


if __name__=='__main__':
    main()
