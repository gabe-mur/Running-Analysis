"""Separate calendar coverage from role/dose construction on a saved failure.

The later calendar is a hindsight diagnostic witness, not a search algorithm.
Reconstruct both saved prescriptions and native prices before comparing it.
No live state, scoring weights, acceptance gates or search policy are changed.
"""
from argparse import ArgumentParser
from dataclasses import asdict
from datetime import date, datetime, timedelta
import json
from pathlib import Path
from time import perf_counter

from run_analysis import weekly_schedule as planner
from run_analysis.aerobic_acceptance import assess_aerobic_exposure
from scripts.audit_conditional_upload import components, score, scoring, window
from scripts.audit_continuation_upload import evaluate_physical, hard_score, reconstruct, unpack_soft
from scripts.audit_volume_search import cadence_trace, timing_stress
from scripts.boundary_funding_prototype import boundary_funding_policy
from scripts.continuation_replan import fingerprint, shortlist
from scripts.volume_path_prototype import volume_path_policy


def witness_roles(report):
    """Map saved physical run dates into the earlier hard horizon."""
    first, later = report['replans']
    origin = date.fromisoformat(first['decision_start_date'])
    boundary = date.fromisoformat(later['decision_start_date'])
    sessions = [r for r in first['planned_sessions']
                if datetime.fromisoformat(r['planned_for']).date() < boundary]
    sessions += later['planned_sessions']
    roles = {}
    for row in sessions:
        offset = (datetime.fromisoformat(row['planned_for']).date()-origin).days
        if 0 <= offset < 21:
            kind = planner.WorkoutType(row['workout_type'])
            roles[offset] = ('long' if kind == planner.WorkoutType.LONG else
                             'quality' if kind in planner.QUALITY_WORKOUT_TYPES else 'easy')
    return roles


def prescriptions(days):
    return [dict(date=d.date.isoformat(), at=d.recommendation.planned_for.isoformat(),
                 role=d.recommendation.workout_type.value,
                 miles=planner._midpoint(d.recommendation))
            for d in days if d.recommendation]


def admission_probe(record, price, acceptance):
    """Test existing shortlist rules without widening its budget or search."""
    row = dict(price, roles={}, acceptance=acceptance)
    original = record['evaluated']
    augmented = dict(record, evaluated=original+[row])
    selected = shortlist(augmented, limit=6)
    return dict(already_evaluated=any(r['offsets']==row['offsets'] for r in original),
                hard_rank_if_added=1+sum(r['total_cost'] < row['total_cost'] for r in original),
                admitted_if_added=any(r is row for r in selected),
                original_evaluated_count=len(original), finalist_budget=6,
                augmented_shortlist_offsets=[r['offsets'] for r in selected])


def investigate(report):
    sandbox = report['synthetic_sandbox']
    if (len(report['replans']) != 2 or not sandbox.get('funding_accounting', {}).get('enabled')
            or sandbox.get('volume_path', {}).get('version') != 'additive_date_path_v1'):
        raise ValueError('Require the saved two-decision additive/full-path-funding trial')
    roles = witness_roles(report)
    captured, proposed = [], []

    def observe(index, inputs, days, evidence):
        captured.append((inputs, days, evidence))
        if index:
            return
        # Construction must run inside the original replay allocator scopes.
        for name, overrides in (
            ('dates_only', None),
            ('dates_and_near_roles', {k:v for k,v in roles.items()
                                      if k < planner.NEAR_TERM_STABILITY_DECISION_OFFSETS}),
            ('dates_and_all_roles', roles),
        ):
            started = perf_counter()
            candidate, price, details = hard_score(inputs, sorted(roles), roles=overrides)
            if candidate is None or details.get('unrepresentable_distance_target'):
                raise ValueError(f'{name}: unrepresentable candidate')
            acceptance = assess_aerobic_exposure(**details['aerobic_acceptance_facts'],
                planning_mode=inputs['target_evidence'].planning_mode.value)
            proposed.append((name, candidate, price, asdict(acceptance), perf_counter()-started))

    reconstruct(report, observe)
    physical = [days+unpack_soft(snapshot['planner_diagnostics']['continuation_selection']['soft_sessions'])
                for (_,days,_),snapshot in zip(captured,report['replans'])]
    output = dict(witness_offsets=sorted(roles), native_controls=[], probes=[])
    output['construction_fingerprints'] = {name:fingerprint(candidate)
        for name,candidate,_,_,_ in proposed}
    output['role_overrides_changed_construction'] = len(set(output['construction_fingerprints'].values())) > 1
    with volume_path_policy(), boundary_funding_policy():
        for i, (inputs, days, evidence) in enumerate(captured):
            end = days[0].date+timedelta(days=21)
            native = score(inputs,evidence,physical[i],end,end+timedelta(days=28),historical_pace=False)
            expected = report['replans'][i]['planner_diagnostics']['winning_candidate']['total_cost']
            assert abs(native['components']['total']-expected) < 1e-8
            output['native_controls'].append(native['components'])
        inputs, days, evidence = captured[0]
        end = days[0].date+timedelta(days=21)
        witness = window(physical[0],days[0].date,captured[1][1][0].date)+physical[1]
        output['exact_later_program_before_upload'] = score(inputs,evidence,witness,
            end,end+timedelta(days=28),historical_pace=False)
        with scoring(inputs,evidence,False):
            for name, candidate, price, acceptance, construction_seconds in proposed:
                started = perf_counter()
                result = evaluate_physical(inputs,candidate,evidence)
                assert abs(price.total_cost-result['hard']['total_cost']) < 1e-8
                row = dict(name=name, construction_seconds=construction_seconds,
                    continuation_seconds=perf_counter()-started, acceptance=acceptance,
                    prescriptions=prescriptions(candidate), cadence=cadence_trace(candidate,inputs),
                    components=components(result), **result)
                output['probes'].append(row)
                print(name, 'hard', price.total_cost, 'full', row['components']['total'],
                      'acceptance', acceptance['reason'], flush=True)
            output['timing_stress'] = timing_stress(inputs,proposed[0][1],evidence)
    output['admission'] = admission_probe(sandbox['calendar_neighbor_prototype'][0],
        proposed[0][2].model_dump(mode='json'), proposed[0][3])
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
    print('Seconds', result['seconds'], flush=True)


if __name__=='__main__':
    main()
