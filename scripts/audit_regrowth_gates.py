"""Check saved regrowth candidates against unchanged single-replan gates."""
from argparse import ArgumentParser
from dataclasses import asdict, replace
import json
from pathlib import Path

from pydantic import TypeAdapter
from run_analysis import weekly_schedule as planner
from run_analysis.adherence_projection import ProjectionReplan, ProjectionPlanSession
from run_analysis.aerobic_acceptance import cadence_exceptions
from run_analysis.projection_gate import ProjectionGateConfig, evaluate_replan_regressions
from scripts.audit_continuation_upload import reconstruct, hard_score
from scripts.typed_continuation_prototype import runs


def check_candidate(saved, inputs, days):
    baseline=TypeAdapter(ProjectionReplan).validate_python(saved)
    planned=runs(days)
    def peak(sequence):
        return planner._peak_projected_continuous_mileage_rate(
            inputs['daily_states'][0].recent_load.continuous_distance_miles,
            inputs['daily_states'][0].as_of,sequence,half_life_days=float(
                inputs['config'].get('coaching',{}).get('continuous_fatigue_half_life_days',7)))
    snapshot=replace(baseline,planned_sessions=tuple(ProjectionPlanSession(
        planned_for=r.planned_for,workout_type=r.workout_type,midpoint_miles=r.planned_distance_miles,
        distance_low_miles=r.distance_range_miles[0],distance_high_miles=r.distance_range_miles[1],
        planning_role=r.planning_role,readiness=r.readiness.value) for r in planned),
        committed_sessions=(),planner_diagnostics=None,
        peak_projected_continuous_mileage_rate=peak(days[:7]),
        target_trajectory=tuple(inputs['expected_target_projector'](planned,len(days))),
        cadence_exception_reasons=cadence_exceptions(reason for r in planned for reason in r.reasons))
    config=ProjectionGateConfig(enforce_stability=True,audit_key_session_cadence=True,
                                enforce_aerobic_exposure_cadence=True)
    return dict(failures=[asdict(f) for f in evaluate_replan_regressions(snapshot,config)],
        whole_hard_peak_failures=[asdict(f) for f in evaluate_replan_regressions(
            replace(snapshot,peak_projected_continuous_mileage_rate=peak(days)),config)],
        strict_cadence_findings=[asdict(f) for f in evaluate_replan_regressions(
            snapshot,replace(config,enforce_key_session_cadence=True))],
        miles=sum(r.planned_distance_miles for r in planned),
        integrated_minimum=sum(t[0] for t in snapshot.target_trajectory)/7,
        peak_visible=peak(days[:7]),peak_hard=peak(days),
        readiness=[r.readiness.value for r in planned],
        cadence_exceptions=list(snapshot.cadence_exception_reasons),
        unchecked=['future daily/upload churn','realized delivery','90-day progression',
                   'timing robustness outside selected timestamps'])


def investigate(report,candidates):
    output=[]
    def observe(index,inputs,days,evidence):
        for candidate in candidates:
            days,price,details=hard_score(inputs,candidate['offsets'],
                roles={int(k):v for k,v in candidate['roles'].items()})
            if 'hard' in candidate:
                assert abs(price.total_cost-candidate['hard']['total_cost'])<1e-8
            output.append(dict(offsets=candidate['offsets'],
                **check_candidate(report['replans'][0],inputs,days)))
    reconstruct(dict(report,replans=report['replans'][:1]),observe)
    return output


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--candidates',type=Path,required=True)
    parser.add_argument('--output-json',type=Path,required=True)
    args=parser.parse_args()
    if args.output_json.exists():
        parser.error('Choose a new output path')
    output=investigate(json.loads(args.report.read_text()),
        json.loads(args.candidates.read_text())['discarded_probes'])
    with args.output_json.open('x') as stream:
        json.dump(output,stream,indent=2,default=str)
        stream.write('\n')
    for row in output:
        print(row['offsets'],'failures',row['failures'],'strict cadence',row['strict_cadence_findings'])


if __name__=='__main__':
    main()
