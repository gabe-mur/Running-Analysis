"""Summarize measured search cost and actual daily/upload outcomes, no search."""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
from statistics import median


def summarize(report):
    timings=[r['planning_seconds'] for r in report['replans']]
    decisions=[]
    for i,snapshot in enumerate(report['replans']):
        selected=snapshot['planner_diagnostics']['continuation_selection']
        refinement=selected.get('distance_refinement')
        decisions.append(dict(index=i,at=snapshot['generated_at'],trigger=snapshot['trigger'],
            planning_seconds=snapshot['planning_seconds'],cost=selected['search_cost'],
            full_fingerprint=selected['selected_program_fingerprint'],
            distance_seconds=refinement['seconds'] if refinement else None,
            distance_evaluations=refinement['evaluations'] if refinement else None,
            selected_locally_converged=refinement['selected_locally_converged'] if refinement else None,
            distance_lanes=refinement['lanes'] if refinement else None,
            near_sessions=selected['hard_sessions'][:3]))
    return dict(stopped=report['stopped_early'],failures=report['failures'],advisories=report['advisories'],
        daily_replans=report['daily_replan_count'],upload_replans=report['post_upload_replan_count'],
        median_planning_seconds=median(timings),max_planning_seconds=max(timings),
        total_planning_seconds=sum(timings),decisions=decisions,churn=report['churn'],
        churn_by_trigger=report['churn_by_trigger'],executions=report['realized_execution']['sessions'])


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument('--control',type=Path,required=True)
    p.add_argument('--prototype',type=Path,required=True)
    p.add_argument('--output-json',type=Path,required=True)
    a=p.parse_args()
    if a.output_json.exists():
        p.error('Require new output path')
    control=json.loads(a.control.read_text()); prototype=json.loads(a.prototype.read_text())
    result=dict(scope=__doc__,control=summarize(control),prototype=summarize(prototype),
        same_realized_executions=control['realized_execution']['sessions']==prototype['realized_execution']['sessions'],
        provenance=[dict(path=str(path),sha256=sha256(path.read_bytes()).hexdigest()) for path in (a.control,a.prototype)],
        limitation='One serial run per branch, not a statistically precise overhead estimate. Changed issued prescriptions can change later histories/search paths; internal distance-pass timing isolates this stage.')
    if control['synthetic_sandbox']['config']!=prototype['synthetic_sandbox']['config']:
        raise ValueError('Configuration differs')
    with a.output_json.open('x') as out:
        json.dump(result,out,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
