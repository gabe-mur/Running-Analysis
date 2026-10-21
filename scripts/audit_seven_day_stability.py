"""Evaluate completed daily/upload replays against a seven-day stability window.

Retain all existing gate thresholds, report every smaller edit too, and do not
confuse a passing four-day gate with seven days of demonstrated stability.
"""
from argparse import ArgumentParser
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path

from pydantic import TypeAdapter
from run_analysis.adherence_projection import ProjectionReplan, summarize_replan_churn
from run_analysis.projection_gate import ProjectionGate, ProjectionGateConfig, _future_distance_changes, _future_schedule_changes
from scripts.audit_backward_controls import plain


def assess(report):
    snapshots=TypeAdapter(list[ProjectionReplan]).validate_python(report['replans'])
    original=ProjectionGateConfig(**report['gate_policy'])
    gate=ProjectionGate(replace(original,stability_horizon_days=7,enforce_stability=True),fail_fast=False)
    for snapshot in snapshots:
        gate.observe(snapshot)
    transitions=[]
    for before,after in zip(snapshots,snapshots[1:]):
        dates,types=_future_schedule_changes(before,after,horizon_days=7)
        doses=_future_distance_changes(before,after,horizon_days=7,tolerance_miles=0.)
        if dates or types or doses:
            transitions.append(dict(at=after.generated_at,trigger=after.trigger,
                material_evidence=list(after.material_evidence_reasons),
                date_changes=dates,type_changes=types,distance_changes=doses))
    daily_dates={s.generated_at.date() for s in snapshots if s.trigger=='scheduled_refresh'}
    consecutive=bool(daily_dates) and (max(daily_dates)-min(daily_dates)).days+1==len(daily_dates)
    complete=(not report['stopped_early'] and report['realized_execution']['complete']
        and len(daily_dates)>=7 and consecutive)
    return dict(scope=__doc__,complete_seven_day_screen=complete,
        daily_dates=sorted(daily_dates),consecutive_daily_refreshes=consecutive,
        original_gate_window=original.stability_horizon_days,
        original_report_failures=report['failures'],seven_day_failures=gate.failures,
        passed_seven_day_gates=complete and not gate.failures,
        zero_seven_day_edits=complete and not transitions,
        daily_decisions=sum(s.trigger=='scheduled_refresh' for s in snapshots),
        upload_decisions=sum(s.trigger=='post_upload' for s in snapshots),
        churn={str(n):asdict(summarize_replan_churn(snapshots,horizon_days=n)) for n in (3,4,7)},
        changed_transitions=transitions,
        max_distance_edit=max((d['movement_miles'] for t in transitions for d in t['distance_changes']),default=0.),
        timing=[dict(at=s.generated_at,seconds=s.planning_seconds,
            distance_refinement=s.planner_diagnostics.get('continuation_selection',{}).get('distance_refinement'))
            for s in snapshots if s.planner_diagnostics],
        limitation='One replay, not population or progression clearance. Uses the existing conservative churn-window definition, including an unconsumed today in addition to the future seven dates.')


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--output-json',type=Path,required=True)
    a=p.parse_args()
    if a.output_json.exists():
        p.error('Require new output path')
    source=a.report.read_bytes()
    result=assess(json.loads(source))
    result.update(source=str(a.report),source_sha256=sha256(source).hexdigest())
    with a.output_json.open('x') as out:
        json.dump(plain(result),out,indent=2,allow_nan=False)
    print(json.dumps(plain({k:result[k] for k in ('complete_seven_day_screen','passed_seven_day_gates',
        'zero_seven_day_edits','seven_day_failures','churn','max_distance_edit')}),indent=2))


if __name__=='__main__':
    main()
