"""Frozen effort-reference diagnostic. No search, integration or target fitting.

All quality rows stay in the report. Saved effort and realized timing are separate
inputs: this is conditional load attribution, NOT a prospective accuracy trial.
The all-steady path on repetitions is an explicitly mismatched-semantics control.
"""
from argparse import ArgumentParser
from datetime import datetime
import json
from pathlib import Path
import sqlite3
from zipfile import ZipFile

import yaml

from run_analysis.adaptive.effort_pace_reference import EffortEvidence, ExecutionPace, effort_reference
from run_analysis.adaptive.requested_quality_forecast import SpeedPhase, forecast_physical
from run_analysis.movement import classify_movement
from run_analysis.phase_partitions import detailed_layout
from run_analysis.projection_evidence import steady_pace_samples
from run_analysis.training_signals import HeartRateReference
from scripts.audit_direct_quality_load import DirectAudit
from scripts.audit_issued_quality_load import matched_assignments, issue_boundary
from scripts.audit_work_stimulus import encode
from scripts.compare_requested_quality_response import saved_effort, summarize, trace_metrics
from scripts.research_manifest import freeze_sources
from scripts.validate_quality_load import audit, INITIAL_RESERVE


def work_mode(payload):
    # Explicit prescription structure, not family/title or target trace duration.
    work = next((s for s in payload.get('structure', ()) if s.get('phase') == 'work'), {})
    return 'commanded_execution' if (work.get('repetitions') or 1) > 1 else 'steady_exposure'


class Comparison:
    def __init__(self, assignments, config):
        self.assignments, self.config = assignments, config
        self.direct = DirectAudit()
        self.history, self.pending = [], {}
        self.hr = HeartRateReference(config['resting_hr'], config['max_hr'], 'frozen exposure normalization')

    def query(self, case, run, layout, actual, scale, issued, *, force_steady):
        if issued is None:
            return dict(status='unknown_historical_intent', metrics=None)
        result = dict(status='unresolved_effort_pace', metrics=None, profiles=[],
            saved_issue=issued['issue'], uses_realized_timing=True, prospective=False,
            uses_target_speed=False, uses_target_hr_for_pace=False,
            limits='Run-start history/fit, saved intent, realized timing; not issuance-time prediction')
        if not layout or case['fit']['dynamic'] is None:
            return dict(result, status='no_response_or_alignment')
        phases = []
        for kind,a,b in detailed_layout(layout):
            effort = saved_effort(issued['payload'],kind)
            if effort is None:
                return dict(result, status='unknown_saved_effort')
            mode = work_mode(issued['payload']) if kind == 'work' and not force_steady else 'steady_exposure'
            reference = effort_reference(self.history, effort, mode=mode, config=self.config,
                as_of=run.start, accounting_key=case['fit']['accounting_key'], bout_seconds=b-a)
            result['profiles'].append(dict(kind=kind, reference=reference))
            if reference['profile'] is None:
                return dict(result, status=reference['status'])
            p = reference['profile']
            phases.append(SpeedPhase(kind,b-a,p.first_speed.estimate,p.last_speed.estimate,p.basis))
        forecast = forecast_physical(phases, case['fit'], issued_at=run.start,
            accounting_key=case['fit']['accounting_key'], initial_reserve=INITIAL_RESERVE)
        return dict(result, status=forecast['status'], forecast=forecast,
            metrics=trace_metrics(forecast.get('normalized_spans',()),actual,layout,scale))

    def __call__(self,case,run,layout,actual,rows,predictions,donors,scale,history):
        self.direct(case,run,layout,actual,rows,predictions,donors,scale,history)
        issued = self.assignments.get(run.key)
        case['effort_aware_scenario'] = self.query(case,run,layout,actual,scale,issued,force_steady=False)
        case['steady_exposure_control'] = self.query(case,run,layout,actual,scale,issued,force_steady=True)
        # Candidate queries are complete before target execution is extracted.
        if issued is not None:
            effort = saved_effort(issued['payload'],'work')
            exposures = []
            windows = [(a,b) for kind,a,b in layout if kind == 'work']
            for bout,(a,b) in zip(run.bouts,windows):
                covered = sum(max(0.,min(b,r.end)-max(a,r.start)) for r in rows)
                speed = (bout.distance_meters/(b-a) if bout.distance_meters is not None else
                    sum(max(0.,min(b,r.end)-max(a,r.start))*r.speed for r in rows)/covered
                    if covered > 0 and abs(covered-(b-a)) < 1e-5 else None)
                if speed is not None and speed > 0:
                    exposures.append(ExecutionPace(b-a,speed,effort))
            self.pending[run.key] = tuple(exposures)

    def activity(self,row,activity,accounting_key,eligible):
        intervals = classify_movement(activity.trackpoints,self.config['moving_time']).intervals
        steady = tuple(steady_pace_samples(intervals,activity.start_time_utc,self.config,self.hr))
        execution = self.pending.pop(row.key,())
        issued = self.assignments.get(row.key)
        self.history.append(EffortEvidence(row.key,row.start,row.end,row.available_at,accounting_key,
            steady,execution,issued['issue'] if issued else None,eligible))


def assert_parent_unchanged(result, parent):
    prior = {c['key']:c for c in parent['cases']}
    if set(prior) != {c['key'] for c in result['cases']}:
        raise AssertionError('Changed original quality inventory')
    fields = ('status','model','baseline_fraction_error','fit','source_keys','programs','pace_references')
    def canonical(value):
        return json.loads(json.dumps(value,default=encode,allow_nan=False))
    for row in result['cases']:
        for name in fields:
            if canonical(row.get(name)) != prior[row['key']].get(name):
                raise AssertionError(f'Changed frozen baseline {row["key"]}/{name}')


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument('--baseline-root',type=Path,required=True)
    p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--as-of',type=datetime.fromisoformat,required=True)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    database = args.baseline_root/'evidence.sqlite'
    with ZipFile(args.baseline_root/'sources.zip') as archive:
        config = yaml.safe_load(archive.read('config.yaml'))
    protocol = dict(version='q5_effort_reference_diagnostic_v1',
        hypotheses=('Measured steady exposure must not pool unrelated hard work or invert HR equilibrium',
            'Repeated-work commands use independently saved effort, not instantaneous HR to select execution pace',
            'Transfer across duration/structure is allowed; unknown historical intent remains unknown',
            'Report an all-steady control separately: its semantics do NOT validate short-effort execution'),
        fixed=('Existing 30-day window, original eligibility and configured steady-window validity rules',
            'One session ownership; geometric means; existing log-duration relevance for known executions',
            'Same original response fit and fixed initial HRR .45; same matched observed HR masks',
            'Full numeric zones for measured exposure; upper/low refinements not assigned invented fractions',
            'No coefficient fitting/sweep, target pace/HR used in reference queries, search or integration'),
        gate='Engineering separation/causality and unchanged Q4 baseline. Sparse intent is NOT a predictive acceptance gate',
        limits=('Previously inspected development data; four labelled-quality saved-intent cases, not a fresh holdout',
            'Current labels and HR normalization; activity availability assumed at completion',
            'Saved commands do not prove adherence; query at run start and realized timing, not true issuance',
            'Measured steady pace is not a response to a requested effort or a calibrated prediction interval',
            'Linear progression endpoint path declared, not learned; standing/nonwork execution remains unresolved',
            'Historical unknown intent cannot validate full-menu prediction; explicit uncertain scenarios still permitted',
            'No live deployment, scoring/fatigue changes, calendar search or daily stability claim'))
    with (args.output/'protocol.json').open('x') as stream:
        json.dump(protocol,stream,indent=2)
    manifest = freeze_sources(args.output/'sources.zip',config=config,entrypoint=__name__,
        inputs=(database,args.parent,args.output/'protocol.json'))
    assignments = {}
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as db:
        for key,pid,saved,payload,started,*_ in matched_assignments(db,args.as_of):
            payload = json.loads(payload)
            assignments[str(key)] = dict(prescription_id=pid,payload=payload,
                issue=issue_boundary(saved,payload,datetime.fromisoformat(started)))
    comparison = Comparison(assignments,config)
    result = audit(database,config,args.as_of,case_observer=comparison,activity_observer=comparison.activity)
    assert_parent_unchanged(result,json.loads(args.parent.read_text()))
    result.update(protocol=protocol,manifest=manifest,original_baseline_unchanged=True,
        evidence_inventory=dict(sessions=len(comparison.history),
            steady_sessions=sum(bool(r.steady_samples) for r in comparison.history),
            known_command_execution_sessions=sum(bool(r.executions) for r in comparison.history)))
    result['effort_summary'] = {name:summarize(result['cases'],name)
        for name in ('effort_aware_scenario','steady_exposure_control')}
    with (args.output/'comparison.json').open('x') as stream:
        json.dump(result,stream,default=encode,indent=2,allow_nan=False)
    print(json.dumps(dict(summary=result['effort_summary'],inventory=result['evidence_inventory'],
        seconds=result['elapsed_seconds']),indent=2))


if __name__ == '__main__':
    main()
