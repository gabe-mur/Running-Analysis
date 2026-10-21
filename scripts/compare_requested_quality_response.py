"""Frozen Q5 attribution: unchanged baseline, speed oracle and saved effort.

Not calendar acceptance or a prospective trial. Unknown historical intention
stays unknown. Recorded-speed diagnostics never construct the intent query.
"""
from argparse import ArgumentParser
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import sqlite3
from statistics import mean
from zipfile import ZipFile

import yaml

from run_analysis.adaptive.quality_prescription_contract import Effort
from run_analysis.adaptive.requested_quality_forecast import SpeedPhase, effort_pace, forecast_physical
from run_analysis.hr_response import predict_response
from run_analysis.phase_partitions import detailed_layout
from run_analysis.phase_calibration import clip_trace
from scripts.audit_direct_quality_load import DirectAudit
from scripts.audit_issued_quality_load import matched_assignments, issue_boundary
from scripts.audit_work_stimulus import encode
from scripts.research_manifest import freeze_sources
from scripts.validate_quality_load import audit, matched_metrics, INITIAL_RESERVE


def trace_metrics(predicted, actual, layout, scale):
    """Score curves on identical observed masks, not average phase rates."""
    elapsed = layout[-1][2]
    result = dict(whole=matched_metrics(predicted, actual, elapsed, scale), phases={})
    for kind in ('warmup', 'work', 'recovery', 'cooldown'):
        pairs = [matched_metrics(clip_trace(predicted, a, b), clip_trace(actual, a, b), elapsed, scale)
                 for k, a, b in detailed_layout(layout) if k == kind]
        pairs = [p for p in pairs if p is not None]
        if not pairs:
            continue
        observed = sum(p['observed_load'] for p in pairs)
        valid = all(p['predicted_load'] is not None for p in pairs)
        expected = sum(p['predicted_load'] for p in pairs) if valid else None
        result['phases'][kind] = dict(known_seconds=sum(p['seconds'] for p in pairs),
            observed_load=observed, predicted_load=expected,
            fraction_error=expected/observed-1 if valid and observed > 0 else None)
    return result


def saved_effort(payload, kind):
    """Read explicit cues only; never infer intention from quality labels/HR.

    Legacy progression's literal gradual/finishing-Z3 instruction identifies
    endpoint order. 'Upper/low' refinements are not assigned invented fractions.
    Legacy recovery/unprescribed finish has no independent cues: the easy
    opening is a disclosed scenario, not evidence of what the athlete did.
    """
    steps = payload.get('structure', [])
    work = next((s for s in steps if s.get('phase') == 'work'), None)
    if work is None or not steps:
        return None
    # A two-step progression ends in WORK, not a cooldown instruction. Never
    # recycle that work cue as a forecast of detected trailing nonwork.
    finish = steps[-1] if steps[-1].get('phase') != 'work' else steps[0]
    step = work if kind == 'work' else finish if kind == 'cooldown' else steps[0]
    cues = tuple(step.get('target_zones') or ())
    text = step.get('instruction', '').lower()
    if kind == 'work' and 'gradually' in text and 'finishing controlled in z3' in text and set(cues) == {'Z2', 'Z3'}:
        return Effort(('Z2',), ('Z3',), 'gradual')
    return Effort(cues, cues, 'purposeful' if kind == 'work' else 'easy') if cues else None


class Comparison:
    def __init__(self, assignments, config):
        self.direct = DirectAudit()
        self.assignments = assignments
        self.config = config

    def __call__(self, case, run, layout, actual, rows, predictions, donors, scale, history):
        self.direct(case, run, layout, actual, rows, predictions, donors, scale, history)
        model = case['fit']['dynamic']
        if rows and model is not None and layout:
            spans, diag = predict_response(rows, model)
            case['recorded_speed_diagnostic'] = dict(status='conditional_recorded_speed',
                diagnostics=diag, metrics=trace_metrics(spans, actual, layout, scale),
                uses_target_speed=True, uses_target_segment_initial_hr=True,
                prospective=False)
        else:
            case['recorded_speed_diagnostic'] = dict(status='no_response_or_alignment', metrics=None)
        program = case.get('programs', {}).get('model')
        case['shared_api_control'] = dict(status=case['status'], metrics=None)
        if program:
            phases = tuple(SpeedPhase(k, seconds, speed, speed, 'unchanged prior-only Q4 pace')
                for (k, _, _), (seconds, speed) in zip(detailed_layout(layout), program))
            if len(phases) != len(layout):
                raise AssertionError('Unchanged layout required')
            control = forecast_physical(phases, case['fit'], issued_at=run.start,
                accounting_key=case['fit']['accounting_key'], initial_reserve=INITIAL_RESERVE)
            metrics = trace_metrics(control.get('normalized_spans', ()), actual, layout, scale)
            old = case.get('model')
            if old and old['predicted_load'] is not None:
                new = metrics['whole']['predicted_load']
                if new is None or abs(new-old['predicted_load']) > 1e-7:
                    raise AssertionError('Shared API changed unchanged physical programme cost')
            case['shared_api_control'] = dict(status=control['status'], metrics=metrics,
                diagnostics=control.get('diagnostics'), source_keys=control['source_keys'])
        issued = self.assignments.get(run.key)
        case['requested_effort_scenario'] = dict(status='unknown_historical_intent', metrics=None)
        if issued is None or not layout:
            return
        query = dict(status='unresolved_requested_effort', metrics=None, profiles=[],
            saved_issue=issued['issue'], prescription_id=issued['prescription_id'],
            response_fit_origin=run.start, uses_realized_timing=True, prospective=False,
            limits='Saved intent, run-start fit, realized timing; NOT actual-issue availability or execution validation')
        phases = []
        for kind, a, b in detailed_layout(layout):
            effort = saved_effort(issued['payload'], kind)
            if effort is None:
                case['requested_effort_scenario'] = query
                return
            pace = effort_pace(effort, case['fit'], self.config, issued_at=run.start)
            query['profiles'].append(dict(kind=kind, effort=effort, result=pace,
                cue_basis='saved work cues' if kind == 'work' else
                    'saved easy bookend or declared easy-opening scenario for detected nonwork'))
            if pace['profile'] is None:
                query['status'] = pace['status']
                case['requested_effort_scenario'] = query
                return
            p = pace['profile']
            phases.append(SpeedPhase(kind, b-a, p.first_speed.estimate, p.last_speed.estimate, p.basis))
        forecast = forecast_physical(phases, case['fit'], issued_at=run.start,
            accounting_key=case['fit']['accounting_key'], initial_reserve=INITIAL_RESERVE)
        query.update(status=forecast['status'], forecast=forecast,
            metrics=trace_metrics(forecast.get('normalized_spans', ()), actual, layout, scale))
        case['requested_effort_scenario'] = query


def summarize(cases, name):
    usable = [c for c in cases if c[name].get('metrics') and c[name]['metrics']['whole']
              and c[name]['metrics']['whole']['fraction_error'] is not None]
    def errors(rows):
        return dict(n=len(rows), mae=mean(abs(r['fraction_error']) for r in rows) if rows else None,
            bias=mean(r['fraction_error'] for r in rows) if rows else None)
    result = dict(total=len(cases), statuses=dict(Counter(c[name]['status'] for c in cases)),
        whole=errors([c[name]['metrics']['whole'] for c in usable]), phases={})
    for kind in ('warmup', 'work', 'recovery', 'cooldown'):
        rows = [c[name]['metrics']['phases'].get(kind) for c in usable]
        result['phases'][kind] = errors([r for r in rows if r and r['fraction_error'] is not None])
    # Compare the existing baseline only over the same support/time mask.
    paired = [c for c in usable if c['direct_load']['metrics']
              and c['direct_load']['metrics']['fraction_error'] is not None
              and abs(c[name]['metrics']['whole']['seconds']-c['direct_load']['metrics']['seconds']) < 1e-6]
    result['paired_current_direct'] = dict(n=len(paired),
        current_mae=mean(abs(c['direct_load']['metrics']['fraction_error']) for c in paired) if paired else None,
        candidate_mae=mean(abs(c[name]['metrics']['whole']['fraction_error']) for c in paired) if paired else None)
    return result


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument('--baseline-root', type=Path, required=True)
    p.add_argument('--as-of', type=datetime.fromisoformat, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    database = args.baseline_root/'evidence.sqlite'
    with ZipFile(args.baseline_root/'sources.zip') as archive:
        config = yaml.safe_load(archive.read('config.yaml'))
    protocol = dict(version='q5_requested_response_attribution_v1',
        hypotheses=('Shared API must preserve Q4 unchanged-programme costs to 1e-7 load units',
            'Recorded-speed/segment-initial-HR control diagnoses response limits, NOT a pre-run candidate',
            'Saved effort zones invert prior affine equilibrium; declared full-band centers/linear progression scenario',
            'Report all historical cases and unknown intention; no target HR/speed enters saved-effort query'),
        fixed=('Existing 30-day causal response fit, tau search and fixed initial HRR .45',
            'Matched observed HR masks; phase curves integrated BEFORE averaging',
            'No residual calibration, family coefficient, window/weight sweep or source database mutation'),
        gate='Mechanics/parity only. No prospective accuracy acceptance from sparse intent or recorded-speed oracle',
        limits=('Previously inspected development data, not fresh holdout',
            'Current historical labels/HR settings; source availability assumed at completion',
            'Intent query uses realized timing and run-start fit, not issuance-time availability',
            'Raw-speed response has no explicit terrain, weather or cardiovascular drift term',
            'No search, planner change, recovery inference, deployment or confidence calibration'))
    with (args.output/'protocol.json').open('x') as stream:
        json.dump(protocol, stream, indent=2)
    manifest = freeze_sources(args.output/'sources.zip', config=config, entrypoint=__name__,
        inputs=(database, args.baseline_root/'baseline.json', args.output/'protocol.json'))
    assignments = {}
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True) as db:
        for key, pid, saved, payload, started, *_ in matched_assignments(db, args.as_of):
            payload = json.loads(payload)
            assignments[str(key)] = dict(prescription_id=pid, payload=payload,
                issue=issue_boundary(saved, payload, datetime.fromisoformat(started)))
    result = audit(database, config, args.as_of, case_observer=Comparison(assignments, config))
    result.update(protocol=protocol, manifest=manifest, saved_assignments=len(assignments))
    result['q5_summary'] = {label: {name:summarize(rows, name) for name in (
        'shared_api_control', 'recorded_speed_diagnostic', 'requested_effort_scenario')}
        for label, rows in (('all', result['cases']),
            ('intervals', [c for c in result['cases'] if c['label']=='intervals']),
            ('tempo_threshold', [c for c in result['cases'] if c['label']=='tempo_threshold']))}
    with (args.output/'comparison.json').open('x') as stream:
        json.dump(result, stream, default=encode, indent=2, allow_nan=False)
    print(json.dumps(dict(summary=result['q5_summary'], seconds=result['elapsed_seconds']), indent=2))


if __name__ == '__main__':
    main()
