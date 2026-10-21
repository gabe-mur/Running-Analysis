"""Q5 pace-source and residual attribution, not a fitted forecasting repair.

All actual-speed/HR/time associations below are held-out-run diagnostics. They
cannot become future inputs, response coefficients, or a correction to load.
"""
from argparse import ArgumentParser
from collections import Counter
from datetime import datetime
import json
from math import fsum, isclose, sqrt
from pathlib import Path
from statistics import mean
from zipfile import ZipFile

import numpy as np
import yaml

from run_analysis.hr_response import predict_response
from run_analysis.phase_calibration import clip_trace, mask_trace
from run_analysis.phase_partitions import detailed_layout
from run_analysis.signal_structure import SignalSpan
from scripts.audit_quality_response_inputs import diagnostic_grid
from scripts.audit_work_stimulus import encode
from scripts.compare_requested_quality_response import trace_metrics
from scripts.research_manifest import freeze_sources
from scripts.validate_quality_load import audit, matched_metrics


def matched_cells(grid, predicted, actual):
    """Joint known support, including exact interpolated endpoints; no gap fill."""
    if len(grid) != len(predicted):
        raise ValueError('One response span per input cell required')
    cells, index = [], 0
    for (_, row), prediction in zip(grid, predicted):
        if (row.start, row.end) != (prediction.start, prediction.end):
            raise ValueError('Response and input boundaries must agree')
        while index < len(actual) and actual[index].end <= row.start:
            index += 1
        j = index
        while j < len(actual) and actual[j].start < row.end:
            a, b = max(row.start, actual[j].start), min(row.end, actual[j].end)
            if b > a:
                p = clip_trace((prediction,), a, b)[0]
                h = clip_trace((actual[j],), a, b)[0]
                cells.append((a, b, row.speed, p.first, p.last, h.first, h.last))
            j += 1
    return tuple(cells)


def residual_associations(cells, scale):
    """Exact duration-weighted linear associations with actual-minus-model HR.

    Joint speed/time coefficients are conditional associations, not evidence of
    causal HR dynamics. Fit on held-out data is diagnostic ONLY; never returned
    as a response model. Two Gauss nodes integrate the quadratic moments exactly
    for the linear endpoints, constant interval speed and elapsed time.
    """
    if not cells:
        return dict(status='no_joint_known_support', known_seconds=0)
    gram, rhs, square, seconds = np.zeros((3, 3)), np.zeros(3), 0., 0.
    for a, b, speed, p0, p1, h0, h1 in cells:
        dt = b-a
        for f in (.5-1/(2*sqrt(3)), .5+1/(2*sqrt(3))):
            x = np.array((1., speed, (a+dt*f)/60))
            y = ((h0-p0)*(1-f)+(h1-p1)*f)*scale
            gram += np.outer(x, x)*(dt/2)
            rhs += x*y*(dt/2)
            square += y*y*(dt/2)
        seconds += dt
    gram, rhs, square = gram/seconds, rhs/seconds, square/seconds
    # Center covariates to distinguish mean residual from slope/time effects.
    transform = np.array(((1., -gram[0, 1], -gram[0, 2]), (0., 1., 0.), (0., 0., 1.)))
    centered, target = transform.T@gram@transform, transform.T@rhs
    rank = int(np.linalg.matrix_rank(centered))
    result = dict(status='identified_diagnostic' if rank == 3 else 'joint_association_unidentified',
        known_seconds=seconds, mean_actual_minus_model_bpm=float(rhs[0]),
        residual_rmse_bpm=sqrt(max(0., square)), rank=rank,
        mean_speed_mps=float(gram[0, 1]), mean_elapsed_minutes=float(gram[0, 2]),
        prospective=False, causal=False)
    if rank == 3:
        coef = np.linalg.solve(centered, target)
        result.update(speed_association_bpm_per_mps=float(coef[1]),
            elapsed_association_bpm_per_minute=float(coef[2]),
            residual_rmse_after_diagnostic_fit_bpm=sqrt(max(0.,
                square-2*coef@target+coef@centered@coef)))
    return result


def phase_halves(predicted, actual, layout, scale):
    """First/last elapsed halves of each bout, then aggregate within source run.

    Half boundaries do not reset predictions. Missing data in either half stays
    unknown. Many repetitions do not count as many independent source sessions.
    """
    elapsed, result = layout[-1][2], {}
    for kind in ('warmup', 'work', 'recovery', 'cooldown'):
        for half in ('first', 'last'):
            windows = []
            for name, a, b in detailed_layout(layout):
                if name == kind:
                    middle = (a+b)/2
                    left, right = (a, middle) if half == 'first' else (middle, b)
                    windows.append(SignalSpan(left, right, 0., 0.))
            if windows:
                result[kind+'_'+half] = matched_metrics(mask_trace(predicted, windows),
                    mask_trace(actual, windows), elapsed, scale)
    return result


def pace_inputs(rows, layout, program):
    inputs = []
    for index, (kind, a, b) in enumerate(detailed_layout(layout)):
        known = [(min(b, r.end)-max(a, r.start), r.speed) for r in rows
                 if min(b, r.end) > max(a, r.start)]
        seconds = fsum(t for t, _ in known)
        measured = fsum(t*v for t, v in known)/seconds if seconds else None
        prior = program[index][1] if program else None
        inputs.append(dict(kind=kind, start=a, end=b, known_seconds=seconds,
            prior_speed_mps=prior, measured_known_mean_speed_mps=measured,
            prior_speed_fraction_error=prior/measured-1 if prior is not None and measured else None))
    return inputs


class BiasAudit:
    def __init__(self, previous):
        self.previous = {c['key']: c for c in previous['cases']}
        self.source_cases = {}

    def __call__(self, case, run, layout, actual, rows, predictions, donors, scale, history):
        previous = self.previous[case['key']]
        # Exact original fits/commands, not just similar average result.
        for key in ('fit', 'programs', 'pace_references', 'layout', 'status'):
            if json.loads(json.dumps(case.get(key), default=encode)) != previous.get(key):
                raise AssertionError('Original causal baseline changed: '+key)
        program = case.get('programs', {}).get('model')
        phase_inputs = pace_inputs(rows, layout, program)
        result = dict(status=case['status'], phase_pace_inputs=phase_inputs,
            pace_reference_accepts_requested_effort=False, prospective=False)
        # Previously completed donor cases only; do not use the target's means.
        result['prior_pace_sources'] = [dict(key=k,
            work_seconds=[b-a for kind, a, b in self.source_cases[k]['layout'] if kind == 'work'],
            saved_match=self.source_cases[k].get('saved_match')) for k in case.get('source_keys', ())]
        if program and rows and case['fit']['dynamic'] is not None:
            grid = diagnostic_grid(rows, layout)
            pred, diag = predict_response(tuple(r for _, r in grid), case['fit']['dynamic'])
            support = tuple(SignalSpan(r.start, r.end, 0., 0.) for _, r in grid)
            actual = mask_trace(actual, support)
            metrics = trace_metrics(pred, actual, layout, scale)
            old = previous['input_attribution']['cells']['recorded_measured']['metrics']
            if metrics != old:
                raise AssertionError('Recorded-input diagnostic no longer matches frozen parent')
            cells = matched_cells(grid, pred, actual)
            if not isclose(fsum(b-a for a, b, *_ in cells), metrics['whole']['seconds'], abs_tol=1e-6):
                raise AssertionError('Residual associations must share the exact known mask')
            result.update(status='diagnosed', metrics=metrics, diagnostics=diag,
                phase_halves=phase_halves(pred, actual, layout, scale),
                associations=residual_associations(cells, scale),
                source_keys=case['fit']['source_keys'])
        case['bias_diagnostic'] = result
        self.source_cases[case['key']] = case


def summarize(cases):
    rows = [c['bias_diagnostic'] for c in cases if c['bias_diagnostic']['status']=='diagnosed']
    result = dict(total=len(cases), diagnosed=len(rows),
        statuses=dict(Counter(c['bias_diagnostic']['status'] for c in cases)), halves={})
    for key in sorted({k for r in rows for k in r['phase_halves']}):
        known = [r['phase_halves'][key] for r in rows if r['phase_halves'].get(key)]
        valid = [r for r in known if r['fraction_error'] is not None]
        result['halves'][key] = dict(n=len(known), valid_load_count=len(valid),
            mean_prediction_minus_actual_hr_bpm=mean(r['hr_bias_bpm'] for r in known),
            mean_load_bias=mean(r['fraction_error'] for r in valid) if valid else None,
            mean_load_mae=mean(abs(r['fraction_error']) for r in valid) if valid else None)
    associations = [r['associations'] for r in rows if r['associations']['status']=='identified_diagnostic']
    result['associations'] = dict(identified=len(associations),
        positive_speed_count=sum(r['speed_association_bpm_per_mps']>0 for r in associations),
        positive_elapsed_count=sum(r['elapsed_association_bpm_per_minute']>0 for r in associations),
        mean_speed_bpm_per_mps=mean(r['speed_association_bpm_per_mps'] for r in associations) if associations else None,
        mean_elapsed_bpm_per_minute=mean(r['elapsed_association_bpm_per_minute'] for r in associations) if associations else None,
        prospective=False, causal=False)
    return result


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument('--baseline-root', type=Path, required=True)
    p.add_argument('--matched-inputs', type=Path, required=True)
    p.add_argument('--as-of', type=datetime.fromisoformat, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    previous = json.loads(args.matched_inputs.read_text())
    with ZipFile(args.baseline_root/'sources.zip') as archive:
        config = yaml.safe_load(archive.read('config.yaml'))
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = dict(version='q5_response_bias_attribution_v1',
        fixed=('Same frozen Q4 history/config/as-of and unchanged prior30-day response/pace construction',
            'Assert exact original fit, commands, pace references, layout/status against matched-input parent',
            'Assert recorded-speed/segment-start-HR whole and phase metrics exactly match frozen parent',
            'First/last elapsed half of each phase, combined within source run; no phase resets or new cutoff',
            'Duration-weighted actual-minus-model HR associations with speed and elapsed minutes, estimated jointly'),
        limits=('Every target-input diagnostic is nonprospective; coefficients never enter forecasts or load accounting',
            'Speed/time associations are not causal identification; environmental, grade and measurement effects remain confounded',
            'Only known speed/HR intervals; no gap filling, output clipping, model/weight/window sweep or significance claim',
            'Retain unknown intent, unsupported cases, sparse interval coverage and any unidentified associations',
            'No calendar search, live writes or release; previously inspected development data'))
    protocol_path = args.output/'protocol.json'
    protocol_path.write_text(json.dumps(protocol, indent=2))
    manifest = freeze_sources(args.output/'sources.zip', config=config, entrypoint=__name__,
        inputs=(args.baseline_root/'evidence.sqlite', args.matched_inputs, protocol_path))
    result = audit(args.baseline_root/'evidence.sqlite', config, args.as_of, case_observer=BiasAudit(previous))
    result.update(bias_protocol=protocol, manifest=manifest)
    result['bias_summary'] = {name:summarize(rows) for name, rows in (
        ('all', result['cases']), ('intervals', [c for c in result['cases'] if c['label']=='intervals']),
        ('sustained', [c for c in result['cases'] if c['label']=='tempo_threshold']),
        ('contiguous', [c for c in result['cases'] if c.get('bias_diagnostic', {}).get('diagnostics', {}).get('segment_initializations')==1]),
        ('gapped', [c for c in result['cases'] if c.get('bias_diagnostic', {}).get('diagnostics', {}).get('segment_initializations',0)>1]))}
    with (args.output/'comparison.json').open('x') as stream:
        json.dump(result, stream, indent=2, default=encode, allow_nan=False)
    print(json.dumps(dict(seconds=result['elapsed_seconds'], summary=result['bias_summary']), indent=2))


if __name__=='__main__':
    main()
