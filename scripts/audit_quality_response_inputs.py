"""Q5 matched-input attribution, not new predictor fitting or acceptance.

Factorial uses one prior response fit and identical input/HR masks. Mean/recorded
speed and segment-initial HR are completed-run diagnostics, NEVER pre-run inputs.
No invented speed in gaps; compare gap-reset effects separately from first HR.
"""
from argparse import ArgumentParser
from collections import Counter
from dataclasses import replace
from datetime import datetime
from math import fsum, isclose
import json
from pathlib import Path
from statistics import mean
from zipfile import ZipFile

import yaml

from run_analysis.hr_response import ResponseInterval, predict_response
from run_analysis.phase_calibration import mask_trace
from run_analysis.signal_structure import SignalSpan
from run_analysis.adaptive.direct_quality_load import score_direct
from scripts.audit_direct_quality_load import DirectAudit
from scripts.compare_requested_quality_response import trace_metrics
from scripts.audit_work_stimulus import encode
from scripts.research_manifest import freeze_sources
from scripts.validate_quality_load import audit, INITIAL_RESERVE

SPEEDS = ('prior', 'phase_mean', 'recorded')
INITIALS = ('fixed', 'measured')


def diagnostic_grid(rows, layout, *, step=.5):
    """Split known input intervals at phase boundaries; never fill a gap.

    Every cell uses the same grid and segment boundaries. Actual HR interpolation
    is only for the initial point of a contiguous segment, never a phase reset.
    """
    if not rows or not layout:
        return ()
    if step <= 0:
        raise ValueError('Positive diagnostic integration step required')
    pieces = []
    for row in rows:
        for phase_index, (_, a, b) in enumerate(layout):
            left, right = max(row.start, a), min(row.end, b)
            if right <= left:
                continue
            at = left
            while at < right:
                end = min(right, at+step)
                if end <= at:
                    raise ValueError('Diagnostic step below time precision')
                first = row.first+(row.last-row.first)*(at-row.start)/(row.end-row.start)
                last = row.first+(row.last-row.first)*(end-row.start)/(row.end-row.start)
                pieces.append((phase_index, ResponseInterval(at, end, row.speed, first, last)))
                at = end
    return tuple(pieces)


def response_cells(rows, layout, prior_program, model):
    if len(prior_program) != len(layout) or any(abs(seconds-(b-a)) > 1e-6
            for (seconds, _), (_, a, b) in zip(prior_program, layout)):
        raise ValueError('Prior query must match the exact diagnostic phase layout')
    grid = diagnostic_grid(rows, layout)
    if not grid:
        return None
    phase_speeds = {}
    for index in range(len(layout)):
        known = [(r.end-r.start, r.speed) for i, r in grid if i == index]
        if known:
            phase_speeds[index] = fsum(seconds*speed for seconds, speed in known)/fsum(s for s, _ in known)
    segments = [dict(start=r.start, observed_initial_reserve=r.first)
        for j, (_, r) in enumerate(grid) if j == 0 or abs(r.start-grid[j-1][1].end)>1e-8]
    result = {}
    for speed in SPEEDS:
        for initial in INITIALS:
            inputs = tuple(replace(r, speed=(prior_program[i][1] if speed=='prior'
                else phase_speeds[i] if speed=='phase_mean' else r.speed),
                first=INITIAL_RESERVE if initial=='fixed' else r.first) for i, r in grid)
            spans, diagnostics = predict_response(inputs, model)
            if diagnostics['segment_initializations'] != len(segments):
                raise AssertionError('Phase/sampling boundary cannot reset response')
            result[speed+'_'+initial] = (spans, diagnostics)
    return dict(cells=result, phase_mean_speeds=phase_speeds, segments=segments,
        input_support=tuple(SignalSpan(r.start, r.end, 0., 0.) for _, r in grid),
        known_input_seconds=fsum(r.end-r.start for _, r in grid))


class InputAudit:
    def __init__(self):
        self.direct = DirectAudit()

    def __call__(self, case, run, layout, actual, rows, predictions, donors, scale, history):
        self.direct(case, run, layout, actual, rows, predictions, donors, scale, history)
        program = case.get('programs', {}).get('model')
        case['input_attribution'] = dict(status=case['status'], cells={}, prospective=False)
        if not program or not rows or not layout or case['fit']['dynamic'] is None:
            return
        result = response_cells(rows, layout, program, case['fit']['dynamic'])
        if result is None:
            case['input_attribution']['status'] = 'no_known_response_inputs'
            return
        actual = mask_trace(actual, result['input_support'])
        cells = {name:dict(metrics=trace_metrics(spans, actual, layout, scale), diagnostics=diag)
                 for name, (spans, diag) in result['cells'].items()}
        cells['continuous_prior_fixed'] = dict(metrics=trace_metrics(predictions['model'][0], actual, layout, scale),
            diagnostics=predictions['model'][1],
            reset_policy='Original uninterrupted declared programme; never reinitializes after observed input gaps')
        seconds = {round(c['metrics']['whole']['seconds'], 6) for c in cells.values()
                   if c['metrics']['whole'] is not None}
        if len(seconds) != 1 or len([c for c in cells.values() if c['metrics']['whole']]) != 7:
            raise AssertionError('All controls must use exactly the same observed mask')
        direct = score_direct(case['direct_load']['forecast'], actual)
        if direct and not isclose(direct['seconds'], next(iter(seconds)), abs_tol=1e-6):
            raise AssertionError('Baseline mask mismatch')
        attribution = dict(status='diagnosed', cells=cells, same_mask_direct=direct,
            segments=result['segments'], phase_mean_speeds=result['phase_mean_speeds'],
            known_input_seconds=result['known_input_seconds'], original_hr_seconds=case['known_hr_seconds'],
            matched_hr_seconds=next(iter(seconds)), prospective=False,
            source_keys=case['fit']['source_keys'], uses_realized_timing=True,
            speed_mean_basis='Elapsed weighted over known input intervals only; no pace in gaps')
        if all(c['metrics']['whole']['fraction_error'] is not None for c in cells.values()):
            errors = {name:c['metrics']['whole']['fraction_error'] for name,c in cells.items()}
            contrasts = {
                'gap_reset_and_grid_bridge':errors['prior_fixed']-errors['continuous_prior_fixed'],
                'mean_pace_at_fixed_initial':errors['phase_mean_fixed']-errors['prior_fixed'],
                'pace_variation_at_fixed_initial':errors['recorded_fixed']-errors['phase_mean_fixed'],
                'measured_initial_at_recorded_pace':errors['recorded_measured']-errors['recorded_fixed'],
                'total_original_to_recorded_measured':errors['recorded_measured']-errors['continuous_prior_fixed']}
            if not isclose(fsum(v for k,v in contrasts.items() if not k.startswith('total_')),
                    contrasts['total_original_to_recorded_measured'], abs_tol=1e-10):
                raise AssertionError('Signed path attribution must telescope')
            attribution['signed_prediction_contrasts'] = contrasts
        case['input_attribution'] = attribution


def summarize(cases):
    names = tuple(s+'_'+i for s in SPEEDS for i in INITIALS)+('continuous_prior_fixed',)
    paired = [c for c in cases if c['input_attribution']['status']=='diagnosed'
        and all(c['input_attribution']['cells'][n]['metrics']['whole']['fraction_error'] is not None for n in names)
        and c['input_attribution']['same_mask_direct'] is not None]
    def errors(rows):
        return dict(n=len(rows), mae=mean(abs(r['fraction_error']) for r in rows) if rows else None,
            bias=mean(r['fraction_error'] for r in rows) if rows else None)
    result = dict(total=len(cases), paired=len(paired), paired_keys=[c['key'] for c in paired],
        statuses=dict(Counter(c['input_attribution']['status'] for c in cases)), cells={},
        direct=errors([c['input_attribution']['same_mask_direct'] for c in paired]))
    for name in names:
        rows = [c['input_attribution']['cells'][name]['metrics'] for c in paired]
        phases = {}
        for kind in ('warmup', 'work', 'recovery', 'cooldown'):
            known = [r['phases'].get(kind) for r in rows]
            phases[kind] = errors([r for r in known if r and r['fraction_error'] is not None])
        result['cells'][name] = dict(whole=errors([r['whole'] for r in rows]), phases=phases)
    result['contrasts'] = {}
    if paired:
        for name in paired[0]['input_attribution']['signed_prediction_contrasts']:
            changes = [c['input_attribution']['signed_prediction_contrasts'][name] for c in paired]
            result['contrasts'][name] = dict(mean_signed_prediction_change=mean(changes),
                mean_absolute_prediction_change=mean(abs(v) for v in changes))
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
    protocol = dict(version='q5_matched_input_factorial_v1',
        cells=dict(speed=SPEEDS, initial=INITIALS),
        fixed=('One prior30-day response fit per origin, existing tau grid and load integral',
            'Same known speed/HR grid and observed masks in every control; no gap filling',
            'First-order state never resets at phase boundaries; all grid cells have identical segment reset points',
            'Separate original continuous-programme control attributes the reset/grid bridge',
            'Phase-mean speed weighted by known input seconds; temporal variation is a separate control'),
        interpretation=('All measured-speed/initial-HR controls are completed-input diagnostics, NEVER prospective forecasts',
            'Report whole and work/recovery/bookend MAE/bias, contiguous and gapped subgroups separately',
            'Signed prediction changes telescope; changes in absolute error are NOT additive causal shares',
            'Retain all36 cases and unsupported/invalid outcomes, no output clipping or weight/window selection'),
        limits=('Previously inspected development data, current HR settings/labels and completion-time availability assumption',
            'No new effort-to-pace mapping, model fitting, scientific confidence or all-menu predictive acceptance',
            'No calendar search, production change, recovery inference or deployment'))
    with (args.output/'protocol.json').open('x') as stream:
        json.dump(protocol, stream, indent=2)
    manifest = freeze_sources(args.output/'sources.zip', config=config, entrypoint=__name__,
        inputs=(database, args.baseline_root/'baseline.json', args.output/'protocol.json'))
    result = audit(database, config, args.as_of, case_observer=InputAudit())
    result.update(input_protocol=protocol, manifest=manifest)
    result['input_summary'] = {name:summarize(cases) for name,cases in (
        ('all', result['cases']), ('intervals', [c for c in result['cases'] if c['label']=='intervals']),
        ('sustained', [c for c in result['cases'] if c['label']=='tempo_threshold']),
        ('contiguous', [c for c in result['cases'] if len(c['input_attribution'].get('segments',()))==1]),
        ('gapped', [c for c in result['cases'] if len(c['input_attribution'].get('segments',()))>1]))}
    with (args.output/'comparison.json').open('x') as stream:
        json.dump(result, stream, default=encode, indent=2, allow_nan=False)
    print(json.dumps(dict(seconds=result['elapsed_seconds'],
        summary={name:dict(paired=s['paired'], cells={n:r['whole'] for n,r in s['cells'].items()},
            direct=s['direct'],contrasts=s['contrasts']) for name,s in result['input_summary'].items()}), indent=2))


if __name__=='__main__':
    main()
