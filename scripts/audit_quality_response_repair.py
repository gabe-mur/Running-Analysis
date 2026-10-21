"""Isolate execution versus HR-response errors without modifying v1 forecasts."""
from argparse import ArgumentParser
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean

from run_analysis.config import load_config
from run_analysis.hr_response import predict_speed_program
from run_analysis.phase_calibration import mask_trace
from run_analysis.signal_structure import SignalSpan
from scripts.audit_work_stimulus import encode
from scripts.validate_quality_load import audit, INITIAL_RESERVE, POLICY, matched_metrics
from scripts.session_centered_hr_response import fit_session_centered


def observed_speed_control(program, speed_spans):
    """Actual-speed diagnostic only. Unknown speed retains original forecast.

    No target HR initializations or gap resets. Score on the joint known mask,
    disclose gap imputation, and do not call this an executable forecast.
    """
    boundaries = [0.]
    for seconds, _ in program:
        boundaries.append(boundaries[-1]+seconds)
    elapsed = boundaries[-1]
    if any(s.start < 0 or s.end > elapsed+1e-8 for s in speed_spans):
        raise ValueError('Speed spans must be inside the program')
    if any(b.start < a.end for a,b in zip(speed_spans,speed_spans[1:])):
        raise ValueError('Ordered nonoverlapping speed spans required')
    cuts = sorted(set(boundaries+ [max(0., min(elapsed,t)) for s in speed_spans for t in (s.start,s.end)]))
    rows, known, i, phase = [], 0., 0, 0
    for left,right in zip(cuts,cuts[1:]):
        while phase+1 < len(program) and boundaries[phase+1] <= left:
            phase += 1
        while i < len(speed_spans) and speed_spans[i].end <= left:
            i += 1
        speed = program[phase][1]
        if i < len(speed_spans) and speed_spans[i].start <= left and speed_spans[i].end >= right:
            s = speed_spans[i]
            speed = s.first+(s.last-s.first)*((left+right)/2-s.start)/(s.end-s.start)
            known += right-left
        rows.append((right-left,speed))
    return rows, elapsed-known


def diagnosis(case, run, layout, actual, response_rows, predictions, donors, scale, response_history=()):
    if not predictions:
        return
    speeds = tuple(SignalSpan(r.start,r.end,r.speed,r.speed) for r in response_rows)
    program, imputed = observed_speed_control(case['programs']['model'],speeds)
    pred, diag = predict_speed_program(program,case['fit']['dynamic'],initial_reserve=INITIAL_RESERVE)
    elapsed = (run.end-run.start).total_seconds()
    common = mask_trace(actual,speeds)
    case['execution_diagnostic'] = dict(speed_unknown_seconds=imputed, diagnostics=diag,
        forecast_speed=matched_metrics(predictions['model'][0],common,elapsed,scale),
        actual_speed=matched_metrics(pred,common,elapsed,scale),
        phases={kind: dict(
            forecast_speed=matched_metrics(mask_trace(predictions['model'][0],tuple(SignalSpan(a,b,0,0) for k,a,b in layout if k==kind)),common,elapsed,scale),
            actual_speed=matched_metrics(mask_trace(pred,tuple(SignalSpan(a,b,0,0) for k,a,b in layout if k==kind)),common,elapsed,scale))
            for kind in ('work','nonwork')})


from run_analysis.phase_partitions import detailed_layout


def measured_phase_speeds(layout, speed_spans):
    result = {}
    for kind in ('warmup','recovery','cooldown'):
        spans = mask_trace(speed_spans,tuple(SignalSpan(a,b,0,0) for k,a,b in detailed_layout(layout) if k==kind))
        known = sum(s.end-s.start for s in spans)
        result[kind] = dict(known_seconds=known,
            speed=sum((s.end-s.start)*(s.first+s.last)/2 for s in spans)/known if known else None)
    return result


def phase_pace_program(layout, original, source_keys, history, as_of):
    """Only replace pooled non-work speed. Same donors, work pace and timing."""
    if len(layout) != len(original):
        raise ValueError('Aligned original program required')
    accepted = [history[k] for k in source_keys]
    if any(d['end'] >= as_of or d['available_at'] > as_of or
           not 0 < (as_of-d['start']).total_seconds() <= POLICY.current_days*86400 for d in accepted):
        raise ValueError('Prior current-window phase evidence required')
    result, sources = [], {}
    for (kind,a,b),(seconds,speed) in zip(detailed_layout(layout),original):
        if abs(seconds-(b-a)) > 1e-6:
            raise ValueError('Phase timing changed')
        if kind != 'work':
            keys = [k for k in source_keys if history[k]['phases'][kind]['speed'] is not None]
            # New structure is permitted. If a phase has no recent observations,
            # retain the original pooled non-work expectation, explicitly flagged.
            if keys:
                speed = mean(history[k]['phases'][kind]['speed'] for k in keys)
            sources[kind] = dict(source_keys=keys,fallback_pooled_nonwork=not bool(keys))
        result.append((seconds,speed))
    return result,sources


class PhasePaceRepair:
    def __init__(self):
        self.history = {}

    def __call__(self,case,run,layout,actual,response_rows,predictions,donors,scale,response_history=()):
        diagnosis(case,run,layout,actual,response_rows,predictions,donors,scale)
        speeds = tuple(SignalSpan(r.start,r.end,r.speed,r.speed) for r in response_rows)
        if predictions:
            program,sources = phase_pace_program(layout,case['programs']['model'],case['source_keys'],self.history,run.start)
            pred,diag = predict_speed_program(program,case['fit']['dynamic'],initial_reserve=INITIAL_RESERVE)
            elapsed = (run.end-run.start).total_seconds()
            case['phase_pace_repair'] = dict(program=program,sources=sources,diagnostics=diag,
                metrics=matched_metrics(pred,actual,elapsed,scale),
                phases={kind: matched_metrics(mask_trace(pred,tuple(SignalSpan(a,b,0,0) for k,a,b in layout if k==kind)),actual,elapsed,scale)
                        for kind in ('work','nonwork')})
        self.history[run.key] = dict(start=run.start,end=run.end,available_at=run.available_at,
            phases=measured_phase_speeds(layout,speeds))


def centered_diagnosis(case,run,layout,actual,response_rows,predictions,donors,scale,response_history):
    diagnosis(case,run,layout,actual,response_rows,predictions,donors,scale)
    if not predictions:
        return
    fitted = fit_session_centered(response_history,as_of=run.start,
        accounting_key=case['fit']['accounting_key'],lookback_days=POLICY.current_days)
    if fitted['source_keys'] != case['fit']['source_keys']:
        raise AssertionError('Response comparisons must use identical sources')
    case['centered_fit'] = fitted
    if fitted['dynamic'] is None:
        return
    pred,diag = predict_speed_program(case['programs']['model'],fitted['dynamic'],initial_reserve=INITIAL_RESERVE)
    elapsed = (run.end-run.start).total_seconds()
    case['centered_repair'] = dict(diagnostics=diag,metrics=matched_metrics(pred,actual,elapsed,scale),
        phases={kind:matched_metrics(mask_trace(pred,tuple(SignalSpan(a,b,0,0) for k,a,b in layout if k==kind)),actual,elapsed,scale)
                for kind in ('work','nonwork')})


def repair_summary(cases, key='phase_pace_repair'):
    supported = [c for c in cases if c.get(key)]
    paired = [c for c in supported if c[key]['metrics'] and c['model']
        and c[key]['metrics']['fraction_error'] is not None and c['model']['fraction_error'] is not None
        and c['baseline_fraction_error'] is not None
        and not c[key]['diagnostics']['hr_outside_reference']
        and not c['diagnostics']['model']['hr_outside_reference']]
    return dict(total=len(cases),supported=len(supported),paired=len(paired),
        original_mae=mean(abs(c['model']['fraction_error']) for c in paired) if paired else None,
        repaired_mae=mean(abs(c[key]['metrics']['fraction_error']) for c in paired) if paired else None,
        baseline_mae=mean(abs(c['baseline_fraction_error']) for c in paired) if paired else None,
        repaired_hr_rmse=mean(c[key]['metrics']['hr_rmse_bpm'] for c in paired) if paired else None,
        repaired_work_bias=mean(c[key]['phases']['work']['fraction_error'] for c in paired) if paired else None,
        wins_vs_original=sum(abs(c[key]['metrics']['fraction_error']) < abs(c['model']['fraction_error']) for c in paired),
        invalid_prediction_cases=[c['key'] for c in supported if c[key]['diagnostics']['hr_outside_reference']],
        fallback_cases=[c['key'] for c in supported if any(v['fallback_pooled_nonwork'] for v in c[key].get('sources',{}).values())])


def summary(cases):
    rows = [c['execution_diagnostic'] for c in cases if c.get('execution_diagnostic')]
    paired = [r for r in rows if all(r[k] and r[k]['fraction_error'] is not None for k in ('forecast_speed','actual_speed'))
              and not r['diagnostics']['hr_outside_reference']]
    return dict(total=len(cases),diagnosed=len(rows),paired=len(paired),
        methods={name: dict(mae=mean(abs(r[name]['fraction_error']) for r in paired) if paired else None,
            hr_rmse=mean(r[name]['hr_rmse_bpm'] for r in paired) if paired else None,
            work_bias=mean(r['phases']['work'][name]['fraction_error'] for r in paired) if paired else None)
            for name in ('forecast_speed','actual_speed')},
        imputed_speed_seconds=sum(r['speed_unknown_seconds'] for r in rows))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--phase-pace-repair',action='store_true')
    mode.add_argument('--centered-response-repair',action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output path')
    result = audit(Path('data/run_analysis.sqlite'),load_config(Path('config.yaml')),
        datetime(2026,9,26,4,tzinfo=timezone.utc),case_observer=PhasePaceRepair() if args.phase_pace_repair else
            centered_diagnosis if args.centered_response_repair else diagnosis)
    result['execution_diagnostic_summary'] = {name:summary(rows) for name,rows in (
        ('all',result['cases']),('intervals',[c for c in result['cases'] if c['label']=='intervals']),
        ('tempo_threshold',[c for c in result['cases'] if c['label']=='tempo_threshold']))}
    result['protocol'].append('Separate actual-speed control: actual speed where known, prior forecast in speed gaps, fixed initial HR; score original/control on same joint HR/speed mask')
    if args.phase_pace_repair or args.centered_response_repair:
        key = 'phase_pace_repair' if args.phase_pace_repair else 'centered_repair'
        result['repair_summary'] = {name:repair_summary(rows,key) for name,rows in (
            ('all',result['cases']),('intervals',[c for c in result['cases'] if c['label']=='intervals']),
            ('tempo_threshold',[c for c in result['cases'] if c['label']=='tempo_threshold']))}
        result['protocol'].append('Repair: separate prior warmup/recovery/cooldown pace; same contributing sessions, work pace and HR fit; explicit pooled fallback for unseen phases; no weight/window tuning'
            if args.phase_pace_repair else 'Repair: per-training-session intercepts, shared within-session speed slope/tau; forecast uses mean PRIOR intercept; same original pace inputs/window/grid and source runs')
        result['repair_directional_check_passed'] = all(s['paired'] > 0 and not s['invalid_prediction_cases'] and s['repaired_mae'] < min(s['original_mae'],s['baseline_mae'])
            for s in result['repair_summary'].values())
    with args.output.open('x') as stream:
        json.dump(result,stream,default=encode,indent=2,allow_nan=False)
    print(json.dumps(result['execution_diagnostic_summary'],indent=2))
    if args.phase_pace_repair or args.centered_response_repair:
        print(json.dumps(dict(repair=result['repair_summary'],passed=result['repair_directional_check_passed']),indent=2))


if __name__ == '__main__':
    main()
