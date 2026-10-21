"""One predeclared Q5 elapsed-response candidate; no tuning or deployment."""
from argparse import ArgumentParser
from collections import Counter
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
from statistics import mean
from zipfile import ZipFile

import yaml

from run_analysis.phase_calibration import mask_trace
from run_analysis.signal_structure import SignalSpan
from scripts.audit_quality_response_inputs import diagnostic_grid
from scripts.audit_work_stimulus import encode
from scripts.compare_requested_quality_response import trace_metrics
from scripts.elapsed_hr_response import ElapsedEvidence, elapsed_statistics, fit_elapsed_response, predict_elapsed_response
from scripts.research_manifest import freeze_sources
from scripts.validate_quality_load import audit, POLICY, INITIAL_RESERVE


class Comparison:
    def __init__(self, previous):
        self.previous = {c['key']:c for c in previous['cases']}
        self.history = []

    def observe(self, row, rows, accounting_key, eligible):
        self.history.append(ElapsedEvidence(row.key, row.end, row.available_at, accounting_key,
            elapsed_statistics(rows), (min(r.speed for r in rows),max(r.speed for r in rows)),
            (min(r.start for r in rows)/60,max(r.end for r in rows)/60), eligible=eligible))

    def __call__(self, case, run, layout, actual, rows, predictions, donors, scale, history):
        old = self.previous[case['key']]
        for key in ('fit','programs','pace_references','layout','status'):
            if json.loads(json.dumps(case.get(key), default=encode)) != old.get(key):
                raise AssertionError('Existing causal baseline changed: '+key)
        fitted = fit_elapsed_response(self.history, as_of=run.start,
            accounting_key=case['fit']['accounting_key'], lookback_days=POLICY.current_days)
        if fitted['source_keys'] != case['fit']['source_keys']:
            raise AssertionError('Elapsed candidate changed prior contributor cohort')
        base = old['input_attribution']
        result = dict(fit=fitted, baseline_status=base['status'], status=base['status'],
            conditional_recorded=None, conditional_prior_pace=None, prospective=False)
        if base['status']=='diagnosed':
            if fitted['dynamic'] is None:
                result['status']='no_identified_elapsed_response'
            else:
                grid = diagnostic_grid(rows, layout)
                support = tuple(SignalSpan(r.start,r.end,0.,0.) for _,r in grid)
                actual = mask_trace(actual, support)
                program = case['programs']['model']
                inputs = {
                    'conditional_recorded':tuple(r for _,r in grid),
                    'conditional_prior_pace':tuple(replace(r, speed=program[i][1],first=INITIAL_RESERVE) for i,r in grid)}
                for name, command in inputs.items():
                    pred, diag = predict_elapsed_response(command, fitted['dynamic'])
                    metrics = trace_metrics(pred, actual, layout, scale)
                    old_name = 'recorded_measured' if name=='conditional_recorded' else 'prior_fixed'
                    old_metrics = base['cells'][old_name]['metrics']
                    if metrics['whole']['seconds'] != old_metrics['whole']['seconds']:
                        raise AssertionError('Old/new candidate must share identical known masks')
                    result[name] = dict(metrics=metrics, diagnostics=diag, original=old_metrics)
                result['status']='compared'
        case['elapsed_candidate'] = result


def summarize(cases):
    baseline = [c for c in cases if c['elapsed_candidate']['baseline_status']=='diagnosed']
    compared = [c['elapsed_candidate'] for c in baseline if c['elapsed_candidate']['status']=='compared']
    result = dict(total=len(cases), baseline_count=len(baseline), compared=len(compared),
        statuses=dict(Counter(c['elapsed_candidate']['status'] for c in cases)), methods={})
    for name in ('conditional_recorded','conditional_prior_pace'):
        known = [r[name] for r in compared]
        out = dict(invalid_hr_count=sum(r['diagnostics']['hr_outside_reference'] for r in known), phases={})
        for phase in ('whole','warmup','work','recovery','cooldown'):
            pairs = [(r['original']['whole'],r['metrics']['whole']) if phase=='whole' else
                     (r['original']['phases'].get(phase),r['metrics']['phases'].get(phase)) for r in known]
            valid = [(a,b) for a,b in pairs if a and b and a['fraction_error'] is not None and b['fraction_error'] is not None]
            expected = sum(a is not None and a['fraction_error'] is not None for a,b in pairs)
            out['phases'][phase] = dict(baseline_count=expected, valid_pairs=len(valid),
                original_mae=mean(abs(a['fraction_error']) for a,b in valid) if valid else None,
                candidate_mae=mean(abs(b['fraction_error']) for a,b in valid) if valid else None,
                original_bias=mean(a['fraction_error'] for a,b in valid) if valid else None,
                candidate_bias=mean(b['fraction_error'] for a,b in valid) if valid else None)
        result['methods'][name]=out
    return result


def diagnostic_gate(summaries):
    """Conditional component gate only; not pre-run/calendar acceptance."""
    failures = []
    for group in ('all','intervals','sustained'):
        s = summaries[group]
        if not s['baseline_count'] or s['compared']!=s['baseline_count']:
            failures.append(group+':lost conditional coverage')
        m = s['methods']['conditional_recorded']
        if m['invalid_hr_count']:
            failures.append(group+':invalid HR')
        for phase,p in m['phases'].items():
            if p['valid_pairs']!=p['baseline_count']:
                failures.append(group+':'+phase+':lost known load')
            if p['valid_pairs'] and p['candidate_mae']>p['original_mae']+1e-12:
                failures.append(group+':'+phase+':higher MAE')
            if phase in ('whole','work') and (not p['valid_pairs'] or p['candidate_mae']>=p['original_mae']):
                failures.append(group+':'+phase+':no improvement')
    return dict(passed=not failures, failures=failures,
        scope='Conditional HR-response component only; execution pace and prospective prediction remain open')


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument('--baseline-root',type=Path,required=True)
    p.add_argument('--matched-inputs',type=Path,required=True)
    p.add_argument('--as-of',type=datetime.fromisoformat,required=True)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    previous = json.loads(args.matched_inputs.read_text())
    with ZipFile(args.baseline_root/'sources.zip') as archive:
        config = yaml.safe_load(archive.read('config.yaml'))
    args.output.mkdir(parents=True,exist_ok=False)
    protocol = dict(version='q5_elapsed_response_comparison_v1',
        hypothesis='One learned elapsed-time equilibrium covariate separates persistent time-related residuals from the prior speed response',
        fixed=('Same prior30-day completed/available eligible sources, equal-session ownership, original HR normalization/load accounting',
            'Same existing41-point tau grid, positive speed slope, no sign restriction on learned elapsed coefficient',
            'No family/phase coefficients, weight/window/grid sweep, target-fitted baseline or actual HR teacher forcing',
            'Time ramp is elapsed minutes from session origin, continuous through all phases; missing-input gaps reset HR only',
            'Exact ramp-input integration, with old endpoint-moment estimation retained; zero elapsed term reproduces old physics',
            'Identical known-time masks for old/new recorded-input and separate unchanged-prior-pace controls'),
        gate=('Full old30-case conditional coverage, no invalid HR, lower whole/work MAE overall and in both historical categories',
            'No worsening MAE in other observed phase aggregates in those three groups; no post-result family hybrid'),
        limits=('This linear elapsed approximation is an empirical candidate, not a demonstrated drift mechanism or physiology law',
            'Speed/conditions/grade/time remain confounded; raw speed and elapsed-domain extrapolation explicitly reported',
            'Recorded-input component pass is not acceptance of predicted pace, saved-intent accuracy, recovery or calendar scoring',
            'Previously inspected development history, current settings/labels, assumed completion availability; no statistical confidence claim',
            'Unsupported cases retained; no output clipping, load reinterpretation, calendar search or live writes'))
    with (args.output/'protocol.json').open('x') as stream:
        json.dump(protocol,stream,indent=2)
    manifest = freeze_sources(args.output/'sources.zip', config=config, entrypoint=__name__,
        inputs=(args.baseline_root/'evidence.sqlite', args.matched_inputs,args.output/'protocol.json'))
    comparison = Comparison(previous)
    result = audit(args.baseline_root/'evidence.sqlite',config,args.as_of,
        case_observer=comparison,response_observer=comparison.observe)
    result.update(elapsed_protocol=protocol,manifest=manifest)
    result['elapsed_summary']={name:summarize(rows) for name,rows in (
        ('all',result['cases']),('intervals',[c for c in result['cases'] if c['label']=='intervals']),
        ('sustained',[c for c in result['cases'] if c['label']=='tempo_threshold']))}
    result['elapsed_gate']=diagnostic_gate(result['elapsed_summary'])
    with (args.output/'comparison.json').open('x') as stream:
        json.dump(result,stream,indent=2,default=encode,allow_nan=False)
    print(json.dumps(dict(seconds=result['elapsed_seconds'],summary=result['elapsed_summary'],gate=result['elapsed_gate']),indent=2))


if __name__=='__main__':
    main()
