"""Fixed-protocol direct load comparison on the complete quality inventory."""
from argparse import ArgumentParser
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
from statistics import mean

from run_analysis.config import load_config
from scripts.audit_work_stimulus import encode
from scripts.direct_quality_load import PhaseHistory, phase_observations, forecast_direct, score_direct
from scripts.validate_quality_load import audit


class DirectAudit:
    def __init__(self):
        self.history = []

    def __call__(self,case,run,layout,actual,response_rows,predictions,donors,scale,response_history):
        key = case['fit']['accounting_key']
        observations=phase_observations(layout,actual)
        for name,conditioned in (('direct_load',True),('unconditioned_phase_load',False)):
            forecast = forecast_direct(layout,self.history,as_of=run.start,accounting_key=key,conditioned=conditioned)
            case[name] = dict(forecast=forecast,metrics=score_direct(forecast,actual),
                phases={kind:score_direct(forecast,actual,kind=kind) for kind in ('work','nonwork')})
            # Report separate recovery/cooldown errors as well as grouped
            # nonwork: signed errors in those phases can otherwise cancel.
            case[name]['detailed_phase_errors']=phase_errors(forecast,observations)
            if case.get('source_keys') is not None and set(forecast['source_keys']) != set(case['source_keys']):
                raise AssertionError('Unexpected change to preceding contributor cohort')
        # No held-out load enters history before prediction and scoring.
        self.history.append(PhaseHistory(run.key,run.start,run.end,run.available_at,key,observations))


def phase_errors(forecast,observations):
    """Fixed realized layout / identical known-HR time; missing is not zero."""
    if forecast['status']!='predicted':
        return None
    if len(forecast['phases'])!=len(observations) or any(p['kind']!=o.kind
            or abs(p['end']-p['start']-o.seconds)>1e-6 for p,o in zip(forecast['phases'],observations)):
        raise ValueError('Phase diagnostic must use the exact forecast layout')
    result={}
    for kind in ('warmup','work','recovery','cooldown'):
        pairs=[(p,o) for p,o in zip(forecast['phases'],observations) if o.kind==kind]
        if not pairs:
            continue
        known=sum(o.known_seconds for _,o in pairs)
        actual=sum(o.load or 0 for _,o in pairs)
        expected=sum(p['rate']*o.known_seconds for p,o in pairs)
        result[kind]=dict(known_seconds=known,elapsed_seconds=sum(o.seconds for _,o in pairs),
            observed_load=actual if known else None,predicted_load=expected if known else None,
            signed_load_error=expected-actual if known else None,
            fraction_error=expected/actual-1 if actual>0 else None)
    return result


def summary(cases):
    original = [c for c in cases if c.get('model') and c['model']['fraction_error'] is not None
        and c['baseline_fraction_error'] is not None and not c['diagnostics']['model']['hr_outside_reference']]
    paired = [c for c in original if all(c[k]['metrics'] and c[k]['metrics']['fraction_error'] is not None
        for k in ('direct_load','unconditioned_phase_load'))]
    result = dict(total=len(cases),original_supported=len(original),paired=len(paired),
        statuses=dict(Counter(c['direct_load']['forecast']['status'] for c in cases)),
        original_mae=mean(abs(c['model']['fraction_error']) for c in paired) if paired else None,
        baseline_mae=mean(abs(c['baseline_fraction_error']) for c in paired) if paired else None)
    for name in ('direct_load','unconditioned_phase_load'):
        result[name] = dict(mae=mean(abs(c[name]['metrics']['fraction_error']) for c in paired) if paired else None,
            bias=mean(c[name]['metrics']['fraction_error'] for c in paired) if paired else None,
            phase_mae={kind:mean(abs(c[name]['phases'][kind]['fraction_error']) for c in paired) if paired else None for kind in ('work','nonwork')},
            phase_bias={kind:mean(c[name]['phases'][kind]['fraction_error'] for c in paired) if paired else None for kind in ('work','nonwork')},
            wins_vs_original=sum(abs(c[name]['metrics']['fraction_error']) < abs(c['model']['fraction_error']) for c in paired),
            fallback_cases=[c['key'] for c in cases if any(p['fallback_pooled_nonwork'] for p in c[name]['forecast']['phases'])])
        result[name]['detailed_phases']={}
        for kind in ('warmup','work','recovery','cooldown'):
            rows=[c[name].get('detailed_phase_errors',{}).get(kind) for c in paired]
            rows=[r for r in rows if r is not None and r['fraction_error'] is not None]
            result[name]['detailed_phases'][kind]=dict(n=len(rows),
                mae=mean(abs(r['fraction_error']) for r in rows) if rows else None,
                bias=mean(r['fraction_error'] for r in rows) if rows else None)
    return result


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output path')
    result = audit(Path('data/run_analysis.sqlite'),load_config(Path('config.yaml')),
        datetime(2026,9,26,4,tzinfo=timezone.utc),case_observer=DirectAudit())
    result['direct_load_summary'] = {label:summary(rows) for label,rows in (
        ('all',result['cases']),('intervals',[c for c in result['cases'] if c['label']=='intervals']),
        ('tempo_threshold',[c for c in result['cases'] if c['label']=='tempo_threshold']))}
    result['direct_directional_check_passed'] = all(s['paired']==s['original_supported'] and s['paired'] > 0
        and s['direct_load']['mae'] < min(s['original_mae'],s['baseline_mae']) for s in result['direct_load_summary'].values())
    result['direct_protocol'] = [
        'Primary: directly predict integrated phase load from preceding HR integrals; no constructed HR curve, HR lag, target pace or target HR',
        'Same 30-day window and original contributors; one session vote; no label/template or exact-repeat gate',
        'Existing 1/(1+log(duration ratio)^2) similarity extended by adding preceding-phase log-duration ratio squared, untuned equal log scales',
        'Warmup/work/recovery/cooldown evidence; pooled nonwork fallback only when the requested nonwork phase has no evidence; missing work remains unknown',
        'Control: identical phase decomposition without duration/preceding-duration conditioning, not another tuned candidate',
        'Primary gate: full original supported cohort and lower total-load MAE than original AND simple baseline overall and in both quality categories',
        'Report work/nonwork errors separately; matched known-HR masks; original-file missingness and cases preserved',
        'Phase source dispersion is descriptive, not a calibrated prediction interval',
        'Conditional on recorded phase timing; historical settings/labels, assumed completion availability; not prospective or independent validation',
        'No weight/window search, actual accounting change, scheduler integration or recovery/benefit claim']
    with args.output.open('x') as stream:
        json.dump(result,stream,default=encode,indent=2,allow_nan=False)
    print(json.dumps(dict(summary=result['direct_load_summary'],passed=result['direct_directional_check_passed'],seconds=result['elapsed_seconds']),indent=2))


if __name__ == '__main__':
    main()
