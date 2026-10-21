"""One predeclared shared upgrade on frozen evidence; no weights or search."""
from argparse import ArgumentParser
from datetime import datetime
import json
from pathlib import Path
from statistics import mean

from run_analysis.adaptive.calibrated_quality_load import (
    PhaseCalibrationHistory,calibration_phases,forecast_calibrated)
from run_analysis.adaptive.direct_quality_load import score_direct,phase_observations
from run_analysis.config import load_config
from scripts.audit_direct_quality_load import DirectAudit,phase_errors
from scripts.audit_work_stimulus import encode
from scripts.research_manifest import freeze_sources
from scripts.validate_quality_load import audit,INITIAL_RESERVE


class Comparison:
    def __init__(self):
        self.direct=DirectAudit();self.history=[]

    def __call__(self,case,run,layout,actual,response_rows,predictions,donors,scale,response_history):
        self.direct(case,run,layout,actual,response_rows,predictions,donors,scale,response_history)
        # Programs and response fit already frozen using prior-only evidence.
        # Never use held-out response_rows/speed/HR to construct the query.
        program=case.get('programs',{}).get('model')
        if program is None:
            case['calibrated_load']=dict(forecast=dict(status=case['status']),metrics=None)
        else:
            forecast=forecast_calibrated(layout,program,case['fit']['dynamic'],self.history,
                as_of=run.start,accounting_key=case['fit']['accounting_key'],initial_reserve=INITIAL_RESERVE)
            observations=phase_observations(layout,actual)
            case['calibrated_load']=dict(forecast=forecast,metrics=score_direct(forecast,actual),
                detailed_phase_errors=phase_errors(forecast,observations))
        # Source's original prior-only curve, not a refit containing this source.
        if 'model' in predictions and not case['diagnostics']['model']['hr_outside_reference']:
            phases=calibration_phases(layout,predictions['model'][0],actual)
            self.history.append(PhaseCalibrationHistory(run.key,run.start,run.end,run.available_at,
                case['fit']['accounting_key'],phases))


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument('--database',required=True,type=Path)
    p.add_argument('--as-of',required=True,type=datetime.fromisoformat)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    protocol=dict(version='shared_response_phase_calibration_v1',
        upgrade='Prior-only HR-response curve at requested phase pace, corrected by earlier out-of-sample observed/modelled phase-load ratios',
        fixed_rules=('Existing30-day policy, log-duration similarity, equal session votes and initialHRR.45',
            'Identity correction without residual support; no tuned blend or label-specific branch',
            'Raw invalid HR is rejected, not clipped; same known-HR time masks',
            'All cases retained; evaluate current direct, simple rate and upgrade on identical paired cohort',
            'Improve whole-load MAE over both baselines overall AND both existing quality categories without coverage loss',
            'Report phase errors and signed bias even if total improves'),
        limits=('Realized timing and prior predicted pace, not verified requested-intensity trials',
            'Current labels/HR configuration, historical availability assumed at completion',
            'No benefit scoring, calibration of uncertainty, live integration or scheduler acceptance'))
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    config=load_config()
    manifest=freeze_sources(args.output/'sources.zip',config=config,entrypoint=__name__,inputs=(args.database,args.output/'protocol.json'))
    result=audit(args.database,config,args.as_of,case_observer=Comparison())
    summaries={}
    for label,rows in (('all',result['cases']),('intervals',[c for c in result['cases'] if c['label']=='intervals']),
                       ('tempo_threshold',[c for c in result['cases'] if c['label']=='tempo_threshold'])):
        supported=[c for c in rows if c['direct_load']['metrics'] and c.get('baseline_fraction_error') is not None]
        paired=[c for c in supported if c['calibrated_load']['metrics'] and c['calibrated_load']['metrics']['fraction_error'] is not None]
        def mae(name):return mean(abs(c[name]['metrics']['fraction_error']) for c in paired) if paired else None
        direct,upgrade=mae('direct_load'),mae('calibrated_load')
        simple=mean(abs(c['baseline_fraction_error']) for c in paired) if paired else None
        summaries[label]=dict(total=len(rows),supported=len(supported),paired=len(paired),
            current_direct_mae=direct,simple_rate_mae=simple,upgrade_mae=upgrade,
            upgrade_bias=mean(c['calibrated_load']['metrics']['fraction_error'] for c in paired) if paired else None,
            passed=bool(paired) and len(paired)==len(supported) and upgrade<min(direct,simple),
            detailed_phases={kind:dict(n=len(phase),mae=mean(abs(r['fraction_error']) for r in phase) if phase else None,
                bias=mean(r['fraction_error'] for r in phase) if phase else None)
                for kind in ('warmup','work','recovery','cooldown')
                for phase in [[c['calibrated_load']['detailed_phase_errors'].get(kind) for c in paired]]
                for phase in [[r for r in phase if r is not None and r['fraction_error'] is not None]]})
    result.update(protocol=protocol,manifest=manifest,upgrade_summary=summaries,
                  upgrade_gate_passed=all(s['passed'] for s in summaries.values()))
    with (args.output/'comparison.json').open('x') as stream:
        json.dump(result,stream,default=encode,indent=2,allow_nan=False)
    print(json.dumps(dict(summary=summaries,passed=result['upgrade_gate_passed'],seconds=result['elapsed_seconds']),indent=2))


if __name__=='__main__':main()
