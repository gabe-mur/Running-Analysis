"""Q3 command/target audit on the frozen Q1 stage; no search or predictor fit.

Paces below are explicitly declared engineering scenarios, NOT new personalized
pace recommendations. Q5 owns the causal requested-effort response validation.
The existing archived calendar is independently repriced, never rerolled.
"""
from argparse import ArgumentParser
from datetime import datetime, timedelta
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import yaml

from run_analysis.adaptive.quality_dose_policy import (
    QualityDosePolicy, QualityDoseStage, compile_dose_menu, purpose_contract,
)
from run_analysis.adaptive.quality_prescription_contract import PaceReference
from run_analysis.workload_accounting import Quantity
from scripts.audit_work_stimulus import encode
from scripts.replay_adaptive_search import load_decision
from scripts.research_manifest import freeze_sources


def audit(baseline, output):
    started = perf_counter()
    original = baseline/'baseline.json'
    frozen = json.loads(original.read_text())
    root = baseline/'runtime'
    decision = root/frozen['decision']
    event = json.loads(decision.read_text())
    inputs = (original, decision, root/event['snapshot_file'], root/'scenario.json', baseline/'sources.zip')
    digests = {str(p): sha256(p.read_bytes()).hexdigest() for p in inputs}
    output.mkdir(parents=True, exist_ok=False)
    with ZipFile(baseline/'sources.zip') as source:
        config = yaml.safe_load(source.read('config.yaml'))
    manifest = freeze_sources(output/'sources.zip', config=config, entrypoint=__name__, inputs=inputs)
    case = load_decision(root, event['index']+1)
    at = datetime.fromisoformat(frozen['as_of'])
    old = case['catalog']['quality:ordinary']
    minutes = sum(seconds for kind, seconds in old.phases if kind == 'work')/60
    s = frozen['stage']
    stage = QualityDoseStage(s['key'], datetime.fromisoformat(s['issued_at']), s['ordinary_miles'],
        minutes, 7/case['state'].context.policy.quality_per_7,
        'Frozen Q1 accepted stage; quality dose from its exact ordinary work phases')
    policy = QualityDosePolicy()
    def pace(speed, purpose):
        basis = 'Declared Q3 engineering scenario for '+purpose+'; not a fitted prediction or confidence interval'
        return PaceReference(Quantity(.9/speed, 1.1/speed, 1/speed, basis),
            at-timedelta(seconds=1), ('q3-scenario:'+purpose,), basis)
    easy = pace(2.5, 'easy')
    purposes = {purpose: pace(speed, purpose) for purpose, speed in
        (('faster', 3.7), ('sustained', 3.), ('gradual', 2.8))}
    contract = purpose_contract(stage, config=config, issued_at=at,
        easy_pace=easy, faster_pace=purposes['faster'], policy=policy)
    menu = compile_dose_menu(stage, contract=contract, easy_pace=easy, purpose_paces=purposes,
        policy=policy, open_recovery_seconds=90,
        scenario_source='Declared 90-second open-recovery scenario; execution remains by feel')
    rows = []
    for p in menu['prescriptions']:
        duration = sum(phase.seconds.estimate for phase in p.phases)/60
        work = sum(phase.seconds.estimate for phase in p.phases if phase.kind == 'work')/60
        displayed = sum(step.duration_minutes for step in p.steps())
        if abs(duration-displayed) > 1e-9:
            raise AssertionError('Displayed command and compiled dose disagree')
        fraction = min(1., minutes/policy.reference_work_minutes)
        if 'reduced' in p.request.variant or p.request.family == 'progression' and p.miles != stage.ordinary_miles:
            fraction *= policy.reduced_fraction
        allocated = (None if p.request.family == 'progression' else
            (18 if p.request.family == 'threshold' else policy.reference_work_minutes
             if p.request.family == 'long_intervals' else policy.faster_reference_minutes
             *(.75 if p.request.family == 'hill_repeats' else 1.))*fraction)
        rows.append(dict(key=p.key, family=p.request.family, variant=p.request.variant,
            primary_units=[w.unit for w in p.request.work], repetitions=len(p.request.work),
            work_minutes=work, native_budget_minutes=allocated,
            command_quantization_minutes=work-allocated if allocated is not None else None,
            running_minutes=duration, committed_miles=p.miles,
            open_recovery=p.request.open_recovery, steps=[step.model_dump() for step in p.steps()],
            prescription=p.definition()))
    missing = set(frozen['enabled_menu'])-{r['family'] for r in rows}
    if missing:
        raise AssertionError('Missing named recipe families: '+str(missing))
    unchanged = all(sha256(Path(path).read_bytes()).hexdigest() == value for path, value in digests.items())
    if not unchanged:
        raise AssertionError('Frozen inputs changed')
    report = dict(version='quality_q3_dose_audit_v1', as_of=at, manifest=manifest,
        stage=stage, policy=policy, contract=contract, menu_key=menu['key'], commands=rows,
        pace_scenarios=dict(easy=easy, **purposes),
        old_quality_command=dict(miles=old.miles, work_minutes=minutes, phases=old.phases),
        original_cost=frozen['archived_cost'], repriced_cost=case['archived_cost'],
        archived_candidate_eligible=True, frozen_inputs_unchanged=unchanged,
        unresolved_families=menu['unresolved_families'], stage_unsupported_families=menu['stage_unsupported_families'],
        scope='Q3 engineering/coaching contract only; not a run recommendation or forecast acceptance',
        search_performed=False, live_write=False, forecasting_validated=False,
        limitations=('Pace/recovery scenarios are declared, not data-derived current expectations',
            'Configured zone anchors are coaching proxies, not laboratory thresholds',
            'Whole-rep and 30-second quantization can overshoot a native allocation',
            'Q5 shared forecast and Q7 existing stage/program constraints must admit actual candidates',
            'No empirical adaptation equivalence, hill pace transfer or full-menu calendar stability proved'),
        elapsed_seconds=perf_counter()-started)
    with (output/'report.json').open('x') as stream:
        json.dump(report, stream, default=encode, indent=2, allow_nan=False)
    return report


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=Path('output/quality-q1-baseline-20261008'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.baseline, args.output)
    print(json.dumps({k: result[k] for k in ('original_cost', 'repriced_cost',
        'frozen_inputs_unchanged', 'search_performed', 'live_write', 'elapsed_seconds')}))
    print(json.dumps([dict(family=r['family'], variant=r['variant'], work_minutes=r['work_minutes'],
        miles=r['committed_miles']) for r in result['commands']]))


if __name__ == '__main__':
    main()
