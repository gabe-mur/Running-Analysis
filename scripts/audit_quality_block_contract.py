"""Bounded Q5 architecture audit; engineering examples, no new run prescriptions."""
from argparse import ArgumentParser
from dataclasses import asdict
from datetime import timedelta
from hashlib import sha256
import json
from math import fsum
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import yaml

from run_analysis.adaptive.quality_block_contract import (
    BlockCommand, BlockRequest, adapt_v2, block_scenario, compile_blocks, forecast_blocks, read_quality_contract,
)
from run_analysis.adaptive.quality_prescription_contract import Effort, Measure, compile_quality, read_prescription
from run_analysis.adaptive.requested_quality_forecast import command_scenario, forecast_prescription
from run_analysis.workload_accounting import Quantity
from scripts.replay_adaptive_search import load_decision
from scripts.research_manifest import freeze_sources


def audit(frozen_commands, baseline, output):
    started = perf_counter()
    inputs = (frozen_commands, baseline/'baseline.json', baseline/'runtime/decision-0011.json',
              baseline/'runtime/scenario.json', baseline/'sources.zip')
    hashes = {str(p):sha256(p.read_bytes()).hexdigest() for p in inputs}
    output.mkdir(parents=True, exist_ok=False)
    with ZipFile(baseline/'sources.zip') as archive:
        config = yaml.safe_load(archive.read('config.yaml'))
    protocol = dict(scope='Architecture engineering controls, NOT measured accuracy or personalized prescriptions',
        checks=('Recompile old commands with unchanged v2 keys, definitions and physical scenarios',
            'Compare old/new physical-query costs at all three frozen scenario points, absolute tolerance1e-10',
            'Exact block provenance, primary units, ordered display and stored/scored targets survive restart',
            'Exercise repeated threshold, higher-effort longer reps, steady/race-pace and heterogeneous blocks'),
        limits=('Synthetic response and explicitly declared engineering paces, not athlete-calibrated new recommendations',
            'No effort-to-pace inference, purpose quota, stage admission, calendar search or deployment',
            'Strides remain an add-on design, not implemented or credited as a full quality session'))
    with (output/'protocol.json').open('x') as stream:
        json.dump(protocol, stream, indent=2)
    manifest = freeze_sources(output/'sources.zip', config=config, entrypoint=__name__,
        inputs=(*inputs, output/'protocol.json'))
    before = json.loads(frozen_commands.read_text())
    old_prescriptions = [read_prescription(r['definition']['numeric_contract']) for r in before]
    at = old_prescriptions[0].request.issued_at
    fit = dict(issued_at=at-timedelta(seconds=1), accounting_key='engineering_units',
        source_keys=('synthetic_response_control',),
        dynamic=dict(intercept=.4, slope=.1, tau_seconds=45., speed_domain=(0.,6.), tau_at_search_boundary=False))
    def forecast(p, old=False):
        return (forecast_prescription if old else forecast_blocks)(p, fit,
            accounting_key='engineering_units', initial_reserve=.45)
    parity = []
    for r, old in zip(before, old_prescriptions):
        recompiled = compile_quality(old.request, easy_pace=old.easy_pace, work_pace=old.work_pace)
        scenarios = {point:[asdict(p) for p in command_scenario(recompiled, point=point)] for point in r['scenarios']}
        if recompiled.key != r['key'] or recompiled.definition() != r['definition'] or scenarios != r['scenarios']:
            raise AssertionError('Existing v2 identity, display or numerical scenario changed')
        new = adapt_v2(old)
        a, b = forecast(old, True), forecast(new)
        differences = {point:b['scenarios'][point]['predicted_load']-a['scenarios'][point]['predicted_load']
                       for point in r['scenarios']}
        if any(abs(delta)>1e-10 for delta in differences.values()) or new.phases != old.phases:
            raise AssertionError('Explicit block adaptation changed existing dose/cost')
        if read_quality_contract(new.definition()['numeric_contract']) != new:
            raise AssertionError('Block restoration changed command')
        parity.append(dict(old_key=old.key, new_key=new.key, family=old.request.family,
            variant=old.request.variant, load_differences=differences, old_identity_unchanged=True))
    easy = old_prescriptions[0].easy_pace
    sustained = next(p.work_pace for p in old_prescriptions if p.request.family=='threshold')
    faster = old_prescriptions[0].work_pace
    steady = next(p.work_pace for p in old_prescriptions if p.request.family=='progression')
    def block(kind, amount, pace, cues, instruction, unit='time'):
        return BlockCommand(kind, Measure(unit, Quantity.point(amount, 'explicit architecture example')),
            Effort(cues, cues, 'purposeful' if kind=='work' else 'easy'), pace, instruction)
    warm = block('warmup', 600, easy, ('Z1','Z2'), 'Warm up easily.')
    cool = block('cooldown', 600, easy, ('Z1','Z2'), 'Cool down easily.')
    rest = block('recovery', 120, easy, ('Z1','Z2'), 'Jog easily')
    threshold = block('work', 360, sustained, ('threshold effort',), 'At threshold effort.')
    high = block('work', 180, faster, ('higher effort',), 'At higher effort, without sprinting.')
    race = block('work', 480, steady, ('declared race-pace scenario',), 'At the declared race-pace scenario.')
    samples = (
        ('threshold', 'cruise_intervals', 'Threshold repetitions', (threshold, rest, threshold, rest, threshold)),
        ('long_intervals', 'higher_effort', 'Higher-effort intervals', (high, rest, high, rest, high, rest, high)),
        ('steady', 'race_pace_blocks', 'Race-pace blocks', (race, rest, race)),
        ('mixed_quality', 'steady_threshold_fast', 'Mixed efforts', (
            block('work', 600, steady, ('steady effort',), 'At steady effort.'),
            block('work', 400, sustained, ('threshold effort',), 'At threshold effort.', 'distance'), rest,
            block('work', 200, faster, ('fast effort',), 'At fast effort, not sprinting.', 'distance'))))
    examples = []
    for family, variant, title, blocks in samples:
        p = compile_blocks(BlockRequest(family, variant, title,
            'intervals' if family=='long_intervals' else 'tempo_threshold', at,
            protocol['scope'], (warm,*blocks,cool)))
        steps = p.steps()
        if abs(sum(s.duration_minutes for s in steps)*60-fsum(s.seconds.estimate for s in p.phases))>1e-9:
            raise AssertionError('Display and scored phase durations disagree')
        if read_quality_contract(p.definition()['numeric_contract']) != p:
            raise AssertionError('Example restart mismatch')
        predicted = forecast(p)
        examples.append(dict(key=p.key, family=family, variant=variant, instructions=[s.instruction for s in steps],
            prescription=p.definition(), physical_nominal=[asdict(s) for s in block_scenario(p)],
            load=predicted['nominal']['predicted_load'], response_diagnostics=predicted['nominal']['diagnostics'],
            forecast_fingerprint=predicted['fingerprint'], planner_influence=False))
    copied = load_decision(baseline/'runtime',11)
    unchanged = all(sha256(Path(p).read_bytes()).hexdigest()==v for p,v in hashes.items())
    if not unchanged:
        raise AssertionError('Frozen input mutation')
    report = dict(version='quality_block_architecture_audit_v1', protocol=protocol, manifest=manifest,
        parity=parity, engineering_examples=examples, archived_cost=copied['event']['cost'],
        repriced_cost=copied['archived_cost'], frozen_inputs_unchanged=unchanged,
        search_performed=False, live_write=False, elapsed_seconds=perf_counter()-started)
    with (output/'report.json').open('x') as stream:
        json.dump(report, stream, indent=2, default=str, allow_nan=False)
    return report


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument('--before', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    r = audit(args.before, args.baseline, args.output)
    print(json.dumps(dict(parity_commands=len(r['parity']), examples=len(r['engineering_examples']),
        max_load_difference=max(abs(d) for row in r['parity'] for d in row['load_differences'].values()),
        archived_cost=r['archived_cost'], repriced_cost=r['repriced_cost'],
        inputs_unchanged=r['frozen_inputs_unchanged'], seconds=r['elapsed_seconds'])))


if __name__ == '__main__':
    main()
