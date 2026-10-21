"""Inspect TCX/FIT workload measurements without importing or changing a plan."""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

import yaml

from run_analysis.fit import parse_fit
from run_analysis.tcx import parse_tcx
from run_analysis.workload_profile import observed_profile


def audit(source, config):
    parser = {'.fit': parse_fit, '.tcx': parse_tcx}.get(source.suffix.lower())
    if parser is None:
        raise ValueError('Provide an extracted .fit or .tcx file')
    activities = parser(source).activities
    profiles = []
    for activity in activities:
        stream = activity.sensor_stream
        profiles.append(dict(
            workload=observed_profile(activity, config),
            sensor_inventory=dict(
                rr_batch_count=len(stream.rr_batches) if stream else 0,
                rr_interval_count=sum(x is not None for batch in stream.rr_batches
                                      for x in batch.intervals_seconds) if stream else 0,
                alignment_status=stream.alignment_status if stream else 'unavailable',
                dfa_status=stream.dfa_status if stream else 'unavailable')))
    return dict(source_name=source.name, source_sha256=sha256(source.read_bytes()).hexdigest(),
                activities=profiles, planner_influence=False)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit(args.source, yaml.safe_load(args.config.read_text()))
    payload = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        # Refuse overwrites, including the input and existing audit artifacts.
        with args.output.open('x') as output:
            output.write(payload+'\n')
    else:
        print(payload)


if __name__ == '__main__':
    main()
