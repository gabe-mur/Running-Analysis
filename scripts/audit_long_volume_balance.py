"""Curve/progression arithmetic only: no search, upload or simulated recovery."""
from argparse import ArgumentParser
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

from run_analysis.long_volume_balance import LongVolumeBalancePolicy, propose_balance_phase
from run_analysis.staged_progression import DevelopmentStage, ProgressionPolicy, _growth_candidate


def audit():
    balance = LongVolumeBalancePolicy()
    origin = datetime(2026, 10, 2, tzinfo=timezone.utc)
    current = DevelopmentStage('arithmetic-only', origin, 20.3, 4.4, 8.9, 17., 7.,
        ProgressionPolicy(14, .05, balance=balance), 'AS109 accepted dose illustration; not an imported stage')
    rows = []
    s = current
    for index in range(80):
        if (s.long_miles >= s.goal_long_miles and
                s.weekly_equivalent_miles >= balance.volume_for_long(s.long_miles)):
            break
        proposal = _growth_candidate(s, s.review_at, s.long_miles, {})
        if proposal['candidate'] is None:
            raise ValueError(f'Arithmetic trajectory stopped: {proposal}')
        next_stage = proposal['candidate']
        rows.append(dict(index=index+1, volume=next_stage.weekly_equivalent_miles,
            long=next_stage.long_miles, extra_nonlong=proposal['extra_nonlong_miles_per_7_days'],
            long_share=next_stage.long_miles/next_stage.weekly_equivalent_miles))
        s = next_stage
    began = perf_counter()
    for _ in range(1000):
        balance.volume_for_long(8.9)
    milliseconds = (perf_counter()-began)*1000/1000
    return dict(policy=asdict(balance), curve=[dict(volume=v,
        long=balance.long_at_volume(v), share=balance.share(v))
        for v in (5., 10., 20., 20.3, 20.9, 30., 40., 60., 80., 100.)],
        current_phase=propose_balance_phase(current),
        volume_for_goal_long=balance.volume_for_long(17.),
        arithmetic_trajectory=rows,
        average_inverse_milliseconds=milliseconds,
        acceptance_scope='Numerical policy/proposal checks only; not a daily/upload planner simulation',
        limitations=['Every arithmetic review ASSUMES funding, response and guard evidence; does not test them',
            'Existing 5%/14-day proposal defaults retained, not validated by this curve',
            '80-mile anchor22%, asymptote20%, low limit45%, shape4 are coaching assumptions',
            'Curve is not a fit to all mileage bins; middle volumes differ from descriptive plan centers',
            'No frequency choice, fatigue change, automatic phase handoff or live mutation'])


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False,
            default=lambda v: asdict(v) if hasattr(v, '__dataclass_fields__') else v.isoformat())
    print(json.dumps(dict(volume_for_current_long=result['current_phase']['required_volume_per_7'],
        volume_for_goal_long=result['volume_for_goal_long'],
        average_inverse_milliseconds=result['average_inverse_milliseconds']), indent=2))


if __name__ == '__main__':
    main()
