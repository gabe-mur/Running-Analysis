"""Cheap synthetic falsification of smoothing ALONE; no planner or live writes.

Fourteen-day mean evidence age reuses the existing modeling assumption. Order
one admits mileage immediately; order three is the existing smoother. Neither
kernel is tuned here. Synthetic seven-day generation blocks define known totals;
they are NOT measurement windows. The estimator consumes every past receipt.
"""
from argparse import ArgumentParser
import json
from math import exp, factorial
from pathlib import Path
from statistics import mean


PATTERNS = {
    "four_runs": ((0, .445), (2, .17), (3, .165), (5, .22)),
    "long_shifted_one_day": ((1, .445), (2, .17), (3, .165), (5, .22)),
    "seven_equal_runs": tuple((day, 1 / 7) for day in range(7)),
}


def receipts(volume, growth, pattern, *, easy_only=False):
    result = []
    # Long flat prehistory prevents empty-history initialization from being
    # mistaken for actual growth. Growth begins at day zero.
    for week in range(-40, 18):
        total = volume * (1 + growth) ** max(week, 0)
        for index, (day, share) in enumerate(PATTERNS[pattern]):
            miles = total * share
            if easy_only:
                # Hold long and quality doses; put actual growth in easy volume.
                miles = (volume * share if index in (0, 1) else
                         (total - volume * .615) * share / .385)
            result.append((week * 7 + day + .5, miles))
    for week in range(-40, 18):
        expected = volume * (1 + growth) ** max(week, 0)
        observed = sum(m for t, m in result if week * 7 <= t < (week + 1) * 7)
        assert abs(expected - observed) < 1e-10
    return result


def level(rows, now, order):
    tau = 14 / order
    return sum(miles * 7 * ((now - at) / tau) ** (order - 1)
               * exp(-(now - at) / tau) / (factorial(order - 1) * tau)
               for at, miles in rows if at <= now)


def probe():
    cases = []
    # Evaluation dates only; all preceding receipts remain in the estimator.
    clocks = [70 + index / 4 for index in range(28 * 4)]
    for order in (1, 3):
        for rate in (0., .02, .04):
            for pattern in PATTERNS:
                for easy_only in ((False, True) if pattern == "four_runs" else (False,)):
                    rows = receipts(20., rate, pattern, easy_only=easy_only)
                    levels = [level(rows, at, order) for at in clocks]
                    # One-day changes are a readout of the continuously smoothed
                    # level, not a new actual-volume aggregation window.
                    local = [(level(rows, at, order) / level(rows, at - 1, order)) ** 7 - 1
                             for at in clocks]
                    # Compare the same cadence phase 28 days apart, exclusively
                    # to check long-run slope recovery, not to drive progression.
                    recovered = [(level(rows, at, order) / level(rows, at - 28, order)) ** .25 - 1
                                 for at in clocks]
                    cases.append(dict(order=order, weekly_growth=rate, pattern=pattern,
                        easy_only_growth=easy_only, level_min=min(levels), level_max=max(levels),
                        local_growth_min=min(local), local_growth_max=max(local),
                        local_mean=mean(local), same_phase_growth_mean=mean(recovered),
                        fraction_local_negative=mean(x < -.001 for x in local),
                        fraction_local_above_4pct=mean(x > .041 for x in local)))
        # Scale equivariance and exact physical ledger credit are prerequisites.
        for volume in (5., 20., 60.):
            rows = receipts(volume, .02, "four_runs")
            base = receipts(20., .02, "four_runs")
            assert abs(level(rows, 100, order) / level(base, 100, order) - volume / 20) < 1e-12
        rows = receipts(20., 0., "four_runs")
        at = 100.25
        extra = [*rows, (at, 1.)]
        ledger_before = sum(m for t, m in rows if t <= at)
        assert abs(sum(m for t, m in extra if t <= at) - ledger_before - 1) < 1e-12
    jumps = {str(order): level([*rows, (at, 1.)], at, order) - level(rows, at, order)
             for order in (1, 3)}
    flat = [case for case in cases if case['weekly_growth'] == 0]
    rejected = any(case['local_growth_max'] > .04 or case['local_growth_min'] < -.04
                   for case in flat)
    return dict(status="smoothing_alone_not_promising" if rejected else "requires_more_testing",
        scope="Measurement-only synthetic probe, not a test of every possible continuous estimator",
        memory_mean_days=14, no_fitted_parameters=True, planner_searches=0, live_changed=False,
        production_files_changed=False, immediate_one_extra_mile_level_change=jumps, cases=cases)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = probe()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))
