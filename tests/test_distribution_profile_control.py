from datetime import date, timedelta

import pytest

from scripts.check_easy_distribution_profiles import Profile, allocations, patterns, run, steady_cost
from run_analysis.adaptive.program_objective_trial import (
    DemandScenario, EasyDurationPolicy, TrialPolicy, TrialSession, components,
)
from run_analysis.workload_accounting import Quantity


@pytest.mark.parametrize('units,pace', [(50, 12), (50, 18), (77, 11), (390, 8), (390, 11)])
def test_allocations_are_unique_exact_tenths_with_only_declared_floor(units, pace):
    for count in range(1, 8):
        rows = allocations(units, count, pace)
        assert len(rows) == len(set(rows))
        for row in rows:
            assert len(row) == count
            assert sum(row) == units
            assert min(row)*pace/10 >= 10-1e-9


def test_profiles_do_not_add_fake_low_volume_key_sessions():
    low = Profile('low', 5, 12, 0, 0, 0, 3)
    high = Profile('high', 60, 8, 14, 7, 20, 4)
    for p in (low, high):
        for pattern in patterns(p):
            assert sum(m for _, m in pattern) == pytest.approx(p.support)
            assert len({day for day, _ in pattern}) == len(pattern)
            if p.long:
                assert not {0, 4}.intersection(day for day, _ in pattern)


@pytest.mark.parametrize('single,coefficient', [(False, .1), (True, -.1), (True, float('nan')), (True, float('inf'))])
def test_invalid_curve_options_rejected_before_creating_artifacts(tmp_path, single, coefficient):
    output = tmp_path/'diagnostic'
    with pytest.raises(ValueError):
        run(tmp_path/'missing', output, single_volume=single, volume_linear=coefficient)
    assert not output.exists()


@pytest.mark.parametrize('compression', [0., .3])
def test_periodic_demand_matches_existing_component_ledger(compression):
    day = date(2026, 1, 1)
    pattern = ((0, 1.3), (2, 1.7), (5, 2.))
    demand = lambda role, miles: tuple(Quantity.point(v, 'test') for v in (miles*12, miles*15, 0))
    scenario = DemandScenario(timing_compression_days=compression)
    scales = (40, 50, 10)
    short = EasyDurationPolicy(10, 25, .01, 'test', 5)
    rows = tuple(TrialSession(str(n), day+timedelta(days=n), m, 'easy', demand('easy', m),
        source='observed' if n < 0 else 'planned')
        for n in range(-140, 28) for offset, m in pattern if n%7 == offset)
    old = components(tuple(r for r in rows if r.day < day), tuple(r for r in rows if r.day >= day),
        stage_start=day, decision=day, end=day+timedelta(days=28), scales=scales,
        policy=TrialPolicy(), scenario=scenario)
    control = steady_cost(pattern, demand=demand, scales=scales, scenario=scenario,
                         exchange=.01, short_policy=short)
    assert control['fatigue'] == pytest.approx(.01*old['demand_cost'], abs=1e-12)
    shifted = tuple(((d+3)%7, m) for d, m in pattern)
    assert steady_cost(shifted, demand=demand, scales=scales, scenario=scenario,
                       exchange=.01, short_policy=short) == pytest.approx(control)
