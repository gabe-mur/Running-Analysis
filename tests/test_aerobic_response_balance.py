from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from run_analysis.aerobic_response_evidence import BalanceBlock, candidate_pairs, losses, score, select, blocks_from_intervals
from run_analysis.hr_response_alignment import response_step
from run_analysis.pace_comparability import fit_comparison_calibration
from tests.test_pace_comparability import config
from tests.test_hr_response_alignment import synthetic


def trace(*, tau=60., minutes=30, gain=1., steady=False, slowdown=False):
    state = .6; blocks = []
    for t in range(0, minutes*60, 120):
        demand = 1. if steady else (1.25 if (t//120) % 2 else .8)
        if slowdown and t == 600:
            demand = .35
        mean, end = response_step(state, demand, 120., tau)
        change = end-state
        blocks.append(BalanceBlock(t, 120., (t+60)/60, demand*2.7*gain*120.,
            mean*120., max(0., change), min(0., change), 120.))
        state = end
    return blocks


def test_integral_balance_recovers_unsettled_start_and_short_slowdown():
    for rows in (trace(), trace(slowdown=True), trace(steady=True)):
        out = score(rows, 60, 60)
        assert out['pace_min_mile'] == pytest.approx(1609.344/60/2.7)
        assert out['blocks'] == len(rows)


def test_startup_imbalance_shrinks_with_duration_without_excluding_warmup():
    bias = [abs(score(trace(steady=True, minutes=n))['pace_min_mile']-1609.344/60/2.7) for n in (10, 20, 40)]
    assert bias[0] > bias[1] > bias[2]
    for n in (10, 20, 40):
        assert score(trace(steady=True, minutes=n), 60, 60)['pace_min_mile'] == pytest.approx(1609.344/60/2.7)


def test_boundary_terms_telescope_and_allow_either_direction():
    rows = trace()
    first = sum(b.rise+b.fall for b in rows)
    assert first != 0
    positive = score([BalanceBlock(0, 120, 1, 324, 120, .2, 0, 120)], 60, 60)
    negative = score([BalanceBlock(0, 120, 1, 324, 120, 0, -.2, 120)], 60, 60)
    assert positive['response_fraction'] == pytest.approx(.1)
    assert negative['response_fraction'] == pytest.approx(-.1)


def test_true_response_selected_without_target_score():
    pairs = candidate_pairs()
    rows = trace()
    costs = losses(rows, pairs)
    assert pairs[int(np.argmin(costs))] == (60., 60.)
    assert min(costs) < 1e-20
    trials = {1: costs, 2: costs, 3: costs, 4: list(reversed(costs))}
    assert pairs[select(trials, 4, pairs)] == (60., 60.)


def test_real_speed_gains_preserved_without_mutating_data():
    rows = trace(); before = deepcopy(rows)
    for pair in ((0, 0), (60, 60), (30, 90)):
        base = score(rows, *pair)
        for gain in (.97, 1.03):
            changed = score([replace(b, effort_distance=b.effort_distance*gain) for b in rows], *pair)
            assert base['pace_min_mile']/changed['pace_min_mile'] == pytest.approx(gain)
    assert rows == before


def test_all_readable_moving_data_retained_including_high_hr(config):
    intervals, _ = synthetic(config, steady=True, steady_speed=3.8)
    for interval in intervals:
        interval.distance_source = 'device'; interval.stopped_time_s = 0.; interval.bearing_degrees = None
    before = deepcopy(intervals)
    cal = fit_comparison_calibration([], config)
    blocks, audit = blocks_from_intervals(intervals, config, cal)
    assert audit['retained_moving_seconds'] == 1800
    assert audit['missing_moving_seconds'] == audit['score_based_exclusions'] == 0
    assert blocks[0].moving_minute < 5
    assert sum(b.high_hr_seconds for b in blocks) > 0
    assert score(blocks)['available']
    assert intervals == before


def test_gap_is_not_filled_and_both_surrounding_bouts_remain(config):
    intervals, _ = synthetic(config, steady=True)
    for interval in intervals:
        interval.distance_source = 'device'; interval.stopped_time_s = 0.; interval.bearing_degrees = None
    intervals[30].elapsed_s = 120.
    cal = fit_comparison_calibration([], config)
    blocks, audit = blocks_from_intervals(intervals, config, cal)
    assert audit['missing_moving_seconds'] == 10.
    assert audit['retained_moving_seconds'] == 1790.
    assert any(b.start < 300 for b in blocks) and any(b.start > 300 for b in blocks)


def test_nonpositive_balance_reported_not_fabricated():
    rows = [BalanceBlock(0, 120, 1, 300, 30, 0, -1, 120)]
    assert not score(rows, 0, 300)['available']


def test_asymmetric_correction_must_not_be_accepted_just_for_lower_fit_error():
    # Alternating measurement noise with constant true effort. An asymmetric
    # derivative term invents a performance bonus; symmetric terms telescope.
    rows = [BalanceBlock(i*120, 120, i*2+1, 324, 120,
        .05 if i % 2 == 0 else 0., -.05 if i % 2 else 0., 120) for i in range(10)]
    assert score(rows, 60, 60)['response_fraction'] == pytest.approx(0.)
    assert score(rows, 30, 180)['response_fraction'] < 0
