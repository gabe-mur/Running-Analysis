from dataclasses import asdict
from datetime import date, timedelta
import json
import math

import pytest

from run_analysis.adaptive.frequency_range_trial import (
    FrequencyRangePolicy, FrequencyStageTarget, assess_frequency, ranged_evaluator)

START = date(2026, 10, 3)
POLICY = FrequencyRangePolicy()


def dates(count, start=START):
    return tuple(start+timedelta(days=i) for i in range(count))


def target(volume=22., at=START, key='stage'):
    return FrequencyStageTarget.freeze(key, at, volume)


def audit(count, volume=22.):
    return assess_frequency((target(volume),), (), dates(count), decision=START,
                            coverage_end=START+timedelta(days=49))


@pytest.mark.parametrize('volume,lower,upper', POLICY.anchors)
def test_reference_points_and_continuity(volume, lower, upper):
    assert POLICY.bounds(volume) == (lower, upper)
    assert all(abs(a-b) < .001 for a, b in zip(
        POLICY.bounds(volume-.001), POLICY.bounds(volume+.001)))


def test_two_three_low_three_five_moderate_five_seven_high_are_available():
    for volume, counts in ((5., (8, 12)), (22., (12, 20)), (60., (20, 28))):
        assert all(audit(n, volume)['eligible'] for n in counts)
        assert not audit(counts[0]-1, volume)['eligible']
        if counts[1] < 28:  # A 28-day interval cannot contain29 distinct dates.
            assert not audit(counts[1]+1, volume)['eligible']


def test_no_specific_center_or_score_penalty_inside_range():
    for n in range(12, 21):
        assert audit(n)['eligible']
        assert audit(n)['changes_score'] is False
    assert audit(11)['construction_eligible']  # Search may fill the incomplete plan.
    assert not audit(21)['construction_eligible']


def test_no_seven_day_quota_and_distinct_dates_not_sessions():
    planned = tuple(START+timedelta(days=i) for i in
                    (0, 1, 2, 3, 4, 5, 6, 14, 16, 18, 20, 22))
    result = assess_frequency((target(),), (), planned+planned,
        decision=START, coverage_end=START+timedelta(days=28))
    assert result['eligible'] and result['blocks'][0]['count'] == 12
    # Consecutive dates are NOT clearance; the existing evaluator still checks them.


def test_rollover_and_upload_replace_dates_without_resetting_window():
    full = dates(16)
    initial = assess_frequency((target(),), (), full, decision=START,
                              coverage_end=START+timedelta(days=49))
    decision = START+timedelta(days=5)
    rolled = assess_frequency((target(),), full[:5], full[5:], decision=decision,
                             coverage_end=START+timedelta(days=49))
    assert initial['blocks'][0]['count'] == rolled['blocks'][0]['count']
    assert initial['blocks'][0]['required_count_min'] == rolled['blocks'][0]['required_count_min']
    assert initial['anchor'] == rolled['anchor']


def test_today_upload_and_duplicates_are_not_counted_twice():
    result = assess_frequency((target(),), (START,), dates(12), decision=START,
                              coverage_end=START+timedelta(days=28))
    block = result['blocks'][0]
    assert block['count'] == 12 == block['observed_count']+block['planned_count']


def test_stage_change_affects_future_bounds_not_past_or_window_anchor():
    change = START+timedelta(days=14)
    rows = (target(5.), target(22., change, 'next'))
    result = assess_frequency(rows, dates(6), dates(8, change), decision=change,
                              coverage_end=START+timedelta(days=49))
    block = result['blocks'][0]
    assert block['lower_expected_count'] == 10  # 2*2 + 2*3, not retroactive12.
    assert block['upper_expected_count'] == 16  # 2*3 + 2*5.
    assert result['anchor'] == START.isoformat()
    assert result['eligible']


def test_closed_block_shortfall_is_not_a_permanent_frequency_veto():
    later = START+timedelta(days=28)
    result = assess_frequency((target(),), (), dates(12, later), decision=later,
                              coverage_end=later+timedelta(days=28))
    assert result['eligible']
    assert result['blocks'][0]['start'] == later.isoformat()


def test_no_future_tail_credit_or_incomplete_window_pass():
    end = START+timedelta(days=28)
    with pytest.raises(ValueError):
        assess_frequency((target(),), (), (*dates(12), end), decision=START, coverage_end=end)
    result = assess_frequency((target(),), (), dates(12), decision=START,
                              coverage_end=START+timedelta(days=21))
    assert not result['eligible']
    assert result['status'] == 'insufficient_projection_coverage'


def test_wrapper_preserves_score_safety_and_underfunding_verdict():
    program = tuple((d.isoformat(), 'easy') for d in dates(16))
    for base in (dict(cost=1., eligible=True),
                 dict(cost=.1, eligible=False, construction_eligible=True),  # underfunded prefix
                 dict(cost=.1, eligible=False, construction_eligible=False)):  # physical rejection
        evaluate = ranged_evaluator(lambda p: dict(base), (target(),), (),
            decision=START, coverage_end=START+timedelta(days=28))
        result = evaluate(program)
        assert result['cost'] == base['cost']
        assert result['eligible'] == base['eligible']
        assert result['construction_eligible'] == base.get('construction_eligible', base['eligible'])
    # An unfinished lower-count prefix isn't accidentally pruned.
    result = ranged_evaluator(lambda p: dict(cost=3., eligible=True), (target(),), (),
        decision=START, coverage_end=START+timedelta(days=28))(())
    assert not result['eligible'] and result['construction_eligible']


def test_target_does_not_depend_on_candidate_volume_or_role():
    expected = audit(16)['blocks'][0]['allowed_count_max']
    wrapped = ranged_evaluator(lambda p: dict(cost=1., eligible=True), (target(),), (),
        decision=START, coverage_end=START+timedelta(days=28))
    for key in ('easy:1.0', 'easy:8.0', 'quality', 'long'):
        result = wrapped(tuple((d.isoformat(), key) for d in dates(16)))
        assert result['frequency_range']['blocks'][0]['allowed_count_max'] == expected


def test_json_round_trip_preserves_frozen_targets_and_policy():
    p = FrequencyRangePolicy(**json.loads(json.dumps(asdict(POLICY))))
    t = json.loads(json.dumps(asdict(target()), default=lambda d: d.isoformat()))
    t['effective_on'] = date.fromisoformat(t['effective_on'])
    assert FrequencyStageTarget(**t) == target() and p == POLICY


@pytest.mark.parametrize('volume', [0, -1, math.nan, math.inf])
def test_invalid_volume_rejected(volume):
    with pytest.raises(ValueError):
        POLICY.bounds(volume)


@pytest.mark.parametrize('options', [dict(window_days=7), dict(window_days=True),
    dict(anchors=((5, 2, 3), (4, 3, 4))), dict(anchors=((5, 2, 3), (20, 3, 8))),
    dict(source='')])
def test_invalid_policy_rejected(options):
    with pytest.raises(ValueError):
        FrequencyRangePolicy(**options)


def test_invalid_target_history_is_not_silently_rebased():
    end = START+timedelta(days=49)
    for rows in ((), (target(at=START+timedelta(days=1)),),
                 (target(), target(key='other')), (target(), target(at=START-timedelta(days=1)))):
        with pytest.raises(ValueError):
            assess_frequency(rows, (), (), decision=START, coverage_end=end)
