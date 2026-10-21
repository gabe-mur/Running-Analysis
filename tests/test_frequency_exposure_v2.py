from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

import pytest

from run_analysis.adaptive.frequency_range_trial import (
    FrequencyContract, compile_frequency_exposure, assess_frequency)

START = date(2026, 10, 3)


def contract():
    return FrequencyContract.freeze(adopted_at=datetime(2026, 10, 3, tzinfo=timezone.utc),
        stage_key='one', effective_on=START, volume_per_7=22.,
        reference_fatigue=.04, reference_days=54, version='frequency_midpoint_v2')


def frame(c=None, *, offset=0, observed=(), span=54):
    decision = START+timedelta(days=offset)
    return compile_frequency_exposure(c or contract(), observed, decision=decision,
        coverage_end=decision+timedelta(days=span), taper_start=decision+timedelta(days=40))


def test_missed_count_is_guide_not_retroactive_catchup_quota():
    c = contract()
    future = tuple(START+timedelta(days=i) for i in range(28, 44))
    for planned in (future, (START+timedelta(days=27), *future)):
        a = frame(c, offset=27, observed=(START,)).assess(planned)
        assert a['eligible'] and a['lower_is_advisory']
    b = frame(c, offset=28, observed=(START,)).assess(future)
    assert b['eligible']
    # Even no new runs is not rejected by a count floor. Other unchanged model
    # terms/growth funding still determine whether that is a usable programme.
    assert frame(c, offset=27, observed=(START,)).assess(())['eligible']


def test_partial_tail_is_priced_from_initial_decision_not_activated_day_two():
    c = contract()
    planned = tuple(START+timedelta(days=i) for i in (*range(2, 18), *range(30, 50)))
    audits = [frame(c, offset=i).assess(planned) for i in range(3)]
    assert all(len(a['blocks']) == 1 for a in audits)
    assert all(a['unassessed_from'] == a['coverage_end'] for a in audits)
    costs = [c.preference.assess(a)['cost'] for a in audits]
    assert costs[0] > 0  # Previously unassessed second block already contributes.
    assert max(abs(b-a) for a, b in zip(costs, costs[1:])) < .0001
    assert audits[0]['blocks'][0]['planned_count'] == len(planned)


def test_upper_applies_to_future_not_excess_observed_history():
    observed = tuple(START+timedelta(days=i) for i in range(27))
    assert frame(offset=28, observed=observed).assess(())['eligible']
    decision = START+timedelta(days=28)
    daily = tuple(decision+timedelta(days=i) for i in range(54))
    assert not frame(offset=28, observed=observed).assess(daily)['eligible']
    assert not frame(offset=28, observed=observed).assess(daily)['construction_eligible']


def test_upload_replacement_preserves_weighted_count_and_never_double_counts():
    planned = (START, START+timedelta(days=2))
    a = frame().assess(planned)
    b = frame(observed=(START,)).assess(planned)
    assert a['blocks'][0]['count'] == b['blocks'][0]['count']
    assert b['blocks'][0]['observed_count'] == 1
    assert b['blocks'][0]['planned_count'] == 1


def test_declining_past_weight_and_tail_weight_not_free_or_future_credit():
    f = frame(offset=28, observed=(START, START+timedelta(days=27)))
    weights = dict(f.weights)
    assert weights[START] == 0
    assert weights[START+timedelta(days=27)] == 27/28
    assert weights[f.end-timedelta(days=1)] == 1/14
    assert f.end not in weights
    with pytest.raises(ValueError):
        f.assess((f.end,))


def test_stage_targets_change_future_rates_not_past_or_normalization():
    c = contract()
    changed = c.accepted_stage('next', START+timedelta(days=14), 42.)
    f = frame(changed, offset=14)
    plain = frame(c, offset=14)
    assert changed.preference == c.preference
    assert f.future_upper > plain.future_upper
    # Historical part retains its original dated lower/upper rates.
    assert f.upper-f.future_upper == pytest.approx(plain.upper-plain.future_upper)
    assert f.anchor == START


def test_legacy_contract_keeps_v1_block_meaning_explicitly():
    c = replace(contract(), version='frequency_midpoint_v1')
    a = assess_frequency(c.targets, (), (), decision=START,
        coverage_end=START+timedelta(days=54), policy=c.policy)
    assert not a['eligible'] and len(a['blocks']) == 1


def test_compiled_frame_validation():
    for kwargs in (dict(taper_start=START), dict(taper_start=START+timedelta(days=54)),
                   dict(observed_dates=(START+timedelta(days=1),))):
        options = dict(observed_dates=(), decision=START, coverage_end=START+timedelta(days=54),
                       taper_start=START+timedelta(days=40))
        options.update(kwargs)
        with pytest.raises(ValueError):
            compile_frequency_exposure(contract(), **options)
