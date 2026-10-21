from dataclasses import asdict, replace
import math

import pytest

from run_analysis.long_volume_balance import LongVolumeBalancePolicy, propose_balance_phase
from run_analysis.staged_progression import ProgressionPolicy, review_recent_delivery
from run_analysis.adaptive.stage_acceptance import encode, read_stage
from test_staged_progression import stage, entry, review


def balanced(**changes):
    return stage(policy=ProgressionPolicy(14, .05, balance=LongVolumeBalancePolicy()), **changes)


def test_curve_has_declared_asymptotes_anchor_and_no_shrinking_long():
    p = LongVolumeBalancePolicy()
    assert p.share(0) == pytest.approx(.45)
    assert p.share(80) == pytest.approx(.22)
    assert p.share(1e100) == pytest.approx(.2)
    volumes = [i/10 for i in range(1, 3001)]
    shares = [p.share(v) for v in volumes]
    longs = [p.long_at_volume(v) for v in volumes]
    assert all(a > b for a, b in zip(shares, shares[1:]))
    assert all(a < b for a, b in zip(longs, longs[1:]))
    # Check smooth derivatives around the apparent transition, not just bins.
    for v in (20., p.transition_volume, 60., 80.):
        left = (p.long_at_volume(v)-p.long_at_volume(v-1e-4))/1e-4
        right = (p.long_at_volume(v+1e-4)-p.long_at_volume(v))/1e-4
        assert left > 0 and right > 0
        assert left == pytest.approx(right, abs=1e-5)


@pytest.mark.parametrize('long_miles', [1., 2.5, 8.9, 17., 20., 40.])
def test_inverse_is_smallest_supported_committed_tenth(long_miles):
    p = LongVolumeBalancePolicy()
    volume = p.volume_for_long(long_miles)
    assert p.long_at_volume(volume)+1e-8 >= long_miles
    assert p.long_at_volume(volume-.1) < long_miles
    assert p.committed_long_at_volume(volume) <= p.long_at_volume(volume)+1e-8


@pytest.mark.parametrize('bad', [-1., math.nan, math.inf])
def test_invalid_volume_is_not_silently_clamped(bad):
    with pytest.raises(ValueError):
        LongVolumeBalancePolicy().share(bad)
    with pytest.raises(ValueError):
        LongVolumeBalancePolicy().volume_for_long(bad)


@pytest.mark.parametrize('options', [dict(low_volume_share=.2), dict(anchor_share=.2),
    dict(anchor_volume=0), dict(shape=math.nan), dict(shape=20), dict(source='')])
def test_invalid_policy_rejected(options):
    with pytest.raises(ValueError):
        LongVolumeBalancePolicy(**options)


def test_phase_proposal_does_not_rebase_or_accept_current_stage():
    s = balanced(weekly_equivalent_miles=20.3, long_miles=8.9)
    result = propose_balance_phase(s)
    assert result['required_volume_per_7'] == 20.9
    assert result['policy'].growth_focus == 'easy_volume'
    assert result['policy'].volume_path == 'gradual'
    assert result['policy'].volume_goal_miles == 20.9
    assert result['stage_target_unchanged'] == s
    assert not result['accepted']
    caught_up = replace(s, weekly_equivalent_miles=20.9)
    result = propose_balance_phase(caught_up)
    assert result['policy'].growth_focus == 'long_and_volume'
    assert result['policy'].volume_path == 'step'  # Not falsely a gradual long ramp.
    assert result['policy'].volume_goal_miles is None


def test_imbalanced_stage_catches_up_before_novel_long_growth():
    s = balanced(weekly_equivalent_miles=20.3, long_miles=8.9)
    result = review(s, [entry(s, 8.9)], longest_prior_30d_miles=8.9)
    assert result['status'] == 'conditional_development_candidate'
    assert result['candidate'].weekly_equivalent_miles == 20.9
    assert result['candidate'].long_miles == 8.9
    assert not result['long_progression']
    assert result['long_volume_balance']['changes_score'] is False
    caught_up = result['candidate']
    result = review(caught_up, [entry(caught_up, 8.9)], longest_prior_30d_miles=8.9)
    assert result['candidate'].weekly_equivalent_miles == 21.9
    assert result['candidate'].long_miles == 9.2
    assert result['candidate'].long_miles <= LongVolumeBalancePolicy().long_at_volume(21.9)


def test_curve_cannot_shrink_established_long_or_replace_session_guard():
    s = balanced(weekly_equivalent_miles=10., long_miles=8.9)
    result = review(s, [entry(s, 8.9)], longest_prior_30d_miles=8.9)
    assert result['candidate'].long_miles == 8.9
    assert result['candidate'].weekly_equivalent_miles == 10.5
    assert review(s, [entry(s, 8.9)], longest_prior_30d_miles=7.)['candidate'] is None


def test_curve_does_not_bypass_health_missing_delivery_or_realized_funding():
    from run_analysis.staged_progression import ResponseReport
    s = balanced(weekly_equivalent_miles=20.3, long_miles=8.9)
    assert review(s, [], ledger_complete=False)['candidate'] is None
    assert review(s, [entry(s, 7.)])['candidate'] is None
    assert review(s, [entry(s, 8.9)], responses=[
        ResponseReport(s.issued_at, 'health_concern', 'test')])['candidate'] is None
    result = review_recent_delivery(s, [entry(s, 8.9)], s.review_at,
        ledger_complete=True, longest_prior_30d_miles=8.9)
    assert result['status'] == 'recent_delivery_hold_targets'


def test_goal_long_requires_support_then_stops_unbounded_growth():
    s = balanced(weekly_equivalent_miles=40., long_miles=17.)
    result = review(s, [entry(s, 17.)], longest_prior_30d_miles=17.)
    assert result['candidate'].long_miles == 17.
    assert result['candidate'].weekly_equivalent_miles == 42.
    funded = replace(s, weekly_equivalent_miles=76.)
    result = review(funded, [], longest_prior_30d_miles=17.)
    assert result['status'] == 'destination_reached_reassess_supporting_program'
    assert result['candidate'] is None


def test_surplus_volume_does_not_prevent_final_goal_increment():
    s = balanced(weekly_equivalent_miles=80., long_miles=16.9)
    result = review(s, [entry(s, 16.9)], longest_prior_30d_miles=16.9)
    assert result['candidate'].long_miles == 17.
    assert result['extra_nonlong_miles_per_7_days'] >= 0


def test_restart_and_mapping_round_trip_preserve_curve_and_legacy_policy():
    s = balanced(weekly_equivalent_miles=20.3, long_miles=8.9)
    assert read_stage(encode(s)) == s
    assert ProgressionPolicy(**asdict(s.policy)) == s.policy
    legacy = asdict(stage().policy)
    legacy.pop('balance')
    assert ProgressionPolicy(**legacy).balance is None
    assert review()['candidate'].long_miles == 8.8


def test_repeated_reviews_are_identical_without_window_dependent_inputs():
    s = balanced(weekly_equivalent_miles=20.3, long_miles=8.9)
    a = review(s, [entry(s, 8.9)], longest_prior_30d_miles=8.9)
    assert review(s, [entry(s, 8.9)], longest_prior_30d_miles=8.9) == a
    assert propose_balance_phase(s) == propose_balance_phase(s)


def test_curve_reuses_atomic_calendar_acceptance_and_survives_harness_restart(tmp_path, monkeypatch):
    from functools import partial
    import test_adaptive_stage_lifecycle as fixtures
    from run_analysis.adaptive.adaptive_stage_lifecycle import advance_stage
    from run_analysis.adaptive.new_model_daily_harness import DailyHarness
    monkeypatch.setattr(fixtures, 'ProgressionPolicy', partial(
        ProgressionPolicy, balance=LongVolumeBalancePolicy()))
    h, inputs, config, at = fixtures.completed_stage(tmp_path, monkeypatch,
        continuation_boundary=True, single_volume=True)
    try:
        updated, result, event = advance_stage(h, inputs, at)
        assert result['status'] == 'conditional_development_candidate'
        assert result['long_volume_balance']['changes_score'] is False
        assert event['stage_acceptance']
        assert updated.stage.long_miles == 8.8
        assert updated.stage.policy.balance == LongVolumeBalancePolicy()
        assert updated.catalog['quality:ordinary'] == h.catalog['quality:ordinary']
        assert updated.training_policy.long_per_7 == h.training_policy.long_per_7
        assert updated.training_policy.quality_per_7 == h.training_policy.quality_per_7
        updated.sandbox.close()
        resumed = DailyHarness.resume(tmp_path/'state', config, as_of=at)
        try:
            assert resumed.stage == updated.stage
            assert resumed.training_policy == updated.training_policy
        finally:
            resumed.sandbox.close()
    finally:
        h.sandbox.close()
