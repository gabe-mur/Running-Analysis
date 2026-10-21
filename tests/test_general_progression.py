"""Phase arithmetic and atomic wiring, not optimizer-stability acceptance."""
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
import json

import pytest

from run_analysis.general_progression import destination, next_general_phase, VERSION, RATE_VERSION
from run_analysis.staged_progression import DevelopmentStage, ProgressionPolicy, tenth

AT = datetime(2026, 10, 17, tzinfo=timezone.utc)


def stage(volume=19., long=8.9, goal=17.):
    return DevelopmentStage('confirmed', AT, volume, 4.2, long, goal, 7.,
        ProgressionPolicy(14, .05, 'easy_volume', volume, 'gradual'), 'test')


def proposal(current):
    return next_general_phase(current, as_of=AT, longest_prior_30d_miles=current.long_miles)


def test_easy_first_uses_existing_rate_and_bounded_destination():
    current = stage()
    target = destination(current)
    assert (target.long_miles, target.volume_per_7) == (17., 42.5)
    result = proposal(current)
    new = result['candidate']
    assert new.weekly_equivalent_miles == 19.
    assert new.long_miles == 8.9
    assert new.policy.volume_goal_miles == 20.
    assert (new.policy.review_days, new.policy.growth_fraction) == (14, .05)
    assert new.review_at == AT + timedelta(days=14)
    assert ':general-phase:' in new.key and ':race-phase:' not in new.key
    assert result['accepted'] is False
    assert result['supporting_volume_required_for_current_long'] == pytest.approx(22.25)


def test_long_growth_uses_confirmed_support_not_next_ramp():
    assert proposal(stage(22.1))['candidate'].long_miles == 8.9
    new = proposal(stage(23.2))['candidate']
    assert new.long_miles == 9.2  # Confirmed support is 23.2 * .4 = 9.28, floored to the committed grid.
    assert new.policy.volume_goal_miles == 24.4
    guarded = next_general_phase(stage(23.2), as_of=AT, longest_prior_30d_miles=8.2)['candidate']
    assert guarded.long_miles == 9.


@pytest.mark.parametrize('volume,long,goal', [(5., 2., 4.), (20., 8., 10.), (60., 15., 17.)])
def test_athlete_scale_no_current_exposure_reduction(volume, long, goal):
    current = stage(volume, long, goal)
    new = proposal(current)['candidate']
    assert new.weekly_equivalent_miles == volume
    assert new.long_miles >= long
    assert new.policy.volume_goal_miles <= max(volume, tenth(volume * 1.05))
    assert new.policy.balance is None  # Never resurrect rejected 17 -> 76 inverse curve.


def test_destination_caps_growth_and_preserves_higher_established_volume():
    assert proposal(stage(42.5, 17.))['candidate'] is None
    assert proposal(stage(60., 17.))['candidate'] is None
    assert proposal(stage(42., 17.))['candidate'].policy.volume_goal_miles == 42.5


def completed(tmp_path, monkeypatch):
    from test_adaptive_stage_lifecycle import completed_stage
    return completed_stage(tmp_path, monkeypatch, growth_focus='easy_volume', gradual=True,
        continuation_boundary=True, single_volume=True, prospective_review_hour=4,
        purpose_continuation=True, continuation_distribution='c1_v1')


@pytest.mark.parametrize('version,goal', [(VERSION, 22.1), (RATE_VERSION, 22.7)])
def test_new_phase_appends_anchor_and_preserves_normalization_quality_and_weights(tmp_path, monkeypatch, version, goal):
    from run_analysis.adaptive.adaptive_stage_lifecycle import advance_stage
    from run_analysis.adaptive.new_model_daily_harness import DailyHarness
    h, inputs, config, at = completed(tmp_path, monkeypatch)
    h.policy = replace(h.policy, general_progression=version)
    # Persist explicit lifecycle policy just as a prepared isolated release does.
    path = h.sandbox.root / 'scenario.json'
    scenario = json.loads(path.read_text())
    scenario['search_policy'] = asdict(h.policy)
    path.write_text(json.dumps(scenario))
    before = h.stage
    try:
        updated, review, event = advance_stage(h, inputs, at)
        assert event and event['stage_acceptance']
        assert 'general_phase' in review
        assert h.stage == before
        assert updated.stage.weekly_equivalent_miles == 21.
        assert updated.stage.policy.volume_goal_miles == goal
        assert updated.stage.long_miles == 8.4
        assert updated.target_history[:-1] == h.target_history
        assert updated.target_history[-1].effective_on == at.date()
        assert updated.target_history[-1].volume_per_7 == 21.
        assert updated.target_history[-1].volume_goal == goal
        path = updated.target_history[-1]
        rates = [path.rate(path.effective_on + timedelta(days=n), 'volume_per_7') for n in range(15)]
        assert rates[0] == 21. and rates[-1] == goal
        assert all(b > a for a, b in zip(rates, rates[1:]))
        assert len({round(b-a, 10) for a, b in zip(rates, rates[1:])}) == 1
        assert updated.scales == h.scales
        assert updated.catalog['quality:ordinary'] == h.catalog['quality:ordinary']
        assert updated.training_policy == replace(h.training_policy, weekly_miles=21.)
        resumed = DailyHarness.resume(tmp_path/'state', config, as_of=at)
        try:
            assert resumed.stage == updated.stage
            assert resumed.target_history == updated.target_history
            assert resumed.policy.general_progression == version
        finally:
            resumed.sandbox.close()
    finally:
        h.sandbox.close()


@pytest.mark.parametrize('version', [VERSION, RATE_VERSION])
def test_failed_next_phase_search_does_not_commit_confirmation_or_anchor(tmp_path, monkeypatch, version):
    from run_analysis.adaptive.adaptive_stage_lifecycle import advance_stage
    from run_analysis.adaptive.stage_acceptance import StageLedger
    from run_analysis.adaptive.continuous_search_adapter import NoEligibleProgram
    import run_analysis.adaptive.new_model_daily_harness as module
    h, inputs, _, at = completed(tmp_path, monkeypatch)
    h.policy = replace(h.policy, general_progression=version)
    before = h.stage, h.target_history, len(h.events)
    def reject(**kwargs):
        raise NoEligibleProgram('explicit test')
    monkeypatch.setattr(module, 'migrated_search', reject)
    try:
        updated, review, event = advance_stage(h, inputs, at)
        assert updated is h and event is None
        assert review['status'] == 'growth_search_no_eligible_candidate'
        assert (h.stage, h.target_history, len(h.events)) == before
        assert StageLedger.for_sandbox(h.sandbox).current == h.stage
    finally:
        h.sandbox.close()


@pytest.mark.parametrize('version', [VERSION, RATE_VERSION])
def test_missed_delivery_cannot_start_next_phase(tmp_path, monkeypatch, version):
    from run_analysis.adaptive.adaptive_stage_lifecycle import advance_stage
    h, inputs, _, at = completed(tmp_path, monkeypatch)
    h.policy = replace(h.policy, general_progression=version)
    import run_analysis.adaptive.adaptive_stage_lifecycle as module
    entries = module.review_entries(h, at)
    monkeypatch.setattr(module, 'review_entries', lambda *args: tuple(replace(e, completion=None) for e in entries))
    try:
        updated, review, event = advance_stage(h, inputs, at)
        assert updated is h and event is None
        assert review['status'] != 'conditional_development_candidate'
    finally:
        h.sandbox.close()


@pytest.mark.parametrize('volume,long,goal', [(5., 2., 4.), (19., 8.9, 17.), (60., 15., 17.)])
def test_four_percent_rate_is_total_volume_and_does_not_accelerate_long(volume, long, goal):
    current = stage(volume, long, goal)
    result = next_general_phase(current, as_of=AT, longest_prior_30d_miles=long,
                                version=RATE_VERSION)
    new = result['candidate']
    assert new.policy.growth_fraction == pytest.approx(1.04**2 - 1)
    assert result['long_growth_fraction'] == .05
    assert new.policy.volume_goal_miles == min(destination(current).volume_per_7, tenth(volume * 1.04**2))
    assert new.long_miles <= tenth(long * 1.05)
    assert current.policy.growth_fraction == .05  # Existing accepted path is never rewritten.
    assert result['destination'].source.endswith(RATE_VERSION)


def test_rate_separation_when_support_allows_full_long_increment():
    current = stage(30., long=8., goal=17.)
    new = next_general_phase(current, as_of=AT, longest_prior_30d_miles=8.,
                             version=RATE_VERSION)['candidate']
    assert new.long_miles == 8.4  # Not 8.7 from accidentally applying volume growth.
    assert new.policy.volume_goal_miles == 32.4
    at_goal = next_general_phase(stage(42.5, 17.), as_of=AT,
        longest_prior_30d_miles=17., version=RATE_VERSION)
    assert at_goal['candidate'] is None


def test_new_rate_still_obeys_observed_long_guard_and_rejects_unknown_version():
    guarded = next_general_phase(stage(30.), as_of=AT, longest_prior_30d_miles=8.2,
                                 version=RATE_VERSION)['candidate']
    assert guarded.long_miles == 9.
    with pytest.raises(ValueError, match='Named'):
        next_general_phase(stage(), as_of=AT, longest_prior_30d_miles=8.9, version='unknown')


@pytest.mark.parametrize('state', ['little_tired', 'health_concern'])
def test_new_rate_does_not_authorize_growth_with_adverse_readiness(tmp_path, monkeypatch, state):
    from run_analysis.adaptive.adaptive_stage_lifecycle import advance_stage
    from run_analysis.session_decision import CurrentContext
    h, inputs, _, at = completed(tmp_path, monkeypatch)
    h.policy = replace(h.policy, general_progression=RATE_VERSION)
    h.current_context = CurrentContext(at.date(), at, state, 'explicit current report')
    before = h.stage, h.target_history, len(h.events)
    try:
        updated, review, event = advance_stage(h, inputs, at)
        assert updated is h and event is None
        assert review['status'] != 'conditional_development_candidate'
        assert (h.stage, h.target_history, len(h.events)) == before
    finally:
        h.sandbox.close()
