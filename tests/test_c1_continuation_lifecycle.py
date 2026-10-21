"""Real scorer/snapshot lifecycle controls; fixture search is not stability evidence."""
from datetime import timedelta

import pytest

from test_adaptive_stage_lifecycle import completed_stage
from test_continuation_boundary_lifecycle import check_replay
from run_analysis.adaptive.new_model_daily_harness import DailyHarness, HarnessPolicy
from run_analysis.adaptive.adaptive_stage_lifecycle import advance_stage


@pytest.mark.parametrize('phase', [None, 'known_full_phase_v1'])
def test_c1_upload_rollover_restart_and_stage_preview(tmp_path, monkeypatch, phase):
    h, inputs, config, at = completed_stage(tmp_path, monkeypatch, growth_focus='easy_volume',
        continuation_boundary=True, purpose_continuation=phase is not None,
        purpose_continuation_phase=phase, continuation_distribution='c1_v1')
    try:
        check_replay(h, h.events[0])
        check_replay(h, h.replan(at, reason='post_upload'))
        at += timedelta(seconds=1)
        updated, review, event = advance_stage(h, inputs, at)
        assert event['stage_acceptance']
        assert updated.policy.continuation_distribution == 'c1_v1'
        check_replay(updated, event)
        root = updated.sandbox.root
    finally:
        h.sandbox.close()
    reopened = DailyHarness.resume(root, config, as_of=at)
    try:
        assert reopened.policy.continuation_distribution == 'c1_v1'
        check_replay(reopened, reopened.events[-1])
        check_replay(reopened, reopened.replan(at+timedelta(days=1), reason='restored_rollover'))
    finally:
        reopened.sandbox.close()


def test_c1_requires_explicit_boundary_and_legacy_stays_default():
    assert HarnessPolicy().continuation_distribution == 'linear_v1'
    for value in ('c1_v1', 'unknown'):
        with pytest.raises(ValueError, match='continuation distribution'):
            HarnessPolicy(continuation_distribution=value)
