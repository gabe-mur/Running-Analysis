from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from scripts import audit_first_calendar_refresh as audit


@pytest.mark.parametrize("raises", [False, True])
def test_score_probe_restores_materializer_and_separates_coaching(monkeypatch, raises):
    monkeypatch.setattr(audit, "capture_allocations", lambda: nullcontext({"days": []}))
    monkeypatch.setattr(audit.planner, "_joint_candidate_program_cost", lambda *a, **k: (0, 0, 0, 2))
    monkeypatch.setattr(audit.planner, "typical_easy_distance", lambda state: (3, 4))
    original = lambda *a, **k: []
    monkeypatch.setattr(audit.planner, "_materialize_candidate_sessions", original)

    def coaching(*args, **kwargs):
        assert kwargs["include_mileage_path"] is False
        assert kwargs["include_recovery_interactions"] is False
        assert kwargs["include_cadence_pressure"] is False
        assert audit.planner._materialize_candidate_sessions is not original
        if raises:
            raise RuntimeError("probe failure")
        return 3

    monkeypatch.setattr(audit.planner, "_adaptive_candidate_cost", coaching)
    inputs = dict(daily_states=[SimpleNamespace()], daily_state_options=[], request=None,
                  config={}, target_distance_range=(15, 16), completed_miles_by_offset={},
                  expected_target_projector=None, prior_distance_midpoints_by_date={})
    if raises:
        with pytest.raises(RuntimeError, match="probe failure"):
            audit.score_calendar(inputs, (1, 3))
    else:
        row = audit.score_calendar(inputs, (1, 3))
        assert row["coaching"] == row["coaching_other_cost"] == 3
        assert row["total"] == 5
    assert audit.planner._materialize_candidate_sessions is original
