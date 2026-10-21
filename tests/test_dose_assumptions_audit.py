"""Tests for the offline audit, not endorsement of its parameter alternatives."""

from copy import deepcopy
import json
import sqlite3
import sys
from datetime import timedelta

import pytest

from scripts import audit_dose_assumptions as audit


@pytest.mark.parametrize("size", [2.0, 3.7, 6.0])
@pytest.mark.parametrize("residual", [0.0, .10, .105, .1225, .177, .236, .55, .8])
def test_audit_reproduces_production_formula(size, residual):
    state = audit.synthetic_state(size, residual)
    result = audit.recommendation(state)
    maximum = audit.aerobic_maximum(state)
    assert audit.expansion_reference(state, audit.VARIANTS["current"]) == (
        audit.planner._ordinary_easy_expansion_reference(state)
    )
    assert audit.ceiling(state, result, maximum, audit.VARIANTS["current"]) == pytest.approx(
        audit.planner._easy_recovery_allocation_ceiling(state, result, maximum), abs=1e-12,
    )


@pytest.mark.parametrize("raises", [False, True])
def test_variant_hooks_restore_even_after_failure(raises):
    before_ceiling = audit.planner._easy_recovery_allocation_ceiling
    before_reference = audit.planner._ordinary_easy_expansion_reference
    try:
        with audit.apply_variant(audit.VARIANTS["multiplier_low_all"]):
            assert audit.planner._easy_recovery_allocation_ceiling is not before_ceiling
            assert audit.planner._ordinary_easy_expansion_reference is not before_reference
            if raises:
                raise RuntimeError("intentional audit interruption")
    except RuntimeError:
        assert raises
    assert audit.planner._easy_recovery_allocation_ceiling is before_ceiling
    assert audit.planner._ordinary_easy_expansion_reference is before_reference


def test_envelope_only_variant_does_not_relabel_or_reweight_expansion_reference():
    before_reference = audit.planner._ordinary_easy_expansion_reference
    with audit.apply_variant(audit.VARIANTS["multiplier_low"]):
        assert audit.planner._ordinary_easy_expansion_reference is before_reference


def test_current_instrumentation_measures_without_changing_the_ceiling():
    state = audit.synthetic_state(3.7, .236)
    result = audit.recommendation(state)
    maximum = audit.aerobic_maximum(state)
    original = audit.planner._easy_recovery_allocation_ceiling
    expected = original(state, result, maximum)
    with audit.apply_variant(audit.VARIANTS["current"]) as measurements:
        assert audit.planner._easy_recovery_allocation_ceiling(state, result, maximum) == expected
        assert measurements == {"ceiling_calls": 1, "restrictive_ceiling_calls": 1}
    assert audit.planner._easy_recovery_allocation_ceiling is original


def test_variant_allocation_does_not_modify_inputs():
    state = audit.synthetic_state(3.7, .236)
    result = audit.recommendation(state)
    before = deepcopy((state, result, audit.CONFIG))
    with audit.apply_variant(audit.VARIANTS["envelope_removed"]):
        audit.allocate_one(state, result, (4.0, 4.5))
    assert (state, result, audit.CONFIG) == before


def test_small_synthetic_audit_reports_measurements_and_preserves_cautions():
    report = audit.synthetic_audit(
        [audit.VARIANTS[name] for name in ("current", "single_stage", "envelope_removed")],
        profiles=(3.7,),
    )
    assert len(report["summaries"]) == 3
    assert report["summaries"][0]["changed_dose_cases"] == 0
    assert all(item["caution_failures"] == 0 for item in report["summaries"])
    assert all(item["allocated_case_count"] == 24 for item in report["summaries"])
    # The audit has to expose response, not merely print three identical curves.
    assert report["summaries"][-1]["changed_dose_cases"] > 0


def test_full_pool_preserves_tail_and_respects_availability():
    seed = (1, 3, 5, 8, 10, 12, 14, 17, 19)
    allowed = set(range(21)) - {2}
    pool = audit.full_calendar_pool(seed, allowed, {5})
    assert seed in pool
    assert len(pool) <= 9
    assert any(0 in candidate for candidate in pool)
    assert any(len(candidate) == len(seed) - 1 for candidate in pool)
    for candidate in pool:
        assert set(candidate).issubset(allowed)
        assert 5 in candidate
        assert candidate[-6:] == seed[-6:]


def test_calendar_audit_reconciles_every_candidate_and_uses_read_only_input(tmp_path, monkeypatch):
    database = tmp_path / "snapshot.sqlite"
    states = [audit.synthetic_state(3.7, .236).model_copy(update={"as_of": audit.ORIGIN + timedelta(days=i)}) for i in range(3)]
    payload = {
        "daily_states": [x.model_dump(mode="json") for x in states],
        "daily_state_options": [[x.model_dump(mode="json")] for x in states],
        "request": audit.REQUEST.model_dump(mode="json"), "config": audit.CONFIG,
        "target_distance_range": [10, 12],
    }
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE app_state(key TEXT, value_json TEXT)")
        db.execute("INSERT INTO app_state VALUES (?,?)", ("weekly_planner_snapshot", json.dumps(payload)))
    before = database.read_bytes()
    calls = []
    original_connect = sqlite3.connect

    def readonly_connect(database_uri, **kwargs):
        assert database_uri.endswith("?mode=ro") and kwargs == {"uri": True}
        return original_connect(database_uri, **kwargs)

    def joint(offsets, **kwargs):
        assert kwargs["reconcile_allocated_prefix"] is True
        calls.append(offsets)
        return (0, 0, 0, 0)

    monkeypatch.setattr(audit.sqlite3, "connect", readonly_connect)
    monkeypatch.setattr(audit.planner, "_joint_candidate_program_cost", joint)
    monkeypatch.setattr(audit.planner, "_adaptive_candidate_cost", lambda *args, **kwargs: 0)
    report = audit.calendar_audit(
        database, [audit.VARIANTS["current"], audit.VARIANTS["release_wide"]],
        horizon=3, counts=(1, 2), mode="short_exhaustive",
    )
    assert len(calls) == 12
    assert report["variants"][0]["tied_best_count"] == 6
    assert report["expected_target_projector"] is False
    assert database.read_bytes() == before


def test_existing_audit_artifact_is_not_overwritten(tmp_path, monkeypatch):
    output = tmp_path / "existing.json"
    output.write_text("preserved result")
    monkeypatch.setattr(sys, "argv", ["audit", "--output-json", str(output)])
    with pytest.raises(SystemExit) as exc:
        audit.main()
    assert exc.value.code == 2
    assert output.read_text() == "preserved result"
