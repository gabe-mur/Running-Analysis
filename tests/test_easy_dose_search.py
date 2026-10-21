"""Contracts for the offline matched-context diagnostic, not planner policy."""

import pytest

from scripts import audit_easy_dose_search as audit


@pytest.mark.parametrize("raises", [False, True])
def test_trace_restores_all_hooks(raises):
    names = ("adaptive_run_day_offsets", "_joint_candidate_program_cost", "_select_joint_finalists")
    originals = {name: getattr(audit.planner, name) for name in names}
    try:
        with audit.trace_search():
            assert all(getattr(audit.planner, name) is not original for name, original in originals.items())
            if raises:
                raise RuntimeError("intentional")
    except RuntimeError:
        pass
    assert all(getattr(audit.planner, name) is original for name, original in originals.items())


def test_stage_distinguishes_preliminary_and_reconciled_evaluation():
    trace = {"shortlists": [{"ranked": [(1, (1, 3)), (2, (2, 4))]}],
             "joint_calls": [{"offsets": [2, 4], "reconciled": False}]}
    result = audit.search_stage({"offsets": [2, 4]}, trace)
    assert result["preliminary_ranks"] == [2]
    assert not result["reached_reconciled_scoring"]
    trace["joint_calls"].append({"offsets": [2, 4], "reconciled": True})
    assert audit.search_stage({"offsets": [2, 4]}, trace)["reached_reconciled_scoring"]


def test_shortlist_instrumentation_does_not_change_selection():
    candidates = [(float(i), (i, i + 2, i + 4)) for i in range(8)]
    expected = audit.planner._select_joint_finalists(candidates)
    with audit.trace_search() as trace:
        assert audit.planner._select_joint_finalists(candidates) == expected
    assert trace["shortlists"] == [{"ranked": sorted(candidates), "selected": expected}]


@pytest.mark.parametrize("raises", [False, True])
def test_after_search_receives_inputs_and_restores_hook_if_probe_fails(monkeypatch, raises):
    def fake_search(*, _planner_diagnostics, marker):
        return [1, 3]
    monkeypatch.setattr(audit.planner, "adaptive_run_day_offsets", fake_search)
    seen = []

    def inspect(trace):
        seen.append((trace["inputs"]["marker"], trace["offsets"]))
        if raises:
            raise RuntimeError("probe failed")

    try:
        with audit.trace_search(after_search=inspect):
            assert audit.planner.adaptive_run_day_offsets(
                _planner_diagnostics=audit.planner._PlannerSearchDiagnostics(), marker=42) == [1, 3]
    except RuntimeError:
        assert raises
    assert seen == [(42, [1, 3])]
    assert audit.planner.adaptive_run_day_offsets is fake_search
