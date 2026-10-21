from copy import deepcopy

import pytest

from scripts import audit_opportunity_tradeoffs as audit
from scripts.trace_opportunity_survival import TARGET
from test_opportunity_survival import sample_report


def test_funding_pool_is_bounded_and_excludes_unavailable_dates():
    pool = audit.funding_pool(TARGET, range(21))
    assert TARGET in pool and len(pool) <= 128
    assert {len(x) for x in pool} == set(range(6, 13))
    assert pool == audit.funding_pool(reversed(TARGET), reversed(range(21)))
    assert all(3 not in x for x in audit.funding_pool(TARGET, set(range(21)) - {3}))
    with pytest.raises(ValueError):
        audit.funding_pool(TARGET, set(range(21)) - {0})


def test_upper_corner_restricts_only_existing_options_and_restores_on_failure(monkeypatch):
    def original(*a, **k):
        return [2.95, 3.25], {2.95: (2.7, 3.2), 3.25: (3, 3.5)}
    monkeypatch.setattr(audit.planner, "_easy_allocation_options", original)
    with pytest.raises(RuntimeError):
        with audit.upper_easy_options(True):
            assert audit.planner._easy_allocation_options() == ([3.25], {3.25: (3, 3.5)})
            raise RuntimeError("intentional")
    assert audit.planner._easy_allocation_options is original
    with audit.upper_easy_options(False):
        assert audit.planner._easy_allocation_options() == original()


@pytest.mark.parametrize("scale", [.85, 1.])
def test_panel_preserves_frozen_source_and_uses_common_target(monkeypatch, scale):
    report = sample_report()
    before = deepcopy(report)
    monkeypatch.setattr(audit, "funding_pool", lambda seed, allowed: [seed])
    original = audit.planner.adaptive_run_day_offsets
    result = audit.run(report, TARGET, target_scale=scale)
    assert report == before
    assert audit.planner.adaptive_run_day_offsets is original
    assert not audit.planner._OPPORTUNITY_SEARCH_REPAIR.get()
    assert audit.planner._FINALIZED_SCORING_TRIAL.get() is None
    low = sum(pair[0] for pair in result["targets"][:21]) / 7
    for row in result["funding"]:
        assert row["integrated_target"][0] == pytest.approx(low)
        assert row["target_shortfall"] == pytest.approx(max(0., low-row["planned_miles"]))
        assert row["strictly_funded"] == (row["target_shortfall"] <= 1e-9 and row["target_excess"] <= 1e-9)
        assert row["seam_and_gaps_hours"][0] > 0
    for conflict in result["conflicts"]:
        for row in conflict.get("rows", []) + conflict.get("timing_controls", []):
            if "unavailable" in row:
                continue
            assert row["integrated_target"][0] == pytest.approx(low)
            if row["mismatched_roles"] or row["not_ready"] or row["acceptance"]["rejected"]:
                assert not row["eligible"]
        for row in conflict.get("timing_controls", []):
            if row.get("eligible"):
                key = next(s for s in row["sessions"] if s["offset"] == conflict["key_offset"])
                assert key["at"] == conflict["common_key_time"]
    assert len(result["matched_extensions"]) <= 42
    for row in result["matched_extensions"]:
        assert row["integrated_target"][0] == pytest.approx(low)
        assert len(row["offsets"]) == result["matched_funded_base"]["run_count"] + 1
        if row["eligible"]:
            assert not row["mismatched_roles"]
            assert row["roles"][row["added_offset"]] == "easy"


def test_outer_policy_restores_when_scoring_fails(monkeypatch):
    original = audit.planner.adaptive_run_day_offsets
    def failure(*args, **kwargs):
        raise RuntimeError("scorer failed")
    monkeypatch.setattr(audit, "panels", failure)
    with pytest.raises(RuntimeError, match="scorer failed"):
        audit.run(sample_report(), TARGET)
    assert audit.planner.adaptive_run_day_offsets is original
    assert not audit.planner._OPPORTUNITY_SEARCH_REPAIR.get()
    assert audit.planner._FINALIZED_SCORING_TRIAL.get() is None


def test_no_unannounced_target_search():
    with pytest.raises(ValueError, match="Predeclared"):
        audit.run(sample_report(), TARGET, target_scale=.9)


def test_summary_does_not_treat_overfunding_or_rejection_as_within_band():
    def row(label, cost, eligible, within_band):
        return dict(label=label, offsets=(0, 2), run_count=2, planned_miles=10., total_cost=cost,
            eligible=eligible, strictly_funded=within_band, funded_with_existing_tolerance=within_band,
            longest_fraction=.5, seam_and_gaps_hours=[24, 48])
    summary = audit.funding_summary([row("rejected", 0, False, True),
        row("outside_band", 1, True, False), row("within_band", 2, True, True)])
    assert summary[2]["best"]["label"] == "outside_band"
    assert summary[2]["funded"]["label"] == "within_band"
