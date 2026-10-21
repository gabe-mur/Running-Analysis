"""Contracts for offline intentions, not physiological endorsement."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from scripts.easy_dose_prototype import (
    Observation, build_profile, generate_intents, restrict_intents,
)
from scripts import audit_easy_dose_prototype as audit


NOW = datetime(2026, 9, 22, 12, tzinfo=timezone.utc)


def observation(identity="ordinary", distance=4, **kwargs):
    return Observation(identity, NOW - timedelta(days=2), distance, **kwargs)


def profile(items=None, as_of=NOW):
    return build_profile(items if items is not None else [observation()], as_of, half_life_days=84)


def menu(history=None, **kwargs):
    args = dict(useful_floor_miles=3, progression_allowance_miles=.32,
                progression_source="test_explicit_program_allowance")
    args.update(kwargs)
    return generate_intents(history or profile(), **args)


@pytest.mark.parametrize("role", ["ordinary_easy", "medium_long"])
@pytest.mark.parametrize("distance", [3.5, 3.7, 4.0])
def test_in_range_execution_has_same_planning_reference_without_erasing_actual(role, distance):
    p = profile([observation(distance=distance, planning_role=role, prescribed_range=(3.5, 4))])
    assert p.ordinary_planning_miles == 3.75
    assert p.ordinary_actual_miles == distance
    assert p.observations[0].actual_miles == distance
    assert menu(p).ordinary_miles == 3.75


def test_short_support_run_does_not_pull_ordinary_reference_down():
    p = profile([observation(), observation("support", 2.9, planning_role="support_easy", prescribed_range=(2.5, 3))])
    assert p.ordinary_planning_miles == 4
    assert p.support_planning_miles == 2.75
    assert p.ordinary_count == p.support_count == 1


def test_supported_shorter_option_is_retained_inside_useful_band():
    p = profile([observation(), observation("support", 3.1, planning_role="support_easy")])
    choices = menu(p).options
    assert any(x.midpoint_miles == 3.1 and "observed_support_session" in x.sources for x in choices)


def test_future_projected_quality_and_health_caution_are_not_ordinary_evidence():
    items = [observation(), replace(observation("future", 10), occurred_at=NOW + timedelta(days=1)),
             observation("projected", 10, projected=True),
             observation("quality", 10, workout_type="intervals"),
             observation("illness", 10, health="illness")]
    p = profile(items)
    assert p.ordinary_count == 1 and p.ordinary_planning_miles == 4
    assert len(p.exclusions) == 4


def test_completion_and_low_rpe_do_not_assert_recovery_tolerance():
    p = profile([observation(perceived_exertion=2)])
    assert p.recovery_response_count == 0
    assert any("unobserved" in x for x in menu(p).notes)


def test_refresh_without_new_evidence_leaves_options_unchanged():
    items = [replace(observation(str(i), 3 + i / 4), occurred_at=NOW - timedelta(days=3 * i + 2)) for i in range(8)]
    assert menu(profile(items)) == menu(profile(items, NOW + timedelta(hours=1)))
    assert menu(profile(items)) == menu(profile(items, NOW + timedelta(days=1)))


def test_one_large_completion_does_not_redefine_median():
    items = [observation(str(i), 4) for i in range(6)]
    assert profile([*items, observation("large", 8)]).ordinary_planning_miles == 4


def test_new_repeated_history_can_raise_reference():
    items = [replace(observation("old", 3), occurred_at=NOW - timedelta(days=90)),
             observation("new1", 4), observation("new2", 4)]
    assert profile(items).ordinary_planning_miles == 4


def test_progression_is_explicit_and_not_required():
    no_growth = menu(progression_allowance_miles=0)
    assert max(x.midpoint_miles for x in no_growth.options) == 4
    growth = menu()
    assert max(x.midpoint_miles for x in growth.options) == pytest.approx(4.32)
    assert any("ordinary" in x.purposes for x in growth.options)
    assert all("test_explicit_program_allowance" in x.sources for x in growth.options if "progression" in x.purposes)


def test_menu_has_intermediate_options_not_three_discrete_targets():
    choices = menu().options
    assert len(choices) > 3
    assert max(b.midpoint_miles - a.midpoint_miles for a, b in zip(choices, choices[1:])) <= .25 + 1e-9


def test_prior_prescription_is_candidate_not_new_growth_reference_or_lock():
    choices = menu(prior_range=(5, 5.5))
    prior = next(x for x in choices.options if "prior_prescription" in x.purposes)
    assert prior.execution_range == (5, 5.5)
    assert prior.purposes == ("prior_prescription",)
    assert choices.ordinary_miles == 4
    eligible, rejected = restrict_intents(choices, maximum_midpoint_miles=4.5, reason="material_evidence")
    assert prior not in eligible and any(x["option"]["midpoint_miles"] == 5.25 for x in rejected)


def test_restrictions_remove_options_without_inventing_smaller_runs():
    choices = menu()
    eligible, rejected = restrict_intents(choices, maximum_midpoint_miles=2.9, reason="caution")
    assert not eligible and len(rejected) == len(choices.options)
    eligible, _ = restrict_intents(choices, maximum_midpoint_miles=None, no_run=True, reason="unavailable")
    assert not eligible


def test_restrictions_can_relax_without_changing_training_intentions():
    choices = menu()
    first, _ = restrict_intents(choices, maximum_midpoint_miles=3.5, reason="cost")
    later, _ = restrict_intents(choices, maximum_midpoint_miles=4, reason="cost")
    assert set(first).issubset(later)
    assert menu() == choices


def test_missing_history_does_not_invent_a_normal_session():
    p = profile([])
    choices = menu(p)
    assert choices.ordinary_miles is None and not choices.options
    assert "baseline-acquisition" in choices.notes[0]


def test_duplicate_activity_ids_are_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        profile([observation(), observation()])


@pytest.mark.parametrize("kwargs", [
    {"progression_allowance_miles": -1}, {"progression_allowance_miles": float("inf")},
    {"useful_floor_miles": 0}, {"quantum_miles": 0},
    {"progression_source": ""}, {"progression_allowance_miles": 1000000},
])
def test_invalid_or_unbounded_requests_fail(kwargs):
    with pytest.raises(ValueError):
        menu(**kwargs)


def test_restriction_requires_a_reason():
    with pytest.raises(ValueError, match="explanation"):
        restrict_intents(menu(), maximum_midpoint_miles=3)


def test_paired_audit_distinguishes_intent_stability_from_selected_dose():
    rows = audit.paired_probes()
    assert len(rows) == 33 and all(x["prototype_intents_unchanged"] for x in rows)
    assert max(abs(x["current_change"]) for x in rows) == pytest.approx(1)


def test_audit_does_not_overwrite_artifact(tmp_path, monkeypatch):
    import sys
    path = tmp_path / "existing.json"
    path.write_text("keep")
    monkeypatch.setattr(sys, "argv", ["audit", "--output-json", str(path)])
    with pytest.raises(SystemExit):
        audit.main()
    assert path.read_text() == "keep"


def test_funding_probe_keeps_key_work_fixed_and_reports_optimistic_shortfall(monkeypatch):
    monkeypatch.setattr(audit.current, "calendar_audit", lambda *args: {
        "variants": [{"candidates": [{
            "offsets": [0, 2], "allocation_trace": ["0:easy_run:easy:3.500", "2:long_run:long:8.000"],
            "planned_midpoint_miles": 11.5, "integrated_target_low_miles": 14,
        }]}],
    })
    result = audit.funding_probe(None, {"intent_menu": {"options": [{"midpoint_miles": 4}]}})
    row = result["rows"][0]
    assert row["optimistic_prototype_total"] == 12
    assert row["optimistic_shortfall"] == 2
    assert "not an automatic gate failure" in result["scope"]


def test_input_order_does_not_change_history_or_menu():
    items = [observation("a", 3), observation("b", 4), observation("c", 5)]
    assert profile(items) == profile(list(reversed(items)))


@pytest.mark.parametrize("size", [2, 4, 6])
def test_execution_ranges_preserve_selected_midpoint_at_different_runner_sizes(size):
    p = profile([observation(distance=size)])
    choices = menu(p, useful_floor_miles=size * .8, progression_allowance_miles=size * .08)
    for choice in choices.options:
        assert sum(choice.execution_range) / 2 == pytest.approx(choice.midpoint_miles)
