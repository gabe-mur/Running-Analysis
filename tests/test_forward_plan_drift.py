from scripts.audit_forward_plan_drift import anchored_comparisons


def replan(day, miles, *, evidence=()):
    return {"generated_at": f"2026-09-{day:02d}T00:00:00+00:00", "decision_start_date": f"2026-09-{day:02d}",
            "material_evidence_reasons": evidence,
            "planned_sessions": [{"planned_for": "2026-09-25T07:00:00+00:00", "workout_type": "easy", "midpoint_miles": miles}]}


def test_anchor_catches_accumulation_and_retains_intervening_evidence():
    rows = anchored_comparisons([replan(22, 4), replan(23, 4.3, evidence=["missed run"]), replan(24, 4.6)])
    assert len(rows) == 1
    assert round(rows[0]["distance_changes"][0]["delta_miles"], 6) == .6
    assert rows[0]["material_evidence_since_anchor"] == ["missed run"]


def test_consumed_date_is_not_a_surprise_removal():
    before = replan(22, 4)
    after = {**replan(26, 4), "planned_sessions": []}
    rows = anchored_comparisons([before, after])
    assert len(rows) == 1
    assert rows[0]["date_slot_changes"] == []


def test_distant_new_horizon_date_is_not_original_visible_plan_churn():
    before, after = replan(22, 4), replan(24, 4)
    after["planned_sessions"].append({"planned_for": "2026-10-02T07:00:00+00:00", "workout_type": "easy", "midpoint_miles": 4})
    rows = anchored_comparisons([before, after])
    assert rows[0]["date_slot_changes"] == []
