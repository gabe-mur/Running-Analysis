from copy import deepcopy
from datetime import timedelta

import pytest

from scripts.audit_training_opportunities import available_history, restrict_opening_dose, run_comparison, summarize, tail_calendars, UnavailableDose
from run_analysis import weekly_schedule as planner
from test_recorded_hr_replay import START, incoming


def test_available_history_preserves_hr_without_importing_old_prescriptions():
    record = incoming()
    row = dict(available_at=record.available_at.isoformat(),
        start_time=record.run.start_time.isoformat(), distance_miles=record.run.distance_miles,
        moving_minutes=record.run.moving_minutes, workout_type=record.run.workout_type.value,
        difficulty=record.run.difficulty.model_dump(mode="json"))
    report = {"recorded_inputs": [row]}
    before = deepcopy(report)
    assert available_history(report, record.available_at - timedelta(microseconds=1)) == []
    history = available_history(report, record.available_at)
    assert history[0].difficulty == record.run.difficulty
    assert history[0].difficulty.zone_load == 79.2
    assert history[0].prescribed_low_miles is None
    assert report == before


@pytest.mark.parametrize("mode, expected", [("ordinary", 4.), ("shorter", 3.)])
def test_dose_branches_only_restrict_existing_options_and_restore(monkeypatch, mode, expected):
    def options(*args, **kwargs):
        return [3., 3.5, 4.], {4.: (3.7, 4.2)}
    monkeypatch.setattr(planner, "_easy_allocation_options", options)
    with restrict_opening_dose(START.date(), 3.95, mode):
        values, ranges = planner._easy_allocation_options(plan_date=START.date())
        assert values == [expected]
        assert ranges == ({4.: (3.7, 4.2)} if mode == "ordinary" else {})
        assert planner._easy_allocation_options(plan_date=(START + timedelta(days=1)).date()) == options()
    assert planner._easy_allocation_options is options


def test_unavailable_shorter_branch_does_not_fabricate_dose(monkeypatch):
    def options(*args, **kwargs):
        return [2.], {}
    monkeypatch.setattr(planner, "_easy_allocation_options", options)
    with pytest.raises(UnavailableDose, match="distinct"):
        with restrict_opening_dose(START.date(), 4., "shorter"):
            planner._easy_allocation_options(plan_date=START.date())
    assert planner._easy_allocation_options is options


def test_ordinary_branch_cannot_override_recovery_restriction(monkeypatch):
    def options(*args, **kwargs):
        return [2., 2.25], {}
    monkeypatch.setattr(planner, "_easy_allocation_options", options)
    with restrict_opening_dose(START.date(), 4., "ordinary"):
        assert planner._easy_allocation_options(plan_date=START.date()) == ([2.25], {})


@pytest.mark.parametrize("target_scale", [.85, 1., 1.15])
def test_full_comparison_preserves_opening_easy_purpose_and_common_inputs(target_scale):
    from scripts.replay_synthetic_easy_dose import synthetic_history
    from test_recommendation import CONFIG

    _, history = synthetic_history(START)
    report = dict(config=CONFIG, decisions=[dict(at=START.isoformat(), plan=[
        dict(planned_for=(START + timedelta(days=i, hours=7)).isoformat(), workout_type="easy")
        for i in (1, 3, 5, 8, 10, 12, 15, 17, 19)])], recorded_inputs=[
        dict(start_time=r.start_time.isoformat(),
             available_at=(r.start_time + timedelta(minutes=r.moving_minutes)).isoformat(),
             distance_miles=r.distance_miles, moving_minutes=r.moving_minutes,
             workout_type=r.workout_type.value, difficulty=r.difficulty.model_dump(mode="json"))
        for r in history])
    before = deepcopy(report)
    original_search, original_options = planner.adaptive_run_day_offsets, planner._easy_allocation_options
    result = run_comparison(report, 0, target_scale=target_scale, unrestricted_opening=True)
    assert report == before
    assert planner.adaptive_run_day_offsets is original_search
    assert planner._easy_allocation_options is original_options
    assert result["known_runs"] == len(history)
    assert result["prior_schedule"] is None
    assert result["target_scale"] == target_scale
    for actual, original in zip(result["targets"], result["unscaled_targets"]):
        assert actual == pytest.approx([value * target_scale for value in original])
    assert result["comparisons"][-1]["label"] == "unrestricted_today"
    assert result["comparisons"][-1]["branch_realized"]
    rows = result["comparisons"][:3]
    assert len(rows) == 3
    assert all(row["branch_realized"] for row in rows)
    assert [row["sessions"][0]["offset"] for row in rows] == [0, 0, 1]
    assert all(row["sessions"][0]["type"] == "easy" for row in rows)
    assert sum(rows[1]["sessions"][0]["distance_range"]) < sum(rows[0]["sessions"][0]["distance_range"])
    assert len({row["details"]["integrated_target_low_miles"] for row in rows}) == 1
    assert rows[0]["details"]["integrated_target_low_miles"] == pytest.approx(sum(x[0] for x in result["targets"][:21]) / 7)


def test_tail_pool_includes_displacement_and_sparse_alternatives_without_leaking_dates():
    seed = (2, 4, 6)
    pool = tail_calendars(seed, set(range(2, 8)))
    assert seed in pool
    assert (3, 4, 6) in pool  # clear tomorrow's immediate next-day collision
    assert (3, 5, 6) in pool  # allow both downstream sessions to move
    assert (2, 3, 4, 6) in pool  # added opportunity
    assert (4, 6) in pool and (6,) in pool  # one/two removed opportunities
    assert len(pool) == len(set(pool))
    assert all(set(x) <= set(range(2, 8)) and len(x) == len(set(x)) for x in pool)
    assert tail_calendars(reversed(seed), set(range(2, 8))) == pool
    protected = tail_calendars(seed, {2, 4, 5, 6, 7})
    assert not any(3 in candidate for candidate in protected)
    with pytest.raises(ValueError, match="allowed tail"):
        tail_calendars(seed, {2, 4})


def test_summary_never_promotes_unavailable_or_rejected_low_cost_candidates():
    def row(label, count, cost, funded=False, eligible=True):
        return dict(label=label, offsets=tuple(range(count)), total_cost=cost,
            run_count=count, planned_miles=50., integrated_band_funded=funded,
            date_gaps_days=[1], seconds=.1, eligible=eligible)
    result = summarize([row("today", 8, 1.), row("today", 9, 2., funded=True),
        row("tomorrow", 8, -99., funded=True, eligible=False),
        dict(label="shorter", unavailable="No permitted dose", seconds=.1)])
    assert result["evaluated"] == 4 and result["eligible"] == 2
    assert result["best_by_branch"]["tomorrow"] is None
    assert result["best_by_branch"]["shorter"] is None
    assert result["funded_by_count"]["8"] is None
    assert result["funded_by_count"]["9"]["total_cost"] == 2.


def test_spread_controls_extend_pool_without_replacing_local_candidates():
    seed = (2, 4, 6, 9, 12, 13, 17, 18, 20)
    allowed = set(range(2, 21))
    local = set(tail_calendars(seed, allowed))
    extended = set(tail_calendars(seed, allowed, spread_references=True))
    assert local < extended
    assert {len(x) + 1 for x in extended - local} == {8, 9, 10, 11}
    assert len(extended) <= 100
    assert all(set(x) <= allowed and len(x) == len(set(x)) for x in extended)
