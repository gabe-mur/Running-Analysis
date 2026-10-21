from copy import deepcopy
from datetime import timedelta
import sys

import pytest

from scripts.audit_training_opportunities import run_comparison
from scripts.replay_synthetic_easy_dose import synthetic_history
from scripts.trace_opportunity_survival import (
    TARGET, allocation_trace, nested_code, paired_allocations, source_line, trace_calendar_prefix,
    timing_probe,
)
from run_analysis import weekly_schedule as planner
from test_recorded_hr_replay import START
from test_recommendation import CONFIG


def test_trace_markers_resolve_unambiguously_against_current_allocator():
    original = planner._allocate_visible_distance_ranges
    assert nested_code(original, "solve").co_name == "solve"
    assert source_line(original, "dp = next_dp") > original.__code__.co_firstlineno
    with pytest.raises(ValueError, match="changed or ambiguous"):
        source_line(original, "missing marker")


def test_hooks_and_python_tracing_restore_after_exception():
    originals = (planner._allocate_visible_distance_ranges, planner._joint_candidate_program_cost,
                 planner._adaptive_run_day_offsets_for_frequency)
    previous = sys.gettrace()
    with pytest.raises(RuntimeError):
        with paired_allocations(), trace_calendar_prefix(TARGET), allocation_trace(originals[0]):
            assert sys.gettrace() is not previous
            raise RuntimeError("intentional failure")
    assert sys.gettrace() is previous
    assert originals == (planner._allocate_visible_distance_ranges, planner._joint_candidate_program_cost,
                         planner._adaptive_run_day_offsets_for_frequency)


def sample_report():
    _, runs = synthetic_history(START)
    decision = dict(at=START.isoformat(), plan=[
        dict(planned_for=(START + timedelta(days=i, hours=7)).isoformat(), workout_type="easy")
        for i in (1, *TARGET[1:])])
    return dict(config=CONFIG, decisions=[decision] * 4, recorded_inputs=[
        dict(start_time=r.start_time.isoformat(),
             available_at=(r.start_time + timedelta(minutes=r.moving_minutes)).isoformat(),
             distance_miles=r.distance_miles, moving_minutes=r.moving_minutes,
             workout_type=r.workout_type.value, difficulty=r.difficulty.model_dump(mode="json"))
        for r in runs])


def test_same_state_dose_probes_preserve_all_selected_programs():
    report = sample_report()
    before = deepcopy(report)
    baseline = run_comparison(report, 3, unrestricted_opening=True)
    previous = sys.gettrace()
    with paired_allocations() as pairs:
        traced = run_comparison(report, 3, unrestricted_opening=True)
    assert pairs
    for old, new in zip(baseline["comparisons"], traced["comparisons"], strict=True):
        assert old["sessions"] == new["sessions"]
        assert old["total_cost"] == new["total_cost"]
    assert all(row["normal_trace"]["materialized"] for row in pairs)
    assert all(row["shorter"]["assignment"][0] <= 3.75 for row in pairs)
    assert sys.gettrace() is previous and report == before


def test_timing_probe_is_bounded_and_does_not_change_inputs_or_hooks(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Timing probe must stop before anchors or full scoring")

    monkeypatch.setattr(planner, "_cadence_anchor_candidates", forbidden)
    monkeypatch.setattr(planner, "_joint_candidate_program_cost", forbidden)
    report = sample_report()
    before = deepcopy(report)
    original = planner._adaptive_run_day_offsets_for_frequency
    previous = sys.gettrace()
    result = timing_probe(report)
    assert len(result["probes"]) == 2
    for probe in result["probes"]:
        trace = probe["trace"][0]
        assert len(trace["prefixes"]) == len(TARGET)
        assert "after_beam" in trace and "after_anchors" not in trace
    assert result["probes"][1]["raw_time"].hour == 7
    assert report == before
    assert sys.gettrace() is previous
    assert planner._adaptive_run_day_offsets_for_frequency is original
