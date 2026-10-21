from datetime import datetime, timezone
from contextlib import contextmanager
import sys

import pytest

from scripts.replay_synthetic_easy_dose import synthetic_history
from scripts.audit_easy_dose_interactions import allocation_budget
from run_analysis import weekly_schedule as planner


def test_synthetic_seed_has_no_personal_data_or_prescriptions():
    start = datetime(2026, 9, 22, tzinfo=timezone.utc)
    state, runs = synthetic_history(start)
    assert len(runs) == 36
    assert all(run.start_time < start and not run.projected for run in runs)
    assert all(run.prescribed_low_miles is None and run.prescribed_high_miles is None for run in runs)
    assert state.recent_load.trailing_28d.distance_miles == pytest.approx(63)
    assert state.typical_easy_run_miles == pytest.approx(3.75)
    assert state.running_days_28d == 12
    assert state.planned_weather is None


def test_allocation_budget_probe_restores_hook_on_failure():
    original = planner._allocate_visible_distance_ranges
    with pytest.raises(RuntimeError):
        with allocation_budget(assignments=8, finalists=16):
            assert planner._allocate_visible_distance_ranges is not original
            raise RuntimeError("intentional")
    assert planner._allocate_visible_distance_ranges is original


@pytest.mark.parametrize('funding_prototype,regrowth_prototype',[(False,False),(True,False),(True,True)])
def test_continuation_workflow_requires_expected_completion_without_a_separate_flag(monkeypatch, tmp_path,funding_prototype,regrowth_prototype):
    from scripts import replay_synthetic_easy_dose as replay
    from scripts import continuation_replan
    original_selector=continuation_replan.select_program
    from run_analysis.selection_objective import (
        EXPECTED_COMPLETION_STATE, VOLUME_PATH_SCORER, CONTINUATION_FUNDING_SCORER, build_with_selection_history,
    )

    class CheckedBuild(Exception):
        pass

    def build(*args, **kwargs):
        model = EXPECTED_COMPLETION_STATE.get()
        assert model is not None
        assert VOLUME_PATH_SCORER.get() is not None
        assert (CONTINUATION_FUNDING_SCORER.get() is not None)==funding_prototype
        assert (continuation_replan.select_program is not original_selector)==regrowth_prototype
        assert len(model.completed) == 36
        raise CheckedBuild

    @contextmanager
    def no_dose_work(mode):
        yield []

    def simulate(template, history, config, start, **kwargs):
        return build_with_selection_history(replay.projection.build_weekly_schedule,
            template, history, [template], None, config)

    monkeypatch.setattr(replay, 'replay_policy', no_dose_work)
    monkeypatch.setattr(replay.projection, 'build_weekly_schedule', build)
    monkeypatch.setattr(replay.projection, 'simulate_adherence', simulate)
    monkeypatch.setattr(sys, 'argv', ['replay', '--days', '1', '--calendar-neighbors',
        '--candidate-acceptance', '--finalized-scoring', '--opportunity-repair',
        '--canonical-targets', '--shared-objective', '--continuation-selection',
        '--output-json', str(tmp_path / 'report.json')])
    with pytest.raises(CheckedBuild):
        replay.main(funding_prototype=funding_prototype,regrowth_prototype=regrowth_prototype)
    assert EXPECTED_COMPLETION_STATE.get() is None
    assert VOLUME_PATH_SCORER.get() is None
    assert CONTINUATION_FUNDING_SCORER.get() is None
    assert replay.projection.build_weekly_schedule is build
    assert continuation_replan.select_program is original_selector
