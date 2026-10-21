from dataclasses import replace
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from scripts.replay_recorded_hr import RecordedRun, known_runs, inputs_for, hr_reference, load_recorded, upload_comparison, write_report
from scripts.replay_synthetic_easy_dose import synthetic_history
from run_analysis import adherence_projection as projection
from run_analysis.web.schemas import RecommendationResponse, WorkoutType
from test_recommendation import CONFIG
from test_prescription_matching import _schedule


START = datetime(2026, 9, 20, tzinfo=ZoneInfo("America/New_York"))


def history():
    _, runs = synthetic_history(START)
    # Explicit observed fixture HR, consistently present in historical samples.
    return [replace(r, difficulty=r.difficulty.model_copy(update={"zone_load": r.moving_minutes * 2})) for r in runs]


def recommendation():
    return RecommendationResponse(generated_at=START, fitness_state_as_of=START,
        planned_for=START + timedelta(hours=19), workout_type="easy", title="Test",
        distance_range_miles=(3.7, 4.2), confidence="moderate", readiness="ready")


def incoming():
    rec = recommendation()
    when = START + timedelta(hours=18)
    run = projection.ProjectionRun(when, 3.95, 44., WorkoutType.EASY,
        projection._projected_difficulty(rec, 3.95, 44 / 3.95).model_copy(update={"zone_load": 79.2}))
    return RecordedRun(101, when + timedelta(minutes=50), run, .98)


def test_run_becomes_known_at_finish_not_at_start():
    record = incoming()
    assert known_runs([record], record.run.start_time) == []
    assert known_runs([record], record.available_at - timedelta(microseconds=1)) == []
    assert known_runs([record], record.available_at) == [record.run]


def test_future_poison_cannot_change_past_state_or_targets():
    records = [RecordedRun(i, r.start_time + timedelta(minutes=r.moving_minutes), r, .98)
               for i, r in enumerate(history())]
    original = inputs_for(known_runs(records, START), START, CONFIG)
    poisoned = replace(incoming(), run=replace(incoming().run, distance_miles=1000))
    changed = inputs_for(known_runs(records + [poisoned], START), START, CONFIG)
    assert original["daily_states"] == changed["daily_states"]
    assert original["daily_state_options"] == changed["daily_state_options"]
    assert original["target_distance_range"] == changed["target_distance_range"]
    assert original["target_evidence"] == changed["target_evidence"]


def test_upload_and_next_midnight_have_identical_observed_states():
    record = incoming()
    known = history() + [record.run]
    upload = inputs_for(known, record.available_at, CONFIG)
    morning = inputs_for(known, START + timedelta(days=1), CONFIG)
    assert upload["daily_states"] == morning["daily_states"]
    assert upload["daily_state_options"] == morning["daily_state_options"]
    assert upload["target_evidence"] == morning["target_evidence"]
    assert upload["daily_states"][0].as_of.date() == (START + timedelta(days=1)).date()


def test_hr_references_are_past_only_and_purpose_matched():
    rec = recommendation()
    before = history()
    expected = hr_reference(before, rec, START)
    assert expected["samples"] == 12
    assert expected["zone_load_per_minute_min_median_max"] == [2, 2, 2]
    assert hr_reference(before + [incoming().run], rec, START) == expected
    assert expected["midpoint_hr_load_using_prior_median"] == pytest.approx(3.95 * 11 * 2)


def test_hr_observation_is_preserved_and_attribution_does_not_mutate_it():
    record, rec = incoming(), recommendation()
    before = history()
    references = {START.date().isoformat(): hr_reference(before, rec, START)}
    observed = record.run.difficulty.model_dump()
    result = upload_comparison(record, before, _schedule(START, rec), CONFIG, references)
    assert result["hr_load"] == 79.2
    assert result["hr_coverage"] == .98
    assert result["distance_in_range"] and result["type_matches"]
    assert result["timing_delta_hours"] == -1
    assert result["prior_only_hr_reference"] is references[START.date().isoformat()]
    assert record.run.difficulty.model_dump() == observed
    assert result["recorded_recovery"] != result["same_actual_run_without_numeric_hr_recovery"]
    assert record.run.prescribed_low_miles is None


def test_upload_on_rest_day_is_not_called_compliant():
    record, rec = incoming(), recommendation()
    rec.planned_for += timedelta(days=1)
    result = upload_comparison(record, history(), _schedule(START, rec), CONFIG, {})
    assert result["same_date_prescription"] is None
    assert result["distance_in_range"] is None
    assert result["production_plan_projection_recovery"] is None


def test_loader_is_read_only_and_classifies_from_prefix_not_future(monkeypatch, tmp_path):
    import json
    import sqlite3

    def row(index, miles):
        seconds = miles * 11 * 60
        return dict(id=index, start_time_utc=(START + timedelta(days=index)).isoformat(),
                    total_distance_m=miles * 1609.344, total_elapsed_time_s=seconds,
                    calculated_moving_time_s=seconds, device_timer_time_s=seconds,
                    elapsed_time_s=seconds, analysis_distance_m=miles * 1609.344,
                    hr_zone_seconds_json=json.dumps({"z2": seconds}), hr_load_coverage=1.,
                    workout_type=None, perceived_exertion=None,
                    derived_elevation_gain_m=None, derived_elevation_loss_m=None)

    rows = [row(i, 4) for i in range(1, 5)] + [row(5, 7)]
    class Connection:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def execute(self, sql):
            assert "planned_workout_history" not in sql
            assert "model_runs" not in sql
            return self
        def fetchall(self): return list(rows)
    def connect(database_uri, **kwargs):
        assert database_uri.endswith("?mode=ro") and kwargs["uri"]
        return Connection()
    monkeypatch.setattr(sqlite3, "connect", connect)
    before = load_recorded(tmp_path / "unused.sqlite", START.tzinfo)
    rows.extend(row(i, 50) for i in range(6, 20))
    after = load_recorded(tmp_path / "unused.sqlite", START.tzinfo)
    assert before == after[:5]
    assert before[-1].run.workout_type == WorkoutType.LONG
    assert all(x.run.difficulty.zone_load is not None for x in before)
    assert all(x.run.prescribed_low_miles is None for x in before)


def test_report_handles_typed_yaml_dates_and_does_not_overwrite(tmp_path):
    import json
    path = tmp_path / "report.json"
    config = {"coaching": {"goal_date": START.date()}}
    write_report(path, {"config": config, "at": START})
    assert json.loads(path.read_text())["config"]["coaching"]["goal_date"] == "2026-09-20"
    assert config["coaching"]["goal_date"] == START.date()
    with pytest.raises(FileExistsError):
        write_report(path, {})
    broken = tmp_path / "broken.json"
    with pytest.raises(TypeError):
        write_report(broken, {"unserializable": object()})
    assert not broken.exists()
