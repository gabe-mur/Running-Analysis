from __future__ import annotations

from run_analysis.processing import _eligibility
from run_analysis.processing import process_activities
from run_analysis.config import load_config
from run_analysis.db import connect
from run_analysis.importer import import_files
from test_tcx import make_tcx
import pytest


def test_prescription_only_workout_type_is_a_valid_partial_override() -> None:
    config = {
        "model": {
            "minimum_run_minutes": 10,
            "minimum_gps_coverage": 0.8,
            "minimum_hr_coverage": 0.8,
            "maximum_stop_fraction": 0.35,
        },
        "activity_classification": {
            "high_confidence_walk_pace_min_mile": 18,
            "high_confidence_walk_cadence_max_spm": 110,
            "high_confidence_bike_pace_min_mile": 5.5,
        },
    }

    eligible, reasons = _eligibility(
        {"total_distance_m": 0},
        [],
        [],
        {"moving_time_s": 0, "stopped_time_s": 0, "analysis_distance_m": 0},
        config,
        {"workout_type": "easy"},
    )

    assert eligible is False
    assert "manual_exclusion" not in reasons
    assert "workout_type_easy" not in reasons


def test_targeted_repair_preserves_unrelated_fitted_observations(tmp_path):
    make_tcx(tmp_path/'first.tcx')
    make_tcx(tmp_path/'second.tcx',start='2024-07-02T12:00:00Z',
        end='2024-07-02T12:00:20Z',activity_id='2024-07-02T12:00:00Z')
    config=load_config()
    with connect(tmp_path/'app.sqlite') as db:
        import_files(db,tmp_path,'America/New_York')
        process_activities(db,config)
        first,second=[r[0] for r in db.execute('SELECT id FROM activities ORDER BY id')]
        db.execute('UPDATE activity_metrics SET standardized_pace_at_target_hr_min_mile=9.1 WHERE activity_id=?',(second,))
        db.commit()
        previous=dict(db.execute('SELECT * FROM activity_metrics WHERE activity_id=?',(second,)).fetchone())
        result=process_activities(db,config,force=True,activity_ids=(first,))
        assert result.discovered_activities==result.processed_activities==1
        assert dict(db.execute('SELECT * FROM activity_metrics WHERE activity_id=?',(second,)).fetchone())==previous
        assert process_activities(db,config,force=True,activity_ids=()).processed_activities==0
        with pytest.raises(ValueError,match='Positive internal'):
            process_activities(db,config,activity_ids=(True,))
