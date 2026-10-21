from datetime import datetime, timedelta, timezone

import pytest

from scripts.audit_hr_interpolation import audit
from run_analysis.objective_modeling import speed_to_pace_min_mile


def rows():
    result=[]
    for run in range(30):
        for minute,hr in ((12,144),(14,145),(16,148),(18,151),(20,152),(22,148),(24,145),(26,148)):
            result.append(dict(activity_id=run,start_time_utc=(datetime(2026,1,1,tzinfo=timezone.utc)+timedelta(days=run)).isoformat(),
                moving_minutes_into_run=minute,average_hr_bpm=hr,
                moving_pace_min_mile=speed_to_pace_min_mile(2.3+run*.01+.02*(hr-148)),
                heart_rate_change_bpm=0,speed_change_fraction=0,grade_energy_ratio=1,average_grade_percent=0))
    return result


CONFIG={'target_hr':148,'zones':{'z2':[141,153]},'reference_conditions':{'within_run_minutes':20}}


def test_direct_target_check_recovers_known_interpolation_with_chronological_run_holdout():
    source=rows();result=audit(source,source,CONFIG)
    assert result['held_out_matched_runs']==15
    assert result['run_balanced_mae_seconds_per_mile']['hr_only']==pytest.approx(0,abs=1e-8)
    assert result['run_balanced_mae_seconds_per_mile']['hr_controlling_time']==pytest.approx(0,abs=1e-8)
    assert result['comparisons']['hr_only']['run_bootstrap_95'][1]<0
    assert result['pair_counts_by_direction']['anchor_below_target']>0
    assert result['pair_counts_by_direction']['anchor_above_target']>0


def test_missing_stability_evidence_cannot_pass_as_zero_change():
    source=rows()
    for row in source: row['heart_rate_change_bpm']=None
    assert audit(source,source,CONFIG)['status']=='insufficient_matched_observations'


def test_time_gap_cannot_be_relaxed_to_make_distant_samples_validate():
    source=rows()
    for row in source:
        if row['average_hr_bpm']!=148:row['moving_minutes_into_run']=60
    assert audit(source,source,CONFIG)['status']=='insufficient_matched_observations'
