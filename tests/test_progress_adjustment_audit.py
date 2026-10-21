from copy import deepcopy
from datetime import datetime,timedelta,timezone
from pathlib import Path

import pytest

from scripts.audit_progress_adjustments import match_pairs,summary,omission
from run_analysis.config import load_config
from run_analysis.pace_comparability import fit_comparison_calibration,compare_run_windows


def records():
    origin=datetime(2026,1,1,tzinfo=timezone.utc)
    return [dict(activity_id=i,start_time=(origin+timedelta(days=i*2)).isoformat(),fold=0,
        context=dict(device='watch',area='area',role='easy',grade=0.,headwind=0.,crosswind=0.,hr=148.,minutes=20.,heat=10.+i*4),
        full=10.,observed=10+i*.1,omitted=dict(weather=10+i*.1)) for i in range(4)]


def test_matching_uses_context_not_outcomes_and_no_run_is_reused():
    rows=records();first=match_pairs(rows,'weather',7)
    for i,r in enumerate(rows):r.update(full=100-i*10,observed=i*100,omitted={'weather':i*500})
    second=match_pairs(list(reversed(rows)),'weather',7)
    ids=lambda result:[(p['earlier_id'],p['later_id']) for p in result['pairs']]
    assert ids(first)==ids(second)
    all_ids=[i for pair in ids(first) for i in pair]
    assert len(all_ids)==len(set(all_ids))
    assert summary(first['pairs'])['after_minus_before_seconds']<0


def test_missing_context_and_changed_device_cannot_be_called_matched():
    rows=records()[:2];rows[1]['context']['heat']=None
    assert not match_pairs(rows,'weather',7)['pairs']
    rows=records()[:2];rows[1]['context']['device']='other'
    assert not match_pairs(rows,'weather',7)['pairs']
    assert summary([])==dict(pairs=0,calendar_blocks=0)


def test_omitting_each_correction_is_an_exact_reference_identity_without_mutation():
    config=load_config(Path(__file__).parents[1]/'config.example.yaml')
    cal=fit_comparison_calibration([],config)
    row=dict(moving_pace_min_mile=10.,average_hr_bpm=config['target_hr'],
        moving_minutes_into_run=config['reference_conditions']['within_run_minutes'],grade_energy_ratio=1.,
        headwind_signed_mph=0.,crosswind_mph=0.,temperature_f=config['reference_conditions']['temperature_f'],
        dewpoint_f=config['reference_conditions']['dewpoint_f'])
    before=deepcopy((row,cal))
    for factor in ('terrain','wind','heart_rate','duration','weather'):
        assert omission([row],[1.],config,cal,factor)==pytest.approx(10.)
    assert (row,cal)==before


def test_heat_omission_does_not_remove_hr_or_terrain_corrections():
    config=load_config(Path(__file__).parents[1]/'config.example.yaml');cal=fit_comparison_calibration([],config)
    row=dict(moving_pace_min_mile=10.,average_hr_bpm=config['target_hr']+5,
        moving_minutes_into_run=config['reference_conditions']['within_run_minutes'],grade_energy_ratio=1.1,
        headwind_signed_mph=0.,crosswind_mph=0.,temperature_f=85.,dewpoint_f=75.)
    omitted=omission([row],[1.],config,cal,'weather')
    cool=dict(row,temperature_f=config['reference_conditions']['temperature_f'],dewpoint_f=config['reference_conditions']['dewpoint_f'])
    assert omitted==pytest.approx(compare_run_windows([cool],[1.],config,cal)['pace_min_mile'])
