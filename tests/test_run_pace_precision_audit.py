from copy import deepcopy
from pathlib import Path
import math

import numpy as np
import pytest

from run_analysis.config import load_config
from run_analysis import pace_comparability as pc
from scripts.audit_run_pace_precision import sources,sampling_precision,interleaved_check
from tests.test_pace_comparability import row


def test_source_decomposition_matches_exact_shipped_uncertainty_without_mutation():
    config=load_config(Path(__file__).parents[1]/'config.example.yaml')
    calibration=pc.fit_comparison_calibration([],config)
    rows=[row(config,moving_pace_min_mile=10+.2*np.sin(i),moving_minutes_into_run=10+i,
        grade_energy_ratio=1.03 if i%3 else 1.,headwind_signed_mph=2.,crosswind_mph=1.) for i in range(30)]
    rows[0]['temperature_f']=None;before=deepcopy(rows);weights=[1.]*len(rows)
    result=pc.compare_run_windows(rows,weights,config,calibration)
    speeds,parts=sources(rows,weights,config,calibration)
    center,boot=sampling_precision(rows,weights,speeds,32,77)
    assert center==pytest.approx(result['pace_min_mile'])
    assert center*math.expm1(1.96*math.sqrt(sum(parts.values())))==pytest.approx(result['uncertainty_95_min_mile'])
    assert all(v>=0 for v in parts.values())
    assert rows==before
    assert boot==sampling_precision(rows,weights,speeds,32,77)[1]


def test_bootstrap_counts_temporal_blocks_not_overlapping_windows():
    rows=[dict(moving_minutes_into_run=i/2) for i in range(80)]
    speeds=np.array([3.]*80)
    _,result=sampling_precision(rows,[1.]*80,speeds,16,55)
    assert [v['blocks'] for v in result]==[8,4]
    assert all(v['sampling_sd_min_mile']<1e-10 for v in result)
    assert sampling_precision(rows[:10],[1.]*10,speeds[:10],16,55)[1]==[]


def test_interleaved_comparison_uses_supported_disjoint_sections_and_preserves_inputs():
    config=load_config(Path(__file__).parents[1]/'config.example.yaml')
    calibration=pc.fit_comparison_calibration([],config)
    rows=[row(config,moving_minutes_into_run=i) for i in range(80)]
    before=deepcopy(rows)
    result=interleaved_check(rows,[1.]*len(rows),config,calibration,16,55)
    assert [v['block_minutes'] for v in result]==[5,10]
    assert all(abs(v['log_pace_difference'])<1e-10 for v in result)
    assert all(v['proposed']['covered_95'] for v in result)
    assert interleaved_check(rows[:20],[1.]*20,config,calibration,16,55)==[]
    assert rows==before
