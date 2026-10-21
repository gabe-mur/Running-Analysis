"""Fixed-input attribution controls, not forecast acceptance."""
from dataclasses import replace

import pytest

from run_analysis.hr_response import ResponseInterval
from run_analysis.phase_calibration import mask_trace
from run_analysis.signal_structure import SignalSpan
from scripts.audit_quality_response_inputs import diagnostic_grid, response_cells
from scripts.compare_requested_quality_response import trace_metrics

MODEL = dict(intercept=.3, slope=.12, tau_seconds=45., speed_domain=(0.,6.))
LAYOUT = (('nonwork',0.,60.), ('work',60.,120.), ('nonwork',120.,180.))
PROGRAM = ((60.,2.),(60.,4.),(60.,2.))
ROWS = (ResponseInterval(0,60,2.,.3,.5), ResponseInterval(60,120,4.,.5,.8),
        ResponseInterval(120,180,2.,.8,.6))


def test_grid_and_all_cells_preserve_identical_support_without_phase_resets():
    result = response_cells(ROWS,LAYOUT,PROGRAM,MODEL)
    assert len(result['cells'])==6 and len(result['segments'])==1
    assert result['known_input_seconds']==180
    for spans,diag in result['cells'].values():
        assert sum(s.end-s.start for s in spans)==180
        assert diag['segment_initializations']==1
        assert spans[120].first==spans[119].last


def test_gaps_are_not_filled_and_initialization_policy_is_explicit():
    rows = (ROWS[0],replace(ROWS[1],start=70),ROWS[2])
    result = response_cells(rows,LAYOUT,PROGRAM,MODEL)
    assert result['known_input_seconds']==170
    assert [s['start'] for s in result['segments']]==[0,70]
    for spans,diag in result['cells'].values():
        assert diag['segment_initializations']==2
        assert not any(s.start<70 and s.end>60 for s in spans)
    fixed = result['cells']['recorded_fixed'][0]
    measured = result['cells']['recorded_measured'][0]
    assert next(s.first for s in fixed if s.start==70)==.45
    assert next(s.first for s in measured if s.start==70)==.5


def test_later_observed_hr_cannot_affect_any_cell_without_a_gap():
    changed = tuple(replace(r,first=r.first if i==0 else .1,last=.1) for i,r in enumerate(ROWS))
    a = response_cells(ROWS,LAYOUT,PROGRAM,MODEL)
    b = response_cells(changed,LAYOUT,PROGRAM,MODEL)
    assert a['cells']==b['cells']
    changed_initial = (replace(ROWS[0],first=.6),*ROWS[1:])
    c = response_cells(changed_initial,LAYOUT,PROGRAM,MODEL)
    assert c['cells']['recorded_fixed']==a['cells']['recorded_fixed']
    assert c['cells']['recorded_measured']!=a['cells']['recorded_measured']


def test_measured_phase_mean_and_variation_are_separate_not_fitted_models():
    rows = (ResponseInterval(0,30,1.,.3,.4),ResponseInterval(30,60,3.,.4,.5),*ROWS[1:])
    result = response_cells(rows,LAYOUT,PROGRAM,MODEL)
    assert result['phase_mean_speeds'][0]==2.
    assert result['cells']['prior_fixed']==result['cells']['phase_mean_fixed']
    assert result['cells']['recorded_fixed']!=result['cells']['phase_mean_fixed']


def test_boundary_splits_do_not_create_new_hr_initializations():
    rows=(ResponseInterval(0,180,2.,.3,.6),)
    layout=(('nonwork',0,60.25),('work',60.25,120.25),('nonwork',120.25,180))
    result=response_cells(rows,layout,((60.25,2),(60,4),(59.75,2)),MODEL)
    assert len(result['segments'])==1
    assert any(s.end==60.25 for s in result['cells']['prior_fixed'][0])
    assert all(d['segment_initializations']==1 for _,d in result['cells'].values())


def test_phase_mean_is_time_weighted_over_known_inputs_not_missing_time():
    rows=(ResponseInterval(0,10,1.,.3,.4),ResponseInterval(20,50,3.,.4,.6),*ROWS[1:])
    result=response_cells(rows,LAYOUT,PROGRAM,MODEL)
    assert result['phase_mean_speeds'][0]==2.5
    assert result['known_input_seconds']==160


def test_actual_mask_and_invalid_response_are_identical_and_not_clipped():
    result=response_cells(ROWS,LAYOUT,PROGRAM,MODEL)
    actual=mask_trace((SignalSpan(0,40,.4,.4),SignalSpan(80,180,.6,.6)),result['input_support'])
    metrics=[trace_metrics(spans,actual,LAYOUT,130)['whole'] for spans,_ in result['cells'].values()]
    assert all(r['seconds']==140 for r in metrics)
    bad=response_cells(ROWS,LAYOUT,PROGRAM,dict(MODEL,slope=.5))
    spans,diag=bad['cells']['prior_fixed']
    assert diag['hr_outside_reference']
    assert trace_metrics(spans,actual,LAYOUT,130)['whole']['predicted_load'] is None
    assert max(s.last for s in spans)>1


def test_unknown_inputs_and_wrong_prior_layout_do_not_invent_programmes():
    assert diagnostic_grid((),LAYOUT)==()
    assert response_cells((),LAYOUT,PROGRAM,MODEL) is None
    with pytest.raises(ValueError,match='exact diagnostic phase layout'):
        response_cells(ROWS,LAYOUT,((60,2),(30,4),(60,2)),MODEL)
