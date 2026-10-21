"""Diagnostic attribution arithmetic, not prediction validation."""
from math import exp

import pytest

from run_analysis.hr_response import ResponseInterval, predict_response
from run_analysis.signal_structure import SignalSpan
from scripts.audit_quality_response_inputs import diagnostic_grid
from scripts.audit_quality_response_bias import matched_cells, residual_associations, phase_halves, pace_inputs


def test_joint_associations_recover_known_speed_and_time_effects_without_unit_mix():
    cells = []
    for a, b, speed in ((0, 60, 1), (60, 120, 3), (120, 180, 2)):
        y0, y1 = (.03+.02*speed+.01*t/60 for t in (a, b))
        cells.append((a, b, speed, .3, .3, .3+y0, .3+y1))
    r = residual_associations(cells, 100)
    assert r['speed_association_bpm_per_mps']==pytest.approx(2)
    assert r['elapsed_association_bpm_per_minute']==pytest.approx(1)
    assert r['mean_actual_minus_model_bpm']==pytest.approx(8.5)
    assert r['residual_rmse_after_diagnostic_fit_bpm']==pytest.approx(0, abs=1e-6)
    assert r['prospective'] is False and r['causal'] is False


def test_joint_associations_do_not_report_identified_speed_with_constant_input():
    r = residual_associations(((0, 60, 2, .3, .3, .4, .5),), 100)
    assert r['status']=='joint_association_unidentified'
    assert 'speed_association_bpm_per_mps' not in r
    assert residual_associations((), 100)['known_seconds']==0


def test_association_duration_weighting_invariant_to_sampling_density():
    whole = ((0, 60, 1, .3, .3, .34, .36), (60, 120, 3, .3, .3, .36, .38))
    pieces = ((0, 30, 1, .3, .3, .34, .35), (30, 60, 1, .3, .3, .35, .36), whole[1])
    a, b = residual_associations(whole, 100), residual_associations(pieces, 100)
    for k in ('mean_actual_minus_model_bpm','residual_rmse_bpm',
              'speed_association_bpm_per_mps','elapsed_association_bpm_per_minute'):
        assert a[k]==pytest.approx(b[k], abs=1e-9)


def test_joint_cells_clip_interpolate_exact_mask_without_inventing_gaps():
    row = ResponseInterval(0, 60, 2, .3, .6)
    grid = ((0, row),)
    pred = (SignalSpan(0, 60, .4, .7),)
    actual = (SignalSpan(10, 20, .3, .4), SignalSpan(40, 50, .5, .6))
    cells = matched_cells(grid, pred, actual)
    assert sum(b-a for a, b, *_ in cells)==20
    assert cells[0]==pytest.approx((10, 20, 2, .45, .5, .3, .4))
    with pytest.raises(ValueError, match='boundaries'):
        matched_cells(grid, (SignalSpan(1, 60, .4, .7),), actual)
    with pytest.raises(ValueError, match='One response'):
        matched_cells(grid, (), actual)


def test_phase_halves_preserve_state_and_combine_reps_within_run():
    layout = (('nonwork', 0, 60), ('work', 60, 100), ('nonwork', 100, 120),
              ('work', 120, 160), ('nonwork', 160, 180))
    rows = (ResponseInterval(0, 180, 2, .3, .6),)
    grid = diagnostic_grid(rows, layout)
    spans, _ = predict_response(tuple(r for _, r in grid),
        dict(intercept=.3, slope=.1, tau_seconds=45, speed_domain=(0, 5)))
    last = .5+(.3-.5)*exp(-180/45)
    assert spans[-1].last==pytest.approx(last)
    actual = (SignalSpan(0, 180, .5, .5),)
    halves = phase_halves(spans, actual, layout, 100)
    assert halves['work_first']['seconds']==40
    assert halves['work_last']['seconds']==40
    assert halves['warmup_first']['seconds']==30
    assert halves['warmup_last']['seconds']==30
    assert spans[-1].last==pytest.approx(last)  # Scoring never resets response.


def test_half_unknown_time_stays_unknown_not_zero_or_full_bout():
    layout = (('nonwork', 0, 60), ('work', 60, 120), ('nonwork', 120, 180))
    halves = phase_halves((SignalSpan(0, 180, .4, .4),),
        (SignalSpan(20, 40, .5, .5),), layout, 100)
    assert halves['warmup_first']['seconds']==10
    assert halves['warmup_last']['seconds']==10
    assert halves['work_first'] is None


def test_pace_audit_does_not_assign_unknown_pace_or_fill_unknown_speed():
    layout = (('nonwork', 0, 60), ('work', 60, 120))
    rows = (ResponseInterval(10, 20, 1, .3, .4), ResponseInterval(30, 60, 3, .4, .5))
    r = pace_inputs(rows, layout, None)
    assert r[0]['measured_known_mean_speed_mps']==2.5
    assert r[0]['known_seconds']==40
    assert r[0]['prior_speed_fraction_error'] is None
    assert r[1]['measured_known_mean_speed_mps'] is None
