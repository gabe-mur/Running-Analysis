from dataclasses import replace
from datetime import timedelta
from copy import deepcopy
import pytest

from test_athlete_projection import NOW,KEY,point,phase,session
from test_performance_reference import steady
from run_analysis.athlete_projection import snapshot,forecast_trace,ForecastPhase,accounting_signature
from run_analysis.phase_calibration import (
    PhaseResidual,correction,calibrated_forecast,learn_phase_residuals,clip_trace,mask_trace,trace_load,observed_phases,
)
from run_analysis.signal_structure import SignalSpan
from run_analysis.response_model import Policy


def sample(key='prior',kind='easy',pace=1.,load=1.,seconds=600,**changes):
    return replace(PhaseResidual(key,kind,NOW-timedelta(days=1),KEY,seconds,pace,load,1.,'test'),**changes)


def fit(rows,phase='easy',seconds=600,key=KEY):
    return correction(rows,as_of=NOW,accounting_key=key,phase=phase,seconds=seconds)


def test_unseen_phase_and_no_history_keep_explicit_fallback():
    c=fit([])
    assert c['pace_ratio']['factor']==c['load_ratio']['factor']==1
    assert c['load_ratio']['source_count']==0
    assert fit([sample(kind='work')])==c


def test_future_and_different_accounting_cannot_change_correction():
    prior=sample(load=.8)
    later=sample('future',load=2,available_at=NOW+timedelta(seconds=1))
    other=sample('other',load=3,accounting_key='other')
    assert fit([prior])==fit([prior,later,other])


def test_recent_upload_of_old_run_does_not_make_it_recent_calibration_evidence():
    late=sample(completed_at=NOW-timedelta(days=500),available_at=NOW-timedelta(hours=1),load=2)
    assert fit([late])['load_ratio']['source_count']==0


def test_duplicate_repetitions_do_not_multiply_session_weight():
    a,b=sample('a',load=.8),sample('b',load=1.2)
    assert fit([a,b])['load_ratio']['factor']==pytest.approx(fit([a]*20+[b])['load_ratio']['factor'])
    assert fit([a,b])['load_ratio']['source_count']==2


def test_duration_relevance_is_continuous_and_does_not_reject_novel_dose():
    a=sample('a',load=.8,seconds=100)
    b=sample('b',load=1.2,seconds=300)
    assert fit([a,b],seconds=199.999)['load_ratio']['factor']==pytest.approx(fit([a,b],seconds=200.001)['load_ratio']['factor'],rel=1e-5)
    assert fit([a],seconds=1200)['load_ratio']['source_count']==1


def test_missing_hr_does_not_mean_zero_load_or_remove_pace_evidence():
    c=fit([sample(pace=.9,load=None,hr_coverage=0)])
    assert c['pace_ratio']['factor']==pytest.approx(.9)
    assert c['load_ratio']['factor']==1
    assert c['load_ratio']['source_count']==0


def test_phase_load_factors_do_not_change_prescribed_distance():
    state=snapshot([],NOW,KEY)
    phases=[ForecastPhase('easy',None,2000,point(2),point(.6),'test')]
    result=calibrated_forecast(state,phases,[sample(pace=.8,load=.9)],initial_reserve=.6)
    assert result['distance_meters']==2000
    assert result['execution_seconds']==pytest.approx(1250)
    assert result['cardio_load']==pytest.approx(1250/60*.9)
    assert not result['planner_influence']


def test_distance_balancing_after_pace_correction_preserves_total():
    phases=[phase('warmup',seconds=100),ForecastPhase('finish_easy',None,2000,point(2),point(.6),'test',True)]
    result=calibrated_forecast(snapshot([],NOW,KEY),phases,[sample(kind='warmup',pace=.8),sample(kind='cooldown',pace=1.2)])
    assert result['distance_meters']==pytest.approx(2000)
    assert result['phases'][0]['meters']==pytest.approx(160)


def test_exhausted_distance_is_not_silently_inflated_or_clipped():
    phases=[phase('warmup',seconds=600),ForecastPhase('finish_easy',None,2000,point(2),point(.6),'test',True)]
    with pytest.raises(ValueError,match='exhaust'):
        calibrated_forecast(snapshot([],NOW,KEY),phases,[sample(kind='warmup',pace=2)])


def observations(spans=None):
    return [dict(phase='easy',start=0.,end=600.,meters=1200.,moving_seconds=600.,movement_reliable=True,
        spans=spans if spans is not None else (SignalSpan(0,600,.6,.6),),equilibrium_reserve=point(.6),origin='test')]


def completion(**changes):
    return replace(session('done'),start=NOW,end=NOW+timedelta(seconds=600),
        available_at=NOW+timedelta(seconds=600),moving_seconds=600,meters=1200,**changes)


def test_shared_accounting_recovers_identity_without_self_fitting():
    state=snapshot([steady()],NOW,KEY)
    rows=learn_phase_residuals(state,completion(),observations(),initial_reserve=.6)
    assert rows[0].load_ratio==pytest.approx(1)
    assert rows[0].pace_ratio==pytest.approx(1)
    assert fit(rows)['load_ratio']['source_count']==0
    assert correction(rows,as_of=completion().available_at,accounting_key=KEY,phase='easy',seconds=600)['load_ratio']['source_count']==1


def test_gap_uses_same_known_timestamps_on_both_sides():
    rows=learn_phase_residuals(snapshot([steady()],NOW,KEY),completion(),observations((SignalSpan(0,200,.6,.6),SignalSpan(400,600,.6,.6))),initial_reserve=.6)
    assert rows[0].load_ratio==pytest.approx(1)
    assert rows[0].hr_coverage==pytest.approx(2/3)


def test_excluded_capacity_does_not_train_corrections():
    assert not learn_phase_residuals(snapshot([],NOW,KEY),completion(capacity_eligible=False),observations())


def test_unbounded_cardio_reference_is_not_calibrated_from_remaining_in_range_hr():
    rows=learn_phase_residuals(snapshot([steady()],NOW,KEY),completion(cardio=None),observations(),initial_reserve=.6)
    assert rows[0].load_ratio is None
    assert rows[0].pace_ratio==pytest.approx(1)


def test_old_data_changes_neither_current_calibration_nor_absolute_pace_reference():
    from run_analysis.performance_reference import phase_pace_reference
    old=steady('old',days=500,speed=10)
    recent=steady('recent',days=1,speed=2)
    state=snapshot([old,recent],NOW,KEY)
    ref=phase_pace_reference(state,'easy',equilibrium_reserve=point(.6))
    assert ref['source_keys']==('recent',)
    assert ref['speed_mps'].estimate==2
    c=fit([sample('old',pace=.9,available_at=old.end)])
    assert c['pace_ratio']['factor']==1
    assert c['pace_ratio']['source_count']==0
    # The old sample can inform a forecast issued near its original date.
    historical=correction([sample('old',pace=.9,available_at=old.end)],
        as_of=old.end+timedelta(days=1),accounting_key=KEY,phase='easy',seconds=600)
    assert historical['pace_ratio']['factor']==pytest.approx(.9)


def test_issued_result_does_not_change_when_new_samples_arrive():
    state=snapshot([],NOW,KEY);history=[sample(load=.8)]
    first=calibrated_forecast(state,[phase()],history)
    original=deepcopy(first)
    history.append(sample('future',load=2,available_at=NOW+timedelta(seconds=1)))
    assert calibrated_forecast(state,[phase()],history)==first==original


def test_nondefault_accounting_cannot_silently_use_default_kernel():
    key=accounting_signature(50,200,replace(Policy(),intensity_exponent=1.67))
    with pytest.raises(ValueError,match='accounting'):
        calibrated_forecast(snapshot([],NOW,key),[phase()],[])


@pytest.mark.parametrize('change',[dict(seconds=0),dict(load_ratio=-1),dict(hr_coverage=2),dict(available_at=NOW.replace(tzinfo=None))])
def test_invalid_samples_are_rejected(change):
    with pytest.raises(ValueError):sample(**change) if 'seconds' in change else replace(sample(),**change)


def test_trace_mask_preserves_gaps_and_exact_load():
    trace=[SignalSpan(0,100,.6,.6),SignalSpan(100,200,.6,.6)]
    masked=mask_trace(trace,[SignalSpan(25,75,0,0),SignalSpan(150,200,0,0)])
    assert sum(s.end-s.start for s in masked)==100
    assert trace_load(masked)==pytest.approx(100/60)


def test_original_phase_adapter_does_not_relabel_easy_fluctuation_as_work(tmp_path):
    from run_analysis.projection_evidence import evidence_from_activity
    from run_analysis.tcx import parse_tcx
    from run_analysis.workload_profile import observed_profile
    from test_tcx import make_tcx
    from test_workload_profile import CONFIG
    activity=parse_tcx(make_tcx(tmp_path/'run.tcx')).activities[0]
    template=activity.trackpoints[0];start=activity.start_time_utc
    activity.trackpoints=[replace(template,timestamp_utc=start+timedelta(seconds=s),distance_m=s*2,
        heart_rate_bpm=145,point_index=s//10) for s in range(0,2401,10)]
    activity.total_elapsed_time_s=activity.lap_recorded_time_s=2400
    activity.total_distance_m=4800
    profile=observed_profile(activity,CONFIG,workout_type='easy')
    evidence=evidence_from_activity(activity,CONFIG,key='test',available_at=start+timedelta(seconds=2400),workout_type='easy',profile=profile)
    rows=observed_phases(activity,evidence,CONFIG,profile)
    assert [r['phase'] for r in rows]==['warmup','easy']
    assert sum(r['end']-r['start'] for r in rows)==2400
    assert sum(r['meters'] for r in rows)==pytest.approx(4800)
