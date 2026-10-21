from copy import deepcopy
from datetime import datetime,timedelta,timezone
import math

import pytest

from run_analysis.pace_forecast_error import (
    calibrate,error_history,error_radius,matched_error_growth,response_fits,weighted_quantile,weights,
)
from run_analysis.training_response import POLICIES
from scripts.audit_pace_outlook_error_margin import interval,matched_growth

AT=datetime(2025,1,1,tzinfo=timezone.utc)


def errors(power=.21):
    return [dict(origin=(AT+timedelta(days=i*14)).isoformat(),target_id=f'{i}_{h}',
        horizon=h,elapsed=h/28,outcome_end=(AT+timedelta(days=i*14+h)).isoformat(),
        outcome_ids=[i*2+(h==28)],regime='consistent',log_error=(.01+i*.001)*h**power,
        predicted=10.,actual=10*math.exp(-(.01+i*.001)*h**power))
        for i in range(12) for h in (14,28)]


def test_shared_calibration_matches_the_audited_formulas_without_saved_artifacts():
    rows=errors();model=calibrate({28:rows,42:rows})
    assert model['exponent']==pytest.approx(.21)
    assert model['exponent']==pytest.approx(matched_growth(rows)['exponent'])
    assert model['minimum_horizon_days']==14
    for days in (14,28,84,211):
        assert error_radius(model,days)==pytest.approx(interval(rows,days=days,pace=9.,exponent=.21)['absolute_error_80_log'])
    for days in (0,.01,1,7,13.99):
        assert error_radius(model,days) is None


def test_earliest_supported_range_uses_actual_elapsed_time_not_the_nominal_horizon():
    rows=[{**r,'elapsed':r['elapsed']+3/28} for r in errors()]
    model=calibrate({28:rows,42:rows})
    assert model['minimum_horizon_days']==17
    assert error_radius(model,14) is None
    assert error_radius(model,17)>0
    for days in (-1,float('nan'),float('inf')):
        with pytest.raises(ValueError):error_radius(model,days)


def test_growth_is_learned_not_fixed_to_point_two_one_or_square_root():
    for power in (.1,.35,.5,.8):
        assert matched_error_growth(errors(power))['exponent']==pytest.approx(power)
    assert matched_error_growth(errors(-.2)) is None
    assert matched_error_growth(errors()[:8]) is None


def test_calibration_excludes_health_affected_errors_but_does_not_change_control_growth():
    control=errors();candidate=deepcopy(control)
    for r in candidate[:4]:r['regime']='health_affected';r['log_error']=.9
    result=calibrate({28:control,42:candidate})
    assert result['exponent']==pytest.approx(.21)
    assert len(result['errors'])==len(control)-4
    assert all(r['regime']=='consistent' for r in result['errors'])
    assert len(candidate)==len(control)


def test_range_needs_actual_independent_outcome_support_not_repeated_origins():
    rows=errors();candidate=[{**r,'outcome_ids':[1]} for r in rows]
    assert calibrate({28:rows,42:candidate}) is None
    assert calibrate({28:rows,42:rows[:8]}) is None
    assert weighted_quantile([1.,2.],[1.,1.],[.5])[0]==1.5
    repeat={**rows[0],'target_id':'repeat','outcome_ids':[999]}
    assert sum(weights([rows[0],repeat,rows[2]]))==pytest.approx(1.)


def fixture():
    rows=[]
    for i in range(35):
        origin=AT+timedelta(days=i*7)
        rows.append(dict(origin=origin.isoformat(),origin_id=i,target_id=str(i),horizon=14,
            elapsed=.5,log_miles=.1,outcome_ids=[i+100],outcome_end=(origin+timedelta(days=14)).isoformat(),
            response_variance=.001,start_pace=10.,actual=9.9,change=-.01,
            consistency=dict(regime='consistent')))
    policies={p:deepcopy(rows) for p in POLICIES}
    return {28:deepcopy(policies),42:deepcopy(policies)},rows


def test_hindcasts_use_actual_future_runs_and_never_future_fit_labels():
    full,raw=fixture();groups=deepcopy(full)
    first=error_history(full,groups,raw)
    changed=deepcopy(groups);cutoff=raw[22]['origin']
    for policies in changed.values():
        for rows in policies.values():
            for r in rows:
                if r['outcome_end']>=cutoff:r['change']+=1.
    second=error_history(full,changed,raw)
    key=lambda r:(r['origin'],r['target_id'])
    for days in (28,42):
        assert {key(r):r['predicted'] for r in first[days] if r['origin']<=cutoff}=={
            key(r):r['predicted'] for r in second[days] if r['origin']<=cutoff}
        assert all(r['actual']==9.9 for r in first[days])
    new_actual=[{**r,'actual':9.7} for r in raw]
    updated=error_history(full,groups,new_actual)
    assert [r['predicted'] for r in updated[42]]==[r['predicted'] for r in first[42]]
    assert [r['log_error'] for r in updated[42]]!=[r['log_error'] for r in first[42]]


def test_all_policy_fits_require_mature_completed_responses():
    full,rows=fixture();policies=full[42]
    at=rows[20]['origin'];fits=response_fits(policies,at)
    assert fits is not None
    assert all(f['latest_outcome']<at for forms in fits.values() for f in forms.values())
    policies['gap7_fraction0.667']=rows[:3]
    assert response_fits(policies,at) is None
