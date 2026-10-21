from copy import deepcopy
from datetime import datetime,timedelta,timezone
import math

import numpy as np
import pytest

from scripts.audit_growing_pace_error import (
    audit,calibrate_range,chronological,fit_shape,matched_moments,radius,scale,
)

AT=datetime(2024,1,1,tzinfo=timezone.utc)


def examples(*,baseline=.0004,growth=.0009,power=1,n=45):
    rows=[]
    for i in range(n):
        origin=AT+timedelta(days=7*i)
        for j,h in enumerate((14,28,56,84,112)):
            error=math.sqrt(baseline+growth*(h/28)**power)
            rows.append(dict(origin=origin.isoformat(),origin_id=i,target_id=f'{i}_{h}',
                outcome_end=(origin+timedelta(days=h)).isoformat(),elapsed=h/28,horizon=h,
                outcome_ids=[i*10+j],log_error=error,actual=10.,predicted=10*math.exp(error),
                regime='consistent'))
    return rows


@pytest.mark.parametrize('power,method',[(1,'baseline_linear_variance'),(2,'baseline_quadratic_variance')])
def test_recovers_nonzero_baseline_and_growth_instead_of_inventing_a_starting_floor(power,method):
    rows=examples(power=power)
    short=[r for r in rows if r['horizon'] in (14,28)]
    shape=fit_shape(short,method)
    assert shape['baseline']==pytest.approx(.0004)
    assert shape['growth']==pytest.approx(.0009)
    model=calibrate_range(shape,short)
    assert radius(model,0)==pytest.approx(.02)
    for days in (7,28,56,211):
        assert radius(model,days)==pytest.approx(math.sqrt(.0004+.0009*(days/28)**power))
    # Finite initial slope rather than the t**0.21 cusp at zero.
    assert math.isfinite((radius(model,.00001)-radius(model,0))/.00001)


def test_two_horizons_share_exactly_the_same_origin_weights_and_support():
    rows=examples(n=12)
    rows=[r for r in rows if r['horizon']!=28 or r['origin_id']<9]
    matched=matched_moments(rows)
    assert matched['origins']==9
    assert [r['origin'] for r in matched['horizons'][0]]==[r['origin'] for r in matched['horizons'][1]]
    assert np.sum(matched['origin_weights'])==pytest.approx(1.)
    assert matched_moments(rows,minimum_origins=10) is None


def test_negative_growth_or_intercept_is_not_patched_to_produce_a_pretty_fan():
    rows=examples()
    for r in rows:
        if r['horizon']==14:r['log_error']=.1
        if r['horizon']==28:r['log_error']=.01
    assert fit_shape(rows,'baseline_linear_variance') is None
    assert fit_shape(examples(baseline=0,growth=.001,power=2),'baseline_linear_variance') is None
    assert calibrate_range(None,rows) is None
    assert calibrate_range(fit_shape(examples(),'baseline_linear_variance'),rows[:6]) is None
    with pytest.raises(ValueError):scale(dict(baseline=1.,growth=1.,power=1.),-1)


def test_future_labels_and_the_target_runs_cannot_calibrate_historical_bands():
    rows=examples()
    first=chronological(rows,rows)
    assert first['predictions']
    for r in first['predictions']:
        assert r['latest_calibration']<r['origin']
        if r['latest_shape_outcome']:assert r['latest_shape_outcome']<r['origin']
        assert set(r['outcome_ids']).isdisjoint(r['calibration_outcome_ids'])
    cutoff=(AT+timedelta(days=210)).isoformat();changed=deepcopy(rows)
    for r in changed:
        if r['outcome_end']>=cutoff:r['log_error']*=5
    second=chronological(changed,changed)
    key=lambda r:(r['origin'],r['target_id'],r['method'])
    before=lambda report:{key(r):r['radius'] for r in report['predictions'] if r['origin']<=cutoff}
    assert before(first)==before(second)


def test_pairwise_scoring_uses_identical_cases_and_cannot_win_by_discarding_unsupported_fits():
    report=chronological(examples(),examples())
    for method,values in report['pairwise'].items():
        a={(r['origin'],r['target_id']) for r in values['predictions'] if r['method']==method}
        b={(r['origin'],r['target_id']) for r in values['predictions'] if r['method']=='existing_power'}
        assert a==b and a
    # No fabricated healthy-history calibration when the only available
    # completed residuals are explicitly health affected.
    affected=[{**r,'regime':'health_affected'} for r in examples()]
    assert not chronological(affected,affected,consistent=True)['predictions']


def test_audit_keeps_the_pace_projection_and_input_evidence_unchanged():
    rows=examples(n=35)
    source=dict(as_of='2026-10-20',target_days=211,final_sensitivity={'42':{'may_joint':9.1}},
        predictions=[{**r,'bandwidth':bw,'mode':'joint'} for bw in (28,42) for r in rows])
    original=deepcopy(source);result=audit(source)
    assert source==original
    assert not any(result[k] for k in ('live_forecast_changed','planner_changed','database_changed'))
    for scope in result['results'].values():
        for method,model in scope['current'].items():
            if model['available']:assert model['may_center']==9.1
