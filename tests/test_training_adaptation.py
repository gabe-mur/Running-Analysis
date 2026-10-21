from datetime import datetime,timedelta,timezone
import math

import numpy as np
import pytest

from run_analysis.training_adaptation import training_stock,projected_training,fit_change,add_components

AT=datetime(2026,10,20,tzinfo=timezone.utc)


def test_exposure_is_causal_delayed_and_steady_state_has_weekly_units():
    run=dict(time=AT,available_at=AT+timedelta(hours=1),miles=10)
    values=training_stock([run],[AT,run['available_at'],AT+timedelta(days=7)])
    assert values[0]==values[1]==0 and values[2]>0
    history=[dict(time=AT-timedelta(days=d+.5),miles=3) for d in range(1000)]
    assert training_stock(history,[AT])[0]==pytest.approx(21,rel=.001)
    assert training_stock(history,[AT],memory_days=7)[0]==pytest.approx(21,rel=.001)


def test_projection_preserves_actual_initial_state_and_never_substitutes_target():
    history=[dict(time=AT-timedelta(days=d+.5),miles=2) for d in range(500)]
    values=projected_training(history,AT,56,volume=14,growth=.04,target_volume=42)
    assert values[0]==pytest.approx(training_stock(history,[AT])[0])
    assert values[1]<14.01 and values[-1]>values[0]
    with pytest.raises(ValueError):
        projected_training([dict(time=AT+timedelta(days=1),miles=5)],AT,28,volume=14,growth=.04,target_volume=42)


def test_response_fit_has_no_absolute_pace_or_intercept_and_no_nonfinite_inputs():
    examples=[dict(target_id=str(i),outcome_end=AT.isoformat(),elapsed=i/10,trend=0,
        training_change=i/100,temporary=0,change=-i/1000) for i in range(10)]
    fit=fit_change(examples)
    assert set(fit['coefficients'])=={'elapsed','trend','training_change','temporary'}
    assert fit['coefficients']['training_change']<0
    with pytest.raises(ValueError):fit_change(examples,penalty=float('nan'))
    examples[0]['change']=float('nan')
    with pytest.raises(ValueError):fit_change(examples)


def test_identifiable_training_and_transient_components_are_estimated_separately():
    rng=np.random.default_rng(12)
    x=rng.normal(size=(150,4));coefficients=np.array([.001,.2,-.06,-.7])
    examples=[dict(zip(('elapsed','trend','training_change','temporary'),row),
        target_id=str(i),outcome_end=AT.isoformat(),change=float(row@coefficients)) for i,row in enumerate(x)]
    fit=fit_change(examples,penalty=1e-7)
    assert list(fit['coefficients'].values())==pytest.approx(coefficients,abs=1e-8)
    # The synthetic result verifies accounting, not identifiability in real data.


def test_each_outcome_uses_own_training_and_later_runs_cannot_change_it():
    rows=[dict(activity_id=i,start_time=(AT+timedelta(days=i*3)).isoformat(),pace_min_mile=10*math.exp(-i*.001)) for i in range(20)]
    origin=datetime.fromisoformat(rows[15]['start_time'])
    dates=[origin+timedelta(days=21),origin+timedelta(days=35)]
    training=[dict(time=AT+timedelta(days=d),miles=3) for d in range(100)]
    example=dict(origin_id=15,origin=origin.isoformat(),start_level=math.log(9.9),start_trend=-.01,
        elapsed=1.,training_by_outcome=[dict(at=d.isoformat(),weight=.5) for d in dates])
    first=add_components([example],rows,np.full(20,.001),training)[0]
    stocks=training_stock(training,[origin,*dates])
    assert first['training_change']==pytest.approx(.5*sum(np.log(stocks[1:]))-math.log(stocks[0]))
    later=[*training,dict(time=dates[-1]+timedelta(seconds=1),miles=100)]
    assert add_components([example],rows,np.full(20,.001),later)==[first]
    # A run at the outcome start has zero exposure and cannot predict itself.
    same=[*training,dict(time=dates[-1],miles=100)]
    assert add_components([example],rows,np.full(20,.001),same)==[first]
