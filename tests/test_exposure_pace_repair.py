from dataclasses import replace
from math import exp

import pytest

from run_analysis.adaptive.lag_demand_reference import fit_exposure_pace,predict_exposure_pace
from test_lag_demand_reference import sample
from test_quality_dose_policy import AT


def fit(rows,channel='demand'):
    return fit_exposure_pace(rows,as_of=AT,accounting_key='units',channel=channel)


def observations():
    return [sample(str(i),120,u,exp(.3+.8*u)) for i,u in enumerate((.5,.6,.7,.8))]


def test_identical_measurement_lengths_do_not_discard_identifiable_exposure_relation():
    model=fit(observations())
    assert model['status']=='fitted_diagnostic'
    assert model['coefficients']==pytest.approx((.3,.8))
    assert not model['duration_effect_identified']
    assert not model['measurement_duration_is_capacity']
    assert predict_exposure_pace(model,issued_at=AT,intensity=.65)['speed_mps']==pytest.approx(exp(.3+.8*.65))


def test_splitting_identical_observation_support_does_not_increase_source_ownership():
    rows=observations(); rows[0]=replace(rows[0],speed_mps=rows[0].speed_mps*1.1)
    original=fit(rows)
    split=[replace(rows[0],key='split'+str(i),seconds=12) for i in range(10)]
    assert fit([*split,*rows[1:]])['coefficients']==pytest.approx(original['coefficients'])


def test_renaming_families_has_no_effect_and_negative_evidence_is_not_repaired():
    rows=observations()
    assert fit([replace(r,kind='unseen workout') for r in rows])==fit(rows)
    inverted=[replace(r,speed_mps=exp(1.-.8*r.demand)) for r in rows]
    assert fit(inverted)['status']=='nonpositive_intensity_relation'


def test_constant_intensity_and_future_fit_remain_unresolved():
    assert fit([observations()[0]])['status']=='insufficient_identifiable_evidence'
    with pytest.raises(ValueError,match='Causal'):
        predict_exposure_pace(fit(observations()),issued_at=AT.replace(year=2020),intensity=.7)
