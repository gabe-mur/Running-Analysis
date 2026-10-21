from copy import deepcopy
from datetime import datetime, timedelta, timezone
import math
from unittest.mock import patch

import numpy as np
import pytest

from run_analysis import analytics as a
from run_analysis import aerobic_progress as ap
from run_analysis.uncertainty import overlapping_variance


@pytest.mark.parametrize('measurement,observed',[(.01,.04),(.04,.01),(0.,0.),(.03,.03)])
def test_overlapping_terms_are_floors_not_added_causes(measurement,observed):
    assert overlapping_variance(measurement,observed)==max(measurement,observed)


@pytest.mark.parametrize('value',[-1.,float('inf'),float('nan')])
def test_invalid_variance_cannot_manufacture_precision(value):
    with pytest.raises(ValueError):overlapping_variance(value,.1)


def sample():
    origin=datetime(2026,1,1,tzinfo=timezone.utc)
    return [dict(activity_id=i,start_time_utc=(origin+timedelta(days=i*3)).isoformat(),
        standardized_pace=11-i*.012+.10*math.sin(i),uncertainty_95=.65,
        distance_miles=3.+i%4,trend_weight=1.) for i in range(45)]


def analytic_fits(rows):
    end=datetime.fromisoformat(rows[-1]['start_time_utc'])
    current=a._window_estimate(rows,end,28)
    prior=a._window_estimate(rows,end-timedelta(days=28),28)
    comparison=a._distance_adjusted_comparison(rows,current_end=end,
        prior_end=end-timedelta(days=28),days=28,current=current,prior=prior,label='test',robust=True)
    slope=a._window_trend(rows,end,28)
    return current,comparison,slope


def test_all_analytic_fits_preserve_estimates_and_only_reduce_overlapping_uncertainty():
    rows=sample();before=deepcopy(rows)
    with patch.object(a,'overlapping_variance',side_effect=lambda m,o:m+o):old=analytic_fits(rows)
    new=analytic_fits(rows)
    for earlier,later in zip(old,new):
        for key in ('pace_min_mile','pace_change_seconds_per_mile','slope_seconds_per_mile_per_day',
                    'distance_effect_seconds_per_mile_per_added_mile','run_count','effective_run_count'):
            if key in earlier:assert later[key]==earlier[key]
        key='uncertainty_95_min_mile' if 'pace_min_mile' in earlier else 'uncertainty_95_seconds_per_mile'
        assert later[key]<=earlier[key]
    assert rows==before


def test_shared_aerobic_variance_is_not_absorbed_into_the_envelope():
    source=sample()
    rows=[dict(activity_id=r['activity_id'],start_time=r['start_time_utc'],pace_min_mile=r['standardized_pace'],
        uncertainty_95_min_mile=r['uncertainty_95'],trend_weight=1.,included_in_trend=True,
        shared_log_speed_gradient=[.3]) for r in source]
    end=datetime.fromisoformat(rows[-1]['start_time']);cov=np.array([[.001]])
    with patch.object(ap,'overlapping_variance',side_effect=lambda m,o:m+o):old=ap._period(rows,end,cov)
    new=ap._period(rows,end,cov)
    assert new['pace_min_mile']==old['pace_min_mile']
    assert new['random_variance']==max(new['measurement_variance'],new['observed_variance'])
    assert new['shared_calibration_variance']==old['shared_calibration_variance']>0
    assert new['uncertainty_95_min_mile']==pytest.approx(1.96*math.sqrt(new['random_variance']+new['shared_calibration_variance']))
