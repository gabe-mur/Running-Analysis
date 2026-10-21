from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from scripts.audit_relative_weather import prior_anomaly, interpolate, weather_features, paired_range, recent_run_temperature


def archive(start,hours=24*20):
    times=np.array([(start+timedelta(hours=i)).timestamp() for i in range(hours)])
    return dict(lat=0.,lon=0.,times=times,temperature=np.full(hours,80.),source='fixture')


def test_same_time_prior_calendar_days_not_future_or_running_dates():
    data=archive(datetime(2026,7,1,tzinfo=timezone.utc))
    when=datetime(2026,7,10,19,tzinfo=timezone.utc)
    data['temperature'][data['times']==when.timestamp()]=70
    data['temperature'][data['times']>when.timestamp()]=200
    result=prior_anomaly(data,when,7,'America/New_York')
    assert result['prior_mean_f']==80
    assert result['anomaly_c']==pytest.approx(-10*5/9)
    assert datetime.fromisoformat(result['last_prior'])<when


def test_no_boundary_clamping_incomplete_week_or_missing_temperature_imputation():
    start=datetime(2026,7,1,tzinfo=timezone.utc);data=archive(start)
    assert interpolate(data,start-timedelta(seconds=1)) is None
    assert prior_anomaly(data,start+timedelta(days=3),7,'UTC') is None
    data['temperature'][24*8]=np.nan
    assert prior_anomaly(data,start+timedelta(days=10),7,'UTC') is None


def test_same_local_hour_survives_daylight_saving_boundary():
    start=datetime(2026,10,20,tzinfo=timezone.utc);data=archive(start)
    zone=ZoneInfo('America/New_York')
    when=datetime(2026,11,3,18,tzinfo=zone).astimezone(timezone.utc)
    for i in range(1,8):
        expected=(when.astimezone(zone)-timedelta(days=i)).astimezone(timezone.utc).timestamp()
        data['temperature'][data['times']==expected]=70.
    result=prior_anomaly(data,when,7,'America/New_York')
    assert result['prior_mean_f']==70.


def test_choose_complete_nearest_archive_not_best_weather_or_partial_days():
    start=datetime(2026,7,1,tzinfo=timezone.utc)
    incomplete=archive(start,hours=24);complete=archive(start)
    incomplete.update(source='incomplete');complete.update(lat=.01,source='complete')
    row=dict(activity_id=1,start_time_utc='2026-07-10T18:00:00+00:00',sampled_minutes=20.,temperature_f=80.,heat_c=25.)
    result,missing=weather_features([row],[incomplete,complete],{1:(0.,0.)},{1:'UTC'})
    assert result[0]['relative_weather']['7']['source']=='complete'
    assert missing==[]
    assert row.get('relative_weather') is None
    absent,missing=weather_features([row],[complete],{1:(2.,2.)},{1:'UTC'})
    assert absent[0]['relative_weather']=={}
    assert missing[0]['reason']=='no_complete_preceding_week_nearby'


def test_hourly_gaps_are_not_silently_interpolated_and_block_denominators_pair():
    data=archive(datetime(2026,7,1,tzinfo=timezone.utc));data['times']=data['times'][::2];data['temperature']=data['temperature'][::2]
    assert interpolate(data,datetime(2026,7,1,1,tzinfo=timezone.utc)) is None
    cases=[dict(date=f'2026-0{m}-01T00:00:00+00:00',candidate_absolute_error=9.,baseline_absolute_error=10.) for m in (1,2,3)]
    result=paired_range(cases,draws=20)
    assert result['mean_error_difference_seconds']==-1
    assert result['descriptive_95_block_range']==[-1.,-1.]


def test_evening_run_uses_actual_temperature_even_after_hotter_daytime_runs():
    target=dict(start_time_utc='2026-07-07T23:00:00+00:00',temperature_f=70.)
    history=[dict(activity_id=1,start_time_utc='2026-07-05T15:00:00+00:00',temperature_f=80.,location=(0.,0.)),
        dict(activity_id=2,start_time_utc='2026-07-08T15:00:00+00:00',temperature_f=50.,location=(0.,0.)),
        dict(activity_id=3,start_time_utc='2026-07-06T23:00:00+00:00',temperature_f=100.,location=(2.,2.))]
    value=recent_run_temperature(target,history,(0.,0.),7)
    assert value['current_f']==70.
    assert value['prior_mean_f']==80.
    assert value['anomaly_c']==pytest.approx(-10*5/9)
    assert value['prior_ids']==[1]
