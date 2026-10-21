from datetime import datetime, timedelta, timezone
import math

import numpy as np
import pytest

from run_analysis.weather_impact import CONTROLS, analyze_weather_impact


def synthetic(*, coefficient=.006, seasonal_only=False, noise=0.0):
    rng=np.random.default_rng(41013)
    # Align the deliberately constant-per-calendar-block weather case with the
    # declared primary blocks; otherwise it has genuine within-block steps.
    start=datetime(1970,1,1,tzinfo=timezone.utc)+timedelta(days=28*704)
    rows=[]
    for block in range(18):
        season=math.sin(block/3)
        for index in range(10):
            day=block*28+index*2
            heat=12+5*season+(0 if seasonal_only else rng.normal(0,4))
            hr=148+rng.normal(0,2)
            position=20+rng.normal(0,3)
            grade=rng.normal(0,.3)
            headwind=rng.normal(0,2)
            duration=rng.normal(3.5,.2)
            volume=rng.normal(3,.25)
            recovery=rng.normal(1,.3)
            # Independent moving fitness level and local trend, not a constant athlete.
            logpace=(math.log(10)-.008*block-.05*season-.001*index
                     -.008*(hr-148)+.001*(position-20)+.02*grade+.002*headwind
                     +.015*(duration-3.5)+.01*(volume-3)-.005*(recovery-1)
                     +coefficient*heat+rng.normal(0,noise))
            rows.append(dict(activity_id=len(rows)+1,start_time_utc=(start+timedelta(days=day)).isoformat(),
                observed_pace_min_mile=math.exp(logpace),wbgt_c=heat,sampled_hr_bpm=hr,
                sampled_minutes=position,grade_percent=grade,headwind_mph=headwind,
                log_duration=duration,log_previous_7d_miles=volume,log_days_since_hard=recovery))
    return rows


def test_recovers_known_weather_effect_without_using_race_prior_or_fitting_an_athlete_constant():
    result=analyze_weather_impact(synthetic())
    assert result['primary']['log_pace_change_per_wbgt_c']==pytest.approx(.006,abs=.0003)
    assert result['primary']['five_c_warmer_association']['pace_change_percent']==pytest.approx(math.expm1(.03)*100,abs=.2)
    assert not result['adjusted_trend_promoted']
    assert result['status']=='supported_observational_association'
    cv=result['chronological_validation']
    assert cv['status']=='evaluated_relative_contrasts'
    assert cv['with_weather_mean_absolute_contrast_error_seconds']<cv['context_only_mean_absolute_contrast_error_seconds']
    assert cv['held_out_blocks']<cv['pairs_not_independent']


def test_pure_between_season_weather_difference_is_not_identifiable():
    result=analyze_weather_impact(synthetic(seasonal_only=True))
    assert result['status']=='not_estimable'
    assert result['primary']['status']=='weather_not_identifiable'
    assert result['primary']['remaining_weather_sd_c']<1e-8
    assert 'log_pace_change_per_wbgt_c' not in result['primary']


def test_null_weather_with_seasonal_fitness_does_not_inherit_the_seasonal_change():
    result=analyze_weather_impact(synthetic(coefficient=0,noise=.002))
    assert abs(result['primary']['log_pace_change_per_wbgt_c'])<.0005
    assert result['primary']['association_includes_zero']
    assert result['status']=='uncertain_association'


def test_insufficient_or_missing_context_is_explicit_not_imputed_as_zero():
    result=analyze_weather_impact([])
    assert result['status']=='not_estimable'
    rows=synthetic()
    for row in rows[:7]:row['headwind_mph']=None
    result=analyze_weather_impact(rows)
    assert result['complete_context_runs']==len(rows)-7
    assert result['missing_context_counts_not_exclusive']=={'headwind_mph':7}


def test_run_order_and_seed_are_deterministic_and_observations_not_mutated():
    rows=synthetic(noise=.005)
    before=[dict(row) for row in rows]
    result=analyze_weather_impact(rows)
    assert rows==before
    assert result==analyze_weather_impact(rows)
    assert result==analyze_weather_impact(list(reversed(rows)))
    assert result['controls']==list(CONTROLS)


def test_retains_unexpected_negative_association_instead_of_forcing_a_heat_penalty():
    result=analyze_weather_impact(synthetic(coefficient=-.006))
    assert result['primary']['log_pace_change_per_wbgt_c']<0
    assert result['primary']['five_c_warmer_association']['pace_change_percent']<0


def test_reused_activity_cannot_multiply_weather_evidence():
    source=synthetic()
    with pytest.raises(ValueError,match='only once'):
        analyze_weather_impact(source+[source[0]])


def test_bad_dates_are_reported_and_do_not_become_calendar_observations():
    source=synthetic();source[0]['start_time_utc']='not-a-date'
    result=analyze_weather_impact(source)
    assert result['missing_context_counts_not_exclusive']=={'invalid_run_time':1}
    assert result['complete_context_runs']==len(source)-1


def test_one_held_out_block_cannot_produce_a_false_zero_width_validation_interval(monkeypatch):
    from run_analysis import weather_impact
    start=datetime(1970,1,1,tzinfo=timezone.utc)+timedelta(days=28*704)
    rows=[]
    for block in range(3):
        for index in range(2):
            rows.append(dict(activity_id=len(rows)+1,start_time_utc=(start+timedelta(days=28*block+index*2)).isoformat(),
                wbgt_c=10+index*10 if block==2 else 10,observed_pace_min_mile=10+index if block==2 else 10,
                **{key:0 for key in CONTROLS}))
    monkeypatch.setattr(weather_impact,'_fit',lambda *a,**kw:{'status':'estimated_association',
        '_context_only':np.zeros(len(CONTROLS)),'_context_with_weather':np.zeros(len(CONTROLS)),
        'log_pace_change_per_wbgt_c':.01})
    result=weather_impact._chronological_contrasts(rows)
    assert result['held_out_blocks']==1
    assert result['status']=='one_held_out_block_only'
    assert result['block_bootstrap_interval_95'] is None
