from datetime import datetime, timedelta, timezone
from copy import deepcopy

from run_analysis.progress_comparisons import quality_comparisons, endurance_comparisons

END = datetime(2026,10,5,tzinfo=timezone.utc)
CONTEXT = dict(temperature_f=65, dewpoint_f=50, mean_abs_grade_percent=1)


def repeat(key, days, pace=7, distance=.25, count=6, recovery=90):
    return dict(activity_id=key,start_time=(END-timedelta(days=days)).isoformat(),context=dict(CONTEXT),
        interval=dict(available=True,repetitions=[dict(kind='work',distance_miles=distance,
            duration_seconds=pace*60*distance) for _ in range(count)]+[dict(kind='recovery',
                duration_seconds=recovery,pace_min_mile=12) for _ in range(count-1)]))


def drift(key, days, value=3, duration=95, hr=148):
    return dict(activity_id=key,start_time=(END-timedelta(days=days)).isoformat(),context=dict(CONTEXT),
        available=True,decoupling_percent=value,moving_minutes=duration,first_hr=hr)


def test_repeats_not_pooled_across_distances_or_recovery_and_missing_conditions_not_matched():
    rows=[repeat(1,70,8),repeat(2,60,7.8),repeat(3,10,7),repeat(4,2,7.2),
        repeat(5,3,6,distance=.15),repeat(6,4,6,recovery=180)]
    result=quality_comparisons(rows,END)
    assert len(result['repeats'])==3
    c=result['repeats'][0]['comparison']
    assert c['status']=='matched_observations'
    assert c['recent']['count']==2 and c['earlier']['count']==2
    assert c['delta'] < 0
    rows[0]['context']={}; rows[1]['context']={}
    assert quality_comparisons(rows,END)['repeats'][0]['comparison']['status']=='different_conditions_or_sparse'


def test_single_recording_and_cooler_weather_are_not_called_matched_progress():
    rows=[repeat(1,70,8),repeat(2,10,7)]
    rows[1]['context']['temperature_f']=45
    c=quality_comparisons(rows,END)['repeats'][0]['comparison']
    assert c['recent']['count']==1
    assert c['screened']['earlier']['count']==0
    assert c['status']=='different_conditions_or_sparse'


def test_rep_pace_reuses_run_analysis_measurement_instead_of_recalculating():
    earlier, recent=repeat(1,70,8), repeat(2,10,7)
    earlier['interval']['mean_work_pace_min_mile']=8.1
    recent['interval']['mean_work_pace_min_mile']=7.1
    c=quality_comparisons([earlier,recent],END)['repeats'][0]['comparison']
    assert c['earlier']['median']==8.1
    assert c['recent']['median']==7.1


def test_sustained_structure_duration_hr_and_future_records_not_pooled():
    def sustained(key, days, structure='progression_finish',hr=160):
        return dict(activity_id=key,start_time=(END-timedelta(days=days)).isoformat(),context=CONTEXT,
            continuous=dict(pace_min_mile=9,duration_seconds=1500,average_hr_bpm=hr,structure=structure))
    rows=[sustained(1,60),sustained(2,55),sustained(3,10),sustained(4,5),
        sustained(5,60,'interior_work'),sustained(6,60,hr=170)]
    result=quality_comparisons(rows,END)
    assert result['sustained']['comparison']['earlier']['count']==2
    assert result['sustained']['comparison']['status']=='matched_observations'
    assert result==quality_comparisons(rows+[sustained(7,-1,'future')],END)


def test_endurance_matches_starting_intensity_and_does_not_have_90_minute_cliff():
    rows=[drift(1,70,4,duration=89),drift(2,60,5,duration=92),drift(3,10,2),drift(4,5,3),
        drift(5,20,15,hr=170)]
    p=endurance_comparisons(rows,END)[-1]
    c=p['comparison']
    assert c['earlier']['count']==2 and c['recent']['count']==2
    assert c['recent']['median']==2.5 and c['earlier']['median']==4.5
    assert c['status']=='matched_observations'
    assert 5 not in c['recent']['activity_ids']


def test_future_and_health_rows_do_not_change_comparisons_or_mutate_inputs():
    rows=[drift(1,70),drift(2,10)]
    before=deepcopy(rows)
    result=endurance_comparisons(rows,END)
    assert result==endurance_comparisons(rows+[drift(3,-1,100)],END)
    assert rows==before
    bad=repeat(3,2); bad['health_tag']='illness'
    assert quality_comparisons([repeat(1,70),repeat(2,10),bad],END)==quality_comparisons([repeat(1,70),repeat(2,10)],END)


def test_repeat_pace_uses_total_work_time_over_distance_not_average_lap_pace():
    q=repeat(1,10,count=2)
    q['interval']['repetitions'][:2]=[dict(kind='work',distance_miles=.2,duration_seconds=96),
        dict(kind='work',distance_miles=.3,duration_seconds=108)]
    c=quality_comparisons([q],END)['repeats'][0]['comparison']
    assert c['recent']['median']==6.8


def test_overlapping_endurance_profiles_are_not_independent_evidence():
    profiles=endurance_comparisons([drift(1,20,duration=83),drift(2,10,duration=100)],END)
    assert len(profiles)==1
    assert profiles[0]['reference_minutes']==100
