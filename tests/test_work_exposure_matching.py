from dataclasses import replace
from datetime import timedelta

import pytest

from run_analysis.adaptive.direct_quality_load import phase_observations, forecast_direct
from run_analysis.adaptive.work_exposure_matching import (
    PhaseTrace, WorkExposure, band_fraction, forecast_work_exposure, forecast_work_for_ranking,
    requested_work_bands)
from run_analysis.signal_structure import SignalSpan
from test_direct_quality_load import AT, history


LAYOUT = (('warmup',0.,60.),('work',60.,120.),('recovery',120.,180.),
          ('work',180.,240.),('cooldown',240.,300.))


def record(key, work, copies=1):
    layout = (('work',0.,60.),) if copies != 1 else LAYOUT
    if copies != 1:
        layout = tuple(('work',60.*i,60.*(i+1)) for i in range(copies))
    spans = tuple(SignalSpan(a,b,work if k=='work' else .6,work if k=='work' else .6)
                  for k,a,b in layout)
    observations = phase_observations(layout,spans)
    if copies != 1:
        observations = tuple(replace(p,previous_seconds=None) for p in observations)
    source = history(key,phases=observations)
    profiles = tuple(PhaseTrace(a,b,(s,)) for (_,a,b),s in zip(layout,spans))
    return source, profiles


def query(layout, band):
    return tuple(WorkExposure(band,band,'test prescribed band') if k=='work' else None for k,_,_ in layout)


def predict(records, profiles, band=(.5,.6), layout=LAYOUT):
    return forecast_work_exposure(layout,records,profiles,query(layout,band),as_of=AT,accounting_key='units')


def test_only_work_changes_and_source_history_is_not_mutated():
    low,lt = record('low',.55);high,ht = record('high',.85)
    records = [low,high];profiles = {'low':lt,'high':ht}
    baseline = forecast_direct(LAYOUT,records,as_of=AT,accounting_key='units')
    softer = predict(records,profiles);harder = predict(records,profiles,(.8,.9))
    assert harder['predicted_load'] > baseline['predicted_load'] > softer['predicted_load']
    for a,b,c in zip(baseline['phases'],softer['phases'],harder['phases']):
        if a['kind']!='work':
            assert a==b==c
    assert baseline == forecast_direct(LAYOUT,records,as_of=AT,accounting_key='units')


def test_partial_startup_is_matching_affinity_not_truncated_cost():
    layout = (('work',0.,60.),)
    spans = (SignalSpan(0.,60.,.4,.8),)
    source = history(phases=phase_observations(layout,spans))
    f = predict([source],{source.key:(PhaseTrace(0.,60.,spans),)},(.6,.9),layout)
    assert f['predicted_load'] == pytest.approx(source.phases[0].load)
    assert f['phases'][0]['work_exposure']['sources'][0]['match_fractions'] == pytest.approx([.5])


def test_ramp_novel_duration_no_exact_format_gate_and_same_whole_rate():
    spans = (SignalSpan(0.,60.,.4,.8),)
    source = history(phases=phase_observations((('work',0.,60.),),spans))
    layout = (('work',0.,103.),)
    f = forecast_work_exposure(layout,[source],{source.key:(PhaseTrace(0.,60.,spans),)},
        (WorkExposure((.39,.41),(.79,.81),'gradual'),),as_of=AT,accounting_key='units')
    assert f['status']=='predicted'
    assert f['phases'][0]['work_exposure']['supported']
    assert f['predicted_load']==pytest.approx(source.phases[0].load*103/60)


def test_unknown_cue_unsupported_band_missing_trace_keep_baseline_explicitly():
    source,traces = record('past',.55)
    baseline = forecast_direct(LAYOUT,[source],as_of=AT,accounting_key='units')
    missing = predict([source],{})
    unsupported = predict([source],{'past':traces},(.9,1.))
    unknown = forecast_work_exposure(LAYOUT,[source],{'past':traces},[None]*len(LAYOUT),
        as_of=AT,accounting_key='units')
    for f,reason in ((missing,'missing_observed_source_trace'),
                     (unsupported,'no_observed_band_overlap'),(unknown,'unknown_requested_work_exposure')):
        assert f['predicted_load']==baseline['predicted_load']
        assert all(p['work_exposure']['fallback_reason']==reason for p in f['phases'] if p['kind']=='work')
    assert predict([],{},layout=LAYOUT)['predicted_load'] is None


def test_future_old_unavailable_and_other_accounting_do_not_supply_matching_trace():
    source,traces = record('past',.55)
    rejected = [replace(source,key='future',start=AT+timedelta(days=1),end=AT+timedelta(days=1,minutes=5),
                        available_at=AT+timedelta(days=1,minutes=5)),
                replace(source,key='old',start=AT-timedelta(days=31),end=AT-timedelta(days=31,minutes=-5)),
                replace(source,key='late',available_at=AT+timedelta(seconds=1)),
                replace(source,key='other',accounting_key='other')]
    assert predict([source],{'past':traces})==predict([source,*rejected],{'past':traces})


def test_one_source_vote_not_one_vote_per_repetition():
    low,lt = record('low',.55,copies=2);high,ht = record('high',.6,copies=2)
    many,mt = record('low',.55,copies=20)
    layout = (('work',0.,60.),)
    a=predict([low,high],{'low':lt,'high':ht},(.5,.7),layout)
    b=predict([many,high],{'low':mt,'high':ht},(.5,.7),layout)
    assert a['predicted_load']==pytest.approx(b['predicted_load'])


def test_unknown_gaps_are_not_counted_as_exposure_and_masks_must_agree():
    layout = (('work',0.,60.),);spans = (SignalSpan(30.,60.,.55,.55),)
    source = history(phases=phase_observations(layout,spans))
    f = predict([source],{source.key:(PhaseTrace(0.,60.,spans),)},(.5,.6),layout)
    assert f['predicted_load']==pytest.approx(source.phases[0].load*2)
    bad = {source.key:(PhaseTrace(0.,60.,(SignalSpan(0.,60.,.55,.55),)),)}
    with pytest.raises(ValueError,match='known-time mask'):
        predict([source],bad,(.5,.6),layout)


@pytest.mark.parametrize('start,finish', [((-.1,.5),(.5,.6)),((.7,.5),(.5,.6)),
    ((.5,float('nan')),(.5,.6)),((.5,.6),(.8,1.1))])
def test_invalid_requested_exposure_is_not_silently_reinterpreted(start,finish):
    with pytest.raises(ValueError):
        WorkExposure(start,finish,'test')


def test_nonwork_query_and_overlapping_trace_are_rejected():
    with pytest.raises(ValueError,match='Work-only'):
        forecast_work_exposure((('warmup',0.,60.),),[],{},
            (WorkExposure((.5,.6),(.5,.6),'invalid'),),as_of=AT,accounting_key='units')
    with pytest.raises(ValueError):
        PhaseTrace(0.,60.,(SignalSpan(0.,40.,.5,.5),SignalSpan(30.,60.,.5,.5)))
    with pytest.raises(ValueError):
        band_fraction((),0.,0.,(.5,.6),(.5,.6))


def test_compiled_menu_maps_each_phase_effort_without_mutating_commands():
    from dataclasses import asdict
    from test_quality_dose_policy import menu
    config = dict(resting_hr=49,max_hr=194,zones={
        'z1':(128,140),'z2':(141,153),'z3':(154,166),'z4':(167,180),'z5':(181,194)})
    prescriptions = menu()['prescriptions']
    assert len({p.request.family for p in prescriptions}) == 6
    for p in prescriptions:
        before = asdict(p)
        bands = requested_work_bands(p.phases,config)
        assert len(bands) == len(p.phases)
        for phase,band in zip(p.phases,bands):
            if phase.kind != 'work':
                assert band is None
            else:
                assert isinstance(band,WorkExposure)
                assert (band.start_band != band.finish_band) == (p.request.family == 'progression')
        assert asdict(p) == before


def test_hills_share_z4_effort_query_with_short_intervals_not_a_family_multiplier():
    from test_quality_dose_policy import menu
    config = dict(resting_hr=49,max_hr=194,zones={'z4':(167,180)})
    commands = {}
    for p in menu()['prescriptions']:
        if p.request.family in {'hill_repeats','short_intervals'}:
            work = [q for q in requested_work_bands(p.phases,config) if q is not None]
            assert work
            assert all(q == work[0] for q in work)
            commands[p.request.family] = work[0]
    assert commands['hill_repeats'] == commands['short_intervals']


def test_harder_unsupported_query_discloses_baseline_not_intensity_transfer():
    low,lt = record('low',.55);high,ht = record('high',.85)
    records = [low,high];profiles = {'low':lt,'high':ht}
    hard = predict(records,profiles,(.8,.9))
    unsupported = predict(records,profiles,(.95,1.))
    assert unsupported['predicted_load'] < hard['predicted_load']
    assert unsupported['predicted_load'] == forecast_direct(LAYOUT,records,as_of=AT,accounting_key='units')['predicted_load']
    assert not unsupported['work_exposure_matching']['planner_influence']
    assert all(not p['work_exposure']['supported'] for p in unsupported['phases'] if p['kind']=='work')
    assert not unsupported['work_exposure_matching']['requested_intensity_rankable']


def test_ranking_handoff_does_not_turn_missing_intensity_into_a_discount():
    low,lt = record('low',.55);high,ht = record('high',.85)
    records = [low,high];profiles = {'low':lt,'high':ht}
    def ranked(band, layout=LAYOUT):
        return forecast_work_for_ranking(layout,records,profiles,query(layout,band),
            as_of=AT,accounting_key='units')
    assert ranked((.8,.9)) == predict(records,profiles,(.8,.9))
    unknown = ranked((.95,1.))
    assert unknown['status'] == 'unresolved_requested_work_exposure'
    assert unknown['predicted_load'] is None
    assert unknown['work_exposure_matching']['timing_only_proxy_load'] == predict(records,profiles,(.95,1.))['predicted_load']
    assert len(unknown['work_exposure_matching']['unresolved_phases']) == 2
    assert all(p['rate'] is None and p['load'] is None for p in unknown['phases'] if p['kind']=='work')
    assert all(p['rate'] is not None for p in unknown['work_exposure_matching']['diagnostic_work_costs'])
    baseline = forecast_direct(LAYOUT,records,as_of=AT,accounting_key='units')
    assert [p for p in unknown['phases'] if p['kind']!='work'] == [p for p in baseline['phases'] if p['kind']!='work']
    novel = ranked((.8,.9),(('work',0.,137.),))
    assert novel['status'] == 'predicted' and novel['predicted_load'] > 0


def test_ranking_handoff_keeps_unknown_cues_missing_traces_and_empty_history_unresolved():
    source,traces = record('past',.55)
    cases = (([source],{'past':traces},[None]*len(LAYOUT)),
             ([source],{},query(LAYOUT,(.5,.6))),([],{},query(LAYOUT,(.5,.6))))
    for records,profiles,requested in cases:
        result = forecast_work_for_ranking(LAYOUT,records,profiles,requested,as_of=AT,accounting_key='units')
        assert result['predicted_load'] is None
        assert not result['work_exposure_matching']['requested_intensity_rankable']


def test_selection_readiness_does_not_hide_fallback_reversals_behind_supported_pass():
    from scripts.audit_work_exposure_matching import selection_readiness
    probe = dict(rows=[dict(supported=True),dict(supported=False)],
        supported_intensity_reversals=[],duration_reversals=[],
        fallback_intensity_reversals=[dict(from_cue='Z4',to_cue='Z5')])
    recipes = dict(prescriptions=[dict(all_work_exposure_supported=False)])
    result = selection_readiness(probe,recipes)
    assert not result['ready_for_selection_validation'] and not result['production_acceptance']
    assert len(result['blockers']) == 3
    supported = dict(rows=[dict(supported=True)],supported_intensity_reversals=[],
        duration_reversals=[],fallback_intensity_reversals=[])
    assert selection_readiness(supported,dict(prescriptions=[dict(all_work_exposure_supported=True)]))['ready_for_selection_validation']
