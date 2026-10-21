from datetime import datetime, timedelta, timezone
import json

import pytest

from scripts.audit_forecast_scope import baseline, issued_case, load_error, paired_summary, prior_records
from scripts.direct_quality_load import PhaseDose, PhaseHistory


AT = datetime(2026,9,27,tzinfo=timezone.utc)


def record(key='past', **changes):
    result = dict(key=key,role='easy',label='easy',eligible=True,start=AT-timedelta(days=1),
        end=AT-timedelta(days=1)+timedelta(minutes=30),available_at=AT-timedelta(hours=12),
        observed_load=30.,known_seconds=1800.,elapsed_seconds=1800.,miles=3.,complete=True,
        bounds=dict(low=30.,high=30.),coverage=1.)
    return dict(result,**changes)


def match(payload, **changes):
    base = dict(activity_id='target',id=1,workout_type='easy',schedule_generated_at=AT.isoformat(),
                recommendation_json=json.dumps(dict(generated_at=AT.isoformat(),**payload)))
    return dict(base,**changes)


def test_history_cut_is_causal_recent_and_honest_about_assumed_availability():
    rows = [record(),record('late',available_at=AT+timedelta(days=1)),
            record('old',start=AT-timedelta(days=31)),record('unhealthy',eligible=False),
            record('future',end=AT+timedelta(seconds=1)),record('quality',role='quality')]
    assert [r['key'] for r in prior_records(rows,AT,{'easy'},availability='recorded')]==['past']
    assert [r['key'] for r in prior_records(rows,AT,{'easy'},availability='completion_assumed')]==['past','late']


def test_equal_session_baseline_and_missing_hr_are_not_zero_load():
    result = baseline([record(),record('short',known_seconds=60.,observed_load=2.,elapsed_seconds=60.,miles=.1),
                       record('nohr',known_seconds=0.,observed_load=None,complete=False)])
    assert result['rate']==pytest.approx((1/60+2/60)/2)
    assert result['rate_source_keys']==['past','short']
    assert baseline([])['rate'] is None
    assert load_error(30.,dict(low=20.,high=40.),complete=False)['log_error'] is None
    assert load_error(30.,None,complete=False) is None
    assert load_error(0.,dict(low=0.,high=0.),complete=True)['log_error'] is None


def test_complete_hr_paired_summary_does_not_discard_missing_inventory():
    good = load_error(30.,dict(low=30.,high=30.),complete=True)
    partial = load_error(30.,dict(low=20.,high=40.),complete=False)
    result = paired_summary([dict(key='a',x=good,y=good),dict(key='b',x=partial,y=good)],('x','y'))
    assert result['inventory']==2
    assert result['paired_keys']==['a']
    assert result['models']['x']['mean_absolute_log_error']==0


def test_saved_distance_edges_use_prior_pace_not_actual_duration_or_hr():
    target = record('target',start=AT+timedelta(hours=1),end=AT+timedelta(hours=2),
                    available_at=AT+timedelta(hours=3),elapsed_seconds=3600.,miles=4.)
    prior = record()
    phase = PhaseHistory('past',prior['start'],prior['end'],prior['available_at'],'units',
                          (PhaseDose('warmup',1800.,None,1800.,30.),))
    plan = match(dict(distance_range_miles=[3.,4.],structure=[]))
    a = issued_case(plan,[prior,target],[phase],'units')
    changed = dict(target,elapsed_seconds=9000.,observed_load=99.,bounds=dict(low=99.,high=99.))
    b = issued_case(plan,[prior,changed],[phase],'units')
    assert a['scenarios']['low']['seconds']==1800
    assert a['scenarios']['high']['seconds']==2400
    for edge in ('low','high'):
        assert a['scenarios'][edge]['direct']['predicted']==b['scenarios'][edge]['direct']['predicted']
    assert a['response_allowance'] is None
    assert not a['uncertainty_calibrated']


def test_missing_extent_or_history_does_not_invent_precise_duration():
    target = record('target',start=AT+timedelta(hours=1))
    result = issued_case(match(dict(distance_range_miles=[3.,4.])),[target],[],'units')
    assert result['status']=='unresolved_forecast'
    assert not result['scenarios']


def test_saved_after_run_is_not_pre_run_evidence():
    result = issued_case(match({}),[record('target')],[],'units')
    assert result['status']=='invalid_pre_run_provenance'


def test_unknown_quality_timing_never_filled_from_completion():
    target = record('target',role='quality',start=AT+timedelta(hours=1))
    result = issued_case(match(dict(structure=[]),workout_type='intervals'),[target],[],'units')
    assert result['status']=='unresolved_forecast'
    assert result['contract_status']=='unsupported_structure_shape'
