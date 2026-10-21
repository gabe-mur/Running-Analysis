from dataclasses import replace
from datetime import timedelta

import pytest

from scripts.audit_easy_distribution import exposures, program_option
from scripts.audit_easy_placement import ORIGIN
from scripts.audit_easy_tolerance import inputs, policy_for, panel, stacked_records
from scripts.easy_tolerance_trial import assess, prior_exposure
from run_analysis.session_decision import CurrentContext
from run_analysis.workload_accounting import Quantity


@pytest.fixture(scope='module')
def data():
    base, rates, history, refs = inputs()
    policy = policy_for(base, refs)
    option = program_option('eight', exposures(8,ORIGIN), history, rates)
    today = next(e for e in option.demands if e.day == ORIGIN)
    return base, history, policy, today, option


@pytest.fixture(scope='module')
def report():
    return panel()


def test_all_fifteen_improvements_survive_all_six_main_interval_scenarios(report):
    assert len(report['scenarios']) == 6
    for s in report['scenarios'].values():
        assert len(s['improvements']) == 15
        assert all(r['main_statuses'] == {'provisional_dose_option': 8} for r in s['improvements'])
        # Explicit failure, not a shortened horizon that hides the tail.
        assert all(r['tail_statuses'] == {'outside_local_dose_policy': 2} for r in s['improvements'])
    assert not report['program_approved'] and not report['planner_influence']


@pytest.mark.parametrize('name,status', [
    ('unknown_response','provisional_dose_option'),
    ('usual','provisional_dose_option'),
    ('mild_fatigue','provisional_dose_option'),
    ('health_concern','review_current_context'),
    ('unexpected_response_without_current_check','request_current_context'),
    ('future_report_ignored','provisional_dose_option'),
    ('token','below_aerobic_minimum'),
    ('ordinary_after_quality','outside_local_dose_policy'),
    ('large_easy_after_quality','outside_local_dose_policy'),
    ('double_previous_dose','review_prior_dose'),
    ('ordinary_after_one_rest_date','provisional_dose_option'),
    ('unresolved_health_after_rollover','review_current_context'),
])
def test_declared_response_and_dose_controls(report, name, status):
    assert report['controls'][name]['status'] == status
    assert not report['controls'][name]['program_approved']


def test_same_volume_stacked_key_control_exposes_missing_sequence_decision(report):
    original = report['programs']['original_eight']
    stacked = report['programs']['stacked_quality']
    assert stacked['description']['miles'] == pytest.approx(original['description']['miles'])
    assert stacked['description']['counts'] == original['description']['counts']
    assert sorted((e['key'],e['miles'],e['role']) for e in stacked['calendar']) == sorted(
        (e['key'],e['miles'],e['role']) for e in original['calendar'])
    assert len({e.day for e in stacked_records()}) == len(stacked_records())
    assert stacked['constraints']['status'] == 'passes_checked_constraints_conditionally'
    main = [c for c in stacked['easy_checks'] if c['day'] < ORIGIN+timedelta(days=28)]
    assert all(c['status'] == 'provisional_dose_option' for c in main)
    assert stacked['description']['rolling_peaks']['7']['quality_work_minutes'] == 45
    assert original['description']['rolling_peaks']['7']['quality_work_minutes'] == 15
    assert stacked['actual_plus_planned_rolling_peaks']['peaks']['7'][
        'detected_or_prescribed_quality_work_minutes'] > 45
    assert original['actual_plus_planned_rolling_peaks']['peaks']['7']['miles_low'] > 17.3
    assert any(c['order'] not in {'equal','right_no_greater'} for c in stacked['key_comparisons'])
    assert all(c['sequence_review_required'] and not c['program_approved'] for c in main)


def test_canonical_witness_comparison_still_protects_key_opportunities(report):
    witness = report['programs']['witness']
    assert len(witness['key_comparisons']) == 12
    assert all(c['order'] in {'equal','right_no_greater'} for c in witness['key_comparisons'])
    assert any(c['order'] == 'right_no_greater' for c in witness['key_comparisons'])


def test_reduced_policy_does_not_establish_optimal_frequency(report):
    central = report['scenarios']['central:0.65']['distributions']
    upper = report['scenarios']['upper_quartile:0.65']['distributions']
    for count in (8,12,16):
        assert central[str(count)]['main_statuses'] == {'provisional_dose_option':count}
    assert central['20']['main_statuses'] == {'below_aerobic_minimum':20}
    assert upper['16']['main_statuses'] == {'below_aerobic_minimum':16}


def test_rounding_boundary_is_reported_not_silently_changed(report):
    for fraction, within, outside in [('0.5','2.2','2.3'),('0.65','2.8','2.9'),('0.8','3.5','3.6')]:
        assert report['dose_sensitivity'][fraction][within] == 'provisional_dose_option'
        assert report['dose_sensitivity'][fraction][outside] == 'outside_local_dose_policy'


def test_frozen_reference_does_not_learn_from_projected_large_predecessor(data):
    base, history, policy, today, _ = data
    huge = replace(history[-1],source='planned', terms=tuple(replace(t,coefficient=3*t.coefficient)
                                                          for t in history[-1].terms))
    a = assess(today,(*history[:-1],huge),policy,as_of=base['as_of'])
    b = assess(today,(*history[:-1],replace(huge,source='observed')),policy,as_of=base['as_of'])
    assert a['status'] == b['status'] == 'review_prior_dose'
    assert a['previous_is_conditional'] and not b['previous_is_conditional']


def test_short_previous_distance_cannot_hide_definite_cardio_excess(data):
    base, history, policy, today, _ = data
    previous = replace(history[-1],terms=tuple(replace(t,coefficient=t.coefficient*3)
        if t.channel == 'cardio' else t for t in history[-1].terms))
    r = assess(today,(*history[:-1],previous),policy,as_of=base['as_of'])
    assert r['prior_exposure']['channels']['cardio'] == 'above_reference'
    assert r['status'] == 'review_prior_dose'


def test_uncertain_previous_hr_is_neither_fabricated_clearance_nor_automatic_rest(data):
    base, history, policy, today, _ = data
    previous = replace(history[-1],terms=tuple(replace(t,bounds=None)
        if t.channel == 'cardio' else t for t in history[-1].terms))
    r = assess(today,(*history[:-1],previous),policy,as_of=base['as_of'])
    assert r['prior_exposure']['status'] == 'uncertain'
    assert r['status'] == 'provisional_dose_option' and not r['recovery_cleared']


def test_ordinary_familiarity_is_not_matching_exact_workout_bouts(data):
    _, history, policy, _, _ = data
    varied = replace(history[-1],structure=(110.,105.,100.,95.,100.,120.))
    assert prior_exposure(varied,policy)['status'] == prior_exposure(history[-1],policy)['status']


def test_fast_work_cannot_be_relabeled_as_easy(data):
    base, history, policy, today, _ = data
    hidden = replace(today,terms=tuple(replace(t,coefficient=5,bounds=Quantity.point(1,'test'))
        if t.channel == 'work_minutes' else t for t in today.terms))
    assert assess(hidden,history,policy,as_of=base['as_of'])['status'] == 'easy_intensity_not_established'


def test_future_normal_context_cannot_be_invented_and_stale_normal_does_not_carry(data):
    base, history, policy, today, _ = data
    later = replace(today,day=ORIGIN+timedelta(days=2))
    current = CurrentContext(ORIGIN,base['as_of'],'usual','test')
    r = assess(later,history,policy,as_of=base['as_of'],context=current)
    assert r['state'] == 'unknown' and r['current_response_unobserved']
    impossible = replace(current,applies_on=later.day)
    with pytest.raises(ValueError,match='future readiness'):
        assess(later,history,policy,as_of=base['as_of'],context=impossible)


def test_current_mild_fatigue_does_not_predict_future_fatigue_or_recovery(data):
    base, history, policy, today, _ = data
    current = CurrentContext(ORIGIN,base['as_of'],'little_tired','test')
    r = assess(replace(today,day=ORIGIN+timedelta(days=2)),history,policy,as_of=base['as_of'],context=current)
    assert r['status'] == 'request_session_date_context'


def test_same_date_refresh_and_accounting_source_parity(data):
    base, history, policy, today, _ = data
    a = assess(today,history,policy,as_of=base['as_of'])
    b = assess(today,history,policy,as_of=base['as_of']+timedelta(hours=6))
    c = assess(today,(*history[:-1],replace(history[-1],source='planned')),policy,as_of=base['as_of'])
    assert a == b
    assert a == dict(c,previous_is_conditional=False)
    # This is local evaluator parity, not proof of daily replanning stability.


@pytest.mark.parametrize('change',['duplicate','future','units'])
def test_bad_ledger_is_not_accepted(data,change):
    base, history, policy, today, _ = data
    bad = {'duplicate':(*history,history[-1]),
           'future':(*history[:-1],replace(history[-1],day=today.day)),
           'units':(*history[:-1],replace(history[-1],accounting_key='other'))}[change]
    with pytest.raises(ValueError):
        assess(today,bad,policy,as_of=base['as_of'])


@pytest.mark.parametrize('fraction',[0,1,float('nan')])
def test_policy_requires_explicit_nontrivial_reduction(data,fraction):
    with pytest.raises(ValueError):
        replace(data[2],reduced_fraction=fraction)
