from dataclasses import replace
from datetime import timedelta

import pytest

from scripts.audit_easy_distribution import exposures, program_option
from scripts.audit_easy_placement import ORIGIN, records
from scripts.audit_easy_tolerance import inputs, policy_for
from scripts.audit_sequence_dose import panel
from scripts.sequence_dose_trial import compare, concentration
from run_analysis.session_decision import CurrentContext


@pytest.fixture(scope='module')
def report():
    return panel()


@pytest.fixture(scope='module')
def data():
    base,rates,history,refs = inputs()
    reference = program_option('reference',exposures(8,ORIGIN),history,rates)
    candidate = program_option('witness',records(8,(0,4,7,12,14,19,21,26)),history,rates)
    return (reference,candidate,history,policy_for(base,refs)),dict(as_of=base['as_of'],
        start=ORIGIN,main_end=ORIGIN+timedelta(days=28),end=ORIGIN+timedelta(days=42))


def test_useful_move_is_a_disclosed_tradeoff_not_full_program_approval(report):
    r = report['reports']['witness']
    assert r['decision'] == 'provisional_spacing_trial_not_recovery_approval'
    assert r['total_concentration_tradeoff']
    two = next(x for x in r['concentration']['total_miles']['changed_windows'] if x['days']==2)
    # Common tail includes 4.5 quality + 4.4 ordinary easy on adjacent dates.
    assert two['reference'] == pytest.approx(8.9)
    assert two['candidate'] == pytest.approx(10.6)
    assert r['concentration']['long_miles']['order'] == 'equal'
    assert r['concentration']['known_quality_work_minutes']['order'] == 'equal'
    assert any(c['order']=='left_no_greater' for c in r['easy_opportunities'])
    assert all(c['order'] in {'equal','right_no_greater'} for c in r['key_opportunities'])
    assert not r['program_approved'] and not r['recovery_cleared']


def test_all_fifteen_preserve_key_work_but_are_not_declared_optima(report):
    assert report['improvement_decisions'] == {'provisional_spacing_trial_not_recovery_approval':15}
    assert not report['program_approved']
    assert not report['supplied_evening_fit_used_in_noon_snapshot']


@pytest.mark.parametrize('case,decision',[
    ('unchanged','no_key_preparation_preference'),
    ('stacked','reject_uncompensated_key_concentration'),
    ('three_easy','receiving_session_tradeoff_requires_review'),
    ('health_concern','review_current_context'),
    ('unexpected_response','response_specific_dose_review'),
    ('underfunded','changed_volume_requires_stage_review'),
    ('doubled_quality_work','changed_key_prescription_requires_purpose_review'),
    ('changed_tail','different_continuation_requires_joint_evaluation'),
])
def test_contrasting_whole_sequence_cases(report,case,decision):
    assert report['reports'][case]['decision'] == decision


def test_ordinary_tail_easy_is_not_forced_shorter_after_quality(report):
    checks = report['reports']['witness']['easy_dose_checks']
    tail = [c for c in checks if c['day'] >= ORIGIN+timedelta(days=28)]
    assert len(tail)==2
    assert all(c['committed_miles']==4.4 and c['passes_planned_minimum'] for c in tail)


def test_reduction_fraction_is_not_a_hidden_dependency(data):
    args,kwargs = data
    expected = compare(*args,**kwargs)
    for fraction in (.5,.65,.8):
        assert compare(*args[:3],replace(args[3],reduced_fraction=fraction),**kwargs)==expected


def test_same_date_refresh_and_identical_inputs_are_repeatable(data):
    args,kwargs = data
    assert compare(*args,**kwargs) == compare(*args,**dict(kwargs,as_of=kwargs['as_of']+timedelta(hours=6)))
    assert compare(*args,**kwargs) == compare(*args,**kwargs)


def test_profile_is_direct_all_window_sum_with_actual_lead_in(data):
    args,kwargs = data
    _,option,history,_ = args
    actual = concentration(history,option,kwargs['start'],kwargs['end'])
    all_events = (*history,*option.demands)
    assert actual['window_days']==list(range(1,43))
    for span in (1,2,7,21,42):
        peaks = []
        for offset in range(42):
            day = ORIGIN+timedelta(days=offset)
            recent = [e for e in all_events if day-timedelta(days=span-1)<=e.day<=day]
            peaks.append(sum(t.coefficient*t.bounds.low for e in recent for t in e.terms if t.channel=='miles'))
        assert actual['channels']['total_miles'][span-1]==pytest.approx(max(peaks))
    assert actual['channels']['total_miles'][6] > 17.3


def test_identical_physical_exposure_does_not_change_with_source_label(data):
    args,kwargs = data
    _,option,history,_ = args
    before = concentration(history,option,kwargs['start'],kwargs['end'])
    assert before == concentration(tuple(replace(e,source='planned') for e in history),option,
                                   kwargs['start'],kwargs['end'])


def test_fixed_scope_does_not_lose_or_gain_load_from_future_extension(data):
    args,kwargs = data
    _,option,history,_ = args
    added = replace(option.demands[-1],key='outside',day=kwargs['end']+timedelta(days=5))
    exposure = replace(option.exposures[-1],key=added.key,day=added.day)
    extended = replace(option,demands=(*option.demands,added),exposures=(*option.exposures,exposure))
    assert concentration(history,option,kwargs['start'],kwargs['end']) == concentration(
        history,extended,kwargs['start'],kwargs['end'])


def test_unknown_context_does_not_become_health_clearance_and_stale_concern_persists(data):
    args,kwargs=data
    context=CurrentContext(ORIGIN-timedelta(days=1),kwargs['as_of']-timedelta(days=1),'health_concern','test')
    assert compare(*args,**kwargs,context=context)['decision']=='review_current_context'
    future=replace(context,available_at=kwargs['as_of']+timedelta(hours=1))
    assert compare(*args,**kwargs,context=future)['decision']=='provisional_spacing_trial_not_recovery_approval'


def test_accounting_mismatch_cannot_create_better_calendar(data):
    args,kwargs=data
    reference,candidate,history,policy=args
    row=candidate.exposures[0]
    # Existing ProgramOption validation rejects this before selection can see it.
    with pytest.raises(ValueError,match='same distance'):
        replace(candidate,exposures=(replace(row,miles=row.miles+.1),*candidate.exposures[1:]))


def test_missing_external_dose_is_not_zero(data):
    args,kwargs=data
    _,option,history,_=args
    bad=replace(history[-1],terms=tuple(replace(t,bounds=None) if t.channel=='miles' else t for t in history[-1].terms))
    with pytest.raises(ValueError,match='known distance'):
        concentration((*history[:-1],bad),option,kwargs['start'],kwargs['end'])
