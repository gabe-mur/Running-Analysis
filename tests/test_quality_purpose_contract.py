"""Q3 engineering scenarios, not fitted bands or calendar acceptance."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from run_analysis.adaptive.quality_prescription_contract import (
    Effort, Measure, PaceReference, QualityRequest, compile_quality,
)
from run_analysis.adaptive.quality_purpose_contract import (
    PurposeRequirement, QualityPurposeContract, assess_purpose_balance, assess_quality_purposes,
)
from run_analysis.signal_structure import SignalSpan
from run_analysis.work_stimulus import StimulusTarget, WorkSignals
from run_analysis.workload_accounting import Quantity

AT = datetime(2026, 10, 8, tzinfo=timezone.utc)


def point(value):
    return Quantity.point(value, 'test scenario, not accepted physiology')


def contract():
    return QualityPurposeContract(AT, 'fixture-stage', 7., (
        PurposeRequirement('sustained', StimulusTarget('sustained fixture', 'hrr',
            (.7, .8, .9, 1.), 10., 'illustrative test band, not a default'), .5, 300., 'test structure'),
        PurposeRequirement('faster', StimulusTarget('fast fixture', 'speed_over_easy',
            (1.05, 1.25, 1.6, 1.8), 5., 'illustrative test band, not a default'), .5, 0., 'test structure'),
    ), 'explicit test allocation, not accepted training mixture')


def prescription(family='short_intervals', seconds=60, count=8):
    m = lambda value: Measure('time', point(value))
    if family == 'progression':
        request = QualityRequest(family, 'second_half', AT, 'fixture',
            (Measure('portion', point(.5)),), total_miles=4.5)
    else:
        request = QualityRequest(family, 'test_timed', AT, 'fixture',
            (m(seconds),)*count, (m(90),)*(count-1))
    pace = lambda value: PaceReference(point(value), AT-timedelta(days=1), ('fixture-history',), 'test pace')
    return compile_quality(request, easy_pace=pace(.5), work_pace=pace(1/3))


def signals(p=None, *, hrr=.85, speed=3., observed=False, key='observed', windows=None):
    p = p or prescription()
    seconds = sum(phase.seconds.estimate for phase in p.phases)
    offset, work = 0., []
    for phase in p.phases:
        if phase.kind == 'work':
            work.append((offset, offset+phase.seconds.estimate))
        offset += phase.seconds.estimate
    start = AT-timedelta(days=1) if observed else AT+timedelta(days=1)
    end = start+timedelta(seconds=seconds)
    return WorkSignals(key if observed else p.key, 'observed' if observed else 'planned',
        start, end, end if observed else AT, tuple(work) if windows is None else windows,
        (SignalSpan(0, seconds, hrr, hrr),), (SignalSpan(0, seconds, speed, speed),),
        'detected_physical_work' if observed else 'compiled_work', point(2.),
        start-timedelta(days=1), 'prior easy pace scenario')


def assess(e, p=None, c=None, role='quality', as_of=AT):
    return assess_quality_purposes(e, contract=c or contract(), role=role, as_of=as_of, prescription=p)


def test_short_fast_reps_survive_hr_lag_but_do_not_claim_sustained_work():
    p = prescription()
    report = assess(signals(p, hrr=.65), p)
    assert report['credits']['faster'].low == 1.
    assert report['credits']['sustained'].high == 0.
    assert report['opportunity'].low == 1.
    assert not report['physiological_equivalence_established'] and not report['planner_influence']


def test_progression_can_supply_sustained_work_not_faster_work_even_with_fast_trace():
    p = prescription('progression')
    report = assess(signals(p), p)
    assert report['credits']['sustained'].low == 1.
    assert report['credits']['faster'].high == 0.


def test_observed_purposes_follow_physical_work_not_assignment_or_title():
    e = signals(prescription('threshold', 600, 1), observed=True, speed=2.)
    report = assess(e)
    assert report['credits']['sustained'].low == 1.
    assert report['credits']['faster'].high == 0.
    with pytest.raises(ValueError, match='assigned intent'):
        assess(e, prescription())


def test_dual_purpose_work_is_still_one_quality_opportunity():
    p = prescription('long_intervals', 300, 3)
    report = assess(signals(p), p)
    assert all(q.low == 1. for q in report['credits'].values())
    assert report['opportunity'].low == report['opportunity'].high == 1.
    balance = assess_purpose_balance((report,), contract=contract(), quality_opportunities=2.)
    assert balance['targets'] == {'sustained': 1., 'faster': 1.}
    assert balance['delivered_opportunities'].high == 1.
    assert balance['all_purposes_supported']
    assert balance['opportunity_deficit'].low == 1.
    assert not balance['all_requirements_supported']  # Cannot halve overall cadence.


def test_excess_short_work_cannot_pay_sustained_deficit_or_double_cadence():
    a = prescription(count=8)
    b = prescription(count=9)
    rows = (assess(signals(a), a), assess(signals(b), b))
    balance = assess_purpose_balance(rows, contract=contract(), quality_opportunities=2.)
    assert sum(balance['targets'].values()) == 2.
    assert balance['delivery']['faster'].low == 2.
    assert balance['deficits']['faster'].high == 0.
    assert balance['deficits']['sustained'].low == 1.
    assert not balance['all_purposes_supported']


def test_both_purposes_can_be_met_without_forced_family_rotation():
    # Same named family, genuinely different durations and evidence profiles.
    short = prescription('long_intervals', 120, 3)
    long = prescription('long_intervals', 300, 3)
    rows = (assess(signals(short, hrr=.65), short), assess(signals(long, speed=2.), long))
    balance = assess_purpose_balance(rows, contract=contract(), quality_opportunities=2.)
    assert balance['all_requirements_supported']


def test_same_work_minutes_at_easy_effort_do_not_buy_either_purpose():
    p = prescription('long_intervals', 300, 3)
    report = assess(signals(p, hrr=.65, speed=2.), p)
    assert all(q.high == 0. for q in report['credits'].values())


@pytest.mark.parametrize('role', ['easy', 'long'])
def test_hr_drift_and_easy_volume_cannot_buy_quality_credit(role):
    report = assess(signals(observed=True), role=role)
    assert all(q.high == 0. for q in report['credits'].values())


def test_recovery_and_bookend_padding_do_not_buy_purpose_credit():
    p = prescription('long_intervals', 300, 3)
    e = signals(p)
    low = tuple(SignalSpan(a, b, .65, .65) for a, b in e.windows)
    quiet = assess(replace(e, hrr=low), p)
    spans, offset = [], 0.
    for a, b in e.windows:
        if offset < a:
            spans.append(SignalSpan(offset, a, .85, .85))
        spans.append(SignalSpan(a, b, .65, .65)); offset = b
    elapsed = (e.end-e.start).total_seconds()
    if offset < elapsed:
        spans.append(SignalSpan(offset, elapsed, .85, .85))
    padded = assess(replace(e, hrr=tuple(spans)), p)
    assert quiet['credits'] == padded['credits']


def test_missing_signal_is_bounded_not_zero_or_an_exact_template_ban():
    p = prescription('long_intervals', 300, 3)
    e = signals(p)
    report = assess(replace(e, hrr=(), speed_mps=()), p)
    assert all(q.low == 0. and q.high == 1. for q in report['credits'].values())
    balance = assess_purpose_balance((report,), contract=contract(), quality_opportunities=2.)
    assert not balance['all_purposes_supported']
    assert all(q.low == 0. and q.high == 1. for q in balance['deficits'].values())


def test_cannot_attach_inconsistent_intent_or_planned_work_alignment():
    p = prescription()
    e = signals(p)
    with pytest.raises(ValueError, match='exact previously issued'):
        assess(e, prescription(count=7))
    with pytest.raises(ValueError, match='exact compiled'):
        assess(replace(e, windows=((0., 480.),)), p)
    easy = replace(p.phases[1], effort=Effort(('Z1', 'Z2'), ('Z1', 'Z2'), 'easy'))
    q = replace(p, phases=(p.phases[0], easy, *p.phases[2:]))
    report = assess(signals(q), q)
    assert report['reports']['faster']['supported_minutes'].low == pytest.approx(7.)
    assert report['credits']['faster'].low == 1.


def test_evidence_and_targets_must_be_available_and_frozen_before_planned_issue():
    p = prescription()
    e = signals(p)
    with pytest.raises(ValueError, match='not yet available'):
        assess(replace(e, available_at=AT+timedelta(seconds=1)), p)
    with pytest.raises(ValueError, match='not yet available'):
        assess(e, p, replace(contract(), issued_at=AT+timedelta(seconds=1)))
    with pytest.raises(ValueError, match='exact previously issued'):
        assess(e, p, replace(contract(), issued_at=AT+timedelta(seconds=1)), as_of=AT+timedelta(seconds=2))
    # Current frozen purposes may audit older completed data; this is not
    # retroactive evidence about its original prescription or a pre-run forecast.
    assert assess(signals(observed=True))['contract_key'] == contract().key


def test_contract_identity_covers_dose_band_structure_share_and_source():
    c = contract()
    first, second = c.requirements
    for changed in (replace(first, target=replace(first.target, minutes=12)),
                    replace(first, target=replace(first.target, band=(.65, .8, .9, 1.))),
                    replace(first, minimum_bout_seconds=240), replace(first, source='new source')):
        assert replace(c, requirements=(changed, second)).key != c.key
    assert replace(c, cadence_days=8.).key != c.key
    with pytest.raises(ValueError, match='share one existing'):
        replace(c, requirements=(replace(first, opportunity_share=.8), second))
    with pytest.raises(ValueError, match='matching exposure'):
        replace(first, target=second.target)


def test_duplicate_or_mixed_target_assessments_cannot_buy_delivery():
    p = prescription()
    report = assess(signals(p), p)
    with pytest.raises(ValueError, match='Unique evidence'):
        assess_purpose_balance((report, report), contract=contract(), quality_opportunities=2.)
    with pytest.raises(ValueError, match='Unique evidence'):
        assess_purpose_balance((report,), contract=replace(contract(), cadence_days=8.), quality_opportunities=2.)


def test_no_history_required_and_no_default_numerical_bands_are_promoted():
    # All numbers are supplied before assessment, not fitted to the candidate.
    with pytest.raises(TypeError):
        QualityPurposeContract()
    p = prescription()
    report = assess(replace(signals(p), easy_speed_mps=None, reference_at=None, reference_source=''), p)
    assert report['credits']['faster'].low == 0.
    assert report['credits']['faster'].high == 1.


def test_reviewed_hill_z4_intent_is_explicit_but_not_sustained_work():
    p = prescription('hill_repeats', 45, 8)
    report = assess(signals(p), p)
    assert report['credits']['faster'].low == 1.
    assert report['credits']['faster'].high == 1.
    assert report['reports']['faster']['unresolved_intent_windows'] == ()
    assert report['credits']['sustained'].high == 0.  # Not sustained bouts.


def test_splitting_signal_samples_preserves_purpose_credit():
    p = prescription('long_intervals', 300, 3)
    e = signals(p)
    seconds = (e.end-e.start).total_seconds()
    split = replace(e, hrr=(SignalSpan(0, seconds/2, .85, .85), SignalSpan(seconds/2, seconds, .85, .85)),
        speed_mps=(SignalSpan(0, seconds/2, 3., 3.), SignalSpan(seconds/2, seconds, 3., 3.)))
    assert assess(e, p)['credits'] == assess(split, p)['credits']


def test_unaligned_original_case_is_retained_not_fabricated_zero_or_dropped():
    e = replace(signals(observed=True), basis='unknown', windows=())
    report = assess(e)
    assert report['status'] == 'unresolved_physical_work_alignment'
    assert all(q.low == 0. and q.high == 1. for q in report['credits'].values())
    assert not assess_purpose_balance((report,), contract=contract(), quality_opportunities=2.)['all_purposes_supported']


def test_fingerprints_include_physical_signals_even_when_credits_tie():
    p = prescription('long_intervals', 300, 3)
    a, b = signals(p), signals(p, hrr=.86)
    first, second = assess(a, p), assess(b, p)
    assert first['credits'] == second['credits']
    assert first['fingerprint'] != second['fingerprint']
