"""Frozen Q3 policy scenarios; no physiology or daily/calendar acceptance claim."""
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from math import fsum

import pytest

from run_analysis.adaptive.quality_dose_policy import (
    QualityDosePolicy, QualityDoseStage, compile_dose_menu, purpose_contract, recipe_requests,
)
from run_analysis.adaptive.quality_prescription_contract import PaceReference
from run_analysis.adaptive.quality_purpose_contract import assess_quality_purposes, assess_purpose_balance
from run_analysis.signal_structure import SignalSpan
from run_analysis.work_stimulus import (
    CappedStimulusTarget, WorkSignals, assess_work_stimulus, _shared_reference_bounds, _integrated_suitability,
)
from run_analysis.workload_accounting import Quantity

AT = datetime(2026, 10, 8, tzinfo=timezone.utc)
CONFIG = dict(resting_hr=49, max_hr=194, zones=dict(z3=(154, 166), z4=(167, 180)))


def pace(speed, source):
    return PaceReference(Quantity(1/speed*.9, 1/speed*1.1, 1/speed, 'scenario, not confidence'),
        AT-timedelta(days=1), (source,), source)


def stage(work=10.5, ordinary=4.2):
    return QualityDoseStage('accepted-test-stage', AT-timedelta(days=1), ordinary, work, 7., 'frozen test stage')


def contract(s=None, easy=2.5, fast=3.7):
    return purpose_contract(s or stage(), config=CONFIG, issued_at=AT,
        easy_pace=pace(easy, 'earlier easy scenario'), faster_pace=pace(fast, 'earlier faster-effort scenario'))


def menu(s=None, recovery=90):
    s = s or stage()
    return compile_dose_menu(s, contract=contract(s), easy_pace=pace(2.5, 'easy'),
        purpose_paces={'sustained': pace(3., 'sustained'), 'faster': pace(3.7, 'faster'),
            'gradual': pace(2.8, 'gradual')}, open_recovery_seconds=recovery,
        scenario_source='explicit open-recovery scenario, not an instruction')


def trace(p, speed=3.7, hrr=.85):
    offset, windows = 0., []
    for phase in p.phases:
        if phase.kind == 'work':
            windows.append((offset, offset+phase.seconds.estimate))
        offset += phase.seconds.estimate
    start = AT+timedelta(days=1)
    return WorkSignals(p.key, 'planned', start, start+timedelta(seconds=offset), AT,
        tuple(windows), (SignalSpan(0, offset, hrr, hrr),), (SignalSpan(0, offset, speed, speed),),
        'prescribed')


def test_common_targets_use_configured_athlete_inputs_and_one_unchanged_cadence():
    c = contract()
    steady, fast = c.requirements
    assert c.cadence_days == 7.
    assert sum(r.opportunity_share for r in c.requirements) == 1.
    assert steady.target.minutes == pytest.approx(10.5)
    assert fast.target.minutes == pytest.approx(5.6)
    assert steady.target.onset == pytest.approx((154-49)/(194-49))
    assert steady.target.full == pytest.approx((167-49)/(194-49))
    assert fast.target.onset == pytest.approx(2.5) and fast.target.full == pytest.approx(3.7)
    assert fast.target.channel == 'speed_mps'


def test_native_structures_have_distinct_doses_not_six_equal_work_clocks():
    requests = recipe_requests(stage(), issued_at=AT, open_recovery_seconds=90, scenario_source='explicit scenario')
    ordinary = {r.family: r for r in requests if r.variant.endswith('ordinary')}
    assert len(ordinary['short_intervals'].work) == 6
    assert [w.amount.estimate for w in ordinary['short_intervals'].work] == [60]*6
    assert len(ordinary['short_intervals'].recoveries) == 6  # Authored after-each recovery.
    assert [w.amount.estimate for w in ordinary['long_intervals'].work] == [210]*3
    assert len(ordinary['long_intervals'].recoveries) == 2
    assert ordinary['threshold'].work[0].amount.estimate == 750
    assert ordinary['fartlek'].open_recovery
    assert ordinary['hill_repeats'].work[0].amount.estimate == 45
    progression = next(r for r in requests if r.family == 'progression' and r.total_miles == 4.2)
    assert progression.work[0].unit == 'portion' and progression.work[0].amount.estimate == .5


def test_distance_reps_remain_distance_and_time_recipes_remain_time():
    rows = menu()['prescriptions']
    for p in rows:
        primary = p.request.work[0].unit
        if p.request.variant.startswith(('400m', '500m')):
            assert primary == 'distance'
            assert p.request.work[0].amount.estimate in {400, 500}
            assert ' m' in p.steps()[1].instruction
        elif p.request.family == 'progression':
            assert primary == 'portion'
        else:
            assert primary == 'time'
    long = next(p for p in rows if p.request.variant == 'timed_long_ordinary')
    assert '3 × 3:30' in long.steps()[1].instruction


def test_integer_reps_do_not_manufacture_identical_reduced_alternatives():
    rows = menu()['prescriptions']
    identities = [(p.request.family, p.request.work, p.request.recoveries, p.request.total_miles) for p in rows]
    assert len(identities) == len(set(identities))
    assert all(p.miles == round(p.miles, 1) for p in rows)
    for p in rows:
        assert fsum(s.duration_minutes for s in p.steps())*60 == pytest.approx(fsum(s.seconds.estimate for s in p.phases))


def test_same_stage_quality_budget_is_not_increased_with_total_volume():
    a = menu(stage(work=10.5, ordinary=4.2))
    b = menu(stage(work=10.5, ordinary=5.2))
    for family in ('short_intervals', 'long_intervals', 'threshold', 'fartlek', 'hill_repeats'):
        targets = lambda rows: [[(w.unit, w.amount.low, w.amount.high) for w in p.request.work]
            for p in rows['prescriptions'] if p.request.family == family]
        assert targets(a) == targets(b)
    assert [r.target.minutes for r in a['contract'].requirements] == [r.target.minutes for r in b['contract'].requirements]


def test_stage_scale_cannot_compound_quality_growth_past_native_reference():
    a = recipe_requests(stage(work=15), issued_at=AT)
    b = recipe_requests(stage(work=30), issued_at=AT)
    commands = lambda requests: [[[(m.unit, m.amount.low, m.amount.high) for m in measures]
        for measures in (p.work, p.recoveries)] for p in requests]
    assert commands(a) == commands(b)
    assert contract(stage(work=30)).requirements[0].target.minutes == 15.


def test_small_stage_and_novel_duration_do_not_require_matching_history():
    result = menu(stage(work=3, ordinary=1.5))
    assert result['prescriptions'] and not result['forecasting_validated']
    long = next(p for p in result['prescriptions'] if p.request.variant == 'timed_long_ordinary')
    assert len(long.request.work) == 1 and long.request.work[0].amount.estimate == 180
    assert all(w.amount.estimate >= 120 for p in result['prescriptions']
        if p.request.family == 'long_intervals' for w in p.request.work)
    assert result['candidate_admission'] == 'requires_shared_forecast_and_existing_stage_checks'


def test_missing_open_recovery_keeps_unresolved_families_explicit():
    result = menu(recovery=None)
    assert result['unresolved_families'] == ('fartlek', 'hill_repeats')
    assert not any(p.request.family in result['unresolved_families'] for p in result['prescriptions'])
    with pytest.raises(ValueError, match='sourced duration'):
        recipe_requests(stage(), issued_at=AT, open_recovery_seconds=90)


def test_future_wrong_effort_or_pace_scenarios_are_rejected_before_allocation():
    with pytest.raises(ValueError, match='Future pace'):
        purpose_contract(stage(), config=CONFIG, issued_at=AT, easy_pace=pace(2.5, 'easy'),
            faster_pace=replace(pace(3.7, 'fast'), issued_at=AT+timedelta(seconds=1)))
    with pytest.raises(ValueError, match='distinct explicitly sourced'):
        contract(easy=3.7, fast=2.5)
    with pytest.raises(ValueError, match='Stage not yet'):
        recipe_requests(replace(stage(), issued_at=AT+timedelta(seconds=1)), issued_at=AT)


def test_capped_exposure_cannot_drop_when_speed_or_hr_rises_above_target():
    c = contract()
    steady, faster = [r.target for r in c.requirements]
    assert [faster.suitability(v) for v in (2.5, 3.1, 3.7, 5.)] == pytest.approx([0, .5, 1, 1])
    assert steady.suitability(1.) == 1.
    p = menu()['prescriptions'][0]
    low = assess_work_stimulus(trace(p, speed=3.7), faster)
    high = assess_work_stimulus(trace(p, speed=5.), faster)
    assert low['coverage'] == high['coverage']  # No unlimited benefit either.


def test_reference_rescaling_preserves_faster_credit_without_easy_ratio_band():
    p = menu()['prescriptions'][0]
    a = assess_work_stimulus(trace(p, speed=3.1), contract().requirements[1].target)
    b = assess_work_stimulus(trace(p, speed=6.2), contract(easy=5., fast=7.4).requirements[1].target)
    assert a['coverage'].low == pytest.approx(b['coverage'].low)


def test_requested_progression_cannot_replace_faster_work_indefinitely():
    result = menu()
    c = result['contract']
    p = next(p for p in result['prescriptions'] if p.request.family == 'progression')
    report = assess_quality_purposes(trace(p, speed=3.7), contract=c, role='quality', as_of=AT, prescription=p)
    assert report['credits']['sustained'].low == 1.
    assert report['credits']['faster'].high == 0.
    balance = assess_purpose_balance((report,), contract=c, quality_opportunities=2.)
    assert balance['deficits']['faster'].low == 1.


def test_threshold_can_have_partial_fast_exposure_without_claiming_equal_stimulus():
    result = menu()
    c = result['contract']
    p = next(p for p in result['prescriptions'] if p.request.family == 'threshold')
    steady_speed = 2.8
    report = assess_quality_purposes(trace(p, speed=steady_speed), contract=c, role='quality', as_of=AT, prescription=p)
    assert report['credits']['sustained'].low == 1.
    assert 0 < report['credits']['faster'].high < 1.


def test_new_cap_target_keeps_exact_shared_reference_extrema_and_sampling_invariance():
    target = CappedStimulusTarget('test', 'speed_over_easy', 1., 1.3, 5., 'explicit scenario')
    pieces = [(60., 1.8, 3.4), (90., 3.5, 2.)]
    bounds = _shared_reference_bounds(pieces, Quantity(2., 3.), target)
    values = [fsum(dt*_integrated_suitability(a/r, b/r, target)
        for dt, a, b in pieces) for r in (2+n/1000 for n in range(1001))]
    assert bounds[0] <= min(values)+1e-8 and bounds[1] >= max(values)-1e-8
    assert bounds[0] == pytest.approx(min(values)) and bounds[1] == pytest.approx(max(values))


def test_target_and_menu_identities_cover_coaching_references_and_policy():
    a = menu()
    assert menu()['key'] == a['key']
    assert contract(easy=2.6).key != contract().key
    assert replace(QualityDosePolicy(), sustained_share=.6) != QualityDosePolicy()
    assert not a['planner_influence']
    assert asdict(a['contract'])['requirements'][1]['target']['version'] == 'work_signal_capped_exposure_v1'


def test_changed_policy_or_dose_cannot_reuse_old_stage_targets():
    s = stage()
    args = dict(contract=contract(s), easy_pace=pace(2.5, 'easy'),
        purpose_paces={'sustained': pace(3., 'sustained'), 'faster': pace(3.7, 'faster'),
            'gradual': pace(2.8, 'gradual')})
    with pytest.raises(ValueError, match='Common stage contract'):
        compile_dose_menu(s, policy=replace(QualityDosePolicy(), sustained_share=.6), **args)
    with pytest.raises(ValueError, match='Common stage contract'):
        compile_dose_menu(replace(s, quality_work_minutes=12.), **args)


def test_tiny_stage_missing_sustained_variants_are_not_hidden_history_bans():
    result = menu(stage(work=1, ordinary=1.5))
    assert result['stage_unsupported_families'] == ('long_intervals', 'threshold')
    assert not result['unresolved_families']
    assert result['forecasting_validated'] is False


def test_fartlek_retains_flexible_primary_targets_not_identical_stopwatch_reps():
    p = next(p for p in menu()['prescriptions'] if p.request.family == 'fartlek')
    assert p.request.work[0].amount == Quantity(45, 90, 60, p.request.work[0].amount.basis)
    assert '6 × 45–90 seconds' in p.steps()[1].instruction
    assert p.steps()[1].work_duration_range_minutes == (.75, 1.5)
    assert p.steps()[1].work_duration_minutes == 1.  # Explicit forecast nominal, not command rewrite.
    assert p.request.open_recovery
