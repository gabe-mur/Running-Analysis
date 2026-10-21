from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

import pytest

from run_analysis.development_report import ProgramOption
from run_analysis.program_delivery import ProgramExposure
from run_analysis.program_constraints import DateRestriction
from run_analysis.recency_demand import CHANNELS, DemandEvent, DemandTerm
from run_analysis.session_decision import CoachingContract, CurrentContext, DoseLimits, SessionAlternative, decide_session
from run_analysis.workload_accounting import Quantity

DAY = date(2026, 9, 25)
AT = datetime(2026, 9, 25, 12, tzinfo=timezone.utc)


def demand(key, day, miles, purpose='easy', bouts=None, source='planned'):
    quantities = dict(miles=miles, running_minutes=miles*10, cardio=miles*15, work_minutes=sum(bouts or ())/60)
    return DemandEvent(key, day, tuple(DemandTerm(c, 'unit', quantities[c], Quantity.point(1., 'fixture'))
                                      for c in CHANNELS), 'same_units', purpose, bouts, source)


def option(key='issued', miles=4.4, offset=0, purpose='easy', bouts=None, recovery=()):
    target = demand('target', DAY+timedelta(days=offset), miles, purpose, bouts)
    future = demand('following_long', DAY+timedelta(days=4), 8.4, 'long')
    exposures = tuple(ProgramExposure(e.key, e.day, miles if e.key == 'target' else 8.4,
        'quality' if e.purpose == 'threshold' else e.purpose, 'planned',
        quality_purpose=e.purpose if e.purpose == 'threshold' else None,
        work_bout_seconds=e.structure) for e in (target, future))
    program = ProgramOption(key, exposures, (target, future))
    return SessionAlternative(key, program, 'target', 'easy_controlled' if purpose == 'easy' else 'threshold_controlled', recovery)


def contract(quality=False, trial=False):
    q = lambda a, b: Quantity(a, b, basis='declared_fixture_not_validated_default')
    if quality:
        ordinary = DoseLimits(q(4., 5.), q(900, 1200), q(100, 300), q(.4, 1.))
        reduced = DoseLimits(q(3., 4.5), q(600, 600), q(100, 300), q(.4, 1.))
        return CoachingContract(AT-timedelta(days=1), 'explicit_quality_fixture', 'threshold',
                                'threshold_controlled', ordinary, reduced)
    return CoachingContract(AT-timedelta(days=1), 'explicit_easy_fixture', 'easy', 'easy_controlled',
        DoseLimits(q(4., 4.5)), DoseLimits(q(3., 3.5)), DoseLimits(q(4.6, 4.8)) if trial else None,
        trial_target=4.8 if trial else None)


def decide(issued=None, alternatives=(), **kwargs):
    h = (demand('actual_long', DAY-timedelta(days=2), 8.4, 'long', source='observed'),)
    args = dict(as_of=AT, history=h, expected_history=h)
    c = kwargs.pop('coaching_contract', contract())
    args.update(kwargs)
    return decide_session(c, issued or option(), alternatives, **args)


def tired():
    return CurrentContext(DAY, AT, 'little_tired', 'explicit_current_report')


def long_decide(issued=None, alternatives=(), **kwargs):
    """Keep the same numerical fixture, but test a demanding endurance purpose.

    These are synthetic contract limits, not a claim that 4.4 miles is this
    athlete's long run. Easy intensity does not make a long purpose ordinary easy.
    """
    def as_long(o):
        program = replace(o.program,
            exposures=tuple(replace(e, role='long') if e.key == o.target_key else e
                            for e in o.program.exposures),
            demands=tuple(replace(e, purpose='long') if e.key == o.target_key else e
                          for e in o.program.demands))
        return replace(o, program=program)
    c = replace(kwargs.pop('coaching_contract', contract()), purpose='long')
    return decide(as_long(issued or option()), [as_long(o) for o in alternatives],
                  coaching_contract=c, **kwargs)


def test_missing_ratings_and_hrv_do_not_force_rest():
    r = decide()
    assert r['action'] == 'continue_provisionally' and r['context_state'] == 'unknown'
    assert not r['running_authorized'] and not r['physiological_recovery_established']


def test_mild_fatigue_easy_needs_no_invented_reduction_but_long_does():
    c = replace(contract(), reduced=None)
    assert decide(coaching_contract=c)['action'] == 'continue_provisionally'
    r = decide(coaching_contract=c, context=tired(), alternatives=[option('short', 3.5)])
    assert r['action'] == 'continue_provisionally' and r['selected_option_key'] == 'issued'
    r = long_decide(coaching_contract=c, context=tired(), alternatives=[option('short', 3.5)])
    assert r['action'] == 'review_adaptation' and r['selected_option_key'] is None


def test_compiled_point_commands_connect_to_selector_without_reintroducing_ranges():
    from run_analysis.athlete_projection import SessionEvidence
    from run_analysis.coaching_contracts import ReductionPolicy, compile_endurance_contract
    from run_analysis.staged_progression import DevelopmentStage, ProgressionPolicy, MILE
    stage = DevelopmentStage('stage', AT, 17.3, 4.4, 8.4, 17., 8., ProgressionPolicy(14, .05), 'fixture')
    end = AT-timedelta(days=3)
    recorded = SessionEvidence('source', end-timedelta(minutes=44), end, end,
                               'easy', 4.4*MILE, 2640, None, 'fixture')
    c = compile_endurance_contract([recorded], stage, purpose='easy', intensity_key='easy_controlled',
        reduction=ReductionPolicy(.8, 20, 'explicit_test_hypothesis'))['contract']
    assert decide(coaching_contract=c)['action'] == 'continue_provisionally'
    r = decide(coaching_contract=c, context=tired(), alternatives=[option('reduced', 3.5), option('other', 3.6)])
    assert r['selected_option_key'] == 'issued'
    assert r['option_checks']['reduced']['fits']['reduced']
    assert not r['option_checks']['other']['fits']['reduced']


def test_familiar_easy_after_long_uses_contract_not_rest_day_rule():
    h = (demand('long', DAY-timedelta(days=1), 8.4, 'long', source='observed'),)
    assert decide(history=h, expected_history=h)['action'] == 'continue_provisionally'


def test_excess_preceding_work_requires_review_not_automatic_deferral():
    h = (demand('actual_long', DAY-timedelta(days=2), 10., 'long', source='observed'),)
    r = decide(history=h)
    assert r['action'] == 'review_adaptation'
    assert 'greater_preceding_demand_requires_review' in r['option_checks']['issued']['problems']


@pytest.mark.parametrize('state', ['health_concern', 'conflicting'])
def test_current_concern_cannot_be_cancelled_by_tiny_dose_or_low_load(state):
    r = decide(alternatives=[option('small', 3.)], context=CurrentContext(DAY, AT, state, 'current'))
    assert r['action'] == 'review_current_context' and r['selected_option_key'] is None


def test_tired_long_uses_largest_meaningful_reduction_not_cosmetic_cut():
    r = long_decide(alternatives=[option('cosmetic', 4.3), option('small', 3.), option('reduced', 3.5)], context=tired())
    assert r['action'] == 'propose_reduced_trial' and r['selected_option_key'] == 'reduced'
    assert r['program_miles_change'] == pytest.approx(-.9)
    assert r['stage_delivery_recheck_required']


def test_tired_does_not_automatically_assume_tomorrow_recovery():
    r = long_decide(alternatives=[option('tomorrow', offset=1)], context=tired())
    assert r['action'] == 'review_adaptation' and r['selected_option_key'] is None


def test_ordinary_unknown_does_not_trigger_supplied_reduction():
    assert decide(alternatives=[option('reduced', 3.5)])['selected_option_key'] == 'issued'


def test_unexpected_past_response_needs_current_context_not_percentage():
    assert decide(unexpected_previous_response=True)['action'] == 'request_current_context'
    r = decide(unexpected_previous_response=True, context=CurrentContext(DAY, AT, 'usual', 'current'))
    assert r['action'] == 'continue_provisionally'


def test_future_or_other_date_report_is_not_current():
    for context in [replace(tired(), available_at=AT+timedelta(hours=1)), replace(tired(), applies_on=DAY-timedelta(days=1))]:
        assert decide(context=context)['context_state'] == 'unknown'


def test_meaningful_quality_reduction_preserves_purpose():
    issued = option(purpose='threshold', miles=4.5, bouts=(300.,)*3, recovery=(120.,)*2)
    reduced = option('reduced', purpose='threshold', miles=3.8, bouts=(300.,)*2, recovery=(120.,))
    r = decide(issued, [reduced], coaching_contract=contract(quality=True), context=tired())
    assert r['action'] == 'propose_reduced_trial' and r['purpose'] == 'threshold'
    assert r['quality_work_seconds_change'] == -300


def test_novel_format_uses_bouts_recovery_and_purpose_not_donor_count():
    issued = option(purpose='threshold', miles=4.5, bouts=(150.,)*6, recovery=(90.,)*5)
    assert decide(issued, coaching_contract=contract(quality=True))['action'] == 'continue_provisionally'


def test_shortened_recovery_cannot_fake_same_supported_work():
    issued = option(purpose='threshold', miles=4.5, bouts=(150.,)*6, recovery=(15.,)*5)
    with pytest.raises(ValueError, match='outside its own'):
        decide(issued, coaching_contract=contract(quality=True))


def test_easy_substitute_does_not_receive_quality_credit():
    issued = option(purpose='threshold', miles=4.5, bouts=(300.,)*3, recovery=(120.,)*2)
    r = decide(issued, [option('easy', 3.5)], coaching_contract=contract(quality=True), context=tired())
    assert r['selected_option_key'] is None
    assert 'different_purpose_or_intensity' in r['option_checks']['easy']['problems']


def test_reserved_date_can_propose_tentative_deferral():
    n = DateRestriction('rest', DAY, DAY+timedelta(days=1), AT, 'reserved_rest', 'user_choice')
    r = decide(alternatives=[option('tomorrow', offset=1)], restrictions=[n])
    assert r['action'] == 'propose_tentative_deferral'
    assert r['next_date_readiness_recheck_required']


@pytest.mark.parametrize('kind', ['health_review', 'unsafe_weather'])
def test_nonlogistical_restriction_does_not_auto_clear_tomorrow(kind):
    n = DateRestriction('block', DAY, DAY+timedelta(days=1), AT, kind, 'external')
    assert decide(alternatives=[option('tomorrow', offset=1)], restrictions=[n])['action'] == 'review_adaptation'


def test_explicit_trial_is_bounded_and_not_compounded():
    alternatives = [option('trial', 4.8), option('too_much', 5.3)]
    args = dict(alternatives=alternatives, coaching_contract=contract(trial=True), trial_requested=True)
    first = decide(**args)
    assert first['action'] == 'propose_bounded_dose_trial' and first['selected_option_key'] == 'trial'
    assert decide(**args) == first


def test_no_trial_authority_no_growth_bonus():
    r = decide(alternatives=[option('larger', 4.8)], coaching_contract=contract(trial=True))
    assert r['selected_option_key'] == 'issued'
    assert decide(alternatives=[option('larger', 4.8)], trial_requested=True)['action'] == 'review_trial'


def test_dose_and_earlier_date_not_combined():
    issued = option(offset=1)
    candidate = option('earlier_and_larger', 4.8, offset=0)
    r = decide(issued, [candidate], coaching_contract=contract(trial=True), trial_requested=True)
    assert r['action'] == 'review_trial'
    assert 'earlier_date_requires_spacing_trial' in r['option_checks'][candidate.key]['problems']


def test_cannot_pay_for_reduction_by_deleting_downstream_long():
    reduced = option('cheat', 3.5)
    reduced = replace(reduced, program=replace(reduced.program, exposures=reduced.program.exposures[:1], demands=reduced.program.demands[:1]))
    r = long_decide(alternatives=[reduced], context=tired())
    assert 'surrounding_program_changed' in r['option_checks']['cheat']['problems']
    assert r['selected_option_key'] is None


def test_reduction_must_not_compress_quality_recoveries():
    issued = option(purpose='threshold', miles=4.5, bouts=(300.,)*3, recovery=(150.,)*2)
    reduced = option('short_rest', purpose='threshold', miles=3.8, bouts=(300.,)*2, recovery=(120.,))
    r = decide(issued, [reduced], coaching_contract=contract(quality=True), context=tired())
    assert r['selected_option_key'] is None


def test_compliant_history_band_edge_is_not_new_excess():
    actual = demand('actual_long', DAY-timedelta(days=2), 8.6, 'long', source='observed')
    expected = replace(actual, terms=tuple(DemandTerm(c, 'expected_band', scale, Quantity(8.2, 8.6, basis='issued_band'))
        for c, scale in [('miles', 1.), ('running_minutes', 10.), ('cardio', 15.), ('work_minutes', 0.)]))
    assert decide(history=[actual], expected_history=[expected])['action'] == 'continue_provisionally'


def test_reduction_band_cannot_be_a_cosmetic_overlap():
    c = contract()
    with pytest.raises(ValueError, match='meaningfully separate'):
        replace(c, reduced=c.ordinary)


def test_unknown_history_is_review_not_zero_capacity_or_automatic_rest():
    assert decide(history=[], expected_history=[])['action'] == 'review_adaptation'


def test_missing_recovery_details_cannot_be_assumed():
    with pytest.raises(ValueError, match='Every between-work'):
        option(purpose='threshold', miles=4.5, bouts=(300.,)*3)


def test_health_concern_does_not_expire_at_midnight():
    old = CurrentContext(DAY-timedelta(days=1), AT-timedelta(days=1), 'health_concern', 'unresolved')
    assert decide(context=old)['action'] == 'review_current_context'


def test_old_fatigue_requests_context_not_automatic_clearance():
    assert decide(context=replace(tired(), applies_on=DAY-timedelta(days=1)))['action'] == 'request_current_context'


def test_todays_fatigue_does_not_determine_tomorrows_dose():
    assert decide(option(offset=1), context=tired())['action'] == 'request_session_date_context'


def test_trial_reports_added_demand_at_downstream_long():
    r = decide(alternatives=[option('trial', 4.8)], coaching_contract=contract(trial=True), trial_requested=True)
    assert r['downstream_demand_comparisons'][0]['key'] == 'following_long'
    # Left is issued, right is proposed. Additional dose is not free at the tail.
    assert r['downstream_demand_comparisons'][0]['channel_orders']['miles'] == 'left_no_greater'
    assert r['stage_delivery_recheck_required'] and not r['progression_accepted']


def test_reduction_reports_lower_downstream_demand_and_delivery_cost():
    r = long_decide(alternatives=[option('reduced', 3.5)], context=tired())
    assert r['downstream_demand_comparisons'][0]['channel_orders']['miles'] == 'right_no_greater'
    assert r['program_miles_change'] < 0


def test_unknown_hr_does_not_erase_work_or_force_rest():
    o = option()
    demands = tuple(replace(e, terms=tuple(replace(t, bounds=None, variable='unknown_cardio') if t.channel == 'cardio' else t for t in e.terms))
                    for e in o.program.demands)
    o = replace(o, program=replace(o.program, demands=demands))
    assert decide(o)['action'] == 'continue_provisionally'


def test_trial_targets_committed_dose_not_upper_band_edge():
    c = replace(contract(trial=True), trial_target=4.6)
    r = decide(alternatives=[option('target', 4.6), option('upper_edge', 4.8)], coaching_contract=c, trial_requested=True)
    assert r['selected_option_key'] == 'target'


def test_shorter_distance_but_longer_duration_not_automatic_reduction():
    o = option('slower', 3.5)
    e = o.event
    e = replace(e, terms=tuple(replace(t, coefficient=90.) if t.channel == 'running_minutes' else t for t in e.terms))
    o = replace(o, program=replace(o.program, demands=(e, *o.program.demands[1:])))
    assert long_decide(alternatives=[o], context=tired())['selected_option_key'] is None


def test_mild_fatigue_keeps_familiar_easy_without_growing_or_cutting_it():
    r = decide(alternatives=[option('short', 3.5), option('larger', 4.8)],
               context=tired(), coaching_contract=contract(trial=True), trial_requested=True)
    assert r['action'] == 'continue_provisionally'
    assert r['selected_option_key'] == 'issued'
    assert r['program_miles_change'] == 0
    assert not r['stage_delivery_recheck_required']


def test_mild_fatigue_easy_does_not_bypass_restrictions_or_changed_lead_in():
    blocked = DateRestriction('rest', DAY, DAY+timedelta(days=1), AT, 'reserved_rest', 'user')
    assert decide(context=tired(), restrictions=[blocked])['selected_option_key'] is None
    history = (demand('actual_long', DAY-timedelta(days=2), 10., 'long', source='observed'),)
    assert decide(context=tired(), history=history)['selected_option_key'] is None


def test_run_rest_probe_separates_omission_from_future_delivery():
    from scripts.audit_session_decision import run_rest_crossover_probe
    report = run_rest_crossover_probe()
    rows = {(r['memory_days'], r['exchange'], r['preceding_dose_multiple']): r
            for r in report['cases']}
    assert report['options'] == 21 and len(rows) == 81
    assert not report['physiology_validated'] and not report['policy_accepted']
    # These characterize the current experimental score, not desired prescriptions.
    ordinary = rows[2., .01, 1.]
    elevated = rows[2., .01, 2.]
    assert ordinary['no_today_minus_today'] > 0
    assert elevated['no_today_minus_today'] < 0
    assert elevated['omission_minus_today'] > 0
    for r in report['cases']:
        assert r['full_tomorrow_minus_today_delivery'] == pytest.approx(0)
    for memory in (1., 2., 4.):
        for multiple in (0., .5, 1., 2., 4., 8., 16., 32., 64.):
            pair = [rows[memory, exchange, multiple] for exchange in (.005, .01, .02)]
            slopes = [r['full_tomorrow_minus_today']/r['exchange'] for r in pair]
            assert slopes[0] == pytest.approx(slopes[1]) == pytest.approx(slopes[2])


def test_training_logic_panel_uses_shared_score_and_independent_physical_metrics():
    from scripts.audit_session_decision import training_logic_panel
    report = training_logic_panel()
    metrics = report['metrics']
    for key in ('distributed', 'compressed', 'before_quality'):
        assert metrics[key]['miles'] == pytest.approx(100.)
        assert metrics[key]['minutes'] == pytest.approx(1000.)
        assert metrics[key]['runs'] == 20
    assert metrics['underfunded']['miles'] == pytest.approx(64.5)
    assert metrics['compressed']['peak_3day_minutes'] > metrics['distributed']['peak_3day_minutes']
    assert metrics['before_quality']['max_quality_prior_day_minutes'] == 35.
    assert metrics['distributed']['max_quality_prior_day_minutes'] == 0.
    assert len(report['cases']) == 9 and len(report['boundary_cases']) == 3
    assert report['predeclared_checks_pass']
    assert not report['policy_accepted'] and not report['physiology_validated']
    for case in report['cases']:
        scores = case['scores']
        assert scores['distributed']['delivery_cost'] == pytest.approx(scores['compressed']['delivery_cost'])
        assert scores['distributed']['delivery_cost'] == pytest.approx(scores['before_quality']['delivery_cost'])
        assert all(not row['selection_ready'] for row in scores.values())
    # Detect/report boundary effects; do not make the present bug a permanent
    # requirement or assert that either diagnostic calendar is suitable to train.
    for case in report['boundary_cases']:
        assert case['rank_reversal'] == (case['short_winner'] != case['extended_winner'])
        for scope in ('short', 'extended'):
            assert case[scope]['earlier']['delivery_cost'] == pytest.approx(case[scope]['edge']['delivery_cost'])
        for name in ('earlier', 'edge'):
            short, added = case['short'][name], case['extended'][name]
            assert short['targets'] == added['targets']
            assert short['delivery'] == added['delivery']
            assert short['delivery_cost'] == added['delivery_cost']
            assert short['cost'] == case['scope_only'][name]['cost']
            assert added['cost'] == case['more_coverage'][name]['cost']
            assert case['legacy_extended'][name]['targets'] != short['targets']


def test_marginal_panel_explains_actions_without_asserting_recovery_clearance():
    from scripts.audit_session_decision import training_logic_panel
    report = training_logic_panel()
    assert len(report['marginal_cases']) == 45
    options = report['marginal_options']
    for name in ('continue_today', 'defer_one_day', 'redistribute'):
        assert options[name]['miles'] == pytest.approx(100)
        assert options[name]['runs'] == 20
    assert options['reduce_today']['miles'] == pytest.approx(99)
    assert options['omit_today']['miles'] == pytest.approx(96.5)
    for row in report['marginal_cases']:
        for contrast in row['contrasts'].values():
            assert contrast['score_change'] == pytest.approx(
                contrast['delivery_loss_change']-row['exchange']*contrast['demand_saved'])
            assert sum(contrast['shortfall_change_by_purpose'].values()) == pytest.approx(
                contrast['delivery_loss_change'])
        shifted = row['contrasts']['defer_one_day']
        assert shifted['delivery_loss_change'] == pytest.approx(0)
        redistributed = row['contrasts']['redistribute']['shortfall_change_by_purpose']
        # Attribution, not a requirement to preserve the easy-credit penalty.
        assert redistributed['volume'] == pytest.approx(0)
        assert redistributed['long'] == pytest.approx(0)
        assert redistributed['quality'] == pytest.approx(0)
    assert not report['policy_accepted'] and not report['physiology_validated']


def test_small_prior_dose_increases_are_proportional_in_whole_program_score():
    from scripts.audit_session_decision import training_logic_panel
    report = training_logic_panel()
    rows = {(r['preceding_miles'], r['memory_days'], r['exchange']): r
            for r in report['marginal_cases'] if not r['stacked']}
    for memory in (1., 2., 4.):
        for exchange in (.005, .01, .02):
            for option in ('reduce_today', 'omit_today', 'defer_one_day', 'redistribute'):
                costs = [rows[miles, memory, exchange]['contrasts'][option]['score_change']
                         for miles in (8.4, 8.5, 8.6)]
                assert costs[1]-costs[0] == pytest.approx(costs[2]-costs[1], abs=1e-12)
    # Continuity of score does not guarantee continuity of an argmin near a tie.


def test_marginal_panel_is_invariant_to_joint_demand_and_reference_rescaling():
    from scripts.audit_session_decision import training_logic_panel
    original = training_logic_panel()
    scaled = training_logic_panel(role_rates={role: tuple(v*2 for v in values)
        for role, values in original['scenario_rates'].items()})
    for left, right in zip(original['marginal_cases'], scaled['marginal_cases']):
        for key in left['contrasts']:
            assert left['contrasts'][key]['score_change'] == pytest.approx(
                right['contrasts'][key]['score_change'])
    # Unit invariance is not proof that runners with different paces recover alike.
    with pytest.raises(ValueError):
        training_logic_panel(role_rates={})


def test_near_minimum_curve_separates_useful_short_runs_from_fragmentation():
    from scripts.audit_session_decision import training_logic_panel
    from scripts.program_objective_trial import EasyDurationPolicy
    report = training_logic_panel(easy_duration_policy=EasyDurationPolicy(10, 20, .01, 'test'))
    for case in report['cases']:
        scores = case['scores']
        assert scores['distributed']['easy_duration']['cost'] == 0
        assert scores['near_floor_fragments']['easy_duration']['cost'] > 0
        assert len(scores['below_floor']['easy_duration']['below_minimum_keys']) == 5
        assert scores['above_band_split']['easy_duration']['cost'] == 0
        for name in ('near_floor_fragments', 'above_band_split', 'concentrated_easy', 'below_floor'):
            assert report['metrics'][name]['miles'] == pytest.approx(100)
            assert scores[name]['shortfall_by_purpose']['ordinary_easy'] == 0
    for row in report['marginal_cases']:
        for contrast in row['contrasts'].values():
            assert contrast['score_change'] == pytest.approx(contrast['delivery_loss_change']
                + contrast['short_run_cost_change']-row['exchange']*contrast['demand_saved'])
        assert row['contrasts']['redistribute']['short_run_cost_change'] == 0
        assert row['contrasts']['redistribute']['delivery_loss_change'] == 0
    # No assertion that every penalty or fatigue setting produces good training.
