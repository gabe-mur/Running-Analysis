from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from run_analysis.athlete_projection import SessionEvidence
from run_analysis.delivered_training import DoseContract, DoseTarget, DeliveredDose
from run_analysis.staged_progression import (
    MILE, DevelopmentStage, ProgressionPolicy, ReviewEntry, ResponseReport,
    bootstrap_stage, review_stage, tenth,
)
from run_analysis.workload_accounting import Quantity

AT = datetime(2026, 9, 25, tzinfo=timezone.utc)
POLICY = ProgressionPolicy(14, .05)


def stage(**changes):
    base = DevelopmentStage('s1', AT, 17.3, 4.4, 8.4, 17., 8., POLICY, 'fixture')
    return replace(base, **changes)


def entry(s=None, miles=8.4, key='long'):
    s = s or stage()
    start = s.issued_at+timedelta(days=2)
    contract = DoseContract(s.issued_at, 'long_endurance', (
        DoseTarget('distance_meters', Quantity((s.long_miles-.2)*MILE, (s.long_miles+.2)*MILE,
            s.long_miles*MILE, 'issued_fixture')),), 'test_prescription')
    done = DeliveredDose(key, start, start+timedelta(hours=2), start+timedelta(hours=2),
        Quantity.point(miles*MILE, 'actual'), Quantity.point(7200, 'actual'), (), 'unknown')
    return ReviewEntry(key, start, contract, done)


def review(s=None, entries=None, **kwargs):
    s = s or stage()
    options = dict(ledger_complete=True, longest_prior_30d_miles=8.45)
    options.update(kwargs)
    return review_stage(s, [entry(s)] if entries is None else entries, s.review_at, **options)


def history():
    result = []
    for i, (days, miles, role) in enumerate([(26, 4.4, 'easy'), (24, 7.5, 'long'),
        (20, 4.4, 'easy'), (16, 7.7, 'long'), (12, 4.4, 'easy'), (8, 8., 'long'),
        (4, 4.4, 'easy'), (1, 8.45, 'long')]):
        end = AT-timedelta(days=days)
        result.append(SessionEvidence(str(i), end-timedelta(hours=2), end, end, role,
            miles*MILE, 3600, None, 'no_hr_needed'))
    return result


def test_bootstrap_is_provisional_and_does_not_treat_rolling_mean_as_latest_long():
    result = bootstrap_stage(history(), AT, goal_long_miles=17., policy=POLICY)
    assert result['status'] == 'provisional_seed'
    assert result['stage'].ordinary_miles == 4.4
    assert result['stage'].long_miles == 8.5
    assert result['ordinary_reference_provisional']
    assert not result['planner_influence']


def test_exceptional_latest_long_is_not_promoted_into_capacity():
    rows = history()
    rows[-1] = replace(rows[-1], meters=14*MILE)
    assert bootstrap_stage(rows, AT, goal_long_miles=17., policy=POLICY)['status'] == 'latest_long_spike_requires_review'


def test_missing_or_recent_excluded_history_needs_starting_dose_review():
    assert bootstrap_stage([], AT, goal_long_miles=17., policy=POLICY)['stage'] is None
    rows = history()
    rows[0] = replace(rows[0], capacity_eligible=False)
    assert bootstrap_stage(rows, AT, goal_long_miles=17., policy=POLICY)['stage'] is None


def test_future_data_cannot_change_bootstrap():
    rows = history()
    future = replace(rows[-1], key='future', start=AT+timedelta(days=1),
        end=AT+timedelta(days=1, hours=1), available_at=AT+timedelta(days=1, hours=1))
    a = bootstrap_stage(rows, AT, goal_long_miles=17., policy=POLICY)
    b = bootstrap_stage(rows+[future], AT, goal_long_miles=17., policy=POLICY)
    assert a['stage'] == b['stage']


def test_refresh_or_window_expiry_does_not_rebase_committed_stage():
    s = stage()
    for days in [1, 7, 13]:
        r = review_stage(s, [], AT+timedelta(days=days), ledger_complete=False, longest_prior_30d_miles=None)
        assert r['status'] == 'continue_existing_stage'
        assert r['stage_target_unchanged'] == s
    r = review(s, entries=[], ledger_complete=False)
    assert r['stage_target_unchanged'] == s


@pytest.mark.parametrize('miles', [8.2, 8.4, 8.6])
def test_compliant_range_edges_generate_same_conditional_targets(miles):
    result = review(entries=[entry(miles=miles)])
    assert result['status'] == 'conditional_development_candidate'
    assert result['candidate'].long_miles == 8.8
    assert result['candidate'].weekly_equivalent_miles == 18.2
    assert result['candidate'].ordinary_miles == 4.4
    assert result['extra_nonlong_miles_per_7_days'] == pytest.approx(.55)
    assert not result['quality_progression']
    assert not result['running_authorized']


def test_repeated_review_does_not_compound_growth():
    first = review()
    for _ in range(10):
        assert review() == first


@pytest.mark.parametrize('miles', [7., 10.])
def test_imperfect_execution_retains_targets_for_explanation_not_auto_shrink(miles):
    r = review(entries=[entry(miles=miles)])
    assert r['status'] == 'execution_review_hold_targets'
    assert r['stage_target_unchanged'] == stage()
    assert r['delivery'][0]['observed_metrics']['distance_meters'].low == miles*MILE


def test_missing_completion_and_late_upload_are_not_rest():
    a = entry()
    assert review(entries=[replace(a, completion=None)])['delivery_problems'] == [('long', 'uncompleted_or_unuploaded')]
    late = replace(a.completion, available_at=stage().review_at+timedelta(days=1))
    assert review(entries=[replace(a, completion=late)])['delivery_problems'] == [('long', 'completion_not_available')]


def test_missing_feedback_does_not_block_candidate_or_claim_recovery():
    r = review()
    assert r['response_state'] == 'unknown'
    assert r['candidate'] is not None
    assert r['requires_program_feasibility_and_recovery']


def test_adverse_feedback_interrupts_before_review_and_does_not_expire():
    s = stage()
    reports = [ResponseReport(AT, 'adverse', 'explicit_concern'),
               ResponseReport(AT+timedelta(days=1), 'unknown', 'missing_rating')]
    r = review_stage(s, [], AT+timedelta(days=2), ledger_complete=False,
        longest_prior_30d_miles=None, responses=reports)
    assert r['status'] == 'dose_specific_review_required'
    assert review(responses=reports)['status'] == r['status']


def test_rollback_uses_previous_stage_not_an_invented_reduction_percent():
    old = stage(key='old', issued_at=AT-timedelta(days=14), weekly_equivalent_miles=16.5, long_miles=8.)
    r = review(responses=[ResponseReport(AT, 'adverse', 'explicit_report')], previous_stage=old)
    assert r['status'] == 'rollback_candidate'
    assert r['candidate'] == old


def test_health_concern_cannot_be_cancelled_by_numeric_success_or_future_clearance():
    reports = [ResponseReport(AT, 'health_concern', 'athlete_report'),
               ResponseReport(AT+timedelta(days=20), 'clear', 'later_resolution')]
    assert review(responses=reports)['status'] == 'health_review_required'


def test_later_fatigue_report_does_not_clear_health_concern():
    reports = [ResponseReport(AT, 'health_concern', 'pain'),
               ResponseReport(AT+timedelta(days=1), 'adverse', 'fatigue')]
    assert review(responses=reports)['status'] == 'health_review_required'
    reports.append(ResponseReport(AT+timedelta(days=2), 'clear', 'explicit_resolution'))
    assert review(responses=reports)['status'] == 'conditional_development_candidate'


def test_goal_caps_long_growth_and_stops_compounding_on_later_review():
    s = stage(long_miles=16.9, weekly_equivalent_miles=35.)
    r = review(s, [entry(s, 16.9)], longest_prior_30d_miles=16.9)
    assert r['candidate'].long_miles == 17.
    next_stage = r['candidate']
    assert review(next_stage, [], longest_prior_30d_miles=17.)['candidate'] is None


def test_guard_rounding_never_crosses_ceiling():
    r = review(longest_prior_30d_miles=7.91)
    assert r['candidate'].long_miles == 8.7
    assert r['candidate'].long_miles <= 7.91*1.1
    assert review(longest_prior_30d_miles=7.6)['status'] == 'long_growth_resolution_or_guard_limit'


def test_supporting_volume_cannot_shrink_to_pay_for_long_growth():
    assert review(stage(weekly_equivalent_miles=4., observed_long_gap_days=4.))['status'] == 'supporting_volume_conflict'


def test_duplicate_completed_session_cannot_satisfy_two_opportunities():
    a = entry()
    with pytest.raises(ValueError, match='counted twice'):
        review(entries=[a, replace(a, key='second')])


def test_arithmetic_rounding_is_decimal_half_up_and_floor_for_guard():
    assert tenth(8.45) == 8.5
    assert tenth(8.499, down=True) == 8.4


def easy_build(**changes):
    return stage(weekly_equivalent_miles=18.6,
        policy=ProgressionPolicy(14,.05,'easy_volume',19.5),**changes)


@pytest.mark.parametrize('miles',[8.2,8.4,8.6])
def test_easy_volume_phase_holds_long_and_assigns_all_growth_to_nonlong(miles):
    s=easy_build()
    result=review(s,[entry(s,miles)])
    assert result['status']=='conditional_development_candidate'
    assert result['candidate'].weekly_equivalent_miles==19.5
    assert result['candidate'].long_miles==8.4
    assert result['candidate'].ordinary_miles==4.4
    assert result['candidate'].goal_long_miles==17.
    assert result['extra_nonlong_miles_per_7_days']==pytest.approx(.9)
    assert not result['long_progression'] and not result['quality_progression']
    assert result['progression_focus']=='easy_volume'


def test_volume_build_is_bounded_even_while_long_goal_is_paused():
    s=easy_build()
    next_stage=review(s,[entry(s)])['candidate']
    result=review(next_stage,[entry(next_stage)])
    assert result['status']=='volume_phase_complete_reassess_next_stage'
    assert result['candidate'] is None
    assert result['stage_target_unchanged'].weekly_equivalent_miles==19.5
    assert result['stage_target_unchanged'].long_miles==8.4


def test_long_goal_and_unknown_cadence_do_not_prevent_explicit_volume_phase():
    s=easy_build(goal_long_miles=8.4,observed_long_gap_days=None)
    result=review(s,[entry(s)])
    assert result['candidate'].weekly_equivalent_miles==19.5
    assert result['candidate'].long_miles==8.4
    assert result['extra_nonlong_miles_per_7_days']==pytest.approx(.9)


def test_holding_long_does_not_bypass_reference_or_health_or_delivery_checks():
    s=easy_build()
    assert review(s,[entry(s)],longest_prior_30d_miles=None)['candidate'] is None
    assert review(s,[entry(s)],longest_prior_30d_miles=7.)['status']=='held_long_requires_session_guard_review'
    assert review(s,[entry(s,7.)])['candidate'] is None
    assert review(s,[],ledger_complete=False)['candidate'] is None
    assert review(s,[entry(s)],responses=[ResponseReport(AT,'health_concern','test')])['candidate'] is None


def test_volume_phase_requires_explicit_goal_and_old_policy_is_backward_compatible():
    assert ProgressionPolicy(**dict(review_days=14,growth_fraction=.05)).growth_focus=='long_and_volume'
    with pytest.raises(ValueError,match='volume goal'):
        ProgressionPolicy(14,.05,'easy_volume')
    with pytest.raises(ValueError):
        ProgressionPolicy(14,.05,'easy_volume',19.55)
    with pytest.raises(ValueError):
        ProgressionPolicy(14,.05,'unknown')


def test_thin_ledger_never_claims_whole_program_feasibility():
    # Even a numerically successful isolated long does not validate frequency,
    # total funding or a stable calendar. No automatically accepted progression.
    r = review()
    assert r['requires_program_feasibility_and_recovery']
    assert 'Numerical prescription completion does not prove whole-program funding' in r['limitations']
