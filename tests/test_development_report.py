from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from scripts.audit_program_delivery import ORIGIN, ledger, regular
from scripts.audit_recency_demand import future_events
from run_analysis.development_report import ProgramOption, OpportunityPair, development_report
from run_analysis.delivered_training import DoseContract, DoseTarget, DeliveredDose
from run_analysis.program_delivery import TargetSegment
from run_analysis.staged_progression import DevelopmentStage, ProgressionPolicy, ResponseReport, ReviewEntry, MILE
from run_analysis.workload_accounting import Quantity
from run_analysis.program_constraints import DateRestriction

AT = datetime(2026, 9, 25, 12, tzinfo=timezone.utc)
STAGE = DevelopmentStage('stage', AT-timedelta(days=14), 17.3, 4.4, 8.4, 17., 7.,
                         ProgressionPolicy(14, .05), 'fixture_not_accepted_policy')


def option(key, motif=None):
    motif = regular() if motif is None else motif
    rates = {r: {c: {'bounds': Quantity(1., 20., basis='shared_fixture_rate')}
                 for c in ('running_minutes', 'cardio')} for r in ('easy', 'long', 'quality')}
    return ProgramOption(key, tuple(e for e in ledger(motif) if e.source == 'planned'),
                         tuple(future_events(motif, rates, 'same_units', horizon_days=112)))


def completion():
    start = STAGE.issued_at+timedelta(days=2)
    end = start+timedelta(hours=2)
    contract = DoseContract(STAGE.issued_at, 'long_endurance', (
        DoseTarget('distance_meters', Quantity(8.2*MILE, 8.6*MILE, basis='issued_fixture')),), 'fixture')
    done = DeliveredDose('actual_long', start, end, end, Quantity.point(8.4*MILE, 'recorded_fixture'),
                         Quantity.point(7200, 'recorded_fixture'), (), 'unknown')
    return ReviewEntry('issued_long', start, contract, done)


def report(candidate=None, **kwargs):
    a, b = option('reference'), candidate or option('candidate', regular(easy=5.3))
    args = dict(as_of=AT,
        reference_targets=[TargetSegment(ORIGIN, ORIGIN+timedelta(days=112), 17.3)],
        candidate_targets=[TargetSegment(ORIGIN, ORIGIN+timedelta(days=112), 18.2)],
        coverage_start=ORIGIN, coverage_end=ORIGIN+timedelta(days=112),
        evaluation_starts=[ORIGIN], spans=(28,), coverage_complete=True,
        entries=[completion()], ledger_complete=True, longest_prior_30d_miles=8.45,
        opportunity_pairs=[OpportunityPair(a.demands[2].key, b.demands[2].key)])
    args.update(kwargs)
    return development_report(STAGE, a, b, **args)


def test_funded_compliant_candidate_still_has_unresolved_feasibility():
    r = report()
    assert r['execution_and_stage_review']['status'] == 'conditional_development_candidate'
    assert r['program_delivery']['screen']['review_reasons'] == []
    assert r['feasibility']['status'] == 'unresolved'
    assert not r['progression_accepted'] and not r['running_authorized'] and not r['planner_influence']
    assert r['selected_candidate'] is None
    assert r['goal']['candidate']['training_benefit_established'] is False


def test_perfect_delivery_cannot_hide_sparse_program():
    r = report(option('sparse', regular()[:2]))
    assert r['execution_and_stage_review']['status'] == 'conditional_development_candidate'
    assert 'underfunded_across_checked_intervals' in r['feasibility']['known_review_reasons']
    assert not r['progression_accepted']


@pytest.mark.parametrize('state', ['clear', 'unknown', 'adverse', 'health_concern'])
def test_response_states_do_not_become_feasibility_clearance(state):
    r = report(responses=[ResponseReport(AT, state, 'test_report')])
    assert r['response']['state'] == state
    assert r['response']['concern_present'] == (state in {'adverse', 'health_concern'})
    assert r['feasibility']['status'] == 'unresolved'
    assert not r['progression_accepted']


def test_future_response_cannot_clear_known_concern():
    r = report(responses=[ResponseReport(AT, 'health_concern', 'current'),
                          ResponseReport(AT+timedelta(days=1), 'clear', 'not_yet_available')])
    assert r['response']['state'] == 'health_concern'


def test_missing_opportunity_comparisons_stay_visible():
    r = report(opportunity_pairs=[])
    assert len(r['demand']['unmatched_candidate_keys']) == 12
    assert 'uncompared_candidate_opportunities' in r['feasibility']['unmet_requirements']


def test_unknown_history_coverage_not_silently_complete():
    r = report(coverage_complete=False)
    assert 'unverified_ledger_coverage' in r['feasibility']['known_review_reasons']


def test_quality_dose_change_is_visible_even_when_funded():
    r = report(option('quality_growth', regular(quality=5.4, work=(300.,)*4)))
    assert 'quality_dose_or_purpose_changed' in r['feasibility']['known_review_reasons']


def test_no_report_path_selects_lowest_demand():
    r = report(option('hold'), candidate_targets=[TargetSegment(ORIGIN, ORIGIN+timedelta(days=112), 17.3)])
    assert r['demand']['comparisons'][0]['recency_demand_order'] == 'equal'
    assert r['selected_candidate'] is None


def test_larger_long_is_not_reported_as_verified_goal_progress():
    r = report(option('long_growth', regular(long=8.8, easy=4.9)))
    g = r['goal']['candidate']
    assert g['long_prescriptions'][0]['gap_to_destination_miles'] == pytest.approx(8.2)
    assert not g['training_benefit_established']


def test_funding_demand_distance_mismatch_rejected():
    o = option('bad')
    terms = tuple(replace(t, coefficient=t.coefficient+1) if t.channel == 'miles' else t for t in o.demands[0].terms)
    with pytest.raises(ValueError, match='same distance'):
        replace(o, demands=(replace(o.demands[0], terms=terms), *o.demands[1:]))


def test_work_demand_mismatch_rejected():
    o = option('bad')
    terms = tuple(replace(t, coefficient=0) if t.channel == 'work_minutes' else t for t in o.demands[2].terms)
    with pytest.raises(ValueError, match='prescribed bouts'):
        replace(o, demands=(*o.demands[:2], replace(o.demands[2], terms=terms), *o.demands[3:]))


def test_changed_demand_date_rejected():
    o = option('bad')
    with pytest.raises(ValueError, match='Same planned date'):
        replace(o, demands=(replace(o.demands[0], day=ORIGIN+timedelta(days=1)), *o.demands[1:]))


def test_absent_and_duplicate_matching_rejected():
    with pytest.raises(ValueError, match='absent'):
        report(opportunity_pairs=[OpportunityPair('not_here', 'not_here')])
    p = OpportunityPair('planned:0:0', 'planned:0:0')
    with pytest.raises(ValueError, match='unique'):
        report(opportunity_pairs=[p, p])


def test_all_accounting_units_checked_even_without_pairs():
    o = option('bad')
    o = replace(o, demands=(replace(o.demands[0], accounting_key='edwards'), *o.demands[1:]))
    with pytest.raises(ValueError, match='accounting units'):
        report(o, opportunity_pairs=[])


def test_observed_exposure_cannot_be_counted_again_as_projection():
    a = option('ref').demands[0]
    h = replace(a, source='observed', day=ORIGIN-timedelta(days=2))
    with pytest.raises(ValueError, match='also appear'):
        report(history=[h])


def test_same_day_history_is_not_discarded():
    h = replace(option('ref').demands[0], source='observed', key='actual')
    with pytest.raises(ValueError, match='same-day'):
        report(history=[h])


def test_same_day_additional_planned_session_is_not_discarded():
    motif = regular()+[(4, .9, 'easy', .9, None, None)]
    o = option('two_runs', motif)
    key = next(e.key for e in o.demands if e.purpose == 'threshold')
    with pytest.raises(ValueError, match='Same-day multiple'):
        report(o, opportunity_pairs=[OpportunityPair('planned:4:2', key)])


def test_demand_identity_missing_or_duplicated_rejected():
    o = option('bad')
    with pytest.raises(ValueError, match='one-to-one'):
        replace(o, demands=o.demands[1:])
    with pytest.raises(ValueError, match='one-to-one'):
        replace(o, exposures=(*o.exposures, o.exposures[0]))


def test_unknown_observations_do_not_imply_no_concern():
    r = report(responses=[])
    assert r['response']['state'] == 'unknown'
    assert not r['response']['unknown_is_clearance']
    assert 'response_unknown_not_clearance' in r['feasibility']['unmet_requirements']


def test_repeated_reports_do_not_compound_or_mutate_stage():
    first = report()
    assert report() == first
    assert STAGE.long_miles == 8.4 and STAGE.weekly_equivalent_miles == 17.3


def test_feasibility_override_is_not_an_api_option():
    with pytest.raises(TypeError, match='feasibility'):
        report(feasibility=True)


def test_joined_report_keeps_protected_date_conflict():
    n = DateRestriction('rest', ORIGIN, ORIGIN+timedelta(days=1), AT, 'reserved_rest', 'test_user_choice')
    r = report(restrictions=[n])
    assert r['feasibility']['constraint_checks']['status'] == 'constraint_conflict'
    assert 'explicit_program_constraint_conflict' in r['feasibility']['known_review_reasons']


def test_guard_pass_does_not_resolve_recovery_feasibility():
    h = replace(option('reference').demands[0], key='actual_long', day=ORIGIN-timedelta(days=1), source='observed')
    r = report(history=[h])
    assert r['feasibility']['constraint_checks']['status'] == 'passes_checked_constraints_conditionally'
    assert 'candidate_constraints_require_resolution' not in r['feasibility']['unmet_requirements']
    assert r['feasibility']['status'] == 'unresolved'
    assert not r['progression_accepted']
