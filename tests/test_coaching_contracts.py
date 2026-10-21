from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from run_analysis.athlete_projection import SessionEvidence
from run_analysis.coaching_contracts import ReductionPolicy, compile_endurance_contract
from run_analysis.demonstrated_training import SessionIntent
from run_analysis.staged_progression import DevelopmentStage, ProgressionPolicy, bootstrap_stage, MILE

AT = datetime(2026, 9, 25, 12, tzinfo=timezone.utc)
STAGE = DevelopmentStage('stage', AT, 17.3, 4.4, 8.4, 17., 8., ProgressionPolicy(14, .05), 'fixture')


def run(key='easy', miles=4.4, minutes=44., days=2, role='easy'):
    end = AT-timedelta(days=days)
    return SessionEvidence(key, end-timedelta(minutes=minutes), end, end, role,
                           miles*MILE, minutes*60, None, 'fixture')


def compile(rows=None, **kw):
    args = dict(purpose='easy', intensity_key='controlled_easy')
    args.update(kw)
    return compile_endurance_contract([run()] if rows is None else rows, STAGE, **args)


def test_stage_is_authoritative_and_duration_remains_paired():
    r = compile([run(miles=4.45, minutes=50), run('other', miles=6, minutes=35, days=4)])
    assert r['contract'].ordinary.miles.estimate == 4.4
    assert r['source_key'] == 'easy'
    assert r['ordinary_minutes_at_reference_pace'] == pytest.approx(50*4.4/4.45)
    assert r['contract'].ordinary.running_minutes is None
    assert r['ordinary_intent_provisional'] and not r['original_program_justified']


def test_only_one_source_needed_but_zero_does_not_invent_capacity():
    assert compile()['reference_count'] == 1
    assert compile([])['status'] == 'missing_paired_reference'


def test_explicit_support_cannot_supply_ordinary_reference():
    support = SessionIntent('easy', 'support', AT, 'issued_support')
    assert compile(intents=[support])['contract'] is None
    r = compile([run(), run('ordinary', miles=5, minutes=55, days=4)], intents=[support])
    assert r['source_key'] == 'ordinary'


def test_known_ordinary_precedes_nearer_unknown_intent():
    intent = SessionIntent('ordinary', 'ordinary', AT, 'issued_ordinary')
    r = compile([run(), run('ordinary', miles=5, minutes=55, days=4)], intents=[intent])
    assert r['source_key'] == 'ordinary' and not r['ordinary_intent_provisional']


def test_later_upload_and_later_intent_cannot_rewrite_issue_time():
    extra = replace(run('late', days=1), available_at=AT+timedelta(days=1))
    intent = SessionIntent('easy', 'support', AT+timedelta(days=1), 'late_intent')
    a = compile()
    b = compile([run(), extra], intents=[intent])
    assert a == b


def test_later_completed_run_cannot_rewrite_contract():
    assert compile([run(), run('future', days=-1)]) == compile()


def test_recent_exclusion_cannot_be_hidden_by_older_healthy_donor():
    assert compile([run(), replace(run('concern', days=1), capacity_eligible=False)])['status'] == 'recent_exclusions_require_review'


def test_missing_duration_is_not_a_zero_pace_anchor():
    r = replace(run(), moving_seconds=0)
    assert compile([r])['status'] == 'missing_paired_reference'


def test_latest_long_not_lifetime_max_supplies_duration():
    r = compile([run('older', 9, 90, 12, 'long'), run('latest', 8.4, 100, 3, 'long')], purpose='long')
    assert r['source_key'] == 'latest' and r['contract'].ordinary.miles.estimate == 8.4
    assert r['contract'].reduced is None


def test_rounding_committed_once_and_actual_reduction_reported():
    r = compile(reduction=ReductionPolicy(.8, 20, 'hypothesis'))
    assert r['contract'].reduced.miles.estimate == 3.5
    assert r['reduced_minutes_at_reference_pace'] == pytest.approx(35)
    assert r['realized_reduction_fraction'] == pytest.approx(3.5/4.4)
    assert r['contract'].trial is None


@pytest.mark.parametrize('minutes,available', [(22, False), (44, True), (88, True)])
def test_same_mileage_different_duration_does_not_imply_same_meaningful_reduction(minutes, available):
    r = compile([run(minutes=minutes)], reduction=ReductionPolicy(.8, 20, 'hypothetical_floor'))
    assert (r['contract'].reduced is not None) == available


def test_rounding_cannot_create_cosmetic_or_zero_reduction():
    for fraction in (.999, .001):
        r = compile(reduction=ReductionPolicy(fraction, .01, 'test'))
        assert r['contract'].reduced is None
        assert r['reduction_status'] == 'no_meaningful_reduction_after_rounding'


@pytest.mark.parametrize('fraction,floor,source', [(1,20,'x'), (0,20,'x'), (.8,0,'x'), (.8,20,''), (float('nan'),20,'x')])
def test_invalid_reduction_policy_rejected(fraction, floor, source):
    with pytest.raises(ValueError):
        ReductionPolicy(fraction, floor, source)


def test_generic_intervals_cannot_be_promoted_to_adaptation_purpose():
    with pytest.raises(ValueError, match='Quality needs'):
        compile(purpose='high_aerobic')


def test_order_determinism():
    rows = [run(), run('other', miles=5, days=4)]
    assert compile(rows) == compile(reversed(rows))


def test_support_excluded_from_seed_anchor_but_not_total_exposure():
    rows = [run(), run('long', 8.4, 90, 3, 'long'), run('prior_long', 8., 90, 20, 'long')]
    supports = [run(f's{i}', 1, 10, i+5) for i in range(6)]
    intents = [SessionIntent(s.key, 'support', AT, 'support_command') for s in supports]
    args = dict(goal_long_miles=17., policy=STAGE.policy)
    before = bootstrap_stage(rows, AT, **args)
    after = bootstrap_stage(rows+supports, AT, intents=intents, **args)
    assert before['stage'].ordinary_miles == after['stage'].ordinary_miles == 4.4
    assert after['stage'].weekly_equivalent_miles > before['stage'].weekly_equivalent_miles
    only_support = bootstrap_stage(rows[1:]+supports, AT, intents=intents, **args)
    assert only_support['stage'] is None
