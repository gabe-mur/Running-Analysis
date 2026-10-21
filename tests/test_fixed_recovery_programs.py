from dataclasses import replace
import pytest

from scripts.audit_fixed_recovery_programs import PROGRAMS, TAIL, TAILS, TIMINGS, events, metrics, demand_trace, score_trace, history_before_plan
from scripts.audit_recovery_baseline import NOW
from run_analysis.recovery_assessment import AssessmentPolicy, assess_next_session

P = AssessmentPolicy('fixture', 130, 280, 24, 24, 24, 24)


def test_programs_match_counts_and_aggregate_dose():
    values = [metrics(program) for program in PROGRAMS.values()]
    assert all(len(program) == 21 for program in PROGRAMS.values())
    assert len(TAIL) == 14
    assert all(value['counts'] == {'L': 3, 'Q': 3, 'E': 6} for value in values)
    assert all(value['totals'] == values[0]['totals'] for value in values)
    assert values[0]['totals']['running_minutes'] == 585


@pytest.mark.parametrize('timing', list(TIMINGS))
def test_execution_paths_are_ordered_and_include_tail(timing):
    for program in PROGRAMS.values():
        rows = events(program, timing)
        assert any(day >= 21 for day, _, _ in rows)
        assert all(a.end <= b.start for (_, _, a), (_, _, b) in zip(rows, rows[1:]))


def test_cached_demand_scoring_matches_direct_assessment_for_other_budgets():
    program = PROGRAMS['compressed_blocks']
    totals = demand_trace(program, 'alternating_early', P)
    different = replace(P, running_minutes_envelope=110, cardio_points_envelope=230, work_minutes_envelope=18)
    past = list(history_before_plan())
    direct = []
    for day, _, event in events(program, 'alternating_early'):
        result = assess_next_session(past, event.workload, event.start, different,
                                     proposed_work_minutes=event.work_minutes)
        direct.append((day, result['excess_high']))
        past.append(event)
    cached = score_trace(totals, different)
    assert cached['combined_excess'] == pytest.approx(sum(value for _, value in direct))
    assert cached['plan_excess'] == pytest.approx(sum(value for day, value in direct if day < 21))
    assert cached['tail_excess'] == pytest.approx(sum(value for day, value in direct if day >= 21))


def test_extra_work_is_not_hidden_behind_horizon():
    # At a 36-hour work half-life, five days leaves >1 minute of residual work.
    tight = replace(P, work_minutes_envelope=16, work_half_life_hours=36)
    trace = demand_trace(PROGRAMS['spaced'], 'nominal', tight)
    result = score_trace(trace, tight)
    assert result['tail_excess'] > 0
    assert result['combined_excess'] > result['plan_excess']


def test_alternative_tails_preserve_dose_and_do_not_change_prefix_scores():
    from collections import Counter
    assert all(Counter(tail) == Counter(TAIL) for tail in TAILS.values())
    scores = [score_trace(demand_trace(PROGRAMS['compressed_blocks'], 'nominal', P, tail), P)
              for tail in TAILS.values()]
    assert len({row['plan_excess'] for row in scores}) == 1
