from dataclasses import replace
from datetime import timedelta

import pytest

from run_analysis.adaptive.effort_pace_reference import EffortEvidence, ExecutionPace, effort_reference
from run_analysis.adaptive.quality_prescription_contract import Effort, Measure
from run_analysis.adaptive.quality_block_contract import BlockCommand, BlockRequest, compile_blocks
from run_analysis.athlete_projection import PaceSample
from run_analysis.workload_accounting import Quantity
from test_quality_dose_policy import AT, pace
from test_requested_quality_forecast import CFG


def effort(*cues, purpose='purposeful'):
    return Effort(cues, cues, purpose)


def evidence(key='past', *, samples=(), executions=(), **kwargs):
    values = dict(key=key, start=AT-timedelta(days=1), end=AT-timedelta(hours=23),
        available_at=AT-timedelta(hours=23), accounting_key='units', steady_samples=samples,
        executions=executions, command_issued_at=AT-timedelta(days=2) if executions else None)
    return EffortEvidence(**dict(values, **kwargs))


def query(history, request=None, **kwargs):
    return effort_reference(history, request or effort('Z3'), mode=kwargs.pop('mode', 'steady_exposure'),
        config=CFG, as_of=AT, accounting_key='units', bout_seconds=kwargs.pop('bout_seconds', 300), **kwargs)


def sample(speed, bpm):
    return PaceSample('easy', 120, speed*120, 'valid steady measured window', (bpm-49)/(194-49))


def test_moderate_query_does_not_pool_fast_work_or_invert_hr_equilibrium():
    row = evidence(samples=(sample(2.8, 158), sample(3.8, 174)),
        executions=(ExecutionPace(60, 4.4, effort('Z4')),))
    moderate = query([row])
    high = query([row], effort('Z4'))
    assert moderate['profile'].first_speed.estimate == pytest.approx(2.8)
    assert high['profile'].first_speed.estimate == pytest.approx(3.8)
    assert moderate['profile'].first_speed.estimate != 4.4
    assert not moderate['uncertainty_calibrated'] and not moderate['planner_influence']


def test_short_command_uses_known_execution_not_delayed_measured_hr():
    row = evidence(samples=(sample(2.8, 158),), executions=(ExecutionPace(45, 4.4, effort('Z4')),))
    command = query([row], effort('Z4'), mode='commanded_execution', bout_seconds=90)
    measured = query([row], effort('Z4'))
    assert command['profile'].first_speed.estimate == pytest.approx(4.4)
    assert command['duration_transfer']  # No identical duration/template prerequisite.
    assert measured['pace'] is None  # No fabricated fast-HR observation or automatic fallback.


def test_unknown_intent_cannot_be_filled_from_measured_hr_or_label():
    row = evidence(samples=(sample(3.8, 174),), executions=(ExecutionPace(60, 4.4, None),))
    command = query([row], effort('Z4'), mode='commanded_execution')
    assert command['pace'] is None
    assert query([row], effort('Z4'))['pace'] is not None


def test_no_history_keeps_explicit_supplied_scenario_available_not_zero_or_banned():
    supplied = pace(3.5, 'declared uncertain transfer')
    answer = query([], mode='supplied', supplied=supplied)
    assert answer['pace'] is supplied and answer['source_keys'] == supplied.source_keys
    assert query([])['pace'] is None
    with pytest.raises(ValueError, match='causal'):
        query([], mode='supplied', supplied=replace(supplied, issued_at=AT+timedelta(seconds=1)))


def test_equal_session_ownership_not_one_vote_per_window_or_rep():
    a = evidence('a', samples=(sample(2.5, 158),)*20)
    b = evidence('b', samples=(sample(3.2, 158),))
    assert query([a, b])['profile'].first_speed.estimate == pytest.approx((2.5*3.2)**.5)


@pytest.mark.parametrize('changed,reason', [
    ({'end':AT, 'available_at':AT}, 'not_completed'),
    ({'available_at':AT+timedelta(seconds=1)}, 'not_available'),
    ({'accounting_key':'other'}, 'accounting_mismatch'),
    ({'start':AT-timedelta(days=31), 'end':AT-timedelta(days=30),
      'available_at':AT-timedelta(days=30)}, 'outside_current_window'),
    ({'eligible':False}, 'ineligible'),
])
def test_causal_eligible_same_normalization_recent_only(changed, reason):
    row = evidence(samples=(sample(2.8,158),), **changed)
    answer = query([row])
    assert answer['pace'] is None and answer['excluded'] == (('past', reason),)


def test_gradual_uses_distinct_measured_endpoints_and_explicit_linear_scenario():
    row = evidence(samples=(sample(2.5,145), sample(2.9,158)))
    request = Effort(('Z2',), ('Z3',), 'gradual')
    answer = query([row], request)
    p = answer['profile']
    assert (p.first_speed.estimate, p.last_speed.estimate) == pytest.approx((2.5,2.9))
    assert 1/answer['pace'].seconds_per_meter.estimate == pytest.approx(2.7)
    assert answer['progression_profile_is_declared']
    assert query([row], request, mode='commanded_execution')['pace'] is None


def test_effort_qualifiers_not_silently_reinterpreted_as_equivalent_commands():
    row = evidence(executions=(ExecutionPace(300,3.3,effort('upper Z3','low Z4')),))
    assert query([row], effort('Z3','Z4'), mode='commanded_execution')['pace'] is None
    assert query([row], effort(' LOW z4 ', 'Upper z3'), mode='commanded_execution')['pace'] is not None


def test_unknown_numeric_cues_remain_unresolved_and_duplicates_rejected():
    assert query([], effort('threshold'))['status'] == 'unknown_numeric_exposure'
    row = evidence()
    with pytest.raises(ValueError, match='Duplicate'):
        query([row,row])
    with pytest.raises(ValueError, match='provenance'):
        replace(evidence(executions=(ExecutionPace(60,4.,effort('Z4')),)), command_issued_at=None)


def test_known_commands_with_different_durations_share_existing_similarity_not_exact_match():
    row = evidence(executions=(ExecutionPace(60,4.1,effort('Z4')),
                              ExecutionPace(300,3.5,effort('Z4'))))
    short = query([row], effort('Z4'), mode='commanded_execution', bout_seconds=90)
    long = query([row], effort('Z4'), mode='commanded_execution', bout_seconds=240)
    assert short['profile'].first_speed.estimate > long['profile'].first_speed.estimate
    assert short['source_keys'] == long['source_keys'] == ('past',)


def test_conflicting_observed_endpoints_cannot_issue_a_backwards_progression():
    row = evidence(samples=(sample(2.9,145), sample(2.5,158)))
    answer = query([row], Effort(('Z2',),('Z3',),'gradual'))
    assert answer['status'] == 'measured_endpoints_do_not_support_increasing_progression'
    assert answer['pace'] is None and answer['profile'] is None
    assert [q.estimate for q in answer['observed_endpoint_speeds']] == pytest.approx([2.9,2.5])


@pytest.mark.parametrize('unit,amount,seconds,meters', [('distance',400,100,400),('time',90,90,360)])
def test_effort_reference_connects_to_existing_compiler_without_changing_primary_unit(unit,amount,seconds,meters):
    request = effort('Z4')
    reference = query([evidence(executions=(ExecutionPace(60,4.,request),))],
        request, mode='commanded_execution', bout_seconds=100)
    block = BlockCommand('work',Measure(unit,Quantity.point(amount,'committed command')),
        request,reference['pace'],'Run at the requested effort.')
    warmup = BlockCommand('warmup',Measure('time',Quantity.point(600,'declared warm-up')),
        effort('Z1','Z2',purpose='easy'),pace(2.5,'explicit warm-up scenario'),'Warm up easily.')
    compiled = compile_blocks(BlockRequest('any family','any variant','Example','intervals',AT,
        'engineering control, not a recommended dose',(warmup,block)))
    assert compiled.request.blocks[1].primary.unit == unit
    assert (compiled.phases[1].seconds.estimate,compiled.phases[1].meters.estimate) == pytest.approx((seconds,meters))
