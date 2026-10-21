from dataclasses import replace
from types import SimpleNamespace

import pytest

from run_analysis.adaptive.quality_coaching_policy import (
    freeze_sequence, evaluate_coached_program, continuation_reference)
from run_analysis.adaptive.shared_program_evaluator import ProgramCandidate
from run_analysis.workload_accounting import Quantity
from test_shared_program_evaluator import context, session


def dose(key, family, recipe, work, primary, miles=3.4):
    return SimpleNamespace(key=key, role='quality', miles=miles,
        phases=(('warmup', 720), ('work', work * 60), ('cooldown', 600)),
        workout_definition=dict(family=family, recipe_key=recipe,
            numeric_contract=dict(phases=[dict(kind='work', primary=dict(amount=dict(estimate=primary)))])))


def catalogue():
    rows = (
        dose('threshold10', 'threshold', 'threshold_continuous', 10.5, 630),
        dose('threshold6', 'threshold', 'threshold_continuous', 6, 360),
        dose('threshold_reps', 'threshold', 'threshold_repeats', 10, 300),
        dose('short3', 'short_intervals', 'pickups_distance', 5.6, 400, 3.3),
        dose('short2', 'short_intervals', 'pickups_distance', 3.7, 400, 2.9),
        dose('short_timed', 'short_intervals', 'pickups_timed', 6, 60, 3.7),
        dose('long9', 'long_intervals', 'aerobic_timed', 9, 180, 3.7),
        dose('long6', 'long_intervals', 'aerobic_timed', 6, 180, 3.1),
        dose('progression', 'progression', 'progression', 14, 840, 4.2),
        dose('progression_reduced', 'progression', 'progression', 11, 660, 3.4),
        dose('fartlek', 'fartlek', 'fartlek_surges', 5, 60, 3.5))
    result = {d.key: d for d in rows}
    result['easy'] = SimpleNamespace(key='easy', role='easy', miles=3., workout_definition=None)
    return result


def sequence(**kwargs):
    return freeze_sequence(catalogue(), stage_key='stage', issued_at='2026-10-09T00:00:00+00:00',
        ordinary_miles=4.2, quality_work_minutes=10.5, **kwargs)


def test_purpose_family_structure_variation_is_issued_before_fatigue():
    q = sequence()
    assignments = [q.assignment(i) for i in range(q.cycle_length)]
    assert all(r.purpose == ('sustained' if i % 2 == 0 else 'faster') for i, r in enumerate(assignments))
    assert {r.family for r in assignments} == {'threshold', 'progression', 'short_intervals', 'long_intervals', 'fartlek'}
    assert {r.recipe for r in assignments} == {r.recipe for r in q.ordinary}
    assert q.assignment(q.cycle_length) == q.assignment(0)
    assert q.assignment(0).key == 'threshold10'
    assert q.assignment(1).key == 'short3'
    assert q.assignment(2).key == 'progression'
    assert q.assignment(3).key == 'long9'
    # Native duration goals are not HR-exposure normalization targets.
    assert q.assignment(0).requested_work_minutes == 12.6
    assert q.assignment(0).accepted_work_minutes == 10.5
    assert q.assignment(3).commanded_work_minutes == 9.


def test_cost_does_not_select_dose_and_disabled_family_is_not_inserted():
    c = catalogue()
    for d in c.values():
        d.fatigue = -1000 if 'reduced' in d.key or d.key.endswith('6') else 1000
    a = freeze_sequence(c, stage_key='stage', issued_at='2026-10-09T00:00:00+00:00',
        ordinary_miles=4.2, quality_work_minutes=10.5)
    assert a == sequence()
    c = {k: d for k, d in c.items() if d.workout_definition is None or d.workout_definition['family'] != 'fartlek'}
    b = freeze_sequence(c, stage_key='stage', issued_at=a.issued_at, ordinary_miles=4.2, quality_work_minutes=10.5)
    assert all(b.assignment(i).family != 'fartlek' for i in range(b.cycle_length))


def test_sequence_binding_is_idempotent_keeps_dates_and_ignores_requested_cheaper_type():
    q = sequence(); c = catalogue()
    program = (('2026-10-09', 'short2'), ('2026-10-11', 'easy'), ('2026-10-15', 'long6'))
    bound = q.bind(program, c)
    assert bound == (('2026-10-09', 'threshold10'), ('2026-10-11', 'easy'), ('2026-10-15', 'short3'))
    assert q.bind(bound, c) == bound
    moved = q.bind((('2026-10-10', 'short2'), *program[1:]), c)
    assert [k for _, k in moved] == [k for _, k in bound]
    assert q.consumed == 0  # Neither a calendar move nor a missed day advances it.
    assert q.after_completion('threshold10').assignment(0).key == 'short3'
    with pytest.raises(ValueError, match='next reconciled'):
        q.after_completion('short3')


def test_incomplete_purpose_inventory_and_invalid_stage_are_explicit():
    c = {k: d for k, d in catalogue().items() if d.workout_definition is None or
        d.workout_definition['family'] == 'short_intervals'}
    with pytest.raises(ValueError, match='both coaching purposes'):
        freeze_sequence(c, stage_key='stage', issued_at='2026-10-09T00:00:00+00:00', ordinary_miles=4.2, quality_work_minutes=10.5)
    with pytest.raises(ValueError, match='physical stage'):
        freeze_sequence(catalogue(), stage_key='stage', issued_at='2026-10-09T00:00:00+00:00', ordinary_miles=4.2, quality_work_minutes=float('nan'))


def test_coaching_dose_fulfillment_preserves_demands_and_history():
    q = sequence(); ctx = context()
    rows = tuple(replace(session('candidate:'+session('x', i).day.isoformat()+':'+q.assignment(rank).key,
        i, role='quality'), quality_credit=1.) for rank, i in enumerate((0, 7)))
    candidate = ProgramCandidate(rows, (), 'explicit fixture tail')
    original_demands = [r.demand for r in (*ctx.history, *rows)]
    reference = ('quality', 3.5, (Quantity.point(1., 'fixture'),) * 3)
    result = evaluate_coached_program(ctx, candidate, sequence=q, quality_reference=reference)
    assert [r.demand for r in result.conditional_ledger] == original_demands
    assert [r.quality_credit for r in result.conditional_ledger[-2:]] == [1., 1.]
    assert result.conditional_ledger[:-2] == ctx.history
    assert not result.parts['quality_coaching']['adaptation_equivalence_established']
    assert not result.parts['quality_coaching']['dose_choice_by_fatigue']
    with pytest.raises(ValueError, match='pre-scored'):
        evaluate_coached_program(ctx, replace(candidate, hard=(replace(rows[0], key='candidate:2026-09-27:short2'),)),
            sequence=q, quality_reference=reference)


def test_continuation_is_cycle_average_not_cheapest_or_tail_only():
    q = sequence()
    values = {r.key: i + 1 for i, r in enumerate(q.ordinary)}
    physical = {('day', r.key): SimpleNamespace(miles=r.miles,
        demand=(Quantity.point(values[r.key], 'fixture'),) * 3) for r in q.ordinary}
    _, miles, demand = continuation_reference(q, physical, 'day')
    cycle = [q.assignment(i) for i in range(q.cycle_length)]
    assert miles == pytest.approx(sum(r.miles for r in cycle) / len(cycle))
    assert demand[0].estimate == pytest.approx(sum(values[r.key] for r in cycle) / len(cycle))
    assert demand[0].estimate > min(values.values())


def test_cadence_contract_is_frozen_and_old_snapshot_and_sequence_identity_stay_exact():
    from dataclasses import asdict
    from hashlib import sha256
    import json
    from run_analysis.adaptive.decision_snapshot import _encode,_decode
    from run_analysis.adaptive.quality_coaching_policy import VERSION,read_sequence
    old=sequence();payload=asdict(old);payload.pop('cadence_contract');payload.pop('focus')
    expected=VERSION+':'+sha256(json.dumps(payload,sort_keys=True,allow_nan=False).encode()).hexdigest()[:20]
    assert old.key==expected
    encoded=_encode(old)
    assert 'cadence_contract' not in encoded['fields']
    assert 'focus' not in encoded['fields']
    assert _decode(encoded)==old==read_sequence(payload)
    new=sequence(cadence_contract='proportional_grace_v1')
    assert new.ordinary==old.ordinary and new.key!=old.key
    assert _decode(_encode(new))==new


@pytest.mark.parametrize('first_purpose', ['sustained', 'faster'])
def test_vo2_focus_emphasizes_aerobic_power_preserves_variety_and_complete_cycle(first_purpose):
    from collections import Counter
    q = sequence(focus='vo2max', first_purpose=first_purpose)
    rows = [q.assignment(i) for i in range(q.cycle_length)]
    purposes = Counter(r.purpose for r in rows)
    assert purposes['faster'] == 2 * purposes['sustained']
    families = Counter(r.family for r in rows if r.purpose == 'faster')
    assert families['long_intervals'] == 2 * families['short_intervals']
    assert families['short_intervals'] == families['fartlek']
    assert {r.recipe for r in rows} == {r.recipe for r in q.ordinary}
    assert q.assignment(0).purpose == first_purpose
    assert all(q.assignment(i) == q.assignment(i + q.cycle_length) for i in range(q.cycle_length))
    assert q.key != sequence(first_purpose=first_purpose).key
    for rank in range(q.cycle_length):
        assert replace(q, consumed=rank).assignment(0) == q.assignment(rank)


def test_vo2_focus_selects_longer_admitted_commands_without_inventing_a_dose():
    c = catalogue()
    c['long8'] = dose('long8', 'long_intervals', 'aerobic_timed', 8, 240, 3.8)
    c['distance600'] = dose('distance600', 'long_intervals', 'aerobic_distance', 8, 600)
    c['distance800'] = dose('distance800', 'long_intervals', 'aerobic_distance', 8.5, 800)
    kw = dict(stage_key='stage', issued_at='2026-10-09T00:00:00+00:00',
        ordinary_miles=4.2, quality_work_minutes=10.5)
    balanced = freeze_sequence(c, **kw)
    focused = freeze_sequence(c, **kw, focus='vo2max')
    assert {r.recipe:r.key for r in balanced.ordinary}['aerobic_timed'] == 'long9'
    selected = {r.recipe:r for r in focused.ordinary}
    assert selected['aerobic_timed'].key == 'long8'
    assert selected['aerobic_distance'].key == 'distance800'
    assert selected['aerobic_timed'].requested_work_minutes == 10.5
    assert all(r.key in c for r in focused.ordinary)
    # Missing preferred commands use the available admitted options, not new
    # durations/forecasts or a larger hard-work budget.
    assert {r.recipe:r.key for r in sequence(focus='vo2max').ordinary}['aerobic_timed'] == 'long9'


def test_vo2_focus_never_reintroduces_disabled_aerobic_intervals():
    c = {k:d for k,d in catalogue().items() if not d.workout_definition or
        d.workout_definition['family'] != 'long_intervals'}
    q = freeze_sequence(c, stage_key='stage', issued_at='2026-10-09T00:00:00+00:00',
        ordinary_miles=4.2, quality_work_minutes=10.5, focus='vo2max')
    assert all(q.assignment(i).family != 'long_intervals' for i in range(q.cycle_length))


def test_focus_snapshot_roundtrip_and_invalid_focus():
    from dataclasses import asdict
    from run_analysis.adaptive.decision_snapshot import _encode, _decode
    from run_analysis.adaptive.quality_coaching_policy import read_sequence
    q = sequence(focus='vo2max')
    assert _encode(q)['fields']['focus'] == 'vo2max'
    assert _decode(_encode(q)) == read_sequence(asdict(q)) == q
    with pytest.raises(ValueError):
        sequence(focus='random')
