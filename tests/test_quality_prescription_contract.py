"""Compiler-only gates; no calendar optimizer, live source wiring or fitted HR."""
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace

import pytest

from run_analysis.adaptive.quality_prescription_contract import (
    Measure, PaceReference, QualityRequest, causal_paces, clock, compile_quality, read_prescription,
)
from run_analysis.adaptive.new_model_daily_harness import Dose
from run_analysis.staged_progression import MILE
from run_analysis.workload_accounting import Quantity

AT = datetime(2026, 10, 8, tzinfo=timezone.utc)


def measure(unit, amount):
    return Measure(unit, Quantity.point(amount, 'explicit test command'))


def pace(value, key='past', basis='causal test scenario'):
    return PaceReference(Quantity(value*.9, value*1.1, value, basis), AT-timedelta(days=1), (key,), basis)


def compile(family, unit='time', amount=60, count=6, **kwargs):
    request = QualityRequest(family, unit+'_repetitions', AT, 'explicit test allocation',
        (measure(unit, amount),)*count, (measure('time', 90),)*(count-1), **kwargs)
    return compile_quality(request, easy_pace=pace(.45), work_pace=pace(.3))


@pytest.mark.parametrize('family,unit,amount,count', [
    ('short_intervals', 'time', 60, 8), ('short_intervals', 'distance', 400, 6),
    ('short_intervals', 'distance', 500, 3), ('long_intervals', 'time', 210, 3),
    ('long_intervals', 'distance', 1000, 3), ('threshold', 'time', 1080, 1),
    ('fartlek', 'time', 75, 8), ('hill_repeats', 'time', 45, 8),
])
def test_every_family_and_variant_preserves_primary_unit_and_scored_structure(family, unit, amount, count):
    extras = dict(warmup=measure('time', 720), cooldown=measure('time', 600)) if family == 'fartlek' else {}
    p = compile(family, unit, amount, count, **extras)
    before = asdict(p)
    steps = p.steps()
    assert len(steps) == 3
    assert steps[1].prescription_basis == unit
    assert steps[1].repetitions == (count if count > 1 else None)
    assert sum(s.duration_minutes for s in steps)*60 == pytest.approx(sum(s.seconds.estimate for s in p.phases))
    assert sum(s.distance_miles for s in steps) == pytest.approx(sum(s.meters.estimate for s in p.phases)/MILE)
    assert p.miles == round(p.miles, 1)
    assert abs(p.miles-sum(s.distance_miles for s in steps)) <= .05+1e-8
    work = [s for s in p.phases if s.kind == 'work']
    assert all(s.primary.amount.estimate == amount for s in work)
    if unit == 'time':
        assert all(s.seconds.estimate == amount for s in work)
    else:
        assert all(s.meters.estimate == amount for s in work)
        assert all(s.seconds.low < s.seconds.estimate < s.seconds.high for s in work)
    assert asdict(p) == before  # Rendering cannot mutate the scored command.
    definition = json.loads(json.dumps(p.definition()))
    reopened = read_prescription(definition['numeric_contract'])
    assert reopened == p and reopened.key == p.key and reopened.definition() == p.definition()
    Dose(p.key, 'quality', p.miles, tuple((s.kind, s.seconds.estimate) for s in p.phases),
        phase_scenarios=tuple((s.seconds.low, s.seconds.high) for s in p.phases), workout_definition=definition)


def test_mixed_units_and_final_recovery_are_exact_not_inferred_from_family():
    p = compile('short_intervals', 'distance', 400)
    assert '6 × 400 m' in p.steps()[1].instruction
    assert '1:30' in p.steps()[1].instruction
    assert [s.primary.unit for s in p.phases[1:4]] == ['distance', 'time', 'distance']
    r = replace(p.request, recoveries=p.request.recoveries+(measure('time', 90),))
    final = compile_quality(r, easy_pace=p.easy_pace, work_pace=p.work_pace)
    assert 'after each effort' in final.steps()[1].instruction
    assert final.phases[-2].kind == 'recovery'
    assert final.key != p.key
    timed = compile('short_intervals')
    assert '6 × 1 min' in timed.steps()[1].instruction
    assert timed.steps()[1].prescription_basis == 'time'


def test_ranged_time_command_is_displayed_as_range_with_nominal_dose_unchanged():
    p = compile('long_intervals', count=3)
    work = Measure('time', Quantity(180, 300, 210, 'explicit flexible command; nominal3:30'))
    p = compile_quality(replace(p.request, work=(work,)*len(p.request.work)),
        easy_pace=p.easy_pace, work_pace=p.work_pace)
    before = p.key
    assert '3 × 3–5 min' in p.steps()[1].instruction
    assert p.steps()[1].work_duration_range_minutes == (3., 5.)
    assert p.steps()[1].work_duration_minutes == 3.5
    assert p.key == before
    assert read_prescription(p.definition()['numeric_contract']) == p


def test_progression_preserves_distance_halves_and_gradual_effort():
    r = QualityRequest('progression', 'second_half', AT, 'test allocation',
        (measure('portion', .5),), total_miles=4.5)
    p = compile_quality(r, easy_pace=pace(.45), work_pace=pace(.4, basis='explicit gradual-effort pace scenario'))
    assert [s.primary.unit for s in p.phases] == ['portion', 'portion']
    assert [s.meters.estimate/MILE for s in p.phases] == pytest.approx([2.25, 2.25])
    assert p.phases[1].effort.purpose == 'gradual'
    assert p.phases[1].effort.start_cues == ('Z2',)
    assert p.phases[1].effort.finish_cues == ('Z3',)
    assert all(s.prescription_basis == 'portion' for s in p.steps())
    assert 'Gradually' in p.steps()[1].instruction
    assert read_prescription(p.definition()['numeric_contract']) == p


def test_open_recovery_and_bookends_cannot_silently_invent_extent():
    with pytest.raises(ValueError, match='explicit sourced scenario'):
        compile('fartlek')
    p = compile('hill_repeats', amount=45, open_recovery=True)
    assert all(s.open_recovery for s in p.phases if s.kind == 'recovery')
    assert 'forecast scenario' in p.steps()[1].instruction
    assert 'uphill' in p.steps()[1].instruction and 'downhill' in p.steps()[1].instruction
    assert 'flat terrain' in p.steps()[0].instruction


def test_nonuniform_targets_are_not_turned_into_uniform_reps():
    p = compile('short_intervals', 'distance', 400, 3)
    r = replace(p.request, work=(measure('distance', 400), measure('distance', 500), measure('distance', 400)))
    uneven = compile_quality(r, easy_pace=p.easy_pace, work_pace=p.work_pace)
    assert '400 m, 500 m, 400 m' in uneven.steps()[1].instruction
    assert uneven.steps()[1].work_duration_minutes is None
    assert p.key != uneven.key


def test_nonuniform_recoveries_preserve_order_and_repeated_targets():
    p = compile('short_intervals', 'distance', 400, 4)
    r = replace(p.request, recoveries=(measure('time', 60), measure('time', 90), measure('time', 60)))
    uneven = compile_quality(r, easy_pace=p.easy_pace, work_pace=p.work_pace)
    assert '1 min, 1:30, 1 min between efforts, in that order.' in uneven.steps()[1].instruction
    assert uneven.steps()[1].recovery_duration_minutes is None
    assert p.key != uneven.key


def test_identity_covers_units_effort_references_and_version():
    p = compile('long_intervals', amount=210, count=3)
    assert '3 × 3:30' in p.steps()[1].instruction
    assert clock(210) == '3:30' and clock(180) == '3 min' and clock(45) == '45 seconds'
    assert replace(p, work_pace=pace(.3, key='different evidence')).key != p.key
    changed_phase = replace(p.phases[1], effort=replace(p.phases[1].effort, finish_cues=('Z3',)))
    assert replace(p, phases=(p.phases[0], changed_phase, *p.phases[2:])).key != p.key
    with pytest.raises(ValueError):
        replace(p, version='unvalidated future version')
    with pytest.raises(ValueError):
        compile_quality(p.request, easy_pace=replace(p.easy_pace, issued_at=AT+timedelta(seconds=1)), work_pace=p.work_pace)


def run(key, role, seconds=1800, miles=3, **changes):
    start = AT-timedelta(days=1)
    return SimpleNamespace(key=key, role=role, start=start, end=start+timedelta(seconds=seconds),
        available_at=start+timedelta(seconds=seconds), miles=miles,
        demand=(Quantity.point(seconds/60, 'observed moving duration'),), bouts=(), **changes)


def test_paces_are_causal_filtered_one_vote_per_session_and_athlete_relative():
    a = run('easy', 'easy')
    bout = SimpleNamespace(work_seconds=120, distance_meters=400)
    b = run('work', 'quality', work_evidence_basis='detected')
    b.bouts = (bout,)*10
    rejected = run('illness', 'easy', seconds=3000)
    later = run('late', 'easy'); later.available_at = AT+timedelta(seconds=1)
    p = causal_paces([a, b, rejected, later], as_of=AT, eligible_keys={'easy', 'work', 'late'})
    assert p['easy'].source_keys == ('easy',)
    assert p['work'].source_keys == ('work',)
    assert p['work'].seconds_per_meter.estimate == .3
    p2 = causal_paces([a], as_of=AT, eligible_keys={'easy'})
    assert p2['work'] is None
    with pytest.raises(ValueError, match='explicit sourced scenario'):
        compile_quality(compile('short_intervals').request, easy_pace=p2['easy'], work_pace=None)
    fallback = causal_paces([b], as_of=AT, eligible_keys={'work'})
    b.bouts = ()
    whole = causal_paces([b], as_of=AT, eligible_keys={'work'})
    assert 'fallback' in whole['work'].basis and 'measured work-bout' in fallback['work'].basis
    assert p['easy'].seconds_per_meter.estimate == pytest.approx(1800/(3*MILE))
    with pytest.raises(ValueError, match='Duplicate'):
        causal_paces([a, a], as_of=AT, eligible_keys={'easy'})


@pytest.mark.parametrize('changes', [dict(recoveries=()), dict(work=[]), dict(total_miles=4.25),
    dict(warmup=measure('portion', .5))])
def test_incomplete_or_contradictory_commands_rejected(changes):
    with pytest.raises(ValueError):
        replace(compile('short_intervals').request, **changes)
