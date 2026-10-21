"""Architecture controls, not prospective accuracy or calendar acceptance."""
from dataclasses import asdict, replace
from datetime import timedelta
import json
from math import fsum

import pytest

from run_analysis.adaptive.quality_block_contract import (
    BlockCommand, BlockRequest, adapt_v2, block_scenario, compile_blocks, forecast_blocks, read_blocks, read_quality_contract,
)
from run_analysis.adaptive.quality_prescription_contract import Effort, Measure, PaceReference
from run_analysis.adaptive.requested_quality_forecast import PaceProfile, forecast_prescription
from run_analysis.staged_progression import MILE
from run_analysis.workload_accounting import Quantity
from test_quality_dose_policy import AT, menu
from test_requested_quality_forecast import fit


def pace(seconds, key='prior scenario'):
    return PaceReference(Quantity(seconds*.9, seconds*1.1, seconds, 'engineering envelope, not confidence'),
        AT-timedelta(days=1), (key,), 'explicit causal engineering scenario')


def block(kind, seconds=120, *, unit='time', speed_pace=.45, cues=('Z1', 'Z2'), text='Run easily.'):
    return BlockCommand(kind, Measure(unit, Quantity.point(seconds, 'explicit engineering command')),
        Effort(cues, cues, 'easy' if kind != 'work' else 'purposeful'), pace(speed_pace), text)


def prescription(*middle, family='threshold', variant='repeated', title='Threshold repetitions'):
    return compile_blocks(BlockRequest(family, variant, title, 'tempo_threshold', AT,
        'engineering contract control; not a recommended dose',
        (block('warmup', 600), *middle, block('cooldown', 600))))


def repeated():
    work = block('work', 360, speed_pace=.34, cues=('threshold effort',), text='At threshold effort.')
    recovery = block('recovery', 120, text='Jog easily')
    return prescription(work, recovery, work, recovery, work)


def mixed():
    return prescription(
        block('work', 600, speed_pace=.40, cues=('steady effort',), text='At steady effort.'),
        block('work', 400, unit='distance', speed_pace=.32, cues=('threshold effort',), text='At threshold effort.'),
        block('recovery', 90, speed_pace=.5),
        block('work', 200, unit='distance', speed_pace=.25, cues=('fast effort',), text='At fast effort, not sprinting.'),
        family='mixed_quality', variant='steady_threshold_fast', title='Mixed efforts')


def test_threshold_can_be_continuous_or_repeated_without_becoming_higher_intensity():
    p = repeated()
    assert [b.primary.amount.estimate for b in p.request.blocks if b.kind == 'work'] == [360]*3
    assert [b.effort.finish_cues for b in p.request.blocks if b.kind == 'work'] == [('threshold effort',)]*3
    steps = p.steps()
    assert len(steps) == 3 and steps[1].repetitions == 3
    assert '3 × 6 min' in steps[1].instruction and 'between efforts' in steps[1].instruction
    assert 'threshold' in steps[1].instruction and 'hard effort' not in steps[1].instruction
    continuous = prescription(block('work', 1080, speed_pace=.34, cues=('threshold effort',), text='At threshold effort.'),
        variant='continuous')
    assert continuous.steps()[1].repetitions is None
    assert p.key != continuous.key


def test_different_efforts_and_recovery_paces_compile_independently():
    p = mixed()
    assert [b.pace.seconds_per_meter.estimate for b in p.request.blocks[1:-1]] == [.4, .32, .5, .25]
    assert [b.seconds.estimate for b in p.phases[1:-1]] == [600, 128, 90, 50]
    assert [b.meters.estimate for b in p.phases[1:-1]] == [1500, 400, 180, 200]
    assert len(p.steps()) == 5  # A single work/recovery pair is grouped; distinct work cues remain separate.
    text = ' '.join(s.instruction for s in p.steps())
    assert text.index('steady effort') < text.index('threshold effort') < text.index('fast effort')
    assert '400 m' in text and '200 m' in text
    assert p.steps()[1].prescription_basis == 'time'


def test_display_and_forecast_do_not_average_two_work_intensities():
    p = mixed()
    phases = block_scenario(p)
    assert [v.first_speed for v in phases[1:-1]] == pytest.approx([2.5, 3.125, 2., 4.])
    assert sum(s.duration_minutes for s in p.steps())*60 == pytest.approx(fsum(p.seconds.estimate for p in p.phases))
    assert sum(s.distance_miles for s in p.steps()) == pytest.approx(fsum(p.meters.estimate for p in p.phases)/MILE)
    before = p.definition()
    report = forecast_blocks(p, fit(), accounting_key='units', initial_reserve=.45)
    assert report['nominal']['status'] == 'predicted'
    assert report['nominal']['diagnostics']['segment_initializations'] == 1
    assert report['nominal']['predicted_load'] == pytest.approx(sum(v['load'] for v in report['nominal']['phases']))
    assert p.definition() == before and not report['planner_influence']


def test_each_distance_block_derives_its_duration_from_its_own_pace():
    p = mixed()
    slow, fast = (block_scenario(p, point=point) for point in ('lower_speed', 'higher_speed'))
    for i, b in enumerate(p.request.blocks):
        if b.primary.unit == 'distance':
            assert slow[i].meters == pytest.approx(fast[i].meters)
            assert slow[i].seconds > fast[i].seconds
        else:
            assert slow[i].seconds == fast[i].seconds == b.primary.amount.estimate


def test_same_physical_programme_is_label_independent():
    p = mixed()
    q = compile_blocks(replace(p.request, family='anything_else', variant='different name',
        title='Different label', workout_type='intervals'))
    a, b = (forecast_blocks(v, fit(), accounting_key='units', initial_reserve=.45) for v in (p, q))
    assert a['nominal']['normalized_spans'] == b['nominal']['normalized_spans']
    assert a['nominal']['predicted_load'] == b['nominal']['predicted_load']
    assert a['fingerprint'] != b['fingerprint'] and p.key != q.key


def test_frozen_per_block_sources_efforts_and_sequence_survive_restart():
    p = mixed()
    restored = read_blocks(json.loads(json.dumps(p.definition()))['numeric_contract'])
    assert restored == p and restored.key == p.key and restored.definition() == p.definition()
    target = p.request.blocks[2]
    for changed in (replace(target, pace=pace(.32, key='another source')),
                    replace(target, effort=Effort(('Z4',), ('Z4',), 'purposeful')),
                    replace(target, instruction='A different instruction.')):
        q = compile_blocks(replace(p.request, blocks=(*p.request.blocks[:2], changed, *p.request.blocks[3:])))
        assert q.key != p.key
    reversed_work = compile_blocks(replace(p.request, blocks=(p.request.blocks[0],
        *reversed(p.request.blocks[1:-1]), p.request.blocks[-1])))
    assert reversed_work.key != p.key


def test_restoration_rejects_scored_extent_tampering():
    p = mixed()
    payload = p.definition()['numeric_contract']
    payload['phases'][2]['seconds']['estimate'] += 1
    with pytest.raises(ValueError, match='agree exactly'):
        read_blocks(payload)


def test_future_pace_and_profile_rejected_even_if_other_blocks_are_causal():
    p = mixed()
    bad = replace(p.request.blocks[2], pace=replace(p.request.blocks[2].pace, issued_at=AT+timedelta(seconds=1)))
    with pytest.raises(ValueError, match='Future block'):
        compile_blocks(replace(p.request, blocks=(*p.request.blocks[:2], bad, *p.request.blocks[3:])))
    old = next(p for p in menu()['prescriptions'] if p.request.family == 'progression')
    q = adapt_v2(old)
    bad = replace(q.request.blocks[1], profile=replace(q.request.blocks[1].profile, issued_at=AT+timedelta(seconds=1)))
    with pytest.raises(ValueError, match='Future block'):
        replace(q.request, blocks=(q.request.blocks[0], bad))


def test_gradual_requires_explicit_profile_and_consistent_mean():
    b = block('work', 600, speed_pace=.4)
    gradual = Effort(('Z2',), ('Z3',), 'gradual')
    with pytest.raises(ValueError, match='explicit sourced profile'):
        replace(b, effort=gradual)
    q = b.pace.seconds_per_meter
    first = Quantity(2., 2.4, 2.2, 'explicit start scenario')
    last = Quantity(2/q.high-first.low, 2/q.low-first.high, 2/q.estimate-first.estimate, 'explicit finish scenario')
    profile = PaceProfile(first, last, AT-timedelta(days=1), ('prior',), 'explicit engineering profile')
    p = prescription(replace(b, effort=gradual, profile=profile))
    assert block_scenario(p)[1].first_speed < block_scenario(p)[1].last_speed
    assert read_blocks(p.definition()['numeric_contract']) == p
    with pytest.raises(ValueError, match='preserve its own'):
        replace(b, profile=replace(profile, first_speed=Quantity.point(1., 'inconsistent')))


def test_open_recovery_compaction_preserves_after_each_and_known_scenario():
    p = repeated()
    w, r = p.request.blocks[1:3]
    r = replace(r, open_recovery=True)
    q = prescription(w, r, w, r, w, r)
    assert len(q.steps()) == 3
    assert 'after each effort' in q.steps()[1].instruction and 'forecast scenario' in q.steps()[1].instruction
    assert sum(s.duration_minutes for s in q.steps())*60 == pytest.approx(fsum(p.seconds.estimate for p in q.phases))


def test_nonuniform_recovery_is_not_collapsed_and_adjacent_work_is_legal():
    p = repeated()
    w, r = p.request.blocks[1:3]
    q = prescription(w, r, w, replace(r, primary=Measure('time', Quantity.point(90, 'different rest'))), w)
    assert len(q.steps()) == 5
    assert '2 min' in q.steps()[1].instruction and '1:30' in q.steps()[2].instruction
    assert len(prescription(w, w).phases) == 4


def test_bad_portion_partition_and_independent_total_are_rejected():
    p = mixed()
    with pytest.raises(ValueError, match='independent exact total'):
        replace(p.request, total_miles=4.2)
    old = next(p for p in menu()['prescriptions'] if p.request.family == 'progression')
    q = adapt_v2(old)
    bad = replace(q.request.blocks[1], primary=Measure('portion', Quantity.point(.4, 'bad partition')))
    with pytest.raises(ValueError, match='partition'):
        replace(q.request, blocks=(q.request.blocks[0], bad))


def test_all_existing_commands_adapt_without_repricing_or_mutating_v2():
    for p in menu()['prescriptions']:
        old = asdict(p), p.key, p.definition()
        q = adapt_v2(p)
        assert q.phases == p.phases and q.miles == p.miles
        assert read_blocks(q.definition()['numeric_contract']) == q
        x = forecast_prescription(p, fit(), accounting_key='units', initial_reserve=.45)
        y = forecast_blocks(q, fit(), accounting_key='units', initial_reserve=.45)
        for point in ('lower_speed', 'nominal', 'higher_speed'):
            assert x['scenarios'][point]['predicted_load'] == pytest.approx(y['scenarios'][point]['predicted_load'], abs=1e-10)
        assert (asdict(p), p.key, p.definition()) == old


def test_missing_response_remains_unknown_not_a_zero_cost_novel_workout():
    p = mixed()
    result = forecast_blocks(p, None, accounting_key='units', initial_reserve=.45)
    assert result['nominal']['predicted_load'] is None and result['scenario_envelope'] is None
    assert not result['all_scenarios_supported'] and not result['uncertainty_calibrated']


def test_version_dispatch_reads_old_bytes_without_migrating_them():
    old = menu()['prescriptions'][0]
    payload = old.definition()['numeric_contract']
    assert read_quality_contract(payload) == old
    assert read_quality_contract(payload).key == old.key
    new = mixed()
    assert read_quality_contract(new.definition()['numeric_contract']) == new
    with pytest.raises(ValueError, match='Unsupported quality contract version'):
        read_quality_contract(dict(payload, version='unapproved_future'))


def test_changing_one_block_changes_only_its_compiled_extent_before_response():
    p = mixed()
    changed = replace(p.request.blocks[2], pace=pace(.36))
    q = compile_blocks(replace(p.request, blocks=(*p.request.blocks[:2], changed, *p.request.blocks[3:])))
    assert q.phases[:2] == p.phases[:2] and q.phases[3:] == p.phases[3:]
    assert q.phases[2].seconds.estimate == 144 and p.phases[2].seconds.estimate == 128
    assert q.phases[2].meters == p.phases[2].meters
    old, new = (forecast_blocks(v, fit(), accounting_key='units', initial_reserve=.45) for v in (p, q))
    assert old['nominal']['predicted_load'] != new['nominal']['predicted_load']


def test_comparator_preserves_terrain_landmark_and_open_recovery_cues():
    for p in menu()['prescriptions']:
        if p.request.family == 'hill_repeats':
            text = ' '.join(s.instruction for s in adapt_v2(p).steps())
            assert 'uphill' in text and 'downhill' in text and 'flat terrain' in text
            assert 'forecast scenario' in text and 'breathing and form' in text
        elif p.request.family == 'fartlek':
            assert 'landmarks' in ' '.join(s.instruction for s in adapt_v2(p).steps())
