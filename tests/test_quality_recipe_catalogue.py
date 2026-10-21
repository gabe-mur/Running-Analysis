"""Offline recipe/dose contracts, not coaching efficacy or calendar acceptance."""
from dataclasses import replace
from datetime import timedelta
import json
from math import fsum

import pytest

from run_analysis.adaptive.quality_block_contract import adapt_v2, read_quality_contract
from run_analysis.adaptive.quality_recipe_catalogue import (
    DoseWindow, compile_catalogue, compile_recipe_options, progression_options, reviewed_recipes)
from run_analysis.adaptive.work_exposure_matching import requested_work_bands
from run_analysis.staged_progression import MILE
from test_quality_block_contract import pace
from test_quality_dose_policy import AT, CONFIG, menu


def recipes(**changes):
    args = dict(issued_at=AT, easy_pace=pace(.4), threshold_pace=pace(.32),
        aerobic_pace=pace(.28), fast_pace=pace(.27), hill_pace=pace(.5, 'uphill scenario'),
        open_recovery_seconds=90)
    return reviewed_recipes(**dict(args, **changes))


def window(low=360, high=900, **changes):
    args = dict(minimum_work_seconds=low, maximum_work_seconds=high,
        maximum_session_miles=8., maximum_session_seconds=5400., maximum_repetitions=20,
        source='explicit engineering generation envelope, not athlete admission')
    return DoseWindow(**dict(args, **changes))


def catalogue():
    return compile_catalogue(recipes(), {'sustained': window(), 'faster': window(240, 720)})


def test_threshold_and_harder_aerobic_intents_are_explicit_not_label_lookups():
    rs = {r.key:r for r in recipes()}
    threshold, harder = (rs[k] for k in ('threshold_repeats', 'aerobic_timed'))
    assert threshold.base.family == 'threshold' and harder.base.family == 'long_intervals'
    assert threshold.purpose == 'sustained' and harder.purpose == 'faster'
    a, b = (requested_work_bands((r.base.blocks[1],), CONFIG)[0] for r in (threshold, harder))
    assert a.start_band != b.start_band and sum(b.start_band) > sum(a.start_band)
    assert threshold.base.blocks[1].pace != harder.base.blocks[1].pace
    assert threshold.base.blocks[2].primary.amount.estimate == 60
    assert harder.base.blocks[2].primary.amount.estimate == 120


def test_native_grids_vary_counts_time_and_distance_without_converting_workouts():
    rows = catalogue()['candidates']
    assert all(c.prescription.request.family != 'strides' for c in rows)
    for key in ('threshold_repeats', 'aerobic_timed', 'pickups_timed', 'pickups_distance'):
        subset = [c for c in rows if c.recipe_key == key]
        assert len({c.repetitions for c in subset}) > 1
        assert len({c.prescription.request.blocks[1].primary.amount.estimate for c in subset}) > 1
    for c in rows:
        unit = c.prescription.request.blocks[1].primary.unit
        assert unit == ('distance' if 'distance' in c.recipe_key else 'time')
        assert c.prescription.miles == tenth_miles(c.prescription)


def tenth_miles(p):
    from run_analysis.staged_progression import tenth
    return tenth(fsum(q.meters.estimate for q in p.phases)/MILE)


def test_every_declared_valid_count_target_combination_is_present_without_score_pruning():
    r = next(r for r in recipes() if r.key == 'aerobic_timed')
    w = window(240, 720)
    rows = compile_recipe_options(r, w)
    got = {(c.repetitions, c.prescription.request.blocks[1].primary.amount.estimate) for c in rows}
    expected = {(n, s) for s in r.work_targets for n in range(2, 21) if 240 <= n*s <= 720}
    assert got == expected and len({c.prescription.key for c in rows}) == len(rows)
    assert rows == compile_recipe_options(r, w)


def test_full_bookends_and_final_jogs_count_and_display_matches_scored_dose():
    for c in catalogue()['candidates']:
        p = c.prescription
        assert fsum(s.duration_minutes for s in p.steps())*60 == pytest.approx(fsum(q.seconds.estimate for q in p.phases))
        assert fsum(s.distance_miles for s in p.steps()) == pytest.approx(fsum(q.meters.estimate for q in p.phases)/MILE)
        rests = [q for q in p.phases if q.kind == 'recovery']
        assert len(rests) == (0 if c.recipe_key == 'threshold_continuous' else c.repetitions
            if c.recipe_key.startswith('pickups') or c.recipe_key.startswith('fartlek') else c.repetitions-1)
        assert 360 <= c.work_seconds.estimate <= 900 if c.purpose == 'sustained' else 240 <= c.work_seconds.estimate <= 720
    r = next(r for r in recipes() if r.key == 'pickups_timed')
    assert not compile_recipe_options(r, window(240, 300, maximum_session_seconds=1400))


def test_native_targets_frozen_causal_sources_and_purpose_survive_restart():
    for c in catalogue()['candidates']:
        p = c.prescription
        restored = read_quality_contract(json.loads(json.dumps(p.definition()))['numeric_contract'])
        assert restored == p and restored.key == p.key and restored.steps() == p.steps()
        assert all(q.primary.amount.estimate == round(q.primary.amount.estimate) for q in p.phases)
    with pytest.raises(ValueError, match='Future block'):
        recipes(aerobic_pace=replace(pace(.28), issued_at=AT+timedelta(seconds=1)))
    with pytest.raises(ValueError, match='harder aerobic pace'):
        recipes(aerobic_pace=pace(.32))


def test_distance_variant_duration_scales_with_athlete_and_outside_purpose_bounds_is_excluded():
    rs = recipes(aerobic_pace=pace(.31), threshold_pace=pace(.35), easy_pace=pace(.5), fast_pace=pace(.4))
    r = next(r for r in rs if r.key == 'aerobic_distance')
    rows = compile_recipe_options(r, window())
    assert {c.prescription.request.blocks[1].primary.amount.estimate for c in rows} == {600}
    assert all(c.prescription.phases[1].seconds.estimate == 186 for c in rows)
    assert compile_recipe_options(replace(r, work_targets=(1000,)), window()) == ()


def test_hills_do_not_silently_inherit_flat_speed_and_keep_terrain_open_recovery():
    assert 'hill_repeats' not in {r.key for r in recipes(hill_pace=None)}
    rows = [c for c in catalogue()['candidates'] if c.recipe_key == 'hill_repeats']
    assert rows
    for c in rows:
        p = c.prescription
        text = ' '.join(s.instruction for s in p.steps())
        assert all(word in text for word in ('uphill', 'downhill', 'flat terrain', 'forecast scenario'))
        assert p.request.blocks[1].pace.seconds_per_meter.estimate == .5
        assert all(q.open_recovery for q in p.phases if q.kind == 'recovery')


def test_fartlek_retains_landmark_range_and_one_frozen_nominal_not_fake_same_command_variants():
    rows = [c for c in catalogue()['candidates'] if c.recipe_key == 'fartlek_surges']
    assert len({c.repetitions for c in rows}) > 1
    for c in rows:
        p = c.prescription
        work = p.request.blocks[1]
        assert (work.primary.amount.low, work.primary.amount.estimate, work.primary.amount.high) == (45,60,90)
        assert '45–90 seconds' in p.steps()[1].instruction and 'landmarks' in p.steps()[1].instruction
        assert c.work_seconds.low == c.repetitions*45
        assert c.work_seconds.high == c.repetitions*90


def test_growing_declared_window_adds_doses_without_changing_existing_targets():
    r = next(r for r in recipes() if r.key == 'aerobic_timed')
    a, b = (compile_recipe_options(r, window(high=h)) for h in (600, 900))
    # Window provenance unchanged; changing an actual issued command is not allowed.
    assert {c.prescription.key for c in a} <= {c.prescription.key for c in b}
    assert max(c.work_seconds.estimate for c in b) > max(c.work_seconds.estimate for c in a)


def test_progression_varies_committed_distance_without_reconstructing_its_profile():
    old = next(p for p in menu()['prescriptions'] if p.request.family == 'progression')
    base = adapt_v2(old)
    rows = progression_options(base, (3.2, 3.5, 3.8, 3.5), window=window(1, 3600))
    assert [c.prescription.miles for c in rows] == [3.2, 3.5, 3.8]
    assert all(c.purpose == 'gradual' and c.prescription.request.blocks == base.request.blocks for c in rows)
    assert base.request.total_miles == old.request.total_miles
    with pytest.raises(ValueError, match='tenth-mile'):
        progression_options(base, (3.25,), window=window())


def test_invalid_windows_duplicates_missing_purpose_and_empty_recipe_are_explicit():
    for args in ({'maximum_work_seconds': 300}, {'maximum_repetitions': True}, {'source': ''}):
        with pytest.raises(ValueError):
            window(**args)
    with pytest.raises(ValueError, match='Duplicate'):
        compile_catalogue((recipes()[0], recipes()[0]), {'sustained': window()})
    with pytest.raises(ValueError, match='every requested purpose'):
        compile_catalogue(recipes(), {'sustained': window()})
    r = next(r for r in recipes() if r.key == 'threshold_repeats')
    result = compile_catalogue((r,), {'sustained': window(1, 299)})
    assert result['candidates'] == () and result['empty_recipes'] == ('threshold_repeats',)
    assert not result['planner_influence'] and not result['purpose_credit_assigned']
