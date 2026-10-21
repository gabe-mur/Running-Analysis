"""Tests of the exploratory diagnostic, not physiological validation."""
from datetime import datetime, timedelta, timezone
import math

import numpy as np
import pytest

from scripts.audit_additional_pace_signals import design, fit, contrast, matched_time_checks


def planted_rows(effect=.03, confounded=False):
    rng = np.random.default_rng(24)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(200):
        at = start+timedelta(days=i*3)
        block = math.floor(at.timestamp()/86400/56)
        hour = (8 if block % 2 else 18) if confounded else (8 if rng.random() < .5 else 18)
        sine = math.sin(math.tau*hour/24)
        pace = 10*math.exp(-i*.0003+effect*sine+rng.normal(0,.002))
        rows.append(dict(activity_id=i, start_time_utc=at.isoformat(), pace=pace,
                         hour_sin=sine, hour_cos=math.cos(math.tau*hour/24)))
    return rows


def test_cyclic_contrast_recovers_planted_direction_and_rough_magnitude():
    rows = planted_rows(); keys = ('hour_sin','hour_cos')
    model = fit(rows, keys)
    change = contrast(rows, model, keys, {
        'hour_sin': (math.sin(math.tau*8/24),math.sin(math.tau*18/24)),
        'hour_cos': (math.cos(math.tau*8/24),math.cos(math.tau*18/24)),
    })
    expected = np.median([r['pace'] for r in rows])*60*math.expm1(.03*(-1-math.sin(math.tau*8/24)))
    assert change['seconds_per_mile'] == pytest.approx(expected, abs=3)
    assert change['conditional_block_range_95'][1] < 0


def test_calendar_only_time_difference_cannot_teach_clock_effect():
    model = fit(planted_rows(confounded=True), ('hour_sin','hour_cos'))
    assert np.max(np.abs(model['beta'])) < 1e-10


def test_future_data_does_not_change_training_scales_or_categories():
    rows = [dict(x, device='old') for x in planted_rows()]
    _, spec = design(rows, ('hour_sin','device'))
    _, same = design([dict(rows[0], hour_sin=1000, device='new')], (), spec)
    assert same is spec
    assert not any(f.get('level') == 'new' for f in same)


def test_missing_value_not_confused_with_measured_zero():
    rows = [dict(x, extra=(None if i%3 == 0 else float(i%2))) for i,x in enumerate(planted_rows())]
    x, spec = design(rows, ('extra',))
    missing = next(i for i,f in enumerate(spec) if f['kind'] == 'missing')
    assert x[0,missing] == 1
    assert x[2,missing] == 0


def test_context_matches_are_disjoint_and_not_chosen_for_pace_agreement():
    rows = planted_rows()[:6]
    for i, row in enumerate(rows):
        row.update(hour=10 if i%2 == 0 else 18, area='same', workout_type='easy',
                   sampled_hr_bpm=148, heat_c=15, duration_minutes=30, headwind_mph=0)
    first = matched_time_checks(rows)
    for i, row in enumerate(rows):
        row['pace'] = 2+i*4  # Make agreement completely different.
    second = matched_time_checks(rows)
    for a, b in zip(first['results'], second['results']):
        identities = [identity for pair in b['pairs'] for identity in (pair['morning_id'],pair['evening_id'])]
        assert len(identities) == len(set(identities))
        assert [(p['morning_id'],p['evening_id']) for p in a['pairs']] == [(p['morning_id'],p['evening_id']) for p in b['pairs']]


def test_unknown_weather_cannot_qualify_as_a_context_match():
    rows = planted_rows()[:2]
    for i, row in enumerate(rows):
        row.update(hour=10 if i == 0 else 18, area='same', workout_type='easy',
                   sampled_hr_bpm=148, heat_c=None, duration_minutes=30, headwind_mph=0)
    assert all(r['independent_pairs'] == 0 for r in matched_time_checks(rows)['results'])
