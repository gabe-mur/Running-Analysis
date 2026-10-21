from dataclasses import asdict
from datetime import date, timedelta
import json
import math

import pytest

from run_analysis.adaptive.frequency_range_trial import (
    FrequencyRangePolicy, FrequencyStageTarget, FrequencyMidpointPreference,
    assess_frequency, ranged_evaluator, midpoint_evaluator)

START = date(2026, 10, 3)
TARGET = FrequencyStageTarget.freeze('stage', START, 22.)


def audit(count, *, decision=START, observed=(), days=28):
    planned = tuple(START+timedelta(days=i) for i in range(count)
                    if START+timedelta(days=i) >= decision)
    return assess_frequency((TARGET,), observed, planned, decision=decision,
        coverage_end=START+timedelta(days=days))


def test_center_zero_symmetric_edges_explicit_fraction():
    p = FrequencyMidpointPreference(100., 28, .02)
    assert p.assess(audit(16))['cost'] == 0
    assert p.assess(audit(12))['cost'] == p.assess(audit(20))['cost'] == 2
    assert p.assess(audit(14))['cost'] == .5


def test_frozen_reference_not_candidate_cost_or_fatigue():
    p = FrequencyMidpointPreference(100., 28, .02)
    program = tuple(((START+timedelta(days=i)).isoformat(), 'easy') for i in range(20))
    for fatigue in (1., 10000.):
        ranged = ranged_evaluator(lambda x: dict(cost=3., fatigue=fatigue, eligible=True),
            (TARGET,), (), decision=START, coverage_end=START+timedelta(days=28))
        result = midpoint_evaluator(ranged, p)(program)
        assert result['cost'] == 5. and result['model_cost'] == 3.
        assert result['fatigue'] == fatigue and result['eligible']


def test_admission_never_overridden_and_zero_weight_exact():
    program = tuple(((START+timedelta(days=i)).isoformat(), 'easy') for i in range(16))
    ranged = ranged_evaluator(lambda x: dict(cost=3., eligible=False, construction_eligible=False),
        (TARGET,), (), decision=START, coverage_end=START+timedelta(days=28))
    result = midpoint_evaluator(ranged, FrequencyMidpointPreference(100., 28, 0.))(program)
    assert result['cost'] == 3.
    assert not result['eligible'] and not result['construction_eligible']


def test_rollover_replacement_does_not_rebase_preference():
    p = FrequencyMidpointPreference(100., 49, .02)
    original = p.assess(audit(20, days=49))
    rolled = p.assess(audit(20, decision=START+timedelta(days=5),
        observed=tuple(START+timedelta(days=i) for i in range(5)), days=49))
    assert original == rolled
    assert original['cost'] == pytest.approx(2*28/49)


def test_incomplete_tail_not_charged_or_counted_as_rest():
    p = FrequencyMidpointPreference(100., 49, .02)
    assert p.assess(audit(20, days=49))['cost'] == p.assess(audit(20, days=28))['cost']
    assert p.assess(audit(16, days=21))['cost'] == 0


def test_json_roundtrip_and_dated_bounds_center():
    p = FrequencyMidpointPreference(100., 28, .02)
    assert FrequencyMidpointPreference(**json.loads(json.dumps(asdict(p)))) == p
    later = START+timedelta(days=14)
    low = FrequencyStageTarget.freeze('low', START, 5.)
    high = FrequencyStageTarget.freeze('high', later, 22.)
    a = assess_frequency((low, high), (START,),
        tuple(later+timedelta(days=i) for i in range(12)), decision=later,
        coverage_end=START+timedelta(days=28))
    assert p.assess(a)['blocks'][0]['center_count'] == 13
    assert p.assess(a)['cost'] == 0


def test_equal_bound_no_artificial_preference():
    policy = FrequencyRangePolicy(anchors=((5., 3., 3.), (22., 4., 4.)))
    t = FrequencyStageTarget.freeze('narrow', START, 22., policy=policy)
    a = assess_frequency((t,), (), tuple(START+timedelta(days=i) for i in range(16)),
        decision=START, coverage_end=START+timedelta(days=28), policy=policy)
    assert FrequencyMidpointPreference(100., 28, .02).assess(a)['cost'] == 0


@pytest.mark.parametrize('kwargs', [dict(reference_fatigue=0), dict(reference_fatigue=math.nan),
    dict(reference_days=0), dict(reference_days=True), dict(edge_fraction=-1),
    dict(edge_fraction=math.inf)])
def test_invalid_reference_rejected(kwargs):
    with pytest.raises(ValueError):
        FrequencyMidpointPreference(**dict(dict(reference_fatigue=100., reference_days=28,
            edge_fraction=.02), **kwargs))
