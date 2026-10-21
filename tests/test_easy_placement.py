from itertools import product

import numpy as np
import pytest

from scripts.audit_easy_placement import curves, dose_vector, classify, census, records, panel, ORIGIN
from scripts.audit_easy_distribution import exposures


def test_curve_matches_direct_integer_accounting():
    from scripts.audit_easy_placement import COORDINATES
    days = (0,4,7,11,14,18,21,25)
    values = curves([dose_vector(days,8)])[0]
    assert list(values) == [sum(22 for d in days if start <= d < target) for target,start in COORDINATES]


def test_classification_preserves_tradeoffs_not_scalar_sum():
    assert list(classify(np.array([[0,0],[-1,0],[0,1],[-10,1]]))) == [0,1,2,3]


def test_all_fifteen_key_demand_improvements_are_the_same_local_pattern():
    baseline = (0,5,7,12,14,19,21,26)
    reference = curves([dose_vector(baseline,8)])[0]
    improvements = 0
    for moves in product((0,1),repeat=4):
        days = tuple(d for cycle in range(4) for d in (cycle*7,cycle*7+5-moves[cycle]))
        code = classify(curves([dose_vector(days,8)])-reference)[0]
        assert code == (1 if any(moves) else 0)
        improvements += code == 1
    assert improvements == 15


def test_batch_size_does_not_change_census():
    assert census(4,batch_size=37) == census(4,batch_size=1024)


def test_calendar_reconstruction_preserves_distances_key_work_and_tail():
    days = (0,4,7,11,14,18,21,25)
    actual = records(8,days)
    before = exposures(8,ORIGIN)
    assert sorted(e.miles for e in actual) == sorted(e.miles for e in before)
    assert [e for e in actual if e.role != 'easy'] == [e for e in before if e.role != 'easy']
    assert len({e.day for e in actual}) == len(actual)


def test_witness_move_is_already_in_existing_stable_search_neighborhood():
    from scripts.calendar_neighbor_prototype import calendar_neighbors
    before = tuple((e.day-ORIGIN).days for e in exposures(8,ORIGIN) if (e.day-ORIGIN).days < 28)
    after = tuple(sorted((set(before)-{5})|{4}))
    assert after in calendar_neighbors(before,range(28))
    # Proposal coverage only: the old allocator/evaluator still need replacement.


@pytest.mark.parametrize('days', [(0,0,1,2), (0,1,2,3), (0,1,2,28)])
def test_invalid_calendar_rejected(days):
    with pytest.raises(ValueError):
        dose_vector(days,4)


def test_census_and_canonical_original_history_witness_agree_but_not_full_clearance():
    r = panel()
    assert sum(v['calendars_enumerated'] for v in r['results'].values()) == 130815
    eight = r['results']['8']['comparisons']
    assert eight['8']['all']['candidate_no_greater'] == 15
    assert eight['4']['all']['candidate_no_greater'] == 0
    assert r['canonical_opportunities_checked'] == 252
    assert r['canonical_easy_opportunities_checked'] == 68
    witness = next(c for c in r['canonical_checks'] if c['easy_count'] == c['reference_count'] == 8
                   and tuple(c['easy_days']) == (0,4,7,12,14,19,21,26))
    assert all(c['order'] in {'equal','candidate_no_greater'} for c in witness['receiving_opportunities'])
    moved = next(c for c in witness['easy_opportunity_comparisons'] if c['previous_day'] == 5)
    assert moved['candidate_day'] == 4 and moved['order'] == 'candidate_no_less'
    assert witness['constraints'] == 'passes_checked_constraints_conditionally'
    assert all(witness['aerobic_floor_passes'].values())
    assert not r['physiological_optimum_established']
