from copy import deepcopy

import pytest

from scripts.audit_backward_same_origin import aligned, coverage
from scripts.continuation_replan import fingerprint
from test_typed_continuation import fixture


def frame():
    return dict(origin='2026-09-23', hard_end_exclusive='2026-10-14',
        full_end_exclusive='2026-11-11', field_hashes={k:'same' for k in (
            'completed_history','config','daily_states','daily_state_options','prior_prescriptions')})


@pytest.mark.parametrize('key', ['origin','hard_end_exclusive','full_end_exclusive'])
def test_same_origin_rejects_moving_boundaries(key):
    before, after = frame(), frame()
    after[key] = 'different'
    with pytest.raises(ValueError, match='boundaries'):
        aligned(before, after)


def test_changed_prior_is_allowed_but_changed_completed_evidence_is_not():
    before, after = frame(), frame()
    after['field_hashes']['prior_prescriptions'] = 'changed'
    aligned(before, after)
    after['field_hashes']['completed_history'] = 'changed'
    with pytest.raises(ValueError, match='history'):
        aligned(before, after)


def coverage_fixture():
    _, make = fixture()
    full, _ = make(49, {0:('easy',3.5), 25:('easy',4.)})
    hard = full[:21]
    row = dict(offsets=[0], hard_fingerprint=fingerprint(hard),
        tail_fingerprint=fingerprint(full[21:]), combined_cost=1.)
    report = dict(replans=[{'planner_diagnostics': {'continuation_selection': {'finalists':[row]}}}],
        synthetic_sandbox={'calendar_neighbor_prototype':[{'evaluated':[{'offsets':[0]}]}]})
    return report, hard, full


def test_calendar_match_does_not_claim_exact_distance_or_tail_was_scored():
    report, hard, full = coverage_fixture()
    row = report['replans'][0]['planner_diagnostics']['continuation_selection']['finalists'][0]
    row['hard_fingerprint'] = 'different distances'
    actual = coverage(report,0,hard,full)
    assert len(actual['finalist_calendar_rows']) == 1
    assert actual['exact_hard_rows'] == []
    assert not actual['exact_pair_scored']


def test_exact_hard_match_with_other_tail_is_not_full_program_coverage():
    report, hard, full = coverage_fixture()
    report['replans'][0]['planner_diagnostics']['continuation_selection']['finalists'][0]['tail_fingerprint'] = 'other'
    actual = coverage(report,0,hard,full)
    assert len(actual['exact_hard_rows']) == 1
    assert not actual['exact_pair_scored']


def test_exact_scored_pair_is_reported_without_modifying_report():
    report, hard, full = coverage_fixture()
    before = deepcopy(report)
    assert coverage(report,0,hard,full)['exact_pair_scored']
    assert report == before
