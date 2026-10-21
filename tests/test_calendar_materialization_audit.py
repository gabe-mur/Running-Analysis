from copy import deepcopy

from scripts.audit_calendar_materialization import admission_probe, witness_roles


def session(at, role):
    return dict(planned_for=at+'T07:00:00+00:00',workout_type=role)


def test_witness_preserves_consumed_dates_and_trims_same_absolute_horizon():
    report = dict(replans=[
        dict(decision_start_date='2026-09-22',planned_sessions=[
            session('2026-09-22','easy'),session('2026-09-24','long')]),
        dict(decision_start_date='2026-09-23',planned_sessions=[
            session('2026-09-24','easy'),session('2026-09-25','intervals'),
            session('2026-09-28','long'),session('2026-10-11','tempo_threshold'),
            session('2026-10-13','long')]),
    ])
    original = deepcopy(report)
    assert witness_roles(report) == {0:'easy',2:'easy',3:'quality',6:'long',19:'quality'}
    assert report == original


def row(offsets, price, rejected=False):
    return dict(offsets=offsets,total_cost=price,roles={},
                acceptance=dict(rejected=rejected),allocation_trace=[])


def test_admission_probe_uses_actual_shortlist_without_mutating_record():
    seed = row([0,2],1.)
    record = dict(evaluated=[seed,row([0,2,4],4.),row([0,3],5.)],
                  unconstrained=seed,stable=None)
    original = deepcopy(record)
    candidate = row([0,4],2.)
    result = admission_probe(record,candidate,candidate['acceptance'])
    assert result['hard_rank_if_added'] == 2
    assert result['admitted_if_added']
    assert not result['already_evaluated']
    assert result['finalist_budget'] == 6
    assert record == original


def test_rejected_candidate_cannot_claim_admission_even_with_best_price():
    record = dict(evaluated=[row([0,2],1.)])
    candidate = row([0,3],0.,True)
    result = admission_probe(record,candidate,candidate['acceptance'])
    assert result['hard_rank_if_added'] == 1
    assert not result['admitted_if_added']


def test_same_existing_calendar_is_reported_as_covered():
    candidate = row([0,3],2.)
    result = admission_probe(dict(evaluated=[candidate]),candidate,candidate['acceptance'])
    assert result['already_evaluated']
    assert not result['admitted_if_added']  # Deduplicated; existing row already represents it.
