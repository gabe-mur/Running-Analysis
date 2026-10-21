"""Count admission is separate from objective weights; no biological claims."""
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

import pytest

from run_analysis.adaptive.frequency_range_trial import FrequencyContract, compile_frequency_exposure


START = date(2026, 9, 29)


def frame(contract, offset=0, observed=()):
    decision = START+timedelta(days=offset)
    return compile_frequency_exposure(contract, observed, decision=decision,
        coverage_end=decision+timedelta(days=54), taper_start=decision+timedelta(days=40))


def contract():
    return FrequencyContract.freeze(adopted_at=datetime(2026, 9, 29, tzinfo=timezone.utc),
        stage_key='seed', effective_on=START, volume_per_7=19.3,
        reference_fatigue=.042, reference_days=54)


def test_no_evidence_rollover_does_not_change_count_admission_of_same_future_dates():
    c = contract()
    planned = tuple(START+timedelta(days=i) for i in range(2, 38))
    audits = [frame(c, i).assess(planned) for i in range(3)]
    assert all(a['eligible'] for a in audits)
    assert all(a['blocks'][0]['admission_count'] == 36 for a in audits)
    assert all(a['blocks'][0]['admission_upper_bound'] == pytest.approx(36.12100840336136) for a in audits)
    assert not frame(c).assess((*planned, START+timedelta(days=40)))['eligible']


def test_discounted_tail_is_not_discounted_for_admission():
    c = contract()
    planned = tuple(START+timedelta(days=i) for i in range(17, 54))
    old = frame(replace(c, version='frequency_midpoint_v2')).assess(planned)
    new = frame(c).assess(planned)
    assert old['eligible'] and not new['eligible']
    assert new['blocks'][0]['planned_weighted_count'] < 37
    assert new['blocks'][0]['admission_count'] == 37
    assert c.preference.assess(new)['cost'] == c.preference.assess(old)['cost']


def test_upload_and_missed_day_cannot_create_frequency_repayment_or_past_veto():
    c = contract()
    observed = tuple(START+timedelta(days=i) for i in range(28))
    f = frame(c, 28, observed)
    assert f.assess(())['eligible']
    assert f.assess((START+timedelta(days=29),))['eligible']
    assert f.assess((START+timedelta(days=28),))['blocks'][0]['admission_count'] == 1


def test_upper_changes_only_with_covered_duration_or_dated_stage_rate():
    c = contract()
    changed = c.accepted_stage('next', START+timedelta(days=14), 22.)
    assert frame(changed, 14).unweighted_future_upper > frame(c, 14).unweighted_future_upper
    assert changed.preference == c.preference

