from datetime import date
import pytest
from scripts.audit_terminal_continuation import repeat_suffix


def test_suffix_is_candidate_owned_and_repeats_rest_days_without_fake_runs():
    p = (('2026-10-01','easy'),('2026-10-03','long'))
    assert repeat_suffix(p,end=date(2026,10,4),cycle_days=3,extra_days=4)==(
        *p,('2026-10-04','easy'),('2026-10-06','long'),('2026-10-07','easy'))
    assert repeat_suffix(p,end=date(2026,10,4),cycle_days=3,extra_days=0)==p
    with pytest.raises(ValueError):
        repeat_suffix(p,end=date(2026,10,4),cycle_days=0,extra_days=4)
