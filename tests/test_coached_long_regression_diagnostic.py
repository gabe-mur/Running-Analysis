from types import SimpleNamespace
import pytest

from scripts.audit_coached_long_regression import mapped_calendar
from test_quality_coaching_policy import sequence


def test_live_mapping_preserves_dates_roles_and_nonquality_dose():
    source=dict(event=dict(program=[('2026-10-09','old-quality'),('2026-10-10','easy:2.3'),
        ('2026-10-12','long:8.9'),('2026-10-16','old-quality')]),catalog={
        'old-quality':SimpleNamespace(role='quality'),
        'easy:2.3':SimpleNamespace(role='easy'),'long:8.9':SimpleNamespace(role='long')})
    target=dict(dates=('2026-10-09','2026-10-10','2026-10-12','2026-10-16'),
        handoffs={'easy:2.3':None,'long:8.9':None})
    mapped=mapped_calendar(source,target,sequence())
    assert mapped==(('2026-10-09','threshold10'),('2026-10-10','easy:2.3'),
        ('2026-10-12','long:8.9'),('2026-10-16','short3'))


def test_calendar_comparison_rejects_a_different_horizon():
    source=dict(event=dict(program=[('2026-10-09','long:8.9')]),
        catalog={'long:8.9':SimpleNamespace(role='long')})
    with pytest.raises(ValueError,match='same dated horizon'):
        mapped_calendar(source,dict(dates=('2026-10-10',),handoffs={}),sequence())


def test_calendar_comparison_does_not_guess_an_unavailable_nonquality_dose():
    source=dict(event=dict(program=[('2026-10-09','easy:3.3')]),
        catalog={'easy:3.3':SimpleNamespace(role='easy')})
    with pytest.raises(ValueError,match='nonquality dose absent'):
        mapped_calendar(source,dict(dates=('2026-10-09',),handoffs={'easy:3.4':None}),sequence())
