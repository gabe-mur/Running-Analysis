"""Visibility bookkeeping, not a new stability preference or scorer."""
from types import SimpleNamespace

from scripts.audit_adaptive_daily_replay import visible_dose_history


def event(index,day,program):
    return dict(index=index,decision=day,program=program)


def test_new_boundary_day_is_not_a_changed_visible_promise():
    events=[event(0,'2026-10-08',[('2026-10-15','easy')]),
        event(1,'2026-10-09',[])]
    catalogs={i:{'easy':SimpleNamespace(role='easy',miles=3.)} for i in range(2)}
    result=visible_dose_history(events,catalogs,7)
    assert result['date_edits']==0 and result['type_edits']==0
    assert result['changed_dates']==[]


def test_visible_date_type_and_dose_changes_are_not_hidden():
    events=[event(0,'2026-10-08',[('2026-10-12','easy'),('2026-10-13','easy')]),
        event(1,'2026-10-09',[('2026-10-13','quality'),('2026-10-14','easy')]),
        event(2,'2026-10-10',[('2026-10-13','quality'),('2026-10-14','larger')])]
    catalog={'easy':SimpleNamespace(role='easy',miles=3.),
        'larger':SimpleNamespace(role='easy',miles=3.4),
        'quality':SimpleNamespace(role='quality',miles=4.)}
    result=visible_dose_history(events,{i:catalog for i in range(3)},7)
    assert result['date_edits']==2  # One removal, one addition on already-visible dates.
    assert result['type_edits']==1
    assert abs(result['total_dose_movement']-.4)<1e-8


def test_completed_day_expiry_is_not_a_calendar_edit():
    events=[event(0,'2026-10-08',[('2026-10-08','easy')]),event(1,'2026-10-09',[])]
    catalog={'easy':SimpleNamespace(role='easy',miles=3.)}
    result=visible_dose_history(events,{0:catalog,1:catalog},7)
    assert not result['date_edits'] and not result['changed_dates']
