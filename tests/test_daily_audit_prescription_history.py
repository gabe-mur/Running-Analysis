from types import SimpleNamespace

from scripts.audit_adaptive_daily_replay import visible_dose_history


def fixture(values,*,day='2026-10-08',decisions=None):
    events=[dict(index=i,decision=decisions[i] if decisions else '2026-10-06',
        program=((day,str(value)),) if value is not None else ()) for i,value in enumerate(values)]
    catalogs={i:{str(value):SimpleNamespace(role='easy',miles=value)} if value is not None else {}
        for i,value in enumerate(values)}
    return events,catalogs


def test_small_adjacent_edits_do_not_hide_accumulated_change():
    events,catalogs=fixture([3.,3.4,3.7])
    result=visible_dose_history(events,catalogs,7)
    assert round(result['largest_net_dose_edit'],8)==.7
    assert round(result['largest_dose_span'],8)==.7


def test_oscillation_is_visible_even_when_final_target_matches_original():
    events,catalogs=fixture([3.,3.4,3.])
    result=visible_dose_history(events,catalogs,7)
    assert result['largest_net_dose_edit']==0.
    assert round(result['largest_dose_span'],8)==.4
    assert round(result['total_dose_movement'],8)==.8


def test_far_tail_and_expired_completion_are_not_visible_changes():
    events,catalogs=fixture([3.,4.,None],day='2026-10-13',
        decisions=['2026-10-06','2026-10-07','2026-10-14'])
    result=visible_dose_history(events,catalogs,7)
    assert result['changed_dates']==[]
