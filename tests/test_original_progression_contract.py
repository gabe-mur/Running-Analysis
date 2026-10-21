"""Causal counterfactual accounting; no planner or physiological acceptance."""
from dataclasses import replace
from datetime import datetime,timedelta,timezone

import pytest

from scripts.audit_original_progression_contract import (
    IssuedRun,issue_event,nominal,course,frozen_publication,evidence_summary)
from run_analysis.adaptive.historical_progression import Exposure
from run_analysis.stage_target_history import StageTarget

AT=datetime(2026,10,3,tzinfo=timezone.utc)


def intent(issued=AT,start=None,miles=2.):
    return dict(issued_at=issued.isoformat(),suggested_start=(start or issued).isoformat(),
        distance_miles=dict(target=miles),phases=[dict(seconds=dict(target=1200.))])


def actual(key='measured',start=AT,miles=3.):
    return Exposure(key,start,start+timedelta(hours=1),start+timedelta(hours=2),miles,'easy','original')


def seed():
    return tuple(actual(str(n),AT-timedelta(days=n*2+1),2.) for n in range(20))


def targets():
    return (StageTarget('known',AT,AT.date()-timedelta(days=50),5.,2.,'explicit test lead-in'),)


def test_issuance_selects_only_then_known_target_version():
    before=dict(as_of=AT.isoformat(),index=1)
    after=dict(as_of=(AT+timedelta(days=1)).isoformat(),index=2)
    assert issue_event((after,before),AT)['index']==1
    with pytest.raises(ValueError):issue_event((after,),AT)


def test_exact_distance_retains_original_clock_and_availability_not_fake_hr():
    observed=actual()
    run=IssuedRun('measured','2026-10-03','fingerprint',intent(), 'easy',observed)
    changed=nominal(run,'recorded_clock')
    assert changed.miles==2 and observed.miles==3
    assert (changed.start,changed.end,changed.available_at)==(observed.start,observed.end,observed.available_at)
    assert changed.provenance.startswith('counterfactual_')
    assert not hasattr(changed,'demand')


def test_scheduled_hypothesis_cannot_start_before_late_issuance():
    i=intent(AT+timedelta(hours=2),AT)
    changed=nominal(IssuedRun('x','2026-10-03','f',i,'easy',None),'scheduled_clock')
    assert changed.start==AT+timedelta(hours=2)
    assert changed.end==changed.start+timedelta(seconds=1200)
    assert changed.available_at==changed.end


def test_retrospective_nominal_is_not_known_execution():
    run=IssuedRun('measured','2026-10-03','f',intent(AT+timedelta(hours=2)),'easy',actual())
    with pytest.raises(ValueError,match='not known'):nominal(run,'recorded_clock')


def test_counterfactual_replaces_actual_once_and_preserves_unprescribed_history():
    observed=actual();older=seed()
    run=IssuedRun('measured','2026-10-03','f',intent(),'easy',observed)
    records=course((*older,observed),(run,),AT+timedelta(hours=3),'recorded_clock')
    assert len(records)==len(older)+1
    assert next(r for r in records if r.key=='measured').miles==2
    assert all(r in records for r in older)


def test_omitted_opportunity_is_not_counted_until_hypothetical_completion():
    run=IssuedRun('omission','2026-10-03','f',intent(),'easy',None)
    assert course(seed(),(run,),AT,'scheduled_clock')==tuple(sorted(seed(),key=lambda r:(r.start,r.key)))
    complete=course(seed(),(run,),AT+timedelta(hours=1),'scheduled_clock')
    assert len(complete)==len(seed())+1
    assert len(course(seed(),(run,),AT+timedelta(hours=1),'actual'))==len(seed())


def test_late_actual_upload_is_not_early_counterfactual_evidence():
    observed=actual();run=IssuedRun('measured','2026-10-03','f',intent(),'easy',observed)
    records=course(seed(),(run,),observed.end,'recorded_clock')
    assert all(r.key!='measured' for r in records)


def publication():
    event=dict(index=1,as_of=AT.isoformat(),decision='2026-10-03',hard_end='2026-10-05',volume=4.,
        active={'2026-10-03':dict(fingerprint='a',dose='easy:2'),
                '2026-10-04':dict(fingerprint='b',dose='easy:2')})
    issues={'a':intent(start=AT+timedelta(hours=12)),
            'b':intent(start=AT+timedelta(days=1,hours=12))}
    return event,issues


def test_one_frozen_publication_excludes_later_actuals_and_target_revisions():
    event,issues=publication()
    future=actual('future',AT+timedelta(days=1),1000.)
    future_target=StageTarget('future',AT+timedelta(days=1),(AT+timedelta(days=1)).date(),1000.,500.,'later')
    baseline=frozen_publication(event,issues,seed(),targets(),timezone.utc)
    after=frozen_publication(event,issues,(*seed(),future),(*targets(),future_target),timezone.utc)
    assert baseline==after
    assert baseline['issued_miles']==4
    assert baseline['hypothetical_runs']==2


def test_full_publication_detects_missing_or_wrong_frozen_issuance():
    event,issues=publication()
    with pytest.raises(ValueError,match='disagree'):
        frozen_publication(dict(event,volume=5),issues,seed(),targets(),timezone.utc)


def test_growth_reference_sensitivity_changes_only_current_level_not_training():
    before=seed()
    e=evidence_summary(before,targets(),AT,timezone.utc,(5.,6.))
    assert list(e['growth_at_levels'])==['5.0','6.0']
    assert all(p['physical_credit_added']==0 for p in e['growth_at_levels'].values())
    assert seed()==before


def test_nominal_course_rejects_duplicate_physical_receipts():
    run=IssuedRun('measured','2026-10-03','f',intent(),'easy',actual())
    with pytest.raises(ValueError,match='twice'):
        course(seed(),(run,run),AT+timedelta(days=1),'recorded_clock')


def test_future_actual_completion_is_not_used_by_earlier_trace():
    observed=actual(start=AT+timedelta(days=1))
    run=IssuedRun('measured','2026-10-04','f',intent(),'easy',observed)
    assert all(r.key!='measured' for r in course(seed(),(run,),AT,'actual'))
