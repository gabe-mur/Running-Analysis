from dataclasses import asdict,replace
from datetime import datetime,timedelta,timezone
from types import SimpleNamespace as NS
import json
import pytest

from run_analysis.adaptive.coached_runtime import sync,rebind_stage,restore_runtime
from run_analysis.adaptive.quality_assignment_accounting import (
    IssuedQualityAssignment,assignment_json,reconcile_assignment)
from run_analysis.adaptive.execution_upload_sandbox import encode,measured_work_distances
from run_analysis.adaptive.decision_snapshot import serialize,restore,evaluate_snapshot
from run_analysis.adaptive.coached_pipeline import connected_evaluator
from run_analysis.workload_accounting import Quantity
from test_quality_coaching_policy import sequence
from test_continuous_program_adapter import run
from test_quality_purpose_contract import contract,signals,prescription
from test_decision_snapshot import freeze,REF


def fixture(monkeypatch, *, matched=True, work=12.,rank=0):
    import run_analysis.adaptive.coached_runtime as m
    base=sequence();at=datetime.fromisoformat(base.issued_at)+timedelta(days=1)
    actual=replace(run(source='observed',work=work),start=at,end=at+timedelta(minutes=40),
        available_at=at+timedelta(minutes=40))
    a=IssuedQualityAssignment(base.stage_key,rank,base.assignment(rank).key,
        datetime.fromisoformat(base.issued_at),base.assignment(rank).purpose,'time',Quantity.point(720.,'test'))
    rows=[dict(session_key='slot:2026-10-10',fingerprint='issued',accounted=encode(actual))]
    h=NS(coaching_base=base,stage=NS(key=base.stage_key),
        sandbox=NS(completion_rows=lambda at:rows),quality_runtime={},events=[dict(
            stage_runtime=dict(stage=dict(key=base.stage_key)),
            active={'2026-10-10':dict(fingerprint='issued')},
            coached_assignments={'2026-10-10':assignment_json(a)})])
    monkeypatch.setattr(m,'completion_signals',lambda h,at:({actual.key:object()} if matched is not None else {},{}))
    monkeypatch.setattr(m,'read_contract',lambda x:None)
    monkeypatch.setattr(m,'assess_quality_purposes',lambda *a,**k:dict(credits={
        'sustained':Quantity.point(float(bool(matched)),'controlled purpose seam'),
        'faster':Quantity.point(float(bool(matched)),'controlled purpose seam')}))
    h.quality_runtime['purpose_contract']={}
    return h,rows,actual,at+timedelta(days=1)


def test_restart_retry_and_correction_rederive_from_immutable_issued_assignment(monkeypatch):
    h,rows,actual,at=fixture(monkeypatch,work=6.)
    sync(h,at);first=h.coaching_receipts
    assert h.coaching_sequence.consumed==1 and first[0].credit.low==.5
    sync(h,at+timedelta(days=1))
    assert h.coaching_sequence.consumed==1 and h.coaching_receipts==first
    # A corrected measured source can revoke completion; do not reuse old credit.
    corrected=replace(actual,role='easy',bouts=(),work_evidence_basis='no_supported_work',
        demand=(*actual.demand[:2],Quantity.point(0.,'corrected original evidence')))
    rows[0]['accounted']=encode(corrected)
    sync(h,at+timedelta(days=2))
    assert h.coaching_sequence.consumed==0 and h.coaching_receipts[0].credit.high==0.


@pytest.mark.parametrize('matched',[False,None])
def test_mismatch_and_unresolved_do_not_consume_variety(monkeypatch,matched):
    h,rows,actual,at=fixture(monkeypatch,matched=matched)
    sync(h,at)
    assert h.coaching_sequence==h.coaching_base
    assert h.coaching_receipts[0].credit.high==(0. if matched is False else 1.)


def test_miss_or_unowned_upload_does_not_consume_and_rank_cannot_skip(monkeypatch):
    h,rows,actual,at=fixture(monkeypatch,rank=1)
    sync(h,at);assert h.coaching_sequence.consumed==0
    rows.clear();sync(h,at);assert h.coaching_sequence==h.coaching_base


def test_stage_rebind_preserves_policy_and_rederives_receipts_across_both_stages(monkeypatch):
    from copy import copy
    import run_analysis.adaptive.coached_runtime as m
    h,rows,actual,at=fixture(monkeypatch)
    h.catalog={r.key:object() for r in h.coaching_base.ordinary}
    h.coaching_contracts={k:object() for k in h.catalog}
    sync(h,at)
    prior=h.coaching_sequence
    child=copy(h);child.stage=NS(key='accepted-child',issued_at=at)
    monkeypatch.setattr(m,'offline_guard',lambda h:None)
    rebind_stage(child,h,at)
    assert child.coaching_sequence==replace(prior,stage_key=child.stage.key)
    assert child.coaching_base is h.coaching_base and h.coaching_sequence==prior
    assert child.coaching_stage_keys==(h.stage.key,child.stage.key)
    later=replace(actual,key='later-quality',start=at+timedelta(days=1),
        end=at+timedelta(days=1,minutes=40),available_at=at+timedelta(days=1,minutes=40))
    next_command=IssuedQualityAssignment(child.stage.key,1,prior.assignment(0).key,
        datetime.fromisoformat(prior.issued_at),prior.assignment(0).purpose,'time',Quantity.point(720.,'test'))
    rows.append(dict(session_key='slot:2026-10-12',fingerprint='second',accounted=encode(later)))
    child.events=[*h.events,dict(stage_runtime=dict(stage=dict(key=child.stage.key)),
        active={'2026-10-12':dict(fingerprint='second')},
        coached_assignments={'2026-10-12':assignment_json(next_command)})]
    monkeypatch.setattr(m,'completion_signals',lambda h,at:({actual.key:object(),later.key:object()},{}))
    sync(child,later.available_at)
    assert child.coaching_sequence.consumed==2 and len(child.coaching_receipts)==2
    sync(child,later.available_at+timedelta(hours=1))
    assert child.coaching_sequence.consumed==2
    # A source correction before the stage change still revokes stale credit;
    # the child ordinal cannot skip the now-uncompleted first command.
    rows[0]['accounted']=encode(replace(actual,role='easy',bouts=(),work_evidence_basis='no_supported_work'))
    sync(child,later.available_at+timedelta(hours=2))
    assert child.coaching_sequence.consumed==0
    assert child.coaching_sequence.stage_key==child.stage.key


def test_stage_rebind_rejects_unreviewed_quality_changes_and_no_implicit_stage_skip(monkeypatch):
    from copy import copy
    import run_analysis.adaptive.coached_runtime as m
    h,rows,actual,at=fixture(monkeypatch)
    h.catalog={r.key:object() for r in h.coaching_base.ordinary}
    h.coaching_contracts={k:object() for k in h.catalog}
    sync(h,at)
    child=copy(h);child.stage=NS(key='child',issued_at=at)
    child.catalog=dict(h.catalog);child.catalog.pop(h.coaching_base.ordinary[0].key)
    monkeypatch.setattr(m,'offline_guard',lambda h:None)
    with pytest.raises(ValueError,match='preserve issued quality'):
        rebind_stage(child,h,at)
    with pytest.raises(ValueError,match='explicit command rebinding'):
        sync(child,at)


def test_restore_stage_lineage_requires_atomic_ledger_and_unchanged_policy(monkeypatch):
    import sqlite3
    import run_analysis.adaptive.coached_runtime as m
    db=sqlite3.connect(':memory:');db.row_factory=sqlite3.Row
    db.executescript('CREATE TABLE stages(key TEXT,parent TEXT);CREATE TABLE current_stage(singleton INTEGER,key TEXT);')
    base=sequence();current=replace(base,stage_key='child',consumed=2)
    db.executemany('INSERT INTO stages VALUES(?,?)',[(base.stage_key,None),('child',base.stage_key)])
    db.execute('INSERT INTO current_stage VALUES(1,"child")')
    h=NS(stage=NS(key='child'),sandbox=NS(db=db))
    runtime=dict(coaching_base=asdict(base),coaching_stage_keys=(base.stage_key,'child'))
    state=NS(quality_coaching=current,quality_completions=())
    monkeypatch.setattr(m,'offline_guard',lambda h:None)
    try:
        restore_runtime(h,runtime,state)
        assert h.coaching_stage_keys==(base.stage_key,'child') and h.coaching_sequence==current
        with pytest.raises(ValueError,match='frozen quality policy'):
            restore_runtime(h,runtime,NS(quality_coaching=replace(current,first_purpose='faster'),quality_completions=()))
        db.execute('UPDATE stages SET parent="unrelated" WHERE key="child"')
        with pytest.raises(ValueError,match='atomic accepted-stage ledger'):
            restore_runtime(h,runtime,state)
    finally:db.close()


def test_raw_purpose_distinguishes_short_fast_and_sustained_without_using_intent():
    from run_analysis.adaptive.quality_purpose_contract import assess_quality_purposes
    at=datetime(2026,10,8,tzinfo=timezone.utc)
    short=signals(prescription(),observed=True,hrr=.65,speed=3.)
    r=assess_quality_purposes(short,contract=contract(),role='quality',as_of=at)
    assert r['credits']['faster'].low>0 and r['credits']['sustained'].high==0
    long=signals(prescription('threshold',600,1),observed=True,hrr=.85,speed=2.)
    r=assess_quality_purposes(long,contract=contract(),role='quality',as_of=at)
    assert r['credits']['sustained'].low>0 and r['credits']['faster'].high==0


def test_observed_work_distance_integrates_only_its_window_and_preserves_gaps():
    at=datetime(2026,10,9,tzinfo=timezone.utc)
    def interval(a,b,m,flags=()):return NS(start=NS(timestamp_utc=at+timedelta(seconds=a)),
        end=NS(timestamp_utc=at+timedelta(seconds=b)),elapsed_s=b-a,moving_time_s=b-a,
        distance_source='recorded',distance_m=m,flags=flags)
    values=(interval(0,60,120),interval(60,120,400),interval(120,180,100))
    phases=(('warmup',60.),('work',60.),('cooldown',60.))
    assert measured_work_distances(values,at,phases)==(400.,)
    assert measured_work_distances((values[0],values[2]),at,phases)==(None,)
    assert measured_work_distances((values[0],interval(60,120,400,('distance_reset',)),values[2]),at,phases)==(None,)


def test_absent_coaching_fields_do_not_change_old_snapshot_identity():
    state=freeze();raw=serialize(state)
    assert 'quality_coaching' not in json.loads(raw)['state']['fields']
    assert restore(raw,expected_fingerprint=state.fingerprint)==state


@pytest.mark.parametrize('focus', ['variety', 'vo2max'])
def test_coached_snapshot_roundtrip_dispatches_the_same_full_evaluator(focus):
    from test_r3_decision_controls import row
    state=freeze()
    seq=replace(sequence(focus=focus),stage_key=state.stage_key,issued_at=state.as_of.isoformat())
    reference=('quality',3.5,tuple(Quantity.point(v,'frozen fixture reference') for v in (35.,45.,10.)))
    state=replace(state,quality_coaching=seq,quality_coaching_reference=reference)
    actual=row(2,4.5,'quality')
    planned=(replace(actual,key='candidate:'+actual.start.date().isoformat()+':'+seq.assignment(0).key),)
    expected=evaluate_snapshot(state,planned,('quality',),reference=REF,
        evaluator=connected_evaluator(seq,reference))
    restored=restore(serialize(state),expected_fingerprint=state.fingerprint)
    assert evaluate_snapshot(restored,planned,('quality',),reference=REF)==expected
    assert restored.quality_coaching==seq
