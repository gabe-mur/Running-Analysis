"""Original FIT enrichment, causal versions and exactly-once physical ownership."""
from dataclasses import replace
from datetime import datetime,timedelta,timezone
from hashlib import sha256
import json

import pytest
from fit_builder import build_run
from test_execution_upload_sandbox import tcx,START,setup
from test_adaptive_stage_lifecycle import completed_stage
from test_adaptive_service import service_fixture
from run_analysis.config import load_config
from run_analysis.db import connect,initialize
from run_analysis.importer import import_files
from run_analysis.adaptive.live_bridge import app_sources
from run_analysis.adaptive.seed_reconciliation import reconcile_source,phase_records
from run_analysis.adaptive.source_reconciliation import verified_replacement,reconcile_sources
from run_analysis.adaptive.new_model_daily_harness import DailyHarness


def originals(tmp_path,config):
    original=tmp_path/'run.tcx';original.write_bytes(tcx().content)
    richer=tmp_path/'run.fit'
    build_run(richer,start=START,points=31,metres_per_point=.4*1609.344/30,heart_rate=150)
    db=connect(tmp_path/'app.sqlite');initialize(db)
    import_files(db,tmp_path,paths=[original],default_timezone=config['timezone_default'])
    activity_id=str(db.execute('SELECT id FROM activities').fetchone()[0])
    digest=app_sources(db)[activity_id]['sha256']
    summary=import_files(db,tmp_path,paths=[richer],default_timezone=config['timezone_default'])
    assert summary.duplicate_activities==1 and summary.evidence_changed_activity_ids==[int(activity_id)]
    return db,activity_id,digest,richer


def test_real_original_fit_verified_without_conversion(tmp_path):
    config=load_config();db,key,digest,path=originals(tmp_path,config)
    try:
        payload,proof=verified_replacement(db,key,digest,app_sources(db)[key],config,datetime.now(timezone.utc))
        assert payload.content==path.read_bytes() and payload.filename.endswith('.fit')
        assert proof['old_sha256']==digest and proof['new_sha256']==sha256(payload.content).hexdigest()
        assert db.execute('SELECT COUNT(*) FROM activities').fetchone()[0]==1
    finally:db.close()


@pytest.mark.parametrize('bad',['tamper','arbitrary','removed','future','different_distance','different_start'])
def test_unverified_source_changes_still_require_review(tmp_path,bad):
    config=load_config();db,key,digest,path=originals(tmp_path,config)
    at=datetime.now(timezone.utc)
    try:
        if bad=='tamper':path.write_bytes(b'tampered source')
        elif bad=='arbitrary':db.execute('UPDATE activity_sources SET duplicate_reason=NULL WHERE is_primary=1');db.commit()
        elif bad=='removed':digest='a'*64
        elif bad=='future':at-=timedelta(days=1)
        else:
            build_run(path,start=START+timedelta(seconds=1) if bad=='different_start' else START,
                points=31,metres_per_point=(.8 if bad=='different_distance' else .4)*1609.344/30)
            db.execute('UPDATE source_files SET sha256=? WHERE path=?',(sha256(path.read_bytes()).hexdigest(),str(path.resolve())));db.commit()
        with pytest.raises(ValueError):
            verified_replacement(db,key,digest,app_sources(db)[key],config,at)
    finally:db.close()


def test_seed_fit_correction_atomic_dated_and_restart_safe(tmp_path,monkeypatch):
    h,_,config,at=completed_stage(tmp_path,monkeypatch)
    try:
        old=next(r for r in h.history if r.key in h.seed_keys and r.role=='easy')
        phase=next(p for p in h.phase_history if p.key==old.key)
        new=replace(old,demand=(old.demand[0].scale(.99),old.demand[1],old.demand[2]))
        before=[tuple(r) for r in h.sandbox.db.execute('SELECT * FROM sandbox_decisions')]
        h.sandbox.db.execute('CREATE TABLE adaptive_app_sources(activity_id TEXT PRIMARY KEY,physical_key TEXT UNIQUE,source_sha256 TEXT)')
        h.sandbox.db.execute('INSERT INTO adaptive_app_sources VALUES (?,?,?)',(old.key,old.key,'a'*64));h.sandbox.db.commit()
        proof=dict(activity_id=old.key,old_sha256='a'*64,new_sha256='b'*64,duplicate_reason='richer_duplicate_replaced_canonical')
        at+=timedelta(seconds=1)
        # A format upgrade cannot also authorize an unreconciled manual title.
        with pytest.raises(ValueError,match='Manual role change'):
            reconcile_source(h,replace(new,role='quality'),phase,activity_id=old.key,
                old_digest='a'*64,new_digest='b'*64,available_at=at,
                proof=dict(proof,manual_workout_type='intervals'))
        # A concurrent/stale predecessor must roll back both revision and hash.
        with pytest.raises(ValueError,match='predecessor'):
            reconcile_source(h,new,phase,activity_id=old.key,old_digest='c'*64,new_digest='b'*64,
                available_at=at,proof=dict(proof,old_sha256='c'*64))
        assert h.sandbox.db.execute('SELECT COUNT(*) FROM sandbox_seed_revisions').fetchone()[0]==0
        assert h.sandbox.db.execute('SELECT source_sha256 FROM adaptive_app_sources').fetchone()[0]=='a'*64
        reconcile_source(h,new,phase,activity_id=old.key,old_digest='a'*64,new_digest='b'*64,available_at=at,proof=proof)
        assert len(h.history)==len(set(r.key for r in h.history))
        assert h.sandbox.db.execute('SELECT source_sha256 FROM adaptive_app_sources').fetchone()[0]=='b'*64
        assert [tuple(r) for r in h.sandbox.db.execute('SELECT * FROM sandbox_decisions')]==before
        assert json.loads(phase_records(h.sandbox,at-timedelta(microseconds=1))[old.key])['available_at']==phase.available_at.isoformat()
        root=h.sandbox.root;h.sandbox.close()
        earlier=DailyHarness.resume(root,config,as_of=at-timedelta(microseconds=1))
        try:assert next(r for r in earlier.history if r.key==old.key)==old
        finally:earlier.sandbox.close()
        later=DailyHarness.resume(root,config,as_of=at)
        try:
            assert next(r for r in later.history if r.key==old.key)==replace(new,available_at=at)
            later.sync_completions(at+timedelta(seconds=1))
            assert len(later.history)==len(h.history)
            assert later.sandbox.db.execute('SELECT COUNT(*) FROM sandbox_seed_revisions').fetchone()[0]==1
        finally:later.sandbox.close()
    finally:h.sandbox.close()


def test_completed_fit_upgrade_preserves_issue_load_once_and_retry(setup,tmp_path):
    service,day=service_fixture(setup);h=service.harness
    at=datetime.now(timezone.utc)
    first=service.upload(tcx(),day=day,available_at=at-timedelta(seconds=2),replan=False)
    key=first['upload']['comparison']['actual_key'];original=tuple(h.history)
    h.seed_keys=set()
    db,activity_id,digest,_=originals(tmp_path,h.config)
    at=datetime.now(timezone.utc)
    try:
        h.sandbox.db.execute('CREATE TABLE adaptive_app_sources(activity_id TEXT PRIMARY KEY,physical_key TEXT UNIQUE,source_sha256 TEXT)')
        h.sandbox.db.execute('INSERT INTO adaptive_app_sources VALUES (?,?,?)',(activity_id,key,digest));h.sandbox.db.commit()
        known={activity_id:(key,digest)}
        reconcile_sources(service,db,app_sources(db),known,at)
        assert len(h.history)==1 and h.history[0].key==key
        assert h.history[0].demand[1].low>original[0].demand[1].low
        assert h.sandbox.completed_accounted(at-timedelta(seconds=1))[0][1]==original[0]
        assert h.sandbox.completion_rows(at)[0]['fingerprint']==first['upload']['comparison']['issued_fingerprint']
        # Simulated interruption after completion revision, before bridge hash update.
        h.sandbox.db.execute('UPDATE adaptive_app_sources SET source_sha256=?',(digest,));h.sandbox.db.commit()
        reconcile_sources(service,db,app_sources(db),known,at+timedelta(seconds=1))
        assert h.sandbox.db.execute('SELECT COUNT(*) FROM sandbox_completion_versions').fetchone()[0]==1
        assert len(h.history)==1
        tracked={r[0]:(r[1],r[2]) for r in h.sandbox.db.execute('SELECT * FROM adaptive_app_sources')}
        reconcile_sources(service,db,app_sources(db),tracked,at+timedelta(seconds=2))
        assert h.sandbox.db.execute('SELECT COUNT(*) FROM sandbox_completion_versions').fetchone()[0]==1
    finally:db.close()
