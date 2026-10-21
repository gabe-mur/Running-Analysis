"""Dated seed correction ownership; not new search/recovery acceptance."""
from dataclasses import replace
from datetime import timedelta
from hashlib import sha256
import json
import pytest

from run_analysis.adaptive.seed_reconciliation import reconcile,phase_records
from run_analysis.adaptive.execution_upload_sandbox import encode
from run_analysis.adaptive.new_model_daily_harness import DailyHarness
from run_analysis.adaptive.live_bridge import apply_app_inputs
from run_analysis.adaptive.decision_snapshot import CapturedAppInputs
from run_analysis.delivered_training import WorkBout
from run_analysis.workload_accounting import Quantity
from test_adaptive_stage_lifecycle import completed_stage


def correction(h):
    old=next(r for r in reversed(h.history) if r.key in h.seed_keys and r.role=='easy')
    phase=next(p for p in h.phase_history if p.key==old.key)
    seconds=sum(p.seconds for p in phase.phases)
    new=replace(old,role='quality',bouts=(WorkBout(seconds,None,old.miles*1609.344),),
        demand=(*old.demand[:2],Quantity.point(seconds/60,'recorded_continuous_lap')),
        quality=Quantity.point(1,'fixture recorded work'),work_evidence_basis='recorded_continuous_lap')
    newphase=replace(phase,phases=tuple(replace(p,kind='work') for p in phase.phases))
    return old,phase,new,newphase


def test_seed_correction_dated_restart_no_double_load_or_rewritten_issuance(tmp_path,monkeypatch):
    h,_,config,at=completed_stage(tmp_path,monkeypatch)
    try:
        old,oldphase,new,newphase=correction(h)
        original_decisions=[tuple(r) for r in h.sandbox.db.execute('SELECT * FROM sandbox_decisions')]
        original_issues=[tuple(r) for r in h.sandbox.db.execute('SELECT * FROM sandbox_issues')]
        original_seed=[tuple(r) for r in h.sandbox.db.execute('SELECT * FROM sandbox_seed_history')]
        total=sum(r.demand[1].low for r in h.history)
        count=len(h.history)
        at+=timedelta(seconds=1)
        captured=CapturedAppInputs(at,(),h.restrictions,h.responses,((old.key,True,'tempo_threshold'),))
        with pytest.raises(ValueError,match='physical-evidence reconciliation'):
            apply_app_inputs(h,captured)
        result=reconcile(h,new,newphase,expected_fingerprint=sha256(encode(old).encode()).hexdigest(),
            available_at=at,source_sha256='a'*64,reason='explicit fixture recorded work repair')
        apply_app_inputs(h,captured)
        h.sync_completions(at)
        h.sync_completions(at+timedelta(seconds=1))
        assert len(h.history)==count and sum(r.demand[1].low for r in h.history)==total
        assert next(r for r in h.history if r.key==old.key).role=='quality'
        assert original_decisions==[tuple(r) for r in h.sandbox.db.execute('SELECT * FROM sandbox_decisions')]
        assert original_issues==[tuple(r) for r in h.sandbox.db.execute('SELECT * FROM sandbox_issues')]
        assert original_seed==[tuple(r) for r in h.sandbox.db.execute('SELECT * FROM sandbox_seed_history')]
        assert json.loads(phase_records(h.sandbox,at-timedelta(microseconds=1))[old.key])['phases'][0]['kind']=='warmup'
        assert json.loads(phase_records(h.sandbox,at)[old.key])['phases'][0]['kind']=='work'
        root=h.sandbox.root
        h.sandbox.close()
        before=DailyHarness.resume(root,config,as_of=at-timedelta(microseconds=1))
        try:
            assert next(r for r in before.history if r.key==old.key)==old
        finally: before.sandbox.close()
        resumed=DailyHarness.resume(root,config,as_of=at)
        try:
            assert next(r for r in resumed.history if r.key==old.key)==replace(new,available_at=at)
            assert next(p for p in resumed.phase_history if p.key==old.key)==replace(newphase,available_at=at)
            event=resumed.replan(at+timedelta(seconds=1),reason='fixture after explicit repair')
            assert event['index']==1
            assert resumed.sandbox.db.execute('SELECT COUNT(*) FROM sandbox_seed_revisions').fetchone()[0]==1
        finally: resumed.sandbox.close()
        latest=DailyHarness.resume(root,config,as_of=at+timedelta(seconds=1))
        try:
            assert next(r for r in latest.history if r.key==old.key).role=='quality'
            assert len(latest.history)==count
        finally: latest.sandbox.close()
    finally: h.sandbox.close()


@pytest.mark.parametrize('bad', ['identity','cardio','title_only','backdate','predecessor','phase_load'])
def test_seed_repair_rejects_unreviewed_physical_changes(tmp_path,monkeypatch,bad):
    h,_,_,at=completed_stage(tmp_path,monkeypatch)
    try:
        old,_,new,phase=correction(h)
        expected=sha256(encode(old).encode()).hexdigest()
        if bad=='identity': new=replace(new,miles=new.miles+.1)
        elif bad=='cardio': new=replace(new,demand=(new.demand[0],new.demand[1].scale(2),new.demand[2]))
        elif bad=='title_only': new=replace(new,bouts=(),work_evidence_basis='no_supported_work')
        elif bad=='backdate': at=h.last_as_of
        elif bad=='predecessor': expected='wrong'
        elif bad=='phase_load': phase=replace(phase,phases=(replace(phase.phases[0],load=phase.phases[0].load+1),))
        with pytest.raises(ValueError):
            reconcile(h,new,phase,expected_fingerprint=expected,available_at=at,
                source_sha256='a'*64,reason='explicit fixture')
        assert h.sandbox.db.execute('SELECT COUNT(*) FROM sandbox_seed_revisions').fetchone()[0]==0
        assert next(r for r in h.history if r.key==old.key)==old
    finally: h.sandbox.close()
