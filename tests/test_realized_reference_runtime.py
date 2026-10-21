"""Actual allocator/publication/restart controls; NOT a 21-day search pass."""
from dataclasses import replace
from datetime import datetime, timedelta
import json

import pytest

from run_analysis.adaptive.smoothed_general_progression import REALIZED_VERSION
from run_analysis.adaptive.adaptive_stage_lifecycle import advance_stage, review_entries
from run_analysis.adaptive.new_model_daily_harness import DailyHarness, HarnessPolicy, synthetic_payload
from run_analysis.adaptive.realized_reference_runtime import initialize, signal
from run_analysis.adaptive.decision_snapshot import restore
from test_smoothed_general_progression import fixture


def test_new_path_requires_explicit_observed_credit_and_keeps_default_off():
    assert HarnessPolicy().general_progression is None
    with pytest.raises(ValueError,match='Continuous progression'):
        HarnessPolicy(delivery_contract='sustained_v1',general_progression=REALIZED_VERSION)


def test_real_allocator_reference_commits_with_exact_issued_identity_and_survives_restart(tmp_path,monkeypatch):
    from run_analysis.adaptive.live_bridge import runtime_version
    h,inputs,config,at=fixture(tmp_path,monkeypatch)
    h.policy=replace(h.policy,general_progression=REALIZED_VERSION)
    try:
        original=h.stage,h.catalog,h.scales,h.training_policy
        new,review,event=advance_stage(h,inputs,at)
        assert event and event['stage_runtime']['general_progression_state']['version']==REALIZED_VERSION
        assert runtime_version(new)==217
        assert (new.stage,new.catalog,new.scales,new.training_policy)==original
        block=event['realized_reference_allocation']
        assert block['calendar_searches']==0 and block['physical_credit_added']==0
        assert block['full_evaluations']==0 and block['compiled_evaluations']==0
        assert tuple(map(tuple,block['program']))==tuple(map(tuple,event['program']))
        assert len(block['allocations'])==len(event['active'])
        for raw in block['allocations']:
            day=raw['opportunity'].removeprefix('slot:')
            assert raw['issued_fingerprint']==event['active'][day]['fingerprint']
            assert raw['available_at']==event['as_of']
            assert raw['basis']=='issued_calendar_mass_v2'
        snapshot=restore((new.sandbox.root/event['snapshot_file']).read_text(),expected_fingerprint=event['snapshot_fingerprint'])
        assert snapshot.fingerprint==block['source_snapshot']
        reference=block['config']
        assert reference['version']=='calendar_mass_reference_v2'
        reopened=DailyHarness.resume(h.sandbox.root,config,as_of=at)
        try:
            assert reopened.policy.general_progression==REALIZED_VERSION
            assert reopened.general_progression_state==new.general_progression_state
            assert reopened.events[-1]['realized_reference_allocation']==block
            assert initialize(reopened,at+timedelta(hours=1))==reference
            before=json.dumps(block,sort_keys=True)
            rows=review_entries(reopened,at+timedelta(hours=1),original_schedule=True)
            observed=signal(reopened,rows,reference,at+timedelta(hours=1))
            assert observed['reference_quality']=='declared_calendar_mass'
            assert json.dumps(reopened.events[-1]['realized_reference_allocation'],sort_keys=True)==before
        finally: reopened.sandbox.close()
        # Upload then publish again: the completed run keeps the reference
        # associated with its original fingerprint, not a new future allocation.
        original_block=json.dumps(block,sort_keys=True)
        day=min(new.active); chosen=new.active[day]
        dose=new.catalog[chosen['dose']]
        start=datetime.fromisoformat(day).replace(tzinfo=new.zone,hour=16)
        payload,phases,end=synthetic_payload(dose,start,hr=140)
        new.upload(payload,day=day,available_at=end,observed_role=dose.role,observed_phases=phases)
        later=end+timedelta(days=1)
        newer,_,second=advance_stage(new,inputs,later)
        assert second['realized_reference_allocation']['config']==reference
        assert json.dumps(newer.events[-2]['realized_reference_allocation'],sort_keys=True)==original_block
        rows=review_entries(newer,later,original_schedule=True)
        observed=signal(newer,rows,reference,later)
        used=next(r for r in observed['included'] if r['key']=='slot:'+day)
        frozen=next(r for r in block['allocations'] if r['opportunity']=='slot:'+day)
        assert used['issued_miles']==frozen['miles']
        assert used['actual_miles']==pytest.approx(dose.miles)
    finally: h.sandbox.close()


def test_failed_ratio_reference_cannot_resume_as_the_new_measurement():
    from types import SimpleNamespace
    h=SimpleNamespace(general_progression_state=dict(realized_reference=dict(version='fixed_allocator_reference_v1')))
    with pytest.raises(ValueError,match='explicit calendar-mass migration'):
        initialize(h,datetime.now().astimezone())


def test_race_still_requires_an_explicit_migration(tmp_path,monkeypatch):
    h,inputs,_,at=fixture(tmp_path,monkeypatch)
    h.policy=replace(h.policy,general_progression='continuous_general_v6')
    monkeypatch.setattr('run_analysis.adaptive.race_event.preparation_active',lambda *_:True)
    try:
        with pytest.raises(ValueError,match='explicit race-target migration'):
            advance_stage(h,inputs,at)
    finally: h.sandbox.close()


def test_intraday_recognition_is_retained_without_repricing_the_same_dated_target(tmp_path,monkeypatch):
    from run_analysis.adaptive.smoothed_general_progression import propose
    from run_analysis.adaptive.realized_growth import VERSION
    from test_continuous_general_progression import adopt
    h,inputs,_,at=fixture(tmp_path,monkeypatch)
    h.policy=replace(h.policy,general_progression=REALIZED_VERSION)
    try:
        new,_,_=advance_stage(h,inputs,at)
        reference=new.general_progression_state['realized_reference']
        def supplied(_h,_entries,config,clock):
            return dict(version=VERSION,as_of=clock.isoformat(),baseline_key=config['baseline_key'],
                level_ratio=22/config['volume'],growth_per_7=.04,status='growing_at_cap',
                reference_quality='declared_independent_allocation',included=[],unresolved=[])
        monkeypatch.setattr('run_analysis.adaptive.realized_reference_runtime.signal',supplied)
        same=at+timedelta(hours=1)
        first=propose(new,same,review_entries(new,same,original_schedule=True),longest=new.stage.long_miles)
        assert first['continuous_state']['volume']==reference['volume']
        assert first['continuous_state']['deferred_reference_volume']==pytest.approx(22.)
        assert first['continuous_state']['reference_controller']['observed_reference_volume']==pytest.approx(22.)
        adopted=adopt(new,first)
        later=at+timedelta(days=1)
        second=propose(adopted,later,review_entries(adopted,later,original_schedule=True),longest=adopted.stage.long_miles)
        assert second['continuous_state']['volume']==pytest.approx(22.)
        assert second['continuous_state']['reference_controller']['baseline_volume']==reference['volume']
        assert second['continuous_state']['deferred_reference_volume'] is None
    finally: h.sandbox.close()
