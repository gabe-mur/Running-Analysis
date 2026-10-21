from dataclasses import asdict,replace
from datetime import timedelta
from types import SimpleNamespace

import pytest

from run_analysis.adaptive.quality_catalogue_runtime import (completion_signals,decision_evidence,
    exposure_pace,json_value,read_contract)
from run_analysis.adaptive.quality_dose_policy import purpose_contract
from run_analysis.workload_accounting import Quantity
from test_quality_catalogue_forecast import CONFIG,evidence
from test_quality_block_contract import pace
from test_quality_dose_policy import stage,AT
from test_execution_upload_sandbox import setup,tcx,KINDS


def test_contract_json_roundtrip_checks_derived_version():
    c=purpose_contract(stage(),config=CONFIG,issued_at=AT,easy_pace=pace(.4),faster_pace=pace(.28))
    raw=json_value(c)
    assert read_contract(raw)==c
    raw['requirements'][0]['target']['version']='unknown'
    with pytest.raises(ValueError,match='Unknown target'):
        read_contract(raw)


def test_original_completion_profiles_and_windows_not_copied_intent(setup, monkeypatch):
    s,p,h,config=setup
    record=s.publish(p,[h])
    # Fixture ATs differ by several days between modules: use source availability.
    from test_direct_quality_load import AT as actual_at
    at=actual_at+timedelta(days=1)
    s.upload(tcx(hr=170),fingerprint=record.fingerprint,available_at=at,
        observed_phases=tuple((k,60.) for k in KINDS),observed_role='quality',phase_evidence='explicit')
    harness=SimpleNamespace(sandbox=s,config=config,ref=s.reference)
    signals,profiles=completion_signals(harness,at)
    assert signals and set(signals)==set(profiles)
    signal=next(iter(signals.values()))
    assert signal.windows==((60.,120.),(180.,240.))
    assert all(span.first==pytest.approx(.8) for span in signal.hrr)
    # The file contains .4 miles, not the original recipe's synthetic 4 miles.
    assert sum((span.end-span.start)*span.first for span in signal.speed_mps)==pytest.approx(.4*1609.344)
    from run_analysis.adaptive.quality_catalogue_runtime import _original_completion_trace
    hits=_original_completion_trace.cache_info().hits
    assert completion_signals(harness,at)==(signals,profiles)
    assert _original_completion_trace.cache_info().hits==hits+1
    # A warm cache must still reject changed/missing original evidence.
    import run_analysis.importer as importer
    monkeypatch.setattr(importer,'sha256_file',lambda path:'changed')
    with pytest.raises(ValueError,match='exact original source'):
        completion_signals(harness,at)


def test_recent_profile_reduction_is_exact_not_a_new_evidence_window(setup):
    s,p,h,config=setup
    from test_direct_quality_load import AT as actual_at
    observed,profiles=evidence()
    runtime=dict(evidence=dict(available_at=actual_at.isoformat(),profiles=json_value(profiles),config=config))
    # Use this fixture's physically valid source clock.
    prior=replace(observed[0],available_at=actual_at,start=actual_at-timedelta(days=1),
        end=actual_at-timedelta(hours=12))
    ancient=replace(prior,key='outside-window',start=actual_at-timedelta(days=31),
        end=actual_at-timedelta(days=30))
    runtime['evidence']['profiles']['outside-window']=runtime['evidence']['profiles'][prior.key]
    harness=SimpleNamespace(sandbox=s,config=config,ref=s.reference,quality_runtime=runtime,
        phase_history=(prior,ancient))
    filtered,signals=decision_evidence(harness,actual_at)
    assert set(filtered['profiles'])=={prior.key} and not signals
    assert 'outside-window' in runtime['evidence']['profiles']


def test_new_zone_cues_do_not_rewrite_accepted_profiles_or_hr_normalization(setup):
    from copy import deepcopy
    s,p,h,config=setup
    from test_direct_quality_load import AT as at
    original=deepcopy(config)
    runtime=dict(evidence=dict(available_at=at.isoformat(),profiles={},config=original))
    new_config=deepcopy(config)
    new_config['zones']['z4']=[168,181]
    harness=SimpleNamespace(sandbox=s,config=new_config,ref=s.reference,quality_runtime=runtime,phase_history=())
    before=deepcopy(runtime)
    filtered,_=decision_evidence(harness,at)
    assert filtered['config']['zones']==new_config['zones']
    assert filtered['config']['resting_hr']==config['resting_hr']
    assert filtered['config']['max_hr']==config['max_hr']
    assert runtime==before
