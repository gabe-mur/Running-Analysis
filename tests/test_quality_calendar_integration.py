from dataclasses import asdict, replace
from datetime import timedelta
from hashlib import sha256
import json

import pytest

from run_analysis.adaptive.direct_quality_load import forecast_direct
from run_analysis.adaptive.execution_forecast_contract import (
    FrozenExecution, IntendedExecution, IntendedPhase, TargetBand, issue_execution, layout)
from run_analysis.adaptive.forecast_handoff import ForecastScope
from run_analysis.adaptive.new_model_daily_harness import Dose
from run_analysis.adaptive.quality_block_contract import read_quality_contract
from run_analysis.adaptive.quality_choices import allowed, compile_choices, install
from test_direct_quality_load import history
from test_quality_dose_policy import AT, menu


def intent(p):
    return IntendedExecution('proposal:'+p.key,AT,AT+timedelta(hours=18),
        tuple(IntendedPhase(s.kind,TargetBand(s.seconds.estimate,s.seconds.low*.9,s.seconds.high*1.1,
            'explicit timing scenario')) for s in p.phases),
        TargetBand(p.miles,max(.1,p.miles-.2),p.miles+.2,'existing distance margin'),
        'units','test reference',forecast_scope=ForecastScope('quality',
            'frozen numeric commands','frozen causal pace scenarios'),
        prescription_contract=p.definition()['numeric_contract'])


def test_full_compiled_menu_uses_the_unchanged_direct_predictor(monkeypatch):
    from run_analysis.adaptive import direct_effort_load,calibrated_quality_load
    def forbidden(*a,**k):
        raise AssertionError('Failed enhanced predictor entered planning')
    monkeypatch.setattr(direct_effort_load,'forecast_effort_contrast',forbidden)
    monkeypatch.setattr(calibrated_quality_load,'forecast_calibrated',forbidden)
    compiled=menu()['prescriptions']
    assert {p.request.family for p in compiled}=={
        'short_intervals','long_intervals','threshold','progression','fartlek','hill_repeats'}
    assert {p.request.work[0].unit for p in compiled}=={'time','distance','portion'}
    for p in compiled:
        command=intent(p)
        record=issue_execution(command,[history()])
        direct=forecast_direct(layout((s.kind,s.seconds.estimate) for s in p.phases),
            [history()],as_of=AT,accounting_key='units')
        assert record.payload['target_forecast']['predicted_load']==direct['predicted_load']
        assert record.payload['planning_handoff']['usable']
        saved=read_quality_contract(record.payload['intent']['prescription_contract'])
        assert saved==p and saved.key==p.key


def test_same_physical_query_does_not_gain_a_family_bonus_or_fake_intensity_effect():
    p=next(p for p in menu()['prescriptions'] if p.request.family=='threshold')
    record=issue_execution(intent(p),[history()])
    stripped=issue_execution(replace(intent(p),prescription_contract=None),[history()])
    assert record.payload['target_forecast']==stripped.payload['target_forecast']
    assert record.fingerprint!=stripped.fingerprint  # Metadata is retained, not a load multiplier.


def test_unsupported_history_is_explicit_without_requiring_identical_workout():
    p=menu()['prescriptions'][0]
    no_history=issue_execution(intent(p),[])
    assert no_history.payload['target_forecast']['predicted_load'] is None
    assert not no_history.payload['planning_handoff']['usable']
    assert issue_execution(intent(p),[history()]).payload['planning_handoff']['usable']


def test_numeric_contract_timing_distance_and_availability_cannot_disagree():
    p=menu()['prescriptions'][0]; command=intent(p)
    with pytest.raises(ValueError,match='Frozen numeric'):
        replace(command,phases=command.phases[:-1])
    with pytest.raises(ValueError,match='Frozen numeric'):
        replace(command,distance_miles=TargetBand(p.miles+.1,p.miles,p.miles+.2,'different dose'))
    with pytest.raises(ValueError,match='Frozen numeric'):
        replace(command,issued_at=p.request.issued_at-timedelta(seconds=1))
    dose=Dose(p.key,'quality',p.miles,tuple((s.kind,s.seconds.estimate) for s in p.phases),
        workout_definition=p.definition())
    with pytest.raises(ValueError,match='numeric prescription'):
        replace(dose,key=dose.key+':mutated')


@pytest.mark.parametrize('version',['execution_forecast_contract_v1','execution_forecast_contract_v2'])
def test_old_issued_bytes_remain_readable_without_rewriting(version):
    record=issue_execution(intent(menu()['prescriptions'][0]),[history()])
    payload=record.payload; payload['version']=version
    payload['intent'].pop('prescription_contract')
    encoded=json.dumps(payload,sort_keys=True)
    old=FrozenExecution(encoded,sha256(encoded.encode()).hexdigest())
    assert old.serialized==encoded and old.payload['version']==version


def test_install_preserves_old_doses_and_freezes_the_new_menu(tmp_path,monkeypatch):
    from test_adaptive_stage_lifecycle import completed_stage
    h,_,_,at=completed_stage(tmp_path,monkeypatch)
    try:
        legacy,contracts=compile_choices(h.catalog['quality:ordinary'],h.coaching_contracts['quality:ordinary'])
        h.catalog.update({d.key:d for d in legacy});h.coaching_contracts.update(contracts)
        before={k:asdict(d) for k,d in h.catalog.items()}
        h.last_as_of=at  # New menu must not pretend it existed at the old anchor's issue.
        install(h)
        assert all(asdict(h.catalog[k])==v for k,v in before.items())
        assert all(not allowed(d,h.coaching_contracts[d.key],h.config) for d in legacy)
        new={k:asdict(d) for k,d in h.catalog.items() if k not in before}
        assert new and all(d['workout_definition']['numeric_contract'] for d in new.values())
        install(h)
        assert new=={k:asdict(d) for k,d in h.catalog.items() if k not in before}
        for key in new:
            command=h.intent(h.catalog[key],at.date(),at,'proposal:'+key)
            assert command.prescription_contract==h.catalog[key].workout_definition['numeric_contract']
            assert read_quality_contract(command.prescription_contract).request.issued_at==at
    finally:h.sandbox.close()
