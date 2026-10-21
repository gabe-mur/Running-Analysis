from dataclasses import asdict, replace
import json

import pytest

from run_analysis.adaptive.execution_forecast_contract import issue_execution, _encode
from run_analysis.adaptive.quality_catalogue_forecast import forecast_prescription_point
from run_analysis.adaptive.quality_forecast_runtime import VERSION
from test_quality_calendar_integration import intent
from test_quality_catalogue_forecast import CONFIG, REF, evidence, example


def test_explicit_runtime_issue_matches_shared_supported_forecast_exactly():
    p=example();h,t=evidence()
    command=replace(intent(p),accounting_key=REF.accounting_key)
    quality=dict(version=VERSION,available_at=p.request.issued_at,config=CONFIG,
        profiles={k:[asdict(q) for q in v] for k,v in t.items()},reference=asdict(REF),response=None,anchors=[])
    command=replace(command,quality_evidence=json.loads(json.dumps(quality,default=_encode)))
    record=issue_execution(command,h)
    actual=record.payload['target_forecast']
    expected=forecast_prescription_point(p,h,t,CONFIG,reference=REF)
    assert actual==json.loads(json.dumps(expected,default=_encode))
    assert record.payload['planning_handoff']['requested_effort_conditioned']
    assert record.payload['planning_handoff']['usable']
    assert record.payload['intent']['quality_evidence']==command.quality_evidence


def test_absent_runtime_preserves_old_intent_shape_and_direct_forecast():
    p=example();h,_=evidence();command=replace(intent(p),accounting_key=REF.accounting_key)
    record=issue_execution(command,h)
    assert 'quality_evidence' not in record.payload['intent']
    assert 'work_exposure_matching' not in record.payload['target_forecast']
    with pytest.raises(ValueError,match='numeric command'):
        replace(command,prescription_contract=None,quality_evidence={})
