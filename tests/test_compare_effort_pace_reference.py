import pytest

from scripts.compare_effort_pace_reference import Comparison, assert_parent_unchanged, work_mode
from test_effort_pace_reference import evidence, sample
from test_requested_quality_forecast import CFG


def test_prescription_structure_not_family_or_target_timing_selects_evidence_semantics():
    payload = dict(quality_session_type='anything', structure=[dict(phase='work', repetitions=6)])
    assert work_mode(payload) == 'commanded_execution'
    assert work_mode(dict(payload, quality_session_type='a new name')) == 'commanded_execution'
    assert work_mode(dict(structure=[dict(phase='work')])) == 'steady_exposure'


def test_unknown_target_intent_stays_unknown_even_with_full_measured_history():
    comparison = Comparison({},CFG)
    comparison.history = [evidence(samples=(sample(3.,158),))]
    result = comparison.query({},None,(),(),145,None,force_steady=True)
    assert result == dict(status='unknown_historical_intent',metrics=None)


def test_parent_gate_rejects_inventory_and_model_changes():
    row = dict(key='a',status='predicted',model=dict(predicted_load=3.))
    fields = ('status','model','baseline_fraction_error','fit','source_keys','programs','pace_references')
    parent = dict(cases=[dict(key='a',**{name:row.get(name) for name in fields})])
    assert_parent_unchanged(dict(cases=[row]),parent)
    with pytest.raises(AssertionError,match='baseline'):
        assert_parent_unchanged(dict(cases=[dict(row,model=dict(predicted_load=4.))]),parent)
    with pytest.raises(AssertionError,match='inventory'):
        assert_parent_unchanged(dict(cases=[]),parent)
