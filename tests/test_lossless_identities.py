"""Exact canonical bytes, not approximate numerical or winner equality."""
from dataclasses import asdict,dataclass,replace,is_dataclass
from datetime import date,datetime,timedelta,timezone
from unittest.mock import patch
from zoneinfo import ZoneInfo
import json
import pytest

from run_analysis.adaptive.identity_json import identity_fields,identity_dumps


@dataclass(frozen=True)
class Nested:
    name:str
    numbers:tuple
    metadata:dict


@pytest.mark.parametrize('separators',[None,(',',':')])
def test_lazy_encoding_matches_full_recursive_copy_bytes(separators):
    obj=Nested('é · "\\',(-0.,1.,1e-18,None,True),{
        'nested':Nested('child',(float(3),),{'day':date(2026,10,4)}),
        'time':datetime(2026,10,4,1,23,tzinfo=timezone(timedelta(hours=-4))),
        'mutable_list':[Nested('item',(),{})]})
    def encode(value):
        if isinstance(value,(date,datetime)):
            return value.isoformat()
        raise TypeError(type(value).__name__)
    kwargs=dict(sort_keys=True,allow_nan=False,default=encode,separators=separators)
    assert identity_dumps(identity_fields(obj),**kwargs)==json.dumps(asdict(obj),**kwargs)
    assert obj.metadata['mutable_list'][0].name=='item'


def test_strict_nan_and_unsupported_values_stay_errors():
    for value in (float('nan'),float('inf'),float('-inf')):
        with pytest.raises(ValueError):
            identity_dumps(Nested('bad',(value,),{}),allow_nan=False)
    with pytest.raises(TypeError):
        identity_dumps(Nested('unsupported',(object(),),{}))
    with pytest.raises(TypeError):
        identity_fields(Nested)


def test_full_program_evaluation_is_identical_including_fingerprint():
    from test_shared_program_evaluator import context,candidate
    from run_analysis.adaptive import shared_program_evaluator as evaluator
    from run_analysis.adaptive import program_evidence_adapter as adapter
    from run_analysis.adaptive.shared_program_evaluator import evaluate_program
    ctx,plan=context(),candidate()
    def original_object(value,*,fragments,default,compact=False):
        def encode(item):
            return asdict(item) if is_dataclass(item) else default(item)
        return json.dumps(value,sort_keys=True,allow_nan=False,default=encode,
            separators=(',',':') if compact else (', ',': '))
    # Explicit old serializer, not current-vs-current after implementation.
    with patch.object(evaluator,'identity_object_dumps',original_object),patch.object(adapter,'identity_object_dumps',original_object):
        original=evaluate_program(ctx,plan)
    changed=evaluate_program(ctx,plan)
    assert changed==original
    assert changed.fingerprint==original.fingerprint


@pytest.mark.parametrize('compact',[False,True])
@pytest.mark.parametrize('fold',[0,1])
def test_owned_fragments_preserve_complete_bytes_and_separate_date_encoders(compact,fold):
    from test_shared_program_evaluator import session
    from run_analysis.adaptive.identity_json import identity_object_dumps,identity_sequence_dumps
    def dates(value):return value.isoformat()
    def utc(value):return value.astimezone(timezone.utc).isoformat() if isinstance(value,datetime) else value.isoformat()
    rows=tuple(replace(session('é "\\',0),quality_credit=credit) for credit in (0,-0.,0.,1,1.))
    for encoder in (dates,utc):
        payload=dict(rows=rows,date=datetime(2026,11,1,1,30,fold=fold,tzinfo=ZoneInfo('America/New_York')))
        expected=identity_dumps(payload,sort_keys=True,allow_nan=False,default=encoder,
            separators=(',',':') if compact else (', ',': '))
        for _ in range(2):
            assert identity_object_dumps(payload,fragments={'rows':identity_sequence_dumps(rows,
                default=encoder,compact=compact)},default=encoder,compact=compact)==expected
    assert rows[0]._identity_fragments is not rows[1]._identity_fragments
    assert '_identity_fragments' not in asdict(rows[0])


def test_mutable_fields_are_not_fragment_cached_and_strict_errors_remain():
    from run_analysis.adaptive.identity_json import FrozenIdentity,identity_sequence_dumps
    @dataclass(frozen=True)
    class Mutable(FrozenIdentity):
        values:list
    row=Mutable([1.])
    assert identity_sequence_dumps((row,),default=None)=='[{"values": [1.0]}]'
    row.values.append(2.)
    assert identity_sequence_dumps((row,),default=None)=='[{"values": [1.0, 2.0]}]'
    assert row._identity_fragments is None
    for value in (float('nan'),float('inf')):
        with pytest.raises(ValueError):identity_sequence_dumps((Mutable([value]),),default=None)


@pytest.mark.parametrize('fold',[0,1])
def test_cached_datetime_fragments_keep_offset_fold_and_encoder_semantics(fold):
    from run_analysis.adaptive.identity_json import FrozenIdentity,identity_sequence_dumps
    @dataclass(frozen=True)
    class Moment(FrozenIdentity):
        at:datetime
    row=Moment(datetime(2026,11,1,1,30,fold=fold,tzinfo=ZoneInfo('America/New_York')))
    local=lambda value:value.isoformat()
    utc=lambda value:value.astimezone(timezone.utc).isoformat()
    for encoder in (local,utc,local):
        assert identity_sequence_dumps((row,),default=encoder)==json.dumps([asdict(row)],
            sort_keys=True,allow_nan=False,default=encoder)
    assert len(row._identity_fragments)==2
    corrected=replace(row,at=row.at+timedelta(hours=2))
    assert corrected._identity_fragments=={}
    assert identity_sequence_dumps((row,),default=utc)!=identity_sequence_dumps((corrected,),default=utc)


def test_cached_horizon_constants_reset_on_context_replacement():
    from test_memory_covered_continuation import guarded
    ctx,_,_=guarded()
    original=ctx.taper_start,ctx.volume_smoothing_days
    old=asdict(ctx)
    assert (ctx.taper_start,ctx.volume_smoothing_days)==original
    changed=replace(ctx,decision=ctx.decision+timedelta(days=1),hard_end=ctx.hard_end+timedelta(days=1))
    assert changed.taper_start==original[0]+timedelta(days=1)
    assert changed.volume_smoothing_days==original[1]
    assert asdict(ctx)==old


@pytest.mark.parametrize('fold',[0,1])
def test_dst_datetime_identity_and_no_cross_value_cache(fold):
    obj=Nested('DST',(),{'time':datetime(2026,11,1,1,30,fold=fold,tzinfo=ZoneInfo('America/New_York'))})
    kwargs=dict(sort_keys=True,allow_nan=False,default=lambda d:d.isoformat())
    first=identity_dumps(obj,**kwargs)
    assert first==json.dumps(asdict(obj),**kwargs)
    obj.metadata['new_field']=[1.,2.]
    second=identity_dumps(obj,**kwargs)
    assert second==json.dumps(asdict(obj),**kwargs)
    assert first!=second  # Only schema metadata, never mutable athlete values, is cached.


def test_reuse_of_normalized_records_matches_fresh_constructor_control():
    from test_program_evidence_adapter import evaluate,fixtures
    from run_analysis.adaptive.program_evidence_adapter import AccountedRun
    from run_analysis.adaptive.program_objective_trial import TrialSession
    from run_analysis.adaptive import shared_program_evaluator as shared
    history,plan=fixtures()
    def fresh(run,day,role,credit):
        return TrialSession(run.key,day,run.miles,role,run.demand,credit,run.source)
    def copied(rows):
        return tuple(sorted((replace(r,demand=tuple(r.demand)) for r in rows),key=lambda r:(r.day,r.key)))
    with patch.object(AccountedRun,'trial_session',fresh),patch.object(shared,'_sessions',copied):
        original=evaluate(history,plan)
    assert evaluate(history,plan)==original
    assert evaluate(history,plan)==original  # Memo hits still encode ALL current fields.


def test_trial_memo_separates_signed_zero_type_and_source_corrections():
    from test_program_evidence_adapter import measured
    r=measured();day=r.start.date()
    positive=r.trial_session(day,'quality',0.)
    assert r.trial_session(day,'quality',0.) is positive
    negative=r.trial_session(day,'quality',-0.)
    integer=r.trial_session(day,'quality',0)
    assert negative is not positive and integer is not positive
    assert negative.quality_credit.hex()!=positive.quality_credit.hex()
    correction=replace(r,miles=r.miles+.1)
    assert correction.trial_session(day,'quality',0.) is not positive
    assert correction.trial_session(day,'quality',0.).miles==correction.miles
    assert '_trial_rows' not in asdict(r)
    for n in range(11):r.trial_session(day,'quality',n/10)
    assert len(r._trial_rows)==8


def test_normalized_tuple_reuse_keeps_mutable_list_boundary_detached():
    from test_shared_program_evaluator import candidate
    from run_analysis.adaptive.shared_program_evaluator import _sessions
    row=candidate().hard[0]
    assert _sessions((row,))[0] is row
    mutable=replace(row,demand=list(row.demand))
    normalized=_sessions((mutable,))[0]
    assert normalized is not mutable and normalized==row
    mutable.demand.clear()
    assert len(normalized.demand)==3
