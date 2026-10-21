from copy import deepcopy
from dataclasses import replace
from datetime import date,timedelta
from types import SimpleNamespace
import sqlite3

import numpy as np
import pytest

from run_analysis import model_windows as windows
from run_analysis.adaptive import program_objective_trial as objective
from run_analysis.adaptive.work_exposure_matching import band_fraction,_band_fraction_uncached,_band_fraction_cached
from run_analysis.program_delivery import TargetSegment,_target_total,_target_total_uncached
from run_analysis.signal_structure import SignalSpan
from test_elevation_quality import config,route,write_tile


def test_preparation_scope_reuses_only_pinned_unchanged_inputs_and_drops_ownership(config,monkeypatch):
    from run_analysis.db import initialize
    from test_progress import _insert_run
    points=route();db=sqlite3.connect(':memory:');db.row_factory=sqlite3.Row;initialize(db)
    _insert_run(db,1,points[0].timestamp_utc,2,24,100)
    db.execute("UPDATE activities SET creator='fr245'")
    db.execute('UPDATE activity_metrics SET model_eligible=1')
    calls=[]
    def load(*args):calls.append(1);return deepcopy(points)
    monkeypatch.setattr(windows,'_load_points',load)
    monkeypatch.setattr(windows,'_hourly_for_activity',lambda *args:{})
    def read():
        return windows.load_model_window_sets(db,config,(120,),check_elevation=True),windows.load_overlapping_model_windows(
            db,config,window_seconds=120,stride_seconds=60,check_elevation=True)
    baseline=read();assert len(calls)==2
    with windows.window_preparation_scope(db,config):
        assert read()==baseline
        assert len(calls)==3
        db.execute("UPDATE activities SET creator='fr265 small'")
        read();assert len(calls)==4
        config['elevation']['smoothing_window_meters']+=1
        read();assert len(calls)==5
    read();assert len(calls)==7
    db.commit()
    with windows.window_preparation_scope(db,config):
        read();assert len(calls)==9  # Autocommit cannot pin one version.
    assert windows._PREPARATION.get() is None


def test_tile_lookup_is_once_per_run_and_does_not_hide_new_maps(config,tmp_path,monkeypatch):
    import run_analysis.elevation_quality as terrain
    points=route();path=tmp_path/'N40W074.hgt'
    original=terrain._read_tile;calls=[]
    def read(*args):calls.append(args);return original(*args)
    monkeypatch.setattr(terrain,'_read_tile',read)
    def prepare():
        movement=windows.classify_movement(points,config['moving_time'])
        return terrain.prepare_terrain(points,movement.intervals,config,'fr245',tmp_path)
    assert prepare()['map_coverage_fraction']==0
    write_tile(path,np.full((1201,1201),10,dtype=np.int16))
    assert prepare()['map_coverage_fraction']==1
    assert len(calls)==1
    write_tile(path,np.full((1201,1201),20,dtype=np.int16))
    assert prepare()['map_coverage_fraction']==1
    assert len(calls)==2
    assert terrain._TILE_LOOKUPS.get() is None


def test_target_and_band_memos_match_original_calculations_and_new_evidence():
    day=date(2026,10,1)
    targets=(TargetSegment(day,day+timedelta(days=10),20.),)
    for rate in (20.,21.,19.):
        current=(replace(targets[0],miles_per_7_days=rate),)
        assert _target_total(current,day,day+timedelta(days=7))==_target_total_uncached(current,day,day+timedelta(days=7))
    _band_fraction_cached.cache_clear()
    for level in (.4,.5,.7):
        spans=(SignalSpan(0.,60.,level,.8),)
        for band in ((.4,.6),(.6,.9)):
            assert band_fraction(spans,0.,60.,band,band)==_band_fraction_uncached(spans,0.,60.,band,band)
            assert band_fraction(list(spans),0.,60.,list(band),list(band))==_band_fraction_uncached(spans,0.,60.,band,band)
    assert _band_fraction_cached.cache_info().hits==6


def test_rolling_frame_keeps_summation_order_and_each_changed_input():
    day=date(2026,10,1);objective._rolling_frame.cache_clear()
    policy=objective.TrialPolicy(weekly_miles=21.,volume_window_days=7)
    history=[SimpleNamespace(day=day-timedelta(days=1),miles=1.e16,role='easy'),
             SimpleNamespace(day=day-timedelta(days=1),miles=.1,role='easy')]
    future=[SimpleNamespace(day=day,miles=.2,role='easy')]
    kwargs=dict(stage_start=day,decision=day,end=day+timedelta(days=2),policy=policy,
        supporting_miles=7.,smoothing_days=1)
    a=objective.rolling_volume_components(history,future,**kwargs)
    assert a['daily'][0]['volume']['delivered']==sum([1.e16,.1,.2])
    b=objective.rolling_volume_components(history,[replace_ns(future[0],miles=3.)],**kwargs)
    assert b['daily'][0]['volume']['delivered']==sum([1.e16,.1,3.])
    assert objective._rolling_frame.cache_info().hits==1
    changed=[replace_ns(history[0],miles=2.),history[1]]
    c=objective.rolling_volume_components(changed,future,**kwargs)
    assert c['daily'][0]['volume']['delivered']==sum([2.,.1,.2])
    assert objective._rolling_frame.cache_info().misses==2
    c['daily'][0]['volume']['target']=1000  # Returned traces never alias the frame.
    assert objective.rolling_volume_components(changed,future,**kwargs)['daily'][0]['volume']['target']==21.


def replace_ns(row,**changes):
    return SimpleNamespace(**(vars(row)|changes))


def test_diagnostic_cache_is_bounded_and_evicted_details_reprice_identically(monkeypatch):
    from test_decision_snapshot import freeze
    from test_quality_coaching_policy import sequence,catalogue
    from run_analysis.adaptive import coached_pipeline as pipeline
    from run_analysis.workload_accounting import Quantity
    state=freeze();seq=sequence();catalog=catalogue()
    days=tuple((state.as_of.date()+timedelta(days=i)).isoformat() for i in range(3))
    physical={(day,r.key):SimpleNamespace(miles=r.miles,
        demand=tuple(Quantity.point(x,'test') for x in (35.,50.,10.)))
        for day in days for r in seq.ordinary}
    calls=[]
    def evaluate(state,rows,keys,**kwargs):
        calls.append(keys)
        return dict(decision_eligible=True,constraints={'status':'allowed'},
            evaluation=SimpleNamespace(total_cost=1.,model_version='test',fingerprint='same'))
    monkeypatch.setattr(pipeline,'evaluate_snapshot',evaluate)
    monkeypatch.setattr(pipeline,'DETAIL_CACHE_LIMIT',1)
    engine=pipeline.coached_evaluator(dict(state=state,catalog=catalog,dates=days,physical=physical),seq)
    programs=[((day,seq.assignment(0).key),) for day in days]
    first=engine['detail'](programs[0])
    assert engine['detail'](programs[0]) is first and len(calls)==1
    for program in programs[1:]:
        engine['detail'](program)
        assert len(engine['cache'])==1
    assert engine['detail'](programs[0])==first and len(calls)==4


def test_continuation_frame_reuses_only_static_inputs_not_history_cost_or_returned_traces():
    from test_memory_covered_continuation import guarded
    from test_soft_horizon_trial import candidate
    from run_analysis.adaptive.shared_program_evaluator import evaluate_program
    from run_analysis.adaptive.continuation_boundary_trial import _continuation_frame
    ctx,refs,rows=guarded();_continuation_frame.cache_clear()
    first=evaluate_program(ctx,candidate(ctx,rows))
    assert evaluate_program(ctx,candidate(ctx,rows))==first
    assert _continuation_frame.cache_info().hits==1
    original=first.total_cost
    first.parts['continuation_boundary']['scenarios'][0]['losses']['volume']=1.e6
    again=evaluate_program(ctx,candidate(ctx,rows))
    assert again.total_cost==original
    assert again.parts['continuation_boundary']['scenarios'][0]['losses']['volume']!=1.e6
    actual=replace(rows[0],key='extra-actual',day=ctx.decision-timedelta(days=1),source='observed')
    changed=replace(ctx,history=(actual,))
    other=evaluate_program(changed,candidate(changed,rows))
    assert other.total_cost!=original
    assert _continuation_frame.cache_info().misses==1
    changed=replace(ctx,policy=replace(ctx.policy,weekly_miles=ctx.policy.weekly_miles+1))
    evaluate_program(changed,candidate(changed,rows))
    assert _continuation_frame.cache_info().misses==2


def test_quality_assignment_preparation_keeps_every_rank_and_resets_on_replacement():
    from dataclasses import asdict
    from test_quality_coaching_policy import sequence
    from run_analysis.adaptive.quality_coaching_policy import FAMILIES
    def original(q,n):
        pattern=('sustained','faster','faster') if q.focus=='vo2max' else ('sustained','faster')
        offset=pattern.index(q.first_purpose);pattern=pattern[offset:]+pattern[:offset]
        cycles,position=divmod(q.consumed+n,len(pattern));purpose=pattern[position]
        occurrence=cycles*pattern.count(purpose)+pattern[:position].count(purpose)
        families=tuple(f for f in FAMILIES[purpose] if any(r.family==f for r in q.ordinary))
        if q.focus=='vo2max' and purpose=='faster' and 'long_intervals' in families:
            families=('long_intervals','long_intervals',*(f for f in families if f!='long_intervals'))
        rounds,slot=divmod(occurrence,len(families));family=families[slot]
        recipes=sorted((r for r in q.ordinary if r.family==family),key=lambda r:r.recipe)
        rank=rounds*families.count(family)+families[:slot].count(family)
        return recipes[rank%len(recipes)]
    base=sequence();before=asdict(base)
    for focus in ('variety','vo2max'):
        for first in FAMILIES:
            for consumed in (0,1,7,100):
                for disable in (None,'long_intervals','progression','short_intervals'):
                    q=replace(base,focus=focus,first_purpose=first,consumed=consumed,
                        ordinary=tuple(r for r in base.ordinary if r.family!=disable))
                    assert [q.assignment(n) for n in range(100)]==[original(q,n) for n in range(100)]
                    assert '_family_slots' not in asdict(q)
    assert asdict(base)==before


def test_continuation_terms_keep_order_bits_and_corrections_and_do_not_memo_mutable_demands():
    from math import exp
    from dataclasses import asdict
    from test_shared_program_evaluator import session
    from run_analysis.adaptive.continuation_boundary_trial import _residual_terms
    from run_analysis.adaptive.program_objective_trial import DemandScenario,demand_value
    row=session('example',0)
    handoffs=tuple(row.day+timedelta(days=d) for d in (-1,0,1,3,10))
    scales=(44.,88.,15.)
    for fraction in (0.,.5,1.):
        scenario=DemandScenario(bound_fraction=fraction,timing_compression_days=.1)
        actual=_residual_terms(row,handoffs,scales,scenario)
        assert _residual_terms(row,handoffs,scales,scenario) is actual
        nominal=np.array([demand_value(q,fraction) for q in row.demand])/scales
        for at,value in zip(handoffs,actual):
            if at<=row.day:assert value is None
            else:
                expected=nominal*exp(-((at-row.day).days-scenario.timing_compression_days)/scenario.memory_days)
                assert value.tobytes()==expected.tobytes()
                assert not value.flags.writeable
    corrected=replace(row,demand=tuple(q.scale(1.1) for q in row.demand))
    assert corrected._continuation_terms=={}
    assert _residual_terms(corrected,handoffs,scales,scenario)[-1].tobytes()!=actual[-1].tobytes()
    assert '_continuation_terms' not in asdict(row)
    mutable=replace(row,demand=list(row.demand))
    first=_residual_terms(mutable,handoffs,scales,scenario)[-1].copy()
    mutable.demand[0]=mutable.demand[0].scale(1.1)
    assert mutable._continuation_terms is None
    assert _residual_terms(mutable,handoffs,scales,scenario)[-1].tobytes()!=first.tobytes()
