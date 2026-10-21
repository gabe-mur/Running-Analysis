from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from run_analysis.adaptive_prescriptions import compile_prescriptions
from run_analysis.athlete_projection import SessionEvidence
from run_analysis.staged_progression import DevelopmentStage, ProgressionPolicy, MILE
from scripts.new_model_daily_harness import accounted
from run_analysis.continuous_training_dose import ContinuousDoseReference
from run_analysis.workload_accounting import Quantity

AT = datetime(2026,9,27,tzinfo=timezone.utc)


def fixture(scale=1.):
    stage = DevelopmentStage('test',AT,20.*scale,4.*scale,8.*scale,17.*scale,8.,ProgressionPolicy(14,.05),'test')
    history = tuple(SessionEvidence(str(i),AT-timedelta(days=i+1,hours=1),AT-timedelta(days=i+1),
        AT-timedelta(days=i+1),'easy',4.*scale*MILE,2400.,None,'test') for i in range(4))
    return stage,history


def test_causal_frozen_construction_and_one_decimal_commands():
    stage,history = fixture()
    a = compile_prescriptions(history,stage)
    late = replace(history[0],key='late',available_at=AT+timedelta(seconds=1),moving_seconds=600.)
    excluded = replace(history[0],key='excluded',capacity_eligible=False,moving_seconds=900.)
    assert compile_prescriptions((*history,late,excluded),stage)==a
    assert a['nominal_minutes_per_mile']==10.
    easy = [p for p in a['prescriptions'] if p.role == 'easy']
    tenths = sorted(round(p.miles*10) for p in easy)
    assert tenths == list(range(tenths[0], tenths[-1]+1))
    for p in a['prescriptions']:
        assert p.miles*10==pytest.approx(round(p.miles*10))
        assert all(lo<=s<=hi for (_,s),(lo,hi) in zip(p.phases,p.phase_scenarios))


def test_duration_floor_and_athlete_scaling_do_not_require_same_mileage():
    a = compile_prescriptions(fixture()[1],fixture()[0])
    b = compile_prescriptions(fixture(2.)[1],fixture(2.)[0])
    assert b['nominal_minutes_per_mile']==a['nominal_minutes_per_mile']/2
    for result in (a,b):
        for p in result['prescriptions']:
            if p.role=='easy':
                assert sum(lo for lo,_ in p.phase_scenarios)>=600.
    assert b['minimum_easy_miles']>a['minimum_easy_miles']


def test_quality_reduction_shares_contract_and_novel_work_is_explicit_trial():
    stage,history = fixture()
    rows = {p.key:p for p in compile_prescriptions(history,stage)['prescriptions']}
    a,b = rows['quality:ordinary'],rows['quality:reduced']
    assert a.contract==b.contract
    assert a.trial and a.contract.trial_target==180.
    assert a.contract.reduced.work_seconds.high<a.contract.ordinary.work_seconds.low
    assert b.contract.reduced.miles.estimate==b.miles


def test_recent_detected_structure_not_label_or_exact_template_count():
    stage,history = fixture()
    ref = ContinuousDoseReference('test')
    start=AT-timedelta(days=1,hours=2)
    phases=(('warmup',600.),('work',180.),('recovery',90.),('work',150.),('cooldown',600.))
    observed=accounted('quality',start,3.,phases,Quantity.point(30,'test'),'quality',ref,
        source='observed',available=start+timedelta(hours=1))
    h=SessionEvidence('quality',observed.start,observed.end,observed.available_at,'intervals',3*MILE,1620,None,ref.accounting_key)
    rows={p.key:p for p in compile_prescriptions((*history,h),stage,physical=(observed,))['prescriptions']}
    assert not rows['quality:ordinary'].trial
    assert [s for k,s in rows['quality:ordinary'].phases if k=='work']==[180.,150.]
    assert rows['quality:ordinary'].contract.trial is None


def test_missing_history_and_impossible_stage_are_explicit_not_guessed():
    stage,history=fixture()
    assert compile_prescriptions([],stage)['status']=='missing_duration_evidence'
    result=compile_prescriptions(history,replace(stage,ordinary_miles=.5))
    assert result['status']=='stage_duration_envelope_conflict'
    assert not result['prescriptions']


@pytest.mark.parametrize('work_minutes',[3.,12.,30.])
def test_stage_quality_completion_does_not_alter_physical_demand(work_minutes):
    from scripts.program_evidence_adapter import stage_quality_credit
    from scripts.program_objective_trial import TrialPolicy
    ref=ContinuousDoseReference('test')
    def run(n):
        return accounted('q',AT,4.,(('warmup',600.),('work',n*60),('cooldown',600.)),
            Quantity.point(n+20,'test'),'quality',ref,source='planned',available=AT)
    full,half=run(work_minutes),run(work_minutes/2)
    policy=TrialPolicy(quality_work_minutes=work_minutes)
    assert stage_quality_credit(full,policy).low==1.
    assert stage_quality_credit(half,policy).low==.5
    assert full.demand[2].low==work_minutes
    assert full.demand[1].low==work_minutes+20
    assert stage_quality_credit(full,TrialPolicy())==full.quality
    unknown=replace(full,bouts=(),work_evidence_basis='unknown',demand=(*full.demand[:2],Quantity(0.,work_minutes)))
    assert stage_quality_credit(unknown,policy).low==0.
