from dataclasses import asdict,replace
from datetime import datetime,timedelta,timezone
import json
from types import SimpleNamespace

import pytest

from run_analysis.adaptive.review_seed import read_entry,validate_seed
from run_analysis.adaptive.adaptive_stage_lifecycle import review_entries
from run_analysis.adaptive.prospective_progression import freeze_volume_review
from run_analysis.delivered_training import DoseContract,DoseTarget,DeliveredDose
from run_analysis.staged_progression import DevelopmentStage,ProgressionPolicy,ReviewEntry,MILE
from run_analysis.stage_target_history import StageTarget
from run_analysis.workload_accounting import Quantity

AT=datetime(2026,10,3,tzinfo=timezone.utc)


def fixture():
    start=AT+timedelta(days=2)
    contract=DoseContract(AT,'long_endurance',(
        DoseTarget('distance_meters',Quantity(8.7*MILE,9.1*MILE,8.9*MILE,'issued band')),),'original issue')
    done=DeliveredDose('actual:long',start,start+timedelta(hours=2),start+timedelta(hours=2),
        Quantity.point(9.5*MILE,'actual'),Quantity.point(7200,'actual'),(),'unknown','long')
    row=ReviewEntry('slot:2026-10-05',start,contract,done)
    history=(SimpleNamespace(key=done.key),)
    return row,history,AT+timedelta(days=3)


def test_seed_preserves_actual_and_issued_credit_through_serialization():
    row,history,origin=fixture()
    seed=validate_seed((row,),history,origin)
    raw=json.loads(json.dumps(asdict(row),default=lambda value:value.isoformat()))
    assert read_entry(raw)==row
    harness=SimpleNamespace(seed_review_entries=seed,events=[],
        sandbox=SimpleNamespace(completed_accounted=lambda at:()),zone=timezone.utc)
    stage=DevelopmentStage('phase',AT,18.1,4.2,8.9,17.,7.,
        ProgressionPolicy(14,.05,'easy_volume',19.,'gradual'),'fixture')
    targets=(StageTarget('path',AT,AT.date(),18.1,5.4,'fixture',14,19.,6.3),)
    rows=review_entries(harness,origin)
    original=freeze_volume_review(stage,targets,(row,),origin,review_at=stage.review_at,timezone='UTC')
    cloned=freeze_volume_review(stage,targets,rows,origin,review_at=stage.review_at,timezone='UTC')
    assert cloned==original
    assert cloned.completed_credit==((row.key,8.9),)  # Actual9.5 is not extra prescription credit.
    assert row.completion.distance_meters.estimate==9.5*MILE  # Physical evidence stays unchanged.


@pytest.mark.parametrize('problem',['uncompleted','future','absent','duplicate','reused_completion'])
def test_seed_cannot_invent_completion_or_use_unavailable_evidence(problem):
    row,history,origin=fixture()
    rows=(row,)
    if problem=='uncompleted':rows=(replace(row,completion=None),)
    if problem=='future':rows=(replace(row,completion=replace(row.completion,available_at=origin+timedelta(days=1))),)
    if problem=='absent':history=()
    if problem=='duplicate':rows=(row,row)
    if problem=='reused_completion':rows=(row,replace(row,key='another_slot'))
    with pytest.raises(ValueError):validate_seed(rows,history,origin)
