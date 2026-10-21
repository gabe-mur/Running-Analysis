from dataclasses import replace
from datetime import datetime,timedelta
from zoneinfo import ZoneInfo

import pytest

from run_analysis.continuous_training_dose import ContinuousDoseReference
from scripts.new_model_daily_harness import Dose,planned_accounted,point
from scripts.program_evidence_adapter import accounted_phases,reconcile_completion


ZONE=ZoneInfo('America/New_York')
START=datetime(2026,9,27,18,tzinfo=ZONE)
REF=ContinuousDoseReference('test')


def row(available=START):
    from dataclasses import asdict
    return planned_accounted('planned',START,Dose('easy','easy',3.,(('warmup',1800.),)),
        dict(usable=True,cardio=asdict(point(30.)),running_minutes=asdict(point(30.))),
        REF,available=available)


def test_same_date_after_suggested_clock_does_not_expire_planning_slot():
    run=row(START+timedelta(hours=3))
    assert run.start==START and run.available_at.hour==21
    assert run.advisory_timezone=='America/New_York'


def test_clock_is_not_permission_to_replan_an_elapsed_date_or_fake_actual():
    with pytest.raises(ValueError,match='current/future'):
        row(START+timedelta(days=1))
    with pytest.raises(ValueError,match='Advisory clock'):
        replace(row(),source='observed',available_at=START+timedelta(hours=1))
    with pytest.raises(ValueError,match='Forecast must'):
        replace(row(),available_at=START+timedelta(hours=1),advisory_timezone=None)


def test_completion_still_requires_forecast_preceding_actual_start():
    actual=accounted_phases('actual',START,3.,(('warmup',1800.),),point(30.),'easy',REF,
        source='observed',available=START+timedelta(minutes=30),moving=point(30.))
    with pytest.raises(ValueError,match='pre-run'):
        reconcile_completion((),(row(START+timedelta(hours=1)),),actual,
            replaces_key='planned',as_of=START+timedelta(hours=2))
    h,p=reconcile_completion((),(row(),),actual,replaces_key='planned',as_of=actual.end)
    assert h==(actual,) and not p
