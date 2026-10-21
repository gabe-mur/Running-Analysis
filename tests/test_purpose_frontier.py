from datetime import date,timedelta
import pytest
from run_analysis.workload_accounting import Quantity
from run_analysis.adaptive.program_objective_trial import TrialSession,TrialPolicy,credits,PURPOSES
from run_analysis.adaptive.timely_purpose_trial import PurposeTimingPolicy,timing_components,envelope_day


@pytest.mark.parametrize('grace',[.2,1.,1.4])
@pytest.mark.parametrize('memory',[.5,1.,2.])
def test_frontier_matches_full_event_envelope_for_every_day(grace,memory):
    start=date(2026,9,27)
    policy=TrialPolicy()
    events=tuple(TrialSession(str(i),start+timedelta(days=d),miles,role,(Quantity.point(1,'oracle fixture'),)*3,
        quality_credit=credit,source='observed' if d<0 else 'planned')
        for i,(d,miles,role,credit) in enumerate((
            (-20,8.4,'long',0),(-9,7.5,'long',0),(-5,4,'quality',1),(-3,8.4,'long',0),
            (0,4,'quality',.5),(2,4,'quality',.3),(4,6.7,'long',0),(8,4,'quality',1),
            (9,8.4,'long',0),(14,4,'quality',.5),(20,4,'quality',1),(28,8.4,'long',0))))
    history=tuple(r for r in events if r.source=='observed')
    planned=tuple(r for r in events if r.source=='planned')
    timing=PurposeTimingPolicy(memory,grace_fraction=grace)
    result=timing_components(history,planned,stage_start=start,decision=start,end=start+timedelta(days=49),
        policy=policy,timing=timing,opening_coverage=(('long',.4),('quality',.8)))
    for purpose,initial in (('long',.4),('quality',.8)):
        reference=[(r.day,float(credits(r,policy)[PURPOSES.index(purpose)])) for r in events]
        for row in result[purpose]['daily']:
            expected=envelope_day(reference,row['day'],grace=7*grace,memory=7*memory,initial=initial,initial_day=start)
            assert row['loss']==pytest.approx(expected['loss'],abs=1e-12)
            assert row['opening_coverage']==pytest.approx(expected['opening'],abs=1e-12)
